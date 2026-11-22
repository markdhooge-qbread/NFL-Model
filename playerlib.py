"""Player projections for the NFL game flow app (shared by player_train.py and app.py).

Approach (validated in the project's props work): projection = player's season-to-date average + a linear
adjustment for last-season level, efficiency regression, role trend, opponent allowed, and the game flow model's
predicted team points, margin and total plays. Anytime TD uses a logistic model.
"""
import numpy as np
import pandas as pd

PCOLS = ["two_point_attempt", "receiver_player_id", "receiver_player_name", "rusher_player_id", "rusher_player_name",
         "passer_player_id", "passer_player_name", "passing_yards", "receiving_yards", "rushing_yards",
         "complete_pass", "air_yards", "yards_after_catch", "pass_attempt", "rush_attempt", "td_player_id",
         "pass_touchdown", "rush_touchdown"]
K = ["game_id", "season", "week", "posteam", "defteam"]
CTX = ["c_pts", "c_margin", "c_plays", "c_home"]

DVP = ["dvp_rec", "dvp_tgt", "dvp_rush"]   # defense vs position: % above/below league average allowed to this position
FEATS = {
    "rec_yds": ["ly_diff", "p_tgt", "p_ypt", "p_yac_r", "p_cr", "ts_trend", "p_ays", "opp_ypt", "dvp_rec", "dvp_tgt",
                "pos_WR", "pos_TE"] + CTX,
    "rec": ["ly_rec_diff", "p_tgt", "p_cr", "ts_trend", "opp_cr", "dvp_tgt", "pos_WR", "pos_TE"] + CTX,
    "rush_yds": ["ly_diff", "p_car", "p_ypc", "cs_trend", "last3", "opp_ypc", "opp_car", "dvp_rush", "pos_QB"] + CTX,
    "car": ["ly_car_diff", "p_cs", "cs_trend", "opp_car", "dvp_rush", "pos_QB"] + CTX,
    "pass_yds": ["ly_diff", "p_db", "opp_ypd", "opp_db", "dvp_pass"] + CTX,
    "td": ["p_td", "p_rz", "p_tgt", "p_car", "ly_td"] + CTX,
}
BASE = {"rec_yds": "p_rec_yds", "rec": "p_rec", "rush_yds": "p_rush_yds", "car": "p_car", "pass_yds": "p_pass_yds"}
FLAG_MIN = {"rec_yds": 8, "rush_yds": 8, "pass_yds": 15, "rec": 0.7, "car": 2.0}


def player_games(raw):
    """One row per skill player per game (targets, catches, yards, carries, TDs) and one per QB per game."""
    d = raw[(raw.season_type == "REG") & (raw.two_point_attempt != 1)]
    d = d.assign(posteam=d.posteam.replace({"OAK": "LV", "SD": "LAC", "STL": "LA"}),
                 defteam=d.defteam.replace({"OAK": "LV", "SD": "LAC", "STL": "LA"}))
    pa = d[(d.pass_attempt == 1) & (d.sack != 1) & d.receiver_player_id.notna()].assign(rz=lambda x: x.yardline_100 <= 20)
    tm = pa.groupby(K).agg(tm_tgt=("pass_attempt", "size"), tm_air=("air_yards", "sum"))
    r = pa.groupby(K + ["receiver_player_id"]).agg(
        name=("receiver_player_name", "first"), tgt=("pass_attempt", "size"), rec=("complete_pass", "sum"),
        rec_yds=("receiving_yards", "sum"), air=("air_yards", "sum"), yac=("yards_after_catch", "sum"),
        rz_t=("rz", "sum")).reset_index().rename(columns={"receiver_player_id": "pid"})
    ru = d[(d.rush_attempt == 1) & d.rusher_player_id.notna()].assign(rz=lambda x: x.yardline_100 <= 20)
    tr = ru.groupby(K).agg(tm_car=("rush_attempt", "size"))
    u = ru.groupby(K + ["rusher_player_id"]).agg(
        name_r=("rusher_player_name", "first"), car=("rush_attempt", "size"), rush_yds=("rushing_yards", "sum"),
        rz_c=("rz", "sum")).reset_index().rename(columns={"rusher_player_id": "pid"})
    s = r.merge(u, on=K + ["pid"], how="outer")
    s["name"] = s.name.fillna(s.name_r)
    s = s.drop(columns="name_r").join(tm, on=K).join(tr, on=K)
    for c in ["tgt", "rec", "rec_yds", "air", "yac", "rz_t", "car", "rush_yds", "rz_c"]:
        s[c] = s[c].fillna(0)
    tds = d[(d.pass_touchdown.eq(1) | d.rush_touchdown.eq(1)) & d.td_player_id.notna()]
    tdc = tds.groupby(["game_id", "td_player_id"]).size().rename("tds")
    s = s.join(tdc, on=["game_id", "pid"]).fillna({"tds": 0})
    s["td"] = (s.tds > 0).astype(float)
    s["rz"] = s.rz_t + s.rz_c
    qb = d[(d.qb_dropback == 1) & d.passer_player_id.notna()]
    q = qb.groupby(K + ["passer_player_id"]).agg(name=("passer_player_name", "first"), db=("qb_dropback", "size"),
                                                 pass_yds=("passing_yards", "sum")).reset_index()
    q = q.rename(columns={"passer_player_id": "pid"})
    q["pass_yds"] = q.pass_yds.fillna(0)
    return s, q


def _prior(df, cols):
    df = df.sort_values(["season", "week"]).copy()
    keys = [df.pid, df.season]; n = df.groupby(keys).cumcount()
    for c in cols:
        v = df[c].fillna(0)
        df["p_" + c] = (v.groupby(keys).cumsum() - v) / n.replace(0, np.nan)
    df["n"] = n
    return df


def _roll3(df, c):
    return df.groupby(["pid", "season"])[c].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())


def _last_season(df, cols):
    real = df[df.fut == 0]
    ls = real.groupby(["pid", "season"])[cols].mean()
    ls.index = ls.index.set_levels(ls.index.levels[1] + 1, level=1)
    return df.join(ls.add_prefix("ly_"), on=["pid", "season"])


def _opp_allowed(df, sums, outs):
    """Defense allowed per game, season to date (prior games only). sums: {name: col}, outs: {feature: fn(agg)}."""
    od = df[df.fut == 0].groupby(["game_id", "season", "week", "defteam"]).agg(**{k: (v, "sum") for k, v in sums.items()})
    od = od.reset_index().sort_values(["season", "week"])
    G = od.groupby(["defteam", "season"])
    cum = {k: G[k].cumsum() for k in sums}
    n = G.cumcount() + 1
    out = od[["season", "defteam", "week"]].copy()
    for f, fn in outs.items():
        out[f] = fn(cum, n)
    # value going INTO the next game = after this week; shift to next game via merge_asof on week
    out = out.sort_values("week")
    res = df[["season", "defteam", "week"]].drop_duplicates().sort_values("week")
    m = pd.merge_asof(res, out.rename(columns={"week": "wk_done"}).sort_values("wk_done"), left_on="week",
                      right_on="wk_done", by=["season", "defteam"], allow_exact_matches=False)
    return df.merge(m.drop(columns="wk_done"), on=["season", "defteam", "week"], how="left")


def add_future(df, games_future, last_n_team_games=2):
    """Append rows for upcoming games: players who appeared in one of their team's last N games this season."""
    if games_future is None or games_future.empty:
        return df.assign(fut=0)
    rows = []
    for (season, week), wk in games_future.groupby(["season", "week"]):
        cur = df[(df.season == season) & (df.week < week)]
        if cur.empty:
            continue
        last = cur.sort_values("week").groupby("pid").tail(1)
        tw = cur.groupby("posteam").week.apply(lambda w: sorted(w.unique())[-last_n_team_games:][0])
        for _, g in wk.iterrows():
            for t, o in ((g.home_team, g.away_team), (g.away_team, g.home_team)):
                x = last[(last.posteam == t) & (last.week >= tw.get(t, 99))].copy()
                if x.empty:
                    continue
                keep = ["pid", "name", "posteam"]
                x = x[keep].assign(game_id=g.game_id, season=season, week=week, defteam=o)
                rows.append(x)
    fut = pd.concat(rows, ignore_index=True).assign(fut=1) if rows else pd.DataFrame()
    return pd.concat([df.assign(fut=0), fut], ignore_index=True)


def skill_features(s):
    s = s.copy()
    real = s.fut == 0
    s["ts"] = (s.tgt / s.tm_tgt).where(real)
    s["ays"] = (s.air / s.tm_air.replace(0, np.nan)).where(real)
    s["cs"] = (s.car / s.tm_car).where(real)
    s["ypt"] = (s.rec_yds / s.tgt.replace(0, np.nan)).where(real)
    s["cr"] = (s.rec / s.tgt.replace(0, np.nan)).where(real)
    s["yac_r"] = (s.yac / s.rec.replace(0, np.nan)).where(real)
    s["ypc"] = (s.rush_yds / s.car.replace(0, np.nan)).where(real)
    s = _prior(s, ["rec_yds", "rec", "tgt", "ts", "ays", "ypt", "cr", "yac_r", "rush_yds", "car", "cs", "ypc", "td", "rz"])
    s["ts_trend"] = _roll3(s, "ts") - s.p_ts
    s["cs_trend"] = _roll3(s, "cs") - s.p_cs
    s["last3"] = _roll3(s, "rush_yds") - s.p_rush_yds
    s = _last_season(s, ["rec_yds", "rec", "rush_yds", "car", "td"])
    s["ly_rec_diff"] = (s.ly_rec - s.p_rec).fillna(0)
    s["ly_car_diff"] = (s.ly_car - s.p_car).fillna(0)
    s["ly_td"] = s.ly_td.fillna(s.p_td)
    s = _opp_allowed(s, {"t": "tgt", "y": "rec_yds", "c": "rec", "car": "car", "ry": "rush_yds"},
                     {"opp_ypt": lambda c, n: c["y"] / c["t"], "opp_cr": lambda c, n: c["c"] / c["t"],
                      "opp_ypc": lambda c, n: c["ry"] / c["car"], "opp_car": lambda c, n: c["car"] / n})
    return s


def _dvp(df, val_cols, pos_col="pos"):
    """Defense vs position. For each defense and position: season-to-date average allowed per game (prior games only),
    as a % above (+) or below (-) the league average per team-game for that position over the same weeks."""
    real = df[df.fut == 0]
    od = real.groupby(["season", "week", "defteam", pos_col])[val_cols].sum().reset_index()
    out = []
    for (season, pos), d in od.groupby(["season", pos_col]):
        weeks = sorted(df[(df.season == season)].week.unique())
        piv = {c: d.pivot_table(index="week", columns="defteam", values=c, aggfunc="sum") for c in val_cols}
        teams_played = d.pivot_table(index="week", columns="defteam", values=val_cols[0], aggfunc="size").notna()
        for c in val_cols:
            p = piv[c].reindex(weeks).fillna(0)
            gp = teams_played.reindex(weeks).fillna(False).astype(int)
            cum = p.cumsum().shift(1); n = gp.cumsum().shift(1)
            team_avg = cum / n.replace(0, np.nan)
            lg_avg = cum.sum(axis=1) / n.sum(axis=1).replace(0, np.nan)
            rel = team_avg.div(lg_avg, axis=0) - 1
            rel = rel.stack().rename(f"dvp_{c}").reset_index().rename(columns={"level_1": "defteam"})
            rel["season"] = season; rel[pos_col] = pos
            out.append(rel.set_index(["season", "week", "defteam", pos_col]))
    D = pd.concat(out).groupby(level=[0, 1, 2, 3]).first()
    return df.join(D, on=["season", "week", "defteam", pos_col])


def add_positions(df, players):
    pos = players.set_index("gsis_id")
    df = df.join(pos[["position", "display_name"]].rename(columns={"display_name": "full_name"}), on="pid")
    df["pos"] = df.position.replace({"FB": "RB", "HB": "RB"})
    df.loc[~df.pos.isin(["QB", "RB", "WR", "TE"]), "pos"] = "OTH"
    for p in ["WR", "TE", "QB"]:
        df[f"pos_{p}"] = (df.pos == p).astype(float)
    df["name"] = df.full_name.fillna(df.name)
    return df


def skill_dvp(s):
    s = _dvp(s.assign(t_=s.tgt, r_=s.rec_yds, u_=s.rush_yds), ["t_", "r_", "u_"])
    return s.rename(columns={"dvp_t_": "dvp_tgt", "dvp_r_": "dvp_rec", "dvp_u_": "dvp_rush"}).drop(columns=["t_", "r_", "u_"])


def qb_dvp(q):
    q = _dvp(q.assign(pos="QB", y_=q.pass_yds), ["y_"])
    return q.rename(columns={"dvp_y_": "dvp_pass"}).drop(columns=["y_"])


def qb_features(q):
    q = _prior(q.copy(), ["pass_yds", "db"])
    q = _last_season(q, ["pass_yds"])
    q = _opp_allowed(q, {"db": "db", "y": "pass_yds"},
                     {"opp_ypd": lambda c, n: c["y"] / c["db"], "opp_db": lambda c, n: c["db"] / n})
    return q


def add_ly_diff(df, market):
    col = {"rec_yds": "rec_yds", "rush_yds": "rush_yds", "pass_yds": "pass_yds"}[market]
    return (df["ly_" + col] - df["p_" + col]).fillna(0)


def design(df, market, fill):
    X = df.copy()
    if market in ("rec_yds", "rush_yds", "pass_yds"):
        X["ly_diff"] = add_ly_diff(X, market)
    cols = FEATS[market]
    X = X[cols].astype(float)
    return X.fillna(pd.Series(fill, index=cols))


def project(PM, df, market):
    m = PM["markets"][market]
    X = design(df, market, m["fill"]).to_numpy()
    z = X @ np.array(m["w"]) + m["b"]
    if market == "td":
        return 1 / (1 + np.exp(-z))
    return np.clip(df[BASE[market]].to_numpy() + z, 0, None)
