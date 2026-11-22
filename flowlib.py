"""Shared NFL game flow feature code (used by train.py offline and app.py live)."""
import numpy as np
import pandas as pd

FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
K = 3
COLS = ["game_id", "play_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team",
        "play_type", "pass", "rush", "qb_dropback", "qb_kneel", "qb_spike", "epa", "success", "wp",
        "score_differential", "qtr", "down", "ydstogo", "yardline_100", "yards_gained", "game_seconds_remaining",
        "half_seconds_remaining", "fixed_drive", "fixed_drive_result", "third_down_converted", "third_down_failed",
        "fourth_down_converted", "fourth_down_failed", "penalty", "penalty_team", "penalty_yards", "sack",
        "interception", "fumble_lost", "field_goal_result", "total_home_score", "total_away_score"]
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.csv.gz"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
NUM_T = ["h_q1", "a_q1", "h_q2", "a_q2", "h_q3", "a_q3", "h_q4", "a_q4", "h_half", "a_half", "half_margin",
         "h_mid8", "a_mid8", "mid8_total", "home_score", "away_score", "result", "total", "total_plays", "total_drives"]
BIN_T = ["home_scores_first", "home_leads_half", "home_last_1h", "home_first_2h", "home_win"]


def load_pbp(seasons, local=None):
    src = (lambda y: f"{local}/pbp_{y}.csv.gz") if local else (lambda y: PBP_URL.format(y=y))
    return pd.concat([pd.read_csv(src(y), usecols=COLS, low_memory=False) for y in seasons], ignore_index=True)


def load_games(local=None):
    games = pd.read_csv(f"{local}/games.csv" if local else GAMES_URL)
    for c in ["home_team", "away_team"]:
        games[c] = games[c].replace(FIX)
    return games[games.game_type == "REG"].copy()


def build(raw, games, J=None):
    """Returns dict: TG (team-game stats + pregame ratings), GF (game outcomes), M (matchup rows),
    sp (scrimmage plays), STATS, FEATS, prev (last-season averages), lg (league averages by season)."""

    raw = raw[raw.season_type == "REG"].sort_values(["game_id", "play_id"]).copy()
    for side in ["home", "away"]:
        s = raw.groupby("game_id")[f"total_{side}_score"].transform(lambda x: x.ffill().fillna(0))
        raw[f"{side}_pts_play"] = s.groupby(raw.game_id).diff().fillna(s).clip(lower=0)
    sc = raw[(raw.home_pts_play > 0) | (raw.away_pts_play > 0)].copy()
    sc["scorer"] = np.where(sc.home_pts_play > 0, sc.home_team, sc.away_team)
    sc["half"] = np.where(sc.qtr <= 2, 1, np.where(sc.qtr <= 4, 2, 3))
    sc["mid8"] = ((sc.qtr == 2) & (sc.half_seconds_remaining <= 240)) | ((sc.qtr == 3) & (sc.half_seconds_remaining >= 1560))
    sc["two_min"] = sc.qtr.isin([2, 4]) & (sc.half_seconds_remaining <= 120)

    def pts_by(mask_col=None, qtr=None):
        d = sc if mask_col is None else sc[sc[mask_col]]
        if qtr is not None: d = d[d.qtr == qtr]
        return d.groupby("game_id")[["home_pts_play", "away_pts_play"]].sum()

    GF = games[games.result.notna()][["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score",
                                      "result", "total", "spread_line", "total_line", "roof"]].set_index("game_id")
    for q in [1, 2, 3, 4]:
        GF = GF.join(pts_by(qtr=q).rename(columns={"home_pts_play": f"h_q{q}", "away_pts_play": f"a_q{q}"}))
    GF = GF.join(pts_by("mid8").rename(columns={"home_pts_play": "h_mid8", "away_pts_play": "a_mid8"}))
    GF = GF.join(pts_by("two_min").rename(columns={"home_pts_play": "h_2min", "away_pts_play": "a_2min"}))
    GF = GF.fillna({c: 0 for c in GF.columns if c[:2] in ("h_", "a_")})
    first = sc.groupby("game_id").first()
    last1 = sc[sc.half == 1].groupby("game_id").last()
    first2 = sc[sc.half == 2].groupby("game_id").first()
    GF["home_scores_first"] = (first.scorer == first.home_team).astype(float).reindex(GF.index)
    GF["home_last_1h"] = (last1.scorer == last1.home_team).astype(float).reindex(GF.index)
    GF["home_first_2h"] = (first2.scorer == first2.home_team).astype(float).reindex(GF.index)
    GF["h_half"] = GF.h_q1 + GF.h_q2; GF["a_half"] = GF.a_q1 + GF.a_q2
    GF["half_margin"] = GF.h_half - GF.a_half
    GF["home_leads_half"] = (GF.half_margin > 0).astype(float)
    GF["home_win"] = (GF.result > 0).astype(float)
    GF["mid8_total"] = GF.h_mid8 + GF.a_mid8


    sp = raw[(raw["pass"].eq(1) | raw["rush"].eq(1)) & raw.qb_kneel.ne(1) & raw.qb_spike.ne(1) & raw.epa.notna()].copy()
    sp["state"] = pd.cut(sp.score_differential, [-99, -17, -9, -4, -1, 0, 3, 8, 16, 99]).astype(str) + sp.qtr.clip(upper=4).astype(str)
    sp["epa_adj"] = sp.epa - sp.groupby(["season", "state"]).epa.transform("mean")
    sp["explosive"] = ((sp["pass"] == 1) & (sp.yards_gained >= 20)) | ((sp["rush"] == 1) & (sp.yards_gained >= 10))
    sp["early"] = sp.down.isin([1, 2])
    sp["neutral"] = (sp.qtr <= 3) & (sp.score_differential.abs() <= 7)
    sp["lead7"] = sp.score_differential >= 7
    sp["trail7"] = sp.score_differential <= -7
    sp["h2"] = sp.qtr >= 3
    sp["giveaway"] = sp.interception.fillna(0) + sp.fumble_lost.fillna(0)

    def agg_side(d, key):
        g = d.groupby(["game_id", key])
        out = pd.DataFrame({
            "plays": g.size(),
            "epa": g.epa.mean(),
            "epa_adj": g.epa_adj.mean(),
            "early_sr": d[d.early].groupby(["game_id", key]).success.mean(),
            "explosive": g.explosive.mean(),
            "sack_rate": d[d.qb_dropback == 1].groupby(["game_id", key]).sack.mean(),
            "giveaways": g.giveaway.sum(),
            "neutral_pass": d[d.neutral].groupby(["game_id", key])["pass"].mean(),
            "pass_lead": d[d.lead7].groupby(["game_id", key])["pass"].mean(),
            "pass_trail": d[d.trail7].groupby(["game_id", key])["pass"].mean(),
            "epa_2h_minus_1h": d[d.h2].groupby(["game_id", key]).epa.mean() - d[~d.h2].groupby(["game_id", key]).epa.mean(),
        })
        out.index.names = ["game_id", "team"]
        return out

    dn = raw[raw.down.isin([3, 4]) & raw.posteam.notna()].copy()
    d3 = dn[(dn.down == 3) & (dn.third_down_converted.fillna(0) + dn.third_down_failed.fillna(0) > 0)]
    d4 = dn[(dn.down == 4) & dn.play_type.isin(["pass", "run", "punt", "field_goal"])]
    def third(d, key):
        g = d.groupby(["game_id", key])
        o = pd.DataFrame({"third_conv": g.third_down_converted.mean(), "third_ytg": g.ydstogo.mean(), "third_att": g.size()})
        o.index.names = ["game_id", "team"]; return o
    def fourth(d, key):
        o = d.assign(go=d.play_type.isin(["pass", "run"])).groupby(["game_id", key]).go.mean().to_frame("fourth_go")
        o.index.names = ["game_id", "team"]; return o

    dr = (sp.groupby(["game_id", "posteam", "fixed_drive"])
            .agg(defteam=("defteam", "first"), result=("fixed_drive_result", "first"), min_yl=("yardline_100", "min"),
                 start_yl=("yardline_100", "first"), n=("play_id", "size")).reset_index())
    dr["pts"] = dr.result.map({"Touchdown": 7, "Field goal": 3}).fillna(0)
    dr["rz"] = dr.min_yl <= 20
    dr["rz_td"] = dr.rz & (dr.result == "Touchdown")
    dr["three_out"] = (dr.n <= 3) & (dr.result == "Punt")
    dr["own_start"] = 100 - dr.start_yl
    def drives(d, key):
        g = d.groupby(["game_id", key])
        o = pd.DataFrame({"drives": g.size(), "pts_drive": g.pts.mean(), "rz_trips": g.rz.sum(),
                          "rz_td_pct": d[d.rz].groupby(["game_id", key]).rz_td.mean(), "three_out": g.three_out.mean(),
                          "start_fp": g.own_start.mean()})
        o.index.names = ["game_id", "team"]; return o

    pc = sp[sp.neutral].sort_values(["game_id", "play_id"]).copy()
    pc["gap"] = -pc.groupby(["game_id", "fixed_drive", "qtr"]).game_seconds_remaining.diff()
    pace = pc[(pc.gap > 0) & (pc.gap <= 60)].groupby(["game_id", "posteam"]).gap.mean().to_frame("pace")
    pace.index.names = ["game_id", "team"]

    pen = raw[raw.penalty == 1].groupby(["game_id", "penalty_team"]).agg(penalties=("penalty", "size"), pen_yds=("penalty_yards", "sum"))
    pen.index.names = ["game_id", "team"]
    st = raw[raw.play_type.isin(["punt", "kickoff", "field_goal", "extra_point"]) & raw.epa.notna()]
    st_for = st.groupby(["game_id", "posteam"]).epa.sum(); st_ag = st.groupby(["game_id", "defteam"]).epa.sum()
    st_for.index.names = st_ag.index.names = ["game_id", "team"]
    stt = st_for.sub(st_ag, fill_value=0).to_frame("st_epa")
    fg = raw[raw.play_type == "field_goal"].assign(made=lambda x: x.field_goal_result.eq("made"))
    fgp = fg.groupby(["game_id", "posteam"]).made.mean().to_frame("fg_pct"); fgp.index.names = ["game_id", "team"]

    OFF = agg_side(sp, "posteam").join(third(d3, "posteam")).join(fourth(d4, "posteam")).join(drives(dr, "posteam")) \
            .join(pace).join(pen).join(stt).join(fgp)
    DEF = agg_side(sp, "defteam").join(third(d3, "defteam")).join(drives(dr, "defteam"))
    DEF = DEF[["plays", "epa", "epa_adj", "early_sr", "explosive", "sack_rate", "giveaways", "third_conv", "third_ytg",
               "drives", "pts_drive", "rz_trips", "rz_td_pct", "three_out", "start_fp"]]
    TG = OFF.add_prefix("o_").join(DEF.add_prefix("d_")).reset_index()
    TG["team"] = TG.team.replace(FIX)
    TG[["o_penalties", "o_pen_yds"]] = TG[["o_penalties", "o_pen_yds"]].fillna(0)

    long = pd.concat([
        GF.reset_index().assign(team=lambda x: x.home_team, opp=lambda x: x.away_team, pf=lambda x: x.home_score,
            pa=lambda x: x.away_score, mid8_f=lambda x: x.h_mid8, mid8_a=lambda x: x.a_mid8,
            twomin_f=lambda x: x.h_2min, first_score=lambda x: x.home_scores_first, last1h=lambda x: x.home_last_1h,
            first2h=lambda x: x.home_first_2h, half_m=lambda x: x.half_margin, home=1,
            **{f"q{q}_f": (lambda q: lambda x: x[f"h_q{q}"])(q) for q in range(1, 5)},
            **{f"q{q}_a": (lambda q: lambda x: x[f"a_q{q}"])(q) for q in range(1, 5)}),
        GF.reset_index().assign(team=lambda x: x.away_team, opp=lambda x: x.home_team, pf=lambda x: x.away_score,
            pa=lambda x: x.home_score, mid8_f=lambda x: x.a_mid8, mid8_a=lambda x: x.h_mid8,
            twomin_f=lambda x: x.a_2min, first_score=lambda x: 1 - x.home_scores_first, last1h=lambda x: 1 - x.home_last_1h,
            first2h=lambda x: 1 - x.home_first_2h, half_m=lambda x: -x.half_margin, home=0,
            **{f"q{q}_f": (lambda q: lambda x: x[f"a_q{q}"])(q) for q in range(1, 5)},
            **{f"q{q}_a": (lambda q: lambda x: x[f"h_q{q}"])(q) for q in range(1, 5)})])
    long["double_dip"] = long.last1h * long.first2h      # scored last in 1H AND first in 2H
    long["blown_lead"] = ((long.half_m > 0) & (long.pf < long.pa)).astype(float)
    long["comeback"] = ((long.half_m < 0) & (long.pf > long.pa)).astype(float)
    TG = TG.merge(long[["game_id", "team", "season", "week", "home", "pf", "pa", "mid8_f", "mid8_a", "twomin_f",
                        "first_score", "last1h", "first2h", "double_dip", "blown_lead", "comeback"] +
                       [f"q{q}_{s}" for q in range(1, 5) for s in "fa"]], on=["game_id", "team"])

    if J is not None:
        J = J.rename(columns={"off": "inj_off", "def": "inj_def"})
        J["team"] = J.team.replace(FIX)
        TG = TG.merge(J, on=["team", "season", "week"], how="left").fillna({"inj_off": 0, "inj_def": 0})
    else:
        TG["inj_off"] = np.nan; TG["inj_def"] = np.nan


    STATS = [c for c in TG.columns if c.startswith(("o_", "d_"))] + \
            ["pf", "pa", "mid8_f", "mid8_a", "twomin_f", "first_score", "last1h", "first2h", "double_dip",
             "blown_lead", "comeback"] + [f"q{q}_{s}" for q in range(1, 5) for s in "fa"]
    K = 3
    TG = TG.sort_values(["team", "season", "week"]).reset_index(drop=True)
    season_avg = TG.groupby(["team", "season"])[STATS].mean()
    prev = season_avg.copy(); prev.index = prev.index.set_levels(prev.index.levels[1] + 1, level=1)
    lg = TG.groupby("season")[STATS].mean()
    grp = TG.groupby(["team", "season"])
    sh = grp[STATS].shift()
    cum = sh.fillna(0).groupby([TG.team, TG.season]).cumsum()
    cnt = sh.notna().astype(int).groupby([TG.team, TG.season]).cumsum()
    pri = prev.reindex(pd.MultiIndex.from_frame(TG[["team", "season"]])).reset_index(drop=True)
    pri = pri.fillna(lg.reindex(TG.season).reset_index(drop=True))
    R = (cum + K * pri) / (cnt + K)
    R.columns = ["r_" + c for c in STATS]
    TG = pd.concat([TG, R], axis=1)
    TG["games_played"] = grp.cumcount()
    RCOLS = list(R.columns)
    RCOLS += ["inj_off", "inj_def"]   # injuries are known pregame, used as-is for that week


    h = TG[["game_id", "team"] + RCOLS + ["games_played"]].add_prefix("H_").rename(columns={"H_game_id": "game_id", "H_team": "home_team"})
    a = TG[["game_id", "team"] + RCOLS + ["games_played"]].add_prefix("A_").rename(columns={"A_game_id": "game_id", "A_team": "away_team"})
    M = GF.reset_index().merge(h, on=["game_id", "home_team"]).merge(a, on=["game_id", "away_team"])
    M = M[M.week >= 2].copy()
    M["indoor"] = M.roof.isin(["dome", "closed"]).astype(float)
    FEATS = [c for c in M.columns if c.startswith(("H_r_", "A_r_", "H_inj", "A_inj"))] + ["indoor"]
    plays = TG.groupby("game_id").o_plays.sum(); drv = TG.groupby("game_id").o_drives.sum()
    M["total_plays"] = M.game_id.map(plays); M["total_drives"] = M.game_id.map(drv)
    return dict(TG=TG, GF=GF, M=M, sp=sp, STATS=STATS, FEATS=FEATS, prev=prev, lg=lg)


def current_ratings(B, season):
    """Each team's latest rating (all games so far this season, shrunk toward last season)."""
    TG, STATS, prev, lg = B["TG"], B["STATS"], B["prev"], B["lg"]
    out = {}
    cur = TG[TG.season == season]
    for team, d in cur.groupby("team"):
        p = prev.loc[(team, season)] if (team, season) in prev.index else lg.loc[season]
        p = p.fillna(lg.loc[season])
        s = d[STATS].sum(); n = d[STATS].notna().sum()
        out[team] = ((s + K * p) / (n + K)).add_prefix("r_")
    return pd.DataFrame(out).T


def matchup_frame(R, home, away, indoor, feats, inj=None, inj_default=(0.0, 0.0)):
    inj = inj or {}
    r = {}
    for side, tm in (("H_", home), ("A_", away)):
        for k, v in R.loc[tm].items(): r[side + k] = v
        io, idf = inj.get(tm, inj_default)
        r[side + "inj_off"], r[side + "inj_def"] = io, idf
    r["indoor"] = float(indoor)
    return pd.DataFrame([r])[feats]


def predict(model, X):
    """model: dict from model.json. X: DataFrame with model['features'] columns. Returns dict of predictions."""
    x = X[model["features"]].to_numpy(dtype=float)
    x = np.where(np.isnan(x), np.array(model["fill"])[None, :], x)
    out = {}
    for t, m in model["targets"].items():
        z = x @ np.array(m["w"]) + m["b"]
        out[t] = 1 / (1 + np.exp(-z)) if m["kind"] == "bin" else z
    return out
