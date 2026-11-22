"""Offline training for the Players tab. Writes player_model.json (run after train.py, once a season).

    python player_train.py --local ~/d --last 2026

Backtest: fit on 2015-2022, graded on 2023+ (weeks with 3+ prior games). Final weights: every season before the
current one. Team context = the game flow model's pregame predictions (team points, margin, total plays).
"""
import argparse, json, datetime
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import flowlib as fl
import playerlib as pl

ap = argparse.ArgumentParser()
ap.add_argument("--local", default=None)
ap.add_argument("--injuries", default="injury_load_hist.csv")
ap.add_argument("--first", type=int, default=2015)
ap.add_argument("--last", type=int, default=datetime.date.today().year if datetime.date.today().month >= 8
                else datetime.date.today().year - 1)
ap.add_argument("--test-from", type=int, default=2023)
a = ap.parse_args()

raw = fl.load_pbp(range(a.first, a.last + 1), a.local, extra=pl.PCOLS)
games = fl.load_games(a.local)
try:
    J = pd.read_csv(a.injuries)
except FileNotFoundError:
    J = None
B = fl.build(raw, games, J)
M = B["M"].copy(); FEATS = B["FEATS"]
M[[f for f in FEATS if "inj" in f]] = M[[f for f in FEATS if "inj" in f]].fillna(0)
M = M.dropna(subset=FEATS)


def flow_ctx(fit_rows):
    preds = {}
    for t in ["home_score", "away_score", "total_plays"]:
        d = fit_rows.dropna(subset=[t])
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 20))).fit(d[FEATS], d[t])
        preds[t] = m.predict(M[FEATS])
    h = pd.DataFrame({"game_id": M.game_id, "posteam": M.home_team, "c_pts": preds["home_score"],
                      "c_margin": preds["home_score"] - preds["away_score"], "c_plays": preds["total_plays"], "c_home": 1.0})
    w = pd.DataFrame({"game_id": M.game_id, "posteam": M.away_team, "c_pts": preds["away_score"],
                      "c_margin": preds["away_score"] - preds["home_score"], "c_plays": preds["total_plays"], "c_home": 0.0})
    return pd.concat([h, w])


PLAYERS = pd.read_csv(f"{a.local}/players.csv" if a.local else
                      "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv", low_memory=False)
s, q = pl.player_games(raw)
S = pl.skill_dvp(pl.skill_features(pl.add_positions(s.assign(fut=0), PLAYERS)))
Q = pl.qb_dvp(pl.qb_features(q.assign(fut=0)))
ELIG = {
    "rec_yds": lambda d: (d.n >= 3) & (d.p_tgt >= 2.5),
    "rec": lambda d: (d.n >= 3) & (d.p_tgt >= 2.5),
    "rush_yds": lambda d: (d.n >= 3) & (d.p_car >= 4),
    "car": lambda d: (d.n >= 3) & (d.p_car >= 4),
    "pass_yds": lambda d: (d.n >= 3) & (d.p_db >= 20) & (d.db >= 15),
    "td": lambda d: (d.n >= 3) & ((d.p_tgt + d.p_car) >= 3),
}
TARGET = {"rec_yds": "rec_yds", "rec": "rec", "rush_yds": "rush_yds", "car": "car", "pass_yds": "pass_yds", "td": "td"}


def frame(market, ctx):
    d = (Q if market == "pass_yds" else S).merge(ctx, on=["game_id", "posteam"], how="inner")
    d = d[ELIG[market](d)].copy()
    if market in ("rec_yds", "rush_yds", "pass_yds"):
        d["ly_diff"] = pl.add_ly_diff(d, market)
    return d


def fit(market, d):
    cols = pl.FEATS[market]
    fill = d[cols].median()
    X = d[cols].fillna(fill)
    if market == "td":
        m = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=8, max_iter=3000)).fit(X, d.td)
        coef, b0 = m[-1].coef_[0], m[-1].intercept_[0]
    else:
        y = d[TARGET[market]] - d[pl.BASE[market]]
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 20))).fit(X, y)
        coef, b0 = m[-1].coef_, m[-1].intercept_
    sc = m[0]
    return {"w": [float(v) for v in coef / sc.scale_], "b": float(b0 - np.sum(coef * sc.mean_ / sc.scale_)),
            "fill": [float(v) for v in fill]}


ctx_bt = flow_ctx(M[M.season < a.test_from])
grades = {}
for mk in pl.FEATS:
    d = frame(mk, ctx_bt)
    tr, te = d[d.season < a.test_from], d[d.season >= a.test_from].copy()
    w = fit(mk, tr)
    te_proj = pl.project({"markets": {mk: w}}, te, mk)
    if mk == "td":
        base = tr.td.mean()
        g = {"n": int(len(te)), "brier_model": float(((te_proj - te.td) ** 2).mean()),
             "brier_season_rate": float(((te.p_td.clip(0, 1) - te.td) ** 2).mean()),
             "calibration": pd.DataFrame({"p": te_proj, "y": te.td}).groupby(pd.cut(te_proj, [0, .15, .3, .45, .6, 1]),
                                         observed=True).agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
                                         .round(3).reset_index(drop=True).to_dict("records")}
    else:
        y, base = te[TARGET[mk]], te[pl.BASE[mk]]
        diff = te_proj - base
        mn = pl.FLAG_MIN[mk]
        over = (diff >= mn) & (diff / base >= .10)
        under = (diff <= -mn) & (diff / base <= -.10)
        g = {"n": int(len(te)), "mae_season_avg": float(np.abs(y - base).mean()), "mae_model": float(np.abs(y - te_proj).mean()),
             "over_flags": int(over.sum()), "over_beat_avg": float((y[over] > base[over]).mean()),
             "base_beat_avg": float((y > base).mean()),
             "under_flags": int(under.sum()), "under_below_avg": float((y[under] < base[under]).mean()),
             "base_below_avg": float((y < base).mean()), "resid_sd": float((y - te_proj).std()),
             "median_minus_avg": float((y - te_proj).median())}
    grades[mk] = g
    print(mk, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items() if k != "calibration"})
    if mk == "td": print(pd.DataFrame(g["calibration"]))

# final weights on every season before the current one, with team context from the full flow model
ctx_final = flow_ctx(M[M.season < a.last])
markets = {}
for mk in pl.FEATS:
    d = frame(mk, ctx_final)
    markets[mk] = fit(mk, d[d.season < a.last])
    if mk != "td":
        markets[mk]["sd"] = grades[mk]["resid_sd"]; markets[mk]["median_shift"] = grades[mk]["median_minus_avg"]
json.dump({"trained": datetime.date.today().isoformat(), "seasons": [a.first, a.last - 1], "test_from": a.test_from,
           "markets": markets, "grades": grades}, open("player_model.json", "w"))
print("wrote player_model.json")
