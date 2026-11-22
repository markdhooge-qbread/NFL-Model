"""Offline training for the NFL game flow app. Run once a week (or season) and commit model.json.

    python train.py                      # downloads 2015-current from nflverse (~5 min)
    python train.py --local ~/d          # use local pbp_YYYY.csv.gz + games.csv

Writes model.json: feature list, fill values, and plain linear / logistic weights for every flow outcome,
plus the out-of-sample grades (fit 2015-2022, graded on 2023+). The app needs no scikit-learn.
"""
import argparse, json, datetime
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import flowlib as fl

ap = argparse.ArgumentParser()
ap.add_argument("--local", default=None)
ap.add_argument("--injuries", default="injury_load_hist.csv")
ap.add_argument("--first", type=int, default=2015)
ap.add_argument("--last", type=int, default=datetime.date.today().year if datetime.date.today().month >= 9
                else datetime.date.today().year - 1)
ap.add_argument("--test-from", type=int, default=2023)
a = ap.parse_args()

raw = fl.load_pbp(range(a.first, a.last + 1), a.local)
games = fl.load_games(a.local)
try:
    J = pd.read_csv(a.injuries)
except FileNotFoundError:
    J = None
B = fl.build(raw, games, J)
M, FEATS = B["M"], B["FEATS"]
M = M.dropna(subset=[f for f in FEATS if "inj" not in f])
M[[f for f in FEATS if "inj" in f]] = M[[f for f in FEATS if "inj" in f]].fillna(0)
fill = M[FEATS].mean()


def fit(t, d):
    d = d.dropna(subset=[t])
    if t in fl.BIN_T:
        m = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, max_iter=3000)).fit(d[FEATS], d[t])
        coef, b0 = m[-1].coef_[0], m[-1].intercept_[0]
    else:
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 20))).fit(d[FEATS], d[t])
        coef, b0 = m[-1].coef_, m[-1].intercept_
    sc = m[0]
    w = coef / sc.scale_; b = b0 - float(np.sum(coef * sc.mean_ / sc.scale_))
    return {"kind": "bin" if t in fl.BIN_T else "num", "w": [float(v) for v in w], "b": float(b)}, m


# out-of-sample grades
train, test = M[M.season < a.test_from], M[M.season >= a.test_from]
grades = []
for t in fl.NUM_T + fl.BIN_T:
    _, m = fit(t, train); te = test.dropna(subset=[t]); base = train[t].mean()
    if t in fl.BIN_T:
        p = m.predict_proba(te[FEATS])[:, 1]
        grades.append({"outcome": t, "metric": "hit rate", "model": float(((p > .5) == (te[t] > .5)).mean()),
                       "gain_vs_avg_pct": float(100 * (1 - ((p - te[t]) ** 2).mean() / ((base - te[t]) ** 2).mean()))})
    else:
        p = m.predict(te[FEATS])
        grades.append({"outcome": t, "metric": "MAE", "model": float(np.abs(te[t] - p).mean()),
                       "gain_vs_avg_pct": float(100 * (1 - np.abs(te[t] - p).mean() / np.abs(te[t] - base).mean()))})

# final model: every season before the current one, so the app's Record tab is graded on unseen games
final = M[M.season < a.last]
targets = {t: fit(t, final)[0] for t in fl.NUM_T + fl.BIN_T}
inj_default = [float(B["TG"].inj_off.fillna(0).mean()), float(B["TG"].inj_def.fillna(0).mean())]
json.dump({"trained": datetime.date.today().isoformat(), "seasons": [a.first, a.last - 1], "games": int(len(final)),
           "test_from": a.test_from, "test_games": int(len(test)), "features": FEATS,
           "fill": [float(v) for v in fill], "inj_default": inj_default, "targets": targets, "grades": grades},
          open("model.json", "w"))
print("wrote model.json:", len(final), "games,", len(FEATS), "features")
print(pd.DataFrame(grades).round(3).to_string(index=False))
