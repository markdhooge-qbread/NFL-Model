# NFL Model

NFL game flow model: predicts how each game unfolds, adapted from the NHL Game Flow Model app to NFL play-by-play.

## What it predicts
For every game: points by quarter for each team, halftime score, middle-eight points, final score, total, total plays
(both teams), total drives, and the chance the home team scores first, leads at half, scores last before half, scores
first after half, and wins.

Inputs (season to date, shrunk toward last season early in the year): 3rd down % and 3rd-down distance, red zone trips
and TD %, middle-eight points, double dip (scored last in 1H and first in 2H), penalties and penalty yards, injuries,
pace, plays, points per drive, drives, three-and-outs, field position, early-down success, sack rate, explosive plays,
giveaways, pass rate leading and trailing, 4th down go rate, two-minute points, quarter splits, special teams, and
score-adjusted EPA (EPA minus the league average for the same score and quarter, the NFL version of hockey score effects).

## Streamlit app
`app.py` has six tabs: Slate (every game in a week, with finals graded), Matchup (any two teams, with injury inputs), Players (player projections with defense vs position and a DK line check),
Teams (season stat profiles), Flow Chart (cumulative score-adjusted EPA and win probability for any finished game) and
Record (this season's hit rates on games the model never trained on).

- Run locally: `pip install -r requirements.txt`, then `streamlit run app.py`
- Deploy: on Streamlit Community Cloud, create an app from this repo, branch `main`, main file `app.py`.

The app downloads this season's and last season's nflverse play-by-play when it starts (cached for 6 hours), so team
ratings update on their own each week. Upcoming games use league-average injuries; set real ones in the Matchup tab.

## Players tab
For each game: QB passing yards and every regular RB/WR/TE's receptions, receiving yards, carries, rushing yards and
anytime TD chance. A projection starts from the player's season average and adjusts for last season, role trend
(target/carry share last 3 games), efficiency regression, what the defense allows, **defense vs position** (yards and
targets this defense allows to WRs, TEs, RBs, QBs vs league average), and the game flow model's script for the game
(team points, margin, total plays). Finished games show projection vs actual. Out/Doubtful players are removed using
nflverse injury reports.

Backtest (fit 2015-22, graded 2023-26): projections beat the season average on every market (rec yds MAE 22.4 vs 23.2,
pass yds 59.1 vs 61.4). Flags (projection 10%+ off average) went: rec yds over 57% / under 71% (base 42% / 58%),
rush yds over 54% / under 66%, pass yds over 71% / under 62%. Anytime TD is well calibrated. Defense vs position on its
own is weaker than the player's average (WR/TE corr ~0.04); it matters more for RB rushing and QB passing (~0.14).
None of this is tested against real DraftKings lines.

## Retraining
`model.json` holds the trained weights (no scikit-learn needed in the app). Retrain once a season, or whenever the
model code changes:

    pip install scikit-learn pandas numpy
    python train.py          # game flow weights -> model.json (about 5 minutes)
    python player_train.py   # player weights -> player_model.json (about 5 minutes)

`train.py` fits on every season before the current one, so the Record tab is graded on games the model never saw.
`injury_load_hist.csv` is the snap-weighted injury history used in training.

## How well it works
Fit on 2015-2022 and graded on 2023-2026 (903 games): winner 65%, halftime leader 61%, final margin, plays and drives
clearly better than guessing the average; who scores first, last before half and first after half about 55%; middle-eight
and 4th-quarter points close to no skill. The betting market's spread and total are still more accurate.

## Notebook
`notebooks/nfl_game_flow_model.ipynb` is the Colab version with the full backtest and the drop-one-stat test.
`flowlib.py` holds the feature code shared by the app and `train.py`.
