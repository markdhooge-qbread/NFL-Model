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
`app.py` has five tabs: Slate (every game in a week, with finals graded), Matchup (any two teams, with injury inputs),
Teams (season stat profiles), Flow Chart (cumulative score-adjusted EPA and win probability for any finished game) and
Record (this season's hit rates on games the model never trained on).

- Run locally: `pip install -r requirements.txt`, then `streamlit run app.py`
- Deploy: on Streamlit Community Cloud, create an app from this repo, branch `main`, main file `app.py`.

The app downloads this season's and last season's nflverse play-by-play when it starts (cached for 6 hours), so team
ratings update on their own each week. Upcoming games use league-average injuries; set real ones in the Matchup tab.

## Retraining
`model.json` holds the trained weights (no scikit-learn needed in the app). Retrain once a season, or whenever the
model code changes:

    pip install scikit-learn pandas numpy
    python train.py        # downloads 2015-current from nflverse, about 5 minutes

`train.py` fits on every season before the current one, so the Record tab is graded on games the model never saw.
`injury_load_hist.csv` is the snap-weighted injury history used in training.

## How well it works
Fit on 2015-2022 and graded on 2023-2026 (903 games): winner 65%, halftime leader 61%, final margin, plays and drives
clearly better than guessing the average; who scores first, last before half and first after half about 55%; middle-eight
and 4th-quarter points close to no skill. The betting market's spread and total are still more accurate.

## Notebook
`notebooks/nfl_game_flow_model.ipynb` is the Colab version with the full backtest and the drop-one-stat test.
`flowlib.py` holds the feature code shared by the app and `train.py`.
