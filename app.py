"""NFL Game Flow Model (Streamlit).

Predicts how each game unfolds: points by quarter, halftime score, middle-eight points, total plays and drives,
and the chance the home team scores first, leads at half, scores last before half, scores first after half, and wins.

Weights come from model.json (built offline by train.py). On start the app downloads this season's and last
season's nflverse play-by-play to compute current team ratings.

Tabs: Slate, Matchup, Teams, Flow Chart, Record.
"""
import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import flowlib as fl

EST = ZoneInfo("America/New_York")
TODAY = datetime.datetime.now(EST).date()
SEASON = TODAY.year if TODAY.month >= 8 else TODAY.year - 1
HERE = Path(__file__).parent

st.set_page_config(page_title="NFL Game Flow Model", page_icon="🏈", layout="centered")


def st_html(content):
    lines = [ln.strip() for ln in str(content).splitlines()]
    st.markdown("\n".join(ln for ln in lines if ln), unsafe_allow_html=True)


st_html(
    """
    <style>
      .block-container {padding-top: 3rem; padding-left: 0.8rem; padding-right: 0.8rem;}
      .card {background:#0f172a;border:1px solid #334155;border-radius:10px;padding:10px;
             margin-bottom:12px;font-family:-apple-system,sans-serif;color:#f8fafc;}
      .card-head {display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px;margin-bottom:6px;}
      .card-title {font-size:15px;font-weight:800;}
      .card-sub {font-size:11px;color:#94a3b8;font-weight:400;}
      .summary {display:flex;flex-wrap:wrap;gap:6px 14px;justify-content:space-between;
                background:#1e293b;padding:8px 10px;border-radius:6px;margin-bottom:8px;font-size:13px;}
      .tbl-wrap {overflow-x:auto;}
      .tbl {width:100%;border-collapse:collapse;font-size:12px;color:#cbd5e1;}
      .tbl th {color:#94a3b8;background:#1e293b;padding:4px;font-weight:600;}
      .tbl td {padding:4px;text-align:center;border-bottom:1px solid #334155;}
      .tbl td:first-child, .tbl th:first-child {text-align:left;}
      .badge {padding:1px 5px;border-radius:4px;font-size:10px;font-weight:700;white-space:nowrap;}
      .hit {background:#14532d;color:#86efac;}
      .miss {background:#7f1d1d;color:#fca5a5;}
      .final {background:#334155;color:#cbd5e1;}
      .section {font-size:13px;font-weight:800;color:#38bdf8;margin:4px 0;}
      .meta {color:#94a3b8;font-size:12px;margin-bottom:8px;}
    </style>
    """)


# ----------------------------------------------------------------------------
# Data + model
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_model():
    return json.loads((HERE / "model.json").read_text())


@st.cache_data(ttl=6 * 3600, show_spinner="Loading nflverse play-by-play (about 30 seconds)...")
def load_data(season):
    raw = fl.load_pbp([season - 1, season])
    games = fl.load_games()
    inj_path = HERE / "injury_load_hist.csv"
    J = pd.read_csv(inj_path) if inj_path.exists() else None
    B = fl.build(raw, games, J)
    R = fl.current_ratings(B, season)
    keep = ["game_id", "posteam", "home_team", "away_team", "epa_adj", "wp", "game_seconds_remaining", "play_id"]
    sp = B["sp"][B["sp"].season == season][keep]
    TG = B["TG"][B["TG"].season == season]
    M = B["M"][B["M"].season == season]
    return games[games.season == season], R, M, TG, sp


MODEL = load_model()
FEATS = MODEL["features"]
INJ_DEFAULT = tuple(MODEL["inj_default"])
try:
    GAMES, RATINGS, MATCH, TEAMG, PLAYS = load_data(SEASON)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load nflverse data: {exc}")
    st.stop()


def predict_row(X):
    return {k: float(v[0]) for k, v in fl.predict(MODEL, X).items()}


def predict_upcoming(home, away, indoor, inj=None):
    X = fl.matchup_frame(RATINGS, home, away, indoor, FEATS, inj, INJ_DEFAULT)
    return predict_row(X)


def predict_played(game_id):
    row = MATCH[MATCH.game_id == game_id]
    if row.empty:
        return None
    X = row[FEATS].copy()
    inj = [f for f in FEATS if "inj" in f]
    X[inj] = X[inj].fillna(0)
    return predict_row(X)


def pct(x):
    return f"{100 * x:.0f}%"


def game_card(g, p, actual=None):
    a, h = g["away_team"], g["home_team"]
    kick = f"{g.get('weekday', '')} {g.get('gametime', '')} ET" if isinstance(g.get("gametime"), str) else ""
    fav, fav_p = (h, p["home_win"]) if p["home_win"] >= 0.5 else (a, 1 - p["home_win"])
    q = lambda side, n: p[f"{side}_q{n}"]
    rows = [(a, "a"), (h, "h")]
    body = "".join(
        f"<tr><td><b>{tm}</b></td>" + "".join(f"<td>{q(s, n):.1f}</td>" for n in range(1, 5)) +
        f"<td>{p[f'{s}_half']:.1f}</td><td>{p[f'{s}_mid8']:.1f}</td>"
        f"<td><b>{p['away_score' if s == 'a' else 'home_score']:.1f}</b></td></tr>" for tm, s in rows)
    act = ""
    if actual is not None:
        win_ok = (p["home_win"] >= .5) == (actual["result"] > 0)
        half_ok = (p["home_leads_half"] >= .5) == (actual.get("half_margin", 0) > 0)
        badge = lambda ok: "<span class='badge hit'>HIT</span>" if ok else "<span class='badge miss'>MISS</span>"
        act = (f"<div class='meta'><span class='badge final'>FINAL</span> {a} {actual['away_score']:.0f} – "
               f"{h} {actual['home_score']:.0f} · total {actual['total']:.0f} · winner {badge(win_ok)} · "
               f"half leader {badge(half_ok)}</div>")
    return f"""
    <div class='card'>
      <div class='card-head'><div class='card-title'>{a} @ {h}</div><div class='card-sub'>{kick}</div></div>
      {act}
      <div class='summary'>
        <span>Final <b>{p['away_score']:.1f}–{p['home_score']:.1f}</b></span>
        <span>{fav} <b>{pct(fav_p)}</b></span>
        <span>Total <b>{p['total']:.1f}</b></span>
        <span>Plays <b>{p['total_plays']:.0f}</b></span>
        <span>Drives <b>{p['total_drives']:.1f}</b></span>
      </div>
      <div class='tbl-wrap'><table class='tbl'>
        <tr><th>Team</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Half</th><th>Mid 8</th><th>Final</th></tr>
        {body}
      </table></div>
      <div class='meta' style='margin-top:8px'>
        {h} scores first <b>{pct(p['home_scores_first'])}</b> · leads at half <b>{pct(p['home_leads_half'])}</b> ·
        scores last 1H <b>{pct(p['home_last_1h'])}</b> · scores first 2H <b>{pct(p['home_first_2h'])}</b>
      </div>
    </div>"""


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
st_html(f"<div class='section'>🏈 NFL Game Flow Model · {SEASON} season</div>"
        f"<div class='meta'>Ratings use every {SEASON} game through the latest nflverse update, shrunk toward "
        f"{SEASON - 1} early in the year. Model trained on {MODEL['seasons'][0]}–{MODEL['seasons'][1]} "
        f"({MODEL['games']:,} games).</div>")

tab_slate, tab_match, tab_teams, tab_flow, tab_rec = st.tabs(["Slate", "Matchup", "Teams", "Flow Chart", "Record"])

with tab_slate:
    weeks = sorted(GAMES.week.unique())
    unplayed = GAMES[GAMES.result.isna()]
    default_week = int(unplayed.week.min()) if len(unplayed) else int(max(weeks))
    week = st.selectbox("Week", weeks, index=weeks.index(default_week), key="slate_week")
    wk = GAMES[GAMES.week == week].sort_values(["gameday", "gametime"])
    expand = st.toggle("Expand all games", key="slate_expand")
    for _, g in wk.iterrows():
        try:
            played = pd.notna(g.result)
            p = predict_played(g.game_id) if played else None
            if p is None:
                p = predict_upcoming(g.home_team, g.away_team, g.roof in ("dome", "closed"))
            actual = None
            if played:
                m = MATCH[MATCH.game_id == g.game_id]
                actual = {"result": g.result, "home_score": g.home_score, "away_score": g.away_score,
                          "total": g.total, "half_margin": float(m.half_margin.iloc[0]) if len(m) else 0}
            fav = g.home_team if p["home_win"] >= .5 else g.away_team
            label = (f"**{g.away_team} @ {g.home_team}** · {p['away_score']:.1f}–{p['home_score']:.1f}"
                     f" · {fav} {pct(max(p['home_win'], 1 - p['home_win']))} · Tot {p['total']:.1f}"
                     + (" · FINAL" if played else ""))
            with st.expander(label, expanded=expand):
                st_html(game_card(g, p, actual))
        except Exception as exc:  # noqa: BLE001
            st.error(f"{g.away_team} @ {g.home_team}: could not predict ({exc})")

with tab_match:
    teams = sorted(RATINGS.index)
    c1, c2 = st.columns(2)
    away = c1.selectbox("Away", teams, index=teams.index("MIA") if "MIA" in teams else 0)
    home = c2.selectbox("Home", teams, index=teams.index("BUF") if "BUF" in teams else 1)
    indoor = st.toggle("Indoor (dome or closed roof)", value=False)
    st_html("<div class='meta'>Injuries: starters out this week, weighted by snap share and position "
            f"(league average is about {INJ_DEFAULT[0]:.1f} on offense and {INJ_DEFAULT[1]:.1f} on defense).</div>")
    c3, c4, c5, c6 = st.columns(4)
    ai_o = c3.number_input(f"{away} off", 0.0, 10.0, round(INJ_DEFAULT[0], 1), 0.5)
    ai_d = c4.number_input(f"{away} def", 0.0, 10.0, round(INJ_DEFAULT[1], 1), 0.5)
    hi_o = c5.number_input(f"{home} off", 0.0, 10.0, round(INJ_DEFAULT[0], 1), 0.5)
    hi_d = c6.number_input(f"{home} def", 0.0, 10.0, round(INJ_DEFAULT[1], 1), 0.5)
    if home == away:
        st.warning("Pick two different teams.")
    else:
        p = predict_upcoming(home, away, indoor, {away: (ai_o, ai_d), home: (hi_o, hi_d)})
        st_html(game_card({"away_team": away, "home_team": home}, p))

with tab_teams:
    show = {"o_third_conv": "3rd% off", "d_third_conv": "3rd% def", "o_third_ytg": "3rd ytg off",
            "d_third_ytg": "3rd ytg def", "o_rz_trips": "RZ trips", "d_rz_trips": "RZ allowed",
            "o_rz_td_pct": "RZ TD% off", "d_rz_td_pct": "RZ TD% def", "mid8_f": "Mid8 for", "mid8_a": "Mid8 ag",
            "double_dip": "Double dip", "o_penalties": "Pen", "o_pen_yds": "Pen yds", "o_pace": "Pace",
            "o_plays": "Off plays", "d_plays": "Def plays", "o_pts_drive": "Pts/drive", "d_pts_drive": "Pts/drive ag",
            "o_epa_adj": "Adj EPA off", "d_epa_adj": "Adj EPA def", "o_early_sr": "Early SR", "o_sack_rate": "Sack% off",
            "o_explosive": "Explosive", "o_three_out": "3&out", "first_score": "Scores 1st"}
    prof = TEAMG.groupby("team")[list(show)].mean().rename(columns=show)
    prof.insert(0, "GP", TEAMG.groupby("team").size())
    prof["Total plays"] = prof["Off plays"] + prof["Def plays"]
    pct_cols = ["3rd% off", "3rd% def", "RZ TD% off", "RZ TD% def", "Double dip", "Early SR", "Sack% off",
                "Explosive", "3&out", "Scores 1st"]
    fmt = {c: "{:.0%}" for c in pct_cols}
    fmt.update({c: "{:.2f}" for c in ["Adj EPA off", "Adj EPA def", "Pts/drive", "Pts/drive ag"]})
    fmt.update({c: "{:.1f}" for c in prof.columns if c not in fmt and c != "GP"})
    st_html("<div class='meta'>Season averages per game. Pace = seconds per snap in neutral situations. "
            "Adj EPA = EPA per play after removing score-state effects (defense: lower is better).</div>")
    st.dataframe(prof.sort_values("Adj EPA off", ascending=False).style.format(fmt), width="stretch",
                 height=600)

with tab_flow:
    done = GAMES[GAMES.result.notna()].sort_values(["week", "gameday"], ascending=False)
    labels = [f"Wk {r.week}: {r.away_team} {r.away_score:.0f} @ {r.home_team} {r.home_score:.0f}" for r in done.itertuples()]
    if not labels:
        st.info("No finished games yet this season.")
    else:
        sel = st.selectbox("Game", labels, key="flow_game")
        gid = done.game_id.iloc[labels.index(sel)]
        x = PLAYS[PLAYS.game_id == gid].sort_values("play_id")
        h = done.home_team.iloc[labels.index(sel)]
        df = pd.DataFrame({
            "minute": (3600 - x.game_seconds_remaining) / 60,
            "flow": np.where(x.posteam == h, x.epa_adj, -x.epa_adj).cumsum(),
            "home_wp": np.where(x.posteam == h, x.wp, 1 - x.wp)})
        share = (df.flow > 0).mean()
        st_html(f"<div class='meta'>Blue line: cumulative score-adjusted EPA, home minus away (above 0 = {h} "
                f"controlling play). Orange line: {h} win probability. {h} was ahead on flow for {share:.0%} of plays."
                "</div>")
        base = alt.Chart(df).encode(x=alt.X("minute:Q", title="Game minute", scale=alt.Scale(domain=[0, 60])))
        flow = base.mark_line(color="#38bdf8", strokeWidth=2.5).encode(y=alt.Y("flow:Q", title="Flow (adj EPA)"))
        wp = base.mark_line(color="#f97316", opacity=.75).encode(
            y=alt.Y("home_wp:Q", title=f"{h} win prob", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")))
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#64748b").encode(y="y:Q")
        st.altair_chart(alt.layer(flow + zero, wp).resolve_scale(y="independent").properties(height=320),
                        width="stretch")

with tab_rec:
    rows = []
    for r in MATCH.itertuples():
        g = GAMES[GAMES.game_id == r.game_id]
        if g.empty or pd.isna(g.result.iloc[0]):
            continue
        p = predict_played(r.game_id)
        rows.append({"week": r.week, "game": f"{r.away_team} @ {r.home_team}",
                     "pred": f"{p['away_score']:.1f}–{p['home_score']:.1f}",
                     "final": f"{r.away_score:.0f}–{r.home_score:.0f}",
                     "winner": (p["home_win"] >= .5) == (r.result > 0),
                     "half_leader": (p["home_leads_half"] >= .5) == (r.half_margin > 0),
                     "first_score": (p["home_scores_first"] >= .5) == (r.home_scores_first == 1),
                     "margin_err": abs(p["result"] - r.result), "total_err": abs(p["total"] - r.total),
                     "plays_err": abs(p["total_plays"] - r.total_plays)})
    rec = pd.DataFrame(rows)
    if rec.empty:
        st.info("No graded games yet.")
    else:
        st_html(f"<div class='meta'>{SEASON} games the model never trained on (weeks 2+), using only data from "
                "before each game.</div>")
        c = st.columns(3)
        c[0].metric("Winner", f"{rec.winner.mean():.0%}", f"{rec.winner.sum()}–{(~rec.winner).sum()}")
        c[1].metric("Halftime leader", f"{rec.half_leader.mean():.0%}")
        c[2].metric("Scores first", f"{rec.first_score.mean():.0%}")
        c = st.columns(3)
        c[0].metric("Margin error", f"{rec.margin_err.mean():.1f} pts")
        c[1].metric("Total error", f"{rec.total_err.mean():.1f} pts")
        c[2].metric("Plays error", f"{rec.plays_err.mean():.1f}")
        st.dataframe(rec.sort_values("week", ascending=False), width="stretch", hide_index=True)
    with st.expander("How the model was graded (2023+ backtest)"):
        gr = pd.DataFrame(MODEL["grades"])
        st.dataframe(gr.round(3), width="stretch", hide_index=True)
        st_html("<div class='meta'>Fit on seasons before 2023, graded on 2023 onward. gain_vs_avg_pct = how much "
                "the model beats guessing the league average every game. Quarter, middle-eight and first-score "
                "predictions are close to coin flips; winner, margin, halftime lead, plays and drives carry real signal."
                "</div>")
