"""NFL Game Flow Model (Streamlit).

Predicts how each game unfolds: points by quarter, halftime score, middle-eight points, total plays and drives,
and the chance the home team scores first, leads at half, scores last before half, scores first after half, and wins.

Weights come from model.json (built offline by train.py). On start the app downloads this season's and last
season's nflverse play-by-play to compute current team ratings.

Tabs: Bets (every pick, graded live), Games (model preview, then live score), Players, Model (Teams, Any matchup,
Flow Chart, Accuracy). Each stat lives on one tab: picks and their grading only on Bets.
"""
import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import importlib

import flowlib as fl
import livelib as lv
import pagelib as pg
import playerlib as pl

# Streamlit Cloud keeps imported helper modules in memory across code updates, so a new app.py can run against an
# old flowlib.py/playerlib.py until the app is rebooted. Reload them so app.py and its helpers always match.
fl = importlib.reload(fl)
pl = importlib.reload(pl)
lv = importlib.reload(lv)
pg = importlib.reload(pg)

EST = ZoneInfo("America/New_York")
TODAY = datetime.datetime.now(EST).date()
SEASON = TODAY.year if TODAY.month >= 8 else TODAY.year - 1
HERE = Path(__file__).parent
PLAYERS_URL = "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv"
INJ_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"

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
      .pend {background:#1e293b;color:#94a3b8;border:1px solid #334155;}
      .push {background:#713f12;color:#fde68a;}
      .livedot {display:inline-block;width:8px;height:8px;border-radius:50%;background:#ef4444;margin-right:5px;}
      .score {display:flex;justify-content:space-between;align-items:center;font-size:22px;font-weight:800;
              padding:4px 2px 6px 2px;}
      .score .tm {display:flex;align-items:center;gap:6px;}
      .poss {font-size:12px;color:#fbbf24;}
      .bet-row {padding:7px 8px;border-bottom:1px solid #334155;font-size:13px;}
      .bet-top {display:flex;justify-content:space-between;align-items:center;gap:6px;}
      .bet-stats {font-size:12px;color:#cbd5e1;margin:3px 0 1px 0;}
      .flag-row {display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font-size:12px;
                 padding:5px 0;border-top:1px solid #334155;color:#cbd5e1;}
    </style>
    """)


# ----------------------------------------------------------------------------
# Data + model
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_model():
    return json.loads((HERE / "model.json").read_text())


@st.cache_data(show_spinner=False)
def load_player_model():
    return json.loads((HERE / "player_model.json").read_text())


@st.cache_data(ttl=6 * 3600, show_spinner="Loading nflverse play-by-play (about 30 seconds)...")
def load_data(season):
    raw = fl.load_pbp([season - 1, season], extra=pl.PCOLS)
    games = fl.load_games()
    inj_path = HERE / "injury_load_hist.csv"
    J = pd.read_csv(inj_path) if inj_path.exists() else None
    B = fl.build(raw, games, J)
    R = fl.current_ratings(B, season)
    keep = ["game_id", "posteam", "home_team", "away_team", "epa_adj", "wp", "game_seconds_remaining", "play_id"]
    sp = B["sp"][B["sp"].season == season][keep]
    TG = B["TG"][B["TG"].season == season]
    M = B["M"][B["M"].season == season]
    # players: history + rows for the next unplayed week
    players = pd.read_csv(PLAYERS_URL, low_memory=False, usecols=["gsis_id", "display_name", "position"])
    gs = games[games.season == season]
    nxt = gs[gs.result.isna()]
    nxt = nxt[nxt.week == nxt.week.min()] if len(nxt) else nxt
    s, q = pl.player_games(raw)
    S = pl.skill_dvp(pl.skill_features(pl.add_positions(pl.add_future(s, nxt), players)))
    qf = pl.add_future(q, nxt)
    listed = set(nxt.home_qb_id.dropna()) | set(nxt.away_qb_id.dropna())
    qf = qf[(qf.fut == 0) | qf.pid.isin(listed) | ~qf.posteam.isin(
        set(nxt.loc[nxt.home_qb_id.notna(), "home_team"]) | set(nxt.loc[nxt.away_qb_id.notna(), "away_team"]))]
    Q = pl.qb_dvp(pl.qb_features(qf))
    Q = Q.join(players.set_index("gsis_id").display_name.rename("full_name"), on="pid")
    Q["name"] = Q.full_name.fillna(Q.name)
    S, Q = S[S.season == season], Q[Q.season == season]
    try:
        inj = pd.read_csv(INJ_URL.format(season=season), usecols=["season", "week", "gsis_id", "report_status"])
    except Exception:  # noqa: BLE001
        inj = pd.DataFrame(columns=["season", "week", "gsis_id", "report_status"])
    return gs, R, M, TG, sp, S, Q, inj


MODEL = load_model()
PMODEL = load_player_model()
FEATS = MODEL["features"]
INJ_DEFAULT = tuple(MODEL["inj_default"])
try:
    GAMES, RATINGS, MATCH, TEAMG, PLAYS, SKILL, QBS, INJ = load_data(SEASON)
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
        <span>{fav} wins <b>{pct(fav_p)}</b></span>
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

tab_bets, tab_games, tab_record, tab_players, tab_model = st.tabs(["This week", "Games", "Record", "Players", "Model"])
BETS = lv.load_bets(HERE / "bets.csv")


@st.cache_data(ttl=25, show_spinner=False)
def live_scoreboard():
    return lv.parse_scoreboard(lv.fetch_scoreboard())


def live_model(g):
    """Model projection for a live game (pre-game ratings), or None if the game is not in the nflverse schedule."""
    rows = GAMES[(GAMES.home_team == g["home"]) & (GAMES.away_team == g["away"])].sort_values("gameday")
    if rows.empty:
        return None, None
    row = rows.iloc[-1]
    try:
        p = predict_played(row.game_id) if pd.notna(row.result) else None
        if p is None:
            p = predict_upcoming(g["home"], g["away"], row.roof in ("dome", "closed"))
    except Exception:  # noqa: BLE001
        return None, row
    return p, row


BADGE = {"winning": ("hit", "WINNING"), "won": ("hit", "WON"), "losing": ("miss", "LOSING"),
         "lost": ("miss", "LOST"), "push": ("push", "PUSH"), "pending": ("pend", "PENDING"),
         "void": ("pend", "VOID")}


def live_card(g, p, flags):
    a, h = g["away"], g["home"]
    if g["state"] == "in":
        status = f"<span class='livedot'></span><b>{g['detail']}</b>"
    elif g["state"] == "post":
        status = f"<span class='badge final'>{g['detail'] or 'FINAL'}</span>"
    else:
        status = g["kick_utc"].tz_convert(EST).strftime("%a %-I:%M %p ET") if pd.notna(g["kick_utc"]) else g["detail"]
    sub = " · ".join(x for x in [status, g["broadcast"]] if x)
    pos = lambda t: " <span class='poss'>🏈</span>" if g["state"] == "in" and g["possession"] == t else ""
    score = (f"<div class='score'><span class='tm'>{a}{pos(a)}</span><span>{g['away_score']:.0f} – "
             f"{g['home_score']:.0f}</span><span class='tm'>{pos(h)}{h}</span></div>") if g["state"] != "pre" else ""
    sit = ""
    if g["state"] == "in" and (g["down_dist"] or g["last_play"]):
        rz = " · <b style='color:#f87171'>RED ZONE</b>" if g["red_zone"] else ""
        sit = f"<div class='meta'>{g['down_dist']}{rz}{' · ' + g['last_play'] if g['last_play'] else ''}</div>"
    summ = []
    if p is not None:
        summ.append(f"<span>Model <b>{a} {p['away_score']:.1f}–{p['home_score']:.1f} {h}</b></span>")
        summ.append(f"<span>Model total <b>{p['total']:.1f}</b></span>")
    pt = lv.pace_total(g, p["total"] if p is not None else None)
    if g["state"] == "in" and pt is not None:
        summ.append(f"<span>On pace <b>{pt:.1f}</b></span>")
    if g["espn_line"] or g["espn_total"]:
        summ.append(f"<span>Line <b>{g['espn_line'] or '–'}</b> · O/U <b>{g['espn_total'] or '–'}</b></span>")
    summ = f"<div class='summary'>{''.join(summ)}</div>" if summ else ""
    qt = ""
    if p is not None:
        nq = max(4, len(g["home_q"]))
        head = "".join(f"<th>{'Q' + str(i) if i <= 4 else 'OT'}</th>" for i in range(1, nq + 1))

        def qrow(tm, side, actual):
            cells = ""
            for i in range(1, nq + 1):
                m = f"{p[f'{side}_q{i}']:.1f}" if i <= 4 else "–"
                act = f"<b>{actual[i - 1]:.0f}</b> / " if i <= len(actual) else ""
                cells += f"<td>{act}{m}</td>"
            return f"<tr><td><b>{tm}</b></td>{cells}</tr>"

        qt = (f"<div class='tbl-wrap'><table class='tbl'><tr><th>Actual / model</th>{head}</tr>"
              f"{qrow(a, 'a', g['away_q'])}{qrow(h, 'h', g['home_q'])}</table></div>")
    return f"""
    <div class='card'>
      <div class='card-head'><div class='card-title'>{a} @ {h}</div><div class='card-sub'>{sub}</div></div>
      {score}{sit}{summ}{qt}{flags_html(g, p, flags)}
    </div>"""


def flags_html(g, p, flags):
    fl_html = ""
    for _, f in flags.iterrows():
        r = lv.grade_bet(f, g, game_model=p)
        stt, m, txt = r["status"], r["margin"], r["text"]
        cls, lab = BADGE[stt]
        side = f["side"].upper() if f["market"] == "total" else f"{f['side']} {f['line']:+g}"
        line = f" {f['line']:g}" if f["market"] == "total" else ""
        mtxt = f" ({m:+.1f})" if m is not None else ""
        fl_html += (f"<div class='flag-row'><span><span class='badge {'hit' if f['tier'] == 'BET' else 'final'}'>"
                    f"{f['tier']}</span> <b>{f['bet']}: {side}{line}</b> <span class='card-sub'>{f['note']}</span>"
                    f"</span><span>{txt}{mtxt} <span class='badge {cls}'>{lab}</span></span></div>")
    return fl_html


@st.cache_data(ttl=30, show_spinner=False)
def week_board_live(season, week):
    return lv.parse_scoreboard(lv.fetch_week(season, week))


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def week_board_done(season, week):
    return lv.parse_scoreboard(lv.fetch_week(season, week))


@st.cache_data(ttl=45, show_spinner=False)
def box_live(event_id):
    return lv.parse_box(lv.fetch_summary(event_id))


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def box_done(event_id):
    return lv.parse_box(lv.fetch_summary(event_id))


def week_games(season, week):
    games = week_board_done(season, week)
    if any(g["state"] != "post" for g in games):  # week still in play: use the 30-second copy
        games = week_board_live(season, week)
    return {g["game"]: g for g in games}


GRADE_RANK = {"Best": 0, "Good": 1, "Pass": 2}
STATUS_LOOK = {"won": ("hit", "HIT"), "lost": ("miss", "MISSED"), "winning": ("hit", "ON PACE"),
               "losing": ("miss", "BEHIND"), "pending": ("pend", "NOT STARTED"), "push": ("push", "PUSH"),
               "void": ("pend", "VOID")}


def bet_grade(tier, is_prop, edge, line):
    """Best / Good / Pass from what held up in testing, not just the size of the gap.

    Best: v3 bets (injury gap ATS ~56% over 2015-26, 15+ mph wind unders). Good: v3 leans (primetime unders) and
    props the props model still likes by its own flag rule (10%+ and 8+ yds past the line). Pass: watch-only
    flags, tiebreakers, and props whose edge shrank once a real line went in. Props top out at Good because they
    have not been tested against real DK lines.
    """
    if tier == "BET":
        return "Best"
    if tier == "LEAN":
        return "Good"
    if is_prop and edge is not None and line and edge >= 8 and edge / abs(line) >= 0.10:
        return "Good"
    return "Pass"


def team_model(game_key, g=None):
    away, home = game_key.split("@")
    return live_model(g if g is not None else {"home": home, "away": away})[0]


def grade_week(season, week):
    rows = []
    bets = BETS[(BETS.season == season) & (BETS.week == week)]
    if bets.empty:
        return pd.DataFrame()
    try:
        games = week_games(season, week)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Week {week}: could not reach ESPN ({exc})")
        games = {}
    for _, b in bets.iterrows():
        g = games.get(b["game"])
        is_prop = b["market"] not in ("spread", "total")
        box, gm = None, None
        if not is_prop:
            gm = team_model(b["game"], g)
        elif g is not None and g["state"] != "pre":
            try:
                box = box_done(g["espn_id"]) if g["state"] == "post" else box_live(g["espn_id"])
            except Exception:  # noqa: BLE001
                box = None
        r = lv.grade_bet(b, g, box, gm)
        side, line = str(b["side"]), float(b["line"])
        proj, proj_txt, edge = None, "–", None
        if b["market"] == "spread" and gm is not None:
            away, home = b["game"].split("@")
            proj = gm["result"] * (1 if side == home else -1)
            proj_txt = f"{side} {'by' if proj >= 0 else 'lose by'} {abs(proj):.1f}"
            edge = proj + line
        elif b["market"] == "total" and gm is not None:
            proj = gm["total"]
            proj_txt = f"{proj:.1f}"
            edge = (proj - line) if side.lower() == "over" else (line - proj)
        elif is_prop and pd.notna(b["model"]):
            proj = float(b["model"])
            proj_txt = f"{proj:.1f}"
            edge = (proj - line) if side.lower() == "over" else (line - proj)
        grade = bet_grade(b["tier"], is_prop, edge, line)
        if is_prop:
            what = f"{b['player']} {side.capitalize()}"
            sub = f"{b['bet']} · {b['game'].replace('@', ' @ ')}"
            line_txt = f"{line:g}"
            edge_txt = "–" if edge is None else f"{edge:+.1f} ({edge / abs(line):+.0%})" if line else f"{edge:+.1f}"
            norm = 0 if edge is None or not line else edge / abs(line) / 0.25
        elif b["market"] == "total":
            what = f"{b['game'].replace('@', ' @ ')} {side.capitalize()}"
            sub = b["bet"]
            line_txt = f"{line:g}"
            edge_txt = "–" if edge is None else f"{edge:+.1f}"
            norm = 0 if edge is None else edge / 7
        else:
            what = f"{side} {line:+g}"
            sub = f"{b['bet']} · {b['game'].replace('@', ' @ ')}"
            line_txt = f"{line:+g}"
            edge_txt = "–" if edge is None else f"{edge:+.1f}"
            norm = 0 if edge is None else edge / 7
        kick = g["kick_utc"] if g else pd.NaT
        rows.append({"week": week, "tier": b["tier"], "grade": grade, "kind": "Prop" if is_prop else "Game",
                     "type": b["bet"], "bet": what, "sub": sub, "game": b["game"],
                     "state": g["state"] if g else "pre", "detail": g["detail"] if g else "",
                     "kick": kick, "kick_day": kick.tz_convert(EST).date() if pd.notna(kick) else None,
                     "line": line, "line_txt": line_txt, "proj": proj, "proj_txt": proj_txt, "edge": edge,
                     "edge_txt": edge_txt, "norm": norm, "now": r["now"], "pace": r["pace"], "margin": r["margin"],
                     "status": r["status"], "note": b["note"]})
    return pd.DataFrame(rows)


def units(d):
    w, l = int((d.status == "won").sum()), int((d.status == "lost").sum())
    return w, l, w * 100 / 110 - l


def board_html(df):
    fmt = lambda v: "–" if v is None or pd.isna(v) else f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"
    rows = ""
    for r in df.itertuples():
        cls, lab = STATUS_LOOK[r.status]
        color = {"hit": "#22c55e", "miss": "#ef4444"}.get(cls, "#475569")
        gcls = {"Best": "hit", "Good": "push", "Pass": "pend"}[r.grade]
        live = "<span class='livedot'></span>" if r.state == "in" else ""
        when = r.detail if r.state != "pre" else (r.kick.tz_convert(EST).strftime("%a %-I:%M %p")
                                                   if pd.notna(r.kick) else "")
        ecol = "#86efac" if (r.edge or 0) > 0 else "#fca5a5" if (r.edge or 0) < 0 else "#cbd5e1"
        stats = [f"Line <b>{r.line_txt}</b>", f"Proj <b>{r.proj_txt}</b>",
                 f"Edge <b style='color:{ecol}'>{r.edge_txt}</b>"]
        if r.state == "in":
            stats.append(f"Now <b>{fmt(r.now)}</b> · Pace <b>{fmt(r.pace)}</b>")
        elif r.state == "post" and r.now is not None and not pd.isna(r.now):
            stats.append(f"Final <b>{fmt(r.now)}</b>")
        rows += (f"<div class='bet-row' style='border-left:4px solid {color}'>"
                 f"<div class='bet-top'><span><span class='badge {gcls}'>{r.grade.upper()}</span> <b>{r.bet}</b></span>"
                 f"<span class='badge {cls}'>{lab}</span></div>"
                 f"<div class='bet-stats'>{' · '.join(stats)}</div>"
                 f"<div class='card-sub'>{r.sub}{' · ' + live + when if when else ''}</div></div>")
    return f"<div class='card' style='padding:4px'>{rows}</div>"


def record_table(df):
    done = df[df.status.isin(["won", "lost", "push"])].copy()
    if done.empty:
        return None
    done["bet type"] = done.grade + " · " + done.type
    g = done.groupby("bet type").status.value_counts().unstack(fill_value=0)
    for c in ("won", "lost", "push"):
        if c not in g:
            g[c] = 0
    g["record"] = g.won.astype(str) + "–" + g.lost.astype(str) + g.push.map(lambda x: f"–{x}" if x else "")
    g["win %"] = (g.won / (g.won + g.lost).replace(0, np.nan) * 100).round(0)
    g["units (-110)"] = (g.won * 100 / 110 - g.lost).round(2)
    out = g[["record", "win %", "units (-110)"]].reset_index()
    out["_r"] = out["bet type"].str.split(" · ").str[0].map(GRADE_RANK)
    return out.sort_values(["_r", "bet type"]).drop(columns="_r")


# ----------------------------------------------------------------------------
# The weekly page (same picks, wording and angles as the "NFL This Week" web page), graded live
# ----------------------------------------------------------------------------
PAGE_WEEKS = pg.page_weeks(HERE, SEASON)
TIER_CLS = {"BET": "hit", "LEAN": "push", "PROP": "push", "PROP_REST": "pend", "WATCH": "pend", "HOLD": "miss"}


def page_for(week):
    return pg.load_page(HERE, SEASON, week)


def grade_page(page, week):
    """Every pick on the page with its live status. Uses ESPN (scores, box scores) and the game flow model."""
    try:
        games = week_games(SEASON, week)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Live scores unavailable (ESPN: {exc}).")
        games = {}
    boxes, models, rows = {}, {}, []
    for p in page["picks"]:
        g = games.get(p["game"])
        b = pg.pick_to_bet(p)
        box = gm = None
        if g is not None and g["state"] != "pre":
            if b is not None and b["player"]:
                if p["game"] not in boxes:
                    try:
                        boxes[p["game"]] = box_done(g["espn_id"]) if g["state"] == "post" else box_live(g["espn_id"])
                    except Exception:  # noqa: BLE001
                        boxes[p["game"]] = None
                box = boxes[p["game"]]
            elif b is not None:
                if p["game"] not in models:
                    models[p["game"]] = team_model(p["game"], g)
                gm = models[p["game"]]
        r = pg.grade_pick(p, g, box, gm)
        rows.append({**p, **{k: r[k] for k in ("status", "now", "pace", "margin", "text")},
                     "state": g["state"] if g else "pre", "detail": g["detail"] if g else ""})
    return pd.DataFrame(rows), games


def kick_txt(s):
    t = pd.Timestamp(s)
    return t.strftime("%a %-I:%M %p")


def pick_rows_html(df, games_meta, show_game=True):
    fmt = lambda v: "–" if v is None or pd.isna(v) else f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"
    out = ""
    for r in df.itertuples():
        cls, lab = STATUS_LOOK[r.status]
        color = {"hit": "#22c55e", "miss": "#ef4444"}.get(cls, "#475569")
        live = "<span class='livedot'></span>" if r.state == "in" else ""
        when = r.detail if r.state != "pre" else kick_txt(r.kick)
        stats = ""
        if r.state == "in" and r.now is not None and not pd.isna(r.now):
            stats = f"<div class='bet-stats'>{r.text} · {'ahead' if (r.margin or 0) > 0 else 'behind'} by {fmt(abs(r.margin or 0))}</div>"
        elif r.state == "post" and r.now is not None and not pd.isna(r.now):
            stats = f"<div class='bet-stats'>Final: {r.text}</div>"
        where = f"{r.game} · " if show_game else ""
        out += (f"<div class='bet-row' style='border-left:4px solid {color}'>"
                f"<div class='bet-top'><span><b>{r.pick}</b></span><span class='badge {cls}'>{lab}</span></div>"
                f"{stats}<div class='card-sub'>{pg.plain(r._asdict(), games_meta)}</div>"
                f"<div class='card-sub'>{where}{live}{when}</div></div>")
    return f"<div class='card' style='padding:4px'>{out}</div>"


def tier_head(key):
    t = pg.TIERS[key]
    rate = f" <span class='card-sub'>· {t['rate']}</span>" if t["rate"] else ""
    return f"<div class='section'><span class='badge {TIER_CLS[key]}'>{t['label']}</span> {t['title']}{rate}</div>"


def home_spread(g):
    a, h = g["game"].split("@")
    sp = g["spread"]
    return "Pick" if sp == 0 else (f"{h} {sp:+g}" if sp < 0 else f"{a} {-sp:+g}")


def final_grade(gm, e):
    """Score and how the DK line and the page's model numbers graded, from the live ESPN result."""
    a, h = gm["game"].split("@")
    hs, as_ = e["home_score"], e["away_score"]
    mg, tot = hs - as_, hs + as_
    c = mg + gm["spread"]
    line = "Push" if c == 0 else (f"{h} covered {gm['spread']:+g}" if c > 0 else f"{a} covered {-gm['spread']:+g}")
    ou = ("Over" if tot > gm["total"] else "Under" if tot < gm["total"] else "Push") + f" {gm['total']:g}"
    txt = f"DK {home_spread(gm)}, O/U {gm['total']:g}: {line}, {ou}."
    ms = gm.get("model_spread")
    if ms is not None:
        side = h if ms < gm["spread"] else a
        ok = "push" if c == 0 else ("right" if (c > 0) == (side == h) else "wrong")
        tside = "Over" if gm["model_total"] > gm["total"] else "Under"
        tok = "push" if tot == gm["total"] else ("right" if (tot > gm["total"]) == (tside == "Over") else "wrong")
        txt += (f" Model had {h if ms < 0 else a} by {abs(ms):.1f}, total {gm['model_total']}: {side} side {ok}, "
                f"{tside.lower()} {tok} (off by {abs(gm['model_total'] - tot):.1f}).")
    return txt


def page_view(page, week):
    meta = {g["game"]: g for g in page["games"]}
    df, espn = grade_page(page, week)
    now = pd.Timestamp.now(tz=EST)
    P = df[df.tier != "PROP_REST"] if len(df) else df
    of = lambda k: P[P.tier == k] if len(P) else P
    bets, leans, props, watch, hold = (of(k) for k in ("BET", "LEAN", "PROP", "WATCH", "HOLD"))
    st_html(f"<div class='card-title' style='font-size:26px'>Week {page['week']}</div>"
            f"<div style='font-size:15px'><b>{len(bets)} bets</b>, <b>{len(leans)} leans</b>, "
            f"{len(props)} props and <b>{len(watch)} game{'s' if len(watch) != 1 else ''} to watch</b>."
            f"{' One game is on hold for a QB check.' if len(hold) else ''}</div>"
            f"<div class='meta'>Picks from the {page['run']} run. "
            f"{page.get('lines') or 'Lines are nflverse consensus, so check each against DraftKings.'}</div>")
    graded = P[P.tier.isin(["BET", "LEAN", "PROP"])] if len(P) else P
    if len(graded):
        w, l, u = units(graded)
        on, behind = int((graded.status == "winning").sum()), int((graded.status == "losing").sum())
        todo = int((graded.status == "pending").sum())
        parts = [f"Week {week} <b>{w}–{l}</b>, <b>{u:+.1f}u</b>"]
        if on or behind:
            parts.append(f"Live <b style='color:#86efac'>{on} on pace</b>, <b style='color:#fca5a5'>{behind} behind</b>")
        if todo:
            parts.append(f"{todo} to play")
        st_html(f"<div style='background:#1e293b;padding:8px 10px;border-radius:6px;font-size:14px;color:#f8fafc;"
                f"margin-bottom:6px'>{' · '.join(parts)}</div>"
                f"<div class='meta'>Bets, leans and top props · units at -110 · updated {now.strftime('%-I:%M:%S %p ET')}</div>")
    live = [(meta[k], e) for k, e in espn.items() if k in meta and e["state"] == "in"]
    if live:
        st_html("<div class='section'><span class='livedot'></span>Live now</div>" + "".join(
            f"<div class='bet-row'><b>{e['away']} {e['away_score']:.0f} – {e['home_score']:.0f} {e['home']}</b> "
            f"<span class='card-sub'>· {e['detail']} · DK {home_spread(gm)}, O/U {gm['total']:g} · "
            f"on pace {lv.pace_total(e, gm['model_total']) or 0:.1f}</span></div>" for gm, e in live))
    done = [(gm, espn[gm["game"]]) for gm in page["games"] if gm["game"] in espn and espn[gm["game"]]["state"] == "post"]
    if done:
        html = ""
        for gm, e in done:
            gp = df[df.game == gm["game"]]
            w_, l_ = int((gp.status == "won").sum()), int((gp.status == "lost").sum())
            res = ", ".join(f"{r.pick.split(' (')[0]} {'won' if r.status == 'won' else 'lost' if r.status == 'lost' else r.status}"
                            for r in gp.itertuples())
            win, lose = (e["home"], e["away"]) if e["home_score"] >= e["away_score"] else (e["away"], e["home"])
            sc = lambda t: e["home_score"] if t == e["home"] else e["away_score"]
            html += (f"<div class='bet-row'><b>{win} {sc(win):.0f}, {lose} {sc(lose):.0f}</b>"
                     f"<div class='card-sub'>{final_grade(gm, e)}</div>"
                     + (f"<div class='card-sub'>Picks in this game: {w_}-{l_} ({res}).</div>" if len(gp) else "")
                     + "</div>")
        st_html("<div class='section'>Played so far</div>" f"<div class='card' style='padding:4px'>{html}</div>")
    for key, part, limit in (("BET", bets, None), ("LEAN", leans, None), ("PROP", props, 5), ("WATCH", watch, None)):
        if not len(part):
            continue
        st_html(tier_head(key))
        st_html(pick_rows_html(part.head(limit) if limit else part, meta))
        if limit and len(part) > limit:
            with st.expander(f"Show {len(part) - limit} more props"):
                st_html(pick_rows_html(part.iloc[limit:], meta))
    if len(hold):
        st_html(tier_head("HOLD"))
        st_html("".join(f"<div class='bet-row'><b>{gk}: confirm QB</b><div class='card-sub'>Listed QB did not start "
                        f"the last game.</div></div>" for gk in hold.game.unique()))
    with st.expander("How the tiers work"):
        st_html("<div class='meta'><b>Bet:</b> injury gap of 2+ non-QB starters with the healthier team favored "
                "(83-56 ATS since 2015), 15+ mph outdoor wind unders (118-96), and 6-point teaser legs on small "
                "underdogs (77% of legs won). <b>Lean:</b> Sunday and Monday night unders (224-183 overall, 46-50 "
                "since 2024) and player prop unders, tested against season averages, not DraftKings lines. "
                "<b>Watch:</b> model total 7+ points off the market; 113-111 over 11 seasons, never bets. "
                "<b>Hold:</b> the listed QB did not start the team's last game. <b>Live status:</b> now = current "
                "number, pace = now plus the model's projection for the time left. Teaser legs are graded on the "
                "teased number.</div>")
    return df


def page_game_html(page, key, graded=None):
    """Angles, QBs, model total and picks for one game, as on the web page's Games tab."""
    meta = {g["game"]: g for g in page["games"]}
    gm = meta.get(key)
    if gm is None:
        return ""
    out = (f"<div class='meta'>QBs {gm['qbs']} · DK {home_spread(gm)}, O/U {gm['total']:g} · model total "
           f"{gm['model_total']} ({gm['model_total'] - gm['total']:+.1f} vs market)</div>")
    ang = [a for a in page["angles"] if a["game"] == key]
    if ang:
        out += "<div class='section'>Matchup angles</div>"
        for a in ang:
            to = " · ".join(x for x in [f"Points to {a['points_to']}" if a.get("points_to") and a["points_to"] != "No side" else "",
                                        f"Prop: {a['props']}" if a.get("props") else ""] if x)
            ck = a.get("check") or {}
            cf = ck.get("agree") is False or ck.get("angle_conflict")
            ctag = (" <span class='badge push'>⚠ CONFLICT: NO PLAY</span>" if cf else
                    " <span class='badge hit'>PROPS MODEL AGREES</span>" if ck.get("agree") else "")
            out += (f"<div class='flag-row' style='display:block'><b>{a['title']}</b> <span class='badge final'>"
                    f"{a['evidence']}</span>{ctag}<div class='card-sub'>{a['detail']}</div>"
                    + (f"<div class='card-sub'>{to}</div>" if to else "")
                    + (f"<div class='card-sub'>{ck['text']}</div>" if ck.get("text") else "") + "</div>")
    if graded is not None and len(graded):
        gp = graded[graded.game == key]
        if len(gp):
            out += "<div class='section'>Model picks</div>" + pick_rows_html(gp, meta, show_game=False)
    return f"<div class='card'>{out}</div>"


with tab_bets:
    bet_weeks = sorted(set(BETS[BETS.season == SEASON].week.unique().tolist()) | set(PAGE_WEEKS))
    if not bet_weeks:
        st.info("No picks for this season yet.")
    else:
        cur_week = bet_weeks[-1]
        try:
            cur = live_scoreboard()
            cur_week = max((w for w in bet_weeks if w <= (cur[0]["week"] if cur else cur_week)), default=cur_week)
        except Exception:  # noqa: BLE001
            pass
        if PAGE_WEEKS and cur_week < PAGE_WEEKS[-1] and cur_week not in PAGE_WEEKS:
            cur_week = PAGE_WEEKS[-1]
        c1, c2 = st.columns([1, 1.4])
        bweek = c1.selectbox("Week", bet_weeks, index=bet_weeks.index(cur_week), key="bets_week")
        PAGE = page_for(bweek)
        if PAGE is None:
            show_pass = c2.toggle("Show Pass picks too", value=False, key="bets_pass")
            grades = ["Best", "Good", "Pass"] if show_pass else ["Best", "Good"]
        else:
            c2.markdown("<div class='meta' style='margin-top:30px'>Same picks as the NFL This Week page, graded live</div>",
                        unsafe_allow_html=True)

        @st.fragment(run_every=60)
        def page_board():
            page_view(PAGE, bweek)

        @st.fragment(run_every=60)
        def bets_board():
            df = grade_week(SEASON, bweek)
            if df.empty:
                st.info("No bets for this week.")
                return
            df = df[df.grade.isin(grades)]
            if df.empty:
                st.info("No bets match these filters.")
                return
            now = pd.Timestamp.now(tz=EST)
            w, l, u = units(df)
            td = df[df.kick_day == now.date()]
            tw, tl, tu = units(td)
            on, behind = int((df.status == "winning").sum()), int((df.status == "losing").sum())
            todo = int((df.status == "pending").sum())
            parts = [f"Week {bweek} <b>{w}–{l}</b>, <b>{u:+.1f}u</b>"]
            if len(td):
                parts.append(f"Today <b>{tw}–{tl}</b>, <b>{tu:+.1f}u</b>")
            if on or behind:
                parts.append(f"Live <b style='color:#86efac'>{on} on pace</b>, <b style='color:#fca5a5'>{behind} behind</b>")
            if todo:
                parts.append(f"{todo} to play")
            st_html(f"<div style='background:#1e293b;padding:8px 10px;border-radius:6px;font-size:14px;color:#f8fafc;"
                    f"margin-bottom:6px'>{' · '.join(parts)}</div>"
                    f"<div class='meta'>Updated {now.strftime('%-I:%M:%S %p ET')} · ranked best bet first</div>")
            df = df.assign(_g=df.grade.map(GRADE_RANK)).sort_values(["_g", "norm"], ascending=[True, False])
            st_html(board_html(df))

        if PAGE is not None:
            page_board()
        else:
            bets_board()

with tab_record:
    rec_page = page_for(PAGE_WEEKS[-1]) if PAGE_WEEKS else None
    if rec_page is not None:
        rr = rec_page["record"]
        st_html("<div class='section'>Season to date</div>")
        st.dataframe(pd.DataFrame(rr["rows"], columns=["Type", f"{SEASON} so far", "Long-run rate"]), width="stretch",
                     hide_index=True)
        st_html(f"<div class='meta'>Tracking started Week 11 (graded through Week {rr['through_week']}). Long-run rates "
                "come from 2015-26 backtests and are what each tier is judged against. Same table as the web page.</div>")
    bet_weeks = sorted(w for w in BETS[BETS.season == SEASON].week.unique().tolist() if w not in PAGE_WEEKS)
    if bet_weeks:
        with st.expander("Older bets.csv record by bet type"):
            allw = pd.concat([grade_week(SEASON, w) for w in bet_weeks], ignore_index=True)
            rec = record_table(allw) if len(allw) else None
            if rec is None:
                st.info("No settled bets yet.")
            else:
                st.dataframe(rec, width="stretch", hide_index=True)
                wk_rec = []
                for w_, d in allw[allw.grade != "Pass"].groupby("week"):
                    ww, ll, uu = units(d)
                    if ww + ll:
                        wk_rec.append({"week": w_, "record": f"{ww}–{ll}", "units (-110)": round(uu, 2)})
                if wk_rec:
                    st_html("<div class='meta'>By week, Best and Good only</div>")
                    st.dataframe(pd.DataFrame(wk_rec), width="stretch", hide_index=True)
        with st.expander("What the grades and numbers mean"):
            st_html("<div class='meta'><b>Grade</b> reflects what held up in testing, not just the gap: "
                "<b>Best</b> = v3 bets (injury gap ATS, 15+ mph wind under). <b>Good</b> = v3 leans (Sunday/Monday "
                "night unders) and props the props model still likes by 10%+ and 8+ yds. <b>Pass</b> = watch-only "
                "flags, tiebreakers and thin props. Props top out at Good until they're tested against real DK "
                "lines. <b>Proj</b> = game flow model (games) or props model (players). <b>Edge</b> = projection "
                "minus the line in your bet's favor. <b>Now / pace</b> = current number / current plus the "
                "model's projection for the time left. Green = hit or on pace, red = missed or behind, gray = not "
                "started. Units at -110, 1 unit per bet. Prop lines are the season average until a DK line is "
                "entered in <code>bets.csv</code>.</div>")

with tab_games:
    weeks = sorted(GAMES.week.unique())
    unplayed = GAMES[GAMES.result.isna()]
    default_week = int(unplayed.week.min()) if len(unplayed) else int(max(weeks))
    c1, c2 = st.columns([1, 2])
    gweek = c1.selectbox("Week", weeks, index=weeks.index(default_week), key="games_week")
    gshow = c2.radio("Show", ["All", "Live now", "Has picks"], horizontal=True, key="games_show")
    st_html("<div class='meta'>Model preview until kickoff, then the live score with actual vs model by quarter. "
            "Open a game for its matchup angles and picks, graded live.</div>")

    @st.fragment(run_every=60)
    def games_board():
        wk = GAMES[GAMES.week == gweek].sort_values(["gameday", "gametime"])
        try:
            espn = week_games(SEASON, int(gweek))
        except Exception as exc:  # noqa: BLE001
            espn = {}
            st.warning(f"Live scores unavailable (ESPN: {exc}). Showing model previews.")
        picks = BETS[(BETS.season == SEASON) & (BETS.week == gweek)]
        no_flags = picks.iloc[0:0]
        gpage = page_for(int(gweek))
        graded = None
        if gpage is not None:
            graded, _ = grade_page(gpage, int(gweek))
            picks = graded[graded.tier != "PROP_REST"] if len(graded) else graded
        extra = lambda key: page_game_html(gpage, key, graded) if gpage is not None else ""
        now = pd.Timestamp.now(tz=EST)
        n_live = sum(1 for g in espn.values() if g["state"] == "in")
        st_html(f"<div class='meta'>Updated {now.strftime('%-I:%M:%S %p ET')} · {n_live} live</div>")
        items = []
        for _, g in wk.iterrows():
            key = f"{g.away_team}@{g.home_team}"
            e = espn.get(key)
            state = e["state"] if e else ("post" if pd.notna(g.result) else "pre")
            if gshow == "Live now" and state != "in":
                continue
            if gshow == "Has picks" and key not in set(picks.game):
                continue
            items.append((g, e, state, key))
        if not items:
            st.info("No games match.")
            return
        order = {"in": 0, "pre": 1, "post": 2}
        items.sort(key=lambda t: (order[t[2]], str(t[0].gameday), str(t[0].gametime)))
        for g, e, state, key in items:
            n_picks = int((picks.game == key).sum())
            more = (f"<div class='meta'>{n_picks} pick{'s' if n_picks != 1 else ''} for this game on the This week tab</div>"
                    if n_picks and gpage is None else "")
            try:
                if state == "in" and e is not None:
                    p, _ = live_model(e)
                    st_html(live_card(e, p, no_flags) + more)
                    if gpage is not None:
                        with st.expander(f"{key.replace('@', ' at ')}: matchup angles and picks"):
                            st_html(extra(key))
                    continue
                played = pd.notna(g.result)
                p = predict_played(g.game_id) if played else None
                if p is None:
                    p = predict_upcoming(g.home_team, g.away_team, g.roof in ("dome", "closed"))
                fav = g.home_team if p["home_win"] >= .5 else g.away_team
                if state == "post":
                    if e is not None:
                        a_s, h_s = e["away_score"], e["home_score"]
                    else:
                        a_s, h_s = g.away_score, g.home_score
                    label = f"**{g.away_team} {a_s:.0f} @ {g.home_team} {h_s:.0f}** · FINAL"
                else:
                    kick = f"{g.weekday} {g.gametime} ET" if isinstance(g.gametime, str) else ""
                    label = f"**{g.away_team} @ {g.home_team}** · {kick} · {fav} {pct(max(p['home_win'], 1 - p['home_win']))}"
                if n_picks:
                    label += f" · {n_picks} pick{'s' if n_picks != 1 else ''}"
                with st.expander(label, expanded=False):
                    if state == "post" and e is not None:
                        st_html(live_card(e, p, no_flags) + more + extra(key))
                        continue
                    actual = None
                    if played:
                        m = MATCH[MATCH.game_id == g.game_id]
                        actual = {"result": g.result, "home_score": g.home_score, "away_score": g.away_score,
                                  "total": g.total, "half_margin": float(m.half_margin.iloc[0]) if len(m) else 0}
                    if state == "pre" and st.toggle("Adjust injuries", key=f"inj_{g.game_id}"):
                        st_html("<div class='meta'>Starters out, weighted by snap share and position (league "
                                f"average about {INJ_DEFAULT[0]:.1f} offense, {INJ_DEFAULT[1]:.1f} defense).</div>")
                        c = st.columns(4)
                        ao = c[0].number_input(f"{g.away_team} off", 0.0, 10.0, round(INJ_DEFAULT[0], 1), 0.5, key=f"ao_{g.game_id}")
                        ad = c[1].number_input(f"{g.away_team} def", 0.0, 10.0, round(INJ_DEFAULT[1], 1), 0.5, key=f"ad_{g.game_id}")
                        ho = c[2].number_input(f"{g.home_team} off", 0.0, 10.0, round(INJ_DEFAULT[0], 1), 0.5, key=f"ho_{g.game_id}")
                        hd = c[3].number_input(f"{g.home_team} def", 0.0, 10.0, round(INJ_DEFAULT[1], 1), 0.5, key=f"hd_{g.game_id}")
                        p = predict_upcoming(g.home_team, g.away_team, g.roof in ("dome", "closed"),
                                             {g.away_team: (ao, ad), g.home_team: (ho, hd)})
                    st_html(game_card(g, p, actual))
                    if more:
                        st_html(more)
                    if gpage is not None:
                        st_html(extra(key))
            except Exception as exc:  # noqa: BLE001
                st.error(f"{g.away_team} @ {g.home_team}: could not load ({exc})")

    games_board()

with tab_players:
    from math import erf, sqrt

    def norm_cdf(z):
        return 0.5 * (1 + erf(z / sqrt(2)))

    pweeks = sorted(GAMES.week.unique())
    p_unplayed = GAMES[GAMES.result.isna()]
    p_default = int(p_unplayed.week.min()) if len(p_unplayed) else int(max(pweeks))
    pc1, pc2 = st.columns([1, 3])
    pweek = pc1.selectbox("Week", pweeks, index=pweeks.index(p_default), key="pl_week")
    pwk = GAMES[GAMES.week == pweek].sort_values(["gameday", "gametime"])
    glabels = [f"{r.away_team} @ {r.home_team}" for r in pwk.itertuples()]
    gsel = pc2.selectbox("Game", glabels, key="pl_game")
    g = pwk.iloc[glabels.index(gsel)]
    played = pd.notna(g.result)
    gp = (predict_played(g.game_id) if played else None) or predict_upcoming(
        g.home_team, g.away_team, g.roof in ("dome", "closed"))
    ctx = {g.home_team: {"c_pts": gp["home_score"], "c_margin": gp["home_score"] - gp["away_score"],
                         "c_plays": gp["total_plays"], "c_home": 1.0},
           g.away_team: {"c_pts": gp["away_score"], "c_margin": gp["away_score"] - gp["home_score"],
                         "c_plays": gp["total_plays"], "c_home": 0.0}}
    upcoming_week = not played and (SKILL[(SKILL.game_id == g.game_id) & (SKILL.fut == 1)].shape[0] > 0)
    if not played and not upcoming_week:
        st.info("Player projections are available for the next unplayed week and for finished games.")
    else:
        inj_wk = INJ[(INJ.season == SEASON) & (INJ.week == pweek)]
        out_ids = set(inj_wk[inj_wk.report_status.isin(["Out", "Doubtful"])].gsis_id)
        q_ids = set(inj_wk[inj_wk.report_status == "Questionable"].gsis_id)

        def with_ctx(df):
            c = pd.DataFrame([ctx[t] for t in df.posteam], index=df.index)
            return pd.concat([df.drop(columns=[x for x in pl.CTX if x in df.columns]), c], axis=1)

        sk = SKILL[SKILL.game_id == g.game_id]
        sk = sk[(sk.n >= 2) & ((sk.p_tgt >= 2) | (sk.p_car >= 3)) & sk.pos.isin(["RB", "WR", "TE", "QB"])]
        if not played:
            sk = sk[~sk.pid.isin(out_ids)]
        sk = with_ctx(sk.copy())
        for mk in ["rec_yds", "rec", "rush_yds", "car"]:
            sk["proj_" + mk] = pl.project(PMODEL, sk, mk) if len(sk) else []
        sk["proj_td"] = pl.project(PMODEL, sk, "td") if len(sk) else []
        qb = QBS[(QBS.game_id == g.game_id) & (QBS.n >= 1) & (QBS.p_db >= 15)]
        if played:
            qb = qb.sort_values("db", ascending=False).groupby("posteam").head(1)
        qb = with_ctx(qb.copy())
        qb["proj_pass_yds"] = pl.project(PMODEL, qb, "pass_yds") if len(qb) else []

        # DraftKings prop lines from the weekly page file (pulled via The Odds API; no extra credits here)
        DK_MK = {"pass_yds": "passing_yards", "rec_yds": "receiving_yards", "rush_yds": "rushing_yards",
                 "rec": "receptions", "car": "rush_attempts"}
        pw_page = page_for(int(pweek))
        DKL = {(lv._norm(d["player"]), d["market"]): d for d in (pw_page or {}).get("dk_props", [])}

        def dk_for(name, mk):
            return DKL.get((lv._norm(name), DK_MK.get(mk, mk)))

        def lean(proj, line, mk, ok=True):
            """Over/under tag: projection vs the DraftKings line, only when DK has one."""
            if not ok or line is None or line <= 0:
                return ""
            d = proj - line
            if abs(d) >= pl.FLAG_MIN[mk] and abs(d) / line >= .10:
                return "<span class='badge hit'>▲ over</span>" if d > 0 else "<span class='badge miss'>▼ under</span>"
            return ""

        def dvp_txt(v):
            if pd.isna(v):
                return "–"
            col = "#86efac" if v >= .10 else ("#fca5a5" if v <= -.10 else "#cbd5e1")
            return f"<b style='color:{col}'>{v:+.0%}</b>"

        def cell(proj, avg, mk, actual=None, show=True, ok=True, name=None):
            dk = dk_for(name, mk) if name else None
            if not show and dk is None:
                return "<td>–</td>"
            fmt = "{:.1f}" if mk in ("rec", "car") else "{:.0f}"
            act = f"<br><span class='card-sub'>act {fmt.format(actual)}</span>" if actual is not None and pd.notna(actual) else ""
            dk_txt = (f"<br><span style='color:#38bdf8;font-weight:700'>DK {dk['line']:g}</span> "
                      f"<span class='card-sub'>o{dk['over']:+d} u{dk['under']:+d}</span>") if dk else ""
            return (f"<td><b>{fmt.format(proj)}</b> <span class='card-sub'>avg {fmt.format(avg)}</span>{dk_txt} "
                    f"{lean(proj, dk['line'] if dk else None, mk, ok)}{act}</td>")

        # One row per DK prop: our projection, the DK line and odds, and the edge (projection minus line).
        # A row is a play (green over / red under) when the edge clears the same bar that held up in testing:
        # 8+ yds and 10%+ of the line (15+ yds on passing). Everything else is greyed out.
        def dk_rows(team):
            out = []
            for r in qb[qb.posteam == team].itertuples():
                out.append((r, "pass_yds", "Pass yds", r.proj_pass_yds, r.pass_yds if played else None))
            for r in sk[sk.posteam == team].itertuples():
                for mk, lab in (("rec_yds", "Rec yds"), ("rush_yds", "Rush yds"), ("rec", "Receptions"), ("car", "Carries")):
                    out.append((r, mk, lab, getattr(r, "proj_" + mk), getattr(r, mk) if played else None))
            rows = []
            for r, mk, lab, proj, act in out:
                dk = dk_for(r.name, mk)
                if dk is None or pd.isna(proj):
                    continue
                edge = proj - dk["line"]
                play = abs(edge) >= pl.FLAG_MIN[mk] and dk["line"] > 0 and abs(edge) / dk["line"] >= .10
                # matchup angles on this same prop: one pointing the other way is a conflict, so no play
                sides = pg.angle_sides(pw_page, f"{g.away_team}@{g.home_team}", r.name, DK_MK[mk])
                mside = "over" if edge > 0 else "under"
                conflict = [t for sd, t in sides if sd != mside] if play else []
                agree = [t for sd, t in sides if sd == mside] if play else []
                rows.append(dict(name=r.name, pos=getattr(r, "pos", "QB"), team=team, mk=mk, lab=lab, proj=proj,
                                 line=dk["line"], over=dk["over"], under=dk["under"], edge=edge,
                                 play=play and not conflict, conflict=conflict, agree=agree, act=act))
            return sorted(rows, key=lambda x: -abs(x["edge"]))

        def odds_for(x):
            return x["over"] if x["edge"] > 0 else x["under"]

        def edge_row(x, show_team=False):
            if x["play"]:
                over = x["edge"] > 0
                bg, col = ("#14532d", "#86efac") if over else ("#7f1d1d", "#fca5a5")
                tag = f"<span class='badge {'hit' if over else 'miss'}'>{'▲ OVER' if over else '▼ UNDER'}</span>"
                style = f"background:{bg}33;color:#f8fafc"
            elif x["conflict"]:
                col, style = "#fde68a", "color:#cbd5e1"
                tag = "<span class='badge push'>⚠ CONFLICT</span>"
            else:
                col, tag, style = "#94a3b8", "", "color:#64748b"
            act = (f"<br><span class='card-sub'>act {x['act']:.0f}</span>" if x["act"] is not None and pd.notna(x["act"]) else "")
            who = f"{x['name']}{' (' + x['team'] + ')' if show_team else ''}"
            note = ""
            if x["conflict"]:
                note = f"<br><span class='card-sub' style='color:#fde68a'>Angle says {'over' if x['edge'] < 0 else 'under'}: {x['conflict'][0]}. No play.</span>"
            elif x["agree"]:
                note = "<br><span class='card-sub'>matchup angle agrees</span>"
            return (f"<tr style='{style}'><td style='text-align:left'><b>{who}</b><br><span class='card-sub'>{x['lab']}</span>{note}</td>"
                    f"<td>{x['line']:g}<br><span class='card-sub'>o{x['over']:+d} u{x['under']:+d}</span></td>"
                    f"<td><b>{x['proj']:.0f}</b>{act}</td><td><b style='color:{col}'>{x['edge']:+.1f}</b><br>{tag}</td></tr>")

        all_rows = {t: dk_rows(t) for t in (g.away_team, g.home_team)}
        best = sorted([x for t in all_rows.values() for x in t if x["play"]], key=lambda x: -abs(x["edge"]))[:3]
        pulled = (f"DK lines pulled {pd.Timestamp(pw_page['dk_props'][0]['pulled']).tz_convert(EST).strftime('%a %-I:%M %p ET')}"
                  if pw_page and pw_page.get("dk_props") else "No DK prop lines pulled for this week")
        st_html(f"<div class='meta'>{pulled}. Edge = our projection minus the DK line. A prop is a play (green over, "
                "red under) when the edge is 8+ yds and 10%+ of the line (15+ yds on passing yards), the gap that held "
                "up in testing; the rest are greyed out. ⚠ Conflict = a matchup angle points the other way on that prop, "
                "so it is not a play. Props DK isn't offering are left out."
                + (" Out and Doubtful players are removed." if not played else "") + "</div>")
        if best:
            st_html("<div class='card'><div class='card-head'><div class='card-title'>⭐ Best plays</div>"
                    "<div class='card-sub'>biggest edges vs DK in this game</div></div>"
                    + "".join(f"<div class='bet-row'><div class='bet-top'><span><b>{x['name']}</b> ({x['team']}) "
                              f"{x['lab'].lower()} {'over' if x['edge'] > 0 else 'under'} {x['line']:g} "
                              f"<span class='card-sub'>({odds_for(x):+d})</span></span><span class='badge "
                              f"{'hit' if x['edge'] > 0 else 'miss'}'>{x['edge']:+.1f}</span></div>"
                              f"<div class='card-sub'>Projection {x['proj']:.0f} vs DK {x['line']:g}</div></div>"
                              for x in best) + "</div>")
        elif pw_page and pw_page.get("dk_props"):
            st_html("<div class='meta'>No prop in this game clears the edge bar.</div>")
        for team, opp in ((g.away_team, g.home_team), (g.home_team, g.away_team)):
            rows = all_rows[team]
            body = "".join(edge_row(x) for x in rows) or "<tr><td colspan=4>No DK prop lines for this team</td></tr>"
            st_html(f"""
            <div class='card'>
              <div class='card-head'><div class='card-title'>{team} vs {opp} defense</div>
                <div class='card-sub'>projected {ctx[team]['c_pts']:.1f} pts</div></div>
              <div class='tbl-wrap'><table class='tbl'>
                <tr><th style='text-align:left'>Player</th><th>DK line</th><th>Proj</th><th>Edge</th></tr>{body}</table></div>
            </div>""")
            with st.expander(f"{team}: every projection, Any TD and Def vs pos"):
                t_sk = sk[sk.posteam == team].copy()
                t_qb = qb[qb.posteam == team]
                qrows = "".join(
                    f"<tr><td><b>{r.name}</b> QB{' (Q)' if r.pid in q_ids and not played else ''}</td>"
                    + cell(r.proj_pass_yds, r.p_pass_yds, "pass_yds", r.pass_yds if played else None, True, False)
                    + f"<td>{dvp_txt(r.dvp_pass)}</td></tr>" for r in t_qb.itertuples())
                t_sk["order"] = t_sk.proj_rec_yds + t_sk.proj_rush_yds
                srows = "".join(
                    f"<tr><td><b>{r.name}</b> {r.pos}{' (Q)' if r.pid in q_ids and not played else ''}</td>"
                    + cell(r.proj_rec, r.p_rec, "rec", r.rec if played else None, r.p_tgt >= 1, False)
                    + cell(r.proj_rec_yds, r.p_rec_yds, "rec_yds", r.rec_yds if played else None, r.p_tgt >= 1, False)
                    + cell(r.proj_car, r.p_car, "car", r.car if played else None, r.p_car >= 2, False)
                    + cell(r.proj_rush_yds, r.p_rush_yds, "rush_yds", r.rush_yds if played else None, r.p_car >= 2, False)
                    + f"<td><b>{r.proj_td:.0%}</b>{'<br><span class=card-sub>TD</span>' if played and r.td > 0 else ''}</td>"
                    + f"<td>{dvp_txt(r.dvp_rush if r.pos == 'QB' or (r.pos == 'RB' and r.p_car >= r.p_tgt) else r.dvp_rec)}</td></tr>"
                    for r in t_sk.sort_values("order", ascending=False).itertuples())
                st_html(f"""
                <div class='card'>
                  <div class='meta'>Bold = projection, gray = season average. <b>Def vs pos</b> = what this defense allows to
                  the position vs league average (green = soft matchup).</div>
                  <div class='tbl-wrap'><table class='tbl'>
                    <tr><th>QB</th><th>Pass yds</th><th>Def vs pos</th></tr>{qrows or "<tr><td colspan=3>No QB with a start this season yet</td></tr>"}</table></div>
                  <div class='tbl-wrap' style='margin-top:8px'><table class='tbl'>
                    <tr><th>Player</th><th>Rec</th><th>Rec yds</th><th>Car</th><th>Rush yds</th><th>Any TD</th><th>Def vs pos</th></tr>
                    {srows}</table></div>
                </div>""")

        # DK line check
        st_html("<div class='section'>🔎 Line check</div>")
        opts = [(f"{r.name} ({r.posteam}) · Pass yds", "pass_yds", r) for r in qb.itertuples()]
        for r in sk.sort_values("proj_rec_yds", ascending=False).itertuples():
            if r.p_tgt >= 1:
                opts += [(f"{r.name} ({r.posteam}) · Rec yds", "rec_yds", r), (f"{r.name} ({r.posteam}) · Receptions", "rec", r)]
            if r.p_car >= 2:
                opts += [(f"{r.name} ({r.posteam}) · Rush yds", "rush_yds", r), (f"{r.name} ({r.posteam}) · Carries", "car", r)]
        if opts:
            lc1, lc2 = st.columns([3, 1])
            pick = lc1.selectbox("Player and market", [o[0] for o in opts], key="pl_lc")
            _, mk, r = opts[[o[0] for o in opts].index(pick)]
            proj = getattr(r, "proj_" + mk)
            dk = dk_for(r.name, mk)
            line = lc2.number_input("DK line", 0.0, 600.0, float(dk["line"]) if dk else
                                    float(round(proj * 2) / 2 - 0.5 if proj > 1 else 0.5), 0.5, key=f"pl_line_{pick}")
            m = PMODEL["markets"][mk]
            med = proj + m.get("median_shift", 0)
            p_over = 1 - norm_cdf((line - med) / m["sd"])
            side = "OVER" if p_over > .5 else "UNDER"
            st_html(f"<div class='card'><div class='summary'><span>Projection <b>{proj:.1f}</b></span>"
                    f"<span>Typical outcome (median) <b>{med:.1f}</b></span><span>Over {line:g} <b>{p_over:.0%}</b></span>"
                    f"<span>Under <b>{1 - p_over:.0%}</b></span><span>Model side <b>{side}</b></span></div>"
                    f"<div class='meta'>Rough odds from the model's typical miss ({m['sd']:.0f} on yards in testing). "
                    + (f"DK line {dk['line']:g} (over {dk['over']:+d}, under {dk['under']:+d}) filled in from the latest pull. " if dk else "No DK line pulled for this one; type it in. ")
                    + "Not yet proven against DraftKings lines over a full season, so treat it as a lean, not an edge.</div></div>")

with tab_model:
    model_sec = st.radio("Section", ["Teams", "Any matchup", "Flow Chart", "Accuracy"], horizontal=True, key="model_sec",
                         label_visibility="collapsed")

    if model_sec == "Teams":
        show = {"o_epa_adj": "Adj EPA off", "d_epa_adj": "Adj EPA def", "o_pts_drive": "Pts/drive",
                "d_pts_drive": "Pts/drive ag", "o_third_conv": "3rd% off", "d_third_conv": "3rd% def",
                "o_rz_td_pct": "RZ TD% off", "d_rz_td_pct": "RZ TD% def", "o_pace": "Pace",
                "o_plays": "Off plays", "d_plays": "Def plays"}
        prof = TEAMG.groupby("team")[list(show)].mean().rename(columns=show)
        prof.insert(0, "GP", TEAMG.groupby("team").size())
        prof["Total plays"] = prof["Off plays"] + prof["Def plays"]
        prof = prof.drop(columns=["Off plays", "Def plays"])
        pct_cols = ["3rd% off", "3rd% def", "RZ TD% off", "RZ TD% def"]
        fmt = {c: "{:.0%}" for c in pct_cols}
        fmt.update({c: "{:.2f}" for c in ["Adj EPA off", "Adj EPA def", "Pts/drive", "Pts/drive ag"]})
        fmt.update({c: "{:.1f}" for c in prof.columns if c not in fmt and c != "GP"})
        st_html("<div class='meta'>Season averages per game. Pace = seconds per snap in neutral situations. "
                "Adj EPA = EPA per play after removing score-state effects (defense: lower is better).</div>")
        st.dataframe(prof.sort_values("Adj EPA off", ascending=False).style.format(fmt), width="stretch",
                     height=600)

    if model_sec == "Any matchup":
        teams = sorted(RATINGS.index)
        c1, c2 = st.columns(2)
        away = c1.selectbox("Away", teams, index=teams.index("MIA") if "MIA" in teams else 0)
        home = c2.selectbox("Home", teams, index=teams.index("BUF") if "BUF" in teams else 1)
        indoor = st.toggle("Indoor (dome or closed roof)", value=False)
        c3, c4, c5, c6 = st.columns(4)
        ai_o = c3.number_input(f"{away} off inj", 0.0, 10.0, round(INJ_DEFAULT[0], 1), 0.5)
        ai_d = c4.number_input(f"{away} def inj", 0.0, 10.0, round(INJ_DEFAULT[1], 1), 0.5)
        hi_o = c5.number_input(f"{home} off inj", 0.0, 10.0, round(INJ_DEFAULT[0], 1), 0.5)
        hi_d = c6.number_input(f"{home} def inj", 0.0, 10.0, round(INJ_DEFAULT[1], 1), 0.5)
        if home == away:
            st.warning("Pick two different teams.")
        else:
            p = predict_upcoming(home, away, indoor, {away: (ai_o, ai_d), home: (hi_o, hi_d)})
            st_html(game_card({"away_team": away, "home_team": home}, p))

    if model_sec == "Flow Chart":
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

    if model_sec == "Accuracy":
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
