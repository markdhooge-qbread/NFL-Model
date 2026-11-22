"""NFL Game Flow Model (Streamlit).

Predicts how each game unfolds: points by quarter, halftime score, middle-eight points, total plays and drives,
and the chance the home team scores first, leads at half, scores last before half, scores first after half, and wins.

Weights come from model.json (built offline by train.py). On start the app downloads this season's and last
season's nflverse play-by-play to compute current team ratings.

Tabs: Live, Slate, Matchup, Players, Teams, Flow Chart, Record.
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
import playerlib as pl

# Streamlit Cloud keeps imported helper modules in memory across code updates, so a new app.py can run against an
# old flowlib.py/playerlib.py until the app is rebooted. Reload them so app.py and its helpers always match.
fl = importlib.reload(fl)
pl = importlib.reload(pl)
lv = importlib.reload(lv)

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

tab_bets, tab_live, tab_slate, tab_match, tab_players, tab_teams, tab_flow, tab_rec = st.tabs(
    ["Bets", "Live", "Slate", "Matchup", "Players", "Teams", "Flow Chart", "Record"])
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
    return f"""
    <div class='card'>
      <div class='card-head'><div class='card-title'>{a} @ {h}</div><div class='card-sub'>{sub}</div></div>
      {score}{sit}{summ}{qt}{fl_html}
    </div>"""


with tab_live:
    c1, c2, c3 = st.columns([1.3, 1, 1])
    show = c1.radio("Show", ["Live now", "Today", "Whole week"], index=1, horizontal=True, key="live_show")
    only_flags = c2.toggle("Flagged games only", key="live_flags_only")
    auto = c3.toggle("Auto-refresh (60s)", value=True, key="live_auto")
    FLAGS = BETS[BETS.market.isin(["spread", "total"])]

    @st.fragment(run_every=60 if auto else None)
    def live_board():
        try:
            games = live_scoreboard()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not reach the ESPN scoreboard: {exc}")
            return
        if not games:
            st.info("ESPN has no games on the scoreboard right now.")
            return
        now = pd.Timestamp.now(tz=EST)
        if show == "Live now":
            games = [g for g in games if g["state"] == "in"]
        elif show == "Today":
            games = [g for g in games if pd.notna(g["kick_utc"]) and g["kick_utc"].tz_convert(EST).date() == now.date()]
        wk = FLAGS[FLAGS.season == SEASON]
        names = {g["game"] for g in live_scoreboard()}
        wk = wk[wk.game.isin(names)]  # this ESPN week's flags only
        if only_flags:
            games = [g for g in games if g["game"] in set(wk.game)]
        # flag tally across the whole week (bets + leans; watch shown separately)
        allg = {g["game"]: g for g in live_scoreboard()}
        tally = {}
        for _, f in wk.iterrows():
            g = allg.get(f["game"])
            stt = lv.grade_bet(f, g, game_model=live_model(g)[0] if g else None)["status"]
            grp = "watch" if f["tier"] == "WATCH" else "play"
            tally.setdefault(grp, {}).setdefault(stt, 0)
            tally[grp][stt] += 1

        def tally_txt(d):
            order = ["won", "lost", "push", "winning", "losing", "pending"]
            return " · ".join(f"{d[k]} {k}" for k in order if d.get(k)) or "none"

        st_html(f"<div class='meta'>Updated {now.strftime('%-I:%M:%S %p ET')} from ESPN · "
                f"<b>Bets & leans:</b> {tally_txt(tally.get('play', {}))} · "
                f"<b>Watch:</b> {tally_txt(tally.get('watch', {}))}</div>")
        if not games:
            st.info({"Live now": "No games in progress right now.",
                     "Today": "No games today. Switch to Whole week."}.get(show, "No games match."))
            return
        order = {"in": 0, "pre": 1, "post": 2}
        games = sorted(games, key=lambda g: (order.get(g["state"], 3), g["kick_utc"]))
        for g in games:
            p, _ = live_model(g)
            f = wk[wk.game == g["game"]].sort_values("tier_rank")
            st_html(live_card(g, p, f))

    live_board()
    st_html("<div class='meta'>Flags come from <code>bets.csv</code> (v3: BET = injury gap ATS and 15+ mph wind "
            "under; LEAN = Sunday/Monday night under; WATCH = model total 7+ pts off and tiebreakers, not bets). "
            "Lines are the ones logged when each flag fired. Model numbers are the pre-game game flow projection; "
            "'On pace' adds the model's scoring rate for the time left to the current total, and live totals flags "
            "are graded on that pace (an under only counts as lost once the line is passed). Quarter cells read "
            "actual / model.</div>")

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


def grade_week(season, week):
    rows = []
    bets = BETS[(BETS.season == season) & (BETS.week == week)]
    if bets.empty:
        return pd.DataFrame()
    try:
        games = week_games(season, week)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Week {week}: could not reach ESPN ({exc})")
        return pd.DataFrame()
    for _, b in bets.iterrows():
        g = games.get(b["game"])
        box, gm = None, None
        if g is not None and g["state"] != "pre":
            if b["market"] in ("spread", "total"):
                gm = live_model(g)[0]
            else:
                try:
                    box = box_done(g["espn_id"]) if g["state"] == "post" else box_live(g["espn_id"])
                except Exception:  # noqa: BLE001
                    box = None
        r = lv.grade_bet(b, g, box, gm)
        is_prop = b["market"] not in ("spread", "total")
        if is_prop:
            what = f"{b['player']} {str(b['side']).upper()} {b['line']:g}"
        elif b["market"] == "total":
            what = f"{b['game'].replace('@', ' @ ')} {str(b['side']).upper()} {b['line']:g}"
        else:
            what = f"{b['side']} {b['line']:+g}"
        rows.append({"week": week, "tier": b["tier"], "type": f"{b['tier']} · {b['bet']}", "bet": what,
                     "game": b["game"], "state": g["state"] if g else "pre", "detail": g["detail"] if g else "",
                     "kick": g["kick_utc"] if g else pd.NaT, "line": b["line"], "now": r["now"], "pace": r["pace"],
                     "margin": r["margin"], "status": r["status"], "text": r["text"], "note": b["note"]})
    return pd.DataFrame(rows)


def bets_table(df):
    body = ""
    for r in df.itertuples():
        cls, lab = BADGE[r.status]
        tier_cls = "hit" if r.tier == "BET" else "final"
        fmt = lambda v: "–" if v is None or pd.isna(v) else f"{v:.1f}" if abs(v - round(v)) > 0.05 else f"{v:.0f}"
        live = "<span class='livedot'></span>" if r.state == "in" else ""
        body += (f"<tr><td><span class='badge {tier_cls}'>{r.tier}</span> <b>{r.bet}</b>"
                 f"<div class='card-sub'>{r.type.split(' · ', 1)[1]} · {r.game.replace('@', ' @ ')} · {live}{r.detail}</div></td>"
                 f"<td>{fmt(r.now)}</td><td>{fmt(r.pace)}</td>"
                 f"<td>{'–' if r.margin is None or pd.isna(r.margin) else f'{r.margin:+.1f}'}</td>"
                 f"<td><span class='badge {cls}'>{lab}</span></td></tr>")
    return (f"<div class='card'><div class='tbl-wrap'><table class='tbl'><tr><th>Bet</th><th>Now</th><th>Pace</th>"
            f"<th>vs line</th><th>Status</th></tr>{body}</table></div></div>")


def record_table(df):
    done = df[df.status.isin(["won", "lost", "push"])]
    if done.empty:
        return None
    g = done.groupby("type").status.value_counts().unstack(fill_value=0)
    for c in ("won", "lost", "push"):
        if c not in g:
            g[c] = 0
    g["record"] = g.won.astype(str) + "–" + g.lost.astype(str) + g.push.map(lambda x: f"–{x}" if x else "")
    g["win %"] = (g.won / (g.won + g.lost).replace(0, np.nan) * 100).round(0)
    g["units (-110)"] = (g.won * 100 / 110 - g.lost).round(2)
    return g[["record", "win %", "units (-110)"]].reset_index().sort_values("type")


with tab_bets:
    bet_weeks = sorted(BETS[BETS.season == SEASON].week.unique().tolist())
    if not bet_weeks:
        st.info("No bets in bets.csv for this season yet.")
    else:
        cur_week = bet_weeks[-1]
        try:
            cur = live_scoreboard()
            cur_week = max((w for w in bet_weeks if w <= (cur[0]["week"] if cur else cur_week)), default=cur_week)
        except Exception:  # noqa: BLE001
            pass
        c1, c2, c3 = st.columns([1, 1.6, 1])
        bweek = c1.selectbox("Week", bet_weeks, index=bet_weeks.index(cur_week), key="bets_week")
        tiers = c2.multiselect("Show", ["BET", "LEAN", "WATCH", "PROP"], default=["BET", "LEAN", "PROP"],
                               key="bets_tiers")
        bauto = c3.toggle("Auto-refresh (60s)", value=True, key="bets_auto")

        @st.fragment(run_every=60 if bauto else None)
        def bets_board():
            df = grade_week(SEASON, bweek)
            if df.empty:
                st.info("No bets for this week.")
                return
            df = df[df.tier.isin(tiers)]
            now = pd.Timestamp.now(tz=EST)
            n = df.status.value_counts()
            st_html(f"<div class='meta'>Updated {now.strftime('%-I:%M:%S %p ET')} · "
                    + " · ".join(f"<b>{n.get(k, 0)}</b> {k}" for k in
                                 ["winning", "losing", "won", "lost", "push", "pending", "void"] if n.get(k, 0))
                    + "</div>")
            live = df[df.state == "in"].assign(k=lambda d: d.margin.abs()).sort_values("k")
            pre = df[df.state == "pre"].sort_values("kick")
            done = df[df.state == "post"].sort_values(["status", "margin"])
            for title, part in [("In progress · closest to the line first", live), ("Upcoming", pre),
                                ("Settled", done)]:
                if len(part):
                    st_html(f"<div class='section'>{title} ({len(part)})</div>")
                    st_html(bets_table(part))

        bets_board()

        st_html("<div class='section'>Record by bet type</div>")
        allw = pd.concat([grade_week(SEASON, w) for w in bet_weeks], ignore_index=True)
        rec = record_table(allw) if len(allw) else None
        if rec is None:
            st.info("No settled bets yet.")
        else:
            st.dataframe(rec, width="stretch", hide_index=True)
            wk_rec = []
            for w, d in allw.groupby("week"):
                d = d[d.status.isin(["won", "lost", "push"]) & (d.tier != "WATCH")]
                if len(d):
                    wk_rec.append({"week": w, "won": int((d.status == "won").sum()),
                                   "lost": int((d.status == "lost").sum()), "push": int((d.status == "push").sum())})
            if wk_rec:
                st_html("<div class='meta'>By week (bets, leans and props; watch excluded)</div>")
                st.dataframe(pd.DataFrame(wk_rec), width="stretch", hide_index=True)
        st_html("<div class='meta'>Bets come from <code>bets.csv</code> in the repo: v3 team flags (BET = injury gap "
                "ATS and 15+ mph wind under, LEAN = Sunday/Monday night under, WATCH = not a bet) and player props "
                "flagged by the props model. Prop lines are the player's season average until you enter the DK line "
                "in the file. <b>Pace</b> = current number plus the model's projection for the time left; live bets "
                "are graded on pace, and an over counts as won (an under as lost) once the line is passed. "
                "Records are graded at -110.</div>")

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

        def lean(proj, avg, mk, ok=True):
            if not ok or pd.isna(avg) or avg <= 0:
                return ""
            d = proj - avg
            if abs(d) >= pl.FLAG_MIN[mk] and abs(d) / avg >= .10:
                return "<span class='badge hit'>▲ over</span>" if d > 0 else "<span class='badge miss'>▼ under</span>"
            return ""

        def dvp_txt(v):
            if pd.isna(v):
                return "–"
            col = "#86efac" if v >= .10 else ("#fca5a5" if v <= -.10 else "#cbd5e1")
            return f"<b style='color:{col}'>{v:+.0%}</b>"

        def cell(proj, avg, mk, actual=None, show=True, ok=True):
            if not show:
                return "<td>–</td>"
            fmt = "{:.1f}" if mk in ("rec", "car") else "{:.0f}"
            act = f"<br><span class='card-sub'>act {fmt.format(actual)}</span>" if actual is not None and pd.notna(actual) else ""
            return (f"<td><b>{fmt.format(proj)}</b> <span class='card-sub'>avg {fmt.format(avg)}</span> "
                    f"{lean(proj, avg, mk, ok)}{act}</td>")

        st_html(f"<div class='meta'>Projections use each player's season so far, last season, role trend, the "
                f"defense's allowed stats and defense vs position, and the game flow model's script for this game "
                f"({g.away_team} {gp['away_score']:.1f} – {g.home_team} {gp['home_score']:.1f}, {gp['total_plays']:.0f} plays). "
                "<b>Def vs pos</b> = what this defense allows to the player's position, vs league average "
                "(green = soft matchup; QBs show rushing allowed to QBs). ▲/▼ = projection 10%+ off the season average for "
                "regular players (3+ games, 2.5+ targets or 4+ carries), the gaps that held up in testing."
                + (" Players ruled Out or Doubtful are removed." if not played else "") + "</div>")
        for team, opp in ((g.away_team, g.home_team), (g.home_team, g.away_team)):
            t_sk = sk[sk.posteam == team].copy()
            t_qb = qb[qb.posteam == team]
            dv = t_sk.groupby("pos")[["dvp_rec", "dvp_rush"]].mean()
            qdv = t_qb.dvp_pass.mean() if len(t_qb) else np.nan
            chips = " · ".join([f"QB pass {dvp_txt(qdv)}"] +
                               [f"{p} rec {dvp_txt(dv.dvp_rec.get(p))}" for p in ("WR", "TE", "RB") if p in dv.index] +
                               ([f"RB rush {dvp_txt(dv.dvp_rush.get('RB'))}"] if "RB" in dv.index else []))
            qrows = "".join(
                f"<tr><td><b>{r.name}</b> QB{' (Q)' if r.pid in q_ids and not played else ''}</td>"
                + cell(r.proj_pass_yds, r.p_pass_yds, "pass_yds", r.pass_yds if played else None, True, r.n >= 3)
                + f"<td>{dvp_txt(r.dvp_pass)}</td></tr>" for r in t_qb.itertuples())
            t_sk["order"] = t_sk.proj_rec_yds + t_sk.proj_rush_yds
            srows = "".join(
                f"<tr><td><b>{r.name}</b> {r.pos}{' (Q)' if r.pid in q_ids and not played else ''}</td>"
                + cell(r.proj_rec, r.p_rec, "rec", r.rec if played else None, r.p_tgt >= 1, r.n >= 3 and r.p_tgt >= 2.5)
                + cell(r.proj_rec_yds, r.p_rec_yds, "rec_yds", r.rec_yds if played else None, r.p_tgt >= 1,
                       r.n >= 3 and r.p_tgt >= 2.5)
                + cell(r.proj_car, r.p_car, "car", r.car if played else None, r.p_car >= 2, r.n >= 3 and r.p_car >= 4)
                + cell(r.proj_rush_yds, r.p_rush_yds, "rush_yds", r.rush_yds if played else None, r.p_car >= 2,
                       r.n >= 3 and r.p_car >= 4)
                + f"<td><b>{r.proj_td:.0%}</b>{'<br><span class=card-sub>TD</span>' if played and r.td > 0 else ''}</td>"
                + f"<td>{dvp_txt(r.dvp_rush if r.pos == 'QB' or (r.pos == 'RB' and r.p_car >= r.p_tgt) else r.dvp_rec)}</td></tr>"
                for r in t_sk.sort_values("order", ascending=False).itertuples())
            st_html(f"""
            <div class='card'>
              <div class='card-head'><div class='card-title'>{team} vs {opp} defense</div>
                <div class='card-sub'>projected {ctx[team]['c_pts']:.1f} pts</div></div>
              <div class='meta'>{opp} allows: {chips}</div>
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
            line = lc2.number_input("DK line", 0.0, 600.0, float(round(proj * 2) / 2 - 0.5 if proj > 1 else 0.5), 0.5,
                                    key=f"pl_line_{pick}")
            m = PMODEL["markets"][mk]
            med = proj + m.get("median_shift", 0)
            p_over = 1 - norm_cdf((line - med) / m["sd"])
            side = "OVER" if p_over > .5 else "UNDER"
            st_html(f"<div class='card'><div class='summary'><span>Projection <b>{proj:.1f}</b></span>"
                    f"<span>Typical outcome (median) <b>{med:.1f}</b></span><span>Over {line:g} <b>{p_over:.0%}</b></span>"
                    f"<span>Under <b>{1 - p_over:.0%}</b></span><span>Model side <b>{side}</b></span></div>"
                    f"<div class='meta'>Rough odds from the model's typical miss ({m['sd']:.0f} on yards in testing). "
                    "Not tested against real DraftKings lines, so treat it as a lean, not an edge.</div></div>")

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
