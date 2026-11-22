"""Live tracker helpers: ESPN public scoreboard + grading the model's flags against live scores.

Kept free of Streamlit so it can be tested offline. ESPN's scoreboard JSON needs no key:
https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard
"""
import json
import urllib.request

import pandas as pd

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
# ESPN abbreviations that differ from nflverse
ESPN_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA"}
TIER_ORDER = {"BET": 0, "LEAN": 1, "WATCH": 2}


def team(abbr):
    return ESPN_TO_NFLVERSE.get(abbr, abbr)


SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={}"
# bets.csv prop market -> (ESPN box score category, stat key, fallback column label)
PROP_STATS = {
    "passing_yards": ("passing", "passingYards", "YDS"),
    "rushing_yards": ("rushing", "rushingYards", "YDS"),
    "receiving_yards": ("receiving", "receivingYards", "YDS"),
    "receptions": ("receiving", "receptions", "REC"),
    "rush_attempts": ("rushing", "rushingAttempts", "CAR"),
}


def _get(url, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (nfl-model live tracker)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def fetch_scoreboard(dates=None, timeout=10):
    return _get(SCOREBOARD + (f"?dates={dates}" if dates else ""), timeout)


def fetch_week(season, week, timeout=10):
    return _get(f"{SCOREBOARD}?dates={season}&seasontype=2&week={week}", timeout)


def fetch_summary(event_id, timeout=10):
    return _get(SUMMARY.format(event_id), timeout)


def parse_box(data):
    """{team: [(display_name, short_name, {market: value})]} from an ESPN game summary."""
    out = {}
    for tb in (data.get("boxscore") or {}).get("players", []):
        tm = team(tb.get("team", {}).get("abbreviation", ""))
        players = {}
        for cat in tb.get("statistics", []):
            keys, labels = cat.get("keys") or [], cat.get("labels") or []
            for ath in cat.get("athletes", []):
                a = ath.get("athlete", {})
                nm = a.get("displayName", "")
                rec = players.setdefault(nm, (nm, a.get("shortName", ""), {}))
                vals = ath.get("stats", [])
                for mk, (c, key, lab) in PROP_STATS.items():
                    if cat.get("name") != c:
                        continue
                    i = keys.index(key) if key in keys else labels.index(lab) if lab in labels else None
                    if i is not None and i < len(vals):
                        v = _num(str(vals[i]).replace(",", ""))
                        if v is not None:
                            rec[2][mk] = v
        out[tm] = list(players.values())
    return out


def _norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalpha())


def match_player(name, tm, box):
    """Match 'T.McBride' / 'Mi.Wilson' / 'Trey McBride' to an ESPN box score row on that team."""
    rows = box.get(tm, [])
    name = str(name).strip()
    if "." in name and " " not in name:
        first, last = name.split(".", 1)
    else:
        parts = name.split(" ", 1)
        first, last = (parts[0], parts[1]) if len(parts) == 2 else ("", name)
    first, last = _norm(first), _norm(last)
    for full, _short, stats in rows:
        fp = full.split(" ", 1)
        if len(fp) == 2 and _norm(fp[1]) == last and _norm(fp[0]).startswith(first):
            return full, stats
    # suffixes (Jr., III) or odd spacing: fall back to last name contained in the full name
    hits = [(f, s) for f, _s, s in rows if last and last in _norm(f) and _norm(f).startswith(first[:1])]
    return hits[0] if len(hits) == 1 else (None, None)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_scoreboard(data):
    """One dict per game, teams in nflverse abbreviations."""
    out = []
    for ev in data.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        h, a = sides.get("home", {}), sides.get("away", {})
        st = ev.get("status") or comp.get("status") or {}
        stype = st.get("type", {})
        sit = comp.get("situation") or {}
        odds = (comp.get("odds") or [{}])[0]
        poss_id = sit.get("possession")
        poss = None
        for c in (h, a):
            if poss_id and c.get("team", {}).get("id") == poss_id:
                poss = team(c["team"].get("abbreviation"))
        g = {
            "espn_id": ev.get("id"),
            "week": int((ev.get("week") or {}).get("number") or 0),
            "kick_utc": pd.to_datetime(ev.get("date"), utc=True, errors="coerce"),
            "home": team(h.get("team", {}).get("abbreviation", "")),
            "away": team(a.get("team", {}).get("abbreviation", "")),
            "home_score": _num(h.get("score")) or 0.0,
            "away_score": _num(a.get("score")) or 0.0,
            "home_q": [_num(x.get("value")) or 0.0 for x in h.get("linescores", [])],
            "away_q": [_num(x.get("value")) or 0.0 for x in a.get("linescores", [])],
            "state": stype.get("state", "pre"),  # pre / in / post
            "detail": stype.get("shortDetail") or stype.get("detail") or "",
            "period": int(st.get("period") or 0),
            "clock": float(st.get("clock") or 0),
            "possession": poss,
            "down_dist": sit.get("downDistanceText") or sit.get("shortDownDistanceText") or "",
            "red_zone": bool(sit.get("isRedZone")),
            "last_play": (sit.get("lastPlay") or {}).get("text", ""),
            "espn_line": odds.get("details") or "",
            "espn_total": _num(odds.get("overUnder")),
            "indoor": bool((comp.get("venue") or {}).get("indoor")),
            "broadcast": ", ".join(n for b in comp.get("broadcasts", []) for n in b.get("names", [])),
        }
        g["game"] = f"{g['away']}@{g['home']}"
        g["total"] = g["home_score"] + g["away_score"]
        g["elapsed"] = elapsed_seconds(g)
        out.append(g)
    return out


def elapsed_seconds(g):
    """Game seconds played (regulation 3600; OT counts past it)."""
    if g["state"] == "pre":
        return 0.0
    if g["state"] == "post":
        return max(3600.0, 3600.0 + 600.0 * (g["period"] - 4)) if g["period"] > 4 else 3600.0
    p = max(g["period"], 1)
    qlen = 900.0 if p <= 4 else 600.0
    done = 900.0 * min(p - 1, 4) + 600.0 * max(p - 5, 0)
    return done + max(qlen - g["clock"], 0.0)


def pace_total(g, model_total=None):
    """Projected final total: points so far plus the model's rate for the time left (falls back to the live pace)."""
    el = g["elapsed"]
    if g["state"] == "post":
        return g["total"]
    left = max(3600.0 - el, 0.0)
    if model_total is not None:
        return g["total"] + model_total * left / 3600.0
    return g["total"] * 3600.0 / el if el > 0 else None


BET_COLS = ["season", "week", "game", "tier", "bet", "market", "player", "team", "side", "line", "model", "note"]


def load_bets(path):
    try:
        f = pd.read_csv(path, dtype={"line": float, "model": float})
    except FileNotFoundError:
        return pd.DataFrame(columns=BET_COLS + ["tier_rank"])
    for c in BET_COLS:
        if c not in f:
            f[c] = None
    f["note"] = f.note.fillna("")
    f["tier_rank"] = f.tier.map(TIER_ORDER).fillna(9)
    return f


def left_frac(g):
    return 0.0 if g["state"] == "post" else max(3600.0 - g["elapsed"], 0.0) / 3600.0


def grade_bet(bet, g, box=None, game_model=None):
    """Grade any bets.csv row. Returns dict(status, now, pace, margin, text).

    status: pending / winning / losing / push / won / lost / void. margin > 0 = ahead of the line, measured on pace
    while live. game_model is the game flow projection dict (home margin 'result', 'total') for team bets.
    """
    out = {"status": "pending", "now": None, "pace": None, "margin": None, "text": ""}
    if g is None:
        out["text"] = "game not found"
        return out
    if g["state"] == "pre":
        out["text"] = "not started"
        return out
    line, side, mk, lf = float(bet["line"]), str(bet["side"]).lower(), bet["market"], left_frac(g)
    if mk == "spread":
        tm = str(bet["side"])
        if tm not in (g["home"], g["away"]):
            out["text"] = f"{tm} not in this game"
            return out
        sign = 1 if tm == g["home"] else -1
        now = sign * (g["home_score"] - g["away_score"])
        mm = sign * game_model["result"] if game_model is not None else 0.0
        pace = now + mm * lf
        out.update(now=now, pace=pace, margin=pace + line, text=f"{tm} {line:+g}: {'up' if now > 0 else 'down' if now < 0 else 'tied'} {abs(now):.0f}")
    else:
        if mk == "total":
            now = g["total"]
            model = game_model["total"] if game_model is not None else None
            label = "total"
        else:
            name, stats = match_player(bet["player"], bet["team"], box or {})
            if name is None and g["state"] == "post":
                out.update(status="void", text=f"{bet['player']} has no stats (inactive?)")
                return out
            now = (stats or {}).get(mk, 0.0)
            model = bet.get("model")
            model = None if model is None or pd.isna(model) else float(model)
            label = PROP_STATS.get(mk, ("", "", mk))[0] + " yds" if "yards" in mk else mk.replace("_", " ")
        if lf <= 0:
            pace = now
        elif model is not None:
            pace = now + model * lf
        else:
            el = g["elapsed"]
            pace = now * 3600.0 / el if el > 0 else now
        margin = (pace - line) if side == "over" else (line - pace)
        out.update(now=now, pace=pace, margin=margin,
                   text=f"{label} {now:.0f}" + (f", pace {pace:.1f}" if g["state"] == "in" else ""))
    m = out["margin"]
    if g["state"] == "post":
        out["status"] = "won" if m > 0 else "lost" if m < 0 else "push"
    else:
        out["status"] = "winning" if m > 0 else "losing" if m < 0 else "push"
        if mk != "spread" and out["now"] > line:  # an over already cashed / an under already busted
            out["status"] = "won" if side == "over" else "lost"
    return out
