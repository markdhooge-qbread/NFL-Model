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


def fetch_scoreboard(dates=None, timeout=10):
    url = SCOREBOARD + (f"?dates={dates}" if dates else "")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (nfl-model live tracker)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


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


def load_flags(path):
    try:
        f = pd.read_csv(path, dtype={"line": float})
    except FileNotFoundError:
        return pd.DataFrame(columns=["season", "week", "game", "tier", "flag", "market", "side", "line", "note"])
    f["tier_rank"] = f.tier.map(TIER_ORDER).fillna(9)
    return f


def grade_flag(flag, g, pace=None):
    """Return (status, margin, text). status: pending / winning / losing / push / won / lost.

    market 'total': side over/under at line. market 'spread': side = backed team, line = that team's spread.
    margin > 0 means the flag is ahead by that many points. For a total in progress, pass the projected final
    total as pace so an under isn't called "winning" just because the game is young.
    """
    if g is None:
        return "pending", None, "game not on the scoreboard"
    if g["state"] == "pre":
        return "pending", None, "not started"
    line = float(flag["line"])
    if flag["market"] == "total":
        cur = g["total"] if g["state"] == "post" or pace is None else pace
        m = (cur - line) if flag["side"] == "over" else (line - cur)
        txt = f"total {g['total']:.0f}" + (f", on pace {pace:.1f}" if g["state"] == "in" and pace is not None else "")
        if flag["side"] == "over" and g["total"] > line:
            m = g["total"] - line  # over already cashed
    else:
        side = flag["side"]
        if side == g["home"]:
            diff = g["home_score"] - g["away_score"]
        elif side == g["away"]:
            diff = g["away_score"] - g["home_score"]
        else:
            return "pending", None, f"team {side} not in this game"
        m = diff + line
        txt = f"{side} {line:+g}: {'up' if diff > 0 else 'down' if diff < 0 else 'tied'} {abs(diff):.0f}"
    if g["state"] == "post":
        status = "won" if m > 0 else "lost" if m < 0 else "push"
    else:
        status = "winning" if m > 0 else "losing" if m < 0 else "push"
        if flag["market"] == "total" and flag["side"] == "under" and g["total"] > line:
            status = "lost"  # under already busted
        if flag["market"] == "total" and flag["side"] == "over" and g["total"] > line:
            status = "won"
    return status, m, txt
