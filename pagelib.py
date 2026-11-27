"""The weekly page ("NFL This Week" artifact) inside the app.

page/<season>_wk<week>.json is the exact data the web page is built from (simplify/build_page.sh writes it as
simplify/page_data_<S>_wk<W>.json). This module turns its picks into bets.csv-style rows so livelib.grade_bet can
grade them live, and keeps the page's plain-language wording. Kept free of Streamlit so it can be tested offline.
"""
import json
import re
from pathlib import Path

import pandas as pd

import livelib as lv

TIERS = {  # same labels and rates as the web page
    "BET": dict(label="BET", title="Bets", rate="Teaser legs win ~77% (need 74%)"),
    "LEAN": dict(label="LEAN", title="Leans", rate="~52-54%"),
    "PROP": dict(label="PROPS", title="Player props", rate="~54% vs season avg, unproven vs DK"),
    "WATCH": dict(label="WATCH", title="Watch only", rate="113-111 historically"),
    "HOLD": dict(label="HOLD", title="Confirm the QB first", rate=""),
}
PROP_RE = re.compile(r"(.+?) (passing|rushing|receiving) yards (over|under)(?: ([\d.]+))?")


def page_path(here, season, week):
    return Path(here) / "page" / f"{season}_wk{week}.json"


def page_weeks(here, season):
    out = []
    for p in (Path(here) / "page").glob(f"{season}_wk*.json"):
        m = re.search(r"_wk(\d+)\.json$", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def load_page(here, season, week):
    p = page_path(here, season, week)
    return json.loads(p.read_text()) if p.exists() else None


def plain(p, games):
    """The page's one-line 'why' for a pick (same wording as the web page)."""
    if p["cat"] == "Teaser leg":
        return ("Small underdog. Teasing 6 points moves it past 3 and 7, the two most common final margins. "
                "Pair it with another leg.")
    if p["cat"] == "Coach fade":
        return p.get("why") or ""
    if p["cat"] == "Primetime under":
        return "Night-game unders have hit 55% since 2015, but only 46-50 since 2024."
    if p["cat"] == "Watch total":
        g = games.get(p["game"], {})
        return f"Model says {g.get('model_total')}, market {g.get('total')}. Gaps this big have been a coin flip."
    why = p.get("why") or ""
    k = re.search(r"Projection (\d+) vs DK ([\d.]+) \(season avg (\d+)\)", why)
    if k:
        return f"Projects {k[1]} yds vs the DK line of {k[2]} (season avg {k[3]})."
    m = re.search(r"Projection (\d+) vs season avg (\d+)", why)
    return f"Projects {m[1]} yds vs his {m[2]} average." if m else why


def pick_to_bet(p):
    """One page pick -> a bets.csv-style dict for livelib.grade_bet, or None if it cannot be graded."""
    pick, cat = p["pick"], p["cat"]
    base = dict(game=p["game"], tier=p["tier"], bet=cat, player=None, team=None, model=None, note="")
    nums = re.findall(r"[+-]?\d+(?:\.\d+)?", pick)
    if cat == "Coach fade":
        # 'PIT +3.5 (-101)': the spread is the first number, the DK odds the second
        return {**base, "market": "spread", "side": pick.split()[0], "line": float(nums[0])}
    if cat == "Teaser leg" or cat.startswith("Injury"):
        # 'GB +2.5 -> +8.5': grade the teased number (last one)
        return {**base, "market": "spread", "side": pick.split()[0], "line": float(nums[-1])}
    if pick.split()[0].lower() in ("under", "over") and nums:
        return {**base, "market": "total", "side": pick.split()[0].lower(), "line": float(nums[-1])}
    m = PROP_RE.match(pick)
    if m:
        name, stat, side, ln = m.groups()
        line = float(ln) if ln else p.get("line")
        if line is None:
            return None
        pj = re.search(r"Projection (\d+)", p.get("why") or "")
        return {**base, "market": f"{stat}_yards", "player": name, "side": side, "line": float(line),
                "model": float(pj[1]) if pj else None}
    return None


def player_team(name, game, box):
    """Which side of the game a short name like 'M.Golden' plays for, from the live box score."""
    away, home = game.split("@")
    for tm in (home, away):
        if lv.match_player(name, tm, box or {})[0] is not None:
            return tm
    return home


def grade_pick(p, g, box=None, game_model=None):
    """Live status for one page pick: dict(status, now, pace, margin, text) as livelib.grade_bet returns."""
    b = pick_to_bet(p)
    if b is None:
        return {"status": "pending", "now": None, "pace": None, "margin": None, "text": "not gradable"}
    if b["player"]:
        b["team"] = player_team(b["player"], b["game"], box)
    return lv.grade_bet(pd.Series(b), g, box, game_model)


def same_player(short, full):
    """'A.St. Brown' / 'K.Johnson' vs a full name like 'Amon-Ra St. Brown' (first initial + last name)."""
    if not short or not full or "." not in short:
        return False
    first, last = short.split(".", 1)
    fp = str(full).split(" ", 1)
    if len(fp) < 2:
        return False
    norm = lambda x: "".join(ch for ch in str(x).lower() if ch.isalpha())
    sfx = {"jr", "sr", "ii", "iii", "iv", "v"}
    rest = " ".join(w for w in fp[1].split() if norm(w) not in sfx)
    return norm(rest) == norm(last) and norm(fp[0]).startswith(norm(first))


def angle_sides(page, game, full_name, market):
    """(side, angle) for each of this game's matchup angles that names this player prop."""
    out = []
    for a in (page or {}).get("angles", []):
        k = a.get("prop_key")
        if k and k[0] == game and k[2] == market and same_player(k[1], full_name):
            out.append((k[3], a))
    return out
