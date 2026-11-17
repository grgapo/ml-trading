"""playbook.py: regime playbook sizing (passed its honest test Nov 17 2026: locked months +$12.0k vs +$0.8k).

Rule (fixed before the test, unchanged here):
  * the day type comes from QQQ's first trading hour (9:30-10:30 New York):
      strong = the first-hour move is bigger than its usual size (average of the last 20 days), up or down
      quiet  = smaller than usual
  * a trade opened at/after 10:30 gets FULL size on a strong day and QUARTER size (at least 1 contract) on a quiet day
  * a trade before 10:30 is unchanged (the day type isn't known yet)
  * NEVER stops a bot for the day; if the day type can't be worked out, trades go at full size (today's behaviour)
The day type is worked out once per day and shared: ~/projects/paper-state/regime/today.json
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
QUIET_W = float(os.environ.get("PLAYBOOK_QUIET_WEIGHT", "0.25"))
KNOWN_AT = dt.time(10, 30)
CACHE = Path(os.environ.get("PLAYBOOK_CACHE", Path.home() / "projects/paper-state/regime/today.json"))


def classify(first_moves: list[float], today_move: float) -> dict:
    """first_moves: the previous 20+ days' first-hour returns (oldest first); today_move: today's."""
    prior = [abs(x) for x in first_moves[-20:]]
    usual = sum(prior) / len(prior) if len(prior) >= 10 else 0.0
    z = today_move / usual if usual > 0 else 0.0
    kind = "strong up" if z > 1 else ("strong down" if z < -1 else "quiet")
    return {"type": kind, "first_hour": round(today_move, 5), "usual": round(usual, 5), "z": round(z, 2)}


def _first_hours(stocks, today: dt.date) -> tuple[list[float], float | None]:
    """First-hour return per session from 30-minute QQQ bars (9:30 bar's open -> 10:00 bar's close)."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    req = StockBarsRequest(symbol_or_symbols="QQQ", timeframe=TimeFrame(30, TimeFrameUnit.Minute),
                           start=dt.datetime.combine(today - dt.timedelta(days=45), dt.time(0), NY), feed=DataFeed.IEX)
    per_day: dict[dt.date, dict] = {}
    for b in stocks.get_stock_bars(req).data.get("QQQ", []):
        t = b.timestamp.astimezone(NY)
        if t.time() == dt.time(9, 30):
            per_day.setdefault(t.date(), {})["o"] = float(b.open)
        elif t.time() == dt.time(10, 0):
            per_day.setdefault(t.date(), {})["c"] = float(b.close)
    moves = {d: v["c"] / v["o"] - 1 for d, v in sorted(per_day.items()) if "o" in v and "c" in v and v["o"] > 0}
    return [m for d, m in moves.items() if d < today], moves.get(today)


def day_type(stocks, now: dt.datetime) -> dict | None:
    today = now.date()
    try:
        c = json.loads(CACHE.read_text())
        if c.get("date") == str(today):
            return c
    except Exception:
        pass
    if now.time() < KNOWN_AT or stocks is None:
        return None
    prior, today_move = _first_hours(stocks, today)
    if today_move is None:
        return None
    out = {"date": str(today), **classify(prior, today_move), "made": now.isoformat(timespec="seconds")}
    if out["usual"] <= 0:
        return None
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(".tmp"); tmp.write_text(json.dumps(out)); tmp.replace(CACHE)
    except Exception:
        pass
    return out


def size(qty: int, stocks, now: dt.datetime) -> tuple[int, str]:
    """The playbook's size for a trade the bot wants to open now at `qty` contracts, plus a one-line reason."""
    if qty < 1:
        return qty, ""
    if now.astimezone(NY).time() < KNOWN_AT:
        return qty, "playbook: before 10:30, full size"
    try:
        t = day_type(stocks, now.astimezone(NY))
    except Exception as exc:
        return qty, f"playbook: day type unavailable ({type(exc).__name__}), full size"
    if t is None:
        return qty, "playbook: day type unknown, full size"
    if t["type"] == "quiet":
        new = max(1, int(qty * QUIET_W))
        return new, f"playbook: quiet first hour ({t['z']:+.1f}x usual) -> {new} of {qty} contracts"
    return qty, f"playbook: {t['type']} first hour ({t['z']:+.1f}x usual), full size"
