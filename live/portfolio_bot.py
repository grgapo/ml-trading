"""Whole-market portfolio bot (Alpaca paper), running the research champion on a $75,000 sleeve.

The options bots only ever hold options, so this bot owns EVERY stock/ETF position in the account.
Its sleeve value = market value of those positions + its own cash (tracked here). Once a month
(first trading day, or when the champion changes) it rebalances to the champion's target weights:
sells first, then buys, using fractional orders during market hours. Every run records the
sleeve's value alongside SPY and 60/40 for the app.

  .venv/bin/python live/portfolio_bot.py              # weekdays 10:00 New York (timer)
  .venv/bin/python live/portfolio_bot.py --dry-run    # print the orders, place nothing
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import portfolio as P  # noqa: E402
from portfolio_search import prices  # noqa: E402

NY = ZoneInfo("America/New_York")
STATE_DIR = Path(os.environ.get("STATE_DIR", Path.home() / "projects" / "paper-state" / "portfolio"))
SLEEVE = float(os.environ.get("PORTFOLIO_SLEEVE", "75000"))
MIN_TRADE = 50.0


def load(name, default):
    p = STATE_DIR / name
    return json.loads(p.read_text()) if p.exists() else default


def save(name, data):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_DIR / (name + ".tmp"); tmp.write_text(json.dumps(data)); tmp.replace(STATE_DIR / name)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); args = ap.parse_args()
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
    tc = TradingClient(os.environ["ALPACA_API_KEY_ID"], os.environ["ALPACA_API_SECRET_KEY"], paper=True)
    now = dt.datetime.now(NY); today = str(now.date())
    clock = tc.get_clock()
    champ = load("champion.json", None)
    book = load("book.json", {"cash": None, "start": today, "start_value": SLEEVE, "last_rebalance": None,
                               "champion": None, "history": []})
    held = {p.symbol: p for p in tc.get_all_positions() if "option" not in str(p.asset_class).lower()}
    value_pos = sum(float(p.market_value) for p in held.values())
    if book["cash"] is None:                                  # first run: the sleeve adopts whatever stocks are held
        book["cash"] = SLEEVE - value_pos
    sleeve = value_pos + book["cash"]
    px = prices(refresh_days=0.5)
    px = px[[c for c in P.UNIVERSE if c in px.columns]]
    actions = []
    status = "waiting for research" if not champ else "holding"
    due = champ and clock.is_open and (book["last_rebalance"] is None or book["last_rebalance"][:7] != today[:7]
                                       or book["champion"] != champ["desc"])
    if due:
        status = "rebalancing"
        target = P.target_weights(px, champ["params"])
        want = {s: float(w) * sleeve for s, w in target.items()}
        have = {s: float(p.market_value) for s, p in held.items()}
        sells = {s: have.get(s, 0) - want.get(s, 0) for s in set(have) | set(want) if have.get(s, 0) - want.get(s, 0) > MIN_TRADE}
        buys = {s: want.get(s, 0) - have.get(s, 0) for s in set(want) if want.get(s, 0) - have.get(s, 0) > MIN_TRADE}
        for s, amt in sorted(sells.items(), key=lambda x: -x[1]):
            if s not in want:
                actions.append(f"sell all {s}")
                if not args.dry_run:
                    tc.close_position(s)
            else:
                qty = round(amt / float(held[s].current_price), 4)
                actions.append(f"sell {qty} {s} (${amt:,.0f})")
                if not args.dry_run and qty > 0:
                    tc.submit_order(MarketOrderRequest(symbol=s, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
            book["cash"] += amt
        for s, amt in sorted(buys.items(), key=lambda x: -x[1]):
            amt = min(amt, max(0.0, book["cash"]))
            if amt < MIN_TRADE:
                continue
            actions.append(f"buy ${amt:,.0f} {s}")
            if not args.dry_run:
                tc.submit_order(MarketOrderRequest(symbol=s, notional=round(amt, 2), side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
            book["cash"] -= amt
        if not args.dry_run:
            book["last_rebalance"] = today; book["champion"] = champ["desc"]
    spy = float(px["SPY"].dropna().iloc[-1]); ief = float(px["IEF"].dropna().iloc[-1])
    if not book["history"] or book["history"][-1][0] != today:
        book["history"].append([today, round(sleeve, 2), spy, ief])
    else:
        book["history"][-1] = [today, round(sleeve, 2), spy, ief]
    if not args.dry_run:
        save("book.json", book)
    h = book["history"]; s0, i0 = h[0][2], h[0][3]
    curve = [{"date": d, "sleeve": v / h[0][1], "spy": sp / s0, "sixty40": 0.6 * sp / s0 + 0.4 * ie / i0} for d, v, sp, ie in h]
    positions = [{"symbol": s, "name": P.UNIVERSE.get(s, s), "class": P.CLASS.get(s, "other"), "value": float(p.market_value),
                  "weight": float(p.market_value) / sleeve if sleeve else 0, "pl": float(p.unrealized_pl or 0),
                  "plpc": float(p.unrealized_plpc or 0)} for s, p in held.items()]
    positions.sort(key=lambda x: -x["value"])
    result = {"updated": now.isoformat(timespec="seconds"), "phase": status, "sleeve": round(sleeve, 2), "start_value": book["start_value"],
              "cash": round(book["cash"], 2), "since_start": round(sleeve / book["start_value"] - 1, 4), "champion": champ,
              "last_rebalance": book["last_rebalance"], "positions": positions, "curve": curve[-400:], "actions": actions,
              "target": P.target_weights(px, champ["params"]).round(4).to_dict() if champ else {}}
    if not args.dry_run:
        save("status.json", result)
    print(f"{today} sleeve ${sleeve:,.0f} ({status})" + ("".join(f"\n  {a}" for a in actions) if actions else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
