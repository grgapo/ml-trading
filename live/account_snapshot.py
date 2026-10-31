"""Write the Alpaca paper account's live state to STATE_DIR/account.json for the lab app.

Equity, cash, buying power, today's change (vs yesterday's close) and every open
position with its unrealized P&L. Runs every 5 minutes from a systemd timer; read-only
(it never places orders). Paper only: the client is hard-wired to paper=True.

  .venv/bin/python live/account_snapshot.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
STATE_DIR = Path(os.environ.get("STATE_DIR", Path.home() / "projects" / "paper-state"))


def snapshot() -> dict:
    from alpaca.trading.client import TradingClient
    client = TradingClient(os.environ["ALPACA_API_KEY_ID"], os.environ["ALPACA_API_SECRET_KEY"], paper=True)
    acct = client.get_account()
    clock = client.get_clock()
    equity, last = float(acct.equity), float(acct.last_equity)
    positions = []
    for p in client.get_all_positions():
        positions.append({
            "symbol": p.symbol, "asset_class": str(getattr(p.asset_class, "value", p.asset_class)),
            "qty": float(p.qty), "avg_entry": float(p.avg_entry_price),
            "price": float(p.current_price or 0), "market_value": float(p.market_value or 0),
            "unrealized_pl": float(p.unrealized_pl or 0), "unrealized_plpc": float(p.unrealized_plpc or 0),
            "today_pl": float(p.unrealized_intraday_pl or 0),
        })
    positions.sort(key=lambda x: -abs(x["market_value"]))
    return {
        "updated": dt.datetime.now(NY).isoformat(timespec="seconds"),
        "market_open": bool(clock.is_open),
        "equity": equity, "last_equity": last, "day_change": equity - last,
        "day_change_pct": (equity / last - 1) if last else 0.0,
        "cash": float(acct.cash), "buying_power": float(acct.buying_power),
        "long_value": float(acct.long_market_value or 0), "start_equity": 100_000.0,
        "positions": positions,
    }


def main() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = snapshot()
    except Exception as exc:
        data = {"updated": dt.datetime.now(NY).isoformat(timespec="seconds"), "error": f"{type(exc).__name__}: {exc}"}
    tmp = STATE_DIR / "account.json.tmp"
    tmp.write_text(json.dumps(data))
    tmp.replace(STATE_DIR / "account.json")
    print(data.get("error") or f"equity ${data['equity']:,.2f} ({data['day_change']:+,.2f} today), {len(data['positions'])} positions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
