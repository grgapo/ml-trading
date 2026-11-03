"""Shared trading module: the ONE place the lab's bots buy, sell and confirm, plus the risk hooks.

Rules that every bot gets from here (learned the hard way on Nov 2, 2026):
  * A fill is confirmed by the ACTUAL POSITION at the broker, never by parsing order-status text.
  * If a buy can't be confirmed in time, its order is cancelled and the position re-checked before giving up.
  * A sell is confirmed when the position is gone (or reduced by the sold amount).
  * Every bot publishes what it believes it holds ("claims"); the risk guard compares that with the broker.
  * Before any new entry a bot asks halted(bot_id): the risk guard (or Greg) can pause one bot or all of them.
Exits are always allowed while halted: pausing must never trap a position.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
RISK = Path(os.environ.get("RISK_DIR", Path.home() / "projects" / "paper-state" / "risk"))


def _save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(data, default=str)); tmp.replace(path)


# ---- risk hooks ---------------------------------------------------------------------------------------------
def halted(bot_id: str) -> tuple[bool, str]:
    """(True, reason) if the risk guard or Greg has paused new entries for this bot (or for everything).
    Pauses carry an `until` date; a manual one (no date) lasts until the file entry is removed."""
    f = RISK / "halt.json"
    try:
        h = json.loads(f.read_text()) if f.exists() else {}
    except Exception:
        return True, "halt file unreadable (pausing to be safe)"
    today = str(dt.datetime.now(NY).date())
    for key in ("all", bot_id):
        e = h.get(key)
        if e and (e.get("until") is None or e["until"] >= today):
            return True, f"{'everything' if key == 'all' else key} paused: {e.get('reason', '?')}"
    return False, ""


def set_halt(key: str, reason: str, until: str | None) -> None:
    f = RISK / "halt.json"
    h = json.loads(f.read_text()) if f.exists() else {}
    h[key] = {"reason": reason, "until": until, "since": dt.datetime.now(NY).isoformat(timespec="seconds")}
    _save(f, h)


def clear_halt(key: str) -> None:
    f = RISK / "halt.json"
    if f.exists():
        h = json.loads(f.read_text()); h.pop(key, None); _save(f, h)


def size_cap(bot_id: str) -> int | None:
    """Max contracts/shares per trade while new order code proves itself (risk/size_cap.json), or None."""
    f = RISK / "size_cap.json"
    try:
        e = (json.loads(f.read_text()) if f.exists() else {}).get(bot_id)
    except Exception:
        return 1                                           # unreadable: smallest size to be safe
    if e and e.get("until", "9999") >= str(dt.datetime.now(NY).date()):
        return int(e["max_qty"])
    return None


def write_claims(bot_id: str, holdings: dict[str, float]) -> None:
    """What this bot believes it holds right now ({symbol: qty}); the risk guard checks it every minute."""
    _save(RISK / "claims" / f"{bot_id}.json", {"updated": dt.datetime.now(NY).isoformat(timespec="seconds"),
                                               "holdings": {k: float(v) for k, v in holdings.items() if v}})


# ---- orders -------------------------------------------------------------------------------------------------
class Broker:
    """Buy/sell with confirmation by position. `client` is an alpaca-py TradingClient (or a test fake)."""

    def __init__(self, client, confirm_s: int = 60, poll_s: float = 1.0):
        self.c, self.confirm_s, self.poll_s = client, confirm_s, poll_s

    def held(self, symbol: str):
        """(qty, avg entry price) at the broker, or None."""
        try:
            p = self.c.get_open_position(symbol)
            q = float(p.qty)
            return (q, float(p.avg_entry_price)) if q else None
        except Exception:
            return None

    def cancel_open(self, symbol: str) -> None:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        for o in self.c.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])):
            self.c.cancel_order_by_id(o.id)

    def buy(self, symbol: str, qty: float, tag: str = "lab"):
        """Market buy. Returns (avg price of the shares this order added, qty added), or None if nothing filled."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        before = self.held(symbol) or (0.0, 0.0)
        self.c.submit_order(MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                                               client_order_id=f"{tag}-{symbol}-buy-{int(time.time() * 1000)}"))
        deadline = time.time() + self.confirm_s
        while time.time() < deadline:
            h = self.held(symbol)
            if h and h[0] >= before[0] + qty - 1e-9:
                break
            time.sleep(self.poll_s)
        else:
            self.cancel_open(symbol)
            time.sleep(min(2.0, self.poll_s * 2))
        h = self.held(symbol)
        if not h or h[0] <= before[0] + 1e-9:
            return None
        added = h[0] - before[0]
        price = (h[0] * h[1] - before[0] * before[1]) / added if added else h[1]
        return round(price, 4), added

    def sell(self, symbol: str, qty: float | None = None, fallback_price=None):
        """Sell `qty` (or everything). Returns the exit price once the position is confirmed reduced/gone, else None."""
        from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
        from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
        before = self.held(symbol)
        if not before:
            return None
        if qty is None or qty >= before[0]:
            self.c.close_position(symbol); target = 0.0
        else:
            self.c.submit_order(MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
            target = before[0] - qty
        deadline = time.time() + self.confirm_s
        while time.time() < deadline:
            h = self.held(symbol)
            if (h is None and target == 0) or (h and h[0] <= target + 1e-9):
                orders = self.c.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[symbol], limit=5))
                fills = [float(o.filled_avg_price) for o in orders
                         if o.filled_avg_price and str(getattr(o.side, "value", o.side)).lower() == "sell"]
                if fills:
                    return fills[0]
                return fallback_price() if callable(fallback_price) else fallback_price
            time.sleep(self.poll_s)
        return None
