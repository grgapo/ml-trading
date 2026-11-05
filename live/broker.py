"""Shared trading module: the ONE place the lab's bots buy, sell and confirm, plus the risk hooks.

Rules that every bot gets from here (learned the hard way on Nov 2, 2026):
  * A fill is confirmed by the ACTUAL POSITION at the broker, never by parsing order-status text.
  * If a buy can't be confirmed in time, its order is cancelled and the position re-checked before giving up.
  * A sell is confirmed when the position is gone (or reduced by the sold amount).
  * Every bot publishes what it believes it holds ("claims"); the risk guard compares that with the broker.
  * Before any new entry a bot asks halted(bot_id): the risk guard (or Greg) can pause one bot or all of them.
Exits are always allowed while halted: pausing must never trap a position.
Self-healing (Nov 3, 2026): one bot's trouble stops only that bot.
  * 3 orders in a row that don't fill (or are rejected) -> only that bot pauses for the day (record_order).
  * A sell that isn't confirmed is retried (cancel anything pending, re-send) up to 3 times.
  * Every order carries its bot's tag in client_order_id, so owner_of() can tell who placed a stray position.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.request
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


def notify(title: str, text: str, priority: str = "high") -> None:
    topic = os.environ.get("NTFY_TOPIC")
    print(f"[{priority}] {title}: {text}", flush=True)
    if topic:
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{topic}", data=text.encode(),
                                                          headers={"Title": title, "Priority": priority, "Tags": "warning"}), timeout=10)
        except Exception:
            pass


def notify_once(key: str, title: str, text: str, priority: str = "high") -> None:
    """notify() at most once per key per day (retries every minute must not spam Greg's phone)."""
    f = RISK / "notified.json"
    today = str(dt.datetime.now(NY).date())
    try:
        d = json.loads(f.read_text()) if f.exists() else {}
    except Exception:
        d = {}
    if d.get("day") != today:
        d = {"day": today, "keys": []}
    if key in d["keys"]:
        print(f"[{priority}] {title}: {text} (already sent today)", flush=True); return
    d["keys"].append(key); _save(f, d)
    notify(title, text, priority)


STRIKES_TO_PAUSE = int(os.environ.get("RISK_ORDER_STRIKES", "3"))
TAG_TO_BOT = {"pg": "proving", "proving": "proving", "options-spy": "options-spy", "options-qqq": "options-qqq",
              "options-exp": "options-exp"}


def record_order(bot_id: str | None, ok: bool, detail: str = "") -> int:
    """Count consecutive failed orders per bot (resets daily and on any success).
    At STRIKES_TO_PAUSE in a row, only that bot is paused for the rest of the day. Returns the streak."""
    if not bot_id:
        return 0
    f = RISK / "strikes.json"
    today = str(dt.datetime.now(NY).date())
    try:
        s = json.loads(f.read_text()) if f.exists() else {}
    except Exception:
        s = {}
    if s.get("day") != today:
        s = {"day": today}
    n = 0 if ok else int(s.get(bot_id, 0)) + 1
    s[bot_id] = n
    _save(f, s)
    if n == STRIKES_TO_PAUSE:
        set_halt(bot_id, f"{n} orders in a row didn't fill ({detail or 'no position at the broker'})", until=today)
        notify_once(f"strikes-{bot_id}", f"{bot_id} paused for today",
               f"{n} orders in a row didn't fill ({detail or 'no position at the broker'}). Only {bot_id} is paused; "
               f"the other bots keep trading. It resumes on its own tomorrow.")
    return n


def bot_from_order_id(client_order_id: str | None, symbol: str) -> str | None:
    """'options-spy-SPY261104C00600000-buy-1793...' -> 'options-spy' (None if it isn't one of ours)."""
    if not client_order_id or f"-{symbol}-" not in client_order_id:
        return None
    return TAG_TO_BOT.get(client_order_id.split(f"-{symbol}-")[0])


def owner_of(client, symbol: str) -> str | None:
    """Which bot bought `symbol` most recently, from the tag in its order id; None if unknown."""
    try:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, symbols=[symbol], limit=50))
    except Exception:
        return None
    for o in orders:                                           # newest first
        if str(getattr(o.side, "value", o.side)).lower() != "buy":
            continue
        b = bot_from_order_id(getattr(o, "client_order_id", None), symbol)
        if b:
            return b
    return None


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

    def buy(self, symbol: str, qty: float, tag: str = "lab", bot: str | None = None):
        """Market buy. Returns (avg price of the shares this order added, qty added), or None if nothing filled.
        With `bot`, a None result counts as a strike toward pausing that bot (record_order)."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        before = self.held(symbol) or (0.0, 0.0)
        try:
            self.c.submit_order(MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                                                   client_order_id=f"{tag}-{symbol}-buy-{int(time.time() * 1000)}"))
        except Exception as exc:                               # rejected: nothing was bought
            record_order(bot, False, f"rejected: {type(exc).__name__}: {str(exc)[:120]}")
            return None
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
            record_order(bot, False, "not filled in time, cancelled")
            return None
        record_order(bot, True)
        added = h[0] - before[0]
        price = (h[0] * h[1] - before[0] * before[1]) / added if added else h[1]
        return round(price, 4), added

    def _fill_price(self, symbol: str, order_id=None):
        if order_id is not None:
            try:
                o = self.c.get_order_by_id(order_id)
                if o.filled_avg_price:
                    return float(o.filled_avg_price)
            except Exception:
                pass
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        try:
            orders = self.c.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[symbol], limit=5))
        except Exception:
            return None
        fills = [float(o.filled_avg_price) for o in orders
                 if o.filled_avg_price and str(getattr(o.side, "value", o.side)).lower() == "sell"]
        return fills[0] if fills else None

    def sell(self, symbol: str, qty: float | None = None, fallback_price=None, tries: int = 3):
        """Sell `qty` (or everything). Returns the exit price once the position is confirmed reduced/gone, else None.
        An unconfirmed or rejected sell is retried: cancel anything pending for the symbol, re-check, re-send."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        first = self.held(symbol)
        if not first:
            return None
        target = 0.0 if qty is None or qty >= first[0] else first[0] - qty
        last_err = ""
        for attempt in range(1, tries + 1):
            h = self.held(symbol)
            if (h is None and target == 0) or (h and h[0] <= target + 1e-9):
                break                                          # an earlier attempt filled late
            oid = None
            try:
                if target == 0:
                    o = self.c.close_position(symbol)
                else:
                    o = self.c.submit_order(MarketOrderRequest(symbol=symbol, qty=h[0] - target, side=OrderSide.SELL,
                                                               time_in_force=TimeInForce.DAY))
                oid = getattr(o, "id", None)
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {str(exc)[:120]}"
            deadline = time.time() + self.confirm_s
            done = False
            while time.time() < deadline:
                h = self.held(symbol)
                if (h is None and target == 0) or (h and h[0] <= target + 1e-9):
                    done = True; break
                if last_err and oid is None:
                    break                                      # rejected: don't wait, retry now
                time.sleep(self.poll_s)
            if done:
                px = self._fill_price(symbol, oid)
                if px is not None:
                    return px
                return fallback_price() if callable(fallback_price) else fallback_price
            try:
                self.cancel_open(symbol)                       # a stuck pending order blocks the next try
            except Exception:
                pass
            time.sleep(min(2.0, self.poll_s * 2))
        else:
            h = self.held(symbol)
            if not ((h is None and target == 0) or (h and h[0] <= target + 1e-9)):
                notify_once(f"sell-{symbol}", f"Sell not confirmed: {symbol}",
                       f"Tried {tries} times to sell {symbol}; Alpaca still shows {h[0] if h else 0:g}. "
                       f"The bot keeps retrying and the risk guard closes any option still open at 3:50 NY."
                       + (f" Last error: {last_err}" if last_err else ""))
                return None
        px = self._fill_price(symbol)
        if px is not None:
            return px
        return fallback_price() if callable(fallback_price) else fallback_price
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
