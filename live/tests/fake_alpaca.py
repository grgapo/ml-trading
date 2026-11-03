"""A fake Alpaca TradingClient that behaves like the real one where it matters for the bots:
statuses are alpaca-py OrderStatus enums (str() gives "OrderStatus.FILLED"), get_open_position raises
when nothing is held, fills can be delayed / partial / never, and close_position sells everything."""
from __future__ import annotations

import itertools
import time
from types import SimpleNamespace

from alpaca.trading.enums import OrderSide, OrderStatus


class NoPosition(Exception):
    pass


class FakeTradingClient:
    def __init__(self, price=2.0, fill_delay=0.0, fill="full"):
        self.price, self.fill_delay, self.fill = price, fill_delay, fill   # fill: full | none | partial
        self.positions: dict[str, list] = {}       # symbol -> [qty, avg]
        self.orders: list[SimpleNamespace] = []
        self.ids = itertools.count(1)

    # positions
    def get_open_position(self, symbol):
        self._process()
        p = self.positions.get(symbol)
        if not p or p[0] == 0:
            raise NoPosition(f"position does not exist: {symbol}")
        return SimpleNamespace(symbol=symbol, qty=str(p[0]), avg_entry_price=str(p[1]), unrealized_pl="0", asset_class=("us_option" if len(symbol) > 10 else "us_equity"))

    def get_all_positions(self):
        self._process()
        return [self.get_open_position(s) for s, p in self.positions.items() if p[0]]

    # orders
    def submit_order(self, req):
        o = SimpleNamespace(id=next(self.ids), symbol=req.symbol, qty=float(req.qty), side=req.side, status=OrderStatus.NEW,
                            filled_avg_price=None, created=time.time())
        self.orders.append(o); return o

    def close_position(self, symbol):
        p = self.positions.get(symbol)
        if not p or not p[0]:
            raise NoPosition(symbol)
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import TimeInForce
        return self.submit_order(MarketOrderRequest(symbol=symbol, qty=p[0], side=OrderSide.SELL, time_in_force=TimeInForce.DAY))

    def get_orders(self, req):
        self._process()
        st = str(getattr(req.status, "value", req.status)).lower()
        want_open = st == "open"
        out = [o for o in self.orders if (o.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)) == want_open
               and (not req.symbols or o.symbol in req.symbols)]
        return list(reversed(out))[: (req.limit or 500)]

    def cancel_order_by_id(self, oid):
        for o in self.orders:
            if o.id == oid and o.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED):
                o.status = OrderStatus.CANCELED

    def get_order_by_id(self, oid):
        self._process()
        return next(o for o in self.orders if o.id == oid)

    def _process(self):
        for o in self.orders:
            if o.status is not OrderStatus.NEW or time.time() - o.created < self.fill_delay:
                continue
            if o.side == OrderSide.BUY and self.fill == "none":
                continue
            q = o.qty if not (o.side == OrderSide.BUY and self.fill == "partial") else max(1.0, o.qty // 2)
            p = self.positions.setdefault(o.symbol, [0.0, 0.0])
            if o.side == OrderSide.BUY:
                p[1] = (p[0] * p[1] + q * self.price) / (p[0] + q); p[0] += q
            else:
                p[0] -= min(q, p[0])
            o.filled_avg_price = str(self.price)
            o.status = OrderStatus.FILLED if q == o.qty else OrderStatus.PARTIALLY_FILLED
            if o.status is OrderStatus.PARTIALLY_FILLED:
                o.status = OrderStatus.CANCELED if False else OrderStatus.PARTIALLY_FILLED


class FakeWithAccount(FakeTradingClient):
    """Adds the account + clock the risk guard reads, and per-position unrealized P&L."""
    def __init__(self, equity=100000.0, last_equity=100000.0, **kw):
        super().__init__(**kw); self.equity, self.last_equity, self.upl = equity, last_equity, {}

    def get_account(self):
        return SimpleNamespace(equity=str(self.equity), last_equity=str(self.last_equity))

    def get_clock(self):
        return SimpleNamespace(is_open=True)

    def get_open_position(self, symbol):
        p = super().get_open_position(symbol); p.unrealized_pl = str(self.upl.get(symbol, 0.0)); return p

    def hold(self, symbol, qty, price, upl=0.0):
        self.positions[symbol] = [float(qty), float(price)]; self.upl[symbol] = upl
