"""IBKR execution for the lab's bots (through IB Gateway on lab03, paper account first).

Same rules as broker.py (learned on Nov 2): a fill is confirmed by the ACTUAL POSITION at the broker, an unconfirmed
buy is cancelled, a sell is confirmed when the position is gone and retried if not. Contracts are always the standard
option class (tradingClass = the symbol, never the adjusted "2QQQ"-style classes) and verified with IBKR before any
order. Market data and option prices still come from Alpaca; only orders and positions go through IBKR.
"""
from __future__ import annotations

import os
import re
import time

OCC = re.compile(r"^([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")
HOST, PORT = os.environ.get("IB_HOST", "127.0.0.1"), int(os.environ.get("IB_PORT", "4002"))


class IBBroker:
    def __init__(self, client_id: int, ib=None, confirm_s: float = 60, poll_s: float = 1.0, tag: str = "lab"):
        self.client_id, self._ib, self.confirm_s, self.poll_s, self.tag = client_id, ib, confirm_s, poll_s, tag
        self.contracts: dict[str, object] = {}

    # --- connection (reconnects by itself: IB Gateway restarts nightly) --------------------------------------
    @property
    def ib(self):
        if self._ib is None:
            from ib_async import IB
            self._ib = IB()
        if not self._ib.isConnected():
            self._ib.connect(HOST, PORT, clientId=self.client_id, timeout=15)
        return self._ib

    def sleep(self, s: float) -> None:
        self.ib.sleep(s)                                   # keeps IBKR's messages flowing while we wait

    # --- contracts -----------------------------------------------------------------------------------------
    def contract(self, occ: str):
        if occ in self.contracts:
            return self.contracts[occ]
        m = OCC.match(occ)
        if not m:
            raise ValueError(f"not an OCC option symbol: {occ}")
        sym, yy, mm, dd, right, k = m.groups()
        from ib_async import Option
        c = Option(sym, f"20{yy}{mm}{dd}", int(k) / 1000, right, "SMART", multiplier="100", currency="USD", tradingClass=sym)
        q = self.ib.qualifyContracts(c)
        if not q or q[0] is None or not getattr(q[0], "conId", 0):
            raise LookupError(f"IBKR doesn't know {occ} (standard class {sym})")
        self.contracts[occ] = q[0]
        return q[0]

    # --- positions -----------------------------------------------------------------------------------------
    def held(self, occ: str):
        """(qty, avg price per share) at IBKR, or None."""
        c = self.contract(occ)
        for p in self.ib.reqPositions():
            if getattr(p.contract, "conId", None) == c.conId and p.position:
                return float(p.position), float(p.avgCost) / 100.0
        return None

    def cancel_open(self, occ: str) -> None:
        c = self.contract(occ)
        for t in self.ib.openTrades():
            if getattr(t.contract, "conId", None) == c.conId:
                self.ib.cancelOrder(t.order)

    # --- orders ----------------------------------------------------------------------------------------------
    def submit_buy(self, occ: str, qty: int) -> None:
        from ib_async import MarketOrder
        o = MarketOrder("BUY", qty, tif="DAY", orderRef=f"{self.tag}-{occ}-buy-{int(time.time() * 1000)}")
        self.ib.placeOrder(self.contract(occ), o)

    def buy(self, occ: str, qty: int):
        """Market buy, confirmed by position. (avg price added, qty added) or None (cancelled, nothing bought)."""
        before = self.held(occ) or (0.0, 0.0)
        self.submit_buy(occ, qty)
        deadline = time.time() + self.confirm_s
        while time.time() < deadline:
            h = self.held(occ)
            if h and h[0] >= before[0] + qty - 1e-9:
                break
            self.sleep(self.poll_s)
        else:
            self.cancel_open(occ); self.sleep(min(2.0, self.poll_s * 2))
        h = self.held(occ)
        if not h or h[0] <= before[0] + 1e-9:
            return None
        added = h[0] - before[0]
        price = (h[0] * h[1] - before[0] * before[1]) / added if added else h[1]
        return round(price, 4), added

    def sell(self, occ: str, tries: int = 3):
        """Sell everything held; confirmed when the position is gone. Exit price (from IBKR's fills) or None."""
        from ib_async import MarketOrder
        for _ in range(tries):
            h = self.held(occ)
            if not h:
                return self._last_sell_price(occ)
            self.cancel_open(occ)
            self.ib.placeOrder(self.contract(occ), MarketOrder("SELL", h[0], tif="DAY", orderRef=f"{self.tag}-{occ}-sell"))
            deadline = time.time() + self.confirm_s
            while time.time() < deadline:
                if not self.held(occ):
                    return self._last_sell_price(occ)
                self.sleep(self.poll_s)
        return None

    def _last_sell_price(self, occ: str):
        c = self.contract(occ)
        fills = [f for f in self.ib.reqExecutions() if getattr(f.contract, "conId", None) == c.conId and f.execution.side == "SLD"]
        return float(fills[-1].execution.price) if fills else None

    def option_positions(self):
        return [(p.contract, float(p.position)) for p in self.ib.reqPositions()
                if getattr(p.contract, "secType", "") == "OPT" and p.position]
