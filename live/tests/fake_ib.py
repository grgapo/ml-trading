"""A fake ib_async.IB for tests: qualify (standard class only), positions, market orders that fill (or not)."""
import itertools
from types import SimpleNamespace


class FakeIB:
    def __init__(self, fill=True, price=2.0, fill_after=0):
        self.fill, self.price, self.fill_after = fill, price, fill_after
        self.pos, self.trades, self.fills, self.ids, self.connected, self.polls = {}, [], [], itertools.count(100), True, 0

    def isConnected(self): return self.connected
    def connect(self, *a, **k): self.connected = True
    def sleep(self, s):
        import time; time.sleep(s); self._process()

    def qualifyContracts(self, c):
        if c.tradingClass != c.symbol:
            return [None]
        c.conId = abs(hash((c.symbol, c.lastTradeDateOrContractMonth, c.strike, c.right))) % 10**9 or 1
        return [c]

    def reqPositions(self):
        self._process()
        return [SimpleNamespace(contract=SimpleNamespace(conId=k, secType="OPT"), position=v[0], avgCost=v[1] * 100) for k, v in self.pos.items() if v[0]]

    def placeOrder(self, c, o):
        t = SimpleNamespace(contract=c, order=o, status="Submitted", age=0); self.trades.append(t); return t

    def openTrades(self): return [t for t in self.trades if t.status == "Submitted"]
    def cancelOrder(self, o):
        for t in self.trades:
            if t.order is o and t.status == "Submitted": t.status = "Cancelled"

    def reqExecutions(self): return self.fills

    def _process(self):
        for t in self.trades:
            if t.status != "Submitted": continue
            t.age += 1
            if t.age <= self.fill_after: continue
            if t.order.action == "BUY" and not self.fill: continue
            q = float(t.order.totalQuantity); k = t.contract.conId; p = self.pos.setdefault(k, [0.0, 0.0])
            if t.order.action == "BUY":
                p[1] = (p[0] * p[1] + q * self.price) / (p[0] + q); p[0] += q
            else:
                p[0] -= min(q, p[0])
            t.status = "Filled"
            self.fills.append(SimpleNamespace(contract=t.contract, execution=SimpleNamespace(side="BOT" if t.order.action == "BUY" else "SLD", price=self.price)))
