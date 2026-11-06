"""Run: .venv/bin/python live/tests/test_broker_ibkr.py   (no network, fake IBKR)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import broker_ibkr as BI
from fake_ib import FakeIB
failures = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); failures.append(name) if not cond else None
OCC = "QQQ261106C00700000"
f = FakeIB(price=1.45); b = BI.IBBroker(11, ib=f, confirm_s=0.3, poll_s=0.01, tag="options-qqq-ibkr")
c = b.contract(OCC)
check("OCC symbol -> standard IBKR option (QQQ class, 2026-11-06, 700 C)", (c.symbol, c.lastTradeDateOrContractMonth, c.strike, c.right, c.tradingClass) == ("QQQ", "20261106", 700.0, "C", "QQQ"))
check("buy confirmed by the position at IBKR (price, qty)", b.buy(OCC, 2) == (1.45, 2.0))
check("held() reads IBKR's position (avg cost per share)", b.held(OCC) == (2.0, 1.45))
check("every order carries the bot's tag", all(t.order.orderRef.startswith("options-qqq-ibkr-") for t in f.trades))
f.price = 2.1
check("sell confirmed when the position is gone, at IBKR's fill price", b.sell(OCC) == 2.1 and b.held(OCC) is None)
f2 = FakeIB(fill=False); b2 = BI.IBBroker(11, ib=f2, confirm_s=0.2, poll_s=0.01)
check("a buy that never fills returns None", b2.buy(OCC, 1) is None)
check("...and is cancelled (no open orders left)", not f2.openTrades())
f3 = FakeIB(price=1.0); b3 = BI.IBBroker(11, ib=f3, confirm_s=0.2, poll_s=0.01); b3.buy(OCC, 1)
f3.fill_after = 999                                     # the sell hangs
check("a sell that never confirms returns None after retries (bot keeps the trade)", b3.sell(OCC, tries=2) is None and b3.held(OCC) == (1.0, 1.0))
check("...and leaves no duplicate sells pending", len(f3.openTrades()) <= 1)
try:
    BI.IBBroker(11, ib=FakeIB()).contract("not-a-contract"); check("bad symbol rejected", False)
except ValueError:
    check("a malformed symbol is rejected before any order", True)
print(f"\n{len(failures)} failure(s)"); sys.exit(1 if failures else 0)
