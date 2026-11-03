"""Run: .venv/bin/python live/tests/test_broker.py   (no network; exits non-zero if anything fails)"""
import os, sys, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["RISK_DIR"] = tempfile.mkdtemp()
import broker as B
from fake_alpaca import FakeTradingClient
from alpaca.trading.enums import OrderStatus

failures = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: failures.append(name)

# the exact Nov 2 trap: status text is "OrderStatus.FILLED"
check("status text really looks like 'OrderStatus.FILLED'", str(OrderStatus.FILLED) == "OrderStatus.FILLED")

c = FakeTradingClient(price=4.65, fill_delay=0.3); b = B.Broker(c, confirm_s=3, poll_s=0.05)
got = b.buy("SPY261103C00758000", 10, "test")
check("a delayed fill is confirmed by the position (price, qty)", got == (4.65, 10.0))
check("held() sees it", b.held("SPY261103C00758000") == (10.0, 4.65))
c.price = 6.0
check("sell is confirmed when the position is gone, at the fill price", b.sell("SPY261103C00758000") == 6.0)
check("nothing held afterwards", b.held("SPY261103C00758000") is None)

c = FakeTradingClient(fill="none"); b = B.Broker(c, confirm_s=0.5, poll_s=0.05)
check("an order that never fills returns None", b.buy("QQQ261103C00697000", 8, "test") is None)
check("...and is cancelled (no open orders left)", not c.get_orders(type("R", (), {"status": "open", "symbols": None, "limit": None})()))

c = FakeTradingClient(fill="partial"); b = B.Broker(c, confirm_s=0.5, poll_s=0.05)
got = b.buy("QQQ261103C00697000", 8, "test")
check("a partial fill reports the qty actually held", got is not None and got[1] == 4.0)

c = FakeTradingClient(); b = B.Broker(c, confirm_s=0.5, poll_s=0.05)
check("selling something not held returns None (no crash)", b.sell("SPY261103C00700000") is None)

# halts
check("not halted by default", B.halted("options-spy") == (False, ""))
B.set_halt("proving", "daily loss limit", until="2999-01-01")
check("a bot-specific halt pauses that bot", B.halted("proving")[0] is True)
check("...but not the others", B.halted("options-spy")[0] is False)
B.set_halt("all", "account down 3%", until="2999-01-01")
check("a global halt pauses everyone", B.halted("options-qqq")[0] is True)
B.clear_halt("all"); B.clear_halt("proving")
B.set_halt("all", "yesterday's limit", until="2000-01-01")
check("an expired halt no longer pauses", B.halted("options-qqq")[0] is False)
(B.RISK / "halt.json").write_text("{not json")
check("an unreadable halt file pauses (fail safe)", B.halted("options-spy")[0] is True)
B.write_claims("proving", {"SPY261103C00758000": 10, "X": 0})
check("claims are written (zero quantities dropped)", json.loads((B.RISK / "claims" / "proving.json").read_text())["holdings"] == {"SPY261103C00758000": 10.0})

(B.RISK / "size_cap.json").write_text(json.dumps({"proving": {"max_qty": 1, "until": "2999-01-01"}}))
check("a size cap applies to its bot", B.size_cap("proving") == 1 and B.size_cap("options-spy") is None)
(B.RISK / "size_cap.json").write_text(json.dumps({"proving": {"max_qty": 1, "until": "2000-01-01"}}))
check("an expired size cap no longer applies", B.size_cap("proving") is None)

print(f"\n{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
