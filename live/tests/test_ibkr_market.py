"""Run: .venv/bin/python live/tests/test_ibkr_market.py   (no network) — the options bot on IBKR (fake IBKR + quotes)."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["RISK_DIR"] = tempfile.mkdtemp(); os.environ["STATE_DIR"] = os.path.join(tempfile.mkdtemp(), "options-qqq-ibkr")
import options_bot as O, broker_ibkr as BI
from fake_ib import FakeIB
failures = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); failures.append(name) if not cond else None
OCC = "QQQ261106P00690000"
m = object.__new__(O.IBKRMarket); f = FakeIB(price=1.50)
m.ibb = BI.IBBroker(21, ib=f, confirm_s=0.3, poll_s=0.01, tag=O.BOT_ID); quotes = {}
m.bid = lambda c: quotes.get(c)
check("the bot's id is options-qqq-ibkr (its orders and claims are labelled)", O.BOT_ID == "options-qqq-ibkr")
check("before buying: no position", m.position(OCC) is None)
m.buy(OCC, 1); m.ibb.sleep(0.02)
p = m.position(OCC)
check("after the fill: position at IBKR with its entry price", p and p["qty"] == 1.0 and abs(p["entry"] - 1.5) < 1e-9)
check("no quote this minute -> the trade stays steady (not mistaken for unfilled)", p["plpc"] == 0.0)
quotes[OCC] = 2.25
check("with a quote: P&L from the bid (+50%)", abs(m.position(OCC)["plpc"] - 0.5) < 1e-9)
f.price = 2.25
check("close is confirmed by the position at IBKR", m.close(OCC) is True and m.position(OCC) is None)
check("orders were tagged with the bot", all(t.order.orderRef.startswith("options-qqq-ibkr-") for t in f.trades))
print(f"\n{len(failures)} failure(s)"); sys.exit(1 if failures else 0)
