"""Run: .venv/bin/python live/tests/test_moneyness.py   (no network) — the live bot buys the strike the search priced."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["RISK_DIR"] = tempfile.mkdtemp(); os.environ["STATE_DIR"] = os.path.join(tempfile.mkdtemp(), "options-exp")
import options_bot as O
failures = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); failures.append(name) if not cond else None
check("at the money (default): the price itself", O.strike_target(681.4, "call", {"moneyness": "atm"}) == 681.4)
check("no strategy (pre-trained model): the price itself", O.strike_target(681.4, "put", None) == 681.4)
check("$2 in the money call: 2 below the rounded price (search: round(close) - 2)", O.strike_target(681.4, "call", {"moneyness": "itm2"}) == 679)
check("$2 in the money put: 2 above the rounded price (search: round(close) + 2)", O.strike_target(681.6, "put", {"moneyness": "itm2"}) == 684)
sim = O.SimMarket.__new__(O.SimMarket)
import datetime as dt
check("the contract picked is that strike", sim.pick_contract("SPY", "call", dt.date(2026, 11, 9), O.strike_target(681.4, "call", {"moneyness": "itm2"})).endswith("C679"))
print(f"\n{len(failures)} failure(s)"); sys.exit(1 if failures else 0)
