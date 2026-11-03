"""Run: .venv/bin/python live/tests/test_smoke.py   (no network)"""
import os, sys, tempfile, json, datetime as dt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
base = tempfile.mkdtemp(); os.environ["RISK_DIR"] = os.path.join(base, "risk"); os.environ["PAPER_STATE"] = base
os.environ.pop("NTFY_TOPIC", None)
os.environ.update(RISK_SMOKE_CONFIRM_S="1", RISK_SMOKE_RETRY_S="0", RISK_SMOKE_TRIES="2")
import broker as B, risk_guard as G
from fake_alpaca import FakeWithAccount
B.RISK.mkdir(parents=True, exist_ok=True)
failures = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); failures.append(name) if not cond else None
T0 = dt.datetime(2026, 11, 3, 9, 35, tzinfo=B.NY)

# Nov 3: the order never fills
c = FakeWithAccount(fill="none")
check("an unfilled order test reports failure", G.smoke(c, T0) is False)
check("...but pauses nothing (alert-only)", B.halted("all")[0] is False and B.halted("options-spy")[0] is False)
check("...leaves no open order", not c.get_orders(type("R", (), {"status": "open", "symbols": None, "limit": None})()))
check("...and retried once", json.loads((B.RISK / "smoke.json").read_text())["attempts"] == 2)

# normal day
c = FakeWithAccount()
check("a working order path passes", G.smoke(c, T0) is True)
check("...and leaves no SGOV behind", B.Broker(c).held("SGOV") is None)
print(f"\n{len(failures)} failure(s)"); sys.exit(1 if failures else 0)
