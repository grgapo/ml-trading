"""Run: .venv/bin/python live/tests/test_risk_guard.py   (no network)"""
import os, sys, tempfile, json, datetime as dt, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
base = tempfile.mkdtemp(); os.environ["RISK_DIR"] = os.path.join(base, "risk"); os.environ["PAPER_STATE"] = base
os.environ.pop("NTFY_TOPIC", None)
import broker as B, risk_guard as G
from fake_alpaca import FakeWithAccount
failures = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); failures.append(name) if not cond else None
def fresh():
    shutil.rmtree(B.RISK, ignore_errors=True); B.RISK.mkdir(parents=True)
T0 = dt.datetime(2026, 11, 3, 10, 0, tzinfo=B.NY)
fast = lambda c: setattr(B.Broker, "__init__", lambda self, client, confirm_s=1, poll_s=0.02: (setattr(self, "c", client), setattr(self, "confirm_s", 1), setattr(self, "poll_s", 0.02)) and None)
fast(None)

# Nov 2 replay: the proving ground bought, but claimed nothing
fresh(); c = FakeWithAccount(); c.hold("SPY261103C00758000", 10, 4.65, upl=-300)
G.run(c, T0)
check("an unclaimed option is NOT closed within the grace period", "SPY261103C00758000" in c.positions and c.positions["SPY261103C00758000"][0] == 10)
G.run(c, T0 + dt.timedelta(minutes=4))
check("after 3+ min unclaimed it is closed", c.positions.get("SPY261103C00758000", [0])[0] == 0)
check("...and everything is paused for the day", B.halted("options-spy")[0] and B.halted("proving")[0])
st = json.loads((B.RISK / "status.json").read_text())
check("...and the event is recorded for the app", any("unclaimed" in e or "no bot claimed" in e for e in st["events"]))

# a claimed position is left alone
fresh(); c = FakeWithAccount(); c.hold("QQQ261103C00697000", 8, 5.69, upl=-50)
B.write_claims("options-qqq", {"QQQ261103C00697000": 8})
G.run(c, T0); G.run(c, T0 + dt.timedelta(minutes=10))
check("a claimed position is left alone", c.positions["QQQ261103C00697000"][0] == 8 and not B.halted("options-qqq")[0])

# experiment daily loss limit ($1,000)
fresh(); c = FakeWithAccount(); c.hold("SPY261103C00758000", 10, 4.65, upl=-1200)
B.write_claims("proving", {"SPY261103C00758000": 10})
G.run(c, T0)
check("experiment over its $1,000 limit: its position is closed", c.positions.get("SPY261103C00758000", [0])[0] == 0)
check("...the proving ground is paused", B.halted("proving")[0] is True)
check("...but the options bots are not", B.halted("options-spy")[0] is False)

# options bot: more leeway ($2,500)
fresh(); c = FakeWithAccount(); c.hold("SPY261103C00760000", 10, 4.0, upl=-1800)
B.write_claims("options-spy", {"SPY261103C00760000": 10})
G.run(c, T0)
check("options bot at -$1,800 (under its $2,500 limit) keeps trading", c.positions["SPY261103C00760000"][0] == 10 and not B.halted("options-spy")[0])
c.upl["SPY261103C00760000"] = -2600; G.run(c, T0 + dt.timedelta(minutes=1))
check("options bot past -$2,500: closed and paused", c.positions["SPY261103C00760000"][0] == 0 and B.halted("options-spy")[0])

# realized losses count too
fresh(); c = FakeWithAccount()
log = B.RISK.parent / "proving" / "trades.csv"; log.parent.mkdir(parents=True, exist_ok=True)
log.write_text("date,experiment,pnl\n2026-11-03,orb-calm,-700\n2026-11-03,orb-calm,-400\n2026-11-02,orb-calm,-5000\n")
G.run(c, T0)
check("realized losses today (-$1,100) trip the experiment limit; yesterday's don't count", B.halted("proving")[0] is True)
log.unlink()

# shadow trades (no orders) never count toward the real loss limit
fresh(); c = FakeWithAccount()
log = B.RISK.parent / "proving" / "trades.csv"; log.parent.mkdir(parents=True, exist_ok=True)
log.write_text("date,experiment,pnl\n2026-11-03,orb-calm-iwm-shadow,-3000\n2026-11-03,orb-calm,-200\n")
G.run(c, T0)
check("shadow losses don't trip the experiments' real loss limit", B.halted("proving")[0] is False)
log.unlink()

# whole account -3%
fresh(); c = FakeWithAccount(equity=96500, last_equity=100000); c.hold("QQQ261103C00697000", 8, 5.69, upl=-100)
B.write_claims("options-qqq", {"QQQ261103C00697000": 8}); c.hold("USO", 94, 158.0)
G.run(c, T0)
check("account down 3.5%: options closed", c.positions["QQQ261103C00697000"][0] == 0)
check("...portfolio ETFs kept", c.positions["USO"][0] == 94)
check("...everything paused", B.halted("options-qqq")[0] and B.halted("proving")[0])

# ghost claim
fresh(); c = FakeWithAccount(); B.write_claims("options-spy", {"SPY261103C00999000": 5})
G.run(c, T0); G.run(c, T0 + dt.timedelta(minutes=6))
st = json.loads((B.RISK / "status.json").read_text())
check("a claimed-but-missing position is reported", any("believes it holds" in e for e in st["events"]))

# morning order test
fresh(); c = FakeWithAccount(price=100.0)
check("order test passes on a working broker", G.smoke(c, T0) is True and not B.halted("all")[0])
fresh(); c = FakeWithAccount(fill="none")
check("order test failing pauses everything", G.smoke(c, T0) is False and B.halted("options-spy")[0])

print(f"\n{len(failures)} failure(s)"); sys.exit(1 if failures else 0)
