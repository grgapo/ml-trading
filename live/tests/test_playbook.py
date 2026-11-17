"""Tests for live/playbook.py (no network)."""
import datetime as dt, os, sys, tempfile
from pathlib import Path
os.environ["PLAYBOOK_CACHE"] = str(Path(tempfile.mkdtemp()) / "today.json")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import playbook as PB
NY = PB.NY
fails = []
def check(name, c):
    print(("PASS " if c else "FAIL ") + name); c or fails.append(name)
past = [0.004, -0.003] * 10                               # usual first-hour move = 0.35%
check("big up first hour = strong up", PB.classify(past, 0.006)["type"] == "strong up")
check("big down first hour = strong down", PB.classify(past, -0.005)["type"] == "strong down")
check("small first hour = quiet", PB.classify(past, 0.001)["type"] == "quiet")
check("too little history = no usual size", PB.classify(past[:5], 0.01)["usual"] == 0)
class Fake:                                              # stands in for Alpaca: day type from canned moves
    def __init__(self, today_move): self.m = today_move
PB._first_hours = lambda stocks, today: (past, stocks.m)
d = dt.date(2026, 11, 18)
at = lambda h, m: dt.datetime.combine(d, dt.time(h, m), NY)
check("before 10:30: unchanged", PB.size(8, Fake(0.001), at(10, 5))[0] == 8)
q, why = PB.size(8, Fake(0.001), at(11, 0))
check("quiet day after 10:30: quarter size (8 -> 2)", q == 2 and "quiet" in why)
check("quiet day: never below 1 contract", PB.size(2, Fake(0.001), at(11, 5))[0] == 1)
Path(os.environ["PLAYBOOK_CACHE"]).unlink()
check("strong day after 10:30: full size", PB.size(8, Fake(-0.008), at(11, 0))[0] == 8)
check("day type is remembered for the rest of the day", PB.size(8, Fake(0.001), at(13, 0))[0] == 8)
Path(os.environ["PLAYBOOK_CACHE"]).unlink()
check("no data: full size (never blocks)", PB.size(8, None, at(11, 0))[0] == 8)
def boom(s, t): raise RuntimeError("feed down")
PB._first_hours = boom
check("data error: full size (never blocks)", PB.size(8, Fake(0.001), at(11, 0))[0] == 8)
check("zero contracts stay zero", PB.size(0, Fake(0.001), at(11, 0))[0] == 0)
print(f"{len(fails)} failure(s)"); sys.exit(1 if fails else 0)
