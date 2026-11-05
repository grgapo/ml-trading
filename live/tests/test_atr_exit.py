"""Run: .venv/bin/python live/tests/test_atr_exit.py   (no network) — the live ATR exit matches the search's rule."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["RISK_DIR"] = tempfile.mkdtemp(); os.environ["STATE_DIR"] = os.path.join(tempfile.mkdtemp(), "options-exp")
import numpy as np, pandas as pd
import options_bot as O
failures = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); failures.append(name) if not cond else None
class M:
    def __init__(self, prices): self.p, self.i = prices, 0
    def bars(self, sym): return pd.DataFrame({"close": [self.p[self.i]]})
def run(prices, sgn=1, atr=1.0, entry=100.0):
    t = {"symbol": "SPY", "atr": atr, "u_signal": entry, "sgn": sgn, "best": 0.0}; m = M(prices)
    for i in range(len(prices)):
        m.i = i; r = O.atr_exit(t, m, False)
        if r: return r, i
    return None, None
check("call: up 4.5 ATR then back to +2.2 ATR (gave back half) -> atr giveback", run([101, 103, 104.5, 103, 102.2]) == ("atr giveback", 4))
check("call: never reaches 4 ATR -> holds", run([101, 102, 103, 102, 101.5]) == (None, None))
check("call: down 10 ATR -> atr stop", run([99, 95, 90]) == ("atr stop", 2))
check("put: down 5 ATR then back to -2.4 ATR -> atr giveback", run([99, 97, 95, 96.5, 97.6], sgn=-1) == ("atr giveback", 4))
check("end of day always exits", O.atr_exit({"symbol": "SPY", "atr": 1.0, "u_signal": 100, "sgn": 1}, M([100]), True) == "end of day")
check("no ATR recorded -> only the end-of-day exit", run([150, 50], atr=None) == (None, None))
# the same rule as the search's atr_exit_bar (options_pretrain) on a random path
try:
    from options_pretrain import atr_exit_bar, FLATTEN_MIN
    rng = np.random.default_rng(3); ok = True
    for k in range(200):
        path = 100 + np.cumsum(rng.normal(0, 0.6, 120)); path[0] = 100.0
        mins = np.arange(120) + 60; j = atr_exit_bar(path, mins, 0, 119, 1.0, 1, 1)
        r, i = run(list(path[1:]), entry=path[0])
        live = (i + 1) if r else 119
        ok &= (live == j) or (mins[j] + 1 - 1 >= FLATTEN_MIN)
    check("live exit lands on the same bar as the search's rule (200 random paths)", ok)
except ImportError as exc:
    print(f"(skipped search comparison: {exc})")
print(f"\n{len(failures)} failure(s)"); sys.exit(1 if failures else 0)
