"""Run: .venv/bin/python live/tests/test_sizing.py  (no network)"""
import os, sys, json, tempfile, csv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
base = tempfile.mkdtemp(); os.environ["PAPER_STATE"] = base
import sizing as S
fails = []
def check(n, c): print(("PASS " if c else "FAIL ") + n); c or fails.append(n)
plan = lambda **k: {"trades": 50, "used": 0.01, "ladder": "normal", **k}
check("under 30 trades: exactly 1 contract", S.contracts("x", 2.0, 10000, 5, plan={"trades": 12, "used": 0.05})[0] == 1)
check("paper only: 0 contracts", S.contracts("x", 2.0, 10000, 5, plan=plan(ladder="paper only"))[0] == 0)
check("1% of $50k on a $2 option = 2 contracts", S.contracts("x", 2.0, 50000, 9, plan=plan())[0] == 2)
check("never more than the paper bot took", S.contracts("x", 2.0, 10_000_000, 3, plan=plan())[0] == 3)
check("2% cap wins over a big Kelly", S.contracts("x", 1.0, 100000, 99, plan=plan(used=0.08))[0] == 20)
check("half size halves", S.contracts("x", 2.0, 100000, 99, plan=plan(ladder="half size"))[0] == 2)
check("wild day (vol 2x) halves", S.contracts("x", 2.0, 100000, 99, plan=plan(), vol_ratio=2.0)[0] == 2)
check("calm day never sizes up", S.contracts("x", 2.0, 100000, 99, plan=plan(), vol_ratio=0.5)[0] == 5)
check("never below 1 while allowed", S.contracts("x", 9.0, 1000, 5, plan=plan())[0] == 1)
check("no edge -> Kelly 0", S.kelly(S.np.array([-0.5, 0.2, -0.4, 0.1])) == 0.0)
k = S.cautious_kelly([0.6, -0.5, 0.8, -0.5, 0.5, -0.5] * 10)
check("cautious Kelly below full Kelly", 0 < k["cautious"] <= k["full"] and abs(k["used"] - 0.25 * k["cautious"]) < 1e-3)
d = os.path.join(base, "options-qqq"); os.makedirs(d + "/search")
json.dump({"promoted": "2026-11-01"}, open(d + "/search/champion.json", "w"))
with open(d + "/options_trades.csv", "w") as fh:
    w = csv.DictWriter(fh, fieldnames=["date", "opened", "pl_pct"]); w.writeheader()
    w.writerow({"date": "2026-10-30", "opened": "2026-10-30T10:00", "pl_pct": "0.9"}); w.writerow({"date": "2026-11-02", "opened": "2026-11-02T10:00", "pl_pct": "-0.4"})
check("only trades since the champion started count", S.returns("options-qqq") == [-0.4])
S.main(); check("nightly report written", "options-qqq" in json.load(open(base + "/risk/sizing.json")))
print(f"\n{len(fails)} failure(s)"); sys.exit(1 if fails else 0)
