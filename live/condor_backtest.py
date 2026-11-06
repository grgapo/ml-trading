"""Iron condor backtest: does SELLING a range (short call + short put, protected by long wings) pay on quiet days?

Real option prices (Alpaca minute bars), SPY / QQQ / IWM, same-day expiry. Only prices backed by a real trade within
the last 2 minutes on ALL FOUR legs are used, at entry and at exit (no stale prices). Selling gets the price minus half
the spread, buying the price plus half. Max loss per condor = wing width - credit; sized to risk ~$1,000.
Grid: entry time x how far out the short strikes sit x wing width x take-profit (share of the credit kept) x stop
(loss as a multiple of the credit) x filter (none / options rich vs recent movement / VIX above usual / no big gap).
Settings are chosen on the first 60% of days and only then scored on the last 40% ("UNSEEN"). Research only.

  .venv/bin/python live/condor_backtest.py --symbols SPY QQQ IWM
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
import option_history as OH  # noqa: E402
import orb_backtest as ORB  # noqa: E402
from options_pretrain import DATA_DIR, SPREAD  # noqa: E402

OUT = Path.home() / "projects" / "paper-state" / "search" / "condor_report.json"
RISK = 1000.0
FRESH = int(os.environ.get("CONDOR_FRESH_MIN", "2"))
ENTRIES = {"10:00": 30, "11:00": 90, "12:00": 150, "13:00": 210}
DIST = (0.004, 0.006, 0.008)          # short strikes this far from the price
WIDTH = (1, 2)                        # wing width in dollars
TP = (0.25, 0.5, 0.75)                # buy back when this share of the credit is kept
SL = (1.0, 2.0, None)                 # stop when the loss reaches this multiple of the credit
FLAT = ORB.FLAT


def day_legs(sym):
    days = ORB.load_days(sym); out = []
    ivf = DATA_DIR.parent / "iv" / f"{sym}.csv"
    iv = dict(zip(*(lambda t: (t["date"].astype(str), t["iv_rv_10"]))(pd.read_csv(ivf)))) if ivf.exists() else {}
    for d in days:
        if d["date"] < OH.FIRST_DAY:
            continue
        fp = OH.OPT_DIR / sym / f"{d['date']}.csv.gz"
        if not fp.exists():
            continue
        try:
            raw = pd.read_csv(fp); ser = OH.day_series(raw)
        except Exception:
            continue
        age = {}
        for c, g in raw.groupby("contract"):
            last = np.full(390, np.nan); mm = g["minute"].to_numpy(int); ok = (mm >= 0) & (mm < 390); last[mm[ok]] = mm[ok]
            age[c] = np.arange(390) - pd.Series(last).ffill().to_numpy()
        rec = {"date": str(d["date"]), "gap": abs(d["o"][0] / d["prev_close"] - 1), "iv_rv": iv.get(str(d["date"])), "set": {}}
        for (label, m), dist, w in itertools.product(ENTRIES.items(), DIST, WIDTH):
            s = float(d["c"][m - 1])
            kc, kp = int(np.ceil(s * (1 + dist))), int(np.floor(s * (1 - dist)))
            legs = [(True, kc, -1), (True, kc + w, +1), (False, kp, -1), (False, kp - w, +1)]
            names = [OH.occ(sym, d["date"], call, k) for call, k, _ in legs]
            if any(n not in ser for n in names):
                continue
            fresh = np.all([age[n] <= FRESH for n in names], axis=0)
            if not fresh[m]:
                continue
            px = [ser[n] for n in names]
            # credit received: sell shorts at price*(1-s/2), buy wings at price*(1+s/2)
            credit = sum((p[m] * (1 - SPREAD / 2) if sgn < 0 else -p[m] * (1 + SPREAD / 2)) for p, (_, _, sgn) in zip(px, legs))
            if credit <= 0.05:
                continue
            # cost to close each minute: buy back shorts at price*(1+s/2), sell wings at price*(1-s/2)
            close = sum((p * (1 + SPREAD / 2) if sgn < 0 else -p * (1 - SPREAD / 2)) for p, (_, _, sgn) in zip(px, legs))
            close = np.where(fresh, close, np.nan)
            rec["set"][(label, dist, w)] = (float(credit), close[m + 1:FLAT + 1], float(w))
        if rec["set"]:
            out.append(rec)
    return out


def trade(credit, close, w, tp, sl):
    path = close[np.isfinite(close)]
    if not len(path):
        return None
    for c in path:
        pnl = credit - c
        if pnl >= tp * credit or (sl is not None and pnl <= -sl * credit):
            return pnl
    return credit - path[-1]


def run(recs, label, dist, w, tp, sl, filt, vix):
    out = []
    for r in recs:
        leg = r["set"].get((label, dist, w))
        if not leg:
            continue
        if filt == "rich" and not (r["iv_rv"] is not None and r["iv_rv"] == r["iv_rv"] and r["iv_rv"] > 1.1):
            continue
        if filt == "vix_high" and not (vix.get(r["date"], 0) > 0):
            continue
        if filt == "no_gap" and r["gap"] > 0.004:
            continue
        pnl = trade(leg[0], leg[1], leg[2], tp, sl)
        if pnl is None:
            continue
        max_loss = (leg[2] - leg[0]) * 100
        n = int(RISK // max_loss) if max_loss > 0 else 0
        if n >= 1:
            out.append(n * pnl * 100)
    a = np.array(out)
    return {"trades": int(len(a)), "total": round(float(a.sum()), 2) if len(a) else 0.0,
            "per_trade": round(float(a.mean()), 2) if len(a) else 0.0, "win_rate": round(float((a > 0).mean()), 3) if len(a) else 0.0,
            "worst": round(float(a.min()), 2) if len(a) else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "IWM"]); a = ap.parse_args()
    v = features.vix_daily(); vix = {}
    if v is not None:
        med = v.rolling(252, min_periods=120).median()
        vix = {str(dd.date()): float(x / m - 1) for dd, x, m in zip(v.index, v.shift(1), med.shift(1)) if m == m and x == x}
    rep = {"spread": SPREAD, "fresh_min": FRESH, "symbols": {}}
    for sym in a.symbols:
        recs = sorted(day_legs(sym), key=lambda r: r["date"])
        if len(recs) < 80:
            print(f"{sym}: only {len(recs)} usable days, skipped"); continue
        cut = int(len(recs) * 0.6); search, test = recs[:cut], recs[cut:]
        grid = []
        for label, dist, w, tp, sl, filt in itertools.product(ENTRIES, DIST, WIDTH, TP, SL, ("none", "rich", "vix_high", "no_gap")):
            s = run(search, label, dist, w, tp, sl, filt, vix)
            if s["trades"] >= 25:
                grid.append(({"entry": label, "dist": dist, "width": w, "tp": tp, "sl": sl, "filter": filt}, s))
        prof = sum(1 for _, s in grid if s["total"] > 0)
        rows = []
        for p, s in sorted(grid, key=lambda g: g[1]["total"], reverse=True)[:5]:
            rows.append({"params": p, "search": s, "test": run(test, p["entry"], p["dist"], p["width"], p["tp"], p["sl"], p["filter"], vix)})
        rep["symbols"][sym] = {"days": len(recs), "combos": len(grid), "profitable_share": round(prof / max(1, len(grid)), 3), "top": rows}
        print(f"\n{sym}: {len(recs)} usable days ({cut} to choose, {len(test)} unseen) · {len(grid)} settings · {prof / max(1, len(grid)):.0%} profitable on search days")
        for r in rows:
            p, s, t = r["params"], r["search"], r["test"]
            print(f"  {p['entry']} short ±{p['dist']:.1%} wings ${p['width']} keep {p['tp']:.0%} stop {('none' if p['sl'] is None else f'{p['sl']:g}x')} {p['filter']:8s} | "
                  f"search {s['trades']} tr ${s['total']:+,.0f} | UNSEEN {t['trades']} tr ${t['total']:+,.0f} (${t['per_trade']:+.0f}/tr, {t['win_rate']:.0%} won, worst ${t['worst']:,.0f})")
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(rep, default=str))
    print(f"\nreport: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
