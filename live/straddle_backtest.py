"""Straddle backtest: does buying a call AND a put together pay on big moves (either direction) after costs?

Real option prices (Alpaca minute bars, option_history), SPY / QQQ / IWM, every day with cached option data.
A straddle = 1 ATM call + 1 ATM put, same expiry (0DTE or next session), bought at the next minute's price +
half the spread, sold at the minute's price - half the spread. ~$1,000 per straddle.
Grid: entry time x expiry x profit target x stop x max hold x filter (none / VIX above usual / options cheap vs
recent movement / big opening gap). Honest test: settings are chosen on the first 60% of days ("search") and only
then scored on the last 40% they never saw ("test"). Research only; changes nothing the bots use.

  .venv/bin/python live/straddle_backtest.py --symbols SPY QQQ IWM
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
from options_pretrain import DATA_DIR, NY, SPREAD  # noqa: E402

OUT = Path.home() / "projects" / "paper-state" / "search" / "straddle_report.json"
STAKE = 1000.0
FRESH = int(os.environ.get("STRADDLE_FRESH_MIN", "0"))      # >0: both legs must have traded within this many minutes
ENTRIES = {"9:45": 15, "10:00": 30, "10:30": 60, "11:30": 120}
TPS, SLS, HOLDS = (0.2, 0.4, 0.6, 1.0), (-0.2, -0.3, -0.5, None), (60, 120, 240)
FLAT = ORB.FLAT


def paths(sym: str):
    """day -> {(entry, dte): (cost per straddle, value path from entry+1 to the close) } using real prices."""
    days = ORB.load_days(sym); sessions = [d["date"] for d in days]; out = {}
    iv = {}
    f = DATA_DIR.parent / "iv" / f"{sym}.csv"
    if f.exists():
        t = pd.read_csv(f); iv = dict(zip(t["date"].astype(str), t["iv_rv_10"]))
    for i, d in enumerate(days):
        if d["date"] < OH.FIRST_DAY or i + 1 >= len(days):
            continue
        fp = OH.OPT_DIR / sym / f"{d['date']}.csv.gz"
        if not fp.exists():
            continue
        try:
            raw = pd.read_csv(fp); ser = OH.day_series(raw)
        except Exception:
            continue
        age = {}                                                # contract -> minutes since its last real trade, each minute
        if FRESH:
            for cname, g in raw.groupby("contract"):
                last = np.full(390, np.nan); mm = g["minute"].to_numpy(int); ok = (mm >= 0) & (mm < 390)
                last[mm[ok]] = mm[ok]
                last = pd.Series(last).ffill().to_numpy(); age[cname] = np.arange(390) - last
        gap = abs(d["o"][0] / d["prev_close"] - 1)
        rec = {"date": str(d["date"]), "gap": gap, "iv_rv": iv.get(str(d["date"])), "legs": {}}
        for (label, m), (dte, expiry) in itertools.product(ENTRIES.items(), (("0dte", d["date"]), ("1dte", sessions[i + 1]))):
            k = round(float(d["c"][m - 1]))
            c, p = ser.get(OH.occ(sym, expiry, True, k)), ser.get(OH.occ(sym, expiry, False, k))
            if c is None or p is None or not (np.isfinite(c[m]) and np.isfinite(p[m])):
                continue
            v = c + p
            if not np.isfinite(v[m]) or v[m] < 0.2:
                continue
            if FRESH:                                           # only prices backed by a recent real trade on BOTH legs
                ac, ap = age.get(OH.occ(sym, expiry, True, k)), age.get(OH.occ(sym, expiry, False, k))
                fresh = (ac <= FRESH) & (ap <= FRESH)
                if not fresh[m]:
                    continue
                v = np.where(fresh, v, np.nan)                  # a stale minute can't be used to exit either
            rec["legs"][(label, dte)] = (float(v[m] * (1 + SPREAD / 2)), v[m + 1:FLAT + 1] * (1 - SPREAD / 2))
        if rec["legs"]:
            out[rec["date"]] = rec
    return out


def trade(cost, path, tp, sl, hold):
    end = min(len(path), hold) if hold else len(path)
    for j in range(end):
        v = path[j]
        if not np.isfinite(v):
            continue
        r = v / cost - 1
        if r >= tp or (sl is not None and r <= sl):
            return r
    v = path[end - 1] if end else np.nan
    return (v / cost - 1) if np.isfinite(v) else None


def run(recs, entry, dte, tp, sl, hold, filt, vix):
    pnl = []
    for r in recs:
        leg = r["legs"].get((entry, dte))
        if not leg:
            continue
        if filt == "vix_high" and not (vix.get(r["date"], 0) > 0):
            continue
        if filt == "cheap" and not (r["iv_rv"] is not None and r["iv_rv"] == r["iv_rv"] and r["iv_rv"] < 1.0):
            continue
        if filt == "gap" and r["gap"] < 0.004:
            continue
        ret = trade(leg[0], leg[1], tp, sl, hold)
        if ret is None:
            continue
        n = int(STAKE // (leg[0] * 100))
        if n >= 1:
            pnl.append(n * leg[0] * 100 * ret)
    a = np.array(pnl)
    return {"trades": int(len(a)), "total": round(float(a.sum()), 2) if len(a) else 0.0,
            "per_trade": round(float(a.mean()), 2) if len(a) else 0.0, "win_rate": round(float((a > 0).mean()), 3) if len(a) else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "IWM"]); a = ap.parse_args()
    v = features.vix_daily()
    vix_rel = {}
    if v is not None:
        med = v.rolling(252, min_periods=120).median()
        vix_rel = {str(d.date()): float(x / m - 1) for d, x, m in zip(v.index, v.shift(1), med.shift(1)) if m == m and x == x}
    report = {"spread": SPREAD, "stake": STAKE, "symbols": {}}
    for sym in a.symbols:
        recs = sorted(paths(sym).values(), key=lambda r: r["date"])
        if len(recs) < 80:
            print(f"{sym}: only {len(recs)} days with option data, skipped"); continue
        cut = int(len(recs) * 0.6); search, test = recs[:cut], recs[cut:]
        grid = []
        for entry, dte, tp, sl, hold, filt in itertools.product(ENTRIES, ("0dte", "1dte"), TPS, SLS, HOLDS, ("none", "vix_high", "cheap", "gap")):
            s = run(search, entry, dte, tp, sl, hold, filt, vix_rel)
            if s["trades"] >= 25:
                grid.append(({"entry": entry, "dte": dte, "tp": tp, "sl": sl, "hold": hold, "filter": filt}, s))
        profitable = sum(1 for _, s in grid if s["total"] > 0)
        top = sorted(grid, key=lambda g: g[1]["total"], reverse=True)[:5]
        rows = []
        for p, s in top:
            t = run(test, p["entry"], p["dte"], p["tp"], p["sl"], p["hold"], p["filter"], vix_rel)
            rows.append({"params": p, "search": s, "test": t})
        report["symbols"][sym] = {"days": len(recs), "search_days": cut, "test_days": len(test),
                                  "combos": len(grid), "profitable_share": round(profitable / max(1, len(grid)), 3), "top": rows}
        print(f"\n{sym}: {len(recs)} days ({cut} to choose settings, {len(test)} unseen) · {len(grid)} settings tried · "
              f"{profitable / max(1, len(grid)):.0%} profitable on the search days")
        for r in rows:
            p, s, t = r["params"], r["search"], r["test"]
            print(f"  {p['entry']} {p['dte']} tp+{p['tp']:.0%} sl{('' if p['sl'] is None else f'{p['sl']:.0%}') or ' none'} hold {p['hold']}m {p['filter']:8s} | "
                  f"search {s['trades']} tr ${s['total']:+,.0f} (${s['per_trade']:+.0f}/tr) | UNSEEN {t['trades']} tr ${t['total']:+,.0f} (${t['per_trade']:+.0f}/tr, {t['win_rate']:.0%} won)")
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, default=str))
    print(f"\nreport: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
