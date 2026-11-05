"""Implied volatility from the option prices we already have (free): a first test of "options data" as a new input.

For every trading day since Feb 2024 and each symbol, from the cached real option minute bars:
  * ATM implied volatility of the next-session (1DTE) call and put at 10:00, 12:00 and 15:00
  * near skew: IV of a put ~$3 below spot minus IV of a call ~$3 above spot (fear priced in)
  * IV vs realized volatility of the previous 20 days (is the market expecting more movement than usual?)
Then the honest question before wiring it into the bots: does the 10:00 IV picture tell us anything about how
much the rest of the day moves (the thing the calm-day filter tries to predict)? Reported as quintiles.

Output: ~/projects/data/iv/<SYM>.csv and ~/projects/paper-state/search/iv_report.json. Read-only for the bots.
  .venv/bin/python live/iv_build.py --symbols SPY QQQ IWM
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import option_history as OH  # noqa: E402
from options_pretrain import DATA_DIR, NY, _bs  # noqa: E402

OUT_DIR = DATA_DIR.parent / "iv"
REPORT = Path.home() / "projects" / "paper-state" / "search" / "iv_report.json"
TIMES = {"10:00": 30, "12:00": 150, "15:00": 330}


def implied_vol(price, S, K, T, call):
    lo, hi = 0.01, 3.0
    intrinsic = max(0.0, S - K) if call else max(0.0, K - S)
    if not np.isfinite(price) or price <= intrinsic + 0.005:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if float(_bs(S, K, T, mid, call)) > price:
            hi = mid
        else:
            lo = mid
    v = (lo + hi) / 2
    return v if 0.02 < v < 2.9 else None


def years_left(day: dt.date, minute: int, expiry: dt.date) -> float:
    left_today = 390 - minute
    sessions_after = max(0, int(np.busday_count(day, expiry)))
    return (left_today + 390 * sessions_after) / (252 * 390)


def load_days(sym: str):
    files = sorted(glob.glob(str(DATA_DIR / sym / "*.csv.gz")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
    df["m"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute - 570
    df = df[(df["m"] >= 0) & (df["m"] < 390)].drop_duplicates("ts")
    out = {}
    for d, g in df.groupby(df["ts"].dt.date):
        a = np.full(390, np.nan); a[g["m"].to_numpy()] = g["close"].to_numpy(float)
        out[d] = pd.Series(a).ffill().bfill().to_numpy()
    return out


def build(sym: str) -> pd.DataFrame:
    days = load_days(sym); dates = sorted(days)
    rows = []
    for i, d in enumerate(dates):
        if d < OH.FIRST_DAY or i + 1 >= len(dates):
            continue
        f = OH.OPT_DIR / sym / f"{d}.csv.gz"
        if not f.exists():
            continue
        try:
            series = OH.day_series(pd.read_csv(f))
        except Exception:
            continue
        nxt = dates[i + 1]; px = days[d]
        rv = None
        if i >= 21:
            closes = np.array([days[dates[j]][-1] for j in range(i - 20, i)])
            rv = float(np.std(np.diff(np.log(closes))) * math.sqrt(252))
        row = {"date": str(d), "rv20": rv, "open": float(px[0]), "close": float(px[-1]),
               "move_after_10": float(abs(px[-1] / px[30] - 1)), "range_after_10": float((np.max(px[30:]) - np.min(px[30:])) / px[30])}
        for label, m in TIMES.items():
            S = float(px[m]); T = years_left(d, m, nxt); k = round(S)
            ivs = {}
            for name, K, call in (("call_atm", k, True), ("put_atm", k, False), ("put_dn3", k - 3, False), ("call_up3", k + 3, True)):
                s = series.get(OH.occ(sym, nxt, call, K))
                ivs[name] = implied_vol(float(s[m]), S, K, T, call) if s is not None and np.isfinite(s[m]) else None
            atm = [v for v in (ivs["call_atm"], ivs["put_atm"]) if v]
            row[f"iv_{label}"] = round(float(np.mean(atm)), 4) if atm else None
            row[f"skew_{label}"] = round(ivs["put_dn3"] - ivs["call_up3"], 4) if ivs["put_dn3"] and ivs["call_up3"] else None
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "iv_10:00" in df:
        df["iv_rv_10"] = df["iv_10:00"] / df["rv20"]
    return df


def quintiles(df: pd.DataFrame, col: str, target: str = "range_after_10") -> list[dict]:
    d = df[[col, target]].dropna()
    if len(d) < 50:
        return []
    d["q"] = pd.qcut(d[col], 5, labels=False, duplicates="drop")
    return [{"quintile": int(q) + 1, "feature_mid": round(float(g[col].median()), 4), "days": int(len(g)),
             "avg_range_after_10": round(float(g[target].mean()) * 100, 3)} for q, g in d.groupby("q")]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "IWM"]); a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True); rep = {"built": dt.datetime.now().isoformat(timespec="seconds"), "symbols": {}}
    for sym in a.symbols:
        try:
            df = build(sym)
        except Exception as exc:
            print(f"{sym}: {type(exc).__name__}: {exc}"); continue
        if df.empty:
            print(f"{sym}: no option days cached"); continue
        df.to_csv(OUT_DIR / f"{sym}.csv", index=False)
        r = {"days": len(df), "iv_coverage": round(float(df["iv_10:00"].notna().mean()), 3)}
        for col in ("iv_10:00", "iv_rv_10", "skew_10:00"):
            if col in df and df[col].notna().sum() > 50:
                d = df[[col, "range_after_10"]].dropna()
                r[col] = {"corr_with_rest_of_day_range": round(float(d[col].corr(d["range_after_10"], method="spearman")), 3),
                          "quintiles": quintiles(df, col)}
        rep["symbols"][sym] = r
        print(f"\n{sym}: {r['days']} days, IV found on {r['iv_coverage']:.0%} of them")
        for col in ("iv_10:00", "iv_rv_10", "skew_10:00"):
            if col in r:
                q = r[col]["quintiles"]
                print(f"  {col:11s} rank-corr with the rest of the day's range: {r[col]['corr_with_rest_of_day_range']:+.2f} · "
                      f"range by quintile (low->high): " + " / ".join(f"{x['avg_range_after_10']:.2f}%" for x in q))
    REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text(json.dumps(rep))
    print(f"\nreport: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
