"""Does morning implied volatility make the trend-day (calm-day) predictor better? A/B test, read-only.

Same walk-forward as trend_days.py (each month predicted only by models trained on earlier months), run twice:
  A) today's inputs (gap, first 30 minutes, VIX, events, ...)
  B) the same + 10:00 implied volatility (iv_build.py) and IV vs 20-day realized volatility
Reports AUC (0.5 = no skill) and how often the predictor's top/bottom 30% of days really were trend days.
Changes nothing the bots use.   .venv/bin/python live/iv_ab.py --symbols SPY QQQ IWM
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
import orb_backtest as ORB  # noqa: E402
import trend_days as TD  # noqa: E402
from options_pretrain import DATA_DIR  # noqa: E402


def add_iv(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sym, g in df.groupby("symbol"):
        f = DATA_DIR.parent / "iv" / f"{sym}.csv"
        g = g.copy()
        if f.exists():
            iv = pd.read_csv(f).set_index("date")
            key = g["date"].astype(str)
            g["iv10"] = key.map(iv["iv_10:00"]).astype(float)
            g["iv_rv10"] = key.map(iv["iv_rv_10"]).astype(float)
        out.append(g)
    return pd.concat(out).sort_values("date").reset_index(drop=True)


def score(df: pd.DataFrame, only_iv_days: bool) -> dict:
    te = TD.walk_forward(df.copy()).dropna(subset=["p_logit"])
    if only_iv_days and "iv10" in te:
        te = te[te["iv10"].notna()]
    out = {"test_days": len(te)}
    for name in ("p_logit", "p_trees"):
        top, low = te[te[name] >= te[name].quantile(0.7)], te[te[name] <= te[name].quantile(0.3)]
        out[name] = (TD.auc(te["trend"], te[name]), top["trend"].mean(), low["trend"].mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "IWM"]); a = ap.parse_args()
    vix = features.vix_daily()
    df = pd.concat([TD.build(s, ORB.load_days(s), vix) for s in a.symbols], ignore_index=True)
    df["is_iwm"] = (df["symbol"] == "IWM").astype(float)
    with_iv = add_iv(df)
    have = with_iv["iv10"].notna().mean() if "iv10" in with_iv else 0
    print(f"{len(df)} symbol-days · implied volatility available on {have:.0%} of them (option prices start Feb 2024)")
    # compare on the SAME days: only days that have IV, so the test is fair
    base = with_iv.drop(columns=["iv10", "iv_rv10"]) if "iv10" in with_iv else with_iv
    a_res, b_res = score(base.loc[with_iv["iv10"].notna()].reset_index(drop=True), False), score(with_iv.loc[with_iv["iv10"].notna()].reset_index(drop=True), False)
    print(f"\nwalk-forward on {b_res['test_days']} test days with option data:")
    for label, r in (("A  today's inputs     ", a_res), ("B  + implied volatility", b_res)):
        for name, nm in (("p_logit", "logistic"), ("p_trees", "trees")):
            auc, top, low = r[name]
            print(f"  {label} · {nm:8s}: AUC {auc:.3f} · trend days in its top 30%: {top:.0%} · in its bottom 30%: {low:.0%}")
    d = b_res["p_logit"][0] - a_res["p_logit"][0]
    print(f"\nverdict: implied volatility {'HELPS' if d >= 0.02 else 'adds little' if d > -0.005 else 'does not help'} "
          f"(logistic AUC {d:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
