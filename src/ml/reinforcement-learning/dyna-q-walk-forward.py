"""Walk-forward evaluation of the Dyna-Q agent (no changes to the agent itself).

Why: the cross-section script trains once on 2016-2024 and tests once on
2024-09 onward. One split can be lucky or unlucky, and the hyperparameters were
picked on that same test period. Walk-forward repeats the experiment many times
the way it would have run live:

  for each test year Y:  train on the TRAIN_YEARS years before Y  ->  trade year Y only

and reports the policy vs buy-and-hold for every (stock, year, seed), under
three accounting rules:
  * as-is      : same-close fills, no costs (the original assumption)
  * costs      : + COST_BPS per position change (commissions, spread, slippage)
  * realistic  : + costs, and the order fills at the NEXT close (a signal computed
                 from today's close can't trade at today's close)

  MPLBACKEND=Agg python src/ml/reinforcement-learning/dyna-q-walk-forward.py
  ... --symbols AAPL MSFT --years 2021 2022 --epochs 200 --seeds 1   # quick run

Output: figures/dyna-q-walk-forward.csv (one row per stock/year/seed) and a summary.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


xsec = _load("dyna_q_cross_section", "dyna-q-cross-section.py")
dyna_q = xsec.dyna_q
q_learning = dyna_q.q_learning
FLAT, LONG = q_learning.FLAT, q_learning.LONG

TRAIN_YEARS = 4
COST_BPS = 5.0
OUT = HERE / "figures" / "dyna-q-walk-forward.csv"


def run_policy(agent, start: int, end: int, cost_bps: float, delay: int) -> dict:
    """Greedy policy from bar `start` to `end` (inclusive). delay=1: the position chosen at
    close t is held from close t+1. Returns total growth, trades and time in the market."""
    px = agent.df["Adj Close"].to_numpy(float)
    position, held, equity, trades, in_market = FLAT, FLAT, 1.0, 0, 0
    pending = []
    for t in range(start, end):
        action = agent.predict(t, position)
        position = agent._update_state(position, action)
        pending.append(position)
        new_held = pending.pop(0) if len(pending) > delay else held
        if new_held != held:
            equity *= 1.0 - cost_bps / 10_000
            trades += 1
            held = new_held
        if held == LONG:
            equity *= px[t + 1] / px[t]
            in_market += 1
    if held == LONG:
        equity *= 1.0 - cost_bps / 10_000          # close out at the end of the year
    days = max(1, end - start)
    return {"growth": equity - 1.0, "trades": trades, "exposure": in_market / days}


def one_fold(prices: pd.DataFrame, year: int, seed: int, epochs: int) -> dict | None:
    dates = pd.to_datetime(prices["Date"])
    test = np.flatnonzero(dates.dt.year == year)
    first_train = pd.Timestamp(year - TRAIN_YEARS, 1, 1)
    window = prices[(dates >= first_train) & (dates.dt.year <= year)].reset_index(drop=True)
    if len(test) < 100 or (pd.to_datetime(window["Date"]).dt.year < year).sum() < 250 * (TRAIN_YEARS - 1):
        return None
    agent = dyna_q.DynaQ(window, alpha=xsec.ALPHA, planning_steps=xsec.PLANNING_STEPS, epochs=epochs)
    q_learning.CUTOFF = f"{year}-01-01"                    # train strictly before the test year
    agent.train(seed=seed)
    wdates = pd.to_datetime(agent.df["Date"])
    idx = np.flatnonzero(wdates.dt.year == year)
    start, end = int(idx[0]), int(idx[-1])
    if agent.market_bin(start) < 0:
        return None
    px = agent.df["Adj Close"].to_numpy(float)
    hold = px[end] / px[start] - 1.0
    row = {"year": year, "seed": seed, "buy_hold": hold}
    for name, cost, delay in (("as_is", 0.0, 0), ("costs", COST_BPS, 0), ("realistic", COST_BPS, 1)):
        r = run_policy(agent, start, end, cost, delay)
        row[f"{name}_growth"] = r["growth"]
        row[f"{name}_excess"] = r["growth"] - hold
        if name == "realistic":
            row["trades"], row["exposure"] = r["trades"], r["exposure"]
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", nargs="*", default=None, help="default: the cross-section's 19 stocks")
    ap.add_argument("--years", nargs="*", type=int, default=list(range(2020, pd.Timestamp.today().year)))
    ap.add_argument("--epochs", type=int, default=300, help="per fold (cross-section uses 1000)")
    ap.add_argument("--seeds", type=int, default=2, help="repeat each fold with different random seeds")
    args = ap.parse_args()

    symbols = args.symbols or [s for s in xsec.STARTING_STOCKS if s != "SPY"]
    rows, t0 = [], time.time()
    total = len(symbols) * len(args.years) * args.seeds
    done = 0
    for symbol in symbols:
        try:
            prices = xsec.download_and_clean(symbol)
        except Exception as exc:
            print(f"{symbol}: download failed ({exc})")
            continue
        for year in args.years:
            for seed in range(args.seeds):
                row = one_fold(prices.copy(), year, xsec.SEED + seed if hasattr(xsec, "SEED") else seed, args.epochs)
                done += 1
                if row:
                    rows.append({"symbol": symbol, **row})
                    print(f"[{done}/{total}] {symbol} {year} seed {seed}: realistic {row['realistic_growth']:+.1%} "
                          f"vs hold {row['buy_hold']:+.1%}  ({time.time() - t0:.0f}s)", flush=True)
    if not rows:
        print("no folds ran")
        return 1
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\n=== Walk-forward summary (each row: one stock, one test year, one seed) ===")
    print(f"folds: {len(df)}  stocks: {df.symbol.nunique()}  years: {sorted(df.year.unique())}  "
          f"train window: {TRAIN_YEARS}y  cost: {COST_BPS:g} bps per trade")
    for name in ("as_is", "costs", "realistic"):
        ex = df[f"{name}_excess"]
        print(f"{name:>10}: beat buy-and-hold in {(ex > 0).mean():.0%} of folds · "
              f"median excess {ex.median():+.1%} · mean excess {ex.mean():+.1%}")
    by_year = df.groupby("year")[["realistic_excess", "buy_hold"]].median()
    print("\nmedian per test year (realistic):")
    print(by_year.to_string(float_format=lambda v: f"{v:+.1%}"))
    print(f"\ntime in the market (realistic): {df.exposure.mean():.0%} · trades per year: {df.trades.mean():.0f}")
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
