"""Portfolio research: grade every allocation variation, pick the champion for the portfolio bot.

  1. Daily prices for the universe (yfinance, adjusted, from 2005), cached.
  2. Every variation is backtested on the SEARCH period (start of data to 3 years ago).
  3. The top 5 by growth (drawdown no worse than 35%) get one look at the LOCKED last 3 years.
  4. Champion = best locked-period growth whose drawdown isn't much worse than SPY's.
Results (all variations, curves, benchmarks) -> STATE_DIR/research.json; champion -> champion.json.

  .venv/bin/python live/portfolio_search.py            (weekly timer, or the app's button)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import portfolio as P  # noqa: E402

NY = ZoneInfo("America/New_York")
STATE_DIR = Path(os.environ.get("STATE_DIR", Path.home() / "projects" / "paper-state" / "portfolio"))
DATA = Path(os.environ.get("PORTFOLIO_DATA", Path.home() / "projects" / "data" / "portfolio"))
HOLDOUT_YEARS = 3
MAX_DD = -0.35
STATUS = {"name": "Portfolio research", "phase": "starting", "index": 0, "total": None, "results": []}


def write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(data)); tmp.replace(path)


def status(**kw) -> None:
    STATUS.update(kw); STATUS["updated"] = dt.datetime.now(NY).isoformat(timespec="seconds")
    write(STATE_DIR / "research.json", STATUS)


def prices(refresh_days: float = 0.5) -> pd.DataFrame:
    path = DATA / "prices.csv"
    if path.exists() and time.time() - path.stat().st_mtime < refresh_days * 86400:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf
    raw = yf.download(list(P.UNIVERSE), start="2005-01-01", auto_adjust=True, progress=False, threads=True)["Close"]
    raw = raw.dropna(how="all").sort_index()
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    DATA.mkdir(parents=True, exist_ok=True); raw.to_csv(path)
    return raw


def main() -> int:
    STATUS["started"] = dt.datetime.now(NY).isoformat(timespec="seconds")
    status(phase="downloading prices")
    try:
        px = prices()
        px = px[[c for c in P.UNIVERSE if c in px.columns]]
        start = int(np.argmax(px[["SPY", "IEF", "SHY"]].notna().all(axis=1).to_numpy())) + 260
        split = len(px) - HOLDOUT_YEARS * P.TD
        var = P.variations()
        status(phase="testing variations", total=len(var), search_period=[str(px.index[start].date()), str(px.index[split].date())],
               holdout_period=[str(px.index[split].date()), str(px.index[-1].date())])
        bench = {"SPY": P.benchmark(px, start, split, {"SPY": 1.0}), "60/40": P.benchmark(px, start, split, {"SPY": 0.6, "IEF": 0.4})}
        rows, t0 = [], time.time()
        for k, p in enumerate(var, 1):
            try:
                r = P.backtest(px, p, start, split)
            except Exception as exc:
                r = {"cagr": None, "error": str(exc)}
            rows.append({"params": p, "desc": P.describe(p), "search": r})
            STATUS["results"] = sorted([{"desc": x["desc"], "cagr": x["search"].get("cagr"), "max_dd": x["search"].get("max_dd"),
                                         "sharpe": x["search"].get("sharpe")} for x in rows if x["search"].get("cagr") is not None],
                                       key=lambda x: -x["cagr"])[:12]
            rate = (time.time() - t0) / k
            status(index=k, symbol=P.describe(p), seconds_left=round(rate * (len(var) - k)))
        ok = [x for x in rows if x["search"].get("cagr") is not None and x["search"]["max_dd"] >= MAX_DD]
        top = sorted(ok, key=lambda x: -x["search"]["cagr"])[:5]
        status(phase="checking the top 5 on the locked last 3 years", seconds_left=None)
        hb = {"SPY": P.benchmark(px, split, len(px), {"SPY": 1.0}), "60/40": P.benchmark(px, split, len(px), {"SPY": 0.6, "IEF": 0.4})}
        for x in top:
            x["holdout"] = P.backtest(px, x["params"], split, len(px))
        allowed = [x for x in top if x["holdout"]["max_dd"] >= min(hb["SPY"]["max_dd"] * 1.25, -0.25)]
        champ = max(allowed or top, key=lambda x: x["holdout"]["cagr"]) if top else None
        weights = P.target_weights(px, champ["params"]).round(4).to_dict() if champ else {}
        result = {
            "phase": "done", "name": "Portfolio research", "updated": dt.datetime.now(NY).isoformat(timespec="seconds"),
            "started": STATUS["started"], "search_period": STATUS["search_period"], "holdout_period": STATUS["holdout_period"],
            "variations": len(var), "benchmarks_search": {k: {kk: vv for kk, vv in v.items()} for k, v in bench.items()},
            "benchmarks_holdout": hb,
            "top": [{"desc": x["desc"], "search": x["search"], "holdout": x["holdout"]} for x in top],
            "ranking": [{"desc": x["desc"], "cagr": x["search"]["cagr"], "max_dd": x["search"]["max_dd"], "sharpe": x["search"]["sharpe"]}
                        for x in sorted(ok, key=lambda x: -x["search"]["cagr"])[:20]],
            "champion": {"desc": champ["desc"], "params": champ["params"], "search": champ["search"], "holdout": champ["holdout"],
                         "weights_now": weights} if champ else None,
        }
        write(STATE_DIR / "research.json", result)
        if champ:
            write(STATE_DIR / "champion.json", {"desc": champ["desc"], "params": champ["params"], "chosen": result["updated"],
                                                "holdout_cagr": champ["holdout"]["cagr"], "spy_holdout_cagr": hb["SPY"]["cagr"]})
            print(f"champion: {champ['desc']} · locked 3y CAGR {champ['holdout']['cagr']:+.1%} (SPY {hb['SPY']['cagr']:+.1%}) · "
                  f"max drawdown {champ['holdout']['max_dd']:.0%} (SPY {hb['SPY']['max_dd']:.0%})")
        return 0
    except Exception as exc:
        status(phase="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
