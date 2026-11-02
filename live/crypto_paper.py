"""Crypto experiments for the proving ground: several trend variants + a just-holding benchmark,
forward-tested on live hourly Coinbase prices, each with its own $10,000 paper wallet split across
BTC/ETH/SOL. Runs every hour: refreshes the latest candles, then applies each strategy to the actual
prices since its start date, deciding each hour only from the hours before it (same code as the
crypto search, so live and backtest results compare directly). Fills at the next hour's open,
0.26% fee per trade, partial trades for core holdings / volatility sizing.

  .venv/bin/python live/crypto_paper.py           # hourly timer
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crypto_search as C  # noqa: E402

STATE = Path(os.environ.get("STATE_DIR", Path.home() / "projects" / "paper-state" / "crypto-proving"))
PRODUCTS = {"BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD", "SOL/USD": "SOL-USD"}
BASE = {"strategy": "LabRegime", "regime_hours": 720, "band": 0.01, "pair_filter": True, "core": 0.0,
        "fast_reentry": None, "vol_target": None}
DEFAULT = [
    {"id": "crypto-trend", "name": "Crypto · trend (baseline)", "params": BASE},
    {"id": "crypto-core50", "name": "Crypto · trend + 50% core", "params": {**BASE, "core": 0.5}},
    {"id": "crypto-fast", "name": "Crypto · trend, fast re-entry", "params": {**BASE, "fast_reentry": 168}},
    {"id": "crypto-volsize", "name": "Crypto · trend, volatility-sized", "params": {**BASE, "vol_target": 0.6}},
    {"id": "crypto-hold", "name": "Crypto · just holding (benchmark)", "params": {**BASE, "core": 1.0}},
]


def save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(data, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    tmp.replace(path)


def refresh_candles() -> None:
    """Fill in the latest hourly candles (Coinbase public API) in the same files the search uses."""
    end = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - dt.timedelta(hours=290)
    for pair, product in PRODUCTS.items():
        path = C.DATA / f"{pair.replace('/', '_')}-1h.feather"
        try:
            r = requests.get(f"https://api.exchange.coinbase.com/products/{product}/candles",
                             params={"granularity": 3600, "start": start.isoformat(), "end": end.isoformat()},
                             headers={"User-Agent": "homelab-crypto-paper"}, timeout=20)
            r.raise_for_status()
            new = pd.DataFrame([(pd.Timestamp(t, unit="s", tz="UTC"), op, hi, lo, cl, vol) for t, lo, hi, op, cl, vol in r.json()],
                               columns=["date", "open", "high", "low", "close", "volume"])
            new = new[new["date"] < end]                    # only completed hours
            old = pd.read_feather(path) if path.exists() else new.iloc[:0]
            old["date"] = pd.to_datetime(old["date"], utc=True)
            df = pd.concat([old, new]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = df[col].astype(float)
            df.to_feather(path)
        except Exception as exc:
            print(f"candle refresh failed for {pair}: {exc}", flush=True)
        time.sleep(0.3)


def main() -> int:
    reg_path = STATE / "experiments.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else []
    have = {e["id"] for e in reg}
    now = pd.Timestamp.now(tz="UTC").floor("h")
    for e in DEFAULT:
        if e["id"] not in have:
            reg.append({**e, "market": "crypto", "status": "incubating", "started": now.isoformat(),
                        "expectation": {"benchmark": "crypto-hold"}})
    save(reg_path, reg)
    refresh_candles()
    ind = C.indicators(C.load())
    out = []
    for e in reg:
        start = pd.Timestamp(e["started"])
        monthly, trades, daily, exposure = C.simulate_regime(ind, e["params"], start, None)
        run, curve = 0.0, []
        for d in sorted(daily):
            run += daily[d]; curve.append({"date": d, "day": round(daily[d], 2), "total": round(run, 2)})
        sells = [t for t in trades if t.get("kind") == "sell"]
        wins = [t["pnl"] for t in sells if t["pnl"] > 0]; losses = [t["pnl"] for t in sells if t["pnl"] <= 0]
        out.append({**e, "stats": {"trades": len(sells), "total": round(run, 2), "pct": round(run / C.WALLET, 4), "curve": curve,
                                   "win_rate": round(len(wins) / len(sells), 3) if sells else None,
                                   "profit_factor": round(sum(wins) / -sum(losses), 2) if losses and sum(losses) < 0 else None},
                    "today": {"exposure": {k: v["pct"] for k, v in exposure.items()}, "holdings": exposure},
                    "recent_trades": trades[-15:]})
    hold = next((x for x in out if x["id"] == "crypto-hold"), None)
    for x in out:
        if hold and x["id"] != "crypto-hold":
            x["stats"]["vs_hold"] = round(x["stats"]["total"] - hold["stats"]["total"], 2)
    last = ind["date"].iloc[-1]
    save(STATE / "status.json", {"updated": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                                 "prices_through": last.isoformat(), "experiments": out})
    print(f"prices through {last:%Y-%m-%d %H:%M} UTC · " + " · ".join(
        f"{x['name'].replace('Crypto · ', '')}: ${x['stats']['total']:+,.0f}" for x in out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
