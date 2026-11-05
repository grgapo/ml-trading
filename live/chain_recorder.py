"""Option-chain recorder: builds our own free history of full option chains (no vendor has this for free).

Weekdays at 10:00, 12:00 and 15:45 New York, for SPY, QQQ and IWM: every contract expiring within 45 days with a
strike within 10% of the price: bid / ask / sizes, last trade, implied volatility and greeks (Alpaca's free
indicative feed). Saved as ~/projects/data/chains/<SYM>/<YYYY-MM-DD>_<HHMM>.csv.gz (~1-2 MB a day in total).
Read-only for the bots. One summary line per run; an alert only if a whole run fails.

  .venv/bin/python live/chain_recorder.py [--label manual]
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
OUT = Path(os.environ.get("CHAINS_DIR", Path.home() / "projects" / "data" / "chains"))
SYMBOLS = os.environ.get("CHAIN_SYMBOLS", "SPY,QQQ,IWM").split(",")
DAYS_OUT, WIDTH = 45, 0.10


def spot(sym: str) -> float:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    c = StockHistoricalDataClient(os.environ["ALPACA_API_KEY_ID"], os.environ["ALPACA_API_SECRET_KEY"])
    return float(c.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=sym, feed="iex"))[sym].price)


def record(sym: str, now: dt.datetime, label: str) -> int:
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionChainRequest
    c = OptionHistoricalDataClient(os.environ["ALPACA_API_KEY_ID"], os.environ["ALPACA_API_SECRET_KEY"])
    s = spot(sym)
    chain = c.get_option_chain(OptionChainRequest(underlying_symbol=sym, expiration_date_lte=now.date() + dt.timedelta(days=DAYS_OUT),
                                                  strike_price_gte=round(s * (1 - WIDTH), 2), strike_price_lte=round(s * (1 + WIDTH), 2)))
    rows = []
    for occ, snap in chain.items():
        q, t, g = snap.latest_quote, snap.latest_trade, snap.greeks
        rows.append({"contract": occ, "spot": s,
                     "bid": getattr(q, "bid_price", None), "ask": getattr(q, "ask_price", None),
                     "bid_size": getattr(q, "bid_size", None), "ask_size": getattr(q, "ask_size", None),
                     "quote_time": getattr(q, "timestamp", None), "last": getattr(t, "price", None), "last_time": getattr(t, "timestamp", None),
                     "iv": snap.implied_volatility, "delta": getattr(g, "delta", None), "gamma": getattr(g, "gamma", None),
                     "theta": getattr(g, "theta", None), "vega": getattr(g, "vega", None)})
    if not rows:
        return 0
    d = OUT / sym; d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / f"{now:%Y-%m-%d}_{label}.csv.gz", index=False)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--label"); a = ap.parse_args()
    now = dt.datetime.now(NY); label = a.label or f"{now:%H%M}"
    done, failed = [], []
    for sym in SYMBOLS:
        try:
            done.append(f"{sym} {record(sym.strip(), now, label)}")
        except Exception as exc:
            failed.append(f"{sym}: {type(exc).__name__}: {str(exc)[:120]}")
    print(f"{now:%Y-%m-%d %H:%M} chains saved: " + (", ".join(done) or "none") + (" · failed: " + " | ".join(failed) if failed else ""))
    topic = os.environ.get("NTFY_TOPIC")
    if failed and not done and topic:
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{topic}", data="; ".join(failed).encode(),
                                   headers={"Title": "option-chain recorder failed", "Tags": "warning"}), timeout=10)
        except Exception:
            pass
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
