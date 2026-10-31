"""Real historical option prices (Alpaca minute bars) for the backtests.

For each trading day and underlying, downloads the minute bars of the near-the-money calls and
puts expiring that day (0DTE) and the next session (1DTE), for every $1 strike across the day's
price range, in one request, and caches them under data/options/<SYM>/<YYYY-MM-DD>.csv.gz.

The simulator then trades the REAL contract at real prices instead of a Black-Scholes estimate.
Bars come from actual trades; minutes with no trade carry the last price forward.
"""
from __future__ import annotations

import datetime as dt
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from options_pretrain import DATA_DIR, NY

OPT_DIR = DATA_DIR.parent / "options"
FIRST_DAY = dt.date(2024, 2, 1)            # Alpaca's option history starts around here


def occ(sym: str, expiry: dt.date, call: bool, strike: float) -> str:
    return f"{sym}{expiry:%y%m%d}{'C' if call else 'P'}{int(round(strike * 1000)):08d}"


def _client():
    from alpaca.data.historical.option import OptionHistoricalDataClient
    return OptionHistoricalDataClient(os.environ["ALPACA_API_KEY_ID"], os.environ["ALPACA_API_SECRET_KEY"])


def fetch_day(sym: str, day: dt.date, next_day: dt.date, lo: float, hi: float, client=None) -> pd.DataFrame:
    """Download (or load from cache) one day of near-the-money option minute bars."""
    path = OPT_DIR / sym / f"{day}.csv.gz"
    if path.exists():
        return pd.read_csv(path)
    if day < FIRST_DAY:
        return pd.DataFrame(columns=["contract", "minute", "close", "volume"])
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame
    client = client or _client()
    strikes = range(math.floor(lo * 0.995) - 1, math.ceil(hi * 1.005) + 2)
    contracts = [occ(sym, e, c, k) for e in (day, next_day) for c in (True, False) for k in strikes]
    rows = []
    start = dt.datetime.combine(day, dt.time(9, 30), NY); end = dt.datetime.combine(day, dt.time(16, 0), NY)
    for i in range(0, len(contracts), 100):
        res = client.get_option_bars(OptionBarsRequest(symbol_or_symbols=contracts[i:i + 100],
                                                       timeframe=TimeFrame.Minute, start=start, end=end))
        for c, bars in res.data.items():
            for b in bars:
                ts = pd.Timestamp(b.timestamp).tz_convert(NY)
                rows.append((c, ts.hour * 60 + ts.minute - 570, float(b.close), float(b.volume)))
    df = pd.DataFrame(rows, columns=["contract", "minute", "close", "volume"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def day_series(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """contract -> 390 minute prices (last trade carried forward; NaN before the first trade)."""
    out = {}
    for c, g in df.groupby("contract"):
        a = np.full(390, np.nan)
        m = g["minute"].to_numpy(int); ok = (m >= 0) & (m < 390)
        a[m[ok]] = g["close"].to_numpy(float)[ok]
        out[c] = pd.Series(a).ffill().to_numpy()
    return out


def simulate_real(series: np.ndarray | None, m: int, hold: int, sl: float, tp: float, flatten: int,
                  spread: float) -> tuple[float, int, float] | None:
    """Buy at the next minute's real price (+half spread), exit on the bot's rules at real prices
    (-half spread). Returns (return, exit minute, entry price) or None if the contract didn't trade."""
    if series is None or m + 1 >= 390:
        return None
    p0 = series[m + 1]
    if not np.isfinite(p0) or p0 < 0.05:
        return None
    entry = p0 * (1 + spread / 2)
    end = min(389, m + 1 + hold, flatten)
    for j in range(m + 2, end + 1):
        p = series[j]
        if not np.isfinite(p):
            continue
        r = p * (1 - spread / 2) / entry - 1
        if r <= sl or r >= tp or j >= end:
            return float(max(-1.0, min(1.5, r))), j, entry
    p = series[end] if np.isfinite(series[end]) else p0
    return float(max(-1.0, min(1.5, p * (1 - spread / 2) / entry - 1))), end, entry
