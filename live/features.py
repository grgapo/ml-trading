"""Shared intraday indicators for the options bot and its pre-training.

Both the live bot and the historical simulator call the SAME functions, so the
model sees identical inputs in practice and in real trading. Every value at a
bar uses only that bar and earlier bars (no peeking ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "ret_5m", "ret_15m", "vwap_gap", "volume_ratio", "ema9_vs_21", "price_vs_ema21",
    "rsi_14", "macd_hist", "macd_slope", "atr_pct", "atr_regime", "fvg_balance",
    "range_pos", "day_gap", "minutes_since_open", "minutes_to_close", "is_qqq",
]


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def add_indicators(bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """bars: 1-minute regular-session bars with columns
    ts (tz-aware New York time), open, high, low, close, volume. Several days ok."""
    df = bars.sort_values("ts").reset_index(drop=True).copy()
    c, h, lo, v = df["close"], df["high"], df["low"], df["volume"].astype(float)
    day = df["ts"].dt.date
    df["date"] = day

    df["ret_5m"] = c / c.shift(5) - 1
    df["ret_15m"] = c / c.shift(15) - 1

    # VWAP resets every session.
    typical = (h + lo + c) / 3
    pv = (typical * v).groupby(day).cumsum()
    vv = v.groupby(day).cumsum().replace(0, np.nan)
    df["vwap"] = pv / vv
    df["vwap_gap"] = c / df["vwap"] - 1

    # Volume: last 5 minutes vs the typical 5 minutes over the past hour.
    v5 = v.rolling(5).sum()
    df["volume_ratio"] = v5 / (v.rolling(60).mean() * 5).replace(0, np.nan)

    ema9, ema21 = _ema(c, 9), _ema(c, 21)
    df["ema9_vs_21"] = ema9 / ema21 - 1
    df["price_vs_ema21"] = c / ema21 - 1

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    macd = _ema(c, 12) - _ema(c, 26)
    hist = macd - _ema(macd, 9)
    df["macd_hist"] = hist / c * 1e4                     # basis points of price
    df["macd_slope"] = (hist - hist.shift(3)) / c * 1e4

    prev_c = c.shift(1)
    tr = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["atr_pct"] = atr / c
    df["atr_regime"] = atr / atr.rolling(390, min_periods=60).median()

    # Fair value gaps (3-candle imbalance): bullish if this bar's low is above the
    # high two bars back, bearish if its high is below the low two bars back.
    min_gap = c * 0.0002
    bull = (lo - h.shift(2)) > min_gap
    bear = (lo.shift(2) - h) > min_gap
    df["fvg_balance"] = (bull.astype(int) - bear.astype(int)).rolling(15).sum()

    # Where price sits in today's range so far (0 = low of day, 1 = high of day).
    day_hi = h.groupby(day).cummax()
    day_lo = lo.groupby(day).cummin()
    df["range_pos"] = (c - day_lo) / (day_hi - day_lo).replace(0, np.nan)

    first_open = df.groupby(day)["open"].transform("first")
    prev_close = df.groupby(day)["close"].last().shift(1)
    df["day_gap"] = first_open / day.map(prev_close) - 1

    mins = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df["minutes_since_open"] = mins - (9 * 60 + 30)
    df["minutes_to_close"] = (16 * 60) - mins
    df["is_qqq"] = 1.0 if symbol.upper() == "QQQ" else 0.0
    return df


def feature_row(df: pd.DataFrame, i: int) -> np.ndarray:
    return df.loc[i, FEATURES].to_numpy(dtype=float)


def describe(row: dict | pd.Series) -> str:
    """Short human summary of what the bot is looking at (for the app)."""
    def pick(x, lo, hi, a, b, c):
        return a if x < lo else c if x > hi else b
    return " · ".join([
        "vwap " + pick(row["vwap_gap"], -0.0005, 0.0005, "below", "at", "above"),
        "ema " + pick(row["ema9_vs_21"], -0.0002, 0.0002, "bearish", "flat", "bullish"),
        f"rsi {row['rsi_14']:.0f}" if pd.notna(row["rsi_14"]) else "rsi –",
        "macd " + ("rising" if row["macd_slope"] > 0 else "falling"),
        "vol " + pick(row["volume_ratio"], 0.7, 1.5, "light", "normal", "heavy"),
        "fvg " + pick(row["fvg_balance"], -0.5, 0.5, "bear", "none", "bull"),
    ])
