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
    # signal context (meta-labeling): which setup fired, its direction, and its shape
    "price_vs_ema50", "ema50_slope", "or_size_pct", "or_pos",
    "sig_vwap", "sig_ema50", "sig_orb", "sig_fvg", "signal_dir",
]
SIGNALS = ["sig_vwap", "sig_ema50", "sig_orb", "sig_fvg"]
SIGNAL_NAMES = {"sig_vwap": "VWAP cross", "sig_ema50": "50 EMA cross",
                "sig_orb": "opening-range breakout", "sig_fvg": "FVG retest"}
OPENING_RANGE_MIN = 15


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
    add_signals(df)
    return df


def _cross(a: pd.Series, b: pd.Series) -> pd.Series:
    """+1 when a crosses above b on this bar, -1 when it crosses below, else 0."""
    above, was_above = a > b, a.shift(1) > b.shift(1)
    return (above & ~was_above).astype(int) - (~above & was_above).astype(int)


def add_signals(df: pd.DataFrame) -> None:
    """The four setups. Each column is +1 (bullish), -1 (bearish) or 0 on every bar."""
    c, h, lo = df["close"], df["high"], df["low"]
    day = df["date"]
    since_open = df["minutes_since_open"]

    ema50 = _ema(c, 50)
    df["price_vs_ema50"] = c / ema50 - 1
    df["ema50_slope"] = (ema50 / ema50.shift(10) - 1) * 1e4        # bp per 10 minutes

    # 1. VWAP cross and 2. price crossing the 50 EMA.
    df["sig_vwap"] = _cross(c, df["vwap"]).where(since_open >= 1, 0)
    df["sig_ema50"] = _cross(c, ema50)

    # 3. Opening-range breakout: range = first 15 minutes; first close beyond it, once per side per day.
    in_or = since_open < OPENING_RANGE_MIN
    or_hi = h.where(in_or).groupby(day).transform("max")
    or_lo = lo.where(in_or).groupby(day).transform("min")
    df["or_size_pct"] = (or_hi - or_lo) / c
    df["or_pos"] = (c - or_lo) / (or_hi - or_lo).replace(0, np.nan)
    after = since_open >= OPENING_RANGE_MIN
    up = (after & (c > or_hi)).astype(int)
    dn = (after & (c < or_lo)).astype(int)
    first_up = (up == 1) & (up.groupby(day).cumsum() == 1)
    first_dn = (dn == 1) & (dn.groupby(day).cumsum() == 1)
    df["sig_orb"] = first_up.astype(int) - first_dn.astype(int)

    # 4. FVG retest: a fair value gap forms, then price comes back into it and holds.
    #    Bullish gap = this bar's low above the high 2 bars back; signal when a later bar
    #    dips into the gap but closes above its bottom (within 30 minutes). Mirror for bearish.
    sig = np.zeros(len(df), dtype=int)
    hv, lv, cv = h.to_numpy(), lo.to_numpy(), c.to_numpy()
    min_gap = cv * 0.0002
    for _, idx in df.groupby(day).indices.items():
        idx = np.sort(idx)
        bull = bear = None                        # (gap_bottom, gap_top, formed_at)
        for k in range(2, len(idx)):
            i, i2 = idx[k], idx[k - 2]
            if bull and k - bull[2] > 30:
                bull = None
            if bear and k - bear[2] > 30:
                bear = None
            if bull and k > bull[2] and lv[i] <= bull[1] and cv[i] > bull[0]:
                sig[i], bull = 1, None
            elif bear and k > bear[2] and hv[i] >= bear[0] and cv[i] < bear[1]:
                sig[i], bear = -1, None
            if lv[i] - hv[i2] > min_gap[i]:
                bull = (hv[i2], lv[i], k)
            if lv[i2] - hv[i] > min_gap[i]:
                bear = (hv[i], lv[i2], k)
    df["sig_fvg"] = sig

    total = df[SIGNALS].sum(axis=1)
    df["signal_dir"] = np.sign(total).astype(int)   # conflicting signals cancel out


def resample(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Turn 1-minute bars into N-minute bars (within each session, labeled by bar start)."""
    if minutes <= 1:
        return bars
    df = bars.set_index("ts")
    out = df.groupby(df.index.date).resample(f"{minutes}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    out = out.dropna(subset=["close"]).reset_index(level=0, drop=True).reset_index()
    return out.rename(columns={"index": "ts"})


def direction_from(row_or_df, allowed: list[str]):
    """Signal direction using only the chosen setups (+1, -1, 0; conflicts cancel)."""
    total = sum(row_or_df[sg] for sg in allowed)
    return np.sign(total)


def fired_signals(row) -> list[str]:
    return [SIGNAL_NAMES[sg] for sg in SIGNALS if row[sg] != 0]


def feature_row(df: pd.DataFrame, i: int, names=None) -> np.ndarray:
    return df.loc[i, names or FEATURES].to_numpy(dtype=float)


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
