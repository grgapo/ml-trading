"""Whole-market portfolio research: which allocation grows the money best, honestly?

Universe: US-listed ETFs (and a few stocks) covering US sectors, international markets,
bonds and commodities. Strategies are classic, well-studied allocators; each variation is
backtested on daily prices with monthly rebalancing and trading costs, graded on the older
history, then the best few get ONE look at the locked last 3 years. Benchmarks: SPY and 60/40.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

UNIVERSE = {
    # US equity
    "SPY": "US large caps", "QQQ": "US tech (Nasdaq-100)", "IWM": "US small caps",
    "XLK": "US tech sector", "XLF": "US financials", "XLE": "US energy", "XLV": "US health care",
    "XLI": "US industrials", "XLY": "US consumer discretionary", "XLP": "US consumer staples",
    "XLU": "US utilities", "XLB": "US materials", "SMH": "semiconductors",
    # International
    "EFA": "developed markets ex-US", "VWO": "emerging markets", "EWT": "Taiwan", "TSM": "Taiwan Semiconductor (ADR)",
    "EWJ": "Japan", "EWG": "Germany", "INDA": "India", "FXI": "China large caps", "EWZ": "Brazil",
    # Bonds (SHY doubles as 'cash')
    "SHY": "short Treasuries (cash-like)", "IEF": "7-10y Treasuries", "TLT": "20+y Treasuries",
    "LQD": "investment-grade corporate bonds", "HYG": "high-yield bonds", "TIP": "inflation-protected Treasuries",
    # Commodities
    "GLD": "gold", "SLV": "silver", "USO": "oil", "DBC": "broad commodities",
}
CLASS = {**{k: "US equity" for k in ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "SMH"]},
         **{k: "international" for k in ["EFA", "VWO", "EWT", "TSM", "EWJ", "EWG", "INDA", "FXI", "EWZ"]},
         **{k: "bonds" for k in ["SHY", "IEF", "TLT", "LQD", "HYG", "TIP"]},
         **{k: "commodities" for k in ["GLD", "SLV", "USO", "DBC"]}}
CASH = "SHY"
COST_BPS = 5.0
TD = 252


# ---- weights: each takes the price history UP TO the rebalance day ----------------------------
def _avail(px: pd.DataFrame, lookback: int) -> list[str]:
    return [c for c in px.columns if px[c].iloc[-lookback - 1:].notna().all()]


def _cap(w: pd.Series, cap: float) -> pd.Series:
    """Long-only, sum 1, no weight above cap (excess spread over the rest)."""
    w = w.clip(lower=0)
    if w.sum() <= 0:
        return w
    w = w / w.sum()
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum(); w[over] = cap
        free = ~over & (w > 0)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w


def _cov(rets: pd.DataFrame) -> np.ndarray:
    from sklearn.covariance import LedoitWolf
    return LedoitWolf().fit(rets.to_numpy()).covariance_ * TD


def equal_weight(px, p):
    a = _avail(px, p["lookback"]); return pd.Series(1 / len(a), index=a)


def inverse_vol(px, p):
    a = _avail(px, p["lookback"]); r = px[a].pct_change().iloc[-p["lookback"]:]
    return _cap(1 / r.std(), p["cap"])


def risk_parity(px, p):
    a = _avail(px, p["lookback"]); r = px[a].pct_change().iloc[-p["lookback"]:].dropna()
    S = _cov(r); w = np.ones(len(a)) / len(a)
    for _ in range(200):                                   # equal risk contribution, fixed-point iteration
        rc = w * (S @ w); target = rc.sum() / len(a)
        w = w * (target / np.maximum(rc, 1e-12)) ** 0.5; w /= w.sum()
    return _cap(pd.Series(w, index=a), p["cap"])


def momentum(px, p):
    """Own the top_k assets by trailing return (skipping the last month); with the trend filter,
    only assets above their 200-day average qualify; empty slots go to cash."""
    a = _avail(px, max(p["lookback"], 210))
    last = px[a].iloc[-1]
    mom = px[a].iloc[-22] / px[a].iloc[-p["lookback"]] - 1
    if p["trend"]:
        mom = mom[last > px[a].iloc[-200:].mean()]
    top = mom.sort_values(ascending=False).head(p["top_k"])
    top = top[top > 0] if p["trend"] else top
    w = pd.Series(1 / p["top_k"], index=top.index)
    if w.sum() < 1 - 1e-9:
        w[CASH] = w.get(CASH, 0) + 1 - w.sum()
    return w


def min_variance(px, p):
    a = _avail(px, p["lookback"]); r = px[a].pct_change().iloc[-p["lookback"]:].dropna()
    S = _cov(r); inv = np.linalg.pinv(S); w = inv @ np.ones(len(a)); w = np.maximum(w, 0)
    return _cap(pd.Series(w if w.sum() > 0 else np.ones(len(a)), index=a), p["cap"])


def growth_optimal(px, p):
    """Kelly-style: maximize expected log growth  mu.w - 1/2 w'Sw  (long-only, fully invested, capped),
    with mu from trailing returns shrunk halfway to the average (raw estimates are very noisy)."""
    from scipy.optimize import minimize
    a = _avail(px, p["lookback"]); r = px[a].pct_change().iloc[-p["lookback"]:].dropna()
    S = _cov(r); mu = r.mean().to_numpy() * TD; mu = 0.5 * mu + 0.5 * mu.mean()
    n = len(a); x0 = np.ones(n) / n
    res = minimize(lambda w: -(mu @ w - 0.5 * w @ S @ w), x0, method="SLSQP",
                   bounds=[(0, p["cap"])] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return _cap(pd.Series(np.maximum(res.x, 0), index=a), p["cap"])


STRATEGIES = {"equal weight": equal_weight, "inverse volatility": inverse_vol, "risk parity": risk_parity,
              "momentum": momentum, "minimum variance": min_variance, "growth-optimal": growth_optimal}


def target_weights(px: pd.DataFrame, p: dict) -> pd.Series:
    w = STRATEGIES[p["strategy"]](px, p)
    vt = p.get("vol_target")
    if vt:                                                   # scale risk to a target volatility, rest to cash
        names = [c for c in w.index if c in px.columns]
        r = px[names].pct_change().iloc[-63:].dropna()
        if len(r) > 20:
            port_vol = float(np.sqrt(w[names] @ _cov(r) @ w[names]))
            k = min(1.0, vt / port_vol) if port_vol > 0 else 1.0
            w = w * k; w[CASH] = w.get(CASH, 0) + (1 - k)
    w = w[w > 1e-4]
    return w / w.sum()


# ---- backtest ------------------------------------------------------------------------------------
def month_ends(idx: pd.DatetimeIndex) -> list[int]:
    s = pd.Series(range(len(idx)), index=idx)
    return list(s.groupby([idx.year, idx.month]).max().values)


def backtest(px: pd.DataFrame, p: dict, start: int, end: int) -> dict:
    """Rebalance at each month-end close from `start`; daily returns until `end` (exclusive)."""
    rets = px.pct_change().fillna(0).to_numpy()
    cols = list(px.columns); col = {c: i for i, c in enumerate(cols)}
    rebal = [i for i in month_ends(px.index) if start <= i < end - 1]
    w = np.zeros(len(cols)); equity = [1.0]; turnover = 0.0
    for k, i in enumerate(rebal):
        tw = target_weights(px.iloc[: i + 1], p)
        new = np.zeros(len(cols))
        for n, v in tw.items():
            new[col[n]] = v
        cost = np.abs(new - w).sum() * COST_BPS / 1e4; turnover += np.abs(new - w).sum()
        e = equity[-1] * (1 - cost)
        stop = rebal[k + 1] if k + 1 < len(rebal) else end - 1
        cur = new.copy()
        for d in range(i + 1, stop + 1):
            growth = cur * (1 + rets[d]); e *= growth.sum(); cur = growth / growth.sum()
            equity.append(e)
        w = cur
    return stats(np.array(equity), turnover / max(1, len(rebal) / 12))


def stats(eq: np.ndarray, turnover: float = 0.0) -> dict:
    if len(eq) < 30:
        return {"cagr": 0, "vol": 0, "sharpe": 0, "max_dd": 0, "growth": 1, "turnover": 0, "curve": []}
    r = np.diff(eq) / eq[:-1]; years = len(r) / TD
    peak = np.maximum.accumulate(eq)
    cagr = eq[-1] ** (1 / years) - 1; vol = r.std() * math.sqrt(TD)
    step = max(1, len(eq) // 120)
    return {"cagr": round(float(cagr), 4), "vol": round(float(vol), 4), "sharpe": round(float(cagr / vol), 3) if vol else 0,
            "max_dd": round(float((eq / peak - 1).min()), 4), "growth": round(float(eq[-1]), 4),
            "turnover": round(float(turnover), 3), "curve": [round(float(v), 4) for v in eq[::step]]}


def benchmark(px: pd.DataFrame, start: int, end: int, mix: dict) -> dict:
    p = {"strategy": "fixed", "mix": mix}
    rets = px.pct_change().fillna(0)
    rebal = set(month_ends(px.index))
    w = pd.Series(mix); e = [1.0]; cur = w.copy()
    for d in range(start + 1, end):
        g = cur * (1 + rets.iloc[d][cur.index]); e.append(e[-1] * g.sum()); cur = g / g.sum()
        if d in rebal:
            cur = w.copy()
    return stats(np.array(e))


def variations() -> list[dict]:
    out = []
    for lb, cap in itertools.product((63, 126, 252), (0.25, 0.4)):
        for s in ("inverse volatility", "risk parity", "minimum variance", "growth-optimal"):
            for vt in (None, 0.12):
                out.append({"strategy": s, "lookback": lb, "cap": cap, "vol_target": vt})
    for lb, k, tr in itertools.product((126, 252), (3, 5, 8), (True, False)):
        for vt in (None, 0.12):
            out.append({"strategy": "momentum", "lookback": lb, "top_k": k, "trend": tr, "cap": 1.0, "vol_target": vt})
    out.append({"strategy": "equal weight", "lookback": 63, "cap": 1.0, "vol_target": None})
    return out


def describe(p: dict) -> str:
    bits = [p["strategy"]]
    if p["strategy"] == "momentum":
        bits.append(f"top {p['top_k']} by {p['lookback'] // 21}-month return" + (" · trend filter" if p.get("trend") else ""))
    elif p["strategy"] != "equal weight":
        bits.append(f"{p['lookback'] // 21}-month window · max {int(p['cap'] * 100)}% per asset")
    if p.get("vol_target"):
        bits.append(f"volatility target {int(p['vol_target'] * 100)}%")
    return " · ".join(bits)
