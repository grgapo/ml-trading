"""Opening-range breakout + noise band + VWAP (one trade a day), backtested on REAL option prices.

Rules (from the strategy Greg found; gaps filled with standard choices, marked [assumed]):
  * opening range: high/low of 9:30-9:44
  * checks every 10 minutes from 10:00 to 14:00, on the last completed 1-minute candle
  * CALL when it closed above (range high + 0.3 x ATR), above the upper noise band and above VWAP;
    PUT when below (range low - 0.3 x ATR), below the lower noise band and below VWAP
      ATR = 14 periods of 5-minute bars [assumed: length]
      noise band = for this minute of the day, the average |price/open - 1| over the previous
      14 days, around max/min(today's open, yesterday's close) [assumed: Zarattini et al. definition]
  * option: about $2 in the money, next-session expiry (the post says 1-3 days), bought at the next
    minute's real price + half the spread; $5,000 per trade [assumed: size]
  * one trade a day at most; some days none
  * exit: once the underlying has moved 4 ATR its way, sell if it gives back half of that move;
    safety stop if it moves 10 ATR against; otherwise out at 3:45
Also: the same signals traded with plain shares (is the edge in the signal or the options?), and a
sensitivity grid around every setting (robust edge vs one lucky combination).

  .venv/bin/python live/orb_backtest.py --symbols SPY QQQ
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import option_history as OH  # noqa: E402
from options_pretrain import DATA_DIR, NY, SPREAD  # noqa: E402

STATE = Path(os.environ.get("STATE_DIR", Path.home() / "projects" / "paper-state" / "orb"))
STAKE = 5000.0
CHECKS = list(range(30, 271, 10))            # minutes after 9:30: 10:00, 10:10, ... 14:00
FLAT = 375                                   # 15:45


def load_days(symbol: str) -> list[dict]:
    files = sorted(glob.glob(str(DATA_DIR / symbol / "*.csv.gz")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
    df = df.drop_duplicates("ts").sort_values("ts")
    df["m"] = (df["ts"].dt.hour * 60 + df["ts"].dt.minute) - 570
    df = df[(df["m"] >= 0) & (df["m"] < 390)]
    days = []
    for day, g in df.groupby(df["ts"].dt.date):
        a = np.full((390, 5), np.nan)
        a[g["m"].to_numpy(), :] = g[["open", "high", "low", "close", "volume"]].to_numpy(float)
        a = pd.DataFrame(a).ffill().to_numpy()
        if np.isnan(a[0]).any() or np.isnan(a[389, 3]):
            continue
        days.append({"date": day, "o": a[:, 0], "h": a[:, 1], "l": a[:, 2], "c": a[:, 3], "v": np.nan_to_num(a[:, 4])})
    # 5-minute ATR(14), continuous across days; value known at each minute = last completed 5-min bar
    rows = []
    for d in days:
        for k in range(78):
            s = slice(5 * k, 5 * k + 5)
            rows.append((d["h"][s].max(), d["l"][s].min(), d["c"][5 * k + 4]))
    h5, l5, c5 = (np.array(x) for x in zip(*rows))
    prev = np.concatenate([[c5[0]], c5[:-1]])
    tr = np.maximum(h5 - l5, np.maximum(abs(h5 - prev), abs(l5 - prev)))
    atr5 = pd.Series(tr).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy()
    for i, d in enumerate(days):
        d["atr_bar"] = atr5[78 * i: 78 * i + 78]               # ATR at the end of each 5-min bar
        d["move"] = np.abs(d["c"] / d["o"][0] - 1)            # how far from the open, each minute
        d["prev_close"] = days[i - 1]["c"][389] if i else d["o"][0]
        typical = (d["h"] + d["l"] + d["c"]) / 3
        d["vwap"] = np.cumsum(typical * d["v"]) / np.maximum(np.cumsum(d["v"]), 1e-9)
        if i >= 14:
            d["sigma"] = np.mean([days[j]["move"] for j in range(i - 14, i)], axis=0)
    return [d for d in days if "sigma" in d]


def atr_at(d: dict, m: int) -> float:
    """ATR known at minute m: the last 5-minute bar that has fully closed by then."""
    k = m // 5 - 1
    return float(d["atr_bar"][max(k, 0)])


def signal(d: dict, band_atr: float):
    rh, rl = d["h"][:15].max(), d["l"][:15].min()
    up_anchor, dn_anchor = max(d["o"][0], d["prev_close"]), min(d["o"][0], d["prev_close"])
    for m in CHECKS:
        c, atr = d["c"][m - 1], atr_at(d, m)
        upper, lower = up_anchor * (1 + d["sigma"][m - 1]), dn_anchor * (1 - d["sigma"][m - 1])
        if c > rh + band_atr * atr and c > upper and c > d["vwap"][m - 1]:
            return m, "call", atr
        if c < rl - band_atr * atr and c < lower and c < d["vwap"][m - 1]:
            return m, "put", atr
    return None


def manage(d: dict, m: int, side: str, atr: float, trigger: float, giveback: float, stop: float):
    """Exit minute + reason, managed on the underlying in ATR units."""
    sgn = 1 if side == "call" else -1
    entry = d["c"][m]
    best = 0.0
    for j in range(m + 1, FLAT + 1):
        gain = sgn * (d["c"][j] - entry)
        best = max(best, gain)
        if gain <= -stop * atr:
            return j, "safety stop"
        if best >= trigger * atr and gain <= (1 - giveback) * best:
            return j, "gave back half"
    return FLAT, "3:45 exit"


def run(days, sym, series_cache, sessions, p):
    trades = []
    for d in days:
        sg = signal(d, p["band"])
        if not sg:
            continue
        m, side, atr = sg
        j, why = manage(d, m, side, atr, p["trigger"], p["giveback"], p["stop"])
        spot = d["c"][m - 1]
        strike = round(spot - 2) if side == "call" else round(spot + 2)          # ~$2 in the money
        k = sessions.index(d["date"])
        expiry = sessions[k + 1] if k + 1 < len(sessions) else (pd.Timestamp(d["date"]) + pd.offsets.BDay(1)).date()
        ser = series_cache(d, expiry, strike, side)
        u_ret = (1 if side == "call" else -1) * (d["c"][j] / d["c"][m] - 1)
        if ser is None or not np.isfinite(ser[m]) or ser[m] < 0.10 or not np.isfinite(ser[j]):
            trades.append({"date": str(d["date"]), "sym": sym, "side": side, "u_ret": u_ret, "pnl": None, "why": why})
            continue
        buy = ser[m] * (1 + p["spread"] / 2); sell = ser[j] * (1 - p["spread"] / 2)
        n = int(STAKE // (buy * 100))
        trades.append({"date": str(d["date"]), "sym": sym, "side": side, "entry_t": f"{(570 + m) // 60}:{(570 + m) % 60:02d}",
                       "exit_t": f"{(570 + j) // 60}:{(570 + j) % 60:02d}", "why": why, "contracts": n,
                       "buy": round(buy, 2), "sell": round(sell, 2), "pnl": round(n * 100 * (sell - buy), 2),
                       "pct": round(sell / buy - 1, 4), "u_ret": round(u_ret, 5)})
    return trades


def summary(trades):
    t = [x for x in trades if x["pnl"] is not None]
    if not t:
        return {"trades": 0}
    pnl = np.array([x["pnl"] for x in t]); cum = np.cumsum(pnl)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    return {"trades": len(t), "win_rate": round(float((pnl > 0).mean()), 3), "total": round(float(pnl.sum()), 2),
            "avg_win": round(float(wins.mean()), 2) if len(wins) else 0, "avg_loss": round(float(losses.mean()), 2) if len(losses) else 0,
            "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if len(losses) and losses.sum() < 0 else None,
            "max_drawdown": round(float((cum - np.maximum.accumulate(cum)).min()), 2),
            "shares_total_pct": round(100 * float(sum(x["u_ret"] for x in t)), 2)}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"]); args = ap.parse_args()
    base = {"band": 0.3, "trigger": 4.0, "giveback": 0.5, "stop": 10.0, "spread": SPREAD}
    client = None
    out = {"rules": base, "stake": STAKE, "symbols": {}, "grid": None}
    all_days = {}
    for sym in args.symbols:
        days = load_days(sym); all_days[sym] = days
        sessions = [d["date"] for d in days]
        cache: dict = {}

        def series(d, expiry, strike, side, sym=sym, cache=cache):
            nonlocal client
            key = d["date"]
            if key not in cache:
                try:
                    if client is None and d["date"] >= OH.FIRST_DAY:
                        client = OH._client()
                    lo, hi = float(np.nanmin(d["l"])), float(np.nanmax(d["h"]))
                    cache[key] = OH.day_series(OH.fetch_day(sym, d["date"], expiry, lo, hi, client))
                except Exception as exc:
                    print(f"no option prices for {sym} {d['date']}: {exc}"); cache[key] = {}
            return cache[key].get(OH.occ(sym, expiry, side == "call", strike))
        print(f"{sym}: {len(days)} trading days {days[0]['date']} -> {days[-1]['date']}", flush=True)
        trades = run(days, sym, series, sessions, base)
        s = summary(trades)
        by_month = {}
        for x in trades:
            if x["pnl"] is not None:
                by_month[x["date"][:7]] = round(by_month.get(x["date"][:7], 0) + x["pnl"], 2)
        reasons = {}
        for x in trades:
            if x["pnl"] is not None:
                r = reasons.setdefault(x["why"], [0, 0.0]); r[0] += 1; r[1] = round(r[1] + x["pnl"], 2)
        out["symbols"][sym] = {"summary": s, "by_month": by_month, "exits": reasons, "signal_days": len(trades),
                               "no_option_data": sum(1 for x in trades if x["pnl"] is None), "trades": trades[-60:]}
        print(f"  as written: {s}", flush=True)
        print(f"  exits: {reasons}", flush=True)
        out["symbols"][sym]["_series"] = series
        out["symbols"][sym]["_sessions"] = sessions
    # sensitivity grid: every combination of neighbouring settings
    grid = []
    for band, trig, give, stop in itertools.product((0.2, 0.3, 0.4), (3.0, 4.0, 5.0), (0.4, 0.5, 0.6), (8.0, 10.0, 12.0)):
        p = {**base, "band": band, "trigger": trig, "giveback": give, "stop": stop}
        tot = 0.0; n = 0
        for sym in args.symbols:
            s = summary(run(all_days[sym], sym, out["symbols"][sym]["_series"], out["symbols"][sym]["_sessions"], p))
            tot += s.get("total", 0); n += s.get("trades", 0)
        grid.append({"band": band, "trigger": trig, "giveback": give, "stop": stop, "total": round(tot, 2), "trades": n})
    totals = np.array([g["total"] for g in grid])
    out["grid"] = {"combos": len(grid), "profitable_share": round(float((totals > 0).mean()), 3),
                   "median_total": round(float(np.median(totals)), 2), "best": max(grid, key=lambda g: g["total"]),
                   "worst": min(grid, key=lambda g: g["total"]), "all": grid}
    for sym in args.symbols:
        out["symbols"][sym].pop("_series"); out["symbols"][sym].pop("_sessions")
    STATE.mkdir(parents=True, exist_ok=True)
    out["updated"] = dt.datetime.now(NY).isoformat(timespec="seconds")
    (STATE / "results.json").write_text(json.dumps(out, default=str))

    print("\n================ opening-range breakout: results ================")
    for sym, r in out["symbols"].items():
        s = r["summary"]
        if not s.get("trades"):
            print(f"{sym}: no trades with option data"); continue
        print(f"{sym}: {s['trades']} trades on {r['signal_days']} signal days · win rate {s['win_rate']:.0%} · "
              f"total ${s['total']:+,.0f} · avg win ${s['avg_win']:,.0f} / avg loss ${s['avg_loss']:,.0f} · "
              f"profit factor {s['profit_factor']} · worst drawdown ${s['max_drawdown']:,.0f}")
        print(f"      same signals with shares instead of options: {s['shares_total_pct']:+.1f}% summed")
        months = r["by_month"]; green = sum(1 for v in months.values() if v > 0)
        print(f"      months: {green}/{len(months)} green · " + "  ".join(f"{k[2:]}:{v:+,.0f}" for k, v in sorted(months.items())))
    g = out["grid"]
    print(f"\nrobustness: {g['profitable_share']:.0%} of {g['combos']} nearby settings are profitable · median total ${g['median_total']:+,.0f}")
    print(f"   best  {g['best']}")
    print(f"   worst {g['worst']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
