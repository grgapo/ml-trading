"""Simulator check: how well does the backtest's option pricing match real paper fills?

For every real options trade the bot made (options_trades.csv), re-price the same contract at
the same entry and exit minutes the way the simulator does (Black-Scholes, IV = IV_MULT x recent
realized volatility, SPREAD round trip) and compare with what Alpaca actually filled:

  * entry price error   (simulated ask vs real fill)
  * P&L error           (simulated return over the same holding window vs real return)

Then it searches IV_MULT and SPREAD for the values that would have matched the real trades best.
Report only: nothing is changed automatically. Written to STATE_DIR/search/calibration.json and
shown in the app. Runs at the start of the nightly search (or by hand).

  .venv/bin/python live/sim_calibration.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from options_pretrain import IV_MULT, NY, SPREAD, STATE_DIR, _bs, download_month  # noqa: E402

OUT = STATE_DIR / "search" / "calibration.json"
OCC = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")
MIN_TRUST = 30            # trades before a suggestion is worth acting on


def parse_contract(sym: str):
    """OCC symbol -> (underlying, expiry date, is_call, strike)."""
    m = OCC.match(sym or "")
    if not m:
        return None
    u, ymd, cp, k = m.groups()
    return u, dt.datetime.strptime(ymd, "%y%m%d").date(), cp == "C", int(k) / 1000


def minute_bars(symbol: str, day: dt.date, cache: dict) -> pd.DataFrame:
    key = (symbol, day.year, day.month)
    if key not in cache:
        cache[key] = download_month(symbol, day.year, day.month)
    return cache[key]


def realized_sigma(bars: pd.DataFrame, at: pd.Timestamp, step: int = 1) -> float | None:
    """Annualized realized vol from the last trading day of `step`-minute returns.
    step=1 is what the simulator uses today; noisy IEX prints inflate it, longer steps damp that."""
    past = bars[bars["ts"] <= at]["close"].to_numpy(float)[-390:]
    if len(past) < 120:
        return None
    r = np.diff(np.log(past[::step]))
    return float(np.std(r) * math.sqrt(252 * 390 / step))


def years_to_expiry(at: pd.Timestamp, expiry: dt.date) -> float:
    close = pd.Timestamp(dt.datetime.combine(expiry, dt.time(16, 0)), tz=NY)
    trading_minutes = max(1.0, (close - at).total_seconds() / 60)
    if expiry > at.date():                                  # overnight: count ~390 trading minutes per session
        days = np.busday_count(at.date(), expiry)
        trading_minutes = max(1.0, (16 * 60 - (at.hour * 60 + at.minute)) + 390 * (days - 1) + 390)
    return trading_minutes / (252 * 390)


def price_at(bars, at, strike, expiry, call, sigma):
    row = bars[bars["ts"] <= at].tail(1)
    if row.empty:
        return None, None
    spot = float(row["close"].iloc[0])
    return float(_bs(spot, strike, years_to_expiry(at, expiry), sigma, call)), spot


def evaluate(trades: list[dict], iv_mult: float, spread: float, cache: dict, step: int = 1) -> list[dict]:
    out = []
    for t in trades:
        c = parse_contract(t["contract"])
        if not c:
            continue
        u, expiry, call, strike = c
        opened = pd.Timestamp(t["opened"]).tz_convert(NY)
        closed = pd.Timestamp(t["closed"]).tz_convert(NY)
        bars = minute_bars(u, opened.date(), cache)
        if bars.empty:
            continue
        sigma0 = realized_sigma(bars, opened, step)
        if not sigma0:
            continue
        sigma = min(1.2, max(0.08, sigma0 * iv_mult))
        mid_in, spot_in = price_at(bars, opened, strike, expiry, call, sigma)
        mid_out, spot_out = price_at(bars, closed, strike, expiry, call, sigma)
        if not mid_in or mid_in < 0.02 or mid_out is None:
            continue
        sim_entry = mid_in * (1 + spread / 2)
        sim_exit = mid_out * (1 - spread / 2)
        real_entry = float(t["entry"])
        real_ret = float(t["pl_pct"])
        out.append({"opened": t["opened"], "contract": t["contract"], "real_entry": real_entry,
                    "sim_entry": round(sim_entry, 3), "entry_err": round(sim_entry / real_entry - 1, 4),
                    "real_ret": round(real_ret, 4), "sim_ret": round(sim_exit / sim_entry - 1, 4),
                    "ret_err": round((sim_exit / sim_entry - 1) - real_ret, 4),
                    "spot_move": round(spot_out / spot_in - 1, 5), "sigma": round(sigma0, 3)})
    return out


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"trades": 0}
    e = np.array([r["entry_err"] for r in rows]); p = np.array([r["ret_err"] for r in rows])
    return {"trades": len(rows), "entry_err_median": round(float(np.median(e)), 4),
            "entry_abs_err": round(float(np.median(np.abs(e))), 4),
            "ret_err_mean": round(float(np.mean(p)), 4), "ret_abs_err": round(float(np.median(np.abs(p))), 4)}


def main() -> int:
    path = STATE_DIR / "options_trades.csv"
    trades = [r for r in csv.DictReader(path.open())] if path.exists() else []
    trades = [t for t in trades if t.get("contract") and t.get("entry") and t.get("closed")]
    cache: dict = {}
    current = evaluate(trades, IV_MULT, SPREAD, cache)
    best, by_step = None, {}
    if current:
        for step in (1, 5, 15):                                   # which volatility estimate matches reality?
            step_best = None
            for iv in np.arange(0.2, 1.81, 0.05):
                for sp in (0.0, 0.02, 0.04, 0.06, 0.08, 0.12):
                    rows = evaluate(trades, float(iv), sp, cache, step)
                    if not rows:
                        continue
                    s = summarize(rows)
                    score = s["entry_abs_err"] + s["ret_abs_err"]      # match both price level and outcome
                    if step_best is None or score < step_best[0]:
                        step_best = (score, round(float(iv), 2), sp, s)
            if step_best:
                by_step[step] = {"iv_mult": step_best[1], "spread": step_best[2], "score": round(step_best[0], 4),
                                 **step_best[3]}
                if best is None or step_best[0] < best[0]:
                    best = (step_best[0], step_best[1], step_best[2], {**step_best[3], "vol_bars_minutes": step})
    now = summarize(current)
    verdict = ("not enough real trades yet" if now["trades"] == 0 else
               f"based on {now['trades']} real trades" + ("" if now["trades"] >= MIN_TRUST else f" (trust it after {MIN_TRUST})"))
    report = {
        "checked": dt.datetime.now(NY).isoformat(timespec="seconds"),
        "settings": {"iv_mult": IV_MULT, "spread": SPREAD}, "current": now,
        "best_fit": ({"iv_mult": best[1], "spread": best[2], **best[3]} if best else None),
        "by_vol_estimate": by_step,
        "edge_of_grid": bool(best and best[1] <= 0.2),
        "trustworthy": now["trades"] >= MIN_TRUST, "verdict": verdict,
        "trades": current[-40:],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp"); tmp.write_text(json.dumps(report)); tmp.replace(OUT)
    if now["trades"]:
        print(f"{verdict}: simulated entry prices are {now['entry_err_median']:+.1%} off (median), "
              f"returns {now['ret_err_mean']:+.1%} off on average")
        for step, b in by_step.items():
            print(f"  vol from {step:>2}-min bars: best IV x{b['iv_mult']}, spread {b['spread']:.0%} -> "
                  f"entry prices {b['entry_err_median']:+.1%} off, returns {b['ret_err_mean']:+.1%} off (score {b['score']})")
        if best:
            print(f"best overall: {best[3]['vol_bars_minutes']}-min bars, IV x{best[1]}, spread {best[2]:.0%} "
                  f"(simulator now: 1-min bars, IV x{IV_MULT}, spread {SPREAD:.0%})")
    else:
        print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
