"""Where does the simulator's price error really come from? (read-only report, changes nothing)

The nightly search prices a practice trade with REAL option minute bars (option_history.py) when all
four contracts (call/put, 0DTE/1DTE) traded at that moment, and falls back to Black-Scholes otherwise.
sim_calibration.py only checks the Black-Scholes path. This checks the rest:

  1. real fills vs real bars: for every real bot trade, the same contract's minute bar at the fill
     minute (what the search would have paid) vs what Alpaca actually filled, plus how stale that
     bar was (minutes since the contract last traded) and the same at the exit.
  2. how much of each search dataset is real-priced vs Black-Scholes, and how differently they score.

  STATE_DIR=~/projects/paper-state/options-spy .venv/bin/python live/sim_check_real.py
"""
from __future__ import annotations

import csv
import datetime as dt
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from options_pretrain import DATA_DIR, NY, SPREAD, STATE_DIR  # noqa: E402
from sim_calibration import parse_contract  # noqa: E402

OUT = STATE_DIR / "search" / "sim_check_real.json"


def contract_bars(contract: str, day: dt.date) -> pd.DataFrame:
    """That one contract's minute bars for that day: columns minute (0 = 9:30), close, volume."""
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame
    c = OptionHistoricalDataClient(os.environ["ALPACA_API_KEY_ID"], os.environ["ALPACA_API_SECRET_KEY"])
    start = dt.datetime.combine(day, dt.time(9, 30), NY); end = dt.datetime.combine(day, dt.time(16, 0), NY)
    res = c.get_option_bars(OptionBarsRequest(symbol_or_symbols=[contract], timeframe=TimeFrame.Minute, start=start, end=end))
    rows = []
    for b in res.data.get(contract, []):
        ts = pd.Timestamp(b.timestamp).tz_convert(NY)
        rows.append((ts.hour * 60 + ts.minute - 570, float(b.close), float(b.volume)))
    return pd.DataFrame(rows, columns=["minute", "close", "volume"])


def at_minute(bars: pd.DataFrame, m: int):
    """(price the simulator would use at minute m: last trade at or before it, minutes since that trade)."""
    past = bars[bars["minute"] <= m]
    if past.empty:
        return None, None
    last = past.iloc[-1]
    return float(last["close"]), int(m - last["minute"])


def minute_of(ts: str) -> tuple[dt.date, int]:
    t = pd.Timestamp(ts).tz_convert(NY)
    return t.date(), t.hour * 60 + t.minute - 570


def check_fills() -> list[dict]:
    path = STATE_DIR / "options_trades.csv"
    trades = [r for r in csv.DictReader(path.open())] if path.exists() else []
    out = []
    for t in trades:
        c = parse_contract(t.get("contract", ""))
        if not c or not t.get("entry") or not t.get("closed"):
            continue
        day, m_in = minute_of(t["opened"]); _, m_out = minute_of(t["closed"])
        try:
            bars = contract_bars(t["contract"], day)
        except Exception as exc:
            out.append({"contract": t["contract"], "error": f"{type(exc).__name__}: {str(exc)[:80]}"}); continue
        real_in = float(t["entry"]); real_out = real_in * (1 + float(t["pl_pct"]))
        p_in, stale_in = at_minute(bars, m_in)
        p_out, stale_out = at_minute(bars, m_out)
        if p_in is None:
            out.append({"contract": t["contract"], "opened": t["opened"], "error": "no trades in this contract before the fill"}); continue
        sim_in = p_in * (1 + SPREAD / 2)
        row = {"contract": t["contract"], "opened": t["opened"], "real_entry": real_in, "bar_entry": round(p_in, 3),
               "sim_entry": round(sim_in, 3), "entry_err": round(sim_in / real_in - 1, 4), "stale_in_min": stale_in,
               "volume_today": int(bars["volume"].sum()), "reason": t.get("exit_reason", "")}
        if p_out is not None:
            sim_out = p_out * (1 - SPREAD / 2)
            row.update({"real_exit": round(real_out, 3), "bar_exit": round(p_out, 3), "exit_err": round(sim_out / real_out - 1, 4),
                        "stale_out_min": stale_out, "real_ret": round(float(t["pl_pct"]), 4),
                        "sim_ret": round(sim_out / sim_in - 1, 4)})
            row["ret_err"] = round(row["sim_ret"] - row["real_ret"], 4)
        out.append(row)
    return out


def check_datasets() -> list[dict]:
    """Share of each cached search dataset priced with real bars vs Black-Scholes, and their average outcome."""
    out = []
    for f in sorted(glob.glob(str(DATA_DIR.parent / "search_cache" / "*" / "*.npz")))[:400]:
        try:
            d = np.load(f)
            if "real" not in d.files or not len(d["real"]):
                continue
            real = d["real"].astype(bool); R = d["R"]
            ts = pd.to_datetime(d["ts"], utc=True)
            recent = ts >= pd.Timestamp("2024-02-01", tz="UTC")
            best_r = np.nanmax(R, axis=1)
            out.append({"set": Path(f).parent.name, "file": Path(f).stem[:40], "decisions": int(len(real)),
                        "real_share": round(float(real.mean()), 3),
                        "real_share_since_feb24": round(float(real[recent].mean()), 3) if recent.any() else None,
                        "avg_best_real": round(float(np.mean(best_r[real])), 4) if real.any() else None,
                        "avg_best_model": round(float(np.mean(best_r[~real])), 4) if (~real).any() else None})
        except Exception:
            continue
    return out


def main() -> int:
    fills = check_fills()
    good = [r for r in fills if "entry_err" in r]
    ds = check_datasets()
    rep = {"checked": dt.datetime.now(NY).isoformat(timespec="seconds"), "spread_assumed": SPREAD, "fills": fills, "datasets": ds}
    if good:
        e = np.array([r["entry_err"] for r in good]); st = np.array([r["stale_in_min"] for r in good])
        rep["summary"] = {"trades": len(good), "entry_err_median": round(float(np.median(e)), 4),
                          "entry_abs_err_median": round(float(np.median(np.abs(e))), 4),
                          "stale_in_median_min": float(np.median(st)), "stale_over_5min": int((st > 5).sum())}
        rr = [r["ret_err"] for r in good if "ret_err" in r]
        if rr:
            rep["summary"]["ret_err_mean"] = round(float(np.mean(rr)), 4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, default=str))

    print(f"== real fills vs the real option bars the search uses ({STATE_DIR.name}) ==")
    for r in fills:
        if "error" in r:
            print(f"  {r['contract']}: {r['error']}"); continue
        print(f"  {r['opened'][:16]} {r['contract']}: filled ${r['real_entry']:.2f} · bar ${r['bar_entry']:.2f} "
              f"({r['stale_in_min']} min old) · sim pays {r['entry_err']:+.1%}"
              + (f" | exit: real ${r['real_exit']:.2f} bar ${r['bar_exit']:.2f} ({r['stale_out_min']} min old) · "
                 f"return real {r['real_ret']:+.0%} sim {r['sim_ret']:+.0%}" if "real_exit" in r else ""))
    if good:
        s = rep["summary"]
        print(f"  -> median entry error {s['entry_err_median']:+.1%} (typical size {s['entry_abs_err_median']:.1%}), "
              f"bars median {s['stale_in_median_min']:.0f} min old, {s['stale_over_5min']} of {s['trades']} over 5 min old"
              + (f", returns off {s['ret_err_mean']:+.1%} on average" if "ret_err_mean" in s else ""))
    print("== search datasets: how much is priced with real bars vs Black-Scholes ==")
    if not ds:
        print("  no cached search datasets found")
    by_set: dict = {}
    for r in ds:
        by_set.setdefault(r["set"], []).append(r)
    for k, rows in by_set.items():
        rs = [r["real_share"] for r in rows]; rs2 = [r["real_share_since_feb24"] for r in rows if r["real_share_since_feb24"] is not None]
        br = [r["avg_best_real"] for r in rows if r["avg_best_real"] is not None]; bm = [r["avg_best_model"] for r in rows if r["avg_best_model"] is not None]
        print(f"  {k}: {len(rows)} datasets · real-priced {np.median(rs):.0%} overall"
              + (f", {np.median(rs2):.0%} since Feb 2024" if rs2 else "")
              + (f" · best-action avg: real {np.median(br):+.1%} vs model {np.median(bm):+.1%}" if br and bm else ""))
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
