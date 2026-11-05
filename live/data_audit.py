"""Nightly data audit: is the price history the searches learn from clean? (read-only report)

Checks, per symbol (SPY, QQQ, IWM and anything else in the bars folder), over the last DAYS trading days:
  * minutes per day (390 normal, 210 half days), duplicated or out-of-order minutes, gaps of 5+ minutes
  * bad prints: a 1-minute candle that jumps 0.75%+ and reverses the next minute; zero/negative prices
  * Alpaca's last minute close vs Yahoo's official daily close (should agree within ~0.15%)
  * option-price files: one per trading day since Feb 2024, not empty
Report: ~/projects/paper-state/search/data_audit.json. If anything looks wrong, one alert (ntfy -> the lab app).

  .venv/bin/python live/data_audit.py            # nightly via data-audit.timer
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

NY = "America/New_York"
BARS = Path(os.environ.get("BARS_DIR", Path.home() / "projects" / "data" / "bars"))
OPTS = BARS.parent / "options"
OUT = Path(os.environ.get("PAPER_STATE", Path.home() / "projects" / "paper-state")) / "search" / "data_audit.json"
DAYS = int(os.environ.get("AUDIT_DAYS", "120"))
SPIKE = 0.0075
CLOSE_TOL = 0.0015


def load(sym: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(BARS / sym / "*.csv.gz")))[-8:]               # last ~8 months is plenty
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
    df["m"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute - 570
    return df[(df["m"] >= 0) & (df["m"] < 390)]


def yahoo_closes(sym: str, start: dt.date) -> dict:
    try:
        import yfinance as yf
        h = yf.Ticker(sym).history(start=str(start), auto_adjust=False)
        return {d.date(): float(c) for d, c in h["Close"].items()}
    except Exception:
        return {}


def audit_symbol(sym: str) -> dict:
    df = load(sym)
    if df.empty:
        return {"symbol": sym, "error": "no bars"}
    days = sorted(df["ts"].dt.date.unique())[-DAYS:]
    yc = yahoo_closes(sym, days[0])
    issues, per_day = [], []
    for day in days:
        g = df[df["ts"].dt.date == day].sort_values("ts")
        n, dups = g["m"].nunique(), int(g["m"].duplicated().sum())
        mins = np.sort(g["m"].unique())
        gaps = int((np.diff(mins) >= 5).sum()) if len(mins) > 1 else 0
        c = g.drop_duplicates("m")["close"].to_numpy(float)
        r = np.diff(c) / c[:-1] if len(c) > 2 else np.array([])
        spikes = int(((np.abs(r[:-1]) >= SPIKE) & (np.sign(r[:-1]) != np.sign(r[1:])) & (np.abs(r[1:]) >= SPIKE * 0.6)).sum()) if len(r) > 2 else 0
        bad_px = int((g[["open", "high", "low", "close"]] <= 0).to_numpy().sum())
        short = n < 300 and not (200 <= n <= 215)          # quiet minutes on the free IEX feed are normal; flag real holes
        close_err = None
        if day in yc and len(c):
            close_err = round(c[-1] / yc[day] - 1, 5)
        row = {"day": str(day), "minutes": int(n), "dups": dups, "gaps": gaps, "spikes": spikes, "bad_prices": bad_px, "close_vs_yahoo": close_err}
        per_day.append(row)
        why = [w for w, bad in (("short day", short), ("duplicate minutes", dups), ("gaps", gaps > 2), ("spike prints", spikes),
                                ("bad prices", bad_px), ("close off vs Yahoo", close_err is not None and abs(close_err) > CLOSE_TOL)) if bad]
        if why:
            issues.append(f"{sym} {day}: " + ", ".join(why) + (f" ({close_err:+.2%})" if close_err is not None and abs(close_err) > CLOSE_TOL else ""))
    errs = [abs(r["close_vs_yahoo"]) for r in per_day if r["close_vs_yahoo"] is not None]
    return {"symbol": sym, "days": len(days), "first": str(days[0]), "last": str(days[-1]), "issues": issues,
            "median_close_vs_yahoo": round(float(np.median(errs)), 5) if errs else None, "per_day": per_day[-30:]}


def audit_options(sym: str, trading_days: list) -> dict:
    files = {Path(f).name.split(".")[0]: Path(f) for f in glob.glob(str(OPTS / sym / "*.csv.gz"))}
    want = [d for d in trading_days if d >= "2024-02-01"]
    missing = [d for d in want if d not in files]
    empty = [d for d, f in files.items() if f.stat().st_size < 200]
    return {"symbol": sym, "option_days": len(files), "missing": missing[-20:], "missing_count": len(missing), "empty": sorted(empty)[-20:]}


def main() -> int:
    syms = sorted(p.name for p in BARS.iterdir() if p.is_dir()) if BARS.exists() else []
    rep = {"checked": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "symbols": [], "options": []}
    for s in syms:
        a = audit_symbol(s); rep["symbols"].append(a)
        if (OPTS / s).exists() and "per_day" in a:
            rep["options"].append(audit_options(s, [r["day"] for r in a["per_day"]]))
    problems = [i for a in rep["symbols"] for i in a.get("issues", [])]
    problems += [f"{o['symbol']} options: {o['missing_count']} recent days missing" for o in rep["options"] if o["missing_count"]]
    rep["ok"], rep["problems"] = not problems, problems[:50]
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(rep, default=str))
    for a in rep["symbols"]:
        if "error" in a:
            print(f"{a['symbol']}: {a['error']}"); continue
        print(f"{a['symbol']}: {a['days']} days {a['first']} -> {a['last']} · {len(a['issues'])} flagged"
              + (f" · Alpaca vs Yahoo close: median {a['median_close_vs_yahoo']:.3%} apart" if a["median_close_vs_yahoo"] is not None else ""))
    for o in rep["options"]:
        print(f"{o['symbol']} options: {o['option_days']} days of option prices · {o['missing_count']} recent days missing")
    print("OK: data looks clean" if rep["ok"] else "FLAGGED:\n  " + "\n  ".join(problems[:15]))
    topic = os.environ.get("NTFY_TOPIC")
    if problems and topic and os.environ.get("AUDIT_ALERT", "1") == "1":
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{topic}", data=("\n".join(problems[:6])).encode(),
                                   headers={"Title": f"data audit: {len(problems)} thing(s) to look at", "Tags": "warning"}), timeout=10)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
