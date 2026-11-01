"""Crypto walk-forward search, shown live in the lab app like the options searches.

Replays LabRegime and LabTrend (the Freqtrade strategies) hour by hour on Coinbase history for
BTC/ETH/SOL, $10,000 split across the three coins, 0.26% fee per side, entries/exits at the next
hour's open (as Freqtrade does). Every variation is graded month by month on the SEARCH period
(history up to 2024-12-31). Each run, the best new candidate gets one look at the LOCKED period
(2025-01-01 -> now) and becomes the champion only if it makes money there and beats the current
champion. Holding the three coins is reported alongside.

Writes STATE_DIR/search.json, search/leaderboard.json, search/recent_trades.json, search/champion.json
(same shapes as the options search, so the lab card can show it).

  .venv/bin/python live/crypto_search.py --trials 300
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
STATE_DIR = Path(os.environ.get("STATE_DIR", Path.home() / "projects" / "paper-state" / "crypto"))
DATA = Path(os.environ.get("CRYPTO_DATA", Path.home() / "projects" / "freqtrade" / "user_data" / "data" / "coinbase"))
PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD"]
WALLET, FEE = 10_000.0, 0.0026
SPLIT = pd.Timestamp("2025-01-01", tz="UTC")
REGIME_HOURS = [168, 336, 720, 1440, 2400]
SPACE = {
    "LabRegime": {"regime_hours": REGIME_HOURS, "band": [0.0, 0.005, 0.01, 0.02, 0.03, 0.04], "pair_filter": [True, False]},
    "LabTrend": {"atr_mult": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], "breakout_bars": [12, 24, 36, 48, 72, 96, 120],
                 "adx_min": [12, 15, 20, 25, 30, 35], "shock_mult": [3.0, 4.0, 5.0, 6.0, 8.0]},
}
STATUS: dict = {}


def write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=lambda o: o.item() if hasattr(o, "item") else str(o)))   # numpy numbers -> plain
    tmp.replace(path)


def status(**kw) -> None:
    import socket
    STATUS.update(kw); STATUS["updated"] = dt.datetime.now(NY).isoformat(timespec="seconds")
    STATUS["host"] = socket.gethostname()
    write(STATE_DIR / "search.json", STATUS)


def describe(p: dict) -> str:
    if p["strategy"] == "LabRegime":
        return (f"LabRegime · BTC above its {p['regime_hours'] // 24}-day average · band {p['band']:.1%}"
                + (" · coin trending too" if p["pair_filter"] else ""))
    return (f"LabTrend · {p['breakout_bars']}h breakout · ADX>{p['adx_min']} · stop {p['atr_mult']}×ATR · "
            f"shock {p['shock_mult']}×")


# ---- data + indicators (computed once) -------------------------------------------------------------
def load() -> dict:
    out = {}
    for pair in PAIRS:
        f = DATA / f"{pair.replace('/', '_')}-1h.feather"
        df = pd.read_feather(f)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        out[pair] = df.sort_values("date").reset_index(drop=True)
    idx = out["BTC/USD"]["date"]
    for pair in PAIRS:                                     # align every coin to BTC's hours
        out[pair] = out[pair].set_index("date").reindex(idx).ffill().reset_index()
    return out


def indicators(frames: dict) -> dict:
    btc = frames["BTC/USD"]
    ind = {"btc_close": btc["close"].to_numpy(), "date": btc["date"],
           "ns": btc["date"].dt.as_unit("ns").astype("int64").to_numpy()}   # UTC nanoseconds, for fast date lookups
    for h in REGIME_HOURS:
        ind[f"btc_ema{h}"] = btc["close"].ewm(span=h, adjust=False, min_periods=h // 2).mean().to_numpy()
    move = btc["close"].pct_change().abs()
    ind["move_ratio"] = (move / move.rolling(168, min_periods=48).mean()).to_numpy()
    ind["btc_risk_on"] = (btc["close"] > btc["close"].ewm(span=200, adjust=False, min_periods=200).mean()).to_numpy()
    for pair, df in frames.items():
        c, h, l = df["close"], df["high"], df["low"]
        prev = c.shift(1)
        tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        up, dn = h.diff(), -l.diff()
        plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0)); minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0))
        pdi = 100 * plus.ewm(alpha=1 / 14, adjust=False).mean() / atr; mdi = 100 * minus.ewm(alpha=1 / 14, adjust=False).mean() / atr
        adx = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        d = c.diff()
        rs = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        ind[pair] = {
            "open": df["open"].to_numpy(), "high": h.to_numpy(), "low": l.to_numpy(), "close": c.to_numpy(),
            "vol_ok": (df["volume"] > df["volume"].rolling(24).mean()).to_numpy(),
            "ema50": c.ewm(span=50, adjust=False, min_periods=50).mean().to_numpy(),
            "ema200": c.ewm(span=200, adjust=False, min_periods=200).mean().to_numpy(),
            "atr": atr.to_numpy(), "adx": adx.to_numpy(), "rsi": (100 - 100 / (1 + rs)).to_numpy(),
            **{f"ema{hh}": c.ewm(span=hh, adjust=False, min_periods=hh // 2).mean().to_numpy() for hh in REGIME_HOURS},
            "hh": {}, "high_s": h,
        }
    return ind


def signals(ind: dict, pair: str, p: dict) -> tuple[np.ndarray, np.ndarray]:
    x = ind[pair]
    if p["strategy"] == "LabRegime":
        e, b = x[f"ema{p['regime_hours']}"], p["band"]
        be = ind[f"btc_ema{p['regime_hours']}"]; bc = ind["btc_close"]
        enter = bc > be * (1 + b); exit_ = bc < be * (1 - b)
        if p["pair_filter"]:
            enter &= x["close"] > e; exit_ |= x["close"] < e * (1 - b)
        return enter, exit_
    n = p["breakout_bars"]
    if n not in x["hh"]:
        x["hh"][n] = x["high_s"].rolling(n).max().shift(1).to_numpy()
    shock = pd.Series(ind["move_ratio"] > p["shock_mult"]).astype(int).rolling(3, min_periods=1).max().to_numpy()
    enter = ((x["close"] > x["ema50"]) & (x["ema50"] > x["ema200"]) & (x["close"] > x["hh"][n]) & x["vol_ok"]
             & (x["adx"] > p["adx_min"]) & (x["rsi"] < 75) & ind["btc_risk_on"] & (shock == 0))
    return enter, x["close"] < x["ema50"]


def simulate(ind: dict, p: dict, start: pd.Timestamp, end: pd.Timestamp | None):
    """Run one variation between start and end. Returns (monthly $ P&L dict, trades list, equity end)."""
    dates = ind["date"]
    lo = int(np.searchsorted(ind["ns"], start.value))
    hi = len(dates) if end is None else int(np.searchsorted(ind["ns"], end.value))
    months = dates.dt.strftime("%Y-%m").to_numpy()
    monthly: dict = {}
    trades = []
    for pair in PAIRS:
        x = ind[pair]; enter, exit_ = signals(ind, pair, p)
        cash = WALLET / len(PAIRS); qty = 0.0; top = 0.0; entry_px = 0.0; entry_t = None
        o, h, l, c, atr = x["open"], x["high"], x["low"], x["close"], x["atr"]
        value_prev = cash
        for i in range(max(lo, 1), hi):
            if np.isnan(o[i]):
                continue
            # act at this hour's open on the previous hour's signal (as Freqtrade does)
            if qty == 0 and enter[i - 1]:
                qty = cash * (1 - FEE) / o[i]; entry_px = o[i]; top = o[i]; entry_t = dates.iloc[i]; cash = 0.0
            elif qty > 0:
                reason = None; px = o[i]
                if exit_[i - 1]:
                    reason = "trend lost" if p["strategy"] == "LabRegime" else "trend break"
                elif p["strategy"] == "LabTrend":
                    stop = max(top - p["atr_mult"] * atr[i - 1], entry_px * 0.88)
                    if l[i] <= stop:
                        reason, px = "trailing stop", min(o[i], stop)
                elif l[i] <= entry_px * 0.70:
                    reason, px = "backstop", min(o[i], entry_px * 0.70)
                if reason:
                    cash = qty * px * (1 - FEE)
                    cost = qty * entry_px / (1 - FEE)
                    trades.append({"t": entry_t.isoformat(), "sym": pair.split("/")[0], "trade": f"long · {reason}",
                                   "pnl": round(cash - cost, 2), "pct": round(100 * (cash / cost - 1), 1)})
                    qty = 0.0
                else:
                    top = max(top, h[i])
            value = cash + qty * c[i]
            m = months[i]
            monthly[m] = monthly.get(m, 0.0) + (value - value_prev)
            value_prev = value
    return monthly, trades


def hold_return(ind: dict, start, end) -> float:
    lo = int(np.searchsorted(ind["ns"], start.value))
    hi = (len(ind["ns"]) if end is None else int(np.searchsorted(ind["ns"], end.value))) - 1
    r = [ind[p]["close"][hi] / ind[p]["close"][lo] - 1 for p in PAIRS if not np.isnan(ind[p]["close"][lo])]
    return float(np.mean(r))


def grade(ind, p, start, end):
    monthly, trades = simulate(ind, p, start, end)
    months = sorted(monthly)
    pnl = sum(monthly.values())
    curve = np.cumsum([monthly[m] for m in months]) if months else np.array([0.0])
    wins = sum(t["pnl"] > 0 for t in trades)
    return {"months": [[m, round(monthly[m], 2)] for m in months], "pnl": round(pnl, 2), "trades": len(trades),
            "wins": wins, "simulated": len(trades), "max_dd": round(float((curve - np.maximum.accumulate(curve)).min()), 2)}, trades


def random_variation(rng: random.Random) -> dict:
    strat = rng.choice(list(SPACE))
    return {"strategy": strat, **{k: rng.choice(v) for k, v in SPACE[strat].items()}}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=300); args = ap.parse_args()
    board_path = STATE_DIR / "search" / "leaderboard.json"
    board = json.loads(board_path.read_text()) if board_path.exists() else []
    tried = {e["desc"] for e in board}
    champ_path = STATE_DIR / "search" / "champion.json"
    champ = json.loads(champ_path.read_text()) if champ_path.exists() else {}
    old = json.loads((STATE_DIR / "search.json").read_text()) if (STATE_DIR / "search.json").exists() else {}
    STATUS.update(name="Crypto search", trials_total=old.get("trials_total", 0), profitable=old.get("profitable", 0),
                  sim_trades_taken=old.get("sim_trades_taken", 0), sim_wins=old.get("sim_wins", 0),
                  simulated_total=old.get("simulated_total", 0), trials_tonight=0, champion=champ or None)
    status(phase="loading price history")
    frames = load(); ind = indicators(frames)
    start = ind["date"].iloc[0] + pd.Timedelta(hours=2500)          # warm-up for the longest average
    hold_search = hold_return(ind, start, SPLIT); hold_lock = hold_return(ind, SPLIT, None)
    STATUS.update(search_period=[str(start.date()), "2024-12-31"], holdout_period=["2025-01-01", str(ind["date"].iloc[-1].date())],
                  hold_search=round(hold_search, 4), hold_holdout=round(hold_lock, 4))
    rng = random.Random(); recent = []
    rt_path = STATE_DIR / "search" / "recent_trades.json"
    for k in range(args.trials):
        p = random_variation(rng); d = describe(p)
        if d in tried:
            continue
        tried.add(d)
        status(phase="testing", current=d)
        res, trades = grade(ind, p, start, SPLIT)
        board.append({"desc": d, "params": p, "tested": dt.datetime.now(NY).isoformat(timespec="seconds"), "search": res})
        STATUS["trials_total"] += 1; STATUS["trials_tonight"] += 1
        STATUS["profitable"] += res["pnl"] > 0; STATUS["sim_trades_taken"] += res["trades"]; STATUS["sim_wins"] += res["wins"]
        STATUS["simulated_total"] += res["simulated"]
        recent = (recent + trades[-40:])[-300:]
        write(rt_path, {"variation": d, "trades": recent})
        write(board_path, board[-1500:])
    # one look at the locked period for the best untested candidate
    status(phase="checking the best candidate on the locked period", current=None)
    looked = set(champ.get("looked", []))
    cands = sorted([e for e in board if e["search"]["trades"] >= 10 and e["desc"] not in looked], key=lambda e: -e["search"]["pnl"])
    if cands:
        best = cands[0]
        hold_res, _ = grade(ind, best["params"], SPLIT, None)
        looked.add(best["desc"])
        promoted = hold_res["pnl"] > 0 and hold_res["pnl"] > (champ.get("holdout_pnl") or 0)
        entry = {"desc": best["desc"], "params": best["params"], "holdout_pnl": hold_res["pnl"], "holdout_trades": hold_res["trades"],
                 "holdout_months": hold_res["months"], "hold_holdout_pct": round(hold_lock, 4),
                 "checked": dt.datetime.now(NY).isoformat(timespec="seconds")}
        champ = {**(entry if promoted else champ), "looked": sorted(looked)[-500:],
                 "holdout_checks": (champ.get("holdout_checks") or 0) + 1,
                 "last_check": {**entry, "promoted": promoted}}
        write(champ_path, champ)
        print(f"best candidate: {best['desc']} -> locked 2025-now ${hold_res['pnl']:+,.0f} "
              f"(holding the coins: {hold_lock:+.0%}) -> {'PROMOTED' if promoted else 'not promoted'}")
    status(phase="idle", champion=champ or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
