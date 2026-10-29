"""Pre-train the options bot on years of real SPY/QQQ 1-minute history.

  1. Download 1-minute bars (Alpaca, free IEX feed) and cache them by month.
  2. Add indicators (live/features.py: VWAP, volume, FVGs, EMAs, RSI, MACD, ATR...).
  3. Every DECISION_EVERY minutes of each day, simulate ALL four trades
     (1DTE/0DTE call/put) with the bot's real rules: entry/exit spread,
     -35% stop, +50% target, 60-minute max hold, flat by 15:45. Options are
     priced with Black-Scholes using recent realized volatility.
  4. Walk-forward test: train on the past, trade the next month under the daily
     rules (max trades, stop at win target, $ per trade), roll forward. This is
     the honest score; it is compared with picking trades at random.
  5. Train the final model on everything and save it for live trading.

The live bot loads the model automatically and keeps learning from its own trades.
Progress shows in the lab app (research card).

  MPLBACKEND=Agg .venv/bin/python live/options_pretrain.py              # 24 months
  .venv/bin/python live/options_pretrain.py --months 36
  .venv/bin/python live/options_pretrain.py --synthetic                 # offline test
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, add_indicators  # noqa: E402

NY = ZoneInfo("America/New_York")
ACTIONS = ["skip", "call_1dte", "put_1dte", "call_0dte", "put_0dte"]
TRADE_ACTIONS = ACTIONS[1:]

STATE_DIR = Path(os.environ.get("STATE_DIR", Path.home() / "projects" / "paper-state"))
DATA_DIR = Path(os.environ.get("BARS_DIR", Path.home() / "projects" / "data" / "bars"))
MODEL_PATH = STATE_DIR / "options_model.joblib"
DATASET_PATH = DATA_DIR.parent / "options_dataset.npz"

STOP_LOSS = float(os.environ.get("OPT_STOP_LOSS", "-0.35"))
TAKE_PROFIT = float(os.environ.get("OPT_TAKE_PROFIT", "0.50"))
MAX_HOLD_MIN = int(os.environ.get("OPT_MAX_HOLD_MIN", "60"))
PER_TRADE = float(os.environ.get("OPT_PER_TRADE", "1000"))
MAX_TRADES = int(os.environ.get("OPT_MAX_TRADES", "5"))
WIN_TARGET = int(os.environ.get("OPT_WIN_TARGET", "3"))
DAILY_BUDGET = float(os.environ.get("OPT_DAILY_BUDGET", "5000"))
MIN_EDGE = float(os.environ.get("OPT_MIN_EDGE", "0.03"))      # only trade if predicted >= +3%
SPREAD = float(os.environ.get("OPT_SIM_SPREAD", "0.04"))       # 4% round-trip bid/ask cost
IV_MULT = float(os.environ.get("OPT_SIM_IV_MULT", "1.15"))     # implied vol ~ 1.15x realized
DECISION_EVERY = int(os.environ.get("OPT_DECISION_EVERY", "10"))
FIRST_ENTRY_MIN, LAST_ENTRY_MIN, FLATTEN_MIN = 15, 330, 375  # minutes after 9:30
RATE = 0.04
MIN_TRAIN_MONTHS = 6

STATUS = {"name": "Options pre-training", "phase": "starting", "started": None, "symbol": None,
          "index": 0, "total": None, "epoch": None, "epochs": None, "seconds_left": None,
          "results": [], "failures": []}


def write_status(**changes) -> None:
    STATUS.update(changes)
    STATUS["updated"] = dt.datetime.now(NY).isoformat(timespec="seconds")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_DIR / "research.json.tmp"
    tmp.write_text(json.dumps(STATUS))
    tmp.replace(STATE_DIR / "research.json")


# ---- 1. data -------------------------------------------------------------------------
def month_list(months: int) -> list[tuple[int, int]]:
    today = dt.date.today().replace(day=1)
    out = []
    y, m = today.year, today.month
    for _ in range(months):
        out.append((y, m))
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return sorted(out)


def regular_session(df: pd.DataFrame) -> pd.DataFrame:
    mins = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    return df[(mins >= 570) & (mins < 960)].reset_index(drop=True)


def download_month(symbol: str, y: int, m: int) -> pd.DataFrame:
    path = DATA_DIR / symbol / f"{y}-{m:02d}.csv.gz"
    current = (y, m) == (dt.date.today().year, dt.date.today().month)
    if path.exists() and not current:
        return pd.read_csv(path, parse_dates=["ts"]).assign(ts=lambda d: d["ts"].dt.tz_convert(NY))
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY_ID"], os.environ["ALPACA_API_SECRET_KEY"])
    start = dt.datetime(y, m, 1, tzinfo=NY)
    end = dt.datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=NY)
    end = min(end, dt.datetime.now(NY) - dt.timedelta(minutes=16))   # free plan: no last 15 min
    bars = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
                                                  start=start, end=end, feed=DataFeed.IEX)).df
    if bars.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    bars = bars.reset_index()
    df = pd.DataFrame({"ts": pd.to_datetime(bars["timestamp"]).dt.tz_convert(NY),
                       "open": bars["open"], "high": bars["high"], "low": bars["low"],
                       "close": bars["close"], "volume": bars["volume"]})
    df = regular_session(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def synthetic_bars(symbol: str, months: int) -> pd.DataFrame:
    """Fake 1-minute bars for offline testing (random walk with a little momentum)."""
    rng = np.random.default_rng(7 if symbol == "SPY" else 11)
    days = pd.bdate_range(end=dt.date.today() - dt.timedelta(days=1), periods=months * 21)
    rows, px = [], 500.0 if symbol == "SPY" else 420.0
    for d in days:
        t0 = pd.Timestamp(d.date()).tz_localize(NY) + pd.Timedelta(hours=9, minutes=30)
        drift = rng.normal(0, float(os.environ.get("SYN_DRIFT", "0.00005")))
        prev = 0.0
        for k in range(390):
            r = drift + float(os.environ.get("SYN_MOMENTUM", "0.15")) * prev + rng.normal(0, 0.0006)
            prev = r
            o, px = px, px * math.exp(r)
            hi, lo = max(o, px) * (1 + abs(rng.normal(0, 0.0002))), min(o, px) * (1 - abs(rng.normal(0, 0.0002)))
            rows.append((t0 + pd.Timedelta(minutes=k), o, hi, lo, px, int(rng.integers(500, 5000))))
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])


# ---- 2/3. simulate option trades ----------------------------------------------------------
def _bs(S, K, T, sigma, call: bool):
    from scipy.special import ndtr
    T = np.maximum(T, 1e-6)
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (RATE + 0.5 * sigma ** 2) * T) / sq
    d2 = d1 - sq
    if call:
        return S * ndtr(d1) - K * np.exp(-RATE * T) * ndtr(d2)
    return K * np.exp(-RATE * T) * ndtr(-d2) - S * ndtr(-d1)


def simulate_trade(close: np.ndarray, mins: np.ndarray, i: int, end: int, sigma: float,
                   action: str) -> tuple[float, int, float] | None:
    """Return (P&L fraction of premium, exit index, entry price) or None if untradeable."""
    kind, dte = action.split("_")
    call = kind == "call"
    extra = 390 if dte == "1dte" else 0
    S0, K = close[i], round(close[i])
    year_min = 252 * 390
    entry_mid = float(_bs(S0, K, (960 - (mins[i] + 570) + extra) / year_min, sigma, call))
    if entry_mid < 0.05:
        return None
    entry = entry_mid * (1 + SPREAD / 2)
    j = np.arange(i + 1, end + 1)
    if len(j) == 0:
        return None
    T = (960 - (mins[j] + 570) + extra) / year_min
    value = _bs(close[j], K, T, sigma, call) * (1 - SPREAD / 2)
    plpc = value / entry - 1
    held = mins[j] - mins[i]
    hit = (plpc <= STOP_LOSS) | (plpc >= TAKE_PROFIT) | (held >= MAX_HOLD_MIN) | (mins[j] >= FLATTEN_MIN)
    k = int(np.argmax(hit)) if hit.any() else len(j) - 1
    return float(max(-1.0, min(1.5, plpc[k]))), int(j[k]), entry


def build_dataset(frames: dict[str, pd.DataFrame]) -> dict:
    X, R, TS, SYM, EXIT_TS, ENTRY = [], [], [], [], [], []
    days_total = sum(df["date"].nunique() for df in frames.values())
    done, t0 = 0, time.time()
    for symbol, df in frames.items():
        mins = (df["ts"].dt.hour * 60 + df["ts"].dt.minute - 570).to_numpy()
        close = df["close"].to_numpy(float)
        logret = np.diff(np.log(close), prepend=np.log(close[0]))
        rv = pd.Series(logret).rolling(390, min_periods=120).std().to_numpy() * math.sqrt(252 * 390)
        feats = df[FEATURES].to_numpy(float)
        for _, idx in df.groupby("date").indices.items():
            idx = np.sort(idx)
            day_end = idx[-1]
            for i in idx:
                m = mins[i]
                if m < FIRST_ENTRY_MIN or m > LAST_ENTRY_MIN or m % DECISION_EVERY:
                    continue
                f = feats[i].copy()
                if np.isnan(f[FEATURES.index("day_gap")]):
                    f[FEATURES.index("day_gap")] = 0.0
                if np.isnan(f).any() or np.isnan(rv[i]):
                    continue
                sigma = float(np.clip(rv[i] * IV_MULT, 0.08, 1.2))
                end = min(day_end, i + MAX_HOLD_MIN)
                outs = [simulate_trade(close, mins, i, end, sigma, a) for a in TRADE_ACTIONS]
                if any(o is None for o in outs):
                    continue
                X.append(f)
                R.append([o[0] for o in outs])
                EXIT_TS.append([df["ts"].iat[o[1]].value for o in outs])
                ENTRY.append([o[2] for o in outs])
                TS.append(df["ts"].iat[i].value)
                SYM.append(symbol)
            done += 1
            if done % 20 == 0 or done == days_total:
                rate = (time.time() - t0) / done
                write_status(phase="simulating trades", symbol=symbol, epoch=done, epochs=days_total,
                             seconds_left=round(rate * (days_total - done)))
    order = np.argsort(TS)
    return {"X": np.array(X)[order], "R": np.array(R)[order], "ts": np.array(TS)[order],
            "sym": np.array(SYM)[order], "exit_ts": np.array(EXIT_TS)[order], "entry": np.array(ENTRY)[order]}


# ---- 4. models + honest walk-forward test ---------------------------------------------------
def fit_models(X: np.ndarray, R: np.ndarray, weights: np.ndarray | None = None) -> list:
    from sklearn.ensemble import HistGradientBoostingRegressor
    models = []
    for a in range(R.shape[1]):
        mdl = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
                                            min_samples_leaf=150, l2_regularization=1.0, random_state=0)
        mdl.fit(X, R[:, a], sample_weight=weights)
        models.append(mdl)
    return models


def predict(models: list, X: np.ndarray) -> np.ndarray:
    return np.column_stack([m.predict(X) for m in models])


def trade_month(data: dict, rows: np.ndarray, choose) -> dict:
    """Replay one month under the live daily rules. choose(row) -> action index (0..3) or None."""
    days = {}
    for r in rows:
        days.setdefault(pd.Timestamp(data["ts"][r], tz="UTC").tz_convert(NY).date(), []).append(r)
    pnl_days, n_trades, n_wins = [], 0, 0
    for _, rs in sorted(days.items()):
        busy_until, trades, wins, spent, pnl = 0, 0, 0, 0.0, 0.0
        for r in rs:
            if data["ts"][r] < busy_until or trades >= MAX_TRADES or wins >= WIN_TARGET:
                continue
            a = choose(r)
            if a is None:
                continue
            entry = data["entry"][r, a]
            qty = int(min(PER_TRADE, DAILY_BUDGET - spent) // (entry * 100))
            if qty < 1:
                continue
            ret = data["R"][r, a]
            spent += qty * entry * 100
            pnl += qty * entry * 100 * ret
            trades += 1
            wins += ret > 0
            busy_until = data["exit_ts"][r, a]
        pnl_days.append(pnl)
        n_trades += trades
        n_wins += wins
    return {"pnl": float(sum(pnl_days)), "days": len(pnl_days), "green_days": int(sum(p > 0 for p in pnl_days)),
            "trades": n_trades, "wins": n_wins}


def walk_forward(data: dict) -> dict:
    month = np.array([pd.Timestamp(t, tz="UTC").tz_convert(NY).strftime("%Y-%m") for t in data["ts"]])
    months = sorted(set(month))
    folds = months[MIN_TRAIN_MONTHS:]
    write_status(phase="walk-forward test", total=len(folds), index=0, results=[], epoch=None, seconds_left=None)
    rng = random.Random(0)
    model_tot = {"pnl": 0.0, "days": 0, "green_days": 0, "trades": 0, "wins": 0}
    rand_tot = dict(model_tot)
    t0 = time.time()
    for k, test_m in enumerate(folds, 1):
        train = month < test_m
        test_rows = np.flatnonzero(month == test_m)
        models = fit_models(data["X"][train], data["R"][train])
        preds = predict(models, data["X"][test_rows])
        pred_of = dict(zip(test_rows, preds))

        def model_choice(r):
            p = pred_of[r]
            a = int(np.argmax(p))
            return a if p[a] >= MIN_EDGE else None

        def random_choice(r):
            return rng.randrange(4) if rng.random() < 0.3 else None

        mres, rres = trade_month(data, test_rows, model_choice), trade_month(data, test_rows, random_choice)
        for tot, res in ((model_tot, mres), (rand_tot, rres)):
            for key in tot:
                tot[key] += res[key]
        STATUS["results"].append({"symbol": test_m, "test_excess": round(mres["pnl"] / DAILY_BUDGET, 4),
                                  "display": f"{'+' if mres['pnl'] >= 0 else '-'}${abs(mres['pnl']):,.0f}",
                                  "trades": mres["trades"], "random_pnl": round(rres["pnl"], 2)})
        rate = (time.time() - t0) / k
        write_status(index=k, symbol=test_m, seconds_left=round(rate * (len(folds) - k)))
    return {"model": model_tot, "random": rand_tot, "months_tested": len(folds),
            "first_test_month": folds[0] if folds else None, "last_test_month": folds[-1] if folds else None}


# ---- 5. main -----------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-train the options bot on history")
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--symbols", default=os.environ.get("OPT_SYMBOLS", "SPY,QQQ"))
    ap.add_argument("--synthetic", action="store_true", help="fake data, no Alpaca (testing)")
    args = ap.parse_args()
    symbols = args.symbols.replace(" ", "").split(",")
    write_status(started=dt.datetime.now(NY).isoformat(timespec="seconds"), phase="downloading bars")
    try:
        frames = {}
        months = month_list(args.months)
        for s_i, symbol in enumerate(symbols):
            if args.synthetic:
                raw = synthetic_bars(symbol, args.months)
            else:
                parts = []
                for m_i, (y, m) in enumerate(months, 1):
                    write_status(symbol=symbol, epoch=s_i * len(months) + m_i, epochs=len(symbols) * len(months))
                    parts.append(download_month(symbol, y, m))
                raw = pd.concat(parts, ignore_index=True)
            frames[symbol] = add_indicators(raw, symbol)
        write_status(phase="simulating trades", epoch=0)
        data = build_dataset(frames)
        if len(data["X"]) < 1000:
            raise RuntimeError(f"only {len(data['X'])} simulated decisions; need more history")
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(DATASET_PATH, **data)
        wf = walk_forward(data)

        write_status(phase="training final model", symbol=None, seconds_left=None)
        import joblib
        models = fit_models(data["X"], data["R"])
        first = pd.Timestamp(data["ts"][0], tz="UTC").tz_convert(NY).date()
        last = pd.Timestamp(data["ts"][-1], tz="UTC").tz_convert(NY).date()
        bundle = {"models": models, "features": FEATURES, "actions": ACTIONS, "min_edge": MIN_EDGE,
                  "trained_from": str(first), "trained_to": str(last), "decisions": int(len(data["X"])),
                  "walk_forward": wf, "synthetic": args.synthetic,
                  "created": dt.datetime.now(NY).isoformat(timespec="seconds")}
        joblib.dump(bundle, MODEL_PATH)

        m, r = wf["model"], wf["random"]
        usd = lambda v: f"{'-' if v < 0 else '+'}${abs(v):,.0f}"
        summary = {"beat_buy_and_hold": m["green_days"], "stocks": m["days"], "avg_test_excess": None,
                   "text": (f"walk-forward {wf['first_test_month']}→{wf['last_test_month']}: "
                            f"model {usd(m['pnl'])} over {m['trades']} trades ({m['wins']} winners, "
                            f"{m['green_days']}/{m['days']} green days) vs random {usd(r['pnl'])}")}
        write_status(phase="done", summary=summary, symbol=None, seconds_left=None, epoch=None)
        print(summary["text"])
        print(f"model saved: {MODEL_PATH} ({len(data['X'])} simulated decisions, {first} to {last})")
        return 0
    except BaseException as exc:
        write_status(phase="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
