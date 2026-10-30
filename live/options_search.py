"""Nightly strategy search: keeps trying to improve the options bot, honestly.

Each trial picks a variation (signals, 1- or 5-minute bars, stop/target/hold,
1DTE/0DTE, time window, model or take-every-signal, how picky) and grades it
with walk-forward tests on the SEARCH period. The last HOLDOUT_MONTHS are
locked away: the search never sees them. Once a night, the best new
candidate gets one look at the holdout; it is promoted to the live bot only
if it makes money there AND beats the current champion. That keeps lucky
flukes (inevitable after hundreds of tries) from reaching the bot.

  .venv/bin/python live/options_search.py --until 06:00          # run until 6:00 NY time
  .venv/bin/python live/options_search.py --trials 5 --synthetic # quick offline test
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, SIGNALS, add_indicators, attach_daily_vol, daily_vol_table, resample  # noqa: E402
from options_pretrain import (DAILY_BUDGET, DATA_DIR, EMBARGO_DAYS, MODEL_PATH, NY, STATE_DIR,  # noqa: E402
                              build_dataset, download_month, fit_models, month_list, predict,
                              synthetic_bars, trade_month)
from strategy import SIGNAL_SETS, SPACE, VOL_SPLIT, WINDOWS, allowed_actions, dataset_key, describe  # noqa: E402

SEARCH_DIR = STATE_DIR / "search"
CACHE_DIR = DATA_DIR.parent / "search_cache"
LEADERBOARD = SEARCH_DIR / "leaderboard.json"
CHAMPION = SEARCH_DIR / "champion.json"
STATUS_PATH = STATE_DIR / "search.json"
HOLDOUT_MONTHS = int(os.environ.get("OPT_HOLDOUT_MONTHS", "6"))
MIN_TRAIN_MONTHS = 6
MIN_TRADES = int(os.environ.get("OPT_SEARCH_MIN_TRADES", "40"))
COL = {name: FEATURES.index(name) for name in FEATURES}


def load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def notify(text: str) -> None:
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        return
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"https://ntfy.sh/{topic}", data=text.encode(), headers={"Title": "Strategy search"}), timeout=10)
    except Exception:
        pass


# ---- data ------------------------------------------------------------------------------
def load_raw(months: int, symbols: list[str], synthetic: bool) -> dict:
    raw = {}
    for s in symbols:
        if synthetic:
            raw[s] = synthetic_bars(s, months)
        else:
            raw[s] = pd.concat([download_month(s, y, m) for y, m in month_list(months)], ignore_index=True)
    return raw


def dataset_for(p: dict, raw: dict, memo: dict) -> dict:
    key = dataset_key(p)
    if key in memo:
        return memo[key]
    path = CACHE_DIR / f"{key}.npz"
    if path.exists():
        data = dict(np.load(path))
    else:
        frames = {s: attach_daily_vol(add_indicators(resample(df, p["bar_minutes"]), s), daily_vol_table(df))
                  for s, df in raw.items()}
        data = build_dataset(frames, p["stop_loss"], p["take_profit"], p["max_hold"], progress=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data)
    memo.clear()                     # keep memory small on a Pi: one dataset at a time
    memo[key] = data
    return data


def qualify(data: dict, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Rows where the chosen signals fire inside the chosen window, and their direction."""
    X = data["X"]
    total = sum(X[:, COL[s]] for s in p["signals"])
    direction = np.sign(total).astype(int)
    lo, hi = WINDOWS[p["window"]]
    m = X[:, COL["minutes_since_open"]]
    ok = (direction != 0) & (m >= lo) & (m <= hi)
    vf = p.get("vol_filter", "off")
    if vf not in (None, "off"):
        vr = X[:, COL["vol_ratio"]]
        ok &= (vr <= VOL_SPLIT) if vf == "calm" else (vr > VOL_SPLIT)
    return ok, direction


def fit_with_confidence(X: np.ndarray, R: np.ndarray, confidence) -> tuple[list, np.ndarray | None]:
    """Split-conformal prediction (what MAPIE implements): fit on the older 80% of the
    rows, measure how far real outcomes fell below the predictions on the newest 20%,
    and keep that shortfall at the chosen confidence. lower bound = prediction + shortfall."""
    if confidence in (None, "off") or len(X) < 500:
        return fit_models(X, R), None
    cut = int(len(X) * 0.8)
    models = fit_models(X[:cut], R[:cut])
    resid = R[cut:] - predict(models, X[cut:])
    q_low = np.quantile(resid, 1 - float(confidence), axis=0)
    return models, q_low


# ---- grading ---------------------------------------------------------------------------
RECENT_TRADES: list = []        # sample of simulated trades for the app's live stream


def run_folds(data: dict, p: dict, folds: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    ok, direction = qualify(data, p)
    tot = {"pnl": 0.0, "days": 0, "green_days": 0, "trades": 0, "wins": 0}
    by_month: dict = {}
    names = ["call 1DTE", "put 1DTE", "call 0DTE", "put 0DTE"]
    simulated = 0
    for train_mask, test_rows in folds:
        test_rows = test_rows[ok[test_rows]]
        if len(test_rows) == 0:
            continue
        preds, q_low = None, None
        if p["use_model"]:
            tr = train_mask & ok
            if tr.sum() < 300:
                continue                               # not enough history yet for a model
            models, q_low = fit_with_confidence(data["X"][tr], data["R"][tr], p.get("confidence", "off"))
            preds = dict(zip(test_rows, predict(models, data["X"][test_rows])))

        def choose(r, preds=preds, q_low=q_low):
            options = allowed_actions(direction[r], p["dte"])
            if not options:
                return None
            if preds is None:
                return options[0]
            a = max(options, key=lambda x: preds[r][x])
            if preds[r][a] < p["min_edge"]:
                return None
            if q_low is not None and preds[r][a] + q_low[a] < 0:
                return None                            # not sure enough it won't lose
            return a

        res = trade_month(data, test_rows, choose)
        simulated += int(len(test_rows)) * 4               # every qualifying moment, all 4 trades simulated
        for k in tot:
            tot[k] += res[k]
        for r, a, dollars, ret in res["taken"]:
            t = pd.Timestamp(data["ts"][r], tz="UTC").tz_convert(NY)
            by_month[t.strftime("%Y-%m")] = by_month.get(t.strftime("%Y-%m"), 0.0) + dollars
            RECENT_TRADES.append({"t": t.isoformat(timespec="minutes"), "sym": str(data["sym"][r]),
                                  "trade": names[a], "pnl": round(float(dollars), 2)})
    del RECENT_TRADES[:-400]
    tot["months"] = [[m, round(v, 2)] for m, v in sorted(by_month.items())]
    tot["simulated"] = simulated
    return tot


def periods(data: dict):
    ts = pd.to_datetime(data["ts"], utc=True).tz_convert(NY)
    month = np.array(ts.strftime("%Y-%m"))
    months = sorted(set(month))
    return ts, month, months


def search_folds(data: dict):
    """Quarterly walk-forward over the search period (everything before the holdout)."""
    ts, month, months = periods(data)
    search_months = months[:-HOLDOUT_MONTHS]
    folds = []
    for start in range(MIN_TRAIN_MONTHS, len(search_months), 3):
        block = search_months[start:start + 3]
        t0 = pd.Timestamp(block[0] + "-01", tz=NY)
        train = np.asarray(ts < t0 - pd.Timedelta(days=EMBARGO_DAYS))
        folds.append((train, np.flatnonzero(np.isin(month, block))))
    return folds, (search_months[MIN_TRAIN_MONTHS] if len(search_months) > MIN_TRAIN_MONTHS else None,
                   search_months[-1] if search_months else None)


def holdout_fold(data: dict):
    ts, month, months = periods(data)
    hold = months[-HOLDOUT_MONTHS:]
    t0 = pd.Timestamp(hold[0] + "-01", tz=NY)
    train = np.asarray(ts < t0 - pd.Timedelta(days=EMBARGO_DAYS))
    return [(train, np.flatnonzero(np.isin(month, hold)))], (hold[0], hold[-1])


# ---- the loop ----------------------------------------------------------------------------
def sample(rng: random.Random) -> dict:
    p = {k: rng.choice(v) for k, v in SPACE.items()}
    p["signals"] = rng.choice(SIGNAL_SETS)
    if not p["use_model"]:
        p["min_edge"] = None
        p["confidence"] = "off"
    return p


def key_of(p: dict) -> str:
    return json.dumps(p, sort_keys=True)


def summarize(entry: dict) -> dict:
    return {"desc": entry["desc"], "search_pnl": entry["search"]["pnl"], "trades": entry["search"]["trades"],
            "wins": entry["search"]["wins"], "green_days": entry["search"]["green_days"],
            "days": entry["search"]["days"], "holdout_pnl": entry.get("holdout", {}).get("pnl")}


def write_status(**info) -> None:
    info["updated"] = dt.datetime.now(NY).isoformat(timespec="seconds")
    save_json(STATUS_PATH, info)


def promote(entry: dict, raw: dict, memo: dict, champion: dict) -> None:
    import joblib
    p = entry["params"]
    data = dataset_for(p, raw, memo)
    ok, _ = qualify(data, p)
    models, q_low = (fit_with_confidence(data["X"][ok], data["R"][ok], p.get("confidence", "off"))
                     if p["use_model"] else (None, None))
    bundle = {"models": models, "features": FEATURES, "strategy": p, "signal_mode": True, "version": 3,
              "min_edge": p["min_edge"] if p["use_model"] else 0.0, "dataset_key": dataset_key(p),
              "dataset_path": str(CACHE_DIR / f"{dataset_key(p)}.npz"),
              "decisions": int(ok.sum()), "description": entry["desc"],
              "conformal_q_low": q_low.tolist() if q_low is not None else None,
              "trained_from": str(pd.Timestamp(data["ts"][0], tz="UTC").tz_convert(NY).date()),
              "trained_to": str(pd.Timestamp(data["ts"][-1], tz="UTC").tz_convert(NY).date()),
              "walk_forward": {"model": entry["holdout"], "search": entry["search"]},
              "created": dt.datetime.now(NY).isoformat(timespec="seconds")}
    if MODEL_PATH.exists():
        MODEL_PATH.replace(MODEL_PATH.with_name("options_model.prev.joblib"))
    joblib.dump(bundle, MODEL_PATH)
    champion.update(params=p, desc=entry["desc"], search_pnl=entry["search"]["pnl"],
                    holdout_pnl=entry["holdout"]["pnl"], promoted=bundle["created"])
    save_json(CHAMPION, champion)
    notify(f"New champion promoted to the options bot: {entry['desc']} "
           f"(holdout {entry['holdout']['pnl']:+,.0f} over {entry['holdout']['trades']} trades)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Nightly options strategy search")
    ap.add_argument("--until", help="stop at this New York time, HH:MM (next occurrence)")
    ap.add_argument("--trials", type=int, default=10_000)
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--symbols", default=os.environ.get("OPT_SYMBOLS", "SPY,QQQ"))
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    now = dt.datetime.now(NY)
    deadline = None
    if args.until:
        h, m = map(int, args.until.split(":"))
        deadline = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if deadline <= now:
            deadline += dt.timedelta(days=1)

    board = load_json(LEADERBOARD, [])
    champion = load_json(CHAMPION, {"params": None, "desc": "none yet (bot keeps its current model)",
                                    "search_pnl": 0.0, "holdout_pnl": 0.0, "holdout_checks": 0})
    tried = {key_of(e["params"]) for e in board}
    rng = random.Random(time.time())
    started = now.isoformat(timespec="seconds")
    tonight = 0

    def status(phase, current=None):
        ranked = sorted((e for e in board if e["search"]["trades"] >= MIN_TRADES),
                        key=lambda e: e["search"]["pnl"], reverse=True)
        write_status(phase=phase, started=started, trials_tonight=tonight, trials_total=len(board),
                     simulated_total=sum(e["search"].get("simulated", 0) for e in board),
                     sim_trades_taken=sum(e["search"]["trades"] for e in board),
                     sim_wins=sum(e["search"]["wins"] for e in board),
                     profitable=sum(e["search"]["pnl"] > 0 and e["search"]["trades"] >= MIN_TRADES for e in board),
                     deadline=deadline.isoformat(timespec="minutes") if deadline else None,
                     current=current, best=[summarize(e) for e in ranked[:5]],
                     champion=champion, holdout_months=HOLDOUT_MONTHS)

    status("loading price history")
    raw = load_raw(args.months, args.symbols.replace(" ", "").split(","), args.synthetic)
    memo: dict = {}
    try:
        while tonight < args.trials and (deadline is None or dt.datetime.now(NY) < deadline - dt.timedelta(minutes=20)):
            p = sample(rng)
            k = key_of(p)
            if k in tried:
                continue
            tried.add(k)
            status("testing", describe(p))
            data = dataset_for(p, raw, memo)
            folds, (f0, f1) = search_folds(data)
            res = run_folds(data, p, folds)
            board.append({"params": p, "desc": describe(p), "search": res, "search_period": [f0, f1],
                          "tested": dt.datetime.now(NY).isoformat(timespec="seconds")})
            save_json(LEADERBOARD, board)
            save_json(SEARCH_DIR / "recent_trades.json",
                      {"variation": describe(p), "trades": RECENT_TRADES[-300:]})
            tonight += 1

        # One holdout look per night: best untested candidate that clears the bar.
        status("checking the best candidate on the locked holdout")
        candidates = sorted((e for e in board if "holdout" not in e and e["search"]["trades"] >= MIN_TRADES
                             and e["search"]["pnl"] > max(0.0, champion.get("search_pnl") or 0.0)),
                            key=lambda e: e["search"]["pnl"], reverse=True)
        if candidates:
            best = candidates[0]
            data = dataset_for(best["params"], raw, memo)
            folds, (h0, h1) = holdout_fold(data)
            best["holdout"] = run_folds(data, best["params"], folds)
            best["holdout_period"] = [h0, h1]
            champion["holdout_checks"] = champion.get("holdout_checks", 0) + 1
            save_json(LEADERBOARD, board)
            if best["holdout"]["pnl"] > max(0.0, champion.get("holdout_pnl") or 0.0) and \
                    best["holdout"]["trades"] >= MIN_TRADES // 3:
                status("promoting a new champion")
                promote(best, raw, memo, champion)
            save_json(CHAMPION, champion)
        status("idle until tomorrow night")
        return 0
    except BaseException as exc:
        status(f"failed: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
