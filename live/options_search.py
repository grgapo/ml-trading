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
from strategy import SIGNAL_SETS, SPACE, VOL_SPLIT, WINDOWS, allowed_actions, dataset_key, describe, needs_live_support  # noqa: E402

SEARCH_DIR = STATE_DIR / "search"
CACHE_DIR = DATA_DIR.parent / "search_cache"               # + one subfolder per symbol set (per bot)
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
SYNTHETIC = False


def load_raw(months: int, symbols: list[str], synthetic: bool) -> dict:
    raw = {}
    for s in symbols:
        if synthetic:
            raw[s] = synthetic_bars(s, months)
        else:
            raw[s] = pd.concat([download_month(s, y, m) for y, m in month_list(months)], ignore_index=True)
    return raw


def cache_path(p: dict, symbols) -> Path:
    return CACHE_DIR / "-".join(sorted(symbols)) / f"{dataset_key(p)}.npz"


def dataset_for(p: dict, raw: dict, memo: dict) -> dict:
    key = dataset_key(p)
    if key in memo:
        return memo[key]
    path = cache_path(p, raw)
    if path.exists():
        data = dict(np.load(path))
    else:
        frames = {s: attach_daily_vol(add_indicators(resample(df, p["bar_minutes"]), s), daily_vol_table(df))
                  for s, df in raw.items()}
        data = build_dataset(frames, p["stop_loss"], p["take_profit"], p["max_hold"], progress=False,
                             real=not SYNTHETIC, exit_style=p.get("exit_style", "option"),
                             moneyness=p.get("moneyness", "atm"))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data)
    # each practice trade's day, rated calm (1) / trend (0) by the walk-forward calm-day predictor
    days = pd.to_datetime(data["ts"], utc=True).tz_convert(NY).date
    data["calm"] = np.array([CALM.get((str(sym), d), np.nan) for sym, d in zip(data["sym"], days)], dtype=float)
    memo.clear()                     # keep memory small on a Pi: one dataset at a time
    memo[key] = data
    return data


CALM: dict = {}


def load_calm_days(symbols: list[str]) -> None:
    """Rate every past day calm/trend with the trend-day predictor, walk-forward (each month predicted
    by a model trained only on earlier months); calm = predicted trend chance at or below the median
    of all earlier predictions (the proving ground's rule). Days it can't rate stay unknown."""
    try:
        import features as F
        import orb_backtest as ORB
        import trend_days as TD
        vix = F.vix_daily()
        df = TD.walk_forward(pd.concat([TD.build(s, ORB.load_days(s), vix) for s in symbols], ignore_index=True))
        te = df.dropna(subset=["p_logit"]).sort_values("date").reset_index(drop=True)
        for k, r in te.iterrows():
            if k < 20:
                continue
            thr = float(te["p_logit"].iloc[:k].median())
            CALM[(r["symbol"], r["date"])] = 1.0 if r["p_logit"] <= thr else 0.0
        print(f"calm-day ratings for {len(CALM)} symbol-days ({sum(CALM.values()) / max(len(CALM), 1):.0%} calm)", flush=True)
    except Exception as exc:
        print(f"calm-day ratings unavailable ({type(exc).__name__}: {exc}); the calm-day filter will skip", flush=True)


def qualify(data: dict, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Rows where the chosen signals fire inside the chosen window, and their direction."""
    X = data["X"]
    total = sum(X[:, COL[s]] for s in p["signals"])
    direction = np.sign(total).astype(int)
    lo, hi = WINDOWS[p["window"]]
    m = X[:, COL["minutes_since_open"]]
    ok = (direction != 0) & (m >= lo) & (m <= hi)
    need = p.get("confluence") or 1
    if need > 1:
        agree = np.where(direction > 0, X[:, COL["conf_bull"]], X[:, COL["conf_bear"]])
        ok &= agree >= need
    vf = p.get("vol_filter", "off")
    if vf not in (None, "off"):
        vr = X[:, COL["vol_ratio"]]
        ok &= (vr <= VOL_SPLIT) if vf == "calm" else (vr > VOL_SPLIT)
    ef = p.get("event_filter", "off")
    if ef not in (None, "off") and "fomc_day" in COL:
        ev = (X[:, COL["fomc_day"]] + X[:, COL["cpi_day"]] + X[:, COL["jobs_day"]]) > 0
        ok &= ~ev if ef == "skip" else ev
    xf = p.get("vix_filter", "off")
    if xf not in (None, "off") and "vix_rel" in COL:
        vx = X[:, COL["vix_rel"]]
        known = ~np.isnan(vx)
        ok &= ~known | ((vx <= 0) if xf == "low" else (vx > 0))
    dfl = p.get("day_filter", "off")
    if dfl not in (None, "off"):
        calm = data.get("calm")
        if calm is None:
            ok &= False
        else:
            ok &= (calm == 1) if dfl == "calm" else (calm == 0)
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


def run_folds(data: dict, p: dict, folds: list[tuple[np.ndarray, np.ndarray]], keep_trades: bool = False) -> dict:
    ok, direction = qualify(data, p)
    tot = {"pnl": 0.0, "days": 0, "green_days": 0, "trades": 0, "wins": 0}
    per_trade: list[float] = []
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
            per_trade.append(float(dollars))
            t = pd.Timestamp(data["ts"][r], tz="UTC").tz_convert(NY)
            by_month[t.strftime("%Y-%m")] = by_month.get(t.strftime("%Y-%m"), 0.0) + dollars
            RECENT_TRADES.append({"t": t.isoformat(timespec="minutes"), "sym": str(data["sym"][r]),
                                  "trade": names[a], "pnl": round(float(dollars), 2)})
    del RECENT_TRADES[:-400]
    tot["months"] = [[m, round(v, 2)] for m, v in sorted(by_month.items())]
    tot["simulated"] = simulated
    if keep_trades and len(per_trade) > 1:
        x = np.array(per_trade)
        tot["mean"] = round(float(x.mean()), 2)
        tot["se"] = round(float(x.std(ddof=1) / np.sqrt(len(x))), 2)
        tot["t"] = round(float(x.mean() / max(x.std(ddof=1) / np.sqrt(len(x)), 1e-9)), 2)
    return tot


# The promotion bar: a locked-test result has to be clearly positive, not just above zero.
MIN_HOLDOUT_TRADES = int(os.environ.get("OPT_MIN_HOLDOUT_TRADES", "40"))
MIN_EDGE_PER_TRADE = float(os.environ.get("OPT_MIN_EDGE_PER_TRADE", "10"))     # dollars per trade
MIN_T = float(os.environ.get("OPT_MIN_T", "2.0"))                               # profit > 2x its noise


def convincing(h: dict) -> tuple[bool, str]:
    n, mean, t = h.get("trades", 0), h.get("mean"), h.get("t")
    if n < MIN_HOLDOUT_TRADES:
        return False, f"only {n} trades (needs {MIN_HOLDOUT_TRADES}+)"
    if mean is None or mean < MIN_EDGE_PER_TRADE:
        return False, f"avg {mean if mean is not None else 0:+.2f}/trade (needs +{MIN_EDGE_PER_TRADE:.0f}+)"
    if t is None or t < MIN_T:
        return False, f"profit is {t if t is not None else 0:.1f}x its noise (needs {MIN_T:.1f}x+)"
    return True, f"avg {mean:+.2f}/trade over {n} trades, {t:.1f}x its noise"


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
    if p["exit_style"] == "atr":                    # the quick-exit settings don't apply
        p["stop_loss"] = p["take_profit"] = p["max_hold"] = None
    return p


def key_of(p: dict) -> str:
    return json.dumps(p, sort_keys=True)


# Robustness: a real edge survives small changes to its settings; a fluke collapses.
ROBUST_TOP = int(os.environ.get("OPT_ROBUST_TOP", "20"))           # candidates checked each night
ROBUST_MAX_NEIGHBORS = int(os.environ.get("OPT_ROBUST_NEIGHBORS", "12"))
ROBUST_MIN_SHARE = float(os.environ.get("OPT_ROBUST_MIN_SHARE", "0.6"))   # of neighbors that must be profitable
ROBUST_RESERVE_MIN = int(os.environ.get("OPT_ROBUST_RESERVE_MIN", "75"))  # minutes kept for this before the deadline
NEIGHBOR_KEYS = ["bar_minutes", "stop_loss", "take_profit", "max_hold", "dte", "window", "min_edge", "vol_filter",
                 "confidence", "confluence", "event_filter", "vix_filter", "moneyness", "day_filter"]


def neighbors(p: dict, rng: random.Random) -> list[dict]:
    """Variations one step away on a single setting (the next value up or down in its list)."""
    out = []
    for k in NEIGHBOR_KEYS:
        vals = SPACE.get(k)
        if not vals or p.get(k) is None or p.get(k) not in vals:
            continue                                     # e.g. quick-exit settings under ATR exits
        if k == "min_edge" and not p.get("use_model"):
            continue
        if k == "confidence" and not p.get("use_model"):
            continue
        i = vals.index(p[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                out.append({**p, k: vals[j]})
    rng.shuffle(out)
    return out[:ROBUST_MAX_NEIGHBORS]


def summarize(entry: dict) -> dict:
    return {"desc": entry["desc"], "search_pnl": entry["search"]["pnl"], "trades": entry["search"]["trades"],
            "wins": entry["search"]["wins"], "green_days": entry["search"]["green_days"],
            "days": entry["search"]["days"], "holdout_pnl": entry.get("holdout", {}).get("pnl")}


def write_status(**info) -> None:
    import socket
    info["updated"] = dt.datetime.now(NY).isoformat(timespec="seconds")
    info["host"] = socket.gethostname()            # which machine is searching (lab03, lab01, ...)
    save_json(STATUS_PATH, info)


# ---- champion monitoring: demote a live champion that stops working ------------------------
MIN_LIVE_TRADES = int(os.environ.get("OPT_MIN_LIVE_TRADES", "30"))
MAX_LIVE_LOSS = float(os.environ.get("OPT_MAX_LIVE_LOSS", str(3 * DAILY_BUDGET)))


def live_record(champion: dict) -> dict | None:
    """Real paper trades taken since this champion was promoted."""
    import csv
    path = STATE_DIR / "options_trades.csv"
    since = champion.get("promoted")
    if not champion.get("params") or not since or not path.exists():
        return None
    rows = [r for r in csv.DictReader(path.open()) if r.get("opened", "") >= since]
    pnl = [float(r["pl_usd"]) for r in rows if r.get("pl_usd")]
    n = len(pnl)
    mean = sum(pnl) / n if n else 0.0
    sd = (sum((x - mean) ** 2 for x in pnl) / (n - 1)) ** 0.5 if n > 1 else 0.0
    expected = (champion.get("holdout_pnl") or 0.0) / max(1, champion.get("holdout_trades") or 1)
    return {"trades": n, "pnl": round(sum(pnl), 2), "mean": round(mean, 2), "sd": round(sd, 2),
            "expected_mean": round(expected, 2), "wins": sum(x > 0 for x in pnl),
            "checked": dt.datetime.now(NY).isoformat(timespec="seconds")}


def demote(champion: dict, reason: str) -> None:
    prev = MODEL_PATH.with_name("options_model.prev.joblib")
    if prev.exists():
        prev.replace(MODEL_PATH)
        fallback = "previous model restored"
    elif MODEL_PATH.exists():
        MODEL_PATH.unlink()
        fallback = "no model: bot back to its learning table"
    else:
        fallback = "no model installed"
    history = champion.setdefault("history", [])
    history.append({"desc": champion.get("desc"), "promoted": champion.get("promoted"),
                    "demoted": dt.datetime.now(NY).isoformat(timespec="seconds"), "reason": reason,
                    "live": champion.get("live")})
    champion.update(params=None, desc="none (last champion demoted: " + reason + ")",
                    search_pnl=0.0, holdout_pnl=0.0, holdout_trades=0, promoted=None, live=None, regrade=[])
    save_json(CHAMPION, champion)
    notify(f"Options champion demoted: {reason}. {fallback}.")


def monitor_champion(champion: dict, raw: dict, memo: dict) -> None:
    """Nightly: live record vs expectation, plus a re-grade on the current (rolled) holdout."""
    if not champion.get("params"):
        return
    rec = live_record(champion)
    if rec:
        champion["live"] = rec
        n, mean, sd, exp = rec["trades"], rec["mean"], rec["sd"], rec["expected_mean"]
        if rec["pnl"] <= -MAX_LIVE_LOSS:
            return demote(champion, f"lost ${-rec['pnl']:,.0f} live over {n} trades")
        clearly_worse = sd == 0 or (exp - mean) > 2 * sd / (n ** 0.5)
        if n >= MIN_LIVE_TRADES and mean < 0 and clearly_worse:
            return demote(champion, f"live avg ${mean:,.0f}/trade over {n} trades, expected ${exp:,.0f}")
        rec["status"] = "under review" if (n >= MIN_LIVE_TRADES and mean < exp) else "ok" if n >= 10 else "too early to judge"
    try:
        data = dataset_for(champion["params"], raw, memo)
        folds, (h0, h1) = holdout_fold(data)
        res = run_folds(data, champion["params"], folds)
        regrade = champion.setdefault("regrade", [])
        if not regrade or regrade[-1]["to"] != h1:
            regrade.append({"from": h0, "to": h1, "pnl": round(res["pnl"], 2), "trades": res["trades"],
                            "at": dt.datetime.now(NY).isoformat(timespec="seconds")})
            del regrade[:-6]
        if len(regrade) >= 2 and regrade[-1]["pnl"] < 0 and regrade[-2]["pnl"] < 0:
            return demote(champion, f"negative on the rolled holdout twice in a row ({h0} to {h1})")
    except Exception as exc:
        champion["regrade_error"] = f"{type(exc).__name__}: {exc}"
    save_json(CHAMPION, champion)


def promote(entry: dict, raw: dict, memo: dict, champion: dict) -> None:
    import joblib
    p = entry["params"]
    data = dataset_for(p, raw, memo)
    ok, _ = qualify(data, p)
    models, q_low = (fit_with_confidence(data["X"][ok], data["R"][ok], p.get("confidence", "off"))
                     if p["use_model"] else (None, None))
    bundle = {"models": models, "features": FEATURES, "strategy": p, "signal_mode": True, "version": 3,
              "min_edge": p["min_edge"] if p["use_model"] else 0.0, "dataset_key": dataset_key(p),
              "dataset_path": str(cache_path(p, raw)),
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
                    holdout_pnl=entry["holdout"]["pnl"], holdout_trades=entry["holdout"]["trades"],
                    holdout_mean=entry["holdout"].get("mean"), holdout_t=entry["holdout"].get("t"),
                    promoted=bundle["created"], live=None, regrade=[])
    save_json(CHAMPION, champion)
    notify(f"New champion promoted to the options bot: {entry['desc']} "
           f"(holdout {entry['holdout']['pnl']:+,.0f} over {entry['holdout']['trades']} trades)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Nightly options strategy search")
    ap.add_argument("--until", help="stop at this New York time, HH:MM (next occurrence)")
    ap.add_argument("--trials", type=int, default=1_000_000)   # in practice: until the deadline
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

    global SYNTHETIC
    SYNTHETIC = args.synthetic
    status("loading price history")
    raw = load_raw(args.months, args.symbols.replace(" ", "").split(","), args.synthetic)
    if not args.synthetic:
        status("rating past days calm / trend")
        load_calm_days(args.symbols.replace(" ", "").split(","))
    memo: dict = {}
    try:
        status("checking the simulator against real fills")
        try:
            import sim_calibration
            sim_calibration.main()
        except Exception as exc:
            print(f"simulator check skipped: {exc}")
        status("checking the live champion")
        monitor_champion(champion, raw, memo)
        explore_until = deadline - dt.timedelta(minutes=20 + ROBUST_RESERVE_MIN) if deadline else None
        last_save = time.time()                         # results file is big: save once a minute, not every variation
        while tonight < args.trials and (explore_until is None or dt.datetime.now(NY) < explore_until):
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
            tonight += 1
            if time.time() - last_save >= 60:
                save_json(LEADERBOARD, board)
                save_json(SEARCH_DIR / "recent_trades.json", {"variation": describe(p), "trades": RECENT_TRADES[-300:]})
                last_save = time.time()
        save_json(LEADERBOARD, board)                       # everything explored tonight, before the robustness pass
        save_json(SEARCH_DIR / "recent_trades.json", {"variation": board[-1]["desc"] if board else "", "trades": RECENT_TRADES[-300:]})

        # Robustness pass: re-test the top candidates' close neighbours (one setting nudged at a time).
        pool = sorted((e for e in board if "holdout" not in e and e["search"]["trades"] >= MIN_TRADES
                       and e["search"]["pnl"] > max(0.0, champion.get("search_pnl") or 0.0)),
                      key=lambda e: e["search"]["pnl"], reverse=True)[:ROBUST_TOP]
        by_key = {key_of(e["params"]): e for e in board}
        for n_c, cand in enumerate(pool, 1):
            if deadline is not None and dt.datetime.now(NY) >= deadline - dt.timedelta(minutes=20):
                break
            if "robust" in cand:
                continue
            status("robustness check", f"{n_c}/{len(pool)} · {cand['desc']}")
            results = []
            for q in neighbors(cand["params"], rng):
                k = key_of(q)
                if k in by_key:                          # already tested: reuse
                    results.append(by_key[k]["search"]["pnl"]); continue
                data = dataset_for(q, raw, memo)
                folds, (f0, f1) = search_folds(data)
                res = run_folds(data, q, folds)
                e = {"params": q, "desc": describe(q), "search": res, "search_period": [f0, f1],
                     "tested": dt.datetime.now(NY).isoformat(timespec="seconds"), "neighbor_of": cand["desc"]}
                board.append(e); by_key[k] = e; tried.add(k); tonight += 1
                results.append(res["pnl"])
            if results:
                arr = np.array(results)
                cand["robust"] = {"neighbors": len(arr), "profitable_share": round(float((arr > 0).mean()), 3),
                                  "median_pnl": round(float(np.median(arr)), 2), "worst_pnl": round(float(arr.min()), 2)}
            save_json(LEADERBOARD, board)

        # One holdout look per night: the best candidate whose neighbours are mostly profitable too
        # (ranked by the median neighbour, a conservative read of its edge).
        status("checking the best candidate on the locked holdout")
        candidates = sorted((e for e in pool if e.get("robust", {}).get("profitable_share", 0) >= ROBUST_MIN_SHARE
                             and e["robust"]["median_pnl"] > 0),
                            key=lambda e: e["robust"]["median_pnl"], reverse=True)
        if candidates:
            best = candidates[0]
            data = dataset_for(best["params"], raw, memo)
            folds, (h0, h1) = holdout_fold(data)
            best["holdout"] = run_folds(data, best["params"], folds, keep_trades=True)
            best["holdout_period"] = [h0, h1]
            champion["holdout_checks"] = champion.get("holdout_checks", 0) + 1
            ok_bar, why = convincing(best["holdout"])
            best["holdout"]["verdict"] = ("passed: " if ok_bar else "not convincing: ") + why
            save_json(LEADERBOARD, board)
            # a champion promoted under the old, looser bar doesn't block a convincing newcomer
            to_beat = (champion.get("holdout_pnl") or 0.0) if champion.get("holdout_t") is not None else 0.0
            if ok_bar and best["holdout"]["pnl"] > max(0.0, to_beat):
                missing = needs_live_support(best["params"])
                if missing:                                 # passed, but the live bot can't run it yet
                    champion["pending"] = {"desc": best["desc"], "params": best["params"], "holdout_pnl": best["holdout"]["pnl"],
                                           "holdout_trades": best["holdout"]["trades"], "needs": missing,
                                           "found": dt.datetime.now(NY).isoformat(timespec="seconds")}
                    notify(f"A variation passed the locked test but the live bot needs {', '.join(missing)} before it can "
                           f"trade it: {best['desc']} (holdout {best['holdout']['pnl']:+,.0f} over {best['holdout']['trades']} trades). "
                           f"Ask Claude to add live support.")
                else:
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
