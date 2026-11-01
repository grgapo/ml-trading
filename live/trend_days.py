"""Trend-day predictor: by 10:00 New York, will today be a big directional day?

Label (known only after the close): the move from 10:00 to 15:45 is at least half of the
average daily range of the previous 20 days (a "trend day").
Inputs (all known by 10:00): opening gap, first-30-minute range / return / volume vs recent days,
VIX (yesterday's level, 5-day change, vs its 1-year median), Fed/CPI/jobs day flags,
yesterday's range, day of week, which symbol.
Walk-forward by month: each month is predicted by models trained only on earlier months.
Then the opening-range breakout is rerun, trading only on days rated likely trend days
(fixed rule: predicted chance above the median of all earlier predictions; nothing tuned).

  .venv/bin/python live/trend_days.py --symbols SPY QQQ
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
import market_events  # noqa: E402
import orb_backtest as ORB  # noqa: E402
from options_pretrain import NY  # noqa: E402

OUT = Path(os.environ.get("STATE_DIR", Path.home() / "projects" / "paper-state" / "orb")) / "trend_days.json"
TREND = 0.5            # a trend day moves at least half the 20-day average daily range, 10:00 -> 15:45
FIRST = 30             # minutes 9:30-9:59 are the "first 30 minutes"


def build(sym: str, days: list[dict], vix: pd.Series | None) -> pd.DataFrame:
    rows = []
    rng = np.array([d["h"].max() - d["l"].min() for d in days])
    r30 = np.array([d["h"][:FIRST].max() - d["l"][:FIRST].min() for d in days])
    v30 = np.array([d["v"][:FIRST].sum() for d in days])
    for i in range(20, len(days)):
        d = days[i]; avg_rng = rng[i - 20:i].mean()
        p10 = d["c"][FIRST - 1]
        f = {"date": d["date"], "symbol": sym,
             "gap": (d["o"][0] / d["prev_close"] - 1) / (avg_rng / p10),
             "first30_range": r30[i] / r30[i - 20:i].mean(),
             "first30_move": (p10 / d["o"][0] - 1) / (avg_rng / p10),
             "first30_volume": v30[i] / max(v30[i - 20:i].mean(), 1),
             "yday_range": rng[i - 1] / avg_rng,
             "weekday": d["date"].weekday(), "is_qqq": float(sym == "QQQ"),
             **{k: v for k, v in market_events.flags(d["date"]).items() if k != "event_day"}}
        if vix is not None:
            prev = vix[vix.index.date < d["date"]]
            if len(prev) > 260:
                f["vix"] = float(prev.iloc[-1]) / 20
                f["vix_chg5"] = float(prev.iloc[-1] / prev.iloc[-6] - 1)
                f["vix_rel"] = float(prev.iloc[-1] / prev.iloc[-252:].median() - 1)
        f["move_after_10"] = abs(d["c"][ORB.FLAT] - p10) / avg_rng
        f["trend"] = int(f["move_after_10"] >= TREND)
        rows.append(f)
    return pd.DataFrame(rows)


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cols = [c for c in df.columns if c not in ("date", "symbol", "move_after_10", "trend")]
    df = df.sort_values("date").reset_index(drop=True)
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    months = sorted(df["month"].unique())
    df["p_logit"] = np.nan; df["p_trees"] = np.nan
    if len(months) < 8:
        raise SystemExit(f"need at least 8 months of history for the walk-forward (have {len(months)})")
    for m in months[6:]:                       # first 6 months only train
        tr, te = df[df["month"] < m], df[df["month"] == m]
        X, y = tr[cols].fillna(tr[cols].median()), tr["trend"]
        Xt = te[cols].fillna(tr[cols].median())
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(X, y)
        gb = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=30).fit(X, y)
        df.loc[te.index, "p_logit"] = lr.predict_proba(Xt)[:, 1]
        df.loc[te.index, "p_trees"] = gb.predict_proba(Xt)[:, 1]
    df.attrs["cols"] = cols
    df.attrs["coefs"] = dict(zip(cols, np.round(lr.named_steps["logisticregression"].coef_[0], 3)))
    return df


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"]); args = ap.parse_args()
    vix = features.vix_daily()
    data, frames = {}, []
    for sym in args.symbols:
        days = ORB.load_days(sym)
        data[sym] = days
        frames.append(build(sym, days, vix))
        print(f"{sym}: {len(days)} days", flush=True)
    df = walk_forward(pd.concat(frames, ignore_index=True))
    te = df.dropna(subset=["p_logit"])
    base = te["trend"].mean()
    print(f"\n================ trend-day predictor (walk-forward, {len(te)} test days) ================")
    print(f"trend days: {base:.0%} of days (move 10:00->15:45 >= {TREND:.0%} of the 20-day average range)")
    for name in ("p_logit", "p_trees"):
        top = te[te[name] >= te[name].quantile(0.7)]
        low = te[te[name] <= te[name].quantile(0.3)]
        print(f"{'logistic' if name == 'p_logit' else 'trees':>9}: AUC {auc(te['trend'], te[name]):.3f} (0.5 = no skill) · "
              f"trend rate in its top 30%: {top['trend'].mean():.0%} · in its bottom 30%: {low['trend'].mean():.0%}")
    print("what pushes toward a trend day (logistic weights, latest model):")
    print("   " + " · ".join(f"{k} {v:+.2f}" for k, v in sorted(df.attrs["coefs"].items(), key=lambda x: -abs(x[1]))))

    # ---- payoff: the opening-range breakout, only on predicted trend days ----------------------------
    p = {"band": 0.3, "trigger": 4.0, "giveback": 0.5, "stop": 10.0, "spread": ORB.SPREAD, "vwap_exit": False}
    client = None
    results = {}
    for sym, days in data.items():
        sessions = [d["date"] for d in days]; cache = {}

        def series(d, expiry, strike, side, sym=sym, cache=cache):
            nonlocal client
            if d["date"] not in cache:
                try:
                    if client is None and d["date"] >= ORB.OH.FIRST_DAY:
                        client = ORB.OH._client()
                    cache[d["date"]] = ORB.OH.day_series(ORB.OH.fetch_day(sym, d["date"], expiry, float(np.nanmin(d["l"])), float(np.nanmax(d["h"])), client))
                except Exception as exc:
                    print(f"no option prices for {sym} {d['date']}: {exc}"); cache[d["date"]] = {}
            return cache[d["date"]].get(ORB.OH.occ(sym, expiry, side == "call", strike))
        trades = [t for t in ORB.run(days, sym, series, sessions, p) if t["pnl"] is not None]
        pred = te[te["symbol"] == sym].set_index(te[te["symbol"] == sym]["date"].astype(str))
        # fixed rule: trade only if today's predicted chance beats the median of all EARLIER predictions
        hist = te.sort_values("date")[["date", "p_logit"]].reset_index(drop=True)
        thr = {str(r.date): float(hist["p_logit"].iloc[:k].median()) if k >= 20 else np.nan for k, r in enumerate(hist.itertuples())}
        keep, skip = [], []
        for t in trades:
            if t["date"] not in pred.index or not np.isfinite(thr.get(t["date"], np.nan)):
                continue
            (keep if float(pred.loc[t["date"], "p_logit"]) > thr[t["date"]] else skip).append(t)
        results[sym] = {"all": ORB.summary([t for t in trades if t["date"] in pred.index]), "trend_days": ORB.summary(keep),
                        "skipped": ORB.summary(skip)}
    print("\n================ opening-range breakout, filtered by the predictor (same test months) ================")
    for sym, r in results.items():
        for k, label in (("all", "every signal"), ("trend_days", "predicted trend days only"), ("skipped", "the days it skipped")):
            s = r[k]
            if s.get("trades"):
                print(f"{sym} · {label:26s}: {s['trades']:>3} trades · win {s['win_rate']:.0%} · total ${s['total']:+,.0f} · "
                      f"profit factor {s['profit_factor']} · worst drawdown ${s['max_drawdown']:,.0f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"updated": dt.datetime.now(NY).isoformat(timespec="seconds"), "trend_rate": base,
                               "auc_logit": auc(te["trend"], te["p_logit"]), "auc_trees": auc(te["trend"], te["p_trees"]),
                               "weights": df.attrs["coefs"], "orb": results}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
