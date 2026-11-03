"""Proving ground: one paper bot that runs candidate strategies ("experiments") side by side, each
with its own budget, trade log and P&L, so new ideas get clean live data before they're allowed
anywhere near the main bots.

Lifecycle (checked after every session, for each experiment):
  incubating  -> "ready for review" once it has 30+ trades over 8+ weeks and is holding up against
                 its backtest (profit factor >= 1.2, average trade not clearly below what the
                 backtest led us to expect). You get a notification; promotion is always your call.
  incubating  -> "retired" if it clearly underperforms: 30+ trades with profit factor < 0.8 or an
                 average trade more than 2 standard errors below the backtest's, or a total loss
                 beyond $15,000. It stops trading; its record is kept.

Experiment #1  orb-calm : opening-range breakout + noise band + VWAP, traded ONLY on days the
calm-day predictor rates calm (the trend-day model, retrained every morning on all data through
yesterday; a day is calm when its predicted trend chance is at or below the median of the model's
predictions over history). One trade per symbol per day, ~$2 in the money, next-session expiry,
$5,000 per trade; after 4 ATR its way sell on giving back half; 10 ATR safety stop; out at 3:45.

  .venv/bin/python live/proving_ground.py                       # live paper session (timer 9:20 NY)
  .venv/bin/python live/proving_ground.py --replay 2026-10-15   # replay a past day on stored prices
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import broker as BR  # noqa: E402
import features  # noqa: E402
import orb_backtest as ORB  # noqa: E402
import trend_days as TD  # noqa: E402
from options_pretrain import NY  # noqa: E402

OH = ORB.OH
STATE = Path(os.environ.get("STATE_DIR", Path.home() / "projects" / "paper-state" / "proving"))
TRADES = STATE / "trades.csv"
REGISTRY = STATE / "experiments.json"
FIELDS = ["date", "experiment", "symbol", "side", "contract", "qty", "entry_time", "entry_price", "exit_time",
          "exit_price", "pnl", "pct", "exit_reason", "calm_score", "calm_threshold"]
DEFAULT_EXPERIMENTS = [{
    "id": "orb-calm", "name": "Opening range · calm days",
    "description": "Opening-range breakout + noise band + VWAP, only on days the calm-day predictor rates calm.",
    "symbols": ["SPY", "QQQ"], "stake": 2000, "status": "incubating", "started": None,
    "expectation": {"mean_per_trade": 225.0, "profit_factor": 1.8, "trades_per_month": 5.8,
                    "source": "trend_days.py walk-forward, skipped (calm) days, Nov 2024 - Oct 2026"},
}]
MIN_TRADES, MIN_WEEKS, MAX_LOSS = 30, 8, 5000.0      # experiments are retired at -$5,000 lifetime
EXPERIMENT_STAKE_CAP = 2000.0                            # hard cap per trade for any experiment


def notify(text: str) -> None:
    topic = os.environ.get("NTFY_TOPIC")
    if topic:
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{topic}", data=text.encode(),
                                                          headers={"Title": "Proving ground"}), timeout=10)
        except Exception:
            pass


def save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(data, default=str)); tmp.replace(path)


def registry() -> list[dict]:
    reg = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []
    have = {e["id"] for e in reg}
    for e in DEFAULT_EXPERIMENTS:                              # new experiments join automatically
        if e["id"] not in have:
            reg.append({**e, "started": str(dt.date.today())})
    for e in reg:                                             # the hard cap applies to every experiment
        if float(e.get("stake", 0)) > EXPERIMENT_STAKE_CAP:
            scale = EXPERIMENT_STAKE_CAP / float(e["stake"])
            e["stake"] = EXPERIMENT_STAKE_CAP
            if (e.get("expectation") or {}).get("mean_per_trade"):
                e["expectation"]["mean_per_trade"] = round(e["expectation"]["mean_per_trade"] * scale, 2)
    save(REGISTRY, reg)
    return reg


def open_file() -> Path:
    return STATE / "open_positions.json"


def remember_open(contract: str, exp_id: str, sym: str, log: bool = True) -> None:
    f = open_file(); d = json.loads(f.read_text()) if f.exists() else {}
    d[contract] = {"experiment": exp_id, "symbol": sym, "since": dt.datetime.now(NY).isoformat(timespec="seconds"), "log": log}
    save(f, d)


def forget_open(contract: str) -> None:
    f = open_file(); d = json.loads(f.read_text()) if f.exists() else {}
    if d.pop(contract, None) is not None:
        save(f, d)


def close_leftovers(market, log: list[str], when: str) -> None:
    """Anything the proving ground still holds that it shouldn't (carried over from an earlier day, or not
    closed by 3:45): close it and tell Greg."""
    f = open_file(); d = json.loads(f.read_text()) if f.exists() else {}
    for contract, info in list(d.items()):
        h = market.held(contract) if hasattr(market, "held") else None
        if not h:
            forget_open(contract); continue
        px = market.sell(contract, int(h[0]))
        if px is not None:
            forget_open(contract)
            msg = f"closed leftover {contract} ({int(h[0])}x, bought at ${h[1]:.2f}) at ${px:.2f} {when}"
            log.append(f"{market.now():%H:%M} {msg}"); notify(f"Proving ground {msg}")


def read_trades() -> list[dict]:
    return list(csv.DictReader(TRADES.open())) if TRADES.exists() else []


def log_trade(row: dict) -> None:
    new = not TRADES.exists()
    TRADES.parent.mkdir(parents=True, exist_ok=True)
    with TRADES.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


# ---- markets: live (Alpaca) or replay (stored minute bars + stored option prices) ------------------------
class LiveMarket:
    def __init__(self):
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient
        k, s = os.environ["ALPACA_API_KEY_ID"], os.environ["ALPACA_API_SECRET_KEY"]
        self.stocks, self.options = StockHistoricalDataClient(k, s), OptionHistoricalDataClient(k, s)
        self.trading = TradingClient(k, s, paper=True)
        self.broker = BR.Broker(self.trading)
        self.today = dt.datetime.now(NY).date()

    def now(self) -> dt.datetime:
        return dt.datetime.now(NY)

    def wait_next_minute(self) -> None:
        t = self.now(); time.sleep(max(1.0, 65 - t.second - t.microsecond / 1e6))

    def is_open_day(self) -> bool:
        from alpaca.trading.requests import GetCalendarRequest
        cal = self.trading.get_calendar(GetCalendarRequest(start=self.today, end=self.today))
        return bool(cal)

    def next_session(self) -> dt.date:
        from alpaca.trading.requests import GetCalendarRequest
        cal = self.trading.get_calendar(GetCalendarRequest(start=self.today + dt.timedelta(days=1),
                                                           end=self.today + dt.timedelta(days=10)))
        return cal[0].date if cal else (pd.Timestamp(self.today) + pd.offsets.BDay(1)).date()

    def bars(self, sym: str) -> pd.DataFrame:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        start = dt.datetime.combine(self.today, dt.time(9, 30), NY)
        res = self.stocks.get_stock_bars(StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Minute,
                                                          start=start, feed=DataFeed.IEX))
        rows = [(pd.Timestamp(b.timestamp).tz_convert(NY), b.open, b.high, b.low, b.close, b.volume) for b in res.data.get(sym, [])]
        return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])

    def option_price(self, contract: str, side: str) -> float | None:
        from alpaca.data.enums import OptionsFeed
        from alpaca.data.requests import OptionLatestQuoteRequest
        q = self.options.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=contract, feed=OptionsFeed.INDICATIVE))
        q = q.get(contract)
        if not q:
            return None
        return float(q.ask_price if side == "buy" else q.bid_price) or None

    def held(self, contract: str):
        return self.broker.held(contract)

    def buy(self, contract: str, qty: int):
        got = self.broker.buy(contract, qty, "pg", bot="proving")   # no-fill streak pauses only the proving ground
        return (got[0], int(got[1])) if got else None

    def sell(self, contract: str, qty: int):
        return self.broker.sell(contract, None, fallback_price=lambda: self.option_price(contract, "sell"))


class ReplayMarket:
    """A past day, minute by minute, on stored minute bars and stored real option prices."""

    def __init__(self, day: dt.date):
        self.today = day; self.minute = 0; self.series = {}

    def now(self) -> dt.datetime:
        return dt.datetime.combine(self.today, dt.time(9, 30), NY) + dt.timedelta(minutes=self.minute)

    def wait_next_minute(self) -> None:
        self.minute += 1

    def is_open_day(self) -> bool:
        return True

    def next_session(self) -> dt.date:
        return (pd.Timestamp(self.today) + pd.offsets.BDay(1)).date()

    def _day(self, sym):
        if sym not in self.series:
            days = {d["date"]: d for d in ORB.load_days(sym)}
            self.series[sym] = days[self.today]
        return self.series[sym]

    def bars(self, sym: str) -> pd.DataFrame:
        d = self._day(sym); n = min(self.minute, 390)
        ts = [dt.datetime.combine(self.today, dt.time(9, 30), NY) + dt.timedelta(minutes=i) for i in range(n)]
        return pd.DataFrame({"ts": ts, "open": d["o"][:n], "high": d["h"][:n], "low": d["l"][:n], "close": d["c"][:n], "volume": d["v"][:n]})

    def _opt(self, contract):
        sym = contract[:3]; d = self._day(sym)
        key = ("opt", sym)
        if key not in self.series:
            self.series[key] = OH.day_series(OH.fetch_day(sym, self.today, self.next_session(), float(np.nanmin(d["l"])), float(np.nanmax(d["h"]))))
        s = self.series[key].get(contract)
        return None if s is None else s[min(self.minute, 389)]

    def option_price(self, contract, side):
        p = self._opt(contract)
        return None if p is None or not np.isfinite(p) else float(p * (1 + ORB.SPREAD / 2 if side == "buy" else 1 - ORB.SPREAD / 2))

    def held(self, contract):
        return None

    def buy(self, contract, qty):
        px = self.option_price(contract, "buy")
        return (px, qty) if px else None

    def sell(self, contract, qty):
        return self.option_price(contract, "sell")


# ---- experiment #1: opening range on calm days -------------------------------------------------------------
def today_dict(hist: list[dict], bars: pd.DataFrame, day: dt.date) -> dict | None:
    """Today's arrays in the same shape as the backtest's days (minutes not yet seen stay NaN)."""
    if bars.empty:
        return None
    a = np.full((390, 5), np.nan)
    m = ((bars["ts"].dt.hour * 60 + bars["ts"].dt.minute) - 570).to_numpy()
    ok = (m >= 0) & (m < 390)
    a[m[ok]] = bars[["open", "high", "low", "close", "volume"]].to_numpy(float)[ok]
    seen = int(m[ok].max()) + 1 if ok.any() else 0
    a[:seen] = pd.DataFrame(a[:seen]).ffill().to_numpy()
    if np.isnan(a[0]).any():
        return None
    last = hist[-1]
    d = {"date": day, "o": a[:, 0], "h": a[:, 1], "l": a[:, 2], "c": a[:, 3], "v": np.nan_to_num(a[:, 4]), "seen": seen,
         "prev_close": last["c"][389]}
    d["sigma"] = np.mean([h["move"] for h in hist[-14:]], axis=0)
    typical = (d["h"] + d["l"] + d["c"]) / 3
    d["vwap"] = np.cumsum(np.nan_to_num(typical) * d["v"]) / np.maximum(np.cumsum(d["v"]), 1e-9)
    # continue the 5-minute ATR(14) from yesterday's last value, over the 5-minute bars completed so far
    atr, prev_c = float(last["atr_bar"][-1]), float(last["c"][389])
    out = np.full(78, np.nan)
    for k in range(seen // 5):
        s = slice(5 * k, 5 * k + 5)
        hi, lo, cl = d["h"][s].max(), d["l"][s].min(), d["c"][5 * k + 4]
        tr = max(hi - lo, abs(hi - prev_c), abs(lo - prev_c))
        atr = atr + (tr - atr) / 14; prev_c = cl; out[k] = atr
    d["atr_bar"] = out
    d["move"] = np.abs(d["c"] / d["o"][0] - 1)
    return d


class OrbCalm:
    ID = "orb-calm"
    BAND, TRIGGER, GIVEBACK, STOP = 0.3, 4.0, 0.5, 10.0

    def __init__(self, exp: dict, market):
        self.exp, self.mk = exp, market
        self.hist = {s: [d for d in ORB.load_days(s) if d["date"] < market.today] for s in exp["symbols"]}
        self.state = {s: {"phase": "waiting for 10:00"} for s in exp["symbols"]}
        self.shadow = exp.get("mode") == "shadow"            # live decisions + live quotes, NEVER an order
        self.model, self.threshold = self._train()

    def _train(self):
        """The calm-day predictor, trained on every day before today (both symbols)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        vix = features.vix_daily()
        df = pd.concat([TD.build(s, h, vix) for s, h in self.hist.items()], ignore_index=True)
        cols = [c for c in df.columns if c not in ("date", "symbol", "move_after_10", "trend")]
        self.cols, self.medians, self.vix = cols, df[cols].median(), vix
        X = df[cols].fillna(self.medians)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(X, df["trend"])
        threshold = float(np.median(model.predict_proba(X)[:, 1]))
        print(f"calm-day model trained on {len(df)} days; calm when the trend chance <= {threshold:.3f}", flush=True)
        return model, threshold

    def calm_score(self, sym: str, today: dict) -> float:
        row = TD.build(sym, self.hist[sym] + [today], self.vix).iloc[-1:]
        return float(self.model.predict_proba(row[self.cols].fillna(self.medians))[:, 1][0])

    def step(self, minute: int) -> list[dict]:
        events = []
        for sym, st in self.state.items():
            if st["phase"] in ("done for today", "sitting out: trend day predicted", "no signal today"):
                continue
            bars = self.mk.bars(sym)
            d = today_dict(self.hist[sym], bars, self.mk.today)
            if d is None:
                continue
            if "calm" not in st:
                if minute < TD.FIRST:                         # score the day once, at 10:00
                    continue
                p = self.calm_score(sym, d)
                st["calm"] = p
                st["phase"] = "watching for a breakout" if p <= self.threshold else "sitting out: trend day predicted"
                events.append({"sym": sym, "text": f"{sym}: {'calm day' if p <= self.threshold else 'trend day predicted'} "
                                                   f"(score {p:.3f} vs {self.threshold:.3f})"})
                continue
            if st["phase"] == "watching for a breakout":
                paused, why = BR.halted("proving")
                if paused:
                    st["note"] = why; continue                # no new entries; exits are never blocked
                st.pop("note", None)
                if minute > max(ORB.CHECKS) + 2:
                    st["phase"] = "no signal today"; continue
                sg = ORB.signal(d, self.BAND)
                if sg and minute - sg[0] <= 2:                 # act only on a fresh check
                    m, side, atr = sg
                    spot = d["c"][m - 1]
                    strike = round(spot - 2) if side == "call" else round(spot + 2)
                    contract = OH.occ(sym, self.mk.next_session(), side == "call", strike)
                    ask = self.mk.option_price(contract, "buy")
                    if not ask or ask < 0.10:
                        st["phase"] = "done for today"; st["note"] = f"no price for {contract}"; continue
                    qty = int(self.exp["stake"] // (ask * 100))
                    cap = BR.size_cap("proving")
                    if cap is not None:
                        qty = min(qty, cap)                   # new order code: minimum size until it has proven itself
                    if self.shadow:                           # record at the live ask; nothing is sent to the broker
                        got = (ask, qty) if qty > 0 else None
                    else:
                        got = self.mk.buy(contract, qty) if qty > 0 else None
                    if not got:
                        st["phase"] = "done for today"; st["note"] = "buy didn't fill (no position at the broker)"; continue
                    fill, qty = got
                    if not self.shadow:
                        remember_open(contract, self.exp["id"], sym)
                    st.update(phase="in a trade", side=side, contract=contract, qty=qty, entry=fill, atr=atr,
                              entry_u=float(d["c"][min(minute, d["seen"]) - 1]), entry_min=minute, best=0.0,
                              opened=self.mk.now().isoformat(timespec="seconds"), path=[])
                    events.append({"sym": sym, "text": f"{sym}: bought {qty} {contract} at ${fill:.2f} ({side}, breakout at "
                                                       f"{(570 + m) // 60}:{(570 + m) % 60:02d})"})
            elif st["phase"] == "in a trade":
                u = float(d["c"][d["seen"] - 1]); sgn = 1 if st["side"] == "call" else -1
                gain = sgn * (u - st["entry_u"]); st["best"] = max(st["best"], gain)
                st["path"].append([self.mk.now().strftime("%H:%M"), round(gain / st["atr"], 2)])
                try:                                          # real 1-min candles since entry, for the app's monitor
                    since = pd.Timestamp(st["opened"]) - pd.Timedelta(minutes=3)
                    b = bars[pd.to_datetime(bars["ts"]) >= since]
                    st["candles"] = [[pd.Timestamp(r.ts).strftime("%H:%M"), round(float(r.open), 2), round(float(r.high), 2),
                                      round(float(r.low), 2), round(float(r.close), 2)] for r in b.itertuples()][-90:]
                except Exception:
                    pass
                why = None
                if gain <= -self.STOP * st["atr"]:
                    why = "safety stop"
                elif st["best"] >= self.TRIGGER * st["atr"] and gain <= (1 - self.GIVEBACK) * st["best"]:
                    why = "gave back half"
                elif minute >= ORB.FLAT:
                    why = "3:45 exit"
                if why:
                    if self.shadow:                           # record at the live bid; nothing is sent to the broker
                        px = self.mk.option_price(st["contract"], "sell") or 0.0
                    else:
                        px = self.mk.sell(st["contract"], st["qty"])
                        if px is None and self.mk.held(st["contract"]):
                            st["note"] = "sell not confirmed yet, retrying next minute"; continue
                        px = px or self.mk.option_price(st["contract"], "sell") or 0.0
                        forget_open(st["contract"])
                    pnl = round(st["qty"] * 100 * (px - st["entry"]), 2)
                    log_trade({"date": str(self.mk.today), "experiment": self.exp["id"], "symbol": sym, "side": st["side"],
                               "contract": st["contract"], "qty": st["qty"], "entry_time": st["opened"],
                               "entry_price": st["entry"], "exit_time": self.mk.now().isoformat(timespec="seconds"),
                               "exit_price": px, "pnl": pnl, "pct": round(px / st["entry"] - 1, 4), "exit_reason": why,
                               "calm_score": round(st.get("calm", float("nan")), 4), "calm_threshold": round(self.threshold, 4)})
                    st.update(phase="done for today", last=f"{why}: {'+' if pnl >= 0 else '-'}${abs(pnl):,.0f}")
                    events.append({"sym": sym, "text": f"{sym}: sold ({why}) for {'+' if pnl >= 0 else '-'}${abs(pnl):,.0f}"})
        return events

    def snapshot(self) -> dict:
        return {s: {k: v for k, v in st.items() if k != "path"} | ({"path": st["path"][-240:]} if "path" in st else {})
                for s, st in self.state.items()} | {"_threshold": self.threshold}


IMPLEMENTATIONS = {OrbCalm.ID: OrbCalm}


# ---- lifecycle -----------------------------------------------------------------------------------------------
def stats_for(exp_id: str, trades: list[dict]) -> dict:
    t = [x for x in trades if x["experiment"] == exp_id and x["pnl"] not in ("", None)]
    pnl = np.array([float(x["pnl"]) for x in t])
    if not len(pnl):
        return {"trades": 0, "total": 0.0}
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    days = sorted({x["date"] for x in t})
    run, curve, by_day = 0.0, [], {}
    for x in t:
        by_day[x["date"]] = by_day.get(x["date"], 0.0) + float(x["pnl"])
    for d in sorted(by_day):
        run += by_day[d]; curve.append({"date": d, "day": round(by_day[d], 2), "total": round(run, 2)})
    return {"trades": len(pnl), "total": round(float(pnl.sum()), 2), "mean": round(float(pnl.mean()), 2),
            "se": round(float(pnl.std(ddof=1) / math.sqrt(len(pnl))), 2) if len(pnl) > 1 else None,
            "win_rate": round(float((pnl > 0).mean()), 3),
            "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if len(losses) and losses.sum() < 0 else None,
            "first": days[0], "last": days[-1], "curve": curve}


def lifecycle(reg: list[dict], trades: list[dict]) -> None:
    for e in reg:
        if e["status"] != "incubating":
            continue
        s = stats_for(e["id"], trades)
        weeks = (dt.date.today() - dt.date.fromisoformat(e["started"])).days / 7 if e.get("started") else 0
        exp_mean = (e.get("expectation") or {}).get("mean_per_trade")
        if s["trades"] >= MIN_TRADES or s["total"] <= -MAX_LOSS:
            pf, mean, se = s.get("profit_factor") or 0.0, s.get("mean", 0.0), s.get("se") or 0.0
            if s["total"] <= -MAX_LOSS or pf < 0.8 or (exp_mean is not None and mean < exp_mean - 2 * se):
                e["status"] = "retired"; e["ended"] = str(dt.date.today())
                e["verdict"] = f"retired after {s['trades']} trades: total ${s['total']:+,.0f}, profit factor {pf}, avg ${mean:+,.0f}/trade"
                notify(f"Retired '{e['name']}': {e['verdict']}")
            elif weeks >= MIN_WEEKS and pf >= 1.2 and (exp_mean is None or mean >= exp_mean - se):
                e["status"] = "ready for review"
                e["verdict"] = f"{s['trades']} trades over {weeks:.0f} weeks: total ${s['total']:+,.0f}, profit factor {pf}, avg ${mean:+,.0f}/trade"
                notify(f"'{e['name']}' is ready for your review: {e['verdict']}")
    save(REGISTRY, reg)


# ---- the session --------------------------------------------------------------------------------------------
def write_status(reg, runners, phase, log, market):
    trades = read_trades()
    save(STATE / "status.json", {
        "updated": market.now().isoformat(timespec="seconds"), "session": str(market.today), "phase": phase,
        "experiments": [{**{k: v for k, v in e.items()}, "stats": stats_for(e["id"], trades),
                         "today": runners[e["id"]].snapshot() if e["id"] in runners else None} for e in reg],
        "log": log[-30:],
    })


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--replay", help="YYYY-MM-DD: replay a past day on stored prices")
    ap.add_argument("--record", action="store_true",
                    help="with --replay: write the replayed trades into the REAL record (to rebuild a mishandled day)")
    args = ap.parse_args()
    market = ReplayMarket(dt.date.fromisoformat(args.replay)) if args.replay else LiveMarket()
    global STATE, TRADES, REGISTRY
    if args.replay and not args.record:                       # replays never touch the real records (unless --record)
        STATE = STATE / "replay"; TRADES = STATE / "trades.csv"; REGISTRY = STATE / "experiments.json"
    if args.replay and args.record:
        already = [t for t in read_trades() if t["date"] == args.replay]
        if already:
            print(f"{args.replay} is already in the record ({len(already)} trades); nothing to do"); return 0
    reg = registry()
    log: list[str] = []
    if not market.is_open_day():
        write_status(reg, {}, "market closed today", log, market); return 0
    runners = {}
    for e in reg:
        if e["status"] in ("incubating", "ready for review") and e.get("impl", e["id"]) in IMPLEMENTATIONS:
            try:
                runners[e["id"]] = IMPLEMENTATIONS[e.get("impl", e["id"])](e, market)
            except Exception as exc:
                log.append(f"{e['name']}: couldn't start ({type(exc).__name__}: {exc})")
    if not args.replay:                                       # wait for the open
        while market.now().time() < dt.time(9, 31):
            write_status(reg, runners, "waiting for the open", log, market); time.sleep(20)
        f = open_file()
        BR.write_claims("proving", {c: (market.held(c) or (0,))[0] for c in (json.loads(f.read_text()) if f.exists() else {})})
        close_leftovers(market, log, "at the open (held overnight)")
    else:
        market.minute = 1
    while True:
        minute = int((market.now() - dt.datetime.combine(market.today, dt.time(9, 30), NY)).total_seconds() // 60)
        if minute > ORB.FLAT + 1:
            break
        for rid, r in runners.items():
            try:
                for ev in r.step(minute):
                    line = f"{market.now():%H:%M} {ev['text']}"; log.append(line); print(line, flush=True)
            except Exception as exc:
                log.append(f"{market.now():%H:%M} {rid}: error {type(exc).__name__}: {exc}"); print(log[-1], flush=True)
        if not args.replay:
            hold = {st["contract"]: st["qty"] for r in runners.values() if not getattr(r, "shadow", False)
                    for st in r.state.values() if st.get("phase") == "in a trade"}
            f = open_file()
            for c in (json.loads(f.read_text()) if f.exists() else {}):
                hold.setdefault(c, (market.held(c) or (0,))[0])
            BR.write_claims("proving", hold)
        busy = any(st.get("phase") == "in a trade" for r in runners.values() for st in r.state.values())
        write_status(reg, runners, "in a trade" if busy else "watching", log, market)
        market.wait_next_minute()
    if not args.replay:
        close_leftovers(market, log, "after the session (should have closed by 3:45)")
        BR.write_claims("proving", {})
    lifecycle(reg, read_trades())
    write_status(reg, runners, "session over", log, market)
    print(f"{market.today}: session over · " + " · ".join(f"{e['name']}: {e['status']}" for e in reg), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
