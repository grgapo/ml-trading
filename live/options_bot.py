"""Intraday options paper trader (Alpaca paper) with a learning agent.

Runs once per trading day, from just before the open to the close:
  * Watches SPY/QQQ 1-minute bars (free real-time IEX feed).
  * Each check, a small learning agent looks at the situation (symbol, 5-min
    momentum, trend vs the 30-min average, time of day) and picks one of:
    skip, buy a 1DTE call/put, or buy a 0DTE call/put. It learns which works.
  * Only BUYS options (max loss = premium paid). One trade open at a time,
    each with a stop-loss, take-profit and max holding time. Everything is
    closed before the bell.
  * Daily rules: at most MAX_TRADES trades and OPT_DAILY_BUDGET premium; stops
    early once WIN_TARGET trades closed net positive.
  * After the close it "trains overnight": replays every trade it has ever made
    (Dyna-style planning over all recorded outcomes) before saving.

Status for the lab app: STATE_DIR/options_status.json + options_trades.csv.
Create STATE_DIR/options.PAUSE to stop it opening new trades (it still manages
an open one). Paper only: the Alpaca client is hard-wired to paper=True.

  python live/options_bot.py               # real paper session
  python live/options_bot.py --simulate    # fake market, fast clock (testing)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import time
import sys
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))

NY = ZoneInfo("America/New_York")

# ---- settings (override in ~/.config/paper-bot.env) ---------------------------
SYMBOLS = os.environ.get("OPT_SYMBOLS", "SPY,QQQ").replace(" ", "").split(",")
DAILY_BUDGET = float(os.environ.get("OPT_DAILY_BUDGET", "5000"))
MAX_TRADES = int(os.environ.get("OPT_MAX_TRADES", "5"))
WIN_TARGET = int(os.environ.get("OPT_WIN_TARGET", "3"))
PER_TRADE = float(os.environ.get("OPT_PER_TRADE", str(DAILY_BUDGET / MAX_TRADES)))
STOP_LOSS = float(os.environ.get("OPT_STOP_LOSS", "-0.35"))     # -35% of premium
TAKE_PROFIT = float(os.environ.get("OPT_TAKE_PROFIT", "0.50"))  # +50% of premium
MAX_HOLD_MIN = int(os.environ.get("OPT_MAX_HOLD_MIN", "60"))
CHECK_EVERY_S = int(os.environ.get("OPT_CHECK_EVERY_S", "60"))
FIRST_ENTRY = os.environ.get("OPT_FIRST_ENTRY", "09:45")         # skip the opening chaos
LAST_ENTRY = os.environ.get("OPT_LAST_ENTRY", "15:00")
FLATTEN_AT = os.environ.get("OPT_FLATTEN_AT", "15:45")           # close everything
EXPLORATION = float(os.environ.get("OPT_EXPLORATION", "0.25"))          # table agent (no model yet)
MODEL_EXPLORATION = float(os.environ.get("OPT_MODEL_EXPLORATION", "0.10"))  # once pre-trained
LIVE_WEIGHT = float(os.environ.get("OPT_LIVE_WEIGHT", "5"))  # a real paper trade counts 5x a simulated one
REPLAY_SWEEPS = int(os.environ.get("OPT_REPLAY_SWEEPS", "20000"))
STATE_DIR = Path(os.environ.get("STATE_DIR", Path.home() / "projects" / "paper-state"))

ACTIONS = ["skip", "call_1dte", "put_1dte", "call_0dte", "put_0dte"]


def hm(text: str) -> dt.time:
    h, m = text.split(":")
    return dt.time(int(h), int(m))


# ---- the learning agent -----------------------------------------------------------
class Agent:
    """Tabular value learner over (situation, action) with Dyna-style replay.

    Each trade is one episode: situation -> action -> reward (P&L as a fraction
    of premium). Unlike a 'last outcome' model, replay samples from EVERY
    recorded outcome, so it learns the average, not the most recent fluke.
    """

    def __init__(self, path: Path):
        self.path = path
        self.q: dict[str, list[float]] = {}
        self.n: dict[str, list[int]] = {}
        self.memory: list[list] = []          # [state, action_index, reward, date]
        if path.exists():
            data = json.loads(path.read_text())
            self.q, self.n, self.memory = data["q"], data["n"], data["memory"]

    def _row(self, state: str):
        if state not in self.q:
            # Skip is worth exactly 0; untried trades start slightly optimistic
            # so the agent tries each one before judging it.
            self.q[state] = [0.0] + [0.05] * (len(ACTIONS) - 1)
            self.n[state] = [0] * len(ACTIONS)
        return self.q[state], self.n[state]

    def choose(self, state: str, allowed: list[int], explore: float) -> tuple[int, bool]:
        q, _ = self._row(state)
        if random.random() < explore:
            return random.choice(allowed), True
        return max(allowed, key=lambda a: q[a]), False

    def learn(self, state: str, action: int, reward: float, date: str, remember=True) -> None:
        q, n = self._row(state)
        n[action] += 1
        alpha = max(1.0 / n[action], 0.05)
        q[action] += alpha * (reward - q[action])
        if remember:
            self.memory.append([state, action, round(reward, 4), date])

    def replay(self, sweeps: int) -> None:
        """Overnight training: revisit past trades at a small learning rate."""
        if not self.memory:
            return
        for _ in range(sweeps):
            state, action, reward, _ = random.choice(self.memory)
            q, _ = self._row(state)
            q[action] += 0.01 * (reward - q[action])

    # Same interface as ModelAgent.
    mode = "table"

    def decide(self, symbol, bars, now, explore):
        closes = bars["close"].tolist()
        if len(closes) < 30:
            return None
        sit = situation(symbol, closes, now)
        a, explored = self.choose(sit, list(range(len(ACTIONS))), explore)
        return {"action": a, "explored": explored, "situation": sit, "features": None, "preds": None}

    def learn_trade(self, trade: dict, reward: float, date: str) -> None:
        self.learn(trade["situation"], ACTIONS.index(trade["action"]), reward, date)

    def overnight(self) -> None:
        self.replay(REPLAY_SWEEPS)
        self.save()

    def stats(self) -> dict:
        by_action = {}
        for _, a, r, _ in self.memory:
            by_action.setdefault(ACTIONS[a], []).append(r)
        return {
            "mode": "table (no pre-training yet)",
            "trades_learned": len(self.memory),
            "situations_seen": len(self.q),
            "avg_by_action": {k: round(sum(v) / len(v), 4) for k, v in by_action.items()},
            "count_by_action": {k: len(v) for k, v in by_action.items()},
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"q": self.q, "n": self.n, "memory": self.memory}))
        tmp.replace(self.path)


class ModelAgent:
    """Pre-trained model (live/options_pretrain.py): predicts each trade's return
    from the indicators, takes the best one if it clears MIN_EDGE, and retrains
    every night on history + its own paper trades (weighted LIVE_WEIGHT x)."""

    mode = "model"

    def __init__(self, model_path: Path, live_path: Path):
        import joblib
        self.model_path, self.live_path = model_path, live_path
        self.bundle = joblib.load(model_path)
        self.models = self.bundle["models"]
        self.names = self.bundle["features"]
        self.signal_mode = bool(self.bundle.get("signal_mode"))
        self.strategy = self.bundle.get("strategy")     # set when the nightly search promoted a champion
        if self.strategy:
            self.exits = {"sl": self.strategy["stop_loss"], "tp": self.strategy["take_profit"],
                          "hold": self.strategy["max_hold"]}
        self.min_edge = float(os.environ.get("OPT_MIN_EDGE", self.bundle.get("min_edge", 0.03)))
        self.seen: set = set()        # signal bars already decided on
        self.market = None            # set by run_session (daily history for the vol forecast)
        self.vol_cache: dict = {}
        self.q_low = self.bundle.get("conformal_q_low")

    def decide(self, symbol, bars, now, explore):
        if self.strategy:
            return self._decide_strategy(symbol, bars, now, explore)
        import numpy as np
        from features import add_indicators, describe, fired_signals
        df = add_indicators(bars, symbol)
        if len(df) < 60:
            return None
        row = df.iloc[-1].copy()
        if pd_isna(row["day_gap"]):
            row["day_gap"] = 0.0
        allowed = [1, 2, 3, 4]                 # ACTIONS index: call_1dte, put_1dte, call_0dte, put_0dte
        label = ""
        if self.signal_mode:                   # meta-labeling: only act when a setup fires
            key = (symbol, str(row["ts"]))
            direction = int(row["signal_dir"])
            if direction == 0 or key in self.seen:
                return None
            self.seen.add(key)
            allowed = [1, 3] if direction > 0 else [2, 4]
            label = f"{', '.join(fired_signals(row))} {'▲' if direction > 0 else '▼'} · "
        if "garch_vol" in self.names:
            row["garch_vol"], row["vol_ratio"] = self._vol_today(symbol, now, float(row["close"]))
        f = row[self.names].to_numpy(dtype=float)
        if np.isnan(f).any():
            return None
        preds = [float(m.predict(f.reshape(1, -1))[0]) for m in self.models]
        explored = random.random() < explore
        if explored:
            a = random.choice(allowed)
        else:
            best = max(allowed, key=lambda x: preds[x - 1])
            a = best if preds[best - 1] >= self.min_edge else 0
        return {"action": a, "explored": explored, "situation": f"{symbol} · {label}" + describe(row),
                "features": [round(x, 8) for x in f.tolist()],
                "preds": {ACTIONS[i]: round(preds[i - 1], 4) for i in allowed}}

    def _vol_today(self, symbol, now, last_price):
        """(garch_vol, vol_ratio) for today, computed once per symbol per day."""
        import pandas as pd
        from features import daily_vol_table
        today = now.date()
        hit = self.vol_cache.get(symbol)
        if hit and hit[0] == today:
            return hit[1]
        val = (float("nan"), float("nan"))
        try:
            d = self.market.daily_bars(symbol)
            d = d[d["ts"].dt.date < today]
            d = pd.concat([d, pd.DataFrame({"ts": [pd.Timestamp(now)], "close": [last_price]})], ignore_index=True)
            tab = daily_vol_table(d)
            val = (float(tab["garch_vol"].iloc[-1]), float(tab["vol_ratio"].iloc[-1]))
        except Exception as exc:
            print(f"vol forecast unavailable for {symbol}: {exc}")
        self.vol_cache[symbol] = (today, val)
        return val

    def _decide_strategy(self, symbol, bars, now, explore):
        """Run the promoted champion: its bar size, signals, window, DTE, model and threshold."""
        import numpy as np
        from features import SIGNAL_NAMES, add_indicators, describe, direction_from, resample
        from strategy import WINDOWS, allowed_actions
        st = self.strategy
        b = resample(bars, st["bar_minutes"])
        if st["bar_minutes"] > 1 and len(b):          # drop the bar that is still forming
            last_1m = bars["ts"].iloc[-1]
            b = b[b["ts"] + dt.timedelta(minutes=st["bar_minutes"] - 1) <= last_1m].reset_index(drop=True)
        df = add_indicators(b, symbol)
        if len(df) < 30:
            return None
        row = df.iloc[-1].copy()
        if pd_isna(row["day_gap"]):
            row["day_gap"] = 0.0
        direction = int(direction_from(row, st["signals"]))
        lo, hi = WINDOWS[st["window"]]
        key = (symbol, str(row["ts"]))
        if direction == 0 or key in self.seen or not (lo <= row["minutes_since_open"] <= hi):
            return None
        self.seen.add(key)
        row["garch_vol"], row["vol_ratio"] = self._vol_today(symbol, now, float(row["close"]))
        from strategy import confluence_ok, vol_ok
        if not confluence_ok(direction, row.get("conf_bull", 1), row.get("conf_bear", 1), st.get("confluence")):
            return {"action": 0, "explored": False, "situation": f"{symbol} · signal skipped: only "
                    f"{int(row['conf_bull'] if direction > 0 else row['conf_bear'])} setup(s) agree, needs {st['confluence']}",
                    "features": None, "preds": None}
        if not vol_ok(row["vol_ratio"], st.get("vol_filter", "off")):
            return {"action": 0, "explored": False, "situation": f"{symbol} · signal skipped: "
                    f"{st.get('vol_filter')} days only (vol ratio {row['vol_ratio']:.2f})",
                    "features": None, "preds": None}
        options = [a + 1 for a in allowed_actions(direction, st["dte"])]   # -> ACTIONS indices
        fired = [SIGNAL_NAMES[sg] for sg in st["signals"] if row[sg] != 0]
        label = f"{', '.join(fired)} {'▲' if direction > 0 else '▼'} · "
        f = row[self.names].to_numpy(dtype=float)
        if np.isnan(f).any():
            return None
        preds = None
        explored = False
        if self.models is None:                        # champion = take every signal
            a = options[0]
        else:
            preds = [float(m.predict(f.reshape(1, -1))[0]) for m in self.models]
            explored = random.random() < explore
            if explored:
                a = random.choice(options)
            else:
                best = max(options, key=lambda x: preds[x - 1])
                a = best if preds[best - 1] >= (st["min_edge"] or 0.0) else 0
                if a and self.q_low is not None and preds[a - 1] + self.q_low[a - 1] < 0:
                    a = 0                                  # not confident enough it won't lose
                    label += "not confident enough · "
        return {"action": a, "explored": explored, "situation": f"{symbol} · {label}" + describe(row),
                "features": [round(x, 8) for x in f.tolist()],
                "preds": {ACTIONS[i]: round(preds[i - 1], 4) for i in options} if preds else None}

    def learn_trade(self, trade: dict, reward: float, date: str) -> None:
        new = not self.live_path.exists()
        with self.live_path.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["date", "action", "reward", "features"])
            w.writerow([date, trade["action"], round(reward, 4), json.dumps(trade["features"])])

    def _live_rows(self):
        if not self.live_path.exists():
            return []
        with self.live_path.open() as fh:
            return list(csv.DictReader(fh))

    def overnight(self) -> None:
        """Retrain on simulated history + real paper trades."""
        import joblib
        import numpy as np
        from options_pretrain import DATASET_PATH, fit_models
        live = self._live_rows()
        if self.strategy and self.models is None:
            return                                     # take-every-signal champion: nothing to fit
        path = Path(self.bundle.get("dataset_path") or DATASET_PATH)
        if not live or not path.exists():
            return
        data = np.load(path)
        X, R = data["X"], data["R"]
        if self.strategy:                              # same rows the champion was trained on
            from options_search import qualify
            ok, _ = qualify({"X": X}, self.strategy)
            X, R = X[ok], R[ok]
        LX = np.array([json.loads(r["features"]) for r in live])
        # A live trade only reveals the outcome of the action actually taken: use the
        # model's current prediction for the other actions so they are left unchanged.
        LR = np.column_stack([m.predict(LX) for m in self.models])
        for k, r in enumerate(live):
            LR[k, ACTIONS.index(r["action"]) - 1] = float(r["reward"])
        Xall, Rall = np.vstack([X, LX]), np.vstack([R, LR])
        w = np.concatenate([np.ones(len(X)), np.full(len(LX), LIVE_WEIGHT)])
        self.models = fit_models(Xall, Rall, w)
        self.bundle.update(models=self.models, live_trades=len(live),
                           retrained=dt.datetime.now(NY).isoformat(timespec="seconds"))
        tmp = self.model_path.with_suffix(".tmp")
        joblib.dump(self.bundle, tmp)
        tmp.replace(self.model_path)

    def stats(self) -> dict:
        live = self._live_rows()
        by_action = {}
        for r in live:
            by_action.setdefault(r["action"], []).append(float(r["reward"]))
        wf = self.bundle.get("walk_forward", {})
        m = wf.get("model", {})
        return {
            "mode": ("search champion: " + self.bundle.get("description", "")) if self.strategy else "pre-trained model",
            "trades_learned": len(live),
            "avg_by_action": {k: round(sum(v) / len(v), 4) for k, v in by_action.items()},
            "count_by_action": {k: len(v) for k, v in by_action.items()},
            "model": {"trained": f"{self.bundle.get('trained_from')} → {self.bundle.get('trained_to')}",
                      "simulated_decisions": self.bundle.get("decisions"),
                      "walk_forward_pnl": m.get("pnl"), "walk_forward_trades": m.get("trades"),
                      "walk_forward_green_days": f"{m.get('green_days')}/{m.get('days')}",
                      "synthetic": self.bundle.get("synthetic", False)},
        }


def pd_isna(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def situation(symbol: str, closes: list[float], now: dt.datetime) -> str:
    """Describe the market in a few coarse buckets the agent can learn from."""
    mom = closes[-1] / closes[-6] - 1 if len(closes) >= 6 else 0.0
    mom_b = "up" if mom > 0.001 else "down" if mom < -0.001 else "flat"
    avg = sum(closes[-30:]) / len(closes[-30:])
    gap = closes[-1] / avg - 1
    trend_b = "above" if gap > 0.0005 else "below" if gap < -0.0005 else "near"
    minutes = (now.hour * 60 + now.minute) - (9 * 60 + 30)
    tod_b = "open" if minutes < 60 else "mid" if minutes < 300 else "late"
    return f"{symbol}|mom:{mom_b}|trend:{trend_b}|tod:{tod_b}"


# ---- market access: real (Alpaca paper) and simulated ---------------------------
class AlpacaMarket:
    def __init__(self) -> None:
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient
        key, secret = os.environ.get("ALPACA_API_KEY_ID"), os.environ.get("ALPACA_API_SECRET_KEY")
        if not key or not secret:
            raise SystemExit("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY missing (see .env)")
        self.trading = TradingClient(key, secret, paper=True)   # paper only, on purpose
        self.stocks = StockHistoricalDataClient(key, secret)
        self.options = OptionHistoricalDataClient(key, secret)

    def now(self) -> dt.datetime:
        return dt.datetime.now(NY)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def session(self) -> tuple[dt.datetime, dt.datetime] | None:
        """Today's open/close in NY time, or None if the market is closed today."""
        from alpaca.trading.requests import GetCalendarRequest
        today = self.now().date()
        days = self.trading.get_calendar(GetCalendarRequest(start=today, end=today))
        if not days or days[0].date != today:
            return None
        day = days[0]
        return day.open.replace(tzinfo=NY), day.close.replace(tzinfo=NY)

    def next_session_date(self) -> dt.date:
        from alpaca.trading.requests import GetCalendarRequest
        today = self.now().date()
        days = self.trading.get_calendar(GetCalendarRequest(
            start=today + dt.timedelta(days=1), end=today + dt.timedelta(days=10)))
        return days[0].date

    def bars(self, symbol: str):
        """Regular-session 1-minute bars for roughly the last 3 sessions (indicator warm-up)."""
        import pandas as pd
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
                               start=self.now() - dt.timedelta(days=5), feed=DataFeed.IEX)
        raw = self.stocks.get_stock_bars(req).data.get(symbol, [])
        df = pd.DataFrame({"ts": [b.timestamp for b in raw], "open": [float(b.open) for b in raw],
                           "high": [float(b.high) for b in raw], "low": [float(b.low) for b in raw],
                           "close": [float(b.close) for b in raw], "volume": [float(b.volume) for b in raw]})
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_convert(NY)
        mins = df["ts"].dt.hour * 60 + df["ts"].dt.minute
        return df[(mins >= 570) & (mins < 960)].reset_index(drop=True)

    def daily_bars(self, symbol: str):
        """About 3 years of daily closes (for the GARCH volatility forecast)."""
        import pandas as pd
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                               start=self.now() - dt.timedelta(days=1100), feed=DataFeed.IEX)
        raw = self.stocks.get_stock_bars(req).data.get(symbol, [])
        df = pd.DataFrame({"ts": [b.timestamp for b in raw], "close": [float(b.close) for b in raw]})
        if len(df):
            df["ts"] = pd.to_datetime(df["ts"]).dt.tz_convert(NY)
        return df

    def pick_contract(self, symbol: str, kind: str, expiry: dt.date, spot: float) -> str | None:
        from alpaca.trading.enums import ContractType
        from alpaca.trading.requests import GetOptionContractsRequest
        res = self.trading.get_option_contracts(GetOptionContractsRequest(
            underlying_symbols=[symbol], expiration_date=expiry,
            type=ContractType.CALL if kind == "call" else ContractType.PUT,
            strike_price_gte=str(round(spot * 0.98, 2)), strike_price_lte=str(round(spot * 1.02, 2)),
            limit=100))
        contracts = [c for c in (res.option_contracts or []) if c.tradable]
        if not contracts:
            return None
        return min(contracts, key=lambda c: abs(float(c.strike_price) - spot)).symbol

    def ask(self, contract: str) -> float | None:
        from alpaca.data.enums import OptionsFeed
        from alpaca.data.requests import OptionLatestQuoteRequest
        quote = self.options.get_option_latest_quote(
            OptionLatestQuoteRequest(symbol_or_symbols=contract, feed=OptionsFeed.INDICATIVE))[contract]
        return float(quote.ask_price) if quote.ask_price else None

    def buy(self, contract: str, qty: int) -> None:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        self.trading.submit_order(MarketOrderRequest(
            symbol=contract, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))

    def position(self, contract: str) -> dict | None:
        try:
            p = self.trading.get_open_position(contract)
        except Exception:
            return None
        return {"qty": float(p.qty), "entry": float(p.avg_entry_price),
                "plpc": float(p.unrealized_plpc or 0.0), "pl": float(p.unrealized_pl or 0.0)}

    def close(self, contract: str) -> None:
        self.trading.close_position(contract)

    def cancel_orders(self, contract: str) -> None:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        for o in self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[contract])):
            self.trading.cancel_order_by_id(o.id)


class SimMarket:
    """Fake market for testing: random-walk prices, simple option pricing, fast clock."""

    def __init__(self, date: dt.date, seed: int = 0) -> None:
        self.rng = random.Random(seed or date.toordinal())
        self.t = dt.datetime.combine(date, dt.time(9, 30), NY)
        self.px = {s: 100.0 * (i + 5) for i, s in enumerate(SYMBOLS)}
        self.hist = {s: [p * math.exp(self.rng.gauss(0, 0.0007) * k ** 0.5) for k in range(400, 0, -1)]
                     for s, p in self.px.items()}
        self.pos: dict[str, dict] = {}

    def now(self) -> dt.datetime:
        return self.t

    def sleep(self, seconds: float) -> None:
        for _ in range(max(1, int(seconds // 60))):
            self.t += dt.timedelta(minutes=1)
            for s in self.px:
                self.px[s] *= math.exp(self.rng.gauss(0.0, 0.0007))
                self.hist[s].append(self.px[s])

    def session(self):
        d = self.t.date()
        return dt.datetime.combine(d, dt.time(9, 30), NY), dt.datetime.combine(d, dt.time(16, 0), NY)

    def next_session_date(self) -> dt.date:
        d = self.t.date() + dt.timedelta(days=1)
        while d.weekday() >= 5:
            d += dt.timedelta(days=1)
        return d

    def bars(self, symbol: str):
        import pandas as pd
        closes = self.hist[symbol][-1200:]
        n = len(closes)
        ts = [self.t - dt.timedelta(minutes=n - 1 - k) for k in range(n)]
        df = pd.DataFrame({"ts": pd.to_datetime(ts), "open": closes, "high": [c * 1.0002 for c in closes],
                           "low": [c * 0.9998 for c in closes], "close": closes, "volume": [1000.0] * n})
        df["ts"] = df["ts"].dt.tz_convert(NY)
        return df

    def daily_bars(self, symbol: str):
        import pandas as pd
        rng = random.Random(len(symbol))
        days = pd.bdate_range(end=self.t.date() - dt.timedelta(days=1), periods=700)
        px, closes = self.px[symbol], []
        for _ in days:
            px *= math.exp(rng.gauss(0, 0.011))
            closes.append(px)
        return pd.DataFrame({"ts": pd.to_datetime(days).tz_localize(NY), "close": closes})

    def pick_contract(self, symbol, kind, expiry, spot):
        return f"{symbol}{expiry:%y%m%d}{kind[0].upper()}{round(spot)}"

    def _premium(self, contract: str, spot: float) -> float:
        zero_dte = contract[3:9] == f"{self.t:%y%m%d}"
        return spot * (0.0025 if zero_dte else 0.004)

    def ask(self, contract: str) -> float:
        sym = contract[:3]
        return self._premium(contract, self.px[sym])

    def buy(self, contract: str, qty: int) -> None:
        sym = contract[:3]
        self.pos[contract] = {"qty": qty, "entry": self.ask(contract), "s0": self.px[sym], "t0": self.t,
                              "sign": 1 if contract[9] == "C" else -1}

    def position(self, contract: str) -> dict | None:
        p = self.pos.get(contract)
        if not p:
            return None
        sym = contract[:3]
        minutes = (self.t - p["t0"]).total_seconds() / 60
        value = max(0.01, p["entry"] + 0.5 * p["sign"] * (self.px[sym] - p["s0"]) - 0.0006 * p["entry"] * minutes)
        plpc = value / p["entry"] - 1
        return {"qty": p["qty"], "entry": p["entry"], "plpc": plpc, "pl": (value - p["entry"]) * 100 * p["qty"]}

    def close(self, contract: str) -> None:
        self.pos.pop(contract, None)

    def cancel_orders(self, contract: str) -> None:
        pass


# ---- one trading session ------------------------------------------------------------
def write_status(**data) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    data["updated"] = dt.datetime.now(NY).isoformat(timespec="seconds")
    tmp = STATE_DIR / "options_status.json.tmp"
    tmp.write_text(json.dumps(data))
    tmp.replace(STATE_DIR / "options_status.json")


def log_trade(row: dict) -> None:
    path = STATE_DIR / "options_trades.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "opened", "closed", "symbol", "contract", "action",
                                          "qty", "entry", "pl_pct", "pl_usd", "exit_reason", "situation",
                                          "explored"])
        if new:
            w.writeheader()
        w.writerow(row)


def notify(text: str, priority: int = 3) -> None:
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        return
    req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=text.encode(),
                                 headers={"Title": "Options bot", "Priority": str(priority)})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:
        print(f"ntfy failed: {exc}")


def run_session(market, agent: Agent, explore: float) -> dict:
    hours = market.session()
    if hours is None:
        print("market closed today")
        write_status(phase="closed today", agent=agent.stats())
        return {"skipped": True}
    open_t, close_t = hours
    today = open_t.date()
    expiry = {"1dte": market.next_session_date(), "0dte": today}
    first_entry = dt.datetime.combine(today, hm(FIRST_ENTRY), NY)
    last_entry = min(dt.datetime.combine(today, hm(LAST_ENTRY), NY), close_t - dt.timedelta(minutes=45))
    flatten = min(dt.datetime.combine(today, hm(FLATTEN_AT), NY), close_t - dt.timedelta(minutes=15))

    ex = getattr(agent, "exits", None) or {"sl": STOP_LOSS, "tp": TAKE_PROFIT, "hold": MAX_HOLD_MIN}
    day = {"date": str(today), "trades": 0, "wins": 0, "spent": 0.0, "pl_usd": 0.0, "last_closed": None,
           "stopped": None, "open": None, "last_decision": None}
    turn = 0

    agent.market = market
    agent_stats = agent.stats()

    def status(phase):
        open_t = {k: v for k, v in (day["open"] or {}).items() if k != "features"} or None
        write_status(phase=phase, session=str(today), max_trades=MAX_TRADES, win_target=WIN_TARGET,
                     budget=DAILY_BUDGET, exits=ex, **{k: v for k, v in day.items() if k not in ("date", "open")},
                     open=open_t, agent=agent_stats)

    while market.now() < open_t:
        status("waiting for the open")
        market.sleep(min(60, max(1, (open_t - market.now()).total_seconds())))

    while market.now() < close_t:
        now = market.now()
        t = day["open"]
        if t:                                            # manage the open trade
            pos = market.position(t["contract"])
            if pos is None and (now - dt.datetime.fromisoformat(t["opened"])).total_seconds() < 120:
                status("waiting for fill")
            elif pos is None:                            # never filled: give up on it
                market.cancel_orders(t["contract"])
                day["open"] = None
                status("order did not fill, cancelled")
            else:
                held = (now - dt.datetime.fromisoformat(t["opened"])).total_seconds() / 60
                t.update(plpc=round(pos["plpc"], 4), pl_usd=round(pos["pl"], 2), entry=pos["entry"])
                t.setdefault("path", []).append([now.strftime("%H:%M"), round(pos["plpc"], 4)])   # for the app's mini-chart
                del t["path"][:-240]
                try:                                     # underlying's 1-min candles since entry, for the app's monitor
                    b = market.bars(t["symbol"])
                    since = dt.datetime.fromisoformat(t["opened"]) - dt.timedelta(minutes=3)
                    b = b[b["ts"] >= since]
                    t["candles"] = [[r.ts.strftime("%H:%M"), round(float(r.open), 2), round(float(r.high), 2),
                                     round(float(r.low), 2), round(float(r.close), 2)] for r in b.itertuples()][-90:]
                except Exception:
                    pass
                reason = ("stop-loss" if pos["plpc"] <= ex["sl"] else
                          "take-profit" if pos["plpc"] >= ex["tp"] else
                          "max hold" if held >= ex["hold"] else
                          "end of day" if now >= flatten else None)
                if reason:
                    market.close(t["contract"])
                    reward = max(-1.0, min(1.0, pos["plpc"]))
                    agent.learn_trade(t, reward, str(today))
                    agent_stats.update(agent.stats())
                    day["trades"] += 1
                    day["wins"] += pos["pl"] > 0
                    day["pl_usd"] = round(day["pl_usd"] + pos["pl"], 2)
                    log_trade({"date": str(today), "opened": t["opened"], "closed": now.isoformat(timespec="seconds"),
                               "symbol": t["symbol"], "contract": t["contract"], "action": t["action"],
                               "qty": t["qty"], "entry": round(pos["entry"], 2), "pl_pct": round(pos["plpc"], 4),
                               "pl_usd": round(pos["pl"], 2), "exit_reason": reason,
                               "situation": t["situation"], "explored": t["explored"]})
                    day["last_closed"] = {"closed": now.isoformat(timespec="seconds"), "symbol": t["symbol"],
                                          "action": t["action"], "pl_usd": round(pos["pl"], 2),
                                          "plpc": round(pos["plpc"], 4), "reason": reason}
                    day["open"] = None
                    if day["wins"] >= WIN_TARGET:
                        day["stopped"] = f"hit {WIN_TARGET} winners"
                    elif day["trades"] >= MAX_TRADES:
                        day["stopped"] = f"used all {MAX_TRADES} trades"
                    if agent.mode == "table":
                        agent.save()
                status("in a trade" if day["open"] else "watching")
        elif day["stopped"] is None:                     # maybe open a trade
            paused = (STATE_DIR / "options.PAUSE").exists()
            room = DAILY_BUDGET - day["spent"]
            if paused:
                status("paused (options.PAUSE)")
            elif now < first_entry:
                status("watching (entries open at " + FIRST_ENTRY + ")")
            elif now >= last_entry:
                day["stopped"] = "past last entry time"
            elif room < 50:
                day["stopped"] = "daily budget used"
            else:
                # Model with signals: check every symbol each minute. Table agent: alternate.
                if getattr(agent, "signal_mode", False):
                    candidates = SYMBOLS
                else:
                    candidates = [SYMBOLS[turn % len(SYMBOLS)]]
                    turn += 1
                d = None
                for symbol in candidates:
                    bars = market.bars(symbol)
                    d = agent.decide(symbol, bars, now, explore) if len(bars) else None
                    if d:
                        break
                if d:
                    closes = bars["close"].tolist()
                    sit, explored = d["situation"], d["explored"]
                    action = ACTIONS[d["action"]]
                    day["last_decision"] = {"time": now.isoformat(timespec="seconds"), "situation": sit,
                                            "action": action, "explored": explored, "preds": d["preds"]}
                    if action != "skip":
                        kind, dte = action.split("_")
                        contract = market.pick_contract(symbol, kind, expiry[dte], closes[-1])
                        price = market.ask(contract) if contract else None
                        qty = int(min(PER_TRADE, room) // (price * 100)) if price else 0
                        if qty >= 1:
                            market.buy(contract, qty)
                            day["spent"] = round(day["spent"] + qty * price * 100, 2)
                            day["open"] = {"symbol": symbol, "contract": contract, "action": action, "qty": qty,
                                           "opened": now.isoformat(timespec="seconds"), "situation": sit,
                                           "features": d["features"], "explored": explored,
                                           "plpc": 0.0, "pl_usd": 0.0, "u_entry": round(float(closes[-1]), 2)}
                        else:
                            day["last_decision"]["note"] = "too expensive or no contract"
                status("in a trade" if day["open"] else
                       "watching for setups" if getattr(agent, "signal_mode", False) else "watching")
        else:
            if now >= flatten:
                break
            status(f"done for the day ({day['stopped']})")
        market.sleep(CHECK_EVERY_S)

    if day["open"]:                                      # safety: never hold overnight
        market.close(day["open"]["contract"])
        day["open"] = None

    status("training overnight")
    agent.overnight()
    summary = (f"{today}: {day['trades']} trades, {day['wins']} winners, "
               f"P&L ${day['pl_usd']:+,.0f}, spent ${day['spent']:,.0f}"
               + (f" (stopped: {day['stopped']})" if day["stopped"] else ""))
    status("session over")
    print(summary)
    notify(summary)
    return day


def main() -> int:
    ap = argparse.ArgumentParser(description="Intraday options paper trader")
    ap.add_argument("--simulate", action="store_true", help="fake market and fast clock")
    ap.add_argument("--date", default="2026-10-29", help="date for --simulate")
    args = ap.parse_args()
    model_path = STATE_DIR / "options_model.joblib"
    if model_path.exists():
        agent = ModelAgent(model_path, STATE_DIR / "options_live.csv")
        explore = MODEL_EXPLORATION
    else:
        agent = Agent(STATE_DIR / "options_agent.json")
        explore = EXPLORATION
    print(f"agent: {agent.mode}, exploration {explore:.0%}")
    market = SimMarket(dt.date.fromisoformat(args.date)) if args.simulate else AlpacaMarket()
    try:
        run_session(market, agent, explore)
    except Exception as exc:
        write_status(phase="failed", error=f"{type(exc).__name__}: {exc}", agent=agent.stats())
        notify(f"Options bot failed: {exc}", priority=5)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
