"""Daily Dyna-Q paper trader (Alpaca paper account).

Runs once per trading day after the close:
  1. Load daily prices (yfinance, same Adj Close the research code uses).
  2. First run per symbol: pre-train the friend's DynaQ on all history so far.
  3. Every run: learn from yesterday's decision (reward = close-to-close return
     while long, minus a trading cost when the position changed), rehearse with
     the model (planning), then pick today's action.
  4. Send the order to Alpaca paper (queued for the next open) and log it.

State lives in STATE_DIR as JSON + .npy (no pickle). Secrets come from the
environment (.env), never from this file.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import math
import os
import sys
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RL_DIR = REPO / "src" / "ml" / "reinforcement-learning"
sys.path.insert(0, str(REPO))
NY = ZoneInfo("America/New_York")


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, RL_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dyna_q = _load("dyna_q", "dyna-q.py")
q_learning = dyna_q.q_learning   # the exact module DynaQ inherits from
FLAT, LONG, BUY, SELL, HOLD = (q_learning.FLAT, q_learning.LONG,
                               q_learning.BUY, q_learning.SELL, q_learning.HOLD)
ACTION = q_learning.ACTION_NAME

# ---- settings (override with environment variables) -------------------------
SYMBOLS = os.environ.get("SYMBOLS", "SPY").replace(" ", "").split(",")
DOLLARS_PER_SYMBOL = float(os.environ.get("DOLLARS_PER_SYMBOL", "10000"))
COST_BPS = float(os.environ.get("COST_BPS", "5"))          # per position change
ALPHA = float(os.environ.get("ALPHA", "0.0001"))
PLANNING_STEPS = int(os.environ.get("PLANNING_STEPS", "5"))
PRETRAIN_EPOCHS = int(os.environ.get("PRETRAIN_EPOCHS", "1000"))
LIVE_EXPLORATION = float(os.environ.get("LIVE_EXPLORATION", "0.05"))
STATE_DIR = Path(os.environ.get("STATE_DIR", Path.home() / "projects" / "paper-state"))


# ---- prices -----------------------------------------------------------------
def load_prices(symbol: str, csv_path: str | None = None) -> pd.DataFrame:
    """Return Date + Adj Close, oldest first, only completed sessions."""
    if csv_path:
        raw = pd.read_csv(csv_path)
    else:
        import yfinance as yf
        raw = yf.download(symbol, period="10y", auto_adjust=False,
                          multi_level_index=False, progress=False).reset_index()
    df = raw[["Date", "Adj Close"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"{symbol}: no price data")
    return df


# ---- live status (read by the lab control app) ----------------------------------
RUN = {"started": None, "symbols": SYMBOLS, "done": []}


def write_status(phase: str, **info) -> None:
    """Atomically write STATE_DIR/status.json so the app can show progress."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    data = {"phase": phase, "updated": dt.datetime.now(NY).isoformat(timespec="seconds"),
            "started": RUN["started"], "symbols": RUN["symbols"], "done": RUN["done"], **info}
    tmp = STATE_DIR / "status.json.tmp"
    tmp.write_text(json.dumps(data))
    tmp.replace(STATE_DIR / "status.json")


class ProgressDynaQ(dyna_q.DynaQ):
    """Friend's DynaQ, unchanged, plus a progress report every few epochs.

    train() asks for the exploration rate at the start of every epoch, so that
    hook tells us which epoch is running without touching the training code.
    """
    symbol = "?"
    _t0 = None

    def exploration_rate(self, epoch: int) -> float:
        import time
        if epoch == 0 or self._t0 is None:
            self._t0 = time.time()
        if epoch % 10 == 0 or epoch == self.epochs - 1:
            elapsed = time.time() - self._t0
            per_epoch = elapsed / epoch if epoch else None
            write_status("pre-training", symbol=self.symbol, epoch=epoch, epochs=self.epochs,
                         seconds_left=round(per_epoch * (self.epochs - epoch)) if per_epoch else None)
        return super().exploration_rate(epoch)


# ---- agent persistence --------------------------------------------------------
def new_agent(df: pd.DataFrame, symbol: str = "?") -> "dyna_q.DynaQ":
    agent = ProgressDynaQ(df, alpha=ALPHA, planning_steps=PLANNING_STEPS,
                          epochs=PRETRAIN_EPOCHS)
    agent.symbol = symbol
    return agent


def save_agent(agent, symbol: str, extra: dict) -> None:
    d = STATE_DIR / symbol
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "q_table.npy", agent.q_table)
    model = [[s, a, r, ns, npos, done]
             for (s, a), (r, ns, npos, done) in agent.model.items()]
    tmp = d / "state.json.tmp"
    tmp.write_text(json.dumps({"model": model, **extra}, indent=1))
    tmp.replace(d / "state.json")


def load_agent(agent, symbol: str) -> dict | None:
    d = STATE_DIR / symbol
    if not (d / "state.json").exists():
        return None
    agent.q_table = np.load(d / "q_table.npy")
    state = json.loads((d / "state.json").read_text())
    agent.model = {(int(s), int(a)): (float(r), int(ns), int(npos), bool(done))
                   for s, a, r, ns, npos, done in state.pop("model")}
    return state


def pretrain(agent, last_day: pd.Timestamp) -> None:
    """Friend's DynaQ.train on everything up to and including last_day."""
    q_learning.CUTOFF = str((last_day + pd.Timedelta(days=1)).date())
    agent.train()


# ---- broker -------------------------------------------------------------------
class Broker:
    """Alpaca paper account. All orders are day market orders queued for the open."""

    def __init__(self) -> None:
        from alpaca.trading.client import TradingClient
        key, secret = os.environ.get("ALPACA_API_KEY_ID"), os.environ.get("ALPACA_API_SECRET_KEY")
        if not key or not secret:
            raise SystemExit("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY missing (see .env)")
        self.client = TradingClient(key, secret, paper=True)   # paper only, on purpose

    def traded_today(self, today: dt.date) -> bool:
        from alpaca.trading.requests import GetCalendarRequest
        days = self.client.get_calendar(GetCalendarRequest(start=today, end=today))
        return any(day.date == today for day in days)

    def shares(self, symbol: str) -> float:
        try:
            return float(self.client.get_open_position(symbol).qty)
        except Exception:
            return 0.0

    def equity(self) -> float:
        return float(self.client.get_account().equity)

    def buy(self, symbol: str, qty: int) -> str:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        order = self.client.submit_order(MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
        return str(order.id)

    def sell_all(self, symbol: str) -> str:
        return str(self.client.close_position(symbol).id)


class DryRunBroker:
    """Pretend broker for testing: remembers positions in a JSON file."""

    def __init__(self) -> None:
        self.path = STATE_DIR / "dry_run_positions.json"
        self.pos = json.loads(self.path.read_text()) if self.path.exists() else {}

    def traded_today(self, today: dt.date) -> bool:
        return today.weekday() < 5

    def shares(self, symbol: str) -> float:
        return float(self.pos.get(symbol, 0))

    def equity(self) -> float:
        return float("nan")

    def _save(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.pos))

    def buy(self, symbol: str, qty: int) -> str:
        self.pos[symbol] = qty
        self._save()
        return "dry-run"

    def sell_all(self, symbol: str) -> str:
        self.pos[symbol] = 0
        self._save()
        return "dry-run"


# ---- one symbol, one day ------------------------------------------------------
def step_symbol(symbol: str, broker, today: dt.date, csv_path: str | None) -> dict:
    write_status("downloading prices", symbol=symbol)
    df = load_prices(symbol, csv_path)
    df = df[df["Date"] <= pd.Timestamp(today)].reset_index(drop=True)
    if df["Date"].iloc[-1].date() != today:
        raise RuntimeError(f"{symbol}: today's close not available yet "
                           f"(latest {df['Date'].iloc[-1].date()})")
    t = len(df) - 1
    agent = new_agent(df, symbol)               # adds Price_to_SMA_Ratio to df
    state = load_agent(agent, symbol)

    if state is None:                           # first run: learn from history
        print(f"{symbol}: no saved agent, pre-training {PRETRAIN_EPOCHS} epochs...")
        pretrain(agent, df["Date"].iloc[t - 1])
        state = {"last_date": None}
        learned = "pre-trained"
    elif state["last_date"] == str(today):
        return {"symbol": symbol, "skipped": "already ran today"}
    else:
        # Learn from the decision made at the previous run.
        write_status("learning from yesterday", symbol=symbol)
        prev_t = int(np.flatnonzero(df["Date"] == pd.Timestamp(state["last_date"]))[0])
        pos = int(state["position_after"])
        ret = float(df["Adj Close"].iloc[t] / df["Adj Close"].iloc[prev_t] - 1.0)
        reward = (ret if pos == LONG else 0.0)
        if int(state["action"]) != HOLD:
            reward -= COST_BPS / 10_000
        next_id = agent.state_id(pos, agent.market_bin(t))
        args = (int(state["state_id"]), int(state["action"]), reward, next_id, pos, False)
        agent.update(*args)
        agent.record_transition(*args)
        agent.plan()
        learned = f"reward {reward:+.4%}"

    # Decide today's action from the broker's real position.
    write_status("deciding + ordering", symbol=symbol)
    held = broker.shares(symbol)
    position = LONG if held > 0 else FLAT
    bin_now = agent.market_bin(t)
    state_id = agent.state_id(position, bin_now)
    agent.exploration_prob = LIVE_EXPLORATION
    action = agent.select_action(state_id, position)
    target = agent._update_state(position, action)

    price = float(df["Adj Close"].iloc[t])
    order = ""
    if position == FLAT and target == LONG:
        qty = math.floor(DOLLARS_PER_SYMBOL / price)
        if qty < 1:
            raise RuntimeError(f"{symbol}: ${DOLLARS_PER_SYMBOL} buys less than 1 share")
        order = f"buy {qty} ({broker.buy(symbol, qty)})"
    elif position == LONG and target == FLAT:
        order = f"sell all ({broker.sell_all(symbol)})"

    save_agent(agent, symbol, {
        "last_date": str(today), "state_id": int(state_id), "action": int(action),
        "position_after": int(target), "market_bin": int(bin_now),
    })
    return {"symbol": symbol, "date": str(today), "close": round(price, 2),
            "bin": int(bin_now), "held": held, "action": ACTION[action],
            "order": order or "none", "learned": learned}


# ---- reporting ----------------------------------------------------------------
def log_rows(rows: list[dict], equity: float) -> None:
    path = STATE_DIR / "trades.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "symbol", "close", "bin", "held",
                                          "action", "order", "learned", "equity"])
        if new:
            w.writeheader()
        for r in rows:
            if "skipped" not in r:
                w.writerow({**r, "equity": round(equity, 2)})


def notify(text: str, priority: int = 3) -> None:
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        return
    req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=text.encode(),
                                 headers={"Title": "Paper bot", "Priority": str(priority)})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:  # a failed notification must not fail the run
        print(f"ntfy failed: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="no Alpaca, fake positions")
    ap.add_argument("--date", help="pretend today is YYYY-MM-DD (testing)")
    ap.add_argument("--csv", help="price CSV instead of yfinance (testing, one symbol)")
    args = ap.parse_args()

    today = dt.date.fromisoformat(args.date) if args.date else dt.datetime.now(NY).date()
    RUN["started"] = dt.datetime.now(NY).isoformat(timespec="seconds")
    write_status("starting", session=str(today), dry_run=args.dry_run)
    broker = DryRunBroker() if args.dry_run else Broker()
    if not broker.traded_today(today):
        print(f"{today}: market closed, nothing to do")
        write_status("idle", result=f"{today}: market closed")
        return 0

    rows, failures = [], []
    for symbol in SYMBOLS:
        try:
            rows.append(step_symbol(symbol, broker, today, args.csv))
            RUN["done"].append(symbol)
        except Exception as exc:
            failures.append(f"{symbol}: {exc}")
    equity = broker.equity()
    log_rows(rows, equity)

    lines = [f"{r['symbol']}: {r.get('skipped') or r['action'] + ', order ' + r['order']}"
             for r in rows] + [f"FAILED {f}" for f in failures]
    summary = f"{today} equity ${equity:,.0f}\n" + "\n".join(lines)
    print(summary)
    write_status("failed" if failures else "idle", session=str(today), result=summary)
    notify(summary, priority=5 if failures else 3)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
