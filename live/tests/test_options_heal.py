"""Run: .venv/bin/python live/tests/test_options_heal.py   (no network, fake market + fast clock)
The options bot heals itself: data/API errors don't kill the session, an unconfirmed exit is retried,
order failures count toward pausing only this bot, and the size cap applies."""
import os, sys, tempfile, json, datetime as dt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
base = tempfile.mkdtemp(); os.environ["RISK_DIR"] = os.path.join(base, "risk")
os.environ["STATE_DIR"] = os.path.join(base, "options-spy"); os.environ.pop("NTFY_TOPIC", None)
os.environ["OPT_FIRST_ENTRY"] = "09:35"
import broker as B, options_bot as O
failures = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); failures.append(name) if not cond else None
DAY = dt.date(2026, 11, 4)

class Agent:
    mode, signal_mode = "test", True
    def __init__(self): self.trades = 0
    def stats(self): return {}
    def decide(self, symbol, bars, now, explore):
        return {"situation": "x", "explored": False, "action": 1, "preds": {}, "features": {}}
    def learn_trade(self, t, reward, day): self.trades += 1
    def overnight(self): pass
    def save(self): pass

class Market(O.SimMarket):
    simulated = False                                   # exercise the live-only paths (strikes, size cap)
    def __init__(self, *a, bar_errors=0, close_fails=0, fill=True, **k):
        super().__init__(*a, **k); self.bar_errors, self.close_fails, self.fill = bar_errors, close_fails, fill
        self.close_calls, self.bought = 0, []
    def bars(self, symbol):
        if self.bar_errors > 0:
            self.bar_errors -= 1; raise ConnectionError("data API blip")
        return super().bars(symbol)
    def buy(self, contract, qty):
        self.bought.append(qty)
        if self.fill: super().buy(contract, qty)
    def close(self, contract):
        self.close_calls += 1
        if self.close_fails > 0:
            self.close_fails -= 1; return False
        super().close(contract); return True

def fresh():
    import shutil; shutil.rmtree(base, ignore_errors=True); B.RISK.mkdir(parents=True); O.STATE_DIR.mkdir(parents=True)

# 1. three data errors early on: the session survives and still trades
fresh(); m = Market(DAY, bar_errors=3); a = Agent()
day = O.run_session(m, a, 0.0)
check("data errors don't end the session (it still trades)", day["trades"] >= 1)
check("...and 3 errors don't pause the bot", B.halted("options-spy")[0] is False)

# 2. five errors in a row: entries paused for this bot only, the session keeps running
fresh(); m = Market(DAY, bar_errors=6); day = O.run_session(m, Agent(), 0.0)
check("5 errors in a row pause only this bot", B.halted("options-spy")[0] and not B.halted("all")[0] and not B.halted("options-qqq")[0])

# 3. an exit that isn't confirmed is retried; the trade is only logged once it's really closed
fresh(); m = Market(DAY, close_fails=2); day = O.run_session(m, Agent(), 0.0)
rows = list(__import__("csv").DictReader((O.STATE_DIR / "options_trades.csv").open()))
check("unconfirmed exits are retried (close called more than once per trade)", m.close_calls > len(rows) >= 1)
check("...and every logged trade was actually closed", not m.pos)

# 4. orders that never fill: 3 in a row pause only this bot
fresh(); m = Market(DAY, fill=False); day = O.run_session(m, Agent(), 0.0)
check("3 unfilled orders in a row: options-spy paused", B.halted("options-spy")[0] and not B.halted("all")[0])
check("...and it stopped sending orders after that", len(m.bought) == 3)

# 5. size cap on the first live day
fresh(); B._save(B.RISK / "size_cap.json", {"options-spy": {"max_qty": 1, "until": "2999-01-01"}})
m = Market(DAY); O.run_session(m, Agent(), 0.0)
check("size cap: every order is 1 contract", m.bought and all(q == 1 for q in m.bought))

print(f"\n{len(failures)} failure(s)"); sys.exit(1 if failures else 0)
