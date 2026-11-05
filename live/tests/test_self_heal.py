"""Run: .venv/bin/python live/tests/test_self_heal.py   (no network)
Self-healing rules (Nov 3, 2026): one bot's trouble stops only that bot; nothing is held overnight."""
import os, sys, tempfile, json, shutil, datetime as dt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
base = tempfile.mkdtemp(); os.environ["RISK_DIR"] = os.path.join(base, "risk"); os.environ["PAPER_STATE"] = base
os.environ.pop("NTFY_TOPIC", None)
import broker as B, risk_guard as G
from fake_alpaca import FakeTradingClient, FakeWithAccount
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
failures = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); failures.append(name) if not cond else None
def fresh():
    shutil.rmtree(B.RISK, ignore_errors=True); B.RISK.mkdir(parents=True)
fast = dict(confirm_s=0.3, poll_s=0.02)
OPT = "SPY261104C00680000"

# 1. three orders in a row that don't fill pause ONLY that bot
fresh(); b = B.Broker(FakeTradingClient(fill="none"), **fast)
for _ in range(2):
    b.buy(OPT, 1, "pg", bot="proving")
check("2 unfilled orders: proving still trading", B.halted("proving")[0] is False)
b.buy(OPT, 1, "pg", bot="proving")
check("3rd unfilled order in a row: proving paused", B.halted("proving")[0] is True)
check("...the options bots are not", B.halted("options-spy")[0] is False and B.halted("all")[0] is False)

# 2. a success resets the streak
fresh(); bad = B.Broker(FakeTradingClient(fill="none"), **fast); good = B.Broker(FakeTradingClient(), **fast)
bad.buy(OPT, 1, "pg", bot="proving"); bad.buy(OPT, 1, "pg", bot="proving")
good.buy(OPT, 1, "pg", bot="proving")
bad.buy(OPT, 1, "pg", bot="proving"); bad.buy(OPT, 1, "pg", bot="proving")
check("a fill in between resets the count (2 + fill + 2 = not paused)", B.halted("proving")[0] is False)

# 3. a rejected order returns None (no crash) and counts as a strike
fresh(); b = B.Broker(FakeTradingClient(reject_buys=True), **fast)
check("a rejected buy returns None instead of crashing", b.buy(OPT, 1, "pg", bot="proving") is None)
check("...and counts as a strike", json.loads((B.RISK / "strikes.json").read_text())["proving"] == 1)

# 4. orders without a bot (the morning test) never pause anyone
fresh(); b = B.Broker(FakeTradingClient(fill="none"), **fast)
for _ in range(4):
    b.buy("SGOV", 1, "smoke")
check("order-test failures never pause a bot", not any(B.halted(x)[0] for x in ("all", "proving", "options-spy")))

# 5. a sell that doesn't fill is cancelled and retried
fresh(); c = FakeTradingClient(price=3.0, stuck_sells=1); b = B.Broker(c, **fast)
c.positions[OPT] = [2.0, 2.5]
px = b.sell(OPT)
check("first sell stuck, second try fills: sell confirmed", px == 3.0 and b.held(OPT) is None)
check("...and no order is left hanging", not c.get_orders(type("R", (), {"status": "open", "symbols": None, "limit": None})()))
fresh(); c = FakeTradingClient(stuck_sells=5); b = B.Broker(c, **fast); c.positions[OPT] = [2.0, 2.5]
check("a sell that never fills returns None after 3 tries (bot keeps the trade and retries)", b.sell(OPT) is None and b.held(OPT) == (2.0, 2.5))
check("...and only one alert is recorded for it today", json.loads((B.RISK / "notified.json").read_text())["keys"].count(f"sell-{OPT}") == 1)
b.sell(OPT)
check("...a second failure doesn't add another alert", json.loads((B.RISK / "notified.json").read_text())["keys"].count(f"sell-{OPT}") == 1)

# 6. order ids carry the owner
check("owner read from an options-bot order id", B.bot_from_order_id(f"options-spy-{OPT}-buy-1793716261271", OPT) == "options-spy")
check("owner read from a proving-ground order id", B.bot_from_order_id(f"pg-{OPT}-buy-1793716261271", OPT) == "proving")
check("unknown / missing order id -> no owner", B.bot_from_order_id(None, OPT) is None and B.bot_from_order_id("abc-123", OPT) is None)

G_fast = lambda: setattr(B.Broker, "__init__", lambda self, client, confirm_s=1, poll_s=0.02: (setattr(self, "c", client), setattr(self, "confirm_s", 0.3), setattr(self, "poll_s", 0.02)) and None)
orig_init = B.Broker.__init__; G_fast()
T0 = dt.datetime.combine(dt.datetime.now(B.NY).date(), dt.time(10, 0), tzinfo=B.NY)

# 7. a stray position bought by the proving ground (late fill after its cancel) pauses ONLY the proving ground
fresh(); c = FakeWithAccount()
c.submit_order(MarketOrderRequest(symbol=OPT, qty=1, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                                  client_order_id=f"pg-{OPT}-buy-1793716261271"))
c.get_all_positions()                                  # it fills; the bot already gave up on it
G.run(c, T0); G.run(c, T0 + dt.timedelta(minutes=4))
check("stray proving-ground position is closed", c.positions.get(OPT, [0])[0] == 0)
check("...only the proving ground is paused", B.halted("proving")[0] is True and B.halted("all")[0] is False)
check("...the options bots keep trading", B.halted("options-spy")[0] is False and B.halted("options-qqq")[0] is False)

# 8. same for an options bot
fresh(); c = FakeWithAccount()
c.submit_order(MarketOrderRequest(symbol=OPT, qty=2, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                                  client_order_id=f"options-qqq-{OPT}-buy-1793716261271"))
c.get_all_positions()
G.run(c, T0); G.run(c, T0 + dt.timedelta(minutes=4))
check("stray options-qqq position: only options-qqq paused", B.halted("options-qqq")[0] and not B.halted("options-spy")[0] and not B.halted("all")[0])

# 9. owner unknown (e.g. bought by hand / old code): everything pauses, as before
fresh(); c = FakeWithAccount(); c.hold(OPT, 3, 2.0)
G.run(c, T0); G.run(c, T0 + dt.timedelta(minutes=4))
check("unknown owner: everything paused (safe default)", B.halted("all")[0] is True)

# 10. 3:50 sweep: an option still open (even a claimed one) is closed; nothing held overnight
fresh(); c = FakeWithAccount(); c.hold(OPT, 1, 2.0); B.write_claims("options-spy", {OPT: 1})
G.run(c, dt.datetime.combine(dt.datetime.now(B.NY).date(), dt.time(15, 40), tzinfo=B.NY))
check("a claimed option at 3:40 is left alone", c.positions[OPT][0] == 1)
G.run(c, dt.datetime.combine(dt.datetime.now(B.NY).date(), dt.time(15, 51), tzinfo=B.NY))
check("at 3:51 it is closed", c.positions.get(OPT, [0])[0] == 0)
check("...without pausing anyone", not B.halted("all")[0] and not B.halted("options-spy")[0])

# 11. pauses expire on their own
fresh(); B.set_halt("proving", "3 orders in a row didn't fill", until=str(dt.datetime.now(B.NY).date() - dt.timedelta(days=1)))
check("yesterday's pause doesn't carry over", B.halted("proving")[0] is False)

B.Broker.__init__ = orig_init
print(f"\n{len(failures)} failure(s)"); sys.exit(1 if failures else 0)
