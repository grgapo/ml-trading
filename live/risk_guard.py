"""Risk guard: an independent check, every minute of the trading day, that the bots and the broker agree
and that no bot trades outside its own limits. It never trusts a bot's own view, and it only steps in when
something is actually wrong: normal market days (even big ones) never pause anything.

Every minute (9:25-16:20 New York, weekdays):
  1. Unclaimed option positions: an option held at Alpaca that no bot claims (claims: risk/claims/<bot>.json)
     for 3+ minutes -> close the position, alert, and pause ONLY the bot whose order bought it (found from the
     tag in the order id). Everything is paused only if the owner can't be identified.
  2. Ghost claims: a bot claims a position Alpaca doesn't have, for 5+ minutes -> alert.
  3. Per-bot daily loss beyond its whole daily budget (realized today + open P&L): options bots $5,000 each,
     experiments $2,500 shared. A bot that only buys options can't lose more than it spends, so this only
     fires on a malfunction -> close that bot's positions, pause only that bot for the day, alert.
  4. Trading outside its limits: a bot that spent more than its daily budget (options bots $5,000 each,
     experiments $2,500) or made more trades than allowed (options bots 5), per its tagged orders at
     Alpaca -> pause only that bot for the day, alert. (No account-wide limit: the portfolio's ETFs move
     with the market and that is never a reason to stop the trading bots.)
  5. From 10 min before the close (15:50 NY on normal days): any option still held is closed (every bot should be flat by 15:45; nothing is held overnight).
  --smoke (9:35 NY): a live round trip on Alpaca paper with 1 share of SGOV (an ETF no bot holds): buy,
     confirm by position, sell, confirm gone. Alert-only: a failure never pauses the bots.
Every pause expires at the end of the day, so each bot starts fresh the next morning on its own.
Status for the app/watchdog: risk/status.json. Pauses: risk/halt.json (bots check it before every entry).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import broker as B  # noqa: E402

NY = B.NY
STATE = Path(os.environ.get("PAPER_STATE", Path.home() / "projects" / "paper-state"))
RISK = B.RISK
BOTS = {   # bot id -> (trade log, P&L column, daily loss limit)
    "options-spy": (STATE / "options-spy" / "options_trades.csv", "pl_usd", float(os.environ.get("RISK_LIMIT_OPTIONS", "5000"))),
    "options-qqq": (STATE / "options-qqq" / "options_trades.csv", "pl_usd", float(os.environ.get("RISK_LIMIT_OPTIONS", "5000"))),
    "proving": (STATE / "proving" / "trades.csv", "pnl", float(os.environ.get("PROVING_DAILY_BUDGET", "1250"))),
    "options-exp": (STATE / "options-exp" / "options_trades.csv", "pl_usd", float(os.environ.get("EXP_DAILY_BUDGET", "1250"))),
}
BUDGET = {"options-spy": float(os.environ.get("OPT_DAILY_BUDGET", "5000")), "options-qqq": float(os.environ.get("OPT_DAILY_BUDGET", "5000")),
          "proving": float(os.environ.get("PROVING_DAILY_BUDGET", "1250")), "options-exp": float(os.environ.get("EXP_DAILY_BUDGET", "1250"))}
MAX_TRADES = {"options-spy": int(os.environ.get("OPT_MAX_TRADES", "5")), "options-qqq": int(os.environ.get("OPT_MAX_TRADES", "5")),
              "options-exp": int(os.environ.get("EXP_MAX_TRADES", "4"))}
BUDGET_SLACK = 1.10                     # fills can come in a little above the price a bot sized its order at
ORPHAN_GRACE_MIN, GHOST_GRACE_MIN = 3, 5
SMOKE_SYMBOL = os.environ.get("RISK_SMOKE_SYMBOL", "SGOV")


def notify(title: str, text: str, priority: str = "high") -> None:
    topic = os.environ.get("NTFY_TOPIC")
    print(f"[{priority}] {title}: {text}", flush=True)
    if topic:
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{topic}", data=text.encode(),
                                                          headers={"Title": title, "Priority": priority, "Tags": "rotating_light"}), timeout=10)
        except Exception:
            pass


def realized_today(path: Path, col: str, today: str) -> float:
    if not path.exists():
        return 0.0
    try:
        return sum(float(r.get(col) or 0) for r in csv.DictReader(path.open())
                   if r.get("date") == today and not str(r.get("experiment", "")).endswith("-shadow"))   # shadow = no orders
    except Exception:
        return 0.0


def today_usage(tc, now: dt.datetime) -> dict[str, dict]:
    """Per bot: money spent on buys that filled today and how many buys, from the bot tag in each order id."""
    out: dict[str, dict] = {}
    try:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        start = dt.datetime.combine(now.date(), dt.time(4, 0), NY)
        orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, after=start, limit=500))
    except Exception:
        return out
    for o in orders:
        if str(getattr(o.side, "value", o.side)).lower() != "buy":
            continue
        qty = float(getattr(o, "filled_qty", None) or 0) if hasattr(o, "filled_qty") else (float(o.qty) if o.filled_avg_price else 0.0)
        px = float(o.filled_avg_price or 0)
        if qty <= 0 or px <= 0:
            continue
        bot = B.bot_from_order_id(getattr(o, "client_order_id", None), o.symbol)
        if not bot:
            continue
        mult = 100 if len(o.symbol) > 10 else 1
        u = out.setdefault(bot, {"spent": 0.0, "buys": 0}); u["spent"] += qty * px * mult; u["buys"] += 1
    return out


def claims() -> dict[str, str]:
    out = {}
    for f in (RISK / "claims").glob("*.json"):
        if f.stem.endswith("-ibkr"):                    # IBKR bots hold positions at IBKR, not Alpaca (ibkr_guard watches those)
            continue
        try:
            for sym, q in json.loads(f.read_text()).get("holdings", {}).items():
                if q:
                    out[sym] = f.stem
        except Exception:
            pass
    return out


def run(tc, now: dt.datetime) -> dict:
    br = B.Broker(tc)
    today = str(now.date())
    st_path = RISK / "state.json"
    state = json.loads(st_path.read_text()) if st_path.exists() else {}
    if state.get("day") != today:
        state = {"day": today, "seen": {}, "ghost": {}, "events": [], "notified": []}
    events = state["events"]

    def once(key: str, title: str, text: str) -> None:
        if key not in state["notified"]:
            state["notified"].append(key); events.append(f"{now:%H:%M} {text}"); notify(title, text)

    positions = tc.get_all_positions()
    held_opts = {p.symbol: p for p in positions if "option" in str(getattr(p.asset_class, "value", p.asset_class)).lower()}
    owner = claims()
    ts = now.timestamp()
    # 1. unclaimed option positions
    for sym, p in held_opts.items():
        if sym in owner:
            state["seen"].pop(sym, None); continue
        first = state["seen"].setdefault(sym, ts)
        if ts - first >= ORPHAN_GRACE_MIN * 60:
            who = B.owner_of(tc, sym)
            key = who if who in BOTS else "all"
            B.set_halt(key, f"unclaimed position {sym}", until=today)
            px = br.sell(sym)
            scope = f"{who} paused for today (it placed the order); the other bots keep trading" if key != "all" else \
                    "owner unknown, so all new entries are paused for today"
            once(f"orphan-{sym}", "Risk guard: unclaimed position closed",
                 f"{sym} x{p.qty} was held at Alpaca but no bot claimed it for {ORPHAN_GRACE_MIN}+ min. "
                 f"Closed it{f' at ${px:.2f}' if px else ' (close not confirmed yet, retrying)'}; {scope}.")
    for sym in list(state["seen"]):
        if sym not in held_opts:
            state["seen"].pop(sym)
    # 2. ghost claims
    held_all = {p.symbol for p in positions}
    for sym, bot in owner.items():
        if sym in held_all:
            state["ghost"].pop(sym, None); continue
        first = state["ghost"].setdefault(sym, ts)
        if ts - first >= GHOST_GRACE_MIN * 60:
            once(f"ghost-{sym}", "Risk guard: bot and broker disagree",
                 f"{bot} believes it holds {sym}, but Alpaca shows no such position (for {GHOST_GRACE_MIN}+ min).")
    # 3. per-bot daily loss limits
    bots = {}
    for bot, (log, col, limit) in BOTS.items():
        open_pl = sum(float(held_opts[s].unrealized_pl or 0) for s, b in owner.items() if b == bot and s in held_opts)
        pnl = realized_today(log, col, today) + open_pl
        paused, why = B.halted(bot)
        bots[bot] = {"pnl_today": round(pnl, 2), "limit": limit, "paused": paused, "why": why}
        if pnl <= -limit and not paused:
            B.set_halt(bot, f"daily loss limit (${limit:,.0f}) hit: {pnl:+,.0f}", until=today)
            closed = [s for s, b in owner.items() if b == bot and s in held_opts and br.sell(s) is not None]
            once(f"limit-{bot}", f"Risk guard: {bot} stopped for today",
                 f"{bot} lost ${-pnl:,.0f} today (limit ${limit:,.0f}). Closed {', '.join(closed) or 'nothing (no open positions)'}; "
                 f"no new trades from it until tomorrow.")
            bots[bot]["paused"] = True
    # 4. trading outside its own limits (budget / trade count), from the bots' tagged orders at Alpaca
    acct = tc.get_account()
    eq, last = float(acct.equity), float(acct.last_equity or acct.equity)
    dd = eq / last - 1 if last else 0.0
    usage = today_usage(tc, now)
    for bot, u in usage.items():
        if bot not in bots:
            continue
        bots[bot].update(spent_today=round(u["spent"], 2), buys_today=u["buys"], budget=BUDGET.get(bot))
        over_budget = bot in BUDGET and u["spent"] > BUDGET[bot] * BUDGET_SLACK
        over_trades = bot in MAX_TRADES and u["buys"] > MAX_TRADES[bot]
        if (over_budget or over_trades) and not B.halted(bot)[0]:
            what = (f"spent ${u['spent']:,.0f} (budget ${BUDGET[bot]:,.0f})" if over_budget else
                    f"made {u['buys']} buys (limit {MAX_TRADES[bot]})")
            B.set_halt(bot, f"traded outside its limits: {what}", until=today)
            once(f"outside-{bot}", f"Risk guard: {bot} paused (outside its limits)",
                 f"{bot} {what} today, which it should never do. Only {bot} is paused; the other bots keep trading. "
                 f"It keeps managing its open trades and resumes tomorrow.")
            bots[bot]["paused"] = True
    # 5. end of day (10 min before the close, so half days work too): nothing is held overnight
    try:
        nc = getattr(tc.get_clock(), "next_close", None)
    except Exception:
        nc = None
    eod = (nc - now).total_seconds() <= 600 if nc else now.time() >= dt.time(15, 50)
    if eod:
        for sym in list(held_opts):
            px = br.sell(sym)
            once(f"eod-{sym}", "Risk guard: option still open at 3:50",
                 f"{sym} (claimed by {owner.get(sym, 'no bot')}) was still open at 3:50 NY. "
                 f"Closed it{f' at ${px:.2f}' if px else ' (not confirmed yet, retrying every minute)'}.")
    paused_all, why_all = B.halted("all")
    out = {"updated": now.isoformat(timespec="seconds"), "ok": not state["seen"] and not paused_all,
           "account": {"equity": round(eq, 2), "change": round(eq - last, 2), "change_pct": round(dd, 4), "limit_pct": None},
           "positions": {"options": len(held_opts), "unclaimed": sorted(state["seen"]), "claimed_by": owner},
           "bots": bots, "paused_all": paused_all, "paused_all_why": why_all, "events": events[-20:]}
    B._save(st_path, state)
    B._save(RISK / "status.json", out)
    return out


SMOKE_CONFIRM_S = int(os.environ.get("RISK_SMOKE_CONFIRM_S", "180"))
SMOKE_TRIES = int(os.environ.get("RISK_SMOKE_TRIES", "2"))
SMOKE_RETRY_S = float(os.environ.get("RISK_SMOKE_RETRY_S", "60"))


def smoke(tc, now) -> bool:
    """Live 1-share round trip. ALERT-ONLY: a failure sends ntfy but never pauses the bots.
    (Nov 3: a 60 s window at 9:31 missed a paper fill and halted the whole day for nothing.)"""
    import time as _time
    br = B.Broker(tc, confirm_s=SMOKE_CONFIRM_S)
    ok, detail, attempt = False, "", 0
    for attempt in range(1, SMOKE_TRIES + 1):
        try:
            if br.held(SMOKE_SYMBOL):                 # a late fill from a cancelled try: sell it first
                br.sell(SMOKE_SYMBOL)
            got = br.buy(SMOKE_SYMBOL, 1, "smoke")
            ok_buy = got is not None and got[1] >= 1
            px = br.sell(SMOKE_SYMBOL) if ok_buy else None
            ok = ok_buy and px is not None and br.held(SMOKE_SYMBOL) is None
            detail = (f"bought 1 {SMOKE_SYMBOL} at {got[0]}, sold at {px}" if ok else
                      f"buy confirmed: {ok_buy}, sell confirmed: {px is not None}")
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        if ok:
            break
        if attempt < SMOKE_TRIES:
            _time.sleep(SMOKE_RETRY_S)
    B._save(RISK / "smoke.json", {"updated": now.isoformat(timespec="seconds"), "ok": ok,
                                  "detail": detail, "attempts": attempt, "alert_only": True})
    if not ok:
        notify("Risk guard: order test failed (bots NOT paused)",
               f"Morning live order test failed after {attempt} tries ({detail}). Bots keep trading; "
               f"the risk guard's position and loss checks still apply. Worth a look.")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); args = ap.parse_args()
    now = dt.datetime.now(NY)
    from alpaca.trading.client import TradingClient
    tc = TradingClient(os.environ["ALPACA_API_KEY_ID"], os.environ["ALPACA_API_SECRET_KEY"], paper=True)
    if not tc.get_clock().is_open and not args.smoke:
        if now.weekday() < 5 and dt.time(16, 0) <= now.time() <= dt.time(16, 20):
            run(tc, now)                                  # one last pass right after the close
        return 0
    if args.smoke:
        return 0 if smoke(tc, now) else 1
    run(tc, now)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
