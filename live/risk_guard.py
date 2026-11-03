"""Risk guard: an independent check, every minute of the trading day, that the bots and the broker agree,
and that no bot (or the whole account) loses more than its hard limit. It never trusts a bot's own view.

Every minute (9:25-16:20 New York, weekdays):
  1. Unclaimed option positions: an option held at Alpaca that no bot claims (claims: risk/claims/<bot>.json)
     for 3+ minutes -> pause ALL new entries for the day, close the position, alert.
  2. Ghost claims: a bot claims a position Alpaca doesn't have, for 5+ minutes -> alert.
  3. Per-bot daily loss limit (realized today + open P&L of its positions):
        experiments (proving ground) $1,000 · options bots $2,500 each
     -> close that bot's positions, pause it for the rest of the day, alert.
  4. Whole account down 3%+ on the day -> close every option position, pause everything for the day, alert.
     (The portfolio's long-term ETFs are NOT force-sold: selling them in a dip is the wrong move.)
  --smoke (9:31 NY): a live round trip on Alpaca paper with 1 share of SGOV (an ETF no bot holds): buy,
     confirm by position, sell, confirm gone. If the order path doesn't work, everything is paused.
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
    "options-spy": (STATE / "options-spy" / "options_trades.csv", "pl_usd", float(os.environ.get("RISK_LIMIT_OPTIONS", "2500"))),
    "options-qqq": (STATE / "options-qqq" / "options_trades.csv", "pl_usd", float(os.environ.get("RISK_LIMIT_OPTIONS", "2500"))),
    "proving": (STATE / "proving" / "trades.csv", "pnl", float(os.environ.get("RISK_LIMIT_EXPERIMENTS", "1000"))),
}
ACCOUNT_LIMIT = float(os.environ.get("RISK_ACCOUNT_LIMIT_PCT", "0.03"))
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
        return sum(float(r.get(col) or 0) for r in csv.DictReader(path.open()) if r.get("date") == today)
    except Exception:
        return 0.0


def claims() -> dict[str, str]:
    out = {}
    for f in (RISK / "claims").glob("*.json"):
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
            B.set_halt("all", f"unclaimed position {sym}", until=today)
            px = br.sell(sym)
            once(f"orphan-{sym}", "Risk guard: unclaimed position closed",
                 f"{sym} x{p.qty} was held at Alpaca but no bot claimed it for {ORPHAN_GRACE_MIN}+ min. "
                 f"Closed it{f' at ${px:.2f}' if px else ' (close not confirmed yet, retrying)'}; all new entries paused for today.")
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
    # 4. whole account
    acct = tc.get_account()
    eq, last = float(acct.equity), float(acct.last_equity or acct.equity)
    dd = eq / last - 1 if last else 0.0
    if dd <= -ACCOUNT_LIMIT and not B.halted("all")[0]:
        B.set_halt("all", f"account down {dd:.1%} today", until=today)
        closed = [s for s in held_opts if br.sell(s) is not None]
        once("account-limit", "Risk guard: account limit hit",
             f"Account is down {dd:.1%} today (limit {ACCOUNT_LIMIT:.0%}). Closed all options ({', '.join(closed) or 'none'}); "
             f"every bot paused until tomorrow. Portfolio ETFs kept.")
    paused_all, why_all = B.halted("all")
    out = {"updated": now.isoformat(timespec="seconds"), "ok": not state["seen"] and not paused_all,
           "account": {"equity": round(eq, 2), "change": round(eq - last, 2), "change_pct": round(dd, 4), "limit_pct": ACCOUNT_LIMIT},
           "positions": {"options": len(held_opts), "unclaimed": sorted(state["seen"]), "claimed_by": owner},
           "bots": bots, "paused_all": paused_all, "paused_all_why": why_all, "events": events[-20:]}
    B._save(st_path, state)
    B._save(RISK / "status.json", out)
    return out


def smoke(tc, now) -> bool:
    br = B.Broker(tc)
    today = str(now.date())
    try:
        got = br.buy(SMOKE_SYMBOL, 1, "smoke")
        ok_buy = got is not None and got[1] >= 1
        px = br.sell(SMOKE_SYMBOL, 1) if ok_buy else None
        ok = ok_buy and px is not None and br.held(SMOKE_SYMBOL) is None
        detail = f"bought 1 {SMOKE_SYMBOL} at {got[0] if got else '?'}, sold at {px}" if ok else \
                 f"buy confirmed: {ok_buy}, sell confirmed: {px is not None}"
    except Exception as exc:
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    B._save(RISK / "smoke.json", {"updated": now.isoformat(timespec="seconds"), "ok": ok, "detail": detail})
    if not ok:
        B.set_halt("all", f"morning order test failed ({detail})", until=today)
        notify("Risk guard: order test FAILED", f"This morning's live order test on Alpaca paper failed ({detail}). "
                                                f"All new entries are paused for today.")
    else:
        print(f"order test passed: {detail}", flush=True)
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
