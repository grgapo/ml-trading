"""Go-live scorecard: how close each bot is to real money, against the 6 checks we agreed on.

  1 passed the blind test (locked 6-month holdout)       4 worst losing streak no worse than the backtest's
  2 30+ real paper trades over 4+ weeks (since champion)  5 zero incidents in its last 10 trading days
  3 paper avg/trade >= half the backtest's, and positive  6 paper fills within ~5% of the simulator's prices
Writes ~/projects/paper-state/scorecard.json for the app. Read-only.   .venv/bin/python live/scorecard.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
PS = Path.home() / "projects" / "paper-state"
BOTS = {"options-qqq": "Options · QQQ", "options-spy": "Options · SPY", "options-exp": "Experiment · side-search recipe"}
MIN_TRADES, MIN_DAYS, INCIDENT_DAYS = 30, 28, 14


def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def bundle_info(d: Path) -> dict:
    try:
        import joblib
        b = joblib.load(d / "options_model.joblib")
        return {"created": b.get("created"), "desc": b.get("description"), "wf": b.get("walk_forward") or {},
                "experiment": bool(b.get("experiment")), "has_strategy": bool(b.get("strategy"))}
    except Exception:
        return {}


def trades(d: Path, since: dt.date | None) -> list[dict]:
    f = d / "options_trades.csv"
    if not f.exists():
        return []
    rows = [r for r in csv.DictReader(f.open()) if r.get("pl_usd")]
    return [r for r in rows if since is None or r["date"] >= str(since)]


def max_drawdown(pnls: list[float]) -> float:
    peak = run = dd = 0.0
    for x in pnls:
        run += x; peak = max(peak, run); dd = min(dd, run - peak)
    return round(dd, 2)


def incidents(bot: str) -> list[str]:
    since = (dt.datetime.now() - dt.timedelta(days=INCIDENT_DAYS)).strftime("%Y-%m-%d")
    pat = re.compile(rf"{re.escape(bot)}.*(stopped for today|outside its limits|unclaimed|paused for today|5 errors in a row)|"
                     rf"unclaimed position closed.*{re.escape(bot)}", re.I)
    out = []
    for unit in ("risk-guard", f"options-bot-{bot.split('-', 1)[1]}"):
        try:
            txt = subprocess.run(["journalctl", "-u", unit, "--since", since, "--no-pager", "-o", "short-iso"],
                                 capture_output=True, text=True, timeout=60).stdout
        except Exception:
            continue
        out += [ln[:19] + " " + ln.split(": ", 1)[-1][:120] for ln in txt.splitlines() if pat.search(ln)]
    return out


def check(name, ok, value, need):
    return {"name": name, "ok": ok, "value": value, "need": need}


def score(bot: str) -> dict:
    d = PS / bot
    champ = load_json(d / "search" / "champion.json") or {}
    b = bundle_info(d)
    passed = bool(champ.get("params")) and not b.get("experiment")
    start = None
    if b.get("created"):
        start = dt.datetime.fromisoformat(str(b["created"])).date()
    rows = trades(d, start if passed else None)
    pnls = [float(r["pl_usd"]) for r in rows]
    days = (dt.datetime.now(NY).date() - start).days if start else 0
    avg = round(sum(pnls) / len(pnls), 2) if pnls else None
    # what the backtest expected per trade
    wf = b.get("wf") or {}
    exp_total = (wf.get("model") or {}).get("pnl") or champ.get("holdout_pnl")
    exp_trades = (wf.get("model") or {}).get("trades") or champ.get("holdout_trades")
    exp_avg = round(exp_total / exp_trades, 2) if exp_total and exp_trades else None
    exp_dd = (wf.get("model") or {}).get("max_drawdown") or champ.get("holdout_max_drawdown")
    live_dd = max_drawdown(pnls)
    inc = incidents(bot)
    simf = load_json(d / "search" / "sim_check_real.json") or {}
    fill = (simf.get("summary") or {}).get("entry_abs_err_median")
    checks = [
        check("passed the blind test", passed, champ.get("desc") if passed else ("experiment (not tested blind yet)" if b.get("experiment") else "no champion yet"), "a champion that passed the locked 6-month test"),
        check("30+ trades over 4+ weeks", (len(pnls) >= MIN_TRADES and days >= MIN_DAYS) if passed else False,
              f"{len(pnls)} trades over {days} days" if passed else f"{len(pnls)} trades (clock starts with a champion)", f"{MIN_TRADES}+ trades and {MIN_DAYS}+ days"),
        check("paper results match the backtest", (avg is not None and avg > 0 and (exp_avg is None or avg >= exp_avg / 2)) if len(pnls) >= 10 else None,
              f"${avg:+,.2f}/trade" + (f" vs ${exp_avg:+,.2f} expected" if exp_avg is not None else "") if avg is not None else "no trades yet",
              "positive, and at least half the backtest's average"),
        check("losing streak no worse than backtest", (exp_dd is None or live_dd >= exp_dd) if len(pnls) >= 10 else None,
              f"worst ${live_dd:,.0f}" + (f" vs ${exp_dd:,.0f} in the backtest" if exp_dd is not None else ""), "worst run of losses within the backtest's"),
        check("zero incidents (14 days)", not inc, f"{len(inc)} incident(s)" if inc else "none", "no malfunctions or pauses"),
        check("fills match the simulator", (fill <= 0.05) if fill is not None else None,
              f"{fill:.1%} median difference" if fill is not None else "not measured yet", "within ~5%"),
    ]
    oks = [c["ok"] for c in checks]
    ready_on = None
    if passed and start:
        ready_on = str(max(start + dt.timedelta(days=MIN_DAYS), dt.datetime.now(NY).date()))
    return {"bot": bot, "name": BOTS[bot], "checks": checks, "passed": sum(1 for o in oks if o is True), "of": len(checks),
            "ready": all(o is True for o in oks), "earliest": ready_on, "incidents": inc[-5:], "since": str(start) if start else None}


def main() -> int:
    out = {"updated": dt.datetime.now(NY).isoformat(timespec="seconds"), "bots": [score(b) for b in BOTS if (PS / b).exists()]}
    (PS / "scorecard.json").write_text(json.dumps(out, default=str))
    for b in out["bots"]:
        print(f"{b['name']}: {b['passed']}/{b['of']} checks" + (" · READY" if b["ready"] else "") + (f" · earliest {b['earliest']}" if b["earliest"] else ""))
        for c in b["checks"]:
            print(f"   {'✓' if c['ok'] is True else '✗' if c['ok'] is False else '·'} {c['name']}: {c['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
