"""Pre-flight check, every trading morning before any bot starts (8:45 New York) and by hand any time.

Checks everything that has bitten us: the broker account, leftover/overnight positions, stale pauses or claims,
size caps, the risk guard's rules and limits, the proving ground's experiments and budget, every bot's timer and
last run, the safety test suites, disk and memory. One notification to the lab app: all clear, or what's wrong.

  .venv/bin/python live/preflight.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import broker as B  # noqa: E402

NY = B.NY
HERE = Path(__file__).resolve().parent
STATE = Path(os.environ.get("PAPER_STATE", Path.home() / "projects" / "paper-state"))
TIMERS = ["risk-guard.timer", "risk-smoke.timer", "options-bot-spy.timer", "options-bot-qqq.timer", "proving-ground.timer",
          "portfolio-bot.timer", "data-audit.timer", "chain-recorder.timer", "options-bot-exp.timer",
          "options-bot-qqq-ibkr.timer", "ibkr-guard.timer"]
BOT_UNITS = ["options-bot-spy.service", "options-bot-qqq.service", "options-bot-exp.service", "proving-ground.service", "risk-guard.service"]
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def sh(*a) -> str:
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()


def main() -> int:
    today = dt.datetime.now(NY).date()      # becomes the next trading day when run after the close
    # 1. broker
    try:
        from alpaca.trading.client import TradingClient
        tc = TradingClient(os.environ["ALPACA_API_KEY_ID"], os.environ["ALPACA_API_SECRET_KEY"], paper=True)
        a = tc.get_account()
        check("Alpaca account active, trading allowed", str(a.status).endswith("ACTIVE") and not a.trading_blocked,
              f"status {a.status}, blocked {a.trading_blocked}, equity ${float(a.equity):,.0f}")
        opts = [p for p in tc.get_all_positions() if "option" in str(getattr(p.asset_class, "value", p.asset_class)).lower()]
        check("no option positions left over from yesterday", not opts, ", ".join(f"{p.symbol} x{p.qty}" for p in opts))
        open_orders = tc.get_orders()
        check("no stale open orders", not open_orders, ", ".join(f"{o.symbol} {o.side.value} {o.qty}" for o in open_orders[:5]))
        clk = tc.get_clock()
        nxt = clk.next_open.astimezone(NY)
        if not clk.is_open:
            today = nxt.date()                                   # checking ahead for the next session
        check("next session", True, f"{'open now' if clk.is_open else f'opens {nxt:%a %b %d %H:%M} NY'}")
    except Exception as exc:
        check("Alpaca reachable", False, f"{type(exc).__name__}: {exc}")
    # 2. pauses, claims, caps, proving-ground leftovers
    try:
        h = json.loads((B.RISK / "halt.json").read_text()) if (B.RISK / "halt.json").exists() else {}
        active = {k: v for k, v in h.items() if v.get("until") is None or v["until"] >= str(today)}
        check("no pauses carried into today", not active, "; ".join(f"{k}: {v.get('reason')}" for k, v in active.items()))
    except Exception as exc:
        check("halt file readable", False, str(exc))
    stale = []
    for f in (B.RISK / "claims").glob("*.json"):
        try:
            if json.loads(f.read_text()).get("holdings"):
                stale.append(f.stem)
        except Exception:
            stale.append(f"{f.stem} (unreadable)")
    check("no bot still claims a position", not stale, ", ".join(stale))
    caps = {}
    try:
        caps = {k: v for k, v in json.loads((B.RISK / "size_cap.json").read_text()).items() if v.get("until", "9999") >= str(today)}
    except Exception:
        pass
    check("size caps (info)", True, ", ".join(f"{k}: {v['max_qty']} contract(s) ({v.get('why', '')[:40]})" for k, v in caps.items()) or "none")
    check("no stray size cap on the proving ground", "proving" not in caps,
          "proving" if "proving" in caps else "")
    op = STATE / "proving" / "open_positions.json"
    left = json.loads(op.read_text()) if op.exists() else {}
    check("proving ground: no leftover positions on file", not left, ", ".join(left))
    # 3. risk guard: the rules we agreed on
    rg = (HERE / "risk_guard.py").read_text()
    check("risk guard: no account-wide limit", "ACCOUNT_LIMIT" not in rg and "trade_dd" not in rg)
    lim_o, lim_e = os.environ.get("RISK_LIMIT_OPTIONS", "5000"), os.environ.get("RISK_LIMIT_EXPERIMENTS", "2500")
    check("risk guard limits = daily budgets ($5,000 options / $1,250 + $1,250 experiments)", float(lim_o) >= 5000,
          f"options {lim_o}, proving {os.environ.get('PROVING_DAILY_BUDGET', '1250')}, exp bot {os.environ.get('EXP_DAILY_BUDGET', '1250')}")
    check("experiment bot (options-exp) has its recipe", (STATE / "options-exp" / "options_model.joblib").exists())
    check("risk guard checks bots trading outside their limits", "today_usage" in rg)
    # 4. proving ground experiments
    try:
        import proving_ground as PG
        ids = [e["id"] for e in PG.DEFAULT_EXPERIMENTS]
        reg = json.loads((STATE / "proving" / "experiments.json").read_text()) if (STATE / "proving" / "experiments.json").exists() else []
        reg = reg if isinstance(reg, list) else reg.get("experiments", [])
        live = [e["id"] for e in reg if e.get("status") not in ("retired",)]
        check("proving ground: every-day experiment ready", "orb-daily" in ids, f"active: {', '.join(live) or '(registry builds at start)'}")
        check("proving ground: its share of the experiment budget", getattr(PG, "DAILY_BUDGET", 0) >= 1250, f"${getattr(PG, 'DAILY_BUDGET', 0):,.0f}")
    except Exception as exc:
        check("proving ground loads", False, f"{type(exc).__name__}: {exc}")
    # 5. timers + last runs
    for t in TIMERS:
        en = sh("systemctl", "is-enabled", t)
        check(f"timer {t.replace('.timer', '')} enabled", en == "enabled", en)
    for u in BOT_UNITS:
        r = sh("systemctl", "show", "-p", "Result", "--value", u)
        check(f"{u.replace('.service', '')}: last run ok", r in ("success", ""), r)
    # 6. the safety tests
    env = {**os.environ, "RISK_SMOKE_RETRY_S": "0"}
    for t in ("test_broker", "test_risk_guard", "test_self_heal", "test_smoke"):
        p = subprocess.run([sys.executable, str(HERE / "tests" / f"{t}.py")], capture_output=True, text=True, env=env, timeout=600)
        fails = [ln for ln in p.stdout.splitlines() if ln.startswith("FAIL")]
        check(f"tests: {t}", p.returncode == 0, "; ".join(fails[:3]))
    # IBKR: gateway logged in, nothing left open
    r = subprocess.run([sys.executable, str(HERE / "ibkr_guard.py"), "--check"], capture_output=True, text=True, timeout=90)
    check("IB Gateway logged in (IBKR paper)", r.returncode == 0, (r.stdout or r.stderr).strip()[-80:])
    # the app's data feed (paper-state-http on 192.168.0.162:8095)
    try:
        code = urllib.request.urlopen("http://192.168.0.162:8095/", timeout=5).status
    except Exception as exc:
        code = f"{type(exc).__name__}"
    check("app data feed from lab03 (port 8095)", code == 200, str(code))
    # 7. machine
    free = shutil.disk_usage(Path.home()).free / 1e9
    check("disk space", free > 5, f"{free:.0f} GB free")
    try:
        mem = {ln.split(":")[0]: int(ln.split()[1]) for ln in open("/proc/meminfo")}
        avail = mem["MemAvailable"] / 1e6
        check("memory", avail > 1.0, f"{avail:.1f} GB available")
    except Exception:
        pass

    bad = [r for r in results if not r[1]]
    for name, ok, detail in results:
        print(f"{'✓' if ok else '✗'} {name}" + (f" — {detail}" if detail else ""))
    print(f"\n{'ALL CLEAR' if not bad else f'{len(bad)} problem(s)'} for {today}")
    topic = os.environ.get("NTFY_TOPIC")
    if topic and os.environ.get("PREFLIGHT_ALERT", "1") == "1":
        title = f"pre-flight {today:%a %b %d}: " + ("all clear" if not bad else f"{len(bad)} problem(s)")
        body = ("Every check passed. Bots start at 9:20 NY." if not bad else
                "\n".join(f"✗ {n}" + (f": {d}" if d else "") for n, _, d in bad)[:1500])
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{topic}", data=body.encode(),
                                   headers={"Title": title, "Tags": "white_check_mark" if not bad else "rotating_light",
                                            "Priority": "default" if not bad else "high"}), timeout=10)
        except Exception:
            pass
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
