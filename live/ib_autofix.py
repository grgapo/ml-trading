"""ib_autofix.py (lab03): if IB Gateway is down for 5+ minutes on a trading day, restart it automatically.

Reads the IBKR check's status file (paper-state/ibkr/status.json, written every 2 min).
  down  = gateway not connected, or the status file is more than 6 minutes old
  fix   = `docker restart ib-gateway` once it has been down 5+ minutes; at most once every 15 minutes
  give up after 3 restarts in 2 hours and send a "needs you" alert instead of looping
  alerts: restart done, back up, needs you (same phone alerts as the other bots)
Runs every 2 minutes, weekdays 6:00am-4:30pm New York. Usage: python3 ib_autofix.py [--test]
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
PS = Path(os.environ.get("PAPER_STATE", Path.home() / "projects" / "paper-state"))
STATUS, STATE = PS / "ibkr" / "status.json", PS / "ibkr" / "autofix.json"
CONTAINER = os.environ.get("IB_CONTAINER", "ib-gateway")
DOWN_MIN, STALE_MIN, GAP_MIN, MAX_RESTARTS, WINDOW_H = 5, 6, 15, 3, 2


def notify(title: str, text: str) -> None:
    print(f"{title}: {text}", flush=True)
    t = os.environ.get("NTFY_TOPIC")
    if t:
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{t}", data=text.encode(),
                                   headers={"Title": title, "Tags": "wrench"}), timeout=10)
        except Exception:
            pass


def gateway_ok(now: dt.datetime) -> tuple[bool, str]:
    try:
        s = json.loads(STATUS.read_text())
    except Exception as exc:
        return False, f"no status file ({type(exc).__name__})"
    age = (now - dt.datetime.fromisoformat(s["updated"])).total_seconds() / 60
    if age > STALE_MIN:
        return False, f"IBKR check hasn't reported in {age:.0f} min"
    if not (s.get("gateway") or {}).get("connected"):
        return False, "gateway not connected"
    return True, "ok"


def restart() -> bool:
    return subprocess.run(["docker", "restart", CONTAINER], capture_output=True, timeout=120).returncode == 0


def step(now: dt.datetime, do_restart=restart) -> str:
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    ok, why = gateway_ok(now)
    iso = now.isoformat(timespec="seconds")
    recent = [r for r in st.get("restarts", []) if now - dt.datetime.fromisoformat(r) < dt.timedelta(hours=WINDOW_H)]
    st["restarts"] = recent
    if ok:
        if st.get("down_since") and recent:
            notify("IBKR: gateway back", "IB Gateway is answering again after an automatic restart. The IBKR bot can trade.")
        st.pop("down_since", None); st.pop("gave_up", None); act = "ok"
    else:
        st.setdefault("down_since", iso)
        down = (now - dt.datetime.fromisoformat(st["down_since"])).total_seconds() / 60
        last = max(recent) if recent else None
        if down < DOWN_MIN:
            act = f"down {down:.0f} min ({why}), waiting"
        elif len(recent) >= MAX_RESTARTS:
            act = "gave up"
            if not st.get("gave_up"):
                st["gave_up"] = iso
                notify("IBKR: gateway needs you", f"IB Gateway is still down after {len(recent)} automatic restarts ({why}). "
                       f"The IBKR bot can't trade. Run ib-fix.sh on lab03 or check the IBKR login.")
        elif last and now - dt.datetime.fromisoformat(last) < dt.timedelta(minutes=GAP_MIN):
            act = f"down {down:.0f} min, restarted recently, waiting"
        else:
            good = do_restart()
            st["restarts"] = recent + [iso]
            act = "restarted" if good else "restart failed"
            notify("IBKR: gateway restarted automatically",
                   f"IB Gateway was down {down:.0f} min ({why}). Restarted it ({'ok' if good else 'docker error'}); "
                   f"it logs back in within ~2 min.")
    STATE.parent.mkdir(parents=True, exist_ok=True); STATE.write_text(json.dumps(st))
    return act


def in_hours(now: dt.datetime) -> bool:
    return now.weekday() < 5 and dt.time(6, 0) <= now.time() <= dt.time(16, 30)


def selftest() -> int:
    import tempfile
    global STATUS, STATE
    d = Path(tempfile.mkdtemp()); STATUS, STATE = d / "s.json", d / "a.json"
    os.environ.pop("NTFY_TOPIC", None)
    calls, fails = [], []
    fake = lambda: calls.append(1) or True
    t0 = dt.datetime(2026, 11, 16, 10, 0, tzinfo=NY)
    def at(m, connected=True, age=0):
        STATUS.write_text(json.dumps({"updated": (t0 + dt.timedelta(minutes=m - age)).isoformat(), "gateway": {"connected": connected}}))
        return step(t0 + dt.timedelta(minutes=m), fake)
    def check(name, c): print(("PASS " if c else "FAIL ") + name); c or fails.append(name)
    check("healthy gateway: nothing happens", at(0) == "ok" and not calls)
    check("down 2 min: waits", "waiting" in at(2, False) and not calls)
    check("down 6 min: restarts", at(8, False) == "restarted" and len(calls) == 1)
    check("still down right after: no restart loop", "recently" in at(12, False) and len(calls) == 1)
    check("stale status file counts as down", "recently" in at(14, True, age=10))
    check("back up: clears", at(16) == "ok")
    for m in (40, 60, 80, 100): at(m, False)
    check("never more than 3 restarts in 2 hours", len(calls) <= 3)
    check("then asks for help instead", at(110, False) == "gave up")
    print(f"{len(fails)} failure(s)"); return 1 if fails else 0


if __name__ == "__main__":
    if "--test" in sys.argv:
        raise SystemExit(selftest())
    now = dt.datetime.now(NY)
    if in_hours(now):
        print(step(now))
