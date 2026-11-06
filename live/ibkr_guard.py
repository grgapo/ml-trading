"""IBKR end-of-day guard: 10 minutes before the close, close any option still held at IBKR (nothing overnight),
and alert. Also usable as a health check (--check): is IB Gateway logged in and reachable?
  .venv/bin/python live/ibkr_guard.py [--check]
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import broker_ibkr as BI  # noqa: E402


def notify(title, text):
    t = os.environ.get("NTFY_TOPIC"); print(f"{title}: {text}", flush=True)
    if t:
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{t}", data=text.encode(), headers={"Title": title, "Tags": "warning"}), timeout=10)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    b = BI.IBBroker(97, confirm_s=40)
    try:
        acct = b.ib.managedAccounts()
    except Exception as exc:
        notify("IBKR: gateway unreachable", f"IB Gateway on lab03 isn't answering ({type(exc).__name__}). The IBKR bot can't trade until it's back."); return 1
    if a.check:
        print("IB Gateway OK:", ", ".join(acct)); return 0
    left = b.option_positions()
    closed = []
    for c, q in left:
        occ = f"{c.symbol}{c.lastTradeDateOrContractMonth[2:]}{c.right}{int(round(c.strike * 1000)):08d}"
        b.contracts[occ] = c
        px = b.sell(occ)
        closed.append(f"{occ} x{q:g}" + (f" at ${px:.2f}" if px else " (not confirmed, still trying)"))
    if closed:
        notify("IBKR: option still open at 3:50", "Closed " + ", ".join(closed) + ". Nothing is held overnight.")
    else:
        print("IBKR: no option positions open")
    b.ib.disconnect(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
