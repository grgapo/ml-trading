"""Run the friend's 19-stock Dyna-Q research script and report progress to the lab app.

Nothing in his code changes: this wrapper loads dyna-q-cross-section.py, hooks
two spots (each stock starting / finishing, and each training epoch), and keeps
STATE_DIR/research.json up to date. The lab app shows it as a live card.

  MPLBACKEND=Agg .venv/bin/python live/run_research.py
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "src" / "ml" / "reinforcement-learning" / "dyna-q-cross-section.py"
STATE_DIR = Path(os.environ.get("STATE_DIR", Path.home() / "projects" / "paper-state"))
NY = ZoneInfo("America/New_York")
sys.path.insert(0, str(REPO))

STATUS = {"name": "Dyna-Q cross-section (19 stocks)", "phase": "starting", "started": None,
          "symbol": None, "index": 0, "total": None, "epoch": None, "epochs": None,
          "seconds_left": None, "results": [], "failures": []}


def write() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATUS["updated"] = dt.datetime.now(NY).isoformat(timespec="seconds")
    tmp = STATE_DIR / "research.json.tmp"
    tmp.write_text(json.dumps(STATUS))
    tmp.replace(STATE_DIR / "research.json")


def main() -> int:
    STATUS["started"] = dt.datetime.now(NY).isoformat(timespec="seconds")
    write()
    try:
        return run()
    except BaseException as exc:          # any crash, including while loading
        STATUS.update(phase="failed", error=f"{type(exc).__name__}: {exc}")
        write()
        traceback.print_exc()
        return 1


def run() -> int:
    spec = importlib.util.spec_from_file_location("xsec", SCRIPT)
    xsec = importlib.util.module_from_spec(spec)
    sys.modules["xsec"] = xsec
    spec.loader.exec_module(xsec)

    STATUS["total"] = len([s for s in xsec.STARTING_STOCKS if s != "SPY"])

    # Hook 1: each stock (download -> train -> result).
    original_download = xsec.download_and_clean
    original_train = xsec.train_symbol

    def download_and_clean(symbol, *args, **kwargs):
        STATUS.update(phase="downloading", symbol=symbol, index=STATUS["index"] + 1,
                      epoch=None, seconds_left=None)
        write()
        try:
            return original_download(symbol, *args, **kwargs)
        except Exception as exc:
            STATUS["failures"].append(f"{symbol}: {exc}")
            write()
            raise

    def train_symbol(symbol, prices, *args, **kwargs):
        STATUS.update(phase="training", symbol=symbol)
        write()
        row = original_train(symbol, prices, *args, **kwargs)
        STATUS["results"].append({"symbol": symbol,
                                  "test_excess": round(float(row.get("test_excess", 0.0)), 4)})
        write()
        return row

    xsec.download_and_clean = download_and_clean
    xsec.train_symbol = train_symbol

    # Hook 2: each epoch. train() asks for the exploration rate once per epoch.
    agent_cls = xsec.dyna_q.DynaQ
    original_rate = agent_cls.exploration_rate
    clock = {"t0": None}

    def exploration_rate(self, epoch):
        if epoch == 0 or clock["t0"] is None:
            clock["t0"] = time.time()
        if epoch % 10 == 0 or epoch == self.epochs - 1:
            per_epoch = (time.time() - clock["t0"]) / epoch if epoch else None
            STATUS.update(epoch=epoch, epochs=self.epochs,
                          seconds_left=round(per_epoch * (self.epochs - epoch)) if per_epoch else None)
            write()
        return original_rate(self, epoch)

    agent_cls.exploration_rate = exploration_rate

    xsec.main()
    excess = [r["test_excess"] for r in STATUS["results"]]
    STATUS.update(phase="done", epoch=None, seconds_left=None, symbol=None,
                  summary={"beat_buy_and_hold": sum(e > 0 for e in excess), "stocks": len(excess),
                           "avg_test_excess": round(sum(excess) / len(excess), 4) if excess else None})
    write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
