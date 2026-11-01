"""Read the Freqtrade crypto paper bot (its local REST API) and write paper-state/crypto/bot.json for
the lab app: wallet value vs the $10,000 start, open positions, P&L by day, recent closed trades.
Uses the API login from Freqtrade's own config (nothing secret leaves lab03). Read-only.

  .venv/bin/python live/crypto_snapshot.py          # every 5 minutes from a systemd timer
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import requests

FT_CONFIG = Path(os.environ.get("FT_CONFIG", Path.home() / "projects" / "freqtrade" / "user_data" / "config.json"))
FT_URL = os.environ.get("FT_URL", "http://127.0.0.1:8097")
OUT = Path(os.environ.get("STATE_DIR", Path.home() / "projects" / "paper-state" / "crypto")) / "bot.json"
START = 10_000.0


def main() -> int:
    cfg = json.loads(FT_CONFIG.read_text())
    api = cfg.get("api_server", {})
    auth = (api.get("username", ""), api.get("password", ""))
    s = requests.Session(); s.auth = auth

    def get(path, **params):
        r = s.get(f"{FT_URL}/api/v1/{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    try:
        conf = get("show_config")
        bal = get("balance")
        status = get("status")
        profit = get("profit")
        daily = get("daily", timescale=180)
        trades = get("trades", limit=50)
    except Exception as exc:
        data = {"updated": now, "error": f"{type(exc).__name__}: {exc}"}
        OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(data))
        print(data["error"]); return 0
    open_pos = [{"pair": t["pair"], "since": t.get("open_date"), "entry": t.get("open_rate"), "now": t.get("current_rate"),
                 "stake": t.get("stake_amount"), "pl": round(t.get("profit_abs") or 0, 2), "plpc": round(t.get("profit_ratio") or 0, 4),
                 "tag": t.get("enter_tag")} for t in status]
    closed = [t for t in trades.get("trades", []) if not t.get("is_open")]
    log = [{"pair": t["pair"], "opened": t.get("open_date"), "closed": t.get("close_date"), "pl": round(t.get("profit_abs") or 0, 2),
            "plpc": round(t.get("profit_ratio") or 0, 4), "exit": t.get("exit_reason"), "stake": t.get("stake_amount")} for t in closed][-25:][::-1]
    days = sorted(daily.get("data", []), key=lambda x: x["date"])
    first = next((i for i, x in enumerate(days) if x.get("trade_count") or x.get("abs_profit")), len(days))
    run, curve = 0.0, []
    for x in days[first:]:
        run += x.get("abs_profit", 0) or 0
        curve.append({"date": x["date"], "day": round(x.get("abs_profit", 0) or 0, 2), "total": round(run, 2)})
    total = float(bal.get("total", START) or START)
    data = {
        "updated": now, "strategy": conf.get("strategy"), "state": conf.get("state"), "dry_run": conf.get("dry_run"),
        "exchange": conf.get("exchange"), "wallet": round(total, 2), "start": START,
        "open": open_pos, "closed_trades": profit.get("closed_trade_count", 0),
        "winning": profit.get("winning_trades", 0), "losing": profit.get("losing_trades", 0),
        "realized": round(profit.get("profit_closed_coin", 0) or 0, 2), "unrealized": round(sum(p["pl"] for p in open_pos), 2),
        "pnl": curve, "trades": log,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp"); tmp.write_text(json.dumps(data)); tmp.replace(OUT)
    print(f"crypto bot ({data['strategy']}, {data['state']}): wallet ${total:,.2f}, {len(open_pos)} open, {data['closed_trades']} closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
