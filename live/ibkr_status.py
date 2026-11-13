"""IBKR status for the lab app (lab03, every 2 minutes): what the IBKR paper account looks like right now.

Writes ~/projects/paper-state/ibkr/status.json (served to the app like every other bot's status):
  gateway   logged in? which account? (read-only connection, its own client id 97; never places orders)
  account   net value, cash, buying power, today's realised / open P&L
  positions what's open, with live value and P&L
  fills     today's executions (time, contract, side, qty, price, commission)
  mirror    the QQQ mirror vs the Alpaca QQQ bot on the SAME signals: daily totals side by side, and the average
            entry-price gap per matched trade (the go-live "fills match the simulator" check)
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
PS = Path(os.environ.get("PAPER_STATE", Path.home() / "projects" / "paper-state"))
OUT = PS / "ibkr" / "status.json"
HOST, PORT, CID = "127.0.0.1", int(os.environ.get("IBKR_PORT", "4002")), 97


def num(x):
    try:
        v = float(x)
        return None if v != v else round(v, 2)
    except Exception:
        return None


def trades(path: Path) -> list[dict]:
    if not path.exists():
        return []
    cols = ["date", "entry_time", "exit_time", "symbol", "contract", "kind", "qty", "entry", "pct", "pnl", "reason", "signal", "flag"]
    rows = []
    for r in csv.reader(path.open()):
        if not r or r[0] == "date":
            continue
        rows.append(dict(zip(cols, r)))
    return rows


def mirror_compare(days: int = 10) -> dict:
    """Pair each IBKR mirror trade with the Alpaca QQQ trade from the same signal (same contract side, entry within 3 min)."""
    ib, al = trades(PS / "options-qqq-ibkr" / "options_trades.csv"), trades(PS / "options-qqq" / "options_trades.csv")
    since = str(dt.date.today() - dt.timedelta(days=days * 2))
    ib = [r for r in ib if r["date"] >= since]; al = [r for r in al if r["date"] >= since]
    pairs, used = [], set()
    for r in ib:
        try:
            t = dt.datetime.fromisoformat(r["entry_time"])
        except Exception:
            continue
        best = None
        for j, a in enumerate(al):
            if j in used or a["date"] != r["date"] or a["kind"] != r["kind"]:
                continue
            try:
                gap = abs((dt.datetime.fromisoformat(a["entry_time"]) - t).total_seconds())
            except Exception:
                continue
            if gap <= 180 and (best is None or gap < best[0]):
                best = (gap, j)
        if best:
            used.add(best[1]); a = al[best[1]]
            ie, ae = num(r["entry"]), num(a["entry"])
            q_al = max(1.0, float(a["qty"] or 1))
            pairs.append({"date": r["date"], "time": r["entry_time"][11:16], "contract": r["contract"], "ibkr_entry": ie, "alpaca_entry": ae,
                          "entry_gap_pct": round((ie / ae - 1) * 100, 2) if ie and ae else None,
                          "ibkr_pnl": num(r["pnl"]), "alpaca_pnl_per_contract": round((num(a["pnl"]) or 0) / q_al, 2)})
    by_day = {}
    for r in ib:
        by_day.setdefault(r["date"], {"date": r["date"], "ibkr": 0.0, "alpaca_per_contract": 0.0, "ibkr_trades": 0})
        by_day[r["date"]]["ibkr"] += num(r["pnl"]) or 0; by_day[r["date"]]["ibkr_trades"] += 1
    for p in pairs:
        by_day[p["date"]]["alpaca_per_contract"] += p["alpaca_pnl_per_contract"] or 0
    gaps = [abs(p["entry_gap_pct"]) for p in pairs if p["entry_gap_pct"] is not None]
    return {"pairs": pairs[-15:], "days": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()} for d in sorted(by_day.values(), key=lambda d: d["date"])][-days:],
            "matched": len(pairs), "ibkr_trades": len(ib), "avg_entry_gap_pct": round(sum(gaps) / len(gaps), 2) if gaps else None}


def main() -> int:
    out = {"updated": dt.datetime.now(NY).isoformat(timespec="seconds"), "gateway": {"connected": False}}
    try:
        from ib_async import IB
        ib = IB()
        ib.connect(HOST, PORT, clientId=CID, readonly=True, timeout=15)
        try:
            accts = ib.managedAccounts()
            out["gateway"] = {"connected": True, "account": accts[0] if accts else None, "paper": bool(accts and accts[0].startswith("D"))}
            want = {"NetLiquidation": "net_value", "TotalCashValue": "cash", "BuyingPower": "buying_power",
                    "RealizedPnL": "realized_today", "UnrealizedPnL": "open_pnl", "AvailableFunds": "available"}
            acct = {}
            for v in ib.accountSummary():
                if v.tag in want and v.currency in ("USD", "BASE", ""):
                    acct[want[v.tag]] = num(v.value)
            out["account"] = acct
            out["positions"] = [{"symbol": p.contract.symbol, "contract": p.contract.localSymbol or p.contract.symbol, "type": p.contract.secType,
                                 "qty": num(p.position), "avg_cost": num(p.averageCost), "price": num(p.marketPrice),
                                 "value": num(p.marketValue), "open_pnl": num(p.unrealizedPNL)} for p in ib.portfolio()]
            today = dt.datetime.now(NY).date()
            fills = []
            for f in ib.fills():
                t = f.time.astimezone(NY) if f.time else None
                if t and t.date() == today:
                    fills.append({"time": t.strftime("%H:%M:%S"), "contract": f.contract.localSymbol or f.contract.symbol,
                                  "side": f.execution.side, "qty": num(f.execution.shares), "price": num(f.execution.price),
                                  "commission": num(f.commissionReport.commission) if f.commissionReport else None})
            out["fills"] = sorted(fills, key=lambda x: x["time"])
        finally:
            ib.disconnect()
    except Exception as exc:
        out["gateway"] = {"connected": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    try:
        out["mirror"] = mirror_compare()
    except Exception as exc:
        out["mirror"] = {"error": f"{type(exc).__name__}: {exc}"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp"); tmp.write_text(json.dumps(out)); tmp.replace(OUT)
    g = out["gateway"]
    print(f"gateway {'logged in · ' + str(g.get('account')) if g.get('connected') else 'not reachable: ' + str(g.get('error'))}"
          f" · {len(out.get('positions', []))} positions · {len(out.get('fills', []))} fills today · mirror matched {out['mirror'].get('matched')} trades")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
