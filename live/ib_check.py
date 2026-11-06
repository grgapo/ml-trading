"""Read-only IBKR connection check: account, positions, and an option-contract lookup (no orders)."""
import datetime as dt, os, sys, time
from zoneinfo import ZoneInfo
from ib_async import IB, Option, Stock
NY = ZoneInfo("America/New_York")
ib = IB()
for i in range(18):                                     # IB Gateway takes ~30-90 s to log in after starting
    try:
        ib.connect("127.0.0.1", 4002, clientId=91, readonly=True, timeout=10); break
    except Exception as exc:
        print(f"  waiting for IB Gateway to log in ({i + 1}/18): {type(exc).__name__}", flush=True); time.sleep(10)
else:
    print("FAILED: could not connect to IB Gateway on 127.0.0.1:4002"); sys.exit(1)
acct = ib.managedAccounts()
print("  connected · account(s):", ", ".join(acct), "(paper accounts start with DU)")
vals = {v.tag: v.value for v in ib.accountSummary() if v.currency in ("USD", "")}
for tag in ("NetLiquidation", "AvailableFunds", "BuyingPower"):
    if tag in vals: print(f"  {tag}: ${float(vals[tag]):,.2f}")
print("  positions:", len(ib.positions()))
q = Stock("QQQ", "SMART", "USD"); ib.qualifyContracts(q)
chains = ib.reqSecDefOptParams(q.symbol, "", q.secType, q.conId)
smart = next((c for c in chains if c.exchange == "SMART"), chains[0] if chains else None)
if smart:
    exp = sorted(smart.expirations)[0]
    strikes = sorted(smart.strikes); mid = strikes[len(strikes) // 2]
    opt = Option("QQQ", exp, mid, "C", "SMART", tradingClass=smart.tradingClass)
    ok = ib.qualifyContracts(opt)
    print(f"  option lookup: nearest QQQ expiry {exp}, {len(strikes)} strikes · contract check {'OK' if ok else 'FAILED'}")
ib.disconnect(); print("OK")
