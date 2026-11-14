"""sizing.py  -  how many contracts a real-money trade gets.

Rule (all of it has to agree; the smallest answer wins):
  1. Kelly from the bot's own real paper trades: the fraction of the account to put into option premium that would
     have grown it fastest, measured on resampled trade histories and taken at the cautious end (25th percentile).
  2. Use a QUARTER of that (full Kelly is far too jumpy in practice).
  3. Wild-day shrink: when today's volatility runs above normal, size shrinks (down to half). Calm days never size up.
  4. Hard caps: premium per trade <= 2% of the account (OPT_SIZE_MAX_PCT); never more contracts than the paper bot took.
  5. The scorecard's ladder wins over everything: "half size" halves it, "paper only" means 0.
  6. While live and allowed to trade, never below 1 contract. Under 30 real trades: exactly 1 contract.

Nightly:  .venv/bin/python live/sizing.py      -> paper-state/risk/sizing.json (the app shows it)
Bot:      import sizing; sizing.contracts(bot, price, equity, paper_qty, vol_ratio)
"""
from __future__ import annotations

import csv, datetime as dt, json, math, os, sys
from pathlib import Path

import numpy as np

PS = Path(os.environ.get("PAPER_STATE", Path.home() / "projects" / "paper-state"))
OUT = PS / "risk" / "sizing.json"
NY = __import__("zoneinfo").ZoneInfo("America/New_York")
FRACTION = float(os.environ.get("OPT_KELLY_FRACTION", "0.25"))
MAX_PCT = float(os.environ.get("OPT_SIZE_MAX_PCT", "0.02"))
MIN_TRADES = int(os.environ.get("OPT_SIZE_MIN_TRADES", "30"))
SHADOW_EQUITY = float(os.environ.get("OPT_SIZING_EQUITY", "10000"))      # the real account Greg plans to fund
BOTS = ("options-qqq", "options-spy")


def returns(bot: str) -> list[float]:
    """Per-trade return on premium (-1 = lost it all) of the bot's real paper trades since its champion started.
    Learning trades live in a separate file and are never used here."""
    d = PS / bot
    try:
        since = str(json.loads((d / "search" / "champion.json").read_text()).get("promoted") or "")
    except Exception:
        since = ""
    f = d / "options_trades.csv"
    if not f.exists():
        return []
    out = []
    for r in csv.DictReader(f.open()):
        if r.get("pl_pct") not in (None, "") and r.get("opened", "") >= since:
            out.append(max(-1.0, float(r["pl_pct"])))
    return out


def kelly(r: np.ndarray) -> float:
    """Fraction f of the account in premium that maximises average log growth, E[log(1 + f*r)]."""
    if len(r) == 0 or r.mean() <= 0:
        return 0.0
    grid = np.linspace(0, 0.99, 991)
    growth = np.log1p(np.outer(grid, r)).mean(axis=1)
    return float(grid[int(growth.argmax())])


def cautious_kelly(r: list[float], seed: int = 7) -> dict:
    a = np.asarray(r, float)
    if len(a) < 2:
        return {"trades": len(a), "full": 0.0, "cautious": 0.0, "used": 0.0}
    rng = np.random.default_rng(seed)
    boots = [kelly(rng.choice(a, len(a))) for _ in range(300)]
    full, cautious = kelly(a), float(np.percentile(boots, 25))
    return {"trades": len(a), "avg_return": round(float(a.mean()), 4), "win_rate": round(float((a > 0).mean()), 3),
            "full": round(full, 4), "cautious": round(cautious, 4), "used": round(FRACTION * cautious, 4)}


def ladder(bot: str) -> tuple[str, str]:
    try:
        for b in json.loads((PS / "scorecard.json").read_text()).get("bots", []):
            if b.get("bot") == bot:
                return b.get("ladder", "normal"), b.get("ladder_why", "")
    except Exception:
        pass
    return "normal", "no scorecard yet"


def contracts(bot: str, price: float, equity: float, paper_qty: int, vol_ratio: float | None = None,
              plan: dict | None = None) -> tuple[int, str]:
    """Contracts for one real-money trade, and a one-line reason."""
    plan = plan or (json.loads(OUT.read_text()).get(bot) if OUT.exists() else None) or {}
    step, why_step = plan.get("ladder") or ladder(bot)[0], plan.get("ladder_why", "")
    if step == "paper only":
        return 0, f"paper only ({why_step})"
    cost = price * 100
    if cost <= 0 or paper_qty < 1:
        return 0, "no price"
    if plan.get("trades", 0) < MIN_TRADES or plan.get("used", 0) <= 0:
        return 1, f"1 contract until {MIN_TRADES}+ real trades show an edge ({plan.get('trades', 0)} so far)"
    budget = equity * min(plan["used"], MAX_PCT)
    shrink = 1.0
    if vol_ratio and vol_ratio > 1:
        shrink = max(0.5, 1 / vol_ratio)
    budget *= shrink * (0.5 if step == "half size" else 1.0)
    q = max(1, min(int(budget // cost), paper_qty))
    parts = [f"¼ Kelly {plan['used']:.1%} of ${equity:,.0f}" + (" (capped at 2%)" if plan["used"] > MAX_PCT else "")]
    if shrink < 1:
        parts.append(f"wild day x{shrink:.2f}")
    if step == "half size":
        parts.append("half size")
    return q, " · ".join(parts) + f" -> {q} contract(s)"


def main() -> int:
    out = {"updated": dt.datetime.now(NY).isoformat(timespec="seconds"), "fraction": FRACTION, "max_pct": MAX_PCT,
           "shadow_equity": SHADOW_EQUITY}
    for bot in BOTS:
        if not (PS / bot).exists():
            continue
        k = cautious_kelly(returns(bot)); step, why = ladder(bot)
        k.update(ladder=step, ladder_why=why)
        q, reason = contracts(bot, 2.0, SHADOW_EQUITY, 99, plan=k)
        k["example"] = f"a $2.00 option with a ${SHADOW_EQUITY:,.0f} account: {reason}"
        out[bot] = k
        print(f"{bot}: {k['trades']} real trades · full Kelly {k['full']:.1%} · cautious {k['cautious']:.1%} · "
              f"using {k['used']:.1%} · ladder {step}\n   e.g. {k['example']}")
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
