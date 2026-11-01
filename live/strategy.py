"""What a strategy variant is, shared by the nightly search and the live options bot."""
from __future__ import annotations

import itertools

from features import FEATURE_VERSION, SIGNALS

# Entry windows, minutes after 9:30 ET.
WINDOWS = {"all": (15, 330), "morning": (15, 120), "midday": (120, 240), "afternoon": (210, 330)}

SPACE = {
    "bar_minutes": [1, 5],
    "stop_loss": [-0.25, -0.35, -0.5],
    "take_profit": [0.3, 0.5, 1.0],
    "max_hold": [30, 60, 120],
    "dte": ["1dte", "0dte", "both"],
    "window": list(WINDOWS),
    "use_model": [True, False],
    "min_edge": [0.0, 0.03, 0.08],
    # daily volatility regime from the GARCH forecast vs recent realized volatility
    "vol_filter": ["off", "calm", "volatile"],
    # split-conformal check: only trade if the model is this sure the trade won't lose
    "confidence": ["off", 0.7, 0.8],
    # how many different setups must agree (within ~15 bars) before a trade is allowed
    "confluence": [1, 2, 3],
    # scheduled events (Fed decision, CPI, jobs report): trade any day, skip those days, or only those days
    "event_filter": ["off", "skip", "only"],
    # VIX vs its 1-year median: any, calmer than usual, or more nervous than usual
    "vix_filter": ["off", "low", "high"],
    # how a trade ends: quick option exits (stop / target / max hold above) or ATR exits on the
    # underlying (after 4 ATR its way, sell on giving back half; 10 ATR safety stop; out at 3:45)
    "exit_style": ["option", "atr"],
    # which contract: at the money, or about $2 in the money
    "moneyness": ["atm", "itm2"],
    # the calm-day predictor (trend_days.py, walk-forward): any day, calm days only, trend days only
    "day_filter": ["off", "calm", "trend"],
}
VOL_SPLIT = 1.0   # vol_ratio above this = volatility expected to rise ("volatile")
SIGNAL_SETS = [list(c) for n in range(1, len(SIGNALS) + 1) for c in itertools.combinations(SIGNALS, n)]

# Columns of the simulated returns: call_1dte, put_1dte, call_0dte, put_0dte.
_DIR_ACTIONS = {1: {"1dte": [0], "0dte": [2], "both": [0, 2]},
                -1: {"1dte": [1], "0dte": [3], "both": [1, 3]}}

SHORT = {"sig_vwap": "VWAP", "sig_ema50": "EMA50", "sig_orb": "ORB", "sig_fvg": "FVG", "sig_orbnb": "ORB+NB"}


def allowed_actions(direction: int, dte: str) -> list[int]:
    return _DIR_ACTIONS.get(int(direction), {}).get(dte, [])


def describe(p: dict) -> str:
    sigs = "+".join(SHORT[s] for s in p["signals"])
    pick = f"model≥{p['min_edge']:.0%}" if p["use_model"] else "every signal"
    conf = p.get("confidence", "off")
    if p["use_model"] and conf not in (None, "off"):
        pick += f" · {conf:.0%} sure"
    vol = p.get("vol_filter", "off")
    if (p.get("confluence") or 1) > 1:
        pick += f" · {p['confluence']}+ agree"
    ev, vx = p.get("event_filter", "off"), p.get("vix_filter", "off")
    exits = ("ATR exits" if p.get("exit_style") == "atr"
             else f"SL{p['stop_loss']:.0%} TP+{p['take_profit']:.0%} · {p['max_hold']}min")
    df_ = p.get("day_filter", "off")
    return (f"{p['bar_minutes']}m · {sigs} · {exits} · {p['dte'].upper()}"
            + (" · $2 ITM" if p.get("moneyness") == "itm2" else "")
            + f" · {p['window']} · {pick}"
            + ({"calm": " · calm days only", "trend": " · trend days only"}.get(df_, ""))
            + ({"calm": " · low vol forecast", "volatile": " · high vol forecast"}.get(vol, ""))
            + ({"skip": " · skip event days", "only": " · event days only"}.get(ev, ""))
            + ({"low": " · VIX below usual", "high": " · VIX above usual"}.get(vx, "")))


def dataset_key(p: dict) -> str:
    exits = "xatr" if p.get("exit_style") == "atr" else f"sl{p['stop_loss']}_tp{p['take_profit']}_h{p['max_hold']}"
    return f"v{FEATURE_VERSION}_b{p['bar_minutes']}_{exits}" + ("_itm2" if p.get("moneyness") == "itm2" else "")


def needs_live_support(p: dict) -> list[str]:
    """Pieces of a variation the live options bot can't execute yet (so it mustn't be auto-promoted)."""
    out = []
    if p.get("exit_style") == "atr":
        out.append("ATR exits")
    if p.get("moneyness") == "itm2":
        out.append("in-the-money strikes")
    if p.get("day_filter", "off") != "off":
        out.append("the calm-day filter")
    if "sig_orbnb" in (p.get("signals") or []):
        out.append("the ORB+noise band setup")
    return out


def confluence_ok(direction: int, conf_bull: float, conf_bear: float, need) -> bool:
    need = need or 1
    return (conf_bull if direction > 0 else conf_bear) >= need


def event_ok(event_day: float, setting: str | None) -> bool:
    if setting in (None, "off") or event_day != event_day:
        return True
    return (event_day == 0) if setting == "skip" else (event_day == 1)


def vix_ok(vix_rel: float, setting: str | None) -> bool:
    if setting in (None, "off") or vix_rel != vix_rel:     # NaN: no VIX data -> don't filter
        return True
    return vix_rel <= 0 if setting == "low" else vix_rel > 0


def vol_ok(vol_ratio: float, setting: str | None) -> bool:
    if setting in (None, "off"):
        return True
    if vol_ratio != vol_ratio:          # NaN: no forecast yet -> don't filter
        return True
    return vol_ratio <= VOL_SPLIT if setting == "calm" else vol_ratio > VOL_SPLIT
