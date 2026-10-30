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
}
VOL_SPLIT = 1.0   # vol_ratio above this = volatility expected to rise ("volatile")
SIGNAL_SETS = [list(c) for n in range(1, len(SIGNALS) + 1) for c in itertools.combinations(SIGNALS, n)]

# Columns of the simulated returns: call_1dte, put_1dte, call_0dte, put_0dte.
_DIR_ACTIONS = {1: {"1dte": [0], "0dte": [2], "both": [0, 2]},
                -1: {"1dte": [1], "0dte": [3], "both": [1, 3]}}

SHORT = {"sig_vwap": "VWAP", "sig_ema50": "EMA50", "sig_orb": "ORB", "sig_fvg": "FVG"}


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
    return (f"{p['bar_minutes']}m · {sigs} · SL{p['stop_loss']:.0%} TP+{p['take_profit']:.0%} · "
            f"{p['max_hold']}min · {p['dte'].upper()} · {p['window']} · {pick}"
            + (f" · {vol} days" if vol not in (None, "off") else ""))


def dataset_key(p: dict) -> str:
    return f"v{FEATURE_VERSION}_b{p['bar_minutes']}_sl{p['stop_loss']}_tp{p['take_profit']}_h{p['max_hold']}"


def confluence_ok(direction: int, conf_bull: float, conf_bear: float, need) -> bool:
    need = need or 1
    return (conf_bull if direction > 0 else conf_bear) >= need


def vol_ok(vol_ratio: float, setting: str | None) -> bool:
    if setting in (None, "off"):
        return True
    if vol_ratio != vol_ratio:          # NaN: no forecast yet -> don't filter
        return True
    return vol_ratio <= VOL_SPLIT if setting == "calm" else vol_ratio > VOL_SPLIT
