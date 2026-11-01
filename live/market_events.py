"""Scheduled market-moving events, by trading day (New York dates).

Sources: Federal Reserve FOMC calendar (decision = 2nd day of each meeting, 2:00 pm ET) and the
BLS release schedules (CPI and the Employment Situation / jobs report, 8:30 am ET), checked against
bls.gov/schedule/2025 and /2026 - including the 2025 shutdown changes (no October 2025 jobs report or
October CPI; September jobs on Nov 20, November jobs on Dec 16, CPI on Oct 24 and Dec 18) and the
early-2026 delay (January jobs on Feb 11, January CPI on Feb 13).
Add next year's dates when the Fed and BLS publish them.
"""
from __future__ import annotations

import datetime as dt

_D = lambda s: {dt.date.fromisoformat(x) for x in s.split()}  # noqa: E731

FOMC = _D("""
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-10-28 2026-12-09
""")
CPI = _D("""
2024-11-13 2024-12-11
2025-01-15 2025-02-12 2025-03-12 2025-04-10 2025-05-13 2025-06-11 2025-07-15 2025-08-12 2025-09-11 2025-10-24 2025-12-18
2026-01-13 2026-02-13 2026-03-11 2026-04-10 2026-05-12 2026-06-10 2026-07-14 2026-08-12 2026-09-11 2026-10-14 2026-11-10 2026-12-10
""")
JOBS = _D("""
2024-11-01 2024-12-06
2025-01-10 2025-02-07 2025-03-07 2025-04-04 2025-05-02 2025-06-06 2025-07-03 2025-08-01 2025-09-05 2025-11-20 2025-12-16
2026-01-09 2026-02-11 2026-03-06 2026-04-03 2026-05-08 2026-06-05 2026-07-02 2026-08-07 2026-09-04 2026-10-02 2026-11-06 2026-12-04
""")
COVERED_THROUGH = dt.date(2026, 12, 31)


def flags(day: dt.date) -> dict:
    f = {"fomc_day": float(day in FOMC), "cpi_day": float(day in CPI), "jobs_day": float(day in JOBS)}
    f["event_day"] = float(any(f.values()))
    return f


def name(day: dt.date) -> str:
    n = [x for x, s in (("Fed decision", FOMC), ("CPI", CPI), ("jobs report", JOBS)) if day in s]
    return " + ".join(n)
