"""Keep search leaderboards small without losing anything the searches need.

  * leaderboard.json keeps: the 2,000 best by search P&L (40+ trades), every entry that had a blind test or a robustness
    check, and the latest 20,000 variations (in tested order).  Nothing a promotion decision uses is dropped.
  * tried.txt remembers EVERY variation ever tested (one key per line, append-only), so dropped entries are never re-tested.
  * board_recent.json: the latest 1,500, slimmed to what the lab app draws (a few MB instead of hundreds).
"""
from __future__ import annotations

import json
from pathlib import Path

KEEP_BEST, KEEP_RECENT, APP_RECENT, MIN_TRADES_BEST = 2000, 20000, 1500, 40


def key_of(p: dict) -> str:
    return json.dumps(p, sort_keys=True)


def trim(board: list) -> list:
    if len(board) <= KEEP_RECENT + KEEP_BEST:
        return board
    best = sorted((i for i, e in enumerate(board) if (e.get("search") or {}).get("trades", 0) >= MIN_TRADES_BEST),
                  key=lambda i: (board[i].get("search") or {}).get("pnl", 0), reverse=True)[:KEEP_BEST]
    keep = set(best) | {i for i, e in enumerate(board) if "holdout" in e or "robust" in e} | set(range(len(board) - KEEP_RECENT, len(board)))
    return [e for i, e in enumerate(board) if i in keep]


def slim(e: dict) -> dict:
    s = e.get("search") or {}
    return {"desc": e.get("desc"), "tested": e.get("tested"),
            "search": {k: s.get(k) for k in ("pnl", "trades", "wins", "months", "simulated")},
            "holdout": {"pnl": (e.get("holdout") or {}).get("pnl")} if e.get("holdout") else None}


def write_all(search_dir: Path, board: list, written: set) -> set:
    """Trim in place, then write leaderboard.json, board_recent.json and append new keys to tried.txt."""
    board[:] = trim(board)
    new = [k for k in (key_of(e["params"]) for e in board if "params" in e) if k not in written]
    if new:
        with (search_dir / "tried.txt").open("a") as f:
            f.write("\n".join(new) + "\n")
        written |= set(new)
    for name, data in (("leaderboard.json", board), ("board_recent.json", [slim(e) for e in board[-APP_RECENT:]])):
        tmp = search_dir / (name + ".tmp"); tmp.write_text(json.dumps(data)); tmp.replace(search_dir / name)
    return written


def load_tried(search_dir: Path) -> set:
    f = search_dir / "tried.txt"
    return set(l for l in f.read_text().splitlines() if l) if f.exists() else set()


if __name__ == "__main__":            # one-off: shrink the given search folders now
    import sys
    for d in map(Path, sys.argv[1:]):
        lb = d / "leaderboard.json"
        board = json.loads(lb.read_text()); before = len(board)
        written = load_tried(d) | {key_of(e["params"]) for e in board if "params" in e}
        (d / "tried.txt").write_text("\n".join(sorted(written)) + "\n")
        write_all(d, board, written)
        print(f"{d}: {before:,} entries -> {len(board):,} kept · {len(written):,} remembered in tried.txt · "
              f"leaderboard {lb.stat().st_size / 1e6:.0f} MB · board_recent {(d / 'board_recent.json').stat().st_size / 1e6:.1f} MB")
