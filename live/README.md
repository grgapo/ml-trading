# `live/`: paper trading, options research and the nightly strategy search

Everything here runs on Alpaca **paper** accounts only (the clients are hard-wired to `paper=True`).
Your Dyna-Q code is used **unchanged**: these files import it and wrap it.

## How the pieces fit

```
                 ┌──────────── daily stock bot ────────────┐
your DynaQ  ───► │ paper_bot.py: pre-train once, then learn │ ──► Alpaca paper (SPY, QQQ)
(unchanged)      │ one step per day after the close         │
                 └──────────────────────────────────────────┘
features.py ─┬─► options_pretrain.py: simulate history, train a model ─┐
strategy.py ─┤                                                         ├─► options_model.joblib ──► options_bot.py ──► Alpaca paper (SPY/QQQ options)
             └─► options_search.py: try variations nightly, promote ───┘
run_research.py: runs your dyna-q-cross-section.py unchanged with live progress
account_snapshot.py: reads the account every 5 min for the dashboard
```

## Files

| File | What it does |
|---|---|
| `paper_bot.py` | Daily Dyna-Q trader. First run per symbol pre-trains your `DynaQ` on all history; after that, each day after the close it learns from yesterday's decision (reward = close-to-close return while long, minus 5 bps when the position changes), runs planning, picks today's action and sends a market order that fills at the next open. State is JSON + `.npy` (no pickle). |
| `run_research.py` | Runs `dyna-q-cross-section.py` exactly as is, hooking two spots (each stock starting/finishing, each epoch via `exploration_rate`) to report progress. |
| `features.py` | Intraday indicators shared by research and live trading (same code in both, so no train/live mismatch): VWAP, volume ratio, EMA 9/21/50, RSI 14, MACD + slope, ATR + regime, FVG balance, range position, opening gap, time of day; four setups (VWAP cross, 50-EMA cross, 15-min opening-range breakout, FVG retest); setup confluence (how many different setups agree within 15 bars); daily GARCH(1,1) volatility forecast, refit monthly on past data only. Every value uses only that bar and earlier. |
| `options_pretrain.py` | Builds the practice set: at every bar where a setup fires, simulates all four trades (call/put × 1DTE/0DTE) under the live rules (Black-Scholes pricing with IV = 1.15 × realized vol, 4% round-trip spread, −35% stop, +50% target, 60-min max hold, flat by 15:45). Trains gradient-boosted trees (meta-labeling: rules propose, model decides), then grades with monthly walk-forward plus a 5-day embargo, against "take every signal" and random baselines, with per-setup scorecards. |
| `strategy.py` | Defines a strategy variation: setups used, 1- or 5-min bars, stop/target/hold, 1DTE/0DTE/both, time window, model vs take-every-signal, minimum edge, volatility filter (calm/volatile days), conformal confidence (70%/80%), confluence (1/2/3+ setups agreeing). |
| `options_search.py` | Nightly random search over those variations, graded by quarterly walk-forward on the older data. The last 6 months are a **locked holdout** the search never sees; once a night the best new candidate gets one look and is promoted only if it's profitable there **and** beats the current champion. Also monitors the live champion (demotes it if 30+ live trades are clearly worse than its holdout promised, or if it loses $15k) and re-grades it nightly as the holdout rolls forward. |
| `options_bot.py` | Intraday options paper trader: watches SPY/QQQ 1-min bars, acts only when a setup fires, buys calls/puts only (max loss = premium), $1k per trade, $5k/day, max 5 trades, stops after 3 winners, exits on stop/target/time/end of day, retrains overnight with its real trades weighted 5×. Runs whatever the search last promoted. |
| `account_snapshot.py` | Read-only: equity, today's change, positions, every 5 minutes. |

## Honest results so far

- **Your cross-section, reproduced on a Raspberry Pi:** 4/19 stocks beat buy-and-hold on the test period (you had 5/19), average about −18 points; training growth far above test growth on most stocks. That gap is why the walk-forward PR exists.
- **Options pre-training (24 months of SPY/QQQ, 18 test months):** none of the four setups made money as simple rules. The model's picks lost less (−$66k vs −$88k take-every-signal vs −$119k random) and the monthly losses shrank over time, but there is **no positive edge yet**. That's why the nightly search exists; it promotes nothing unless it survives the locked holdout.
- **Sanity checks:** on synthetic data with a planted pattern, the pipeline finds it and the search promotes it; on pure noise it finds no edge and promotes nothing (no look-ahead).

## Known limitations / ideas welcome

- The daily bot learns from close-to-close returns but fills at the next open (fixed in the walk-forward evaluation; not yet in the live learner).
- Option prices are simulated (Black-Scholes, no real historical option quotes on the free data plan).
- Free IEX bars: volume is IEX-only, so volume features are relative, not absolute.
- One Alpaca paper account is shared by both bots.

## Running

```
pip install alpaca-py==0.44.0 scikit-learn==1.9.1 arch==8.0.0 yfinance scipy
# keys in the environment: ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY (paper)
python live/options_pretrain.py --synthetic --months 12     # offline test, no keys needed
python live/options_search.py --synthetic --months 16 --trials 3
python live/options_bot.py --simulate                        # fake market, fast clock
python live/paper_bot.py --dry-run --date 2026-10-28         # no orders
```
