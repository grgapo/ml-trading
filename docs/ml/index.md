# ML experiment log

Living record of learning algorithms tried in this repo: current status, latest metrics, and links to version history.

## Status board

| Method | Version | Status | Data | Target | Symbols trained | Key metrics | Page |
|--------|---------|--------|------|--------|-----------------|-------------|------|
| Parametric linear regression | v8 | current | clean-yfinance, 10y daily | next-day Adj Close return | SPY | train MSE 0.000133, R² 0.006; test MSE 0.000110, R² -0.008 | [parametric-linear-regression](regression/parametric-regression/parametric-linear-regression.md) |
| kNN regression | v8 | current | clean-yfinance, 10y daily | next-day Adj Close return | all 20 | v5-recipe macro test R² k=15 -0.0515, k=21 -0.0325, k=51 -0.0111; k=15 beats k=51 on 1/20 | [knn](regression/knn/knn.md) |
| Decision tree regression | v2 | current | clean-yfinance, 25y SPY | next-day Adj Close return | SPY | train MSE 0.000139, R² 0.047; test MSE 0.000109, R² -0.002; `max_depth=2` | [decision-tree](regression/decision-tree/decision-tree.md) |
| Kernel SVR | v6 | current | clean-yfinance, 10y daily (SPY 25y) | next-day Adj Close return | all 20 | v4-recipe macro test R² -0.0074, 5/20 positive; v5 -0.0137, 4/20; v5 beats v4 on 7/20 | [svm](regression/svm/svm.md) |
| Voting regressor | v2 | current | clean-yfinance, 25y SPY | next-day Adj Close return | SPY | train MSE 0.000122, R² 0.158; test MSE 0.000112, R² -0.028 | [voting-regressor](ensemble-learning/voting-regressor.md) |

## How to read this

One row per method, always the **current** version. Full history and vs-previous notes live on the method page.

Add or update rows only from a documented training run.

## Todo

Not logged runs. Same rules as the board: `data/clean-yfinance`, cutoff `2024-09-01`, chronological split, no shuffled rows. Features are the stationary columns in `src/ml/regression/ai_scripts/parametric_experiment_common.py` (`Ret_1`, `Overnight`, `Intraday`, `Range`, `Close_Location`, `Relative_Volume`, `Vol_20`, `5_Day_Return`, `20_Day_Return`, `Dist_From_SMA`). No raw price levels.

| Experiment | Target | Model | Score against |
|------------|--------|-------|----------------|
| Single-name direction | sign of the next-day `Adj Close` return. Start with SPY | `LogisticRegression` (scaled) and a shallow `RandomForestClassifier` (`max_depth` about 3 to 5, large `min_samples_leaf`) | always-up accuracy, and long/flat growth of $1 vs holding the stock |
| Cross-sectional median | 1 when that name's next-day return beats the same day's median across `STARTING_STOCKS` | same two classifiers, one row per symbol-day | equal-weight basket of those names. Long only the names predicted to beat the median |

`src/ml/reinforcement-learning/dyna-q-cross-section.py` is a different experiment. It fits a separate long/flat policy per name. It does not rank names against each other on the same day.
