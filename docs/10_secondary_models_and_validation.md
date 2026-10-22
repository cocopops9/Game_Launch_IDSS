[Previous: The decision layer](09_decision_layer.md) · [Index](README.md) · **10 Secondary models and validation** · [Next: Limitations](11_limitations.md)

# 10. Secondary models and every reported metric

> **Interactive:** [Lab step 10](https://cocopops9.github.io/Game_Launch_IDSS/#validation): metric cards and held-out sample games.

`models.train_models` (cached with `st.cache_resource`) trains two more `XGBRegressor` models on the 134-feature set of
chapter 4. They feed the *Market Data* tab; they do **not** influence the percentile or the recommendations.

## 10.1 Validation design

| Element | Value |
|---|---|
| Split | temporal: release date before 2018-01-01 for training (16,702 games), from 2018-01-01 for testing (10,373 games) |
| Cross-validation | `cross_val_score(cv=3, scoring='r2')` on the training set only (three contiguous folds, no shuffling), owners model only |
| Missing values | filled with 0 |

The temporal split is the more realistic design for a pre-launch tool: the model only sees the past and is judged on the future.
It is also harsh, because 88.9% of the 2018 to 2019 test games are still in the lowest owners range (they had little time to sell).

## 10.2 Owners regression

```python
XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, min_child_weight=3,
             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, random_state=42)
```

Target: `log1p(owners)`; predictions are transformed back with `expm1`.

| Metric (snapshot run) | Value | Note |
|---|---:|---|
| 3-fold CV R² on the training years | 0.297 ± 0.127 (std) | large spread between periods |
| Test R² (log scale) | **0.362** | about a third of the variance of log owners |
| Test MAE | 23,473 owners | dominated by the many small games |
| Test RMSE | 269,154 owners | dominated by a few large games |
| Test Spearman | **0.429** | order of 2018 to 2019 games predicted from earlier ones |
| "Typical error in percentiles" | 8.83 points | the **25th percentile** of absolute percentile errors (see below) |
| Top 10% / top 5% "classification accuracy" | 91.2% / 95.1% | share of games on the same side of the 90th / 95th percentile threshold |
| Top 20% "classification accuracy" | 20.0% | artifact: the 80th percentile of actual owners equals the lowest range, so every game counts as "actual top" |

## 10.3 Review-ratio regression

```python
XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, min_child_weight=5,
             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=1.5, random_state=42)
```

Target: `positive_ratings / (positive_ratings + negative_ratings)`.

| Metric | Value |
|---|---:|
| Test R² | **0.110** |
| Test MAE | 0.197 (on a 0 to 1 scale) |

Pre-launch features explain about 11% of the variance of the review ratio. This is expected: reviews measure quality and
execution, which no field of `steam.csv` describes before launch. The app says so in the *Market Data* tab.

## 10.4 The ranker, for comparison

| Metric | Value |
|---|---:|
| Spearman, random 80/20 split (shown in the app) | **0.725** |
| 95% bootstrap interval | 0.712 to 0.738 |
| Same recipe on the temporal split | 0.432 |
| Absolute percentile error, 25th / 50th / 75th percentile | 5.9 / 13.5 / 24.3 points |

The last row compares, for each test game, the percentile of its predicted score with the percentile of its actual owners.
Half of the test games are placed within 13.5 points of their actual position, and a quarter are more than 24 points away.

## 10.5 Reading the metrics shown in the app correctly

| Display | What the code computes | Correct reading |
|---|---|---|
| "Ranking Accuracy 72.5%" | `spearman * 100` | Spearman's ρ = 0.725. It is a correlation, not a percentage of correctly ordered games |
| "Typical Precision ±X pts, 75% of predictions are within this distance or better" | `np.percentile(errors, 25)` | only **25%** of games are within that distance; 75% are farther. The median error is the better summary |
| "Spearman Correlation" in the training report | owners model, temporal split | different model and split from the ranker's 0.725 |
| Training report: "Train/Test Split: 80/20 with random_state=42" | fixed text | the owners model actually uses the temporal split (61.7% / 38.3%) |

## 10.6 The training report

Each call of `train_models` writes `training_report_YYYYMMDD_HHMMSS.md` (dataset statistics, feature groups, CV folds,
test metrics, residual analysis, feature importance, a comparison with a mean baseline). Some sections contain fixed text
that does not depend on the run, for example the list of "alternatives considered"; the numerical tables are computed.

---
[Previous: The decision layer](09_decision_layer.md) · [Index](README.md) · **10 Secondary models and validation** · [Next: Limitations](11_limitations.md)
