# Design - Regression Pipeline

**Date:** 2026-05-14

## Goal

Train and evaluate four regression models against both `total_ghg` and `ghg_per_capita` in the cleaned master dataset. Output a report with metrics, predicted vs actual plots, residuals, coefficient/importance views (where applicable), and hyperparameter tuning curves (where applicable).

## Regression Targets

Two regression targets, trained separately:

- `total_ghg`
- `ghg_per_capita`

## Features

Numeric: `gdp`, `gdp_per_capita`, `land_area`, `population`.

Categorical: `continent` (one hot encoded, `drop_first=True` means we have 5 binary columns).

Per target exclusion: when predicting `total_ghg`, drop `ghg_per_capita` from features (and vice versa). Avoids the `total_ghg = ghg_per_capita x population` leakage.

Excluded from features in all cases: `iso_code`, `country`, `year` (not valid predictors, including year will leak some data through).

## Models

| Model | Tuned? | Reason |
| --- | --- | --- |
| Linear Regression | No | Baseline, has no hyperparameters. |
| Elastic Net | Yes | Regularized linear, has hyperparameters. |
| K-Nearest Neighbors | Yes | Captures local non linearity (in theory), has hyperparameters. |
| Random Forest | Yes | Added explicitly to handle non linearity that the linear and KNN models can't capture well, has hyperparameters. |

## Feature Scaling

Standardize features (sklearn `StandardScaler`) for Linear, Elastic Net, KNN. Random Forest gets raw features, as scale doesn't matter.

Scaler is fit on the training data only; the same scaler scales validation and test sets.

## Hyperparameter Grids

Grid search with 5 fold cross validation (`sklearn.model_selection.GridSearchCV` + `KFold(n_splits=5, shuffle=True)`), scored by R^2. `shuffle=True` is required to avoid coutries being broken up into alphabetically sorted blocks. Tuning curves built with `sklearn.model_selection.validation_curve` after the grid search finishes.

**Elastic Net** (`sklearn.linear_model.ElasticNet`):

| Hyperparameter | Values |
| --- | --- |
| `alpha` | 0.001, 0.01, 0.1, 1.0 |
| `l1_ratio` | 0.1, 0.5, 0.9 |

**KNN** (`sklearn.neighbors.KNeighborsRegressor`):

| Hyperparameter | Values |
| --- | --- |
| `n_neighbors` | 3, 5, 7, 10, 15 |
| `p` | 1 (Manhattan), 2 (Euclidean) |

Distance metric stays at the default Minkowski; only `p` changes, selecting Manhattan vs Euclidea; keeps the tuning to two hyperparameters.

**Random Forest** (`sklearn.ensemble.RandomForestRegressor`):

| Hyperparameter | Values |
| --- | --- |
| `n_estimators` | 100, 300 |
| `max_depth` | `None`, 10, 20 |
| `min_samples_split` | 2, 5 |
| `max_features` | `sqrt`, 1.0 |

## Splits

All four models use a **80/20 train/test split**. For the tuned models (Elastic Net, KNN, Random Forest), hyperparameter selection happens using 5 fold cross validation **inside the 80% training set** using `sklearn.model_selection.GridSearchCV` with `KFold(n_splits=5, shuffle=True)`. Per hyperparameter validation curves are then computed with `sklearn.model_selection.validation_curve` (each curve scans one hyperparameter while keeping the others at the best values). The 20% test set is held out throughout and only used for final reporting.

Completely random splitting using `random_state=42` everywhere for reproducibility of results.

**Row level split** (each (country, year) row is independent) instead of country level. The same country can appear i both the train and test sets under different years. This is fine here because both regression targets change year by year; predicting `total_ghg` for France in 2017 is different from predicting `total_ghg` for France in 1985, and the model is forced to learn from features that change over time rather than memorize country identity. For the classification pipeline where the target is constant per country regardless of features changing by time, a country level split is used.

## Metrics

Reported for every (model, target) combination on the test set:

- **RMSE** - root mean squared error
- **MAE** - mean absolute error
- **R^2**

## Report Contents

`reports/regression/report.md` plus `reports/regression/images/`. Sections:

1. **Metrics tables:** one table per target, columns are RMSE/MAE/R^2, rows are the four models.
2. **Best hyperparameters:** table for each tuned model showing the values that won the grid search for each target.
3. **Predicted vs actual scatters:** one plot per (model, target) so 8 plots total.
4. **Residuals:** one plot per (model, target) so 8 plots total.
5. **Coefficient bars:** for Linear and Elastic Net, per target so 4 plots total.
6. **Feature importance:** for Random Forest, per target so 2 plots total.
7. **Tuning curves:** for each tuned model and target, validation metric (R^2) plotted against each hyperparameter, which gives one composite figure per (model, target).

## Output

```
reports/regression/
+-- report.md
\-- images/          # ~30 PNGs for the sections above
```

Gitignored, same convention as `reports/eda/`.

## Architecture

Single script: `src/regression.py`. Flat functions for splits, scaling, fitting each model, evaluation, and plotting. `main()` calls functions to run both targets pipeline and writes the markdown report.

Random seed (`42`) applied to all splits and any model with randomness, which produces reproducible results for future verification.
