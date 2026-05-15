# Design - Classification Pipeline

**Date:** 2026-05-15

## Goal

Train and evaluate four classification models with continent as the target, using the cleaned master dataset. Output a report with overall and per class metrics, confusion matrices, hyperparameter tuning curves, and model specific interpretability views (coefficients for linear, feature importance for tree based).

## Target

`continent` (6 classes): `Africa`, `Asia`, `Europe`, `North America`, `South America`, `Oceania`.

Class counts in the cleaned master after the row drop step lean toward Africa and Asia (~50 countries each) and away from South America (~12) and Oceania (~14). Roughly a 4.5 times imbalance between largest and smallest class, which is handled via per class metrics.

## Features

Numeric: `gdp`, `gdp_per_capita`, `land_area`, `population`, `total_ghg`, `ghg_per_capita`.

Both `total_ghg` and `ghg_per_capita` are included as features, as neither is the target, and they are semi independent.

Excluded: `iso_code`, `country`, `year`, `continent` (target).

No one hot encoding needed; the only categorical column in the master is `continent`, which is our target.

## Models

| Model | Tuned? | Rationale |
| --- | --- | --- |
| Logistic Regression | No | Linear baseline. Default settings, no tuning. |
| K-Nearest Neighbors | Yes | Captures local non linearity, has hyperparameters. |
| SVM (SVC) | Yes | Can capture non linearity through RBF kernel, has hyerparameters. |
| Random Forest | Yes | Ensemble of decision trees; captures non linearity, has hyperparameters. |

**Note on Gaussian Naive Bayes:** considered but nto used. GaussianNB assumes each feature follows a normal distribution within each class. From our EDA, our numeric features (especially `gdp`, `population`, `total_ghg`) are heavily skewed to the right, violating the normal assumption. Random Forest replaced it as an alternative that makes no shape assumptions for the features.

## Feature Scaling

Standardize features (sklearn `StandardScaler`) for all four models. KNN and SVM need it (distance based); Logistic Regression benefits from it as well; Random Forest is doesn't care about scale, but using the same scaled inputs keeps the preprocessing simple.

Scaler is fit on training data only; the same scaler scales validation and test sets.

## Hyperparameter Grids

Grid search with 5 fold cross validation (`sklearn.model_selection.GridSearchCV` + `StratifiedGroupKFold(n_splits=5, shuffle=True)` so CV folds also account for both country groupings and class stratification), scored by accuracy. Tuning curves built with `sklearn.model_selection.validation_curve` after the grid search finishes.

**KNN** (`sklearn.neighbors.KNeighborsClassifier`):

| Hyperparameter | Values |
| --- | --- |
| `n_neighbors` | 3, 5, 7, 10, 15 |
| `p` | 1 (Manhattan), 2 (Euclidean) |

Distance metric stays at default Minkowski; only `p` changes. Same logic as the regression KNN grid.

**SVM** (`sklearn.svm.SVC`):

| Hyperparameter | Values |
| --- | --- |
| `C` | 0.1, 1.0, 10.0 |
| `kernel` | `linear`, `rbf` |

`gamma` left at sklearn default (`'scale'`); other kernels (`poly`, `sigmoid`) not tested as computational cost was just too high for local machine and pipeline took too long to run.

**Random Forest** (`sklearn.ensemble.RandomForestClassifier`):

| Hyperparameter | Values |
| --- | --- |
| `n_estimators` | 100, 300 |
| `max_depth` | 5, 10, 20, `None` |
| `min_samples_split` | 2, 5, 10 |
| `max_features` | `sqrt`, 1.0 |

## Splits

All four models use a single **80/20 train/test split**. For the tuned models (KNN, SVM, Random Forest), hyperparameter choices happen via 5 fold cross validation **inside the 80% training set** using `sklearn.model_selection.GridSearchCV` with `StratifiedGroupKFold(n_splits=5, shuffle=True)`. The 20% test set is held out throughout and only used for final reporting.

**Country level, stratified group split** on `iso_code`. Every country goes entirely in one set; no country appears in more than one split. Within each continent, countries are partitioned across train and test sets so all classes are represented; this is done because of the large class imbalance that we could see in EDA. CV folds inside the training set are also stratified by group, so no country leaks across folds and class distribution is properly preserved.

The reason for country level here: `continent` is constant per country. A row level split would let the model see (France, 1985) -> Europe during training and then predict on (France, 2017) -> Europe during test, which makes the task simple country recognition rather than actually predicting a continent from the other features. The model would essentially learn "France shaped features -> Europe" through stable features like `land_area`. A countr level split forces the test rows to come from countries the model has never seen, so the metrics reflect actual generalization across countries rather than across years of the same countries.

**Why stratified:** the initial pipeline used a unstratified `GroupShuffleSplit`. After running, the split happened to assign 0 of the 14 Oceania countries to the test set, which gave per class metrics of 0 and artificially pulled the F1 score down. A stratified group split sized per class, drawing the test fraction from each continent independently, guarantees every class has at least one country in the test set.

With ~197 countries, 80/20 gives roughly 153 train countries, 38 test. `random_state=42` again, as mentioned in the regression pipeline spec.

## Metrics

Reported for each model on the test set:

- **Accuracy:** overall correct rate
- **Macro F1:** unweighted average of per-class F1 (treats every continent equally regardless of size)
- **Per class precision and recall:** one value per continent
- **Confusion matrix:** 6x6 matrix of true vs predicted classes

Accuracy alone would be misleading given the class imbalance; F1 + per class breakdown is the honest view.

## Report Contents

`reports/classification/report.md` plus `reports/classification/images/`. Sections per model:

1. **Summary metrics table:** one row per model, columns `accuracy`, `F1`. Includes both train and test values for overfitting diagnosis.
2. **Per class metrics tables:** one table per model, rows are continents, columns are precision, recall, F1, support.
3. **Best hyperparameters:** table per tuned model.
4. **Confusion matrices:** one heatmap per model, 6x6, annotated with counts.
5. **Coefficient heatmap:** for Logistic Regression, a heatmap of (continent x feature) coefficients showing which features drive each class.
6. **Feature importance:** for Random Forest, a bar chart.
7. **Tuning curves:** for each tuned model, validation accuracy plotted against each hyperparameter, giving one composite figure per model.

## Output

```
reports/classification/
+-- report.md
\-- images/          # PNGs across the sections above
```

Gitignored, same convention as `reports/regression/`.

## Architecture

Single script: `src/classification.py`. Flat functions for splits, scaling, fitting each model, evaluation, and plotting. `main()` sets up two pipeline runs, one with engineered features, one with raw features, and writes a markdown report that compares both.

`random_state=42` applied to all splits and any model with randomness (SVC, Random Forest) as before.

## Feature Engineering

In addition to the six raw numeric features, the pipeline also runs with engineered features added: for each of `gdp`, `gdp_per_capita`, `land_area`, `population`, `total_ghg`, `ghg_per_capita`, three transformations are appended:

- **Log** (signed): `sign(x) * log1p(|x|)`. Signed log handles the small number of negative GHG value.
- **Squared**: `x^2`.
- **Reciprocal**: `1 / (|x| + 1)`.

So the engineered configuration has 6 original + 18 transformed = 24 numeric features.

Both configurations are run on the same train/test split and the report compares their test metrics.

## Approaches tried to lift test accuracy above ~0.57

The classification task ended up capping out at roughly 0.57 test accuracy / 0.46 F1 on Random Forest, with linear and distance based models even lower. The pipeline includes the following iterations, each motivated by an attempt to improve these scores:

1. **Stratified group split** replaced an initial unstratified `GroupShuffleSplit` after observing that the random country draw could assign zero countries from a small class (Oceania) to the test set and send per class metrics to zero.
2. **Random Forest** replaced the originally planned Decision Tree. Single trees memorized the training data (train accuracy 1.0) without generalizing; the ensemble narrowed the train/test gap.
3. **Feature engineering (log / squared / reciprocal transformations)** was added to give linear and distance models access to non linear feature shapes. The engineered configuration is reported alongside the raw configuration in the final report. Engineering produced a decent increase for Logistic Regression (+11 accuracy points), SVM (+8), and KNN (+6); it slightly hurt Random Forest (-2), probably because `max_features="sqrt"` over 24 features picks too few original features per split. The best single model test accuracy ended at ~0.58 (SVM with engineered features).

The ceiling reflects a feature limitation rather than a model or methods problem: six features do not separate six continents at the country level even after non linear transformations of them. Distinguishing features would require external data that we did not use here (i.e.  energy mix, climate, industrial composition). Within the techniques covered in the course, no remaining technique was available to push above ~0.58.
