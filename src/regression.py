from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (  # all visualization imports
    aes,
    coord_flip,
    facet_wrap,
    geom_abline,
    geom_col,
    geom_hline,
    geom_line,
    geom_point,
    ggplot,
    labs,
    theme_minimal,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    train_test_split,
    validation_curve,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# set up paths and shared output config
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "processed" / "master_clean.csv"
REPORTS_DIR = REPO_ROOT / "reports" / "regression"
IMAGES_DIR = REPORTS_DIR / "images"
REPORT_PATH = REPORTS_DIR / "report.md"

SEED = 42  # seed used everywhere for reproducible but random results
TARGETS = ["total_ghg", "ghg_per_capita"]  # two targets, trained  separately

# hyperparameter grids for the three tuned models (Linear has nothing to tune)
EN_GRID = {"alpha": [0.001, 0.01, 0.1, 1.0], "l1_ratio": [0.1, 0.5, 0.9]}
KNN_GRID = {
    "n_neighbors": [3, 5, 7, 10, 15],
    "p": [1, 2],
}  # p=1 is Manhattan, p=2 is Euclidean
RF_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", 1.0],
}

# default image sizes (saved across all plots in the report)
DPI = 150
WIDTH = 8
HEIGHT = 5


# Saves a plot to the images directory and returns the path (used by the report writer)
def save_plot(plot, name, height=HEIGHT):
    path = IMAGES_DIR / f"{name}.png"
    plot.save(path, dpi=DPI, width=WIDTH, height=height, verbose=False)
    return path


# Builds the feature matrix X and target series y for the given target
# Drops the other target so no leak (total_ghg = ghg_per_capita x population)
# One-hots continent with drop_first=True so the dummies are linearly independent
def build_features(df, target):
    other_target = "ghg_per_capita" if target == "total_ghg" else "total_ghg"
    feature_df = df.drop(columns=["iso_code", "country", "year", target, other_target])
    feature_df = pd.get_dummies(
        feature_df, columns=["continent"], drop_first=True, dtype=int
    )
    return feature_df, df[target]


# 80/20 split (left as row level or country-year, fine because we are predicting a moving target, discussed in spec)
def split_80_20(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=SEED)


# Standardize features using scaler fit on X_train, then apply same scaler to X_test so no test data leaks into fit
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    cols = X_train.columns
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), columns=cols, index=X_train.index
    )
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=cols, index=X_test.index)
    return X_train_s, X_test_s


# Finds RMSE, MAE, and R^2 for pair of true vs predicted values
def compute_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# Fits baseline Linear Regression (no tuning) and returns the result as a dictionary
def fit_linear(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    return {  # return dict for future manipulation
        "model": model,
        "feature_names": list(X_train.columns),
        "y_test": y_test,
        "y_pred": y_pred_test,
        "train_metrics": compute_metrics(y_train, y_pred_train),
        "test_metrics": compute_metrics(y_test, y_pred_test),
    }


# Runs GridSearchCV with 5 fold CV scored by R^2, then builds per hyperparameter validation curves with the other hyperparameters fixed at the best values found
# Returns  final fit, picked params, curves df, and train/test metrics
def tune_with_cv(estimator, grid, X_train, X_test, y_train, y_test):
    cv = KFold(
        n_splits=5, shuffle=True, random_state=SEED
    )  # shuffle so folds not sorted by country
    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring="r2",
        cv=cv,
        n_jobs=-1,  # use all cpu cores, speed up as much as possible
    )
    search.fit(X_train, y_train)

    final = search.best_estimator_  # best model already refit on the full training set
    best_params = search.best_params_
    y_pred_train = final.predict(X_train)
    y_pred_test = final.predict(X_test)

    # build validation curve per hyperparameter, hold others at the best values
    curve_rows = []
    for hp in grid.keys():
        fixed = {k: v for k, v in best_params.items() if k != hp}  # best held params
        estimator.set_params(
            **fixed
        )  # set other params to best, validation_curve goest throught rest
        _, val_scores = validation_curve(
            estimator,
            X_train,
            y_train,
            param_name=hp,
            param_range=grid[hp],
            cv=cv,
            scoring="r2",
            n_jobs=-1,
        )
        val_means = val_scores.mean(axis=1)  # average across folds
        for i, v in enumerate(grid[hp]):
            curve_rows.append(
                {
                    "hyperparameter": hp,
                    "value": str(v),
                    "order": i,
                    "cv_r2": float(val_means[i]),
                }
            )
    tuning_curves = pd.DataFrame(curve_rows)

    return {
        "model": final,
        "feature_names": list(X_train.columns),
        "best_params": best_params,
        "tuning_curves": tuning_curves,
        "y_test": y_test,
        "y_pred": y_pred_test,
        "train_metrics": compute_metrics(y_train, y_pred_train),
        "test_metrics": compute_metrics(y_test, y_pred_test),
    }


# Scatter of predicted vs actual values with y=x reference
# Perfect model would have all points on diagonal (error = 0)
def plot_pred_vs_actual(result, model_label, target):
    df = pd.DataFrame(
        {"actual": result["y_test"].values, "predicted": result["y_pred"]}
    )
    plot = (
        ggplot(df, aes(x="actual", y="predicted"))
        + geom_point(alpha=0.3, size=1)
        + geom_abline(
            slope=1, intercept=0, color="#cc3333", linetype="dashed"
        )  # y=x reference line
        + labs(
            title=f"{model_label}: predicted vs actual ({target})",
            x=f"actual {target}",
            y=f"predicted {target}",
        )
        + theme_minimal()
    )
    return save_plot(
        plot, f"pred_vs_actual_{model_label.lower().replace(' ', '_')}_{target}"
    )


# Residuals plot with y=0 reference line, make sure assumptions hold
def plot_residuals(result, model_label, target):
    residuals = result["y_test"].values - result["y_pred"]
    df = pd.DataFrame({"predicted": result["y_pred"], "residual": residuals})
    plot = (
        ggplot(df, aes(x="predicted", y="residual"))
        + geom_point(alpha=0.3, size=1)
        + geom_hline(yintercept=0, color="#cc3333", linetype="dashed")
        + labs(
            title=f"{model_label}: residuals ({target})",
            x=f"predicted {target}",
            y="residual",
        )
        + theme_minimal()
    )
    return save_plot(
        plot, f"residuals_{model_label.lower().replace(' ', '_')}_{target}"
    )


# Horizontal bar chart of model coefficients (for Linear and Elastic Net)
# Bigger bar = bigger weight given to that feature
def plot_coefficients(result, model_label, target):
    coefs = pd.DataFrame(
        {"feature": result["feature_names"], "coefficient": result["model"].coef_}
    ).sort_values("coefficient")
    coefs["feature"] = pd.Categorical(
        coefs["feature"], categories=coefs["feature"], ordered=True
    )  # ordered category keeps bars sorted in the flipped plot
    plot = (
        ggplot(coefs, aes(x="feature", y="coefficient"))
        + geom_col(fill="#4682b4")
        + coord_flip()  # flip again, just looks better
        + labs(
            title=f"{model_label} coefficients ({target})",
            x="feature",
            y="coefficient",
        )
        + theme_minimal()
    )
    return save_plot(
        plot, f"coefficients_{model_label.lower().replace(' ', '_')}_{target}"
    )


# Horizontal bar chart of Random Forest feature importance
def plot_feature_importance(result, target):
    imps = pd.DataFrame(
        {
            "feature": result["feature_names"],
            "importance": result["model"].feature_importances_,
        }
    ).sort_values("importance")  # sort by importance
    imps["feature"] = pd.Categorical(
        imps["feature"], categories=imps["feature"], ordered=True
    )
    plot = (
        ggplot(imps, aes(x="feature", y="importance"))
        + geom_col(fill="#88aa55")
        + coord_flip()
        + labs(
            title=f"Random Forest feature importance ({target})",
            x="feature",
            y="importance",
        )
        + theme_minimal()
    )
    return save_plot(plot, f"feature_importance_{target}")


# Validation curves colored by hyperparameter (one panel per hyperparameter)
# Each curve shows CV R^2 as the hyperparameter varies, others held at best values (see above, this just plots the curves)
def plot_tuning_curves(result, model_label, target):
    curve_df = result["tuning_curves"].copy().sort_values(["hyperparameter", "order"])
    # dedupe values while keeping order so the x axis reads in sweep order
    seen = set()  # values we have already added (for fast lookup)
    category_order = []  # final ordered list of unique values
    for v in curve_df["value"]:
        if v not in seen:
            seen.add(v)
            category_order.append(v)
    curve_df["value"] = pd.Categorical(
        curve_df["value"], categories=category_order, ordered=True
    )
    plot = (
        ggplot(curve_df, aes(x="value", y="cv_r2", group="hyperparameter"))
        + geom_line(color="#4682b4")
        + geom_point(color="#4682b4")
        + facet_wrap(
            "hyperparameter", scales="free"
        )  # free x so each panel uses its own range
        + labs(
            title=f"{model_label} tuning curves ({target})",
            x="value",
            y="mean CV R^2",
        )
        + theme_minimal()
    )
    return save_plot(
        plot,
        f"tuning_{model_label.lower().replace(' ', '_')}_{target}",
        height=4,
    )


# Runs four models against a single target
# Returns dict of {model_label: results}
def run_target(df, target):
    print(f"\n=== Target: {target} ===")
    X, y = build_features(df, target)
    print(f"  features: {list(X.columns)}")
    print(f"  shape: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = split_80_20(X, y)
    X_train_s, X_test_s = scale_features(
        X_train, X_test
    )  # scaled versions for the three non-RF models

    print("  fitting Linear Regression")
    linear_result = fit_linear(X_train_s, X_test_s, y_train, y_test)

    print("  tuning Elastic Net (5-fold CV)")
    en_result = tune_with_cv(
        ElasticNet(
            max_iter=10000, random_state=SEED
        ),  # max_iter high since elastic net on skewed targets can converge kinda slow
        EN_GRID,
        X_train_s,
        X_test_s,
        y_train,
        y_test,
    )

    print("  tuning KNN (5-fold CV)")
    knn_result = tune_with_cv(
        KNeighborsRegressor(),
        KNN_GRID,
        X_train_s,
        X_test_s,
        y_train,
        y_test,
    )

    print("  tuning Random Forest (5-fold CV)")
    rf_result = tune_with_cv(
        RandomForestRegressor(random_state=SEED, n_jobs=-1),
        RF_GRID,
        X_train,  # RF uses raw features (doesnt care about scale)
        X_test,
        y_train,
        y_test,
    )

    return {
        "Linear": linear_result,
        "Elastic Net": en_result,
        "KNN": knn_result,
        "Random Forest": rf_result,
    }


# Builds all plots for every (model, target) pair, returns a dict keyed by tuple
# Keys are (plot_kind, model_label, target) so write_report can find them easily
def build_plots(results_by_target):
    paths = {}
    for target, results in results_by_target.items():
        for model_label, result in results.items():
            paths[("pred", model_label, target)] = plot_pred_vs_actual(
                result, model_label, target
            )
            paths[("resid", model_label, target)] = plot_residuals(
                result, model_label, target
            )
        # coefficient plots only for the linear models
        paths[("coef", "Linear", target)] = plot_coefficients(
            results["Linear"], "Linear", target
        )
        paths[("coef", "Elastic Net", target)] = plot_coefficients(
            results["Elastic Net"], "Elastic Net", target
        )
        # importance plot only for Random Forest
        paths[("importance", "Random Forest", target)] = plot_feature_importance(
            results["Random Forest"], target
        )
        # tuning curves only for the tuned models (not Linear, untuned)
        paths[("tuning", "Elastic Net", target)] = plot_tuning_curves(
            results["Elastic Net"], "Elastic Net", target
        )
        paths[("tuning", "KNN", target)] = plot_tuning_curves(
            results["KNN"], "KNN", target
        )
        paths[("tuning", "Random Forest", target)] = plot_tuning_curves(
            results["Random Forest"], "Random Forest", target
        )
    return paths


# Builds the regression report markdown, one section per target with all metrics, hyperparameters, and images embedded
def write_report(results_by_target, paths):
    lines = [
        "# Regression Report",
        "",
        "Generated from `data/processed/master_clean.csv` with `random_state=42`.",
        "",
    ]

    for target in TARGETS:
        lines += [f"## {target}", ""]
        lines += ["### Metrics (train / test)", ""]
        lines += [
            "| Model | RMSE | MAE | R^2 |",
            "| --- | --- | --- | --- |",
        ]
        for model_label in ["Linear", "Elastic Net", "KNN", "Random Forest"]:
            tr = results_by_target[target][model_label]["train_metrics"]
            te = results_by_target[target][model_label]["test_metrics"]
            lines.append(
                f"| {model_label} "
                f"| {tr['rmse']:,.4f} / {te['rmse']:,.4f} "
                f"| {tr['mae']:,.4f} / {te['mae']:,.4f} "
                f"| {tr['r2']:.4f} / {te['r2']:.4f} |"
            )
        lines.append("")

        # only the tuned models have best_params, skip linear
        lines += ["### Best hyperparameters", ""]
        lines += ["| Model | Best parameters |", "| --- | --- |"]
        for model_label in ["Elastic Net", "KNN", "Random Forest"]:
            bp = results_by_target[target][model_label]["best_params"]
            bp_str = ", ".join(f"`{k}={v}`" for k, v in bp.items())
            lines.append(f"| {model_label} | {bp_str} |")
        lines.append("")

        lines += ["### Predicted vs actual", ""]
        for model_label in ["Linear", "Elastic Net", "KNN", "Random Forest"]:
            rel = paths[("pred", model_label, target)].relative_to(REPORTS_DIR)
            lines += [f"#### {model_label}", "", f"![pred vs actual]({rel})", ""]

        lines += ["### Residuals", ""]
        for model_label in ["Linear", "Elastic Net", "KNN", "Random Forest"]:
            rel = paths[("resid", model_label, target)].relative_to(REPORTS_DIR)
            lines += [f"#### {model_label}", "", f"![residuals]({rel})", ""]

        lines += ["### Coefficients", ""]
        for model_label in ["Linear", "Elastic Net"]:
            rel = paths[("coef", model_label, target)].relative_to(REPORTS_DIR)
            lines += [f"#### {model_label}", "", f"![coefficients]({rel})", ""]

        lines += ["### Random Forest feature importance", ""]
        rel = paths[("importance", "Random Forest", target)].relative_to(REPORTS_DIR)
        lines += [f"![feature importance]({rel})", ""]

        lines += ["### Tuning curves", ""]
        for model_label in ["Elastic Net", "KNN", "Random Forest"]:
            rel = paths[("tuning", model_label, target)].relative_to(REPORTS_DIR)
            lines += [f"#### {model_label}", "", f"![tuning curves]({rel})", ""]

    REPORT_PATH.write_text("\n".join(lines))


# Runs both targets end to end and writes everything to disk via reports
def main():
    print(f"Loading {INPUT_PATH.relative_to(REPO_ROOT)}")
    df = pd.read_csv(INPUT_PATH)
    print(f"    {len(df):,} rows x {df.shape[1]} columns")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # dict comprehension runs both targets and keeps their result dicts under their name
    results_by_target = {target: run_target(df, target) for target in TARGETS}

    print("\nBuilding plots")
    paths = build_plots(results_by_target)

    print("Writing report")
    write_report(results_by_target, paths)

    print(f"\nWrote {len(paths)} images to {IMAGES_DIR.relative_to(REPO_ROOT)}")
    print(f"Wrote report to {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
