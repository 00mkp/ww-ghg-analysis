from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (  # visualizaiton imports
    aes,
    coord_flip,
    facet_wrap,
    geom_col,
    geom_line,
    geom_point,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_fill_gradient,
    scale_fill_gradient2,
    theme_minimal,
)
from sklearn.linear_model import LogisticRegression  # metrics and models
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# set up paths and shared output config
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "processed" / "master_clean.csv"
REPORTS_DIR = REPO_ROOT / "reports" / "classification"
IMAGES_DIR = REPORTS_DIR / "images"
REPORT_PATH = REPORTS_DIR / "report.md"

SEED = 42  # same seed used across pipelines

# hyperparameter grids for three tuned models (Logistic Regression baseline, no tuning)
KNN_GRID = {
    "n_neighbors": [3, 5, 7, 10, 15],
    "p": [1, 2],
}  # p=1 Manhattan, p=2 Euclidean, only tunes p then
SVM_GRID = {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]}
RF_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", 1.0],
}

# fixed continent order so confusion matrices and per class tables line up for models
CONTINENT_ORDER = [
    "Africa",
    "Asia",
    "Europe",
    "North America",
    "Oceania",
    "South America",
]

DPI = 150
WIDTH = 8
HEIGHT = 5


# Lowercases and swaps spaces for underscores so model labels can be used in filenames
def slug_case(s):
    return s.lower().replace(" ", "_")


# Saves a plot to the images directory and returns the path (used by the report writer)
def save_plot(plot, name, height=HEIGHT):
    path = IMAGES_DIR / f"{name}.png"
    plot.save(path, dpi=DPI, width=WIDTH, height=HEIGHT, verbose=False)
    return path


# same six base features always used, feature engineering adds 18 transformations on top
NUMERIC_FEATURES = [
    "gdp",
    "gdp_per_capita",
    "land_area",
    "population",
    "total_ghg",
    "ghg_per_capita",
]


# Adds signed log, squared, and reciprocal versions of each numeric feature (from ZyBooks)
# Signed log handles the small number of negative GHG values (net land use sinks for GHG emissions)
def add_transformations(df, columns):
    out = df.copy()
    for col in columns:
        x = df[col]
        out[f"log_{col}"] = np.sign(x) * np.log1p(np.abs(x))
        out[f"sq_{col}"] = x**2
        out[f"recip_{col}"] = 1.0 / (
            np.abs(x) + 1.0
        )  # +1 in denom guards against divide by zero
    return out


# Builds (features, target, groups) where groups is iso_code so countries don't leak across folds
# `engineered=False` returns only six raw features, True adds the 18 transformations (default)
def build_features_target_groups(df, engineered=True):
    target = df["continent"]
    groups = df["iso_code"]
    features = df.drop(columns=["iso_code", "country", "year", "continent"])
    if engineered:
        features = add_transformations(features, NUMERIC_FEATURES)
    return features, target, groups


# Stratified group split: each country (group) goes entirely to train or test, AND within each continent (class) the test proportion picked separately
# sklearn's StratifiedGroupKFold can do CV folds but doesn't give us one shot train/test split
def stratified_group_split(X, y, groups, test_size, random_state):
    rng = np.random.default_rng(random_state)
    group_class = pd.DataFrame({"g": groups.values, "c": y.values}).drop_duplicates(
        "g"
    )  # one row per country
    train_groups, test_groups = [], []
    for cls in sorted(group_class["c"].unique()):
        cls_groups = group_class.loc[group_class["c"] == cls, "g"].to_numpy()
        rng.shuffle(cls_groups)
        n_test = max(
            1, int(round(len(cls_groups) * test_size))
        )  # at least one country per class in test
        test_groups.extend(cls_groups[:n_test].tolist())
        train_groups.extend(cls_groups[n_test:].tolist())
    train_mask = groups.isin(train_groups).to_numpy()
    test_mask = groups.isin(test_groups).to_numpy()
    return np.where(train_mask)[0], np.where(test_mask)[0]


# Wrapper applying stratified group split at 80/20, then slicing the dfs accordingly
def split_80_20_group(X, y, groups):
    train_idx, test_idx = stratified_group_split(X, y, groups, 0.2, SEED)
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


# Standardizes features using scaler fit on X_train only, apply same scaler to X_test
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    cols = X_train.columns
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), columns=cols, index=X_train.index
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        columns=cols,
        index=X_test.index,  # preserve row index so test rows still line up with y_test
    )
    return X_train_s, X_test_s


# Returns accuracy and macro F1
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),  # use macro because of class imbalance
    }


# Returns per continent table of precision, recall, F1, and support
def per_class_table(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CONTINENT_ORDER, zero_division=0
    )
    return pd.DataFrame(
        {
            "continent": CONTINENT_ORDER,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


# Fits  baseline Logistic Regression (no tuning, just defaults with max_iter bumped, got ConvergenceWarning in testing sometimes)
def fit_logistic(X_train, X_test, y_train, y_test):
    model = LogisticRegression(
        max_iter=2000, random_state=SEED
    )  # raise max_iter so no convergence warnings
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    return {
        "model": model,
        "feature_names": list(X_train.columns),
        "classes": list(model.classes_),
        "y_train": y_train,
        "y_pred_train": y_pred_train,
        "y_test": y_test,
        "y_pred": y_pred_test,
        "train_metrics": compute_metrics(y_train, y_pred_train),
        "test_metrics": compute_metrics(y_test, y_pred_test),
        "per_class_test": per_class_table(y_test, y_pred_test),
    }


# Runs GridSearchCV with StratifiedGroupKFold (folds honor both country grouping and class stratification designed earlier)
# Builds validation curves per hyperparameter with the others held at the best found values like regression
def tune_with_cv(estimator, grid, X_train, X_test, y_train, y_test, groups_train):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(
        X_train, y_train, groups=groups_train
    )  # groups so countries stay together in folds

    final = search.best_estimator_
    best_params = search.best_params_
    y_pred_train = final.predict(X_train)
    y_pred_test = final.predict(X_test)

    # validation curves sweep one hyperparameter at time, others fixed at best
    curve_rows = []
    for hp in grid.keys():
        fixed = {k: v for k, v in best_params.items() if k != hp}
        estimator.set_params(
            **fixed
        )  # set other params to best, validation_curve goes through rest
        _, val_scores = validation_curve(
            estimator,
            X_train,
            y_train,
            param_name=hp,
            param_range=grid[hp],
            groups=groups_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        val_means = val_scores.mean(axis=1)
        for i, v in enumerate(grid[hp]):
            curve_rows.append(
                {
                    "hyperparameter": hp,
                    "value": str(v),
                    "order": i,
                    "cv_accuracy": float(val_means[i]),
                }
            )
    tuning_curves = pd.DataFrame(curve_rows)

    return {
        "model": final,
        "feature_names": list(X_train.columns),
        "classes": list(final.classes_),
        "best_params": best_params,
        "tuning_curves": tuning_curves,
        "y_train": y_train,
        "y_pred_train": y_pred_train,
        "y_test": y_test,
        "y_pred": y_pred_test,
        "train_metrics": compute_metrics(y_train, y_pred_train),
        "test_metrics": compute_metrics(y_test, y_pred_test),
        "per_class_test": per_class_table(y_test, y_pred_test),
    }


# 6x6 confusion matrix with cell counts added, fixed continent order on both axes
def plot_confusion(result, model_label):
    cm = confusion_matrix(result["y_test"], result["y_pred"], labels=CONTINENT_ORDER)
    rows = []
    for i, true_cls in enumerate(CONTINENT_ORDER):
        for j, pred_cls in enumerate(CONTINENT_ORDER):
            rows.append(
                {"true": true_cls, "predicted": pred_cls, "count": int(cm[i, j])}
            )
    cm_df = pd.DataFrame(rows)
    cm_df["true"] = pd.Categorical(
        cm_df["true"], categories=CONTINENT_ORDER, ordered=True
    )
    cm_df["predicted"] = pd.Categorical(
        cm_df["predicted"], categories=CONTINENT_ORDER, ordered=True
    )
    cm_df["label"] = cm_df["count"].astype(str)  # numeric labels on each cell

    plot = (
        ggplot(cm_df, aes(x="predicted", y="true", fill="count"))
        + geom_tile()
        + geom_text(aes(label="label"), color="black", size=8)
        + scale_fill_gradient(low="#f0f4fa", high="#4682b4")
        + labs(
            title=f"{model_label}: confusion matrix",
            x="predicted continent",
            y="true continent",
        )
        + theme_minimal()
    )
    return save_plot(plot, f"confusion_{slug_case(model_label)}", height=6)


# Heatmap of LR coefficients (one cell for (continent, feature) pair ) with color scale
# Positive means feature pushes prediction toward that continent, negative = pushes away
def plot_lr_coefficients(result, model_label):
    coef = result["model"].coef_
    classes = result["classes"]
    features = result["feature_names"]

    rows = []
    for i, cls in enumerate(classes):
        for j, feat in enumerate(features):
            rows.append({"class": cls, "feature": feat, "coefficient": coef[i, j]})
    coef_df = pd.DataFrame(rows)
    coef_df["class"] = pd.Categorical(
        coef_df["class"], categories=classes, ordered=True
    )
    coef_df["feature"] = pd.Categorical(
        coef_df["feature"], categories=features, ordered=True
    )
    coef_df["label"] = coef_df["coefficient"].apply(
        lambda v: f"{v:.2f}"
    )  # 2 decimal labels on each cell, keep formatting nice

    plot = (
        ggplot(
            coef_df, aes(x="class", y="feature", fill="coefficient")
        )  # continents on x (6 values), features on y (24 stacked vertically to be)
        + geom_tile()
        + geom_text(aes(label="label"), color="black", size=8)
        + scale_fill_gradient2(
            low="#cc6677", mid="#ffffff", high="#4682b4", midpoint=0
        )  # diverging scale centered at 0
        + labs(
            title=f"{model_label} coefficients (continent x feature)",
            x="continent",
            y="feature",
        )
        + theme_minimal()
    )
    return save_plot(
        plot, f"coefficients_{slug_case(model_label)}", height=10
    )  # bump height since features now on y axis


# Horizontal bar chart showing Random Forest feature importances
def plot_rf_importance(result, model_label):
    imps = pd.DataFrame(
        {
            "feature": result["feature_names"],
            "importance": result["model"].feature_importances_,
        }
    ).sort_values("importance")
    imps["feature"] = pd.Categorical(
        imps["feature"], categories=imps["feature"], ordered=True
    )
    plot = (
        ggplot(imps, aes(x="feature", y="importance"))
        + geom_col(fill="#88aa55")
        + coord_flip()
        + labs(
            title=f"{model_label} feature importance",
            x="feature",
            y="importance",
        )
        + theme_minimal()
    )
    return save_plot(plot, f"feature_importance_{slug_case(model_label)}")


# Validation curves colored by hyperparameter (one curve per hyperparameter)
def plot_tuning_curves(result, model_label):
    curve_df = result["tuning_curves"].copy().sort_values(["hyperparameter", "order"])
    # dedupe values while keeping order so x axis reads in sweep order
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
        ggplot(curve_df, aes(x="value", y="cv_accuracy", group="hyperparameter"))
        + geom_line(color="#4682b4")
        + geom_point(color="#4682b4")
        + facet_wrap("hyperparameter", scales="free")
        + labs(
            title=f"{model_label} tuning curves",
            x="value",
            y="mean CV accuracy",
        )
        + theme_minimal()
    )
    return save_plot(plot, f"tuning_{slug_case(model_label)}", height=4)


# Runs all four models (LR + 3 tuned) end to end on a single feature configuration
# `engineered=False` uses raw features; True adds the 18 transformations
def run_all(df, engineered=True):
    X, y, groups = build_features_target_groups(df, engineered=engineered)
    print(f"Features: {list(X.columns)}")
    print(f"Shape: X={X.shape}, y={y.shape}")
    print(f"Unique countries: {groups.nunique()}")

    X_train, X_test, y_train, y_test = split_80_20_group(X, y, groups)
    groups_train = groups.loc[
        X_train.index
    ]  # pull groups for just the training countries
    n_train = groups_train.nunique()
    n_test = groups.loc[X_test.index].nunique()
    print(f"\n80/20 split: train countries={n_train}, test countries={n_test}")

    X_train_s, X_test_s = scale_features(
        X_train, X_test
    )  # scaled versions for non RF models

    print("\nFitting Logistic Regression")
    lr_result = fit_logistic(X_train_s, X_test_s, y_train, y_test)

    print("Tuning KNN (5-fold stratified group CV)")
    knn_result = tune_with_cv(
        KNeighborsClassifier(),
        KNN_GRID,
        X_train_s,
        X_test_s,
        y_train,
        y_test,
        groups_train,
    )

    print("Tuning SVM (5-fold stratified group CV)")
    svm_result = tune_with_cv(
        SVC(random_state=SEED),
        SVM_GRID,
        X_train_s,
        X_test_s,
        y_train,
        y_test,
        groups_train,
    )

    print("Tuning Random Forest (5-fold stratified group CV)")
    rf_result = tune_with_cv(
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        RF_GRID,
        X_train,  # RF uses raw features, no scaling needed
        X_test,
        y_train,
        y_test,
        groups_train,
    )

    return {
        "Logistic Regression": lr_result,
        "KNN": knn_result,
        "SVM": svm_result,
        "Random Forest": rf_result,
    }


# Builds every plot for report and returns dict of paths keyed by (plot_kind, model_label)
def build_plots(results):
    paths = {}
    for model_label, result in results.items():
        paths[("confusion", model_label)] = plot_confusion(result, model_label)
    # coefficient heatmap only for Logistic Regression
    paths[("coefficients", "Logistic Regression")] = plot_lr_coefficients(
        results["Logistic Regression"], "Logistic Regression"
    )
    # feature importance only for Random Forest
    paths[("importance", "Random Forest")] = plot_rf_importance(
        results["Random Forest"], "Random Forest"
    )
    # tuning curves for the tuned models (LR not tuned)
    paths[("tuning", "KNN")] = plot_tuning_curves(results["KNN"], "KNN")
    paths[("tuning", "SVM")] = plot_tuning_curves(results["SVM"], "SVM")
    paths[("tuning", "Random Forest")] = plot_tuning_curves(
        results["Random Forest"], "Random Forest"
    )
    return paths


# Builds the classification report markdown including the raw vs engineered comparison table
def write_report(results, results_raw, paths):
    lines = [
        "# Classification Report",
        "",
        "Generated from `data/processed/master_clean.csv` with `random_state=42`. "
        "Uses country level stratified group splits.",
        "",
        "## Feature engineering comparison (test set)",
        "",
        "Two configurations were trained on the same splits: **raw** uses the six "
        "original numeric features; **engineered** adds log, squared, and reciprocal "
        "transformations of each. Feature engineering helped Logistic Regression, KNN, and "
        "SVM noticeably (roughly 6 to 11 points) and slightly hurt Random "
        "Forest. The rest of the report uses the feature engineered configuration as the "
        "primary view since it gave the best test metric overall.",
        "",
        "| Model | Accuracy (raw) | Accuracy (engineered) | F1 (raw) | F1 (engineered) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for model_label in ["Logistic Regression", "KNN", "SVM", "Random Forest"]:
        raw = results_raw[model_label]["test_metrics"]
        eng = results[model_label]["test_metrics"]
        lines.append(
            f"| {model_label} "
            f"| {raw['accuracy']:.4f} "
            f"| {eng['accuracy']:.4f} "
            f"| {raw['f1']:.4f} "
            f"| {eng['f1']:.4f} |"
        )
    lines.append("")

    lines += [
        "## Summary metrics (train / test) - engineered features",
        "",
        "| Model | Accuracy | F1 |",
        "| --- | --- | --- |",
    ]
    for model_label in ["Logistic Regression", "KNN", "SVM", "Random Forest"]:
        tr = results[model_label]["train_metrics"]
        te = results[model_label]["test_metrics"]
        lines.append(
            f"| {model_label} "
            f"| {tr['accuracy']:.4f} / {te['accuracy']:.4f} "
            f"| {tr['f1']:.4f} / {te['f1']:.4f} |"
        )
    lines.append("")

    lines += ["## Best hyperparameters", ""]
    lines += ["| Model | Best parameters |", "| --- | --- |"]
    for model_label in ["KNN", "SVM", "Random Forest"]:
        bp = results[model_label]["best_params"]
        bp_str = ", ".join(f"`{k}={v}`" for k, v in bp.items())
        lines.append(f"| {model_label} | {bp_str} |")
    lines.append("")

    lines += ["## Per class metrics (test set)", ""]
    for model_label in ["Logistic Regression", "KNN", "SVM", "Random Forest"]:
        lines += [f"### {model_label}", ""]
        pc = results[model_label]["per_class_test"]
        lines += [
            "| Continent | Precision | Recall | F1 | Support |",
            "| --- | --- | --- | --- | --- |",
        ]
        for _, row in pc.iterrows():
            lines.append(
                f"| {row['continent']} "
                f"| {row['precision']:.4f} "
                f"| {row['recall']:.4f} "
                f"| {row['f1']:.4f} "
                f"| {int(row['support'])} |"
            )
        lines.append("")

    lines += ["## Confusion matrices", ""]
    for model_label in ["Logistic Regression", "KNN", "SVM", "Random Forest"]:
        rel = paths[("confusion", model_label)].relative_to(REPORTS_DIR)
        lines += [f"### {model_label}", "", f"![confusion matrix]({rel})", ""]

    lines += ["## Logistic Regression coefficients", ""]
    rel = paths[("coefficients", "Logistic Regression")].relative_to(REPORTS_DIR)
    lines += [f"![coefficients]({rel})", ""]

    lines += ["## Random Forest feature importance", ""]
    rel = paths[("importance", "Random Forest")].relative_to(REPORTS_DIR)
    lines += [f"![feature importance]({rel})", ""]

    lines += ["## Tuning curves", ""]
    for model_label in ["KNN", "SVM", "Random Forest"]:
        rel = paths[("tuning", model_label)].relative_to(REPORTS_DIR)
        lines += [f"### {model_label}", "", f"![tuning curves]({rel})", ""]

    REPORT_PATH.write_text("\n".join(lines))


# Runs classification pipeline twice (with and without feature engineering), writes the report
def main():
    print(f"Loading {INPUT_PATH.relative_to(REPO_ROOT)}")
    df = pd.read_csv(INPUT_PATH)
    print(f"    {len(df):,} rows x {df.shape[1]} columns")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # full pipeline with engineered features (used as primary results)
    print("\n=== Pipeline WITH feature engineering ===")
    results = run_all(df, engineered=True)

    # full pipeline with raw features (used in comparison table)
    print("\n=== Pipeline WITHOUT feature engineering ===")
    results_raw = run_all(df, engineered=False)

    print("\nBuilding plots")
    paths = build_plots(results)  # only engineered configuration gets plots

    print("Writing report")
    write_report(results, results_raw, paths)

    print(f"\nWrote {len(paths)} images to {IMAGES_DIR.relative_to(REPO_ROOT)}")
    print(f"Wrote report to {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
