from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    geom_line,
    geom_point,
    ggplot,
    labs,
    theme_minimal,
)
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# set up paths and shared output config
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "processed" / "master_clean.csv"
REPORTS_DIR = REPO_ROOT / "reports" / "clustering"
IMAGES_DIR = REPORTS_DIR / "images"
REPORT_PATH = REPORTS_DIR / "report.md"

SEED = 42
SNAPSHOT_YEAR = (
    2024  # use the most recent year so the clusters are "where countries are now"
)
K_RANGE = [2, 3, 4, 5, 6]  # k=6 included so we can compare to the six continent split

# six numeric used for the clustering distance calculations
NUMERIC_FEATURES = [
    "gdp",
    "gdp_per_capita",
    "land_area",
    "population",
    "total_ghg",
    "ghg_per_capita",
]

# fixed continent order so contingency tables read same across runs
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


# Saves plot to images directory and returns path (used by  report writer)
def save_plot(plot, name, height=HEIGHT):
    path = IMAGES_DIR / f"{name}.png"
    plot.save(path, dpi=DPI, width=WIDTH, height=height, verbose=False)
    return path


# Loads the cleaned master and filters to just snapshot year
def load_snapshot(path, year):
    df = pd.read_csv(path)
    snap = df[df["year"] == year].copy().reset_index(drop=True)
    return snap


# Standardizes features (zero mean, unit variance) before clustering
# K-Means and HAC distance based, need to scale
def scale_features(features):
    scaler = StandardScaler()
    return pd.DataFrame(
        scaler.fit_transform(features), columns=features.columns, index=features.index
    )


# Applies signed log1p to every column, compresses long tails
# Signed form keeps the direction of negative GHG values (net land use sinks like before )
def log_transform_features(features):
    out = pd.DataFrame(index=features.index)
    for col in features.columns:
        x = features[col]
        out[col] = np.sign(x) * np.log1p(np.abs(x))
    return out


# Fits K-Means and returns cluster labels (n_init=10 reruns from different random starts, keeps best)
def fit_kmeans(X, k):
    return KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(X)


# Fits HAC using Ward linkage
def fit_hac(X, k):
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


# Builds a table of (k, kmeans_inertia, kmeans_silhouette, hac_silhouette) used for elbow and silhouette plots
def k_selection_table(X):
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
        km_labels = km.labels_
        km_inertia = float(
            km.inertia_
        )  # inertia = within cluster sum of squared distances to centroid
        km_sil = float(silhouette_score(X, km_labels)) if k > 1 else float("nan")

        hac_labels = fit_hac(X, k)
        hac_sil = float(silhouette_score(X, hac_labels)) if k > 1 else float("nan")

        rows.append(
            {
                "k": k,
                "kmeans_inertia": km_inertia,
                "kmeans_silhouette": km_sil,
                "hac_silhouette": hac_sil,
            }
        )
    return pd.DataFrame(rows)


# K-Means elbow plot (inertia vs k), "elbow" k is where adding more clusters stops adding anything
def plot_elbow(diagnostics, variant):
    plot = (
        ggplot(diagnostics, aes(x="k", y="kmeans_inertia"))
        + geom_line(color="#4682b4")
        + geom_point(color="#4682b4", size=2)
        + labs(
            title=f"K-Means elbow plot ({variant})",
            x="k",
            y="inertia (within-cluster SSE)",
        )
        + theme_minimal()
    )
    return save_plot(plot, f"elbow_kmeans_{variant}", height=4)


# Side by side silhouette plot for K-Means and HAC across k
def plot_silhouettes(diagnostics, variant):
    # melt wide diagnostics table to long so both algorithms can be in one plot
    long = diagnostics.melt(
        id_vars=["k"],
        value_vars=["kmeans_silhouette", "hac_silhouette"],
        var_name="algorithm",
        value_name="silhouette",
    )
    long["algorithm"] = long["algorithm"].map(
        {"kmeans_silhouette": "K-Means", "hac_silhouette": "HAC"}
    )
    plot = (
        ggplot(long, aes(x="k", y="silhouette", color="algorithm", group="algorithm"))
        + geom_line()
        + geom_point(size=2)
        + labs(
            title=f"Silhouette score by k ({variant})",
            x="k",
            y="silhouette score",
            color="algorithm",
        )
        + theme_minimal()
    )
    return save_plot(plot, f"silhouette_by_k_{variant}", height=4)


# Returns k that maximizes named diagnostic column (used for picking final k for each algorithm)
def pick_best_k(diagnostics, column):
    best_row = diagnostics.loc[diagnostics[column].idxmax()]
    return int(best_row["k"])


# Builds per cluster summary, count of countries and mean of each feature (to get cluster characteristics)
def cluster_summary(snap, labels, features):
    df = snap.copy()
    df["cluster"] = labels
    grouped = df.groupby("cluster")
    rows = []
    for cluster_id, sub in grouped:
        row = {"cluster": int(cluster_id), "count": int(len(sub))}
        for feat in features:
            row[feat] = float(sub[feat].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


# Builds (cluster x continent) contingency table with per cluster purity (fraction from dominant continent)
def continent_breakdown(snap, labels):
    df = snap.copy()
    df["cluster"] = labels
    table = df.groupby(["cluster", "continent"]).size().unstack(fill_value=0)
    # make sure every continent appears as column even if 0 countries in that cluster
    for c in CONTINENT_ORDER:
        if c not in table.columns:
            table[c] = 0
    table = table[CONTINENT_ORDER]  # fix column order
    table["total"] = table.sum(axis=1)
    table["dominant"] = table[CONTINENT_ORDER].idxmax(
        axis=1
    )  # most common continent in cluster
    table["purity"] = (
        table[CONTINENT_ORDER].max(axis=1) / table["total"]
    )  # fraction of cluster that is dominant continent
    return table.reset_index()


# Returns {cluster_id: [sorted country names]} for per cluster country lists in  report
def countries_per_cluster(snap, labels):
    df = snap.copy()
    df["cluster"] = labels
    return {int(c): sorted(sub["country"].tolist()) for c, sub in df.groupby("cluster")}


# 2D PCA projection of already scaled features, colored by either cluster id or continent
# PC1/PC2 axes show how much variance they explain in each title
def plot_pca_scatter(X, labels, color_label, name, variant):
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(X)
    df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], color_label: labels})
    df[color_label] = df[color_label].astype(
        str
    )  # cast to str so plotnine treats as discrete
    plot = (
        ggplot(df, aes(x="PC1", y="PC2", color=color_label))
        + geom_point(size=2, alpha=0.7)
        + labs(
            title=f"PCA projection ({variant}) - colored by {color_label}",
            x=f"PC1 ({pca.explained_variance_ratio_[0]:.0%} variance)",
            y=f"PC2 ({pca.explained_variance_ratio_[1]:.0%} variance)",
        )
        + theme_minimal()
    )
    return save_plot(plot, f"pca_{name}_{variant}", height=5)


# Runs full clustering pipeline for one feature variant (raw or log)
# Picks k via silhouette per algorithm, fits both, builds summaries and all plots
def run_variant(snap, X, variant):
    print(f"[{variant}] computing k selection diagnostics")
    diagnostics = k_selection_table(X)

    km_k = pick_best_k(diagnostics, "kmeans_silhouette")
    hac_k = pick_best_k(diagnostics, "hac_silhouette")
    print(f"[{variant}] K-Means best k = {km_k}, HAC best k = {hac_k}")

    km_labels = fit_kmeans(X, km_k)
    hac_labels = fit_hac(X, hac_k)

    # build per cluster summaries, continent breakdowns, and country lists for both algorithms
    km_summary = cluster_summary(snap, km_labels, NUMERIC_FEATURES)
    hac_summary = cluster_summary(snap, hac_labels, NUMERIC_FEATURES)
    km_breakdown = continent_breakdown(snap, km_labels)
    hac_breakdown = continent_breakdown(snap, hac_labels)
    km_countries = countries_per_cluster(snap, km_labels)
    hac_countries = countries_per_cluster(snap, hac_labels)

    print(f"[{variant}] building plots")
    paths = {
        "elbow": plot_elbow(diagnostics, variant),
        "silhouette": plot_silhouettes(diagnostics, variant),
        "pca_kmeans": plot_pca_scatter(
            X, km_labels, "K-Means cluster", "kmeans", variant
        ),
        "pca_hac": plot_pca_scatter(X, hac_labels, "HAC cluster", "hac", variant),
        "pca_continent": plot_pca_scatter(
            X, snap["continent"].to_numpy(), "continent", "continent", variant
        ),
    }

    return {
        "variant": variant,
        "diagnostics": diagnostics,
        "km_k": km_k,
        "hac_k": hac_k,
        "km_labels": km_labels,
        "hac_labels": hac_labels,
        "km_summary": km_summary,
        "hac_summary": hac_summary,
        "km_breakdown": km_breakdown,
        "hac_breakdown": hac_breakdown,
        "km_countries": km_countries,
        "hac_countries": hac_countries,
        "paths": paths,
    }


# Builds one full report section (k selection table, plots, cluster summaries, country lists) for single variant
def variant_section(snap, result, heading):
    diagnostics = result["diagnostics"]
    km_k = result["km_k"]
    hac_k = result["hac_k"]
    paths = result["paths"]

    lines = [
        f"## {heading}",
        "",
        "### k selection",
        "",
        "| k | K-Means inertia | K-Means silhouette | HAC silhouette |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in diagnostics.iterrows():
        lines.append(
            f"| {int(row['k'])} "
            f"| {row['kmeans_inertia']:.2f} "
            f"| {row['kmeans_silhouette']:.4f} "
            f"| {row['hac_silhouette']:.4f} |"
        )
    lines += [
        "",
        f"Chosen k: **K-Means = {km_k}** (highest silhouette), "
        f"**HAC = {hac_k}** (highest silhouette).",
        "",
        "### Elbow plot (K-Means)",
        "",
        f"![elbow]({paths['elbow'].relative_to(REPORTS_DIR)})",
        "",
        "### Silhouette scores",
        "",
        f"![silhouette]({paths['silhouette'].relative_to(REPORTS_DIR)})",
        "",
    ]

    for algo, summary_key, breakdown_key, countries_key, k in [
        ("K-Means", "km_summary", "km_breakdown", "km_countries", km_k),
        ("HAC", "hac_summary", "hac_breakdown", "hac_countries", hac_k),
    ]:
        summary = result[summary_key]
        breakdown = result[breakdown_key]
        countries = result[countries_key]

        lines += [f"### {algo} (k = {k})", "", "**Cluster summaries**", ""]
        cols = ["cluster", "count"] + NUMERIC_FEATURES
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in summary.iterrows():
            parts = [str(int(row["cluster"])), str(int(row["count"]))]
            for feat in NUMERIC_FEATURES:
                parts.append(f"{row[feat]:,.2f}")
            lines.append("| " + " | ".join(parts) + " |")
        lines.append("")

        lines += ["**Continent breakdown and purity**", ""]
        header = ["cluster"] + CONTINENT_ORDER + ["total", "dominant", "purity"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for _, row in breakdown.iterrows():
            parts = [str(int(row["cluster"]))]
            for c in CONTINENT_ORDER:
                parts.append(str(int(row[c])))
            parts.append(str(int(row["total"])))
            parts.append(str(row["dominant"]))
            parts.append(f"{row['purity']:.2%}")
            lines.append("| " + " | ".join(parts) + " |")
        lines.append("")

        lines += ["**Countries per cluster**", ""]
        for cluster_id, country_list in sorted(countries.items()):
            lines.append(
                f"- **Cluster {cluster_id}** ({len(country_list)} countries): "
                + ", ".join(country_list)
            )
        lines.append("")

    lines += [
        "### PCA scatter",
        "",
        "**K-Means clusters**",
        "",
        f"![pca kmeans]({paths['pca_kmeans'].relative_to(REPORTS_DIR)})",
        "",
        "**HAC clusters**",
        "",
        f"![pca hac]({paths['pca_hac'].relative_to(REPORTS_DIR)})",
        "",
        "**Continent (reference)**",
        "",
        f"![pca continent]({paths['pca_continent'].relative_to(REPORTS_DIR)})",
        "",
    ]
    return lines


# Builds silhouette comparison table that sits at top of report (raw vs log, by k)
def comparison_table(raw, log):
    raw_d = raw["diagnostics"]
    log_d = log["diagnostics"]
    merged = raw_d.merge(
        log_d, on="k", suffixes=("_raw", "_log")
    )  # join on k, suffix so columns do not collide
    lines = [
        "## Variant comparison",
        "",
        "Silhouette scores side by side. Higher is better.",
        "",
        "| k | K-Means raw | K-Means log | HAC raw | HAC log |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in merged.iterrows():
        lines.append(
            f"| {int(row['k'])} "
            f"| {row['kmeans_silhouette_raw']:.4f} "
            f"| {row['kmeans_silhouette_log']:.4f} "
            f"| {row['hac_silhouette_raw']:.4f} "
            f"| {row['hac_silhouette_log']:.4f} |"
        )
    lines.append("")
    return lines


# Builds full clustering markdown report including the why two variants section, comparison table, and one section for each variant
def write_report(snap, raw, log):
    # pull raw cluster size split (like 175 vs 4 or similar) to put into explanation at top
    raw_km_top = raw["km_breakdown"][["cluster", "total"]].sort_values(
        "total", ascending=False
    )
    raw_split_note = (
        f"largest cluster has {int(raw_km_top['total'].iloc[0])} of {len(snap)} "
        f"countries; smallest has {int(raw_km_top['total'].iloc[-1])}"
    )
    lines = [
        "# Clustering Report",
        "",
        f"Generated from `data/processed/master_clean.csv` ({SNAPSHOT_YEAR} snapshot, "
        f"{len(snap)} countries) with `random_state={SEED}`. "
        "Two feature configurations are reported side by side: **raw standardized "
        "features** and **signed log1p transformed standardized features**.",
        "",
        "## Why two variants",
        "",
        "The six features are heavily skewed to the right (GDP, population, "
        "total GHG span a huge scale). Distance based clustering on "
        "the raw standardized values is dominated by a handful of huge economies; "
        "K-Means at k=2 isolates the few largest outliers into one cluster and "
        f"dumps everything else into the other ({raw_split_note}).",
        "",
        "The raw variant's silhouette scores are *higher* than the log variant's "
        "in the table below, but that is exactly the failure that we saw: silhouette "
        "rewards a split where one cluster is just a few points sitting "
        "very far from a very dense blob, even though almost no information "
        "is conveyed outside of the split. Applying a signed `log1p` transform *before* "
        "standardization compresses the long tails, lets distances reflect "
        "relative differences across the full range of countries, and produces "
        "more meaningful groupings at the cost of a lower "
        "silhouette number, which is the correct tradeoff here in our opinion.",
        "",
        "Both variants are kept in this report so the difference can be seen.",
        "",
    ]
    lines += comparison_table(raw, log)
    lines += variant_section(snap, raw, "Variant 1 - Raw features")
    lines += variant_section(snap, log, "Variant 2 - Log transformed features")

    REPORT_PATH.write_text("\n".join(lines))


# Runs both feature variants (raw and log) end to end and writes combined report
def main():
    print(f"Loading {INPUT_PATH.relative_to(REPO_ROOT)}")
    df_full = pd.read_csv(INPUT_PATH)
    snap = load_snapshot(
        INPUT_PATH, SNAPSHOT_YEAR
    )  # just the 2024 row per country (explained elsewhere)
    print(f"    full rows: {len(df_full):,}, {SNAPSHOT_YEAR} snapshot: {len(snap)}")
    if len(snap) == 0:
        raise RuntimeError(
            f"no rows for year {SNAPSHOT_YEAR} after cleaning"
        )  # fail immediately if snapshot year gone after cleaning

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # build both feature matrices, have raw scaled and signed log1p then scaled
    raw_features = snap[NUMERIC_FEATURES]
    X_raw = scale_features(raw_features)
    X_log = scale_features(log_transform_features(raw_features))

    raw_result = run_variant(snap, X_raw, "raw")
    log_result = run_variant(snap, X_log, "log")

    print("Writing report")
    write_report(snap, raw_result, log_result)

    total_images = len(raw_result["paths"]) + len(log_result["paths"])
    print(f"\nWrote {total_images} images to {IMAGES_DIR.relative_to(REPO_ROOT)}")
    print(f"Wrote report to {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
