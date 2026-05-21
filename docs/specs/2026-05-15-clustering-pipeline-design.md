# Design - Clustering Pipeline

**Date:** 2026-05-15

## Goal

Apply unsupervised clustering to country level profiles and see whether the resulting groups are meaningful (i.e., emissions tiers, recognizable continent patterns). Output a report with the cluster assignments, per cluster statistics, visualizations, and a comparison of cluster structure against the known continent labels.

## Granularity

**Country level**, one observation per country, using each country's **most recent year (2024) row** from the cleaned master dataset. Yields ~191 observations. This supports a "groups of similar countries by current developmental profile" view, which is more interpretable than groups of country years.

## Features

The six numeric features:

- `gdp`
- `gdp_per_capita`
- `land_area`
- `population`
- `total_ghg`
- `ghg_per_capita`

All features standardized with `StandardScaler` before clustering. Both algorithms chosen are distance based.

Excluded: `iso_code`, `country`, `year`, `continent`. Continent is held out to be able to be used for analysis of potential continent clusters.

### Feature variants (raw vs log transformed)

The pipeline runs **two feature configurations side by side** and reports both:

1. **Raw**: `StandardScaler` applied directly to the six numeric features.
2. **Log transformed**: signed `log1p` (`sign(x) * log1p(|x|)`) applied to each feature first, then `StandardScaler`. Signed form is required because GHG has a small number of negative values.

Reason for the second variant: the raw features are heavily skewed to the right (GDP, population, total GHG are massively scaled). When K-Means is run on the standardized raw features, the top silhouette score lands at **k=2** with one cluster containing the 4 largest economies (China, US, India, Russia) and the other containing the remaining ~175 countries. Silhouette rewards that split because the outliers are extremely far from everything else, but the result is not interesting; we just got top emitters versus not top emitters.

The log variant produces lower silhouette scores than the raw variant but more interpretable and interesting groupings (e.g. K-Means k=2 splits 57 vs 122 instead of 4 vs 175). Intersetingly the silhouette score remained similar for both clusters; after some research, this is a known limitation of silhouette scores not *informing* the personality or type of clusters made.

## Algorithms

| Algorithm | sklearn class | Rationale |
| --- | --- | --- |
| K-Means | `sklearn.cluster.KMeans` | Minimizes within cluster sum of squares. The default clustering baseline. |
| HAC (Hierarchical Agglomerative Clustering) | `sklearn.cluster.AgglomerativeClustering` | Hierarchical merging with **Ward linkage** (minimizes within cluster variance, pairs naturally with K-Means). Provides a different perspective on the same data and produces a dendrogram structure. |

`random_state=42` for K-Means (as for all other pipelines)

## Choosing k

For both algorithms, evaluate **k in {2, 3, 4, 5, 6}**. Two criteria:

- **Elbow plot:** plot the metric vs k. The "elbow" is the point where adding more clusters stops paying off.
- **Silhouette score:** averaged over points, a single number per k. Higher means better separated. Pick the k that maximizes silhouette, or default to elbow point if both methods disagree.

We include k=6 explicitly so we can compare to the six continents in the cluster vs continent analysis.

## Evaluation

Pure unsupervised, no train/test split. Per algorithm:

1. **Silhouette score** at the chosen k.
2. **Inertia (within cluster sum of squares)** (K-Means) / **per cluster variance** (HAC).
3. **Cluster vs continent contingency table:** counts of countries in each (cluster, continent) cell, and a per cluster **purity** number (fraction of the cluster from its dominant continent). Together these answer "do clusters align with continents?" using simple counts: a cluster that is mostly one continent has high purity; a cluster spread across continents has low purity.
4. **Cluster composition:** how many countries per cluster, what feature distributions look like within each cluster.

## Report Contents

`reports/clustering/report.md` plus `reports/clustering/images/`. Top of the report explains the raw vs log variant motivation, then a side by side silhouette comparison table, then one full section per variant. Each variant section contains:

1. **k selection diagnostics:** elbow and silhouette plots for K-Means and HAC.
2. **Chosen k and rationale:** short note for each algorithm.
3. **Cluster summaries:** per cluster (per algorithm), a table showing number of countries, mean of each numeric feature, dominant continent breakdown.
4. **Cluster vs continent comparison:** contingency table of continent counts per cluster, plus a per cluster purity number.
5. **Country lists:** countries in each cluster, per algorithm.
6. **PCA scatter:** 2D PCA projection of the standardized features. Three plots, colored by K-Means cluster, colored by HAC cluster, colored by continent (for reference).

Filenames include the variant suffix (e.g. `elbow_kmeans_raw.png`, `elbow_kmeans_log.png`) so the two sets of plots both exist in `images/`.

## Output

```
reports/clustering/
├── report.md
└── images/          # elbow, silhouette, PCA scatters
```

Gitignored, same convention as the other pipelines.

## Architecture

Single script: `src/clustering.py`. Flat functions for data preparation, scaling, fitting each algorithm, evaluation, and plotting. `main()` creates two feature matrices (raw scaled and log then scaled), then calls `run_variant()` once per matrix; `write_report()` combines both results into one markdown report.

Random seed `42` for K-Means (as before); HAC is the same with the same data.
