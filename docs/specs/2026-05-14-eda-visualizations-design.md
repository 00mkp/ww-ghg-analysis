# Design - EDA Visualizations

**Date:** 2026-05-14

## Goal

To conduct explanatory data analysis via visualizations on `data/processed/master.csv` to analyze feature distributions, time series behavior, pairwise relationships, the classification target's class balances, and `NaN` counts. Output helps to inform the `NaN` handling strategy and any feature transformationss (log, etc.) used later.

All charts built with `plotnine` (from CPSC392). EDA runs as a standalone Python script (`src/eda.py`), as all other scripts do; outputs are written to repo as PNGs plus a generated markdown report.

## Tasks

- Create smoothed distribution plots for the numeric features.
- Create time series plots of the numeric features against `year`.
- Select bivariate scatterplot pairs and create the scatterplot.
- Create a continent class distribution.
- Create `NaN` counts overview.
- Output summary statistics (mean, median, std dev) with the distribution plots.

## Visualizations

### Smoothed distributions (per numeric feature)

Each (country, year) row is to be treated as one observation. We plot a smoothed curve per feature showing the distribution shape and tail.

Features:
- `total_ghg`
- `ghg_per_capita`
- `gdp`
- `gdp_per_capita`
- `population`

`land_area` is left out; it's basically constant per country, and its near uniform distribution won't really tell us anything.

Per feature summary statistics (mean, median, std dev) are printed near each chart.

A log scale is used where the linear scale chart has most data points near zero (most likely all five due to intuitive outliers like China, USA, or India). A linear scale used where the skew is important (decided case by case).

### Time series (per numeric feature against `year`)

One line per country for the joined year range:
- `total_ghg`
- `ghg_per_capita`
- `gdp`
- `gdp_per_capita`
- `population`

`land_area` left because it is basically constant per country (see `data/README.md`).

A log scale is probably needed where multiple countries share an axis, or USA/China dominate and the rest look like flat lines.

### Bivariate views

A selection of scatter plots between feature pairs, but not a full correlation matrix. Some initial pairs to include:

- `gdp` vs `total_ghg`
- `gdp_per_capita` vs `ghg_per_capita`
- `population` vs `total_ghg`
- `land_area` vs `total_ghg`

Each colored by continent so geographic clusters are visible.

### Continent class distribution

A bar chart of country count per continent. Will show the class imbalance (Africa has ~54 countries , Asia ~50, Europe ~46, North America ~23, Oceania ~14, South America ~12) which matters for classification model evaluation.

### NaN counts

Bar chart of `NaN` count per column. Helps to choose a `NaN` handling strategy (imputation or drop or ignore).

## Output

`src/eda.py` writes to a gitignored `reports/eda/` directory:

```
reports/eda/
├── report.md            # generated markdown report with embedded images + statistics tables
└── images/              # one PNG per chart, named by section + feature
```

The markdown report sections follow the visualization sections mentioned/shown above and insert images via relative paths.
