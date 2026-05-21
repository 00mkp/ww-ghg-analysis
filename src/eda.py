import math
from pathlib import Path

import pandas as pd
from plotnine import (  # import everything for desired EDA
    aes,
    coord_flip,
    facet_wrap,
    geom_col,
    geom_density,
    geom_line,
    geom_point,
    ggplot,
    labs,
    scale_x_log10,
    scale_y_log10,
    theme,
    theme_minimal,
)

import warnings

warnings.filterwarnings("ignore")

# set up all paths
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "processed" / "master.csv"
REPORTS_DIR = REPO_ROOT / "reports" / "eda"
IMAGES_DIR = REPORTS_DIR / "images"
REPORT_PATH = REPORTS_DIR / "report.md"

# features we want to build distributions for
PLOT_FEATURES = ["total_ghg", "ghg_per_capita", "gdp", "gdp_per_capita", "population"]

# predefined pairs to build scatterplots for
SCATTER_PAIRS = [
    ("gdp", "total_ghg"),
    ("gdp_per_capita", "ghg_per_capita"),
    ("population", "total_ghg"),
    ("land_area", "total_ghg"),
]

# features to build a log axis for the x axis or for the y axis for
LOG_X_FEATURES = {"total_ghg", "gdp", "gdp_per_capita", "population"}
LOG_Y_TIMESERIES = {"total_ghg", "gdp", "population"}

# define default image sizes/attributes for output
DPI = 150
WIDTH = 10
HEIGHT = 6


# Formats log axis as 10^n instead of raw 10000000 (standard) and outputs them as a list
def fmt_log(breaks):
    out = []
    for b in breaks:
        if b is None or b <= 0:
            out.append("")  # keeps logical track of breks
            continue
        exp = int(round(math.log10(b)))  # does the actual formatting
        out.append(f"$10^{{{exp}}}$")
    return out


# Saves a plot to the image output directory and returns the path (used for future report writing)
def save_plot(plot, name, height=HEIGHT):
    path = IMAGES_DIR / f"{name}.png"  # define save path for image
    plot.save(
        path, dpi=DPI, width=WIDTH, height=height, verbose=False
    )  # actually save the plot
    return path


# Returns aggregate/summary statistics for a df and a given set of features
# Returns those summary statistics as a df
def summary_stats(df, features):
    stats = (
        df[features].agg(["mean", "median", "std"]).T
    )  # transpose so columns are summary stats not rows
    stats.columns = [
        "mean",
        "median",
        "std_dev",
    ]  # rename columns for future report writing (mainly std -> std_dev)
    return stats


# Makes density distributions for a df, specific to the predefined features to build plots for
def make_distributions(df):
    out = []
    for feat in PLOT_FEATURES:
        plot_df = df.dropna(
            subset=[feat]
        )  # drops na before plotting (doesn't persist, clone made via plot_df)
        if feat in LOG_X_FEATURES:
            plot_df = plot_df[
                plot_df[feat] > 0
            ]  # log can't handle 0/negatives, filter them out
        plot = (
            ggplot(plot_df, aes(x=feat))
            + geom_density(fill="#4682b4", alpha=0.5)
            + labs(title=f"Distribution of {feat}", x=feat, y="density")
            + theme_minimal()
        )
        if feat in LOG_X_FEATURES:
            plot = plot + scale_x_log10(
                labels=fmt_log
            )  # apply log scale + formatted ticks
        out.append((feat, save_plot(plot, f"dist_{feat}")))
    return out


# Makes time series plots for each feature, one line per country, colored by continent
def make_timeseries(df):
    out = []
    for feat in PLOT_FEATURES:
        plot_df = df.dropna(subset=[feat])
        if feat in LOG_Y_TIMESERIES:
            plot_df = plot_df[
                plot_df[feat] > 0
            ]  # same log filter as distributions, just for the y axis
        plot = (
            ggplot(plot_df, aes(x="year", y=feat, group="iso_code", color="continent"))
            + geom_line(alpha=0.4, size=0.4)
            + facet_wrap(
                "continent", scales="free_y"
            )  # one panel per continent, free y so small continents read
            + labs(title=f"{feat} over time, by continent", x="year", y=feat)
            + theme_minimal()
            + theme(
                legend_position="none"
            )  # no legend since color matches facet divisions from facet wrap
        )
        if feat in LOG_Y_TIMESERIES:
            plot = plot + scale_y_log10(labels=fmt_log)
        out.append((feat, save_plot(plot, f"timeseries_{feat}", height=7)))
    return out


# Makes scatter plots for the predefined (above) feature pairs, colored by continent
def make_scatters(df):
    out = []
    for x_feat, y_feat in SCATTER_PAIRS:
        plot_df = df.dropna(subset=[x_feat, y_feat])
        plot_df = plot_df[
            (plot_df[x_feat] > 0) & (plot_df[y_feat] > 0)
        ]  # both axes are log so filter both
        plot = (
            ggplot(plot_df, aes(x=x_feat, y=y_feat, color="continent"))
            + geom_point(alpha=0.3, size=1)
            + scale_x_log10(labels=fmt_log)
            + scale_y_log10(labels=fmt_log)
            + labs(title=f"{x_feat} vs {y_feat}", x=x_feat, y=y_feat)
            + theme_minimal()
        )
        name = f"scatter_{x_feat}_vs_{y_feat}"
        out.append(((x_feat, y_feat), save_plot(plot, name)))
    return out


# Makes a bar chart of country count per continent
# Returns the path and the counts (counts is used in the report table)
def make_continent_distribution(df):
    counts = (
        df.drop_duplicates("iso_code")["continent"].value_counts().sort_values()
    )  # drop duplicates so we count countries not country-years
    plot_df = counts.reset_index()
    plot_df.columns = ["continent", "count"]
    plot_df["continent"] = (
        pd.Categorical(  # set ordered category so the bars stay sorted in the plot
            plot_df["continent"], categories=plot_df["continent"], ordered=True
        )
    )
    plot = (
        ggplot(plot_df, aes(x="continent", y="count", fill="continent"))
        + geom_col()
        + coord_flip()  # horizontal bars are easier to read with long continent labels (purely visual choice)
        + labs(title="Countries per continent", x="continent", y="count")
        + theme_minimal()
        + theme(legend_position="none")
    )
    return save_plot(plot, "continent_distribution", height=4), counts


# Makes a bar chart of NaN counts per column (only columns that actually have NaN)
def make_nan_counts(df):
    nan_counts = df.isna().sum()
    nan_df = (
        nan_counts[nan_counts > 0].sort_values().reset_index()
    )  # filter to columns that actually have NaN
    nan_df.columns = ["column", "nan_count"]
    nan_df["column"] = pd.Categorical(
        nan_df["column"], categories=nan_df["column"], ordered=True
    )
    plot = (
        ggplot(nan_df, aes(x="column", y="nan_count"))
        + geom_col(fill="#cc6677")
        + coord_flip()
        + labs(title="NaN counts per column", x="column", y="NaN count")
        + theme_minimal()
    )
    return save_plot(plot, "nan_counts", height=4)


# Builds the EDA report markdown using the chart paths and the summary stats table
def write_report(
    dist_charts,
    ts_charts,
    scatter_charts,
    continent_chart,
    continent_counts,
    nan_counts_chart,
    stats,
):
    lines = ["# EDA Report", "", "Generated from `data/processed/master.csv`.", ""]

    # summary stats table at top of distributions section
    lines += ["## Smoothed Distributions", ""]
    lines += ["Each row is one (country, year) entry.", ""]
    lines += ["| Feature | Mean | Median | Std Dev |", "| --- | --- | --- | --- |"]
    for feat in PLOT_FEATURES:
        row = stats.loc[feat]
        lines.append(
            f"| `{feat}` | {row['mean']:,.2f} | {row['median']:,.2f} | {row['std_dev']:,.2f} |"
        )
    lines.append("")
    for feat, path in dist_charts:
        rel = path.relative_to(
            REPORTS_DIR
        )  # paths come in absolute, use relative for the report to keep things portable
        lines += [f"### {feat}", "", f"![{feat} distribution]({rel})", ""]

    lines += ["## Time Series", ""]
    for feat, path in ts_charts:
        rel = path.relative_to(REPORTS_DIR)
        lines += [f"### {feat}", "", f"![{feat} over time]({rel})", ""]

    lines += ["## Bivariate Views", ""]  # bivaraite scatters
    for (x_feat, y_feat), path in scatter_charts:
        rel = path.relative_to(REPORTS_DIR)
        lines += [
            f"### {x_feat} vs {y_feat}",
            "",
            f"![{x_feat} vs {y_feat}]({rel})",
            "",
        ]

    lines += ["## Continent Distribution", ""]
    rel = continent_chart.relative_to(REPORTS_DIR)
    lines += [f"![continent distribution]({rel})", ""]
    lines += ["| Continent | Countries |", "| --- | --- |"]
    for cont, n in continent_counts.sort_values(ascending=False).items():
        lines.append(f"| {cont} | {n} |")
    lines.append("")

    lines += ["## NaN Counts", ""]
    rel = nan_counts_chart.relative_to(REPORTS_DIR)
    lines += [f"![NaN counts per column]({rel})", ""]

    REPORT_PATH.write_text("\n".join(lines))


# Runs the full EDA pipeline (stats + all the chart types) and writes the report
def main():
    print(f"Loading {INPUT_PATH.relative_to(REPO_ROOT)}")
    df = pd.read_csv(INPUT_PATH)
    print(f"    {len(df):,} rows x {df.shape[1]} columns")

    IMAGES_DIR.mkdir(
        parents=True, exist_ok=True
    )  # create the output dir if it doesn't exist

    print("Computing summary statistics")
    stats = summary_stats(df, PLOT_FEATURES)

    print("Building distributions")
    dist_charts = make_distributions(df)

    print("Building time series")
    ts_charts = make_timeseries(df)

    print("Building bivariate scatters")
    scatter_charts = make_scatters(df)

    print("Building continent distribution")
    continent_chart, continent_counts = make_continent_distribution(df)

    print("Building NaN counts")
    nan_counts_chart = make_nan_counts(df)

    print("Writing report")
    write_report(
        dist_charts,
        ts_charts,
        scatter_charts,
        continent_chart,
        continent_counts,
        nan_counts_chart,
        stats,
    )

    # add 2 for continent + nan charts; not lists
    n_images = len(dist_charts) + len(ts_charts) + len(scatter_charts) + 2
    print(f"\nWrote {n_images} images to {IMAGES_DIR.relative_to(REPO_ROOT)}")
    print(f"Wrote report to {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
