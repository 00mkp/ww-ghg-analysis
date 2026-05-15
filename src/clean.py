from pathlib import Path

import pandas as pd

# set up paths
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "processed" / "master.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "master_clean.csv"

# columns where NaN means we drop the row (land_area handled separately via country mean)
DROP_NAN_COLUMNS = [
    "total_ghg",
    "ghg_per_capita",
    "gdp",
    "gdp_per_capita",
    "population",
]


# Reads master csv in and returns it as a dataframe
def load_master(path):
    return pd.read_csv(path)


# Fills NaN in land_area with the country's mean across the years we do have
# Land area barely changes year to year, so the mean is a fine stand in
def fill_land_area(df):
    df = df.copy()
    country_mean = df.groupby("iso_code")["land_area"].transform(
        "mean"
    )  # per country mean back to each row (custom imputation)
    df["land_area"] = df["land_area"].fillna(country_mean)
    return df


# Drops rows that have NaN in any of tracked features (everything but land_area)
def drop_nan_rows(df):
    return df.dropna(subset=DROP_NAN_COLUMNS).reset_index(drop=True)


# Runs cleaning pipeline, fill land_area NaNs, drop other NaN rows, write to disk
def main():
    print(f"Loading {INPUT_PATH.relative_to(REPO_ROOT)}")
    df = load_master(INPUT_PATH)
    n_in = len(df)  # save initial row count for the dropped count later
    print(f"    {n_in:,} rows")

    # print NaN counts before cleaning to see what we are dealing with
    print("\nNaN counts before:")
    for col, n in df.isna().sum().items():
        if n > 0:
            print(f"    {col}: {n:,}")

    print("\nFilling land_area with country mean")
    df = fill_land_area(df)

    print("Dropping rows with NaN in tracked features")
    df = drop_nan_rows(df)
    n_out = len(df)

    # confirm no NaN remains in tracked columns after cleaning
    print("\nNaN counts after:")
    after = df.isna().sum()
    leftover = {col: n for col, n in after.items() if n > 0}
    if leftover:
        for col, n in leftover.items():
            print(f"    {col}: {n:,}")
    else:
        print("    none")

    pct = (n_in - n_out) / n_in  # percent of rows lost to dropping
    print(f"\nRows: {n_in:,} -> {n_out:,} ({n_in - n_out:,} dropped, {pct:.1%})")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
