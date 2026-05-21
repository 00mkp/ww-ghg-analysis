from pathlib import Path

import pandas as pd

import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]  # resolves to the root of the repo
DATA_DIR = REPO_ROOT / "data"  # resolves to dir where data lives
OUTPUT_DIR = DATA_DIR / "processed"  # where processed stuff goes (master dataset)
OUTPUT_PATH = OUTPUT_DIR / "master.csv"  # file to output master dataset to

# where all of the data can be found, using constants for simplicity
GHG_PATH = DATA_DIR / "ghg_emissions.csv"
WB_GDP_PATH = DATA_DIR / "wb_gdp.csv"
WB_GDP_PC_PATH = DATA_DIR / "wb_gdp_per_capita.csv"
WB_LAND_PATH = DATA_DIR / "wb_land_area.csv"
WB_POP_PATH = DATA_DIR / "wb_population.csv"
CONTINENT_PATH = DATA_DIR / "iso3_to_continent.csv"

# define range for data
START_YEAR = 1960
END_YEAR = 2024
WB_HEADER_SKIP = 4  # World Bank CSVs have 4 metadata rows before the column header

# final columns data will have
FINAL_COLUMNS = [
    "country",
    "year",
    "iso_code",
    "continent",
    "total_ghg",
    "ghg_per_capita",
    "gdp",
    "gdp_per_capita",
    "land_area",
    "population",
]


# Reads main GHG dataset in and returns it as a dataframe
def load_ghg(path):
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)  # convert to int for type logic

    return df


# Reads in World Bank datasets, skips the 4 row metadata header
# Returns wide to long converted World Bank datasets
def load_wb(path, value_name):
    df = pd.read_csv(path, skiprows=WB_HEADER_SKIP)  # skip first 4 rows
    year_cols = [
        c for c in df.columns if str(c).isdigit()
    ]  # get all columns to see years

    long = df.melt(  # convert from wide to long format
        id_vars=["Country Code"],  # identifier column for every row
        value_vars=year_cols,  # which columns to convert into rows
        var_name="year",  # name of new column to hold converted
        value_name=value_name,  # name of the nuw column
    )

    long = long.rename(
        columns={"Country Code": "iso_code"}
    )  # rename to match the rest of the master dataset (snake case)
    long["year"] = long["year"].astype(int)  # convert years from string into int
    long = long[
        (long["year"] >= START_YEAR) & (long["year"] <= END_YEAR)
    ]  # filter for range

    return long.reset_index(drop=True)


# Loads in continent lookup table for matchin, returns as a dataframe
def load_continent(path):
    return pd.read_csv(path)


# Builds master dataset from GHG core dataset and loaded WB dataset
# Takes in GHG dataset, dictionary of World Bank datasets, and continent lookup table
# Returns master dataset with the above specified columnd (FINAL_COLUMNS)
def build_master(ghg, wb, continent):
    df = ghg[
        (ghg["year"] >= START_YEAR) & (ghg["year"] <= END_YEAR)
    ].copy()  # filter ghg dataset for range
    df = df.merge(
        continent, on="iso_code", how="left"
    )  # left join with continent lookup (NaN values handled later)

    for (
        frame
    ) in wb.values():  # inner join all World Bank datasets so only data we have is kept
        df = df.merge(frame, on=["iso_code", "year"], how="inner")

    unmapped = df.loc[
        df["continent"].isna(), "iso_code"
    ].unique()  # check how many countries unmapped to continent

    if len(unmapped):  # raise error if any unmapped
        raise RuntimeError(f"missing continent for: {sorted(unmapped)}")

    return df[FINAL_COLUMNS].sort_values(["country", "year"]).reset_index(drop=True)


# Runs all above functions to full build master dataset
def main():
    print(f"Loading GHG emissions from {GHG_PATH.name}")
    ghg = load_ghg(GHG_PATH)  # load in GHG
    print(f"    {len(ghg):,} rows")  # see shape

    # build dictionary of all World Bank datasets
    wb = {}
    for path, name in [
        (WB_GDP_PATH, "gdp"),
        (WB_GDP_PC_PATH, "gdp_per_capita"),
        (WB_LAND_PATH, "land_area"),
        (WB_POP_PATH, "population"),
    ]:
        print(f"Loading {name} from {path.name}")
        wb[name] = load_wb(path, name)
        print(f"    {len(wb[name]):,} rows after melt and year filter")

    # load continent lookup table
    print(f"Loading continents from {CONTINENT_PATH.name}")
    continent = load_continent(CONTINENT_PATH)
    print(f"    {len(continent):,} rows")  # see shape

    print("\nBuilding master dataset")
    master = build_master(ghg, wb, continent)  # build master

    # check shape of master
    print(f"\nShape: {master.shape}")
    print(f"Year range: {master['year'].min()}-{master['year'].max()}")
    print(f"Countries: {master['iso_code'].nunique()}")

    # check for NaN
    nan_summary = master.isna().sum()

    if nan_summary.any():
        print("\nNaN counts:")

        for col, n in nan_summary.items():
            if n > 0:
                print(f"  {col}: {n:,}")

    # fatal error on building
    if len(master) == 0:
        raise RuntimeError("master dataset is empty")

    OUTPUT_DIR.mkdir(exist_ok=True)  # create output dir
    master.to_csv(OUTPUT_PATH, index=False)  # write df to csv for persistence
    print(f"\nWrote {len(master):,} rows to {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
