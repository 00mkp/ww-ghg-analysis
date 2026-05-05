# Design - Master Dataset Build Script

**Date:** 2026-05-05
**Updated:** 2026-05-05 (continent column added; see [add-continent-feature-design.md](2026-05-05-add-continent-feature-design.md))

## Goal

Provide a single, reproducible Python script that joins the five raw CSVs in `data/` and the `iso3_to_continent.csv` lookup into one country/year master dataset for later regression, classification, and clustering, as per the project proposal. The script is added to the repo; the output CSV is gitignored and rebuilt each time as part of the full pipeline.

## Tasks

- Load the GHG emissions CSV (long format).
- Load each of the four World Bank CSVs (start in wide format; has 4 row metadata frontmatter at top of each file).
- Load the `iso3_to_continent.csv` lookup.
- Inner join the GHG and WB files on `(iso_code, year)`; left join the continent lookup on `iso_code` and check for any missing values.
- Restrict year range to **1960-2024** (to strip missing World Bank values).
- Preserve `NaN` for any missing values within the joined range (will be dealt with later).
- Return resulting dataset as `data/processed/master.csv`.
- Update main `README.md` and `data/README.md` to reference the new script and output.

## Output Schema

One row per country year; year range 1960-2024.

| Column | Type | Source |
| --- | --- | --- |
| `country` | string | `ghg_emissions.csv` |
| `year` | int | `ghg_emissions.csv` |
| `iso_code` | string | `ghg_emissions.csv` |
| `continent` | string | `iso3_to_continent.csv` |
| `total_ghg` | float | `ghg_emissions.csv` |
| `ghg_per_capita` | float | `ghg_emissions.csv` |
| `gdp` | float | `wb_gdp.csv` (current US$) |
| `gdp_per_capita` | float | `wb_gdp_per_capita.csv` (current US$) |
| `land_area` | float | `wb_land_area.csv` (km^2) |
| `population` | float | `wb_population.csv` (count) |

## Join Logic

- **Join types:** inner join across the GHG and four WB files on `(iso_code, year)`; left join with the continent lookup on `iso_code` followed by a `NaN` check that raises an exception if
any ISO3 code is left unmapped.
- **Year filter:** rows outside 1960-2024 are dropped.
- **Aggregate filtering:** WB aggregate rows ("World", "European Union", any regional groupings, income brackets) have no matching ISO code in GHG, so the inner join drops them.
- **Country mismatches:** countries in GHG but not in WB (or vice versa) are dropped by the inner join, which is fine, as rows without WB features can't contribute to the model anyway.
- **In range `NaN` values are left untouched:** For example, `land_area` is empty in WB for 1960, so 1960 rows will have `land_area = NaN`. Country specific missing values (i.e., GDP missing for a small territory in a given year) are also preserved as `NaN`. All cleanup is to be dealt with later.

## Architecture

Single file script: `src/build_master.py`. Just a file with a bunch of functions to be used in `main()`.

**Functions:**

1. `load_ghg(path) -> DataFrame`: read CSV directly.
2. `load_wb(path, value_name) -> DataFrame`: skip 4 row metadata frontmatter, convert from wide to long with `iso_code` and `year` as keys (using `df.melt()`), rename value column to `value_name` (label of new column), filter year range.
3. `load_continent(path) -> DataFrame`: read the lookup CSV.
4. `build_master(ghg, wb_frames, continent) -> DataFrame`: left join continent on `iso_code`, inner join with each WB frame on `(iso_code, year)`, raise if any continent is unmapped, return final ordered DataFrame.
5. `main()`: call functions, write output file, print progress lines.

**Logging:** basic `print()` calls. Print row count after each load and after each merge step, as well as the final shape and a one line summary of any column with `NaN` counts. Helps make sure the join worked while running.

**Error handling:** use default exceptions. If a file is missing, the script exits with `FileNotFoundError`. If any joined `iso_code` doesn't get a continent assignment, raise a `RuntimeError` showing the unmapped codes. If the final result is empty, raise a `RuntimeError`.

## Repo Changes

Files added:

- `src/build_master.py`: the script.
- `data/iso3_to_continent.csv`: the continent lookup.
- `data/processed/.gitkeep`: keeps the directory in git so it exists after clone and repo structure is visible.

Files modified:

- `.gitignore`: add `data/processed/*` and `!data/processed/.gitkeep` so the directory stays tracked but contents are ignored.
- `README.md`: Workflow section step 1 references the new script and the continent join; Repository Layout updated to show `src/build_master.py` and `data/processed/`; data table includes the continent lookup.
- `data/README.md`: short note at the top pointing to the `build-master script`, and a section documenting `iso3_to_continent.csv`.

## Reproducibility

Anyone with the repo cloned runs:

```bash
python src/build_master.py
```
and gets `data/processed/master.csv`.
