# Design - Clean Master Dataset

**Date:** 2026-05-14

## Goal

Produce a cleaned version of `data/processed/master.csv` usable for regression, classification, and clustering by handling `NaN` values per the rules below. Same long format dataset, with fewer rows.

## NaN Handling

| Column | Rule | Why |
| --- | --- | --- |
| `land_area` | Fill with country mean across in range years | Basically constant per country; 1960 and 2023-2024 are empty everywhere in WB. |
| `total_ghg` | Drop row | Target for regression; can't train on `NaN` target. |
| `ghg_per_capita` | Drop row | Same as above. |
| `gdp` | Drop row | Highly volatile, changes with time; country mean is a bad estimator. |
| `gdp_per_capita` | Drop row | Same as above. |
| `population` | Drop row | Few `NaN` |
| `continent` | n/a | No `NaN` by design. |
| `country`, `year`, `iso_code` | n/a | No `NaN`. |

## Architecture

New script: `src/clean.py`. Functions:

1. `load_master(path)`: read `master.csv`.
2. `fill_land_area(df)`: groupby `iso_code` and replace `NaN` in `land_area` with that country's mean.
3. `drop_nan_rows(df)`: drop rows with `NaN` in any of the other tracked features.
4. `main()`: call functions, write output, print row counts and `NaN` counts before/after.

## Output

`data/processed/master_clean.csv`, gitignored alongside `master.csv`.
