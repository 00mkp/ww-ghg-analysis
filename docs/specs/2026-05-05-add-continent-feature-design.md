# Design - Add Continent Column to Master Dataset

**Date:** 2026-05-05

## Goal

Add a `continent` column to `data/processed/master.csv` so it can serve as a categorical feature for regression and as a
target for later classification work. Continent is derived from `iso_code` using a lookup residing in `data/`.

## Tasks

- Create a `data/iso3_to_continent.csv` mapping every ISO3 code in the master dataset to a continent.
- Modify `src/build_master.py` to load and join created lookup.
- Add `continent` to the master schema as a string.
- Document the new file in `data/README.md`.

## Continent Values

Six values, following UN convention with the Americas split:

- `Africa`
- `Asia`
- `Europe`
- `North America`
- `South America`
- `Oceania`

No `Antarctica` since no countries fall in it.

Boundary cases (assignments that vary depending on source):

| ISO3 | Country | Assignment | Reason |
| --- | --- | --- | --- |
| RUS | Russia | Europe | UN convention; capital and most population west of Ural mountains |
| TUR | Turkey | Asia | Most landmass in Anatolia |
| CYP | Cyprus | Europe | EU member, politically European |
| ARM | Armenia | Asia | South Caucasus |
| AZE | Azerbaijan | Asia | South Caucasus |
| GEO | Georgia | Asia | South Caucasus |
| EGY | Egypt | Africa | Classified as African |
| ISR | Israel | Asia | Western Asia |
| TLS | East Timor | Asia | UN South East Asia subregion |

This is in line with the UN Statistics Division format (M49 Standard). Documented in `data/README.md`.

## New Schema

The `master.csv` schema gets one new column:

| Column | Type | Source |
| --- | --- | --- |
| `country` | string | `ghg_emissions.csv` |
| `year` | int | `ghg_emissions.csv` |
| `iso_code` | string | `ghg_emissions.csv` |
| `continent` | string | `iso3_to_continent.csv` **(new)** |
| `total_ghg` | float | `ghg_emissions.csv` |
| `ghg_per_capita` | float | `ghg_emissions.csv` |
| `gdp` | float | `wb_gdp.csv` |
| `gdp_per_capita` | float | `wb_gdp_per_capita.csv` |
| `land_area` | float | `wb_land_area.csv` |
| `population` | float | `wb_population.csv` |

## Lookup File

`data/iso3_to_continent.csv` has two columns:

| Column | Type | Description |
| --- | --- | --- |
| `iso_code` | string | Three letter ISO3 country code |
| `continent` | string | One of the six values above |

Coverage goal: every ISO3 in `ghg_emissions.csv` (199 countries). Aggregate codes (such as `WLD`, `EUU`) are not
mapped, as the existing inner join already drops those, so the lookup only needs individual country entries.

## Architecture

Changes to `src/build_master.py`:

1. Add `CONTINENT_PATH = DATA_DIR / "iso3_to_continent.csv"` to the constants block to support mapping.
2. Add a `load_continent(path) -> DataFrame` function returning a dataframe with `[iso_code, continent]`.
3. In `build_master`, left join continent onto the GHG frame before the WB inner joins (so the WB inner joins can still get rid of rows lacking WB features).
4. After the joins, verify `continent` has no `NaN`; raise `RuntimeError` listing unmapped codes if it `NaN`s exist.
5. Add `continent` to `FINAL_COLUMNS` between `iso_code` and `total_ghg`.

## Repo Changes

Files added:

- `data/iso3_to_continent.csv`: the lookup.

Files changed:

- `src/build_master.py`: load and join continent, schema update, integrity check.
- `data/README.md`: new section documenting `iso3_to_continent.csv`, including the convention used and boundary case decisions (as above). Header note updated to reflect five raw CSVs plus
the lookup.
- `README.md`: data table gains an `iso3_to_continent.csv` row; Workflow step 1 updated to reference the continent join.
