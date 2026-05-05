# Data

Five raw CSVs and a `iso3_to_continent.csv` lookup. The merged country year master dataset is built by `src/build_master.py` and written to `data/processed/master.csv` (gitignored - rebuild from raw with `python src/build_master.py` during full pipeline runs).

## `ghg_emissions.csv`

- **Description:** Total and per capita greenhouse gas emissions by country, one row per country year.
- **Source:** [Kaggle - Global Greenhouse Gas Emissions by Country, 1950-2024](https://www.kaggle.com/datasets/lucalullo/global-ghg-gas-emissions-by-country-1950-2024), originally from Our World in Data / Climate Watch.
- **Shape:** 14,925 rows x 5 features
- **Format:** long
- **Year range:** 1950-2024
- **Countries:** 199

### Features

| Feature | Type | Description |
| --- | --- | --- |
| `country` | string | Country name |
| `year` | int | Observation year |
| `iso_code` | string | Three letter ISO3 country code |
| `total_ghg` | float | Total emissions, million tonnes CO_2-eq |
| `ghg_per_capita` | float | Per person emissions, tonnes CO_2-eq |

---

## `wb_gdp.csv`

- **Description:** Gross Domestic Product in current US dollars, by country and year.
- **Source:** [World Bank Open Data](https://data.worldbank.org/indicator/NY.GDP.MKTP.CD) - indicator `NY.GDP.MKTP.CD` (World Development Indicators).
- **Shape:** 266 country rows x 70 features
- **Format:** wide (one column per year)
- **Year range:** 1960-2024

### Features

| Feature | Type | Description |
| --- | --- | --- |
| `Country Name` | string | Country name |
| `Country Code` | string | ISO3 country code |
| `Indicator Name` | string | Constant: `GDP (current US$)` |
| `Indicator Code` | string | Constant: `NY.GDP.MKTP.CD` |
| `1960` ... `2024` | float | GDP for that year, current US$ |

---

## `wb_gdp_per_capita.csv`

- **Description:** GDP per capita in current US dollars, by country and year.
- **Source:** [World Bank Open Data](https://data.worldbank.org/indicator/NY.GDP.PCAP.CD) - indicator `NY.GDP.PCAP.CD`.
- **Shape:** 266 country rows x 70 features
- **Format:** wide
- **Year range:** 1960-2024

### Features

| Feature | Type | Description |
| --- | --- | --- |
| `Country Name` | string | Country name |
| `Country Code` | string | ISO3 country code |
| `Indicator Name` | string | Constant: `GDP per capita (current US$)` |
| `Indicator Code` | string | Constant: `NY.GDP.PCAP.CD` |
| `1960` ... `2024` | float | GDP per capita for that year, current US$ |

---

## `wb_land_area.csv`

- **Description:** Total land area in square kilometers, by country and year.
- **Source:** [World Bank Open Data](https://data.worldbank.org/indicator/AG.LND.TOTL.K2) - indicator `AG.LND.TOTL.K2`.
- **Shape:** 266 country rows x 70 features
- **Format:** wide
- **Year range:** 1960-2024

### Features

| Feature | Type | Description |
| --- | --- | --- |
| `Country Name` | string | Country name |
| `Country Code` | string | ISO3 country code |
| `Indicator Name` | string | Constant: `Land area (sq. km)` |
| `Indicator Code` | string | Constant: `AG.LND.TOTL.K2` |
| `1960` ... `2024` | float | Land area for that year, km^2 |

### Notes

The source export has columns for 1960 and 2023-2024 but every country's value is empty in those columns; the WB didn't publish land area data for those years. Those gaps cotninue through the merged dataset as `NaN` for `land_area`. Later options if a model needs full coverage: forward/back fill within country (land area is effectively static), drop affected rows, or treat `land_area` as a constant (single value per country).

---

## `wb_population.csv`

- **Description:** Total population, by country and year.
- **Source:** [World Bank Open Data](https://data.worldbank.org/indicator/SP.POP.TOTL) - indicator `SP.POP.TOTL`.
- **Shape:** 266 country rows x 70 features
- **Format:** wide
- **Year range:** 1960-2024

### Features

| Feature | Type | Description |
| --- | --- | --- |
| `Country Name` | string | Country name |
| `Country Code` | string | ISO3 country code |
| `Indicator Name` | string | Constant: `Population, total` |
| `Indicator Code` | string | Constant: `SP.POP.TOTL` |
| `1960` ... `2024` | float | Population count for that year |

---

## `iso3_to_continent.csv`

- **Description:** Static lookup mapping each ISO3 country code in the GHG dataset to one of six continents.
- **Source:** Manually curated using the [UN (M49)](https://unstats.un.org/unsd/methodology/m49/) standard.
- **Shape:** 199 rows x 2 features
- **Format:** long
- **Continent values:** `Africa`, `Asia`, `Europe`, `North America`, `South America`, `Oceania`

### Features

| Feature | Type | Description |
| --- | --- | --- |
| `iso_code` | string | Three letter ISO3 country code (join key) |
| `continent` | string | One of the six continent values above |

### Boundary cases

A few countries are in two continents; the assignments below follow the UN Statistics Division:

| ISO3 | Country | Assignment | Reason |
| --- | --- | --- | --- |
| RUS | Russia | Europe | UN convention; capital and most population west of Urals |
| TUR | Turkey | Asia | Most landmass in Anatolia |
| CYP | Cyprus | Europe | EU member, politically European |
| ARM | Armenia | Asia | South Caucasus |
| AZE | Azerbaijan | Asia | South Caucasus |
| GEO | Georgia | Asia | South Caucasus |
| EGY | Egypt | Africa | Despite Sinai, classified as African |
| ISR | Israel | Asia | Western Asia |
| TLS | East Timor | Asia | UN South East Asia subregion |

---

## `processed/master.csv`

- **Description:** Country year panel built by joining the GHG file with the four World Bank supplements and the continent lookup on `iso_code` (and `year` for the WB joins). Gitignored; rebuild from raw with `python src/build_master.py`.
- **Source:** Generated by `src/build_master.py` from the files above.
- **Shape:** 12,805 rows x 10 features
- **Format:** long
- **Year range:** 1960-2024
- **Countries:** 197

### Features

| Feature | Type | Description |
| --- | --- | --- |
| `country` | string | Country name |
| `year` | int | Observation year |
| `iso_code` | string | Three letter ISO3 country code |
| `continent` | string | One of `Africa`, `Asia`, `Europe`, `North America`, `South America`, `Oceania` |
| `total_ghg` | float | Total emissions, million tonnes CO_2-eq |
| `ghg_per_capita` | float | Per person emissions, tonnes CO_2-eq |
| `gdp` | float | GDP in current US dollars |
| `gdp_per_capita` | float | GDP per capita in current US dollars |
| `land_area` | float | Land area, km^2 |
| `population` | float | Total population |

---

## `processed/master_clean.csv`

- **Description:** Cleaned version of `master.csv`. `land_area` `NaN`s are filled with each country's mean across observed years (land area is basically cosntant per country); rows with `NaN` in any other tracked feature (`total_ghg`, `ghg_per_capita`, `gdp`, `gdp_per_capita`, `population`) are dropped. Same schema as `master.csv`, less rows. Used by the regression, classification, and clustering pipelines. Gitignored; rebuild with `python src/clean.py` (requires `master.csv`).
- **Source:** Generated by `src/clean.py` from `processed/master.csv`.
- **Shape:** 10,746 rows x 10 features
- **Format:** long
- **Year range:** 1960-2024
- **Countries:** 191

### Features

Identical schema to `processed/master.csv`.
