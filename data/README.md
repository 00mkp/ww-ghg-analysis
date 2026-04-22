# Data

## `ghg_emissions.csv`

- **Description:** Total and per-capita greenhouse gas emissions by country, one row per country-year.
- **Source:** [Kaggle — Global Greenhouse Gas Emissions by Country, 1950–2024](https://www.kaggle.com/), originally from Our World in Data / Climate Watch.
- **Shape:** 14,924 rows × 5 features
- **Format:** long
- **Year range:** 1950–2024
- **Countries:** 199

### Features

| Feature | Type | Description |
| --- | --- | --- |
| `country` | string | Country name |
| `year` | int | Observation year |
| `iso_code` | string | Three-letter ISO3 country code |
| `total_ghg` | float | Total emissions, million tonnes CO₂-eq |
| `ghg_per_capita` | float | Per-person emissions, tonnes CO₂-eq |

---

## `wb_gdp.csv`

- **Description:** Gross Domestic Product in current US dollars, by country and year.
- **Source:** World Bank Open Data — indicator `NY.GDP.MKTP.CD` (World Development Indicators).
- **Shape:** 266 country rows × 70 features
- **Format:** wide (one column per year)
- **Year range:** 1960–2024

### Features

| Feature | Type | Description |
| --- | --- | --- |
| `Country Name` | string | Country name |
| `Country Code` | string | ISO3 country code |
| `Indicator Name` | string | Always `GDP (current US$)` |
| `Indicator Code` | string | Always `NY.GDP.MKTP.CD` |
| `1960` … `2025` | float | GDP for that year, current US$ |

---

## `wb_gdp_per_capita.csv`

- **Description:** GDP per capita in current US dollars, by country and year.
- **Source:** World Bank Open Data — indicator `NY.GDP.PCAP.CD`.
- **Shape:** 266 country rows × 70 features
- **Format:** wide
- **Year range:** 1960–2024

### Features

| Feature | Type | Description |
| --- | --- | --- |
| `Country Name` | string | Country name |
| `Country Code` | string | ISO3 country code |
| `Indicator Name` | string | Always `GDP per capita (current US$)` |
| `Indicator Code` | string | Always `NY.GDP.PCAP.CD` |
| `1960` … `2025` | float | GDP per capita for that year, current US$ |

---

## `wb_land_area.csv`

- **Description:** Total land area in square kilometers, by country and year.
- **Source:** World Bank Open Data — indicator `AG.LND.TOTL.K2`.
- **Shape:** 266 country rows × 70 features
- **Format:** wide
- **Year range:** 1961–2022

### Features

| Feature | Type | Description |
| --- | --- | --- |
| `Country Name` | string | Country name |
| `Country Code` | string | ISO3 country code |
| `Indicator Name` | string | Always `Land area (sq. km)` |
| `Indicator Code` | string | Always `AG.LND.TOTL.K2` |
| `1960` … `2025` | float | Land area for that year, km² |

---

## `wb_population.csv`

- **Description:** Total population, by country and year.
- **Source:** World Bank Open Data — indicator `SP.POP.TOTL`.
- **Shape:** 266 country rows × 70 features
- **Format:** wide
- **Year range:** 1960–2024

### Features

| Feature | Type | Description |
| --- | --- | --- |
| `Country Name` | string | Country name |
| `Country Code` | string | ISO3 country code |
| `Indicator Name` | string | Always `Population, total` |
| `Indicator Code` | string | Always `SP.POP.TOTL` |
| `1960` … `2025` | float | Population count for that year |
