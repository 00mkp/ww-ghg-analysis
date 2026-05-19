# CPSC 393 - Final Project: Global GHG Emissions

| **Members** | | |
| --- | --- | --- |
| Maksim Popov | Daniel Franks | Andrew Blanco |

## Description

Final project for **CPSC 393 - Machine Learning** at Chapman University (Spring 2026, Dr. Rajeev Joshi).

We're analyzing country level greenhouse gas (GHG) emissions across ~80 years to understand how developmental indicators (population, land area, GDP, GDP per capita) relate to total and per capita emissions. The goal is:

1. **Regression**: predict a country's GHG emissions from its developmental features, comparing models from the course.
2. **Classification**: predict a country's continent from its developmental and emissions features, with continent treated as a categorical target sourced from `iso3_to_continent.csv`.
3. **Clustering**: see whether grouping countries by these characteristics produces intuitive emissions tiers (clustering methods pulled from CPSC 392 since they aren't covered in depth in 393).

The motivation: give developing countries a view of their projected emissions trajectory so they can plan mitigation early, and help global reduction efforts identify which country profiles
drive the most emissions.

## Reproducibility and Code Design

All randomness is seeded with `random_state=42`, Python is pinned to 3.12 via `.python-version`, and dependencies are pinned in `requirements.txt`, so running `python src/run_all.py` from a
clean clone regenerates all artifacts (master CSVs and pipeline markdown reports). Each script in `src/` is built as a standalone modular file with helper functions and a single `main()`
runner, so individual phases can be re run in isolation without infringing on the others. Code style is enforced by `ruff` formatting and linting via pre-commit hooks, and the same checks
run on every push and pull request by [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Data

Data lives in `data/`. The core dataset is the Kaggle **Global Greenhouse Gas Emissions by Country, 1950-2024** (originally Our World in Data / Climate Watch). It has 14,925 rows and 5
columns covering 199 countries: `country`, `year`, `iso_code`, `total_ghg`, `ghg_per_capita`. See [`data/README.md`](data/README.md) for per file schemas.

Supplemented with **World Bank Open Data** (ranges per file in the table below), joined by ISO3 code + year:

- GDP (current US$)
- GDP per capita (current US$)
- Land area (km^2)
- Population (total)

| File | Range | Source |
| --- | --- | --- |
| `ghg_emissions.csv` | 1950-2024 | [Kaggle - Global GHG Emissions, 1950-2024](https://www.kaggle.com/datasets/lucalullo/global-ghg-gas-emissions-by-country-1950-2024) (OWID / Climate Watch) |
| `wb_gdp.csv` | 1960-2024 | [World Bank - GDP, current US$](https://data.worldbank.org/indicator/NY.GDP.MKTP.CD) |
| `wb_gdp_per_capita.csv` | 1960-2024 | [World Bank - GDP per capita, current US$](https://data.worldbank.org/indicator/NY.GDP.PCAP.CD) |
| `wb_land_area.csv` | 1960-2024 | [World Bank - Land area, sq. km](https://data.worldbank.org/indicator/AG.LND.TOTL.K2) |
| `wb_population.csv` | 1960-2024 | [World Bank - Population, total](https://data.worldbank.org/indicator/SP.POP.TOTL) |
| `iso3_to_continent.csv` | n/a | Curated ISO3 -> continent lookup ([UN M49](https://unstats.un.org/unsd/methodology/m49/) standard, 6 continent split) |

## Setup

### Initial Setup

Python **3.12** is pinned via [`.python-version`](.python-version). Pick one of the two options below; they will produce an identical `.venv/`.

#### Option A - `venv` + `pip` (stdlib, no extra install)

Requires Python 3.12 already installed (e.g. `brew install python@3.12`, or [python.org](https://www.python.org/downloads/)).

```bash
python3.12 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python --version                # verify: Python 3.12.x
pip install -r requirements.txt
```

#### Option B - `uv` (faster, auto installs Python)

Install `uv` once: `brew install uv`, or `curl -LsSf https://astral.sh/uv/install.sh | sh`.

```bash
uv venv                         # reads .python-version, auto downloads 3.12 if needed
source .venv/bin/activate       # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

The install pulls the following libraries (full list in [`requirements.txt`](requirements.txt)):

- **pandas** - dataframes and CSV I/O
- **numpy** - numerical array math
- **scikit-learn** - ML models, splits, scaling, metrics, cross validation
- **plotnine** - plotting for all charts

> **For development contributions:** also run `pre-commit install` after the steps above to wire up the ruff lint/format + nbstripout git hooks.

Once initial setup is complete, see [Running the project](#running-the-project) below.

## Running the project

Two ways to run the pipelines. Both assume the venv is activated (`source .venv/bin/activate`).

### Option 1 - Run everything at once

```bash
python src/run_all.py
```

Executes steps 1-6 below in order, stops on the first failure or finish. Outputs can be found in `data/processed/` and `reports/<pipeline>/`.

### Option 2 - Run each step manually

Useful when working on a single pipeline. Run from the repo root in order; each step's output feeds the next.

| # | Command | Produces |
| --- | --- | --- |
| 1 | `python src/build_master.py` | `data/processed/master.csv` - inner joins the World Bank files onto the GHG dataset on `(iso_code, year)` and attaches a continent label from `data/iso3_to_continent.csv`. |
| 2 | `python src/clean.py` | `data/processed/master_clean.csv` - fills `land_area` `NaN`s with country mean and drops rows with `NaN` in any other tracked feature. |
| 3 | `python src/eda.py` | `reports/eda/report.md` + `reports/eda/images/` - distributions, time series, bivariate scatterplots, continent class balance, `NaN` counts. |
| 4 | `python src/regression.py` | `reports/regression/report.md` + images - Linear/Elastic Net/KNN/Random Forest predicting `total_ghg` and `ghg_per_capita`. Row level 80/20 split. |
| 5 | `python src/classification.py` | `reports/classification/report.md` + images - Logistic Regression/KNN/SVM/Random Forest predicting `continent`. Country level stratified group split. |
| 6 | `python src/clustering.py` | `reports/clustering/report.md` + images - K-Means and HAC clustering on a country level 2024 snapshot, run in raw and log transformed feature variants. |

After all six scripts run, each markdown report can be opened from `reports/<pipeline>/report.md`. The 5 page final write up and presentation slides (per the handout) are assembled separately from these outputs.

> **NOTE:** The full script/pipeline run through make take up to **10 minutes** to run due to grid searches and tuning.

## Repository Layout

```
.
├── .github/
│   ├── workflows/ci.yml         # runs pre-commit on every PR
│   └── pull_request_template.md
├── data/
│   ├── README.md                # per dataset schemas
│   ├── *.csv                    # raw inputs (GHG + World Bank) + iso3_to_continent.csv lookup
│   └── processed/               # generated master CSVs (gitignored except .gitkeep)
├── docs/
│   ├── specs/                   # design specs per feature
│   ├── CPSC393_ML_Final_Project.pdf
│   └── Project Proposal.pdf
├── reports/                     # generated pipeline reports (gitignored except .gitkeep)
│   ├── eda/                     #   report.md + images/ from src/eda.py
│   ├── regression/              #   report.md + images/ from src/regression.py
│   ├── classification/          #   report.md + images/ from src/classification.py
│   └── clustering/              #   report.md + images/ from src/clustering.py
├── src/
│   ├── build_master.py          # joins raw CSVs into data/processed/master.csv
│   ├── clean.py                 # handles NaN, writes data/processed/master_clean.csv
│   ├── eda.py                   # exploratory charts + report
│   ├── regression.py            # four regression models, both GHG targets
│   ├── classification.py        # four classification models predicting continent
│   ├── clustering.py            # K-Means + HAC on 2024 snapshot, raw and log variants
│   └── run_all.py               # executes the six pipelines in order
├── .gitignore
├── .pre-commit-config.yaml      # ruff + nbstripout hooks
├── .python-version              # pinned to 3.12
├── requirements.txt             # Python dependencies
├── LICENSE
├── README.md
└── REQUIREMENTS.md              # markdown version of the handout
```

## Results

Headline metrics per pipeline; full breakdowns in `reports/<pipeline>/report.md`.

### Regression

Predicting `total_ghg` and `ghg_per_capita` from features.

- `total_ghg`: Random Forest reaches **R^2 = 0.99 / RMSE = 78** on the test set; linear models cap around R^2 = 0.87.
- `ghg_per_capita`: Random Forest reaches **R^2 = 0.68**; linear models near zero (R^2 = 0.10) - the per capita feature is genuinely harder, but we have a good predictor for total GHG and that is sufficient for our use.

### Classification

Predicting `continent` (6 classes) from developmental + emissions features.

- Best test accuracy: **~0.58 (SVM, engineered features)**. Feature engineering (log / squared / reciprocal) raised Logistic Regression, KNN, and SVM performance by 6-11 points but hurt Random Forest slightly.
- The ~0.58 ceiling reflects a feature data limit: six developmental indicators don't cleanly separate six continents.

### Clustering

K-Means and HAC on a 2024 country level snapshot, run in two variants.

- Both algorithms pick **k=2** in raw and log variants.
- **Raw** features yield outlier dominated clusters (4 vs 175 - China, US, India, Russia isolated).
- **Log transformed** features give more interpretable splits (57 vs 122 with K-Means), trading a higher silhouette score for a more useful partition.

## Key Deadlines

| Deliverable | Due |
| --- | --- |
| Project Proposal | Apr 20, 2026 |
| Final Report | May 22, 2026 |
| Final Project Code | May 22, 2026 |
| Presentation | May 6, 2026 |

Full rubric and deliverable specs: see [`REQUIREMENTS.md`](REQUIREMENTS.md).
