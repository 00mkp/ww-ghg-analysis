# CPSC 393 — Final Project: Global GHG Emissions

| **Members** | | |
| --- | --- | --- |
| Maksim Popov | Daniel Franks | Andrew Blanco |

## Description

Final project for **CPSC 393 — Machine Learning** at Chapman University (Spring 2026, Dr. Rajeev Joshi).

We're analyzing country-level greenhouse gas (GHG) emissions across ~80 years to understand how developmental indicators (population, land area, GDP, GDP per capita) relate to total and per-capita emissions. The goal is twofold:

1. **Regression** — predict a country's GHG emissions from its developmental features, comparing models from the course.
2. **Clustering** — see whether grouping countries by these characteristics produces intuitive emissions tiers (clustering methods pulled from CPSC 392 since they aren't covered in 393).

The motivation: give developing countries a view of their projected emissions trajectory so they can plan mitigation early, and help global reduction efforts identify which country profiles drive the most emissions.

## Data

Data lives in `data/`. The core dataset is the Kaggle **Global Greenhouse Gas Emissions by Country, 1950–2024** (originally Our World in Data / Climate Watch): 14,924 rows × 5 columns covering 199 countries — `country`, `year`, `iso_code`, `total_ghg` (Mt CO₂-eq), `ghg_per_capita` (t CO₂-eq). See [`data/README.md`](data/README.md) for per-file schemas.

Supplemented with **World Bank Open Data** (1960–2024), joined by ISO3 code + year:

- GDP (current US$)
- GDP per capita (current US$)
- Land area (km²)
- Population (total)

| File | Range | Source |
| --- | --- | --- |
| `ghg_emissions.csv` | 1950–2024 | Kaggle / OWID |
| `wb_gdp.csv` | 1960–2024 | World Bank — GDP (current US$) |
| `wb_gdp_per_capita.csv` | 1960–2024 | World Bank — GDP per capita |
| `wb_land_area.csv` | 1960–2024 | World Bank — Land area (km²) |
| `wb_population.csv` | 1960–2024 | World Bank — Population |

## Setup

Python **3.12** is pinned via [`.python-version`](.python-version). Pick one of the two options below — they produce an identical `.venv/` and you can mix-and-match across teammates.

### Option A — `venv` + `pip` (stdlib, no extra install)

Requires Python 3.12 already installed (e.g. `brew install python@3.12`, or [python.org](https://www.python.org/downloads/)).

```bash
python3.12 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python --version                # verify: Python 3.12.x
pip install -r requirements.txt
pre-commit install
```

### Option B — `uv` (faster, auto-installs Python)

Install `uv` once: `brew install uv`, or `curl -LsSf https://astral.sh/uv/install.sh | sh`.

```bash
uv venv                         # reads .python-version, auto-downloads 3.12 if needed
source .venv/bin/activate       # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
pre-commit install
```

`pre-commit install` wires the git hooks into your local clone (one-time per clone).

## Workflow

1. **Build the merged dataset** — join the supplemental World Bank files onto the GHG dataset using ISO3 code + year.
2. **Clean & explore** — handle missing values, duplicates, outliers; visualize distributions and relationships; settle on a train/test split.
3. **Regression modeling** — baseline first, then improved models using techniques from the course; tune hyperparameters; compare with course metrics.
4. **Clustering** — run multiple clustering algorithms on the cleaned features, compare them, and evaluate whether resulting groups are intuitive.
5. **Write-up** — produce the 5-page final report and presentation deck per the handout.

## Repository Layout

```
.
├── .github/
│   ├── workflows/ci.yml         # runs pre-commit on every PR
│   └── pull_request_template.md
├── data/
│   ├── README.md                # per-dataset schemas
│   └── *.csv                    # raw CSVs (GHG + World Bank supplements)
├── docs/                        # handout and proposal PDFs
├── src/                         # project source code
├── .gitignore
├── .pre-commit-config.yaml      # ruff + nbstripout hooks
├── .python-version              # pinned to 3.12
├── requirements.txt             # Python dependencies
├── LICENSE
├── README.md
└── REQUIREMENTS.md              # markdown version of the handout
```

## Key Deadlines

| Deliverable | Due |
| --- | --- |
| Project Proposal | Apr 20, 2026 *(submitted)* |
| Final Report | May 22, 2026 |
| Final Project Code | May 22, 2026 |
| Presentation | TBA |

Full rubric and deliverable specs: see [`REQUIREMENTS.md`](REQUIREMENTS.md).
