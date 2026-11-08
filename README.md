# Delhi Air Quality Forecasting (2018–2026)

[![Tests](https://github.com/MohdNoman07/delhi-aqi-explainable-forecast/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/MohdNoman07/delhi-aqi-explainable-forecast/actions/workflows/tests.yml)

Multi-horizon daily Air Quality Index (AQI) forecasting for the National Capital Territory of Delhi, India, utilizing 8.75 years of continuous hourly monitoring data from the Central Pollution Control Board (CPCB) station network.

The system predicts city-wide CPCB-style NAQI from 1 to 5 days ahead ($h=1..5$) with leakage controls with automated tests, chronological evaluation across two holdout winter seasons, paired block-bootstrap confidence intervals, and SHAP tree attribution.

---

## 1. Executive Summary & Key Results

- **Data Ingestion & Quality Control**:
  - Raw input comprises **2,767,032 observations** across 37 monitoring stations from `2018-01-01` to `2026-09-30`.
  - Retained **35 representative stations** with ≥50% temporal coverage. The 2 dropped stations recorded in `outputs/station_selection.csv` are `site_1422_Dwarka_Sector_8_Delhi_DPCC` (8.34% coverage, 6,394 of 76,680 hours) and `site_5395_Lodhi_Road_Delhi_IITM` (34.40% coverage, 26,377 of 76,680 hours).
  - Standardized barometric pressure to hPa ($P \times 1.333224$ for mm Hg readings; valid physical range $[940, 1030]$ hPa).
  - Invalidated 173,979 frozen readings via a 24-hour stuck-sensor filter (≥24 consecutive identical non-zero values set to NaN across all variables except rainfall).
  - Screened 3,691 station-hours where $PM_{2.5} > 1.2 \times PM_{10}$ and $PM_{2.5} > 20\ \mu\text{g/m}^3$ (both set to NaN).
  - Screened targeted VOC spikes on benzene and toluene ($10\times$ rolling 24-hour median filter); preserved criteria pollutants and meteorology without spike filtering.
  - Reindexed stations to continuous 76,680 hours; applied 6-hour bounded linear interpolation strictly inside gaps (`limit=6`, `limit_area="inside"`).
  - City-wide spatial hourly aggregation required ≥3 reporting stations for pollutants and ≥2 for weather; wind direction computed via circular vector mean.
  - Daily aggregation: 24-hour arithmetic means for $PM_{2.5}, PM_{10}, NO_2, SO_2, NH_3$ requiring ≥16 valid hourly readings; continuous 8-hour trailing rolling mean for $CO$ and $O_3$ on hourly data requiring ≥6 of 8 valid hours (`min_periods=6`), followed by taking the daily maximum for the calendar day.
  - Aggregated to 3,195 calendar days (`clean_daily.csv`), achieving **99.87% coverage (3,191 valid days; 4 missing days)**.

- **Feature Engineering & Leakage Controls**:
  - Engineered **38 features** using strictly historical trailing information ($t' \le t$): trailing rolling windows, lags, daily maxima, wind vectors, and cyclical calendar harmonics.
  - Multi-horizon targets: `target_h1` ... `target_h5` ($AQI_{t+h}$). 59 rows dropped due to trailing lags (7 days), forward targets (5 days), and mid-series missing data ripple (47 days); **3,136 model-ready rows**.
  - Chronological partition with 5-day buffer gaps: **Train** (1,887 days, ≤ 2023-03-31), **Validation** (352 days, 2023-04-06 to 2024-03-31), **Holdout Test** (887 days, 2024-04-06 to 2026-09-25, spanning two complete winter seasons).
  - Hyperparameter tuning conducted on Train only via 4-fold expanding-window `TimeSeriesSplit(n_splits=4, gap=5)`.

- **Holdout Test Performance ($N=887$, 2024–2026)**:
  - *Best baseline definition*: The "best baseline" is chosen per horizon based on validation MAE among {Persistence, Climatology, Damped Anomaly Persistence} (`models/best_baseline.json`). Damped Anomaly Persistence achieves the lowest validation MAE across all five horizons ($h=1..5$). (Ridge is a model, not a baseline). All evaluations involving Damped Anomaly Persistence are post-hoc diagnostics.
  - **Horizon $h=1$**: Random Forest achieves **MAE 27.57** (vs Damped Anomaly Persistence 28.90, Ridge 28.49, Persistence 30.97, Climatology 40.37), **RMSE 38.84**, **$R^2$ 0.876**, **Skill vs Best Baseline +4.6%** (95% block CI `[-2.18, -0.48]`, RF clearly better). Comparing models at $h=1$, RF achieves MAE 27.57 vs Ridge 28.49 (Δ MAE = -0.92, 95% block-bootstrap CI [-1.87, -0.03]), which just excludes zero.
  - **Horizon $h=2$**: Random Forest achieves **MAE 36.34** (vs Damped Anomaly Persistence 36.25, Ridge 37.01, Climatology 40.43, Persistence 42.57), **Skill vs Best Baseline -0.3%**. Paired 95% block-bootstrap CI for RF vs best baseline is `[-1.05, +1.22]` (includes zero).
  - **Horizons $h=3..5$**: Random Forest achieves MAE 38.75 ($h=3$), 40.26 ($h=4$), and 41.48 ($h=5$) vs Damped Anomaly Persistence 38.20 ($h=3$), 39.33 ($h=4$), and 39.79 ($h=5$). Paired bootstrap CIs with the best baseline include zero at $h=3,4$ (`[-1.07, +2.08]`, `[-0.88, +2.54]`), while at $h=5$ the CI `[+0.04, +3.22]` excludes zero with Damped Anomaly Persistence outperforming Random Forest.
  - **High-Pollution Alert Detection ($\text{AQI} > 200$)**:
    - Compared to the trivial seasonal rule ("alert on every winter day, never in non-winter"):
      - *Alert Precision*: Random Forest ties the trivial rule at $h=2$ (0.821 vs 0.821) and falls strictly below the trivial rule at horizons **$h=3, 4, 5$** (0.809, 0.781, and 0.761 vs 0.821; 95% block CI at $h=5$ is `[-0.137, +0.013]`).
      - *Alert Recall*: Random Forest exceeds the trivial rule across all five horizons (0.870 down to 0.764 vs 0.746 to 0.752) due to detecting non-winter alerts (12.0% to 51.2% recall) where the trivial rule catches zero. At day +4, Random Forest alert recall is 0.752 vs 0.749 for the trivial rule, and at day +5 it is 0.764 vs 0.752 (95% block CI `[-0.030, +0.053]`, spanning zero). In plain words, at days 3 to 5 Random Forest has no demonstrated advantage over the trivial winter rule (lower precision, recall gap within about one point with confidence interval including zero).

- **Diagnostic & Attribution Insights**:
  - **Winter vs Non-Winter Split**: Alert base rate is 81.8% in winter vs 14.0% in non-winter. In winter, RF achieves 91.8% precision and 99.2% recall ($h=1$). In non-winter, RF detects 51.2% ($h=1$) and 38.6% ($h=2$) of alert episodes where the trivial rule catches 0%.
  - **Diwali Proximity**: Among the 10 worst prediction errors per horizon (50 total worst-day observations), exactly 4 occur within the window from 1 day before to 3 days after Diwali. All 4 correspond to the single target day `2025-10-21` (evaluated at horizons $h=2..5$); 46 worst error observations fall outside the Diwali window.
  - **SHAP Attribution Dynamics**: Pollutants and AQI history account for 90.3% of attribution at $h=1$, declining to 61.5% at $h=5$. Calendar features shift from 3.8% at $h=1$ to 30.7% at $h=5$; the model's attribution shifts from daily readings toward calendar features as the horizon grows.

---

## 2. Test Set Evaluation Benchmark

Evaluated on 887 holdout test days (`2024-04-06` to `2026-09-25`) after refitting on combined Train + Validation data (buffer gaps strictly excluded).

### Primary Model Performance Across Horizons ($h=1..5$)

| Horizon | Model | MAE | RMSE | R² | Mean Bias | Exact Accuracy | Alert Precision | Alert Recall | Skill vs Best Baseline |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Day +1 | Persistence | 30.97 | 44.60 | 0.836 | +0.15 | 69.3% | 0.852 | 0.852 | -0.072 |
| Day +1 | Climatology | 40.37 | 54.23 | 0.757 | +8.46 | 60.8% | 0.819 | 0.779 | -0.397 |
| Day +1 | Damped Anomaly Persistence | 28.90 | 40.70 | 0.863 | +3.05 | 71.3% | 0.853 | 0.861 | 0.000 (ref) |
| Day +1 | Ridge | 28.49 | 39.11 | 0.874 | -2.46 | 69.3% | 0.858 | 0.858 | +0.014 |
| Day +1 | **Random Forest** | 27.57 | 38.84 | 0.876 | +3.02 | 71.0% | 0.842 | 0.870 | +0.046 |
| Day +2 | Persistence | 42.57 | 59.31 | 0.710 | +0.34 | 61.0% | 0.789 | 0.789 | -0.174 |
| Day +2 | Climatology | 40.43 | 54.28 | 0.758 | +8.56 | 60.7% | 0.819 | 0.779 | -0.115 |
| Day +2 | Damped Anomaly Persistence | 36.25 | 49.76 | 0.796 | +5.37 | 65.1% | 0.833 | 0.828 | 0.000 (ref) |
| Day +2 | Ridge | 37.01 | 49.95 | 0.795 | -5.24 | 63.8% | 0.842 | 0.819 | -0.021 |
| Day +2 | **Random Forest** | 36.34 | 49.29 | 0.800 | +2.02 | 64.9% | 0.821 | 0.831 | -0.003 |
| Day +3 | Persistence | 47.25 | 64.95 | 0.654 | +0.53 | 56.5% | 0.764 | 0.764 | -0.237 |
| Day +3 | Climatology | 40.52 | 54.34 | 0.758 | +8.66 | 60.7% | 0.819 | 0.779 | -0.061 |
| Day +3 | Damped Anomaly Persistence | 38.20 | 52.20 | 0.776 | +6.59 | 62.9% | 0.833 | 0.813 | 0.000 (ref) |
| Day +3 | Ridge | 39.71 | 53.56 | 0.764 | -7.67 | 59.9% | 0.833 | 0.770 | -0.040 |
| Day +3 | **Random Forest** | 38.75 | 52.80 | 0.771 | +1.34 | 61.2% | 0.809 | 0.779 | -0.014 |
| Day +4 | Persistence | 49.81 | 68.30 | 0.618 | +0.66 | 53.8% | 0.749 | 0.749 | -0.266 |
| Day +4 | Climatology | 40.53 | 54.34 | 0.758 | +8.70 | 60.5% | 0.819 | 0.779 | -0.031 |
| Day +4 | Damped Anomaly Persistence | 39.33 | 53.23 | 0.768 | +7.30 | 61.2% | 0.815 | 0.785 | 0.000 (ref) |
| Day +4 | Ridge | 40.09 | 54.52 | 0.756 | -4.31 | 61.8% | 0.825 | 0.770 | -0.019 |
| Day +4 | **Random Forest** | 40.26 | 54.35 | 0.758 | +3.02 | 59.8% | 0.781 | 0.752 | -0.024 |
| Day +5 | Persistence | 52.61 | 70.62 | 0.591 | +0.79 | 52.4% | 0.734 | 0.736 | -0.322 |
| Day +5 | Climatology | 40.49 | 54.32 | 0.758 | +8.74 | 60.5% | 0.819 | 0.782 | -0.018 |
| Day +5 | Damped Anomaly Persistence | 39.79 | 53.72 | 0.764 | +7.72 | 60.9% | 0.823 | 0.776 | 0.000 (ref) |
| Day +5 | Ridge | 40.99 | 55.70 | 0.746 | -0.54 | 60.8% | 0.799 | 0.770 | -0.030 |
| Day +5 | **Random Forest** | 41.48 | 55.72 | 0.746 | +5.28 | 58.9% | 0.761 | 0.764 | -0.042 |

*Note*: $R^2$ mostly reflects the seasonal cycle (Climatology alone achieves $R^2 pprox 0.76$). All evaluations involving Damped Anomaly Persistence are post-hoc diagnostics.

### Paired Comparisons & 95% Block-Bootstrap Confidence Intervals

Paired differences in daily absolute error (7-day blocks, 2,000 resamples, seed 42). A block-size sensitivity analysis across 7, 14, 21, and 30 days (`outputs/bootstrap_block_sensitivity.csv`) confirms that Random Forest's advantage over the best baseline at Day +1 is robust to block definition, while intervals at Days 2 to 4 span zero across all block sizes:

| Horizon | Random Forest MAE | Best Baseline | Baseline MAE | Δ MAE (RF - Baseline) [95% CI] | Ridge MAE | Δ MAE (RF - Ridge) [95% CI] |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|
| Day +1 | 27.57 | Damped Anomaly Persistence | 28.90 | -1.33 [-2.18, -0.48] | 28.49 | -0.92 [-1.87, -0.03] |
| Day +2 | 36.34 | Damped Anomaly Persistence | 36.25 | +0.10 [-1.05, +1.22] | 37.01 | -0.67 [-2.06, +0.61] |
| Day +3 | 38.75 | Damped Anomaly Persistence | 38.20 | +0.55 [-1.07, +2.08] | 39.71 | -0.96 [-2.86, +0.81] |
| Day +4 | 40.26 | Damped Anomaly Persistence | 39.33 | +0.93 [-0.88, +2.54] | 40.09 | +0.16 [-1.80, +1.95] |
| Day +5 | 41.48 | Damped Anomaly Persistence | 39.79 | +1.69 [+0.04, +3.22] | 40.99 | +0.49 [-1.60, +2.50] |

### Full Test Set Alert Detection Benchmark (Random Forest vs Trivial Seasonal Rule)

Across all 887 test days (alert base rate = 37.3%), comparison of Random Forest alert precision and recall against the trivial seasonal rule ("alert on every winter day, never in non-winter"):

| Horizon | RF Alert Precision | RF Alert Recall | Trivial Alert Precision | Trivial Alert Recall | RF Non-Winter Alert Recall |
|:---:|:---:|:---:|:---:|:---:|:---:|
| h=1 | 0.842 | 0.870 | 0.818 | 0.746 | 0.512 |
| h=2 | 0.821 | 0.831 | 0.821 | 0.749 | 0.386 |
| h=3 | 0.809 | 0.779 | 0.821 | 0.749 | 0.205 |
| h=4 | 0.781 | 0.752 | 0.821 | 0.749 | 0.120 |
| h=5 | 0.761 | 0.764 | 0.821 | 0.752 | 0.146 |

- **Horizons where Random Forest is at or below the trivial rule**:
  - **Alert Precision**: Random Forest ties the trivial rule at horizon $h=2$ (0.821 vs 0.821) and falls strictly below the trivial rule at horizons **$h=3, h=4$, and $h=5$** (0.809, 0.781, and 0.761 vs 0.821).
  - **Alert Recall**: Random Forest exceeds the trivial rule across all five horizons on the full test set (0.870 down to 0.764 vs 0.746 to 0.752) due to detecting non-winter alerts (12.0% to 51.2% recall) where the trivial rule catches zero. At day +4, Random Forest alert recall is 0.752 vs 0.749 for the trivial rule, and at day +5 it is 0.764 vs 0.752. In plain words, at days 3 to 5 Random Forest has no demonstrated advantage over the trivial winter rule (lower precision: 95% block-bootstrap CI [-0.137, +0.013] at h=5; recall gap within about one point: 95% block-bootstrap CI [-0.030, +0.053] at h=5, spanning zero).

### Chronological Train / Validation / Test Partitions

| Partition | Start Date | End Date | Row Count | Days Dropped in Buffer Gap | Description |
|:---|:---:|:---:|:---:|:---:|:---|
| **Train** | `2018-01-08` | `2023-03-31` | **1887** | — | Training data up to `2023-03-31` (~5.25 years) |
| *Gap 1* | `2023-04-01` | `2023-04-05` | **5** | **5** | 5-day chronological buffer between Train and Val |
| **Validation** | `2023-04-06` | `2024-03-31` | **352** | — | Validation data up to `2024-03-31` (1 full calendar year) |
| *Gap 2* | `2024-04-01` | `2024-04-05` | **5** | **5** | 5-day chronological buffer between Val and Test |
| **Test** | `2024-04-06` | `2026-09-25` | **887** | — | Final evaluation holdout (~2.5 years, through `2026-09-25`) |
| **Total** | `2018-01-08` | `2026-09-25` | **3136** | **10** | All calendar rows accounted for |

---

## 3. Dataset

- **Source**: Kaggle — [Delhi Air Quality CPCB Hourly (2010–2026)](https://www.kaggle.com/datasets/yug201/delhi-air-quality-cpcb-hourly-2010-2026).
- **Download Instructions**:
  1. Download the archive from Kaggle.
  2. Extract the individual station CSV files directly into `data/raw/` (e.g. `data/raw/site_103_CRRI_Mathura_Road_Delhi_IITM.csv`, etc.).
  3. Ensure `data/raw/` contains the 37 monitoring station CSV files covering 2018–2026.
- **Repository Note**: Raw data files are not committed to this repository due to size.
- **Licence Terms**: Licence terms are available on the Kaggle dataset page and have not been independently verified.
- **Git Hygiene**: If this is managed as a git repo, run `git rm --cached data/processed/raw_filtered.parquet` and perform a history-rewrite if desired to keep the repository history compact.

---

## 4. Setup & Environment

The project is developed and verified under Python 3.11–3.12.

```bash
# 1. Navigate to the project root directory
cd <project-root>

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the test suite
pytest -q
```

Compatible-release ranges in `requirements.txt`:
- `pandas>=2.2,<3`
- `numpy>=1.26,<3`
- `scipy>=1.11,<2`
- `scikit-learn>=1.4,<2`
- `joblib>=1.3,<2`
- `pyarrow>=14.0,<20`
- `matplotlib>=3.8,<4`
- `shap>=0.45,<1`
- `pytest>=8,<9`
- `seaborn>=0.13,<1`
- `statsmodels>=0.14,<1`
- `nbformat>=5.9,<6`
- `nbclient>=0.9,<1`

---

## 5. Command-Line Inference Demo (`src/predict.py`)

A command-line tool `src/predict.py` generates 5-day forward forecasts from any date in `clean_daily.csv` or for the latest observation. It loads serialized models (`models/rf_h1.pkl`..`rf_h5.pkl`) and feature definitions (`models/feature_columns.json`) with zero retraining.

### Usage

```bash
# Forecast for a specific date (e.g. Severe winter day)
python src/predict.py --date 2024-11-15

# Forecast for the latest available observation
python src/predict.py --latest
```

### Real Sample Output (`--latest`)

```text
Selecting latest date from daily observations: 2026-09-30

===============================================================================================
DELHI AIR QUALITY 5-DAY FORWARD FORECAST | BASE DATE: 2026-09-30
===============================================================================================
Horizon  | Target Date | Pred AQI | Category      | Typical Error | Actual AQI | Advisory
-----------------------------------------------------------------------------------------------
Day +1   | 2026-10-01  | 146.93   | Moderate      | ±27.57        | N/A        | people with breathing or heart conditions, children and older adults should reduce prolonged outdoor exertion
Day +2   | 2026-10-02  | 137.66   | Moderate      | ±36.34        | N/A        | people with breathing or heart conditions, children and older adults should reduce prolonged outdoor exertion
Day +3   | 2026-10-03  | 133.96   | Moderate      | ±38.75        | N/A        | people with breathing or heart conditions, children and older adults should reduce prolonged outdoor exertion
         * Note: at this horizon the model is statistically indistinguishable from a seasonal average
Day +4   | 2026-10-04  | 136.39   | Moderate      | ±40.26        | N/A        | people with breathing or heart conditions, children and older adults should reduce prolonged outdoor exertion
         * Note: at this horizon the model is statistically indistinguishable from a seasonal average
Day +5   | 2026-10-05  | 138.16   | Moderate      | ±41.48        | N/A        | people with breathing or heart conditions, children and older adults should reduce prolonged outdoor exertion
         * Note: at this horizon the model is statistically indistinguishable from a seasonal average
-----------------------------------------------------------------------------------------------
Informational only, not medical advice.
===============================================================================================

Saved forecast table to outputs/forecast_2026-09-30.csv
Saved forecast figure to outputs/forecast_2026-09-30.png
```

---

## 6. Repository Architecture & Directory Structure

```
. (project-root)
├── data/
│   ├── raw/                           # Raw CPCB station CSVs (2010–2026) and data dictionary
│   └── processed/
│       ├── clean_hourly.parquet       # Cleaned, standardized, interpolated hourly grid (76,680 rows)
│       ├── clean_daily.csv            # City-wide 24h daily aggregates and clipped AQI (3,195 rows)
│       └── model_ready.csv            # 38 features and 5 forward targets without NaNs (3,136 rows)
├── docs/                              # Project documentation, full technical report draft, and tracked numbers
│   ├── numbers_used.csv               # Every single tracked metric and parameter with source provenance
│   ├── report_draft.md                # 11-section publication-ready technical methodology report
│   └── slides_outline.md              # 12-slide executive presentation outline
├── models/                            # Serialized models (scikit-learn >=1.4 / Python 3.11–3.12 pickle compatibility)
│   ├── feature_columns.json           # Exact list of 38 engineered feature names
│   ├── rf_config.json                 # Optimal hyperparameters and target scale per horizon
│   ├── climatology_train_val.json     # Climatology lookup table fit on Train + Validation
│   ├── ridge_h1.pkl ... ridge_h5.pkl  # Trained Ridge regression pipelines (StandardScaler + Ridge)
│   └── rf_h1.pkl ... rf_h5.pkl        # Winning Random Forest models for horizons h=1..5
├── notebooks/
│   └── 01_eda.ipynb                   # Comprehensive exploratory data analysis notebook
├── outputs/
│   ├── metrics.csv                    # Full regression and classification metrics on Test set
│   ├── per_class_metrics.csv          # Per-category precision, recall, F1, and support on Test set
│   ├── test_predictions.csv           # Holdout test set date-by-date truths and model predictions
│   ├── rf_vs_baseline_ci.csv          # Block-bootstrap 95% CIs for RF vs best baseline
│   ├── rf_vs_ridge_ci.csv             # Block-bootstrap 95% CIs for RF vs Ridge
│   ├── winter_split_ci.csv            # Block-bootstrap 95% CIs partitioned by winter vs non-winter
│   ├── worst_days_h1.csv ... h5.csv   # Top 10 worst prediction days per horizon with Diwali proximity
│   ├── error_analysis.md              # Detailed diagnostic report on monthly, seasonal, and extreme errors
│   ├── error_by_month.png             # Monthly MAE bar charts comparing RF vs best baseline
│   ├── winter_timeseries_h1_h5.png    # Time series tracking across test winters for h=1 and h=5
│   ├── residuals_h1_h5.png            # Residual vs predicted scatter and error distributions
│   ├── shap_notes.md                  # Comprehensive SHAP interpretation and credit splitting notes
│   ├── shap_family_importance.csv     # Attributions grouped into Pollutants, AQI, Weather, Calendar
│   ├── shap_family_importance.png     # Stacked bar charts showing shift from observations to season
│   ├── shap_summary_h1.png ... h5.png # SHAP beeswarm summary plots for h=1, 3, 5
│   ├── shap_bar_h1.png ... h5.png     # SHAP mean |SHAP| ranking bar charts for h=1, 3, 5
│   ├── shap_waterfall_*.png           # Single-instance waterfall attribution case studies
│   ├── forecast_*.csv                 # Inference forecast CSV tables
│   └── forecast_*.png                 # Inference forecast and historical trajectory charts
├── scratch/                           # One-off audit scripts, verification tools, and build helpers (untracked)
├── src/
│   ├── config.py                      # Column mappings, physical caps, breakpoints, split dates, random seed
│   ├── load_raw.py                    # Ingests raw station CSVs, filters 2018–2026, logs row counts
│   ├── preprocess.py                  # Cleaning, pressure conversion, stuck-sensor, interpolation, spatial mean
│   ├── compute_aqi.py                 # CPCB-style sub-index calculation and continuous breakpoint logic
│   ├── daily_aggregation.py           # 24h averaging, daily sub-indices, AQI computation
│   ├── features.py                    # Trailing feature engineering and multi-horizon targets
│   ├── split.py                       # Chronological split with 5-day buffer gaps and TimeSeriesSplit CV
│   ├── evaluate.py                    # Metric calculation suite: regression, 6/5-class, alerts, skill
│   ├── train.py                       # Fits baselines and tunes Random Forest on Train / Validation
│   ├── final_evaluate.py              # Refits on Train+Val, evaluates on Holdout Test, computes bootstrap CIs
│   ├── error_analysis.py              # Diagnostic evaluation of monthly, seasonal, and worst-day errors
│   ├── explain.py                     # TreeExplainer SHAP attribution pipeline across test horizons
│   ├── predict.py                     # Command-line inference tool supporting --date and --latest
│   └── make_readme_tables.py          # Programmatic markdown table generator from saved artifacts
├── tests/
│   ├── test_aqi.py                    # Unit tests for CPCB-style sub-indices, breakpoints, clipping
│   ├── test_no_leakage.py             # Unit tests verifying zero temporal leakage in features
│   ├── test_split.py                  # Unit tests verifying chronological split ordering and gap buffers
│   ├── test_evaluate.py               # Unit tests for regression metrics, classifications, and skill score
│   ├── test_rf.py                     # Unit tests for serialized RF model files and prediction shapes
│   ├── test_shap.py                   # Unit tests verifying SHAP additivity (preds == base + sum(SHAP))
│   └── test_predict.py                # Unit tests verifying CLI prediction exact match against test set
├── requirements.txt                   # Pinned dependency specifications from working environment
├── pytest.ini                         # Pytest test path configuration (pythonpath = .)
└── PROJECT_NOTES.md                   # Mandatory rules, run logs, verified numbers, and project history
```

---

## 7. End-to-End Pipeline Reproduction

All scripts run from the repository root using standard Python 3.11–3.12 and rely on the fixed seed `RANDOM_SEED = 42` defined in `src/config.py`:

```bash
# 1. Ingest raw station CSVs (filters to 2018-2026, standardizes column names)
python src/load_raw.py

# 2. Preprocess station data (stuck sensor filter, pressure conversion to hPa,
#    PM inversion screening, 6h bounded interpolation, spatial averaging)
python src/preprocess.py

# 3. Compute daily aggregates and CPCB-style NAQI
python src/daily_aggregation.py

# 4. Construct trailing features and multi-horizon targets (h=1..5)
python src/features.py

# 5. Generate chronological split summary and CV fold diagnostics
python src/split.py

# 6. Fit baselines and tune Random Forest on Train / Validation
python src/train.py

# 7. Execute final holdout test evaluation (refits on Train+Val, evaluates once on Test)
python src/final_evaluate.py

# 8. Run detailed error analysis and seasonal diagnostics
python src/error_analysis.py

# 9. Generate SHAP feature attributions and waterfall case studies
python src/explain.py

# 10. Generate updated README markdown tables from saved outputs
python src/make_readme_tables.py

# 11. Execute the automated test suite
pytest
```

---

## 8. CPCB-Style NAQI Calculation Rules

The daily Air Quality Index follows the Central Pollution Control Board (CPCB) National Air Quality Index methodology (*breakpoints follow CPCB NAQI (2014 revision)*):

1. **Daily Aggregation Protocol**:
   - $PM_{2.5}, PM_{10}, NO_2, SO_2, NH_3$: 24-hour arithmetic mean (requires ≥16 valid hourly readings; otherwise NaN).
   - $CO, O_3$: Continuous 8-hour trailing rolling mean computed on the continuous hourly data (requires ≥6 of 8 valid hours, `min_periods=6`), followed by taking the daily maximum of these 8h rolling means for the calendar day.
2. **Sub-Index Calculation**:
   - Continuous linear interpolation between breakpoint intervals $[B_{lo}, B_{hi}]$ mapped to index ranges $[I_{lo}, I_{hi}]$.
   - Internal sub-indices may extend linearly above 500 for concentrations exceeding the Severe breakpoint.
3. **City AQI & Clipping**:
   - Overall $\text{AQI} = \max(\text{Sub-indices})$.
   - Validity criteria: requires valid sub-indices for at least 3 criteria pollutants, of which at least one must be $PM_{2.5}$ or $PM_{10}$.
   - The final reported AQI and category assignment are strictly clipped to the range **$[0, 500]$**.

---

## 9. Mandatory Project Rules

1. **Column Mapping**: Never assume column names or units; use `COLUMN_MAP` in `src/config.py`. Output pressure unit is standardized to `hPa`.
2. **Chronological Splitting**: Never shuffle time series data. Use chronological splits separated by 5-day buffer gaps. Headline metrics were computed once by final_evaluate.py; later diagnostics (alert comparison vs the trivial seasonal rule, Diwali analysis, error analysis) were exploratory and post-hoc.
3. **Leakage Controls with Automated Tests**: Features use only information available up to the end of today ($t' \le t$). Rolling windows are strictly trailing (`center=False`). No negative shifts allowed in features.
4. **Independent Fitting**: Imputers, scalers, and climatology tables are fit on training data only.
5. **Fixed Random Seed**: Deterministic execution enforced across all models and splits using `RANDOM_SEED = 42`.
6. **Execution & Auditability**: Every script runs from the repository root, saves outputs to `outputs/` or `data/processed/`, and logs all dropped rows.
7. **Verification Standard**: Never state a cause or fact not verified from repo data or documentation. Unverified dates or hypotheses are explicitly labeled as such.

---

## 10. Automated Test Suite

The automated test suite in `tests/` spans 7 test modules covering all pipeline components:

| Test Module | Coverage | Status |
|---|---|:---:|
| `tests/test_aqi.py` | Hand-calculated sub-indices, CPCB breakpoint boundaries, Severe band clipping (PM2.5=250 gives 400, PM2.5=380 gives 500, PM2.5=500 gives 500 after clipping, PM10=430 gives 400, CO=34 gives 400), validity requirements | **Passed** ✅ |
| `tests/test_no_leakage.py` | Target exclusion, absence of negative shifts in feature code, temporal slice invariance across 20 random dates | **Passed** ✅ |
| `tests/test_split.py` | No date overlap across Train/Val/Test, strictly chronological ordering, buffer gaps ≥5 days | **Passed** ✅ |
| `tests/test_evaluate.py` | Metric functions: perfect prediction arrays, constant offset, category boundaries (50/51/200/201), alert detection, skill score | **Passed** ✅ |
| `tests/test_rf.py` | Model serialization, config JSON schema, prediction output shape, range bounds within $[0, 500]$ | **Passed** ✅ |
| `tests/test_shap.py` | Additivity verification: model prediction matches SHAP values sum + base value within $10^{-5}$ | **Passed** ✅ |
| `tests/test_predict.py` | Command-line demo verification: prediction match with saved test predictions within $10^{-13}$ across 3 dates and error handling | **Passed** ✅ |

To execute the test suite:
```bash
pytest
```
*Current test result: 31 passed, 1 warning (R² on single sample edge test) across 7 test files.*

---

## 11. Limitations

1. **AQI Cap at 500 and the 53 Days Clipped**: CPCB AQI is capped at 500. Exactly 53 calendar days in `clean_daily.csv` have unclipped sub-indices exceeding 500 (reaching up to 885.22); the model may under-predict the most extreme days (explanation not tested).
2. **City-Average AQI vs Station Bulletins**: AQI in this project is computed from city-wide spatially averaged concentrations. This produces a single representative metropolitan metric, which differs from averaging station-level AQIs or from official CPCB bulletins that report individual station sub-indices.
3. **Breakpoint Verification**: Breakpoints follow CPCB NAQI (2014 revision). Breakpoints are implemented as continuous-float bands in `src/compute_aqi.py`.
4. **Absence of Numerical Weather Forecasts**: The models rely solely on historical trailing meteorology up to today ($t' \le t$) without forward NWP weather forecast inputs. Rapid meteorological changes (sudden wind shifts, rainfall washout, boundary layer collapses) cannot be anticipated days in advance.
5. **Data Missingness and August 2026 Gap**: 4 calendar days in 2018–2026 have missing daily AQI due to insufficient station coverage. In August 2026, exactly 71 hours have zero reporting stations for $PM_{2.5}$ (53 hours from Aug 14 to Aug 16, and 18 hours from Aug 18 to Aug 19; cause unknown).
6. **Distribution Shift Across Split Periods**: High-pollution frequency differs between partitions: Poor-or-worse days represent 48.97% of Train (132 Severe days), 43.47% of Validation (25 Severe days), and 37.32% of Test (44 Severe days).
7. **Geographic Specificity**: All models and spatial weights are tuned strictly to the National Capital Territory of Delhi and do not generalize to other urban airsheds without retraining.
8. **Long-Range Model Skill ($h \ge 3$) and Damped Anomaly Baseline**: Random Forest achieves statistically significant skill improvements at Day +1 (+4.6% vs Damped Anomaly Persistence, 95% block CI `[-2.18, -0.48]`). Beyond Day +1, Random Forest does not demonstrate superior accuracy over the damped anomaly baseline: at Days 2 to 4 paired bootstrap CIs include zero, and at Day +5 the damped baseline outperforms Random Forest (MAE 39.79 vs 41.48, 95% CI `[+0.04, +3.22]`).
9. **Retrospective Elements in Hourly Cleaning**: While daily features are strictly trailing ($t' \le t$, enforced by tests), hourly data cleaning is retrospective: (a) bounded linear interpolation uses the value after a gap ($\le 6$ hours), (b) the stuck-sensor filter blanks a whole consecutive run once it is known to last $\ge 24$ hours, and (c) station selection evaluates coverage over the full 2018–2026 period.

---

---

## 13. License

The code in this repository is licensed under the [MIT License](./LICENSE).

The underlying AQI dataset (*Delhi Air Quality CPCB Hourly 2010–2026*) is licensed separately on Kaggle and is **not** covered by the MIT license.

## 12. References

1. **Central Pollution Control Board (CPCB)**, Ministry of Environment, Forest and Climate Change, Government of India: *National Air Quality Index* report, 2014.
2. **Kaggle Dataset**: *Delhi Air Quality CPCB Hourly (2010–2026)*, [https://www.kaggle.com/datasets/yug201/delhi-air-quality-cpcb-hourly-2010-2026](https://www.kaggle.com/datasets/yug201/delhi-air-quality-cpcb-hourly-2010-2026).
3. **Scikit-learn**: Pedregosa et al., *Scikit-learn: Machine Learning in Python*, Journal of Machine Learning Research (JMLR), 12, pp. 2825–2830, 2011.
4. **SHAP (SHapley Additive exPlanations)**: Lundberg, S. M., & Lee, S.-I., *A Unified Approach to Interpreting Model Predictions*, Advances in Neural Information Processing Systems (NeurIPS 30), 2017.
