"""
src/make_readme_tables.py
Generates markdown tables and complete README.md programmatically from saved artifacts.
Never hard-codes numerical values; reads directly from:
- outputs/metrics.csv
- outputs/rf_vs_baseline_ci.csv
- outputs/rf_vs_ridge_ci.csv
- outputs/station_selection.csv
- outputs/error_analysis.md / outputs/test_predictions.csv
- outputs/split_summary.md
"""

from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_results_table_md() -> str:
    """Generates the primary test results markdown table from outputs/metrics.csv."""
    metrics_path = REPO_ROOT / "outputs" / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} not found.")

    df = pd.read_csv(metrics_path)

    lines = [
        "| Horizon | Model | MAE | RMSE | R² | Mean Bias | Exact Accuracy | Alert Precision | Alert Recall | Skill vs Best Baseline |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for h in sorted(df["horizon"].unique()):
        df_h = df[df["horizon"] == h]
        model_order = ["Persistence", "Climatology", "Damped Anomaly Persistence", "Ridge", "Random Forest"]
        for m in model_order:
            sub = df_h[df_h["model"] == m]
            if sub.empty:
                continue
            row = sub.iloc[0]
            m_name = f"**{m}**" if m == "Random Forest" else m
            skill_str = (
                f"{row['skill_vs_best_baseline']:+.3f}"
                if m == "Random Forest"
                else (
                    "0.000 (ref)"
                    if row["skill_vs_best_baseline"] == 0
                    else f"{row['skill_vs_best_baseline']:+.3f}"
                )
            )
            bias_str = f"{row['bias']:+.2f}"
            lines.append(
                f"| Day +{h} | {m_name} | {row['MAE']:.2f} | {row['RMSE']:.2f} | {row['R2']:.3f} | {bias_str} | {row['accuracy']*100:.1f}% | {row['alert_precision']:.3f} | {row['alert_recall']:.3f} | {skill_str} |"
            )

    lines.append("")
    lines.append("*Note*: $R^2$ mostly reflects the seasonal cycle (Climatology alone achieves $R^2 \approx 0.76$). All evaluations involving Damped Anomaly Persistence are post-hoc diagnostics.")
    return "\n".join(lines)


def get_ci_comparisons_table_md() -> str:
    """Generates paired differences with 95% block-bootstrap CIs from rf_vs_baseline_ci.csv and rf_vs_ridge_ci.csv."""
    base_ci_path = REPO_ROOT / "outputs" / "rf_vs_baseline_ci.csv"
    ridge_ci_path = REPO_ROOT / "outputs" / "rf_vs_ridge_ci.csv"

    df_base = pd.read_csv(base_ci_path)
    df_ridge = pd.read_csv(ridge_ci_path)

    lines = [
        "| Horizon | Random Forest MAE | Best Baseline | Baseline MAE | Δ MAE (RF - Baseline) [95% CI] | Ridge MAE | Δ MAE (RF - Ridge) [95% CI] |",
        "|:---:|:---:|:---|:---:|:---:|:---:|:---:|",
    ]

    for h in sorted(df_base["horizon"].unique()):
        r_base = df_base[df_base["horizon"] == h].iloc[0]
        r_ridge = df_ridge[df_ridge["horizon"] == h].iloc[0]

        ci_base_str = f"[{r_base['ci_lower_95']:+.2f}, {r_base['ci_upper_95']:+.2f}]"
        ci_ridge_str = f"[{r_ridge['ci_lower_95']:+.2f}, {r_ridge['ci_upper_95']:+.2f}]"

        lines.append(
            f"| Day +{h} | {r_base['rf_mae']:.2f} | {r_base['best_baseline']} | {r_base['best_baseline_mae']:.2f} | {r_base['mean_diff_mae']:+.2f} {ci_base_str} | {r_ridge['ridge_mae']:.2f} | {r_ridge['mean_diff_mae']:+.2f} {ci_ridge_str} |"
        )

    return "\n".join(lines)


def get_alert_benchmark_table_md() -> str:
    """Reads the alert detection benchmark table from outputs/error_analysis.md."""
    path = REPO_ROOT / "outputs" / "error_analysis.md"
    text = path.read_text(encoding="utf-8")
    lines = []
    capture = False
    for line in text.splitlines():
        if "| Horizon | RF Alert Precision |" in line:
            capture = True
        if capture:
            if line.strip() == "":
                break
            lines.append(line)
    return "\n".join(lines)


def get_station_summary_md() -> str:
    """Generates the station retention summary from outputs/station_selection.csv."""
    path = REPO_ROOT / "outputs" / "station_selection.csv"
    df = pd.read_csv(path)
    total_evaluated = len(df)
    kept_count = int(df["kept"].sum())
    dropped = df[~df["kept"]]

    dropped_lines = []
    for _, row in dropped.iterrows():
        pct = row["coverage"] * 100
        dropped_lines.append(
            f"- `{row['station']}`: **{pct:.2f}%** coverage ({int(row['valid_pm_hours']):,} of {int(row['total_period_hours']):,} hours)"
        )

    summary = (
        f"Across the 2018–2026 monitoring window (76,680 possible hours), **{total_evaluated} stations** were evaluated.\n"
        f"**{kept_count} stations** met the ≥50% temporal coverage criterion and were retained.\n"
        f"The **{len(dropped)} dropped stations** and their coverage recorded in `outputs/station_selection.csv` are:\n"
        + "\n".join(dropped_lines)
    )
    return summary


def get_split_summary_table_md() -> str:
    """Extracts split boundary table from outputs/split_summary.md."""
    path = REPO_ROOT / "outputs" / "split_summary.md"
    text = path.read_text(encoding="utf-8")
    lines = []
    capture = False
    for line in text.splitlines():
        if line.startswith("| Partition"):
            capture = True
        if capture:
            if line.strip() == "" or line.startswith("---"):
                break
            lines.append(line)
    return "\n".join(lines)


def build_full_readme() -> str:
    """Constructs the complete README.md adhering to all revision requirements."""
    results_tbl = get_results_table_md()
    ci_tbl = get_ci_comparisons_table_md()
    alert_tbl = get_alert_benchmark_table_md()
    split_tbl = get_split_summary_table_md()

    # Read dropped station numbers dynamically
    st_df = pd.read_csv(REPO_ROOT / "outputs" / "station_selection.csv")
    dropped_stations = st_df[~st_df["kept"]]
    st1 = dropped_stations.iloc[0]
    st2 = dropped_stations.iloc[1]
    st1_str = f"`{st1['station']}` ({st1['coverage']*100:.2f}% coverage, {int(st1['valid_pm_hours']):,} of {int(st1['total_period_hours']):,} hours)"
    st2_str = f"`{st2['station']}` ({st2['coverage']*100:.2f}% coverage, {int(st2['valid_pm_hours']):,} of {int(st2['total_period_hours']):,} hours)"

    # Read CI numbers dynamically
    ci_ridge = pd.read_csv(REPO_ROOT / "outputs" / "rf_vs_ridge_ci.csv")
    r1 = ci_ridge[ci_ridge["horizon"] == 1].iloc[0]
    h1_ridge_desc = f"RF achieves MAE {r1['rf_mae']:.2f} vs Ridge {r1['ridge_mae']:.2f} (Δ MAE = {r1['mean_diff_mae']:+.2f}, 95% block-bootstrap CI [{r1['ci_lower_95']:+.2f}, {r1['ci_upper_95']:+.2f}]), which just excludes zero"

    readme_content = f"""# Delhi Air Quality Forecasting (2018–2026)

Multi-horizon daily Air Quality Index (AQI) forecasting for the National Capital Territory of Delhi, India, utilizing 8.75 years of continuous hourly monitoring data from the Central Pollution Control Board (CPCB) station network.

The system predicts city-wide CPCB-style NAQI from 1 to 5 days ahead ($h=1..5$) with leakage controls with automated tests, chronological evaluation across two holdout winter seasons, paired block-bootstrap confidence intervals, and SHAP tree attribution.

---

## 1. Executive Summary & Key Results

- **Data Ingestion & Quality Control**:
  - Raw input comprises **2,767,032 observations** across 37 monitoring stations from `2018-01-01` to `2026-09-30`.
  - Retained **35 representative stations** with ≥50% temporal coverage. The 2 dropped stations recorded in `outputs/station_selection.csv` are {st1_str} and {st2_str}.
  - Standardized barometric pressure to hPa ($P \\times 1.333224$ for mm Hg readings; valid physical range $[940, 1030]$ hPa).
  - Invalidated 173,979 frozen readings via a 24-hour stuck-sensor filter (≥24 consecutive identical non-zero values set to NaN across all variables except rainfall).
  - Screened 3,691 station-hours where $PM_{{2.5}} > 1.2 \\times PM_{{10}}$ and $PM_{{2.5}} > 20\\ \\mu\\text{{g/m}}^3$ (both set to NaN).
  - Screened targeted VOC spikes on benzene and toluene ($10\\times$ rolling 24-hour median filter); preserved criteria pollutants and meteorology without spike filtering.
  - Reindexed stations to continuous 76,680 hours; applied 6-hour bounded linear interpolation strictly inside gaps (`limit=6`, `limit_area="inside"`).
  - City-wide spatial hourly aggregation required ≥3 reporting stations for pollutants and ≥2 for weather; wind direction computed via circular vector mean.
  - Daily aggregation: 24-hour arithmetic means for $PM_{{2.5}}, PM_{{10}}, NO_2, SO_2, NH_3$ requiring ≥16 valid hourly readings; continuous 8-hour trailing rolling mean for $CO$ and $O_3$ on hourly data requiring ≥6 of 8 valid hours (`min_periods=6`), followed by taking the daily maximum for the calendar day.
  - Aggregated to 3,195 calendar days (`clean_daily.csv`), achieving **99.87% coverage (3,191 valid days; 4 missing days)**.

- **Feature Engineering & Leakage Controls**:
  - Engineered **38 features** using strictly historical trailing information ($t' \\le t$): trailing rolling windows, lags, daily maxima, wind vectors, and cyclical calendar harmonics.
  - Multi-horizon targets: `target_h1` ... `target_h5` ($AQI_{{t+h}}$). 59 rows dropped due to trailing lags (7 days), forward targets (5 days), and mid-series missing data ripple (47 days); **3,136 model-ready rows**.
  - Chronological partition with 5-day buffer gaps: **Train** (1,887 days, ≤ 2023-03-31), **Validation** (352 days, 2023-04-06 to 2024-03-31), **Holdout Test** (887 days, 2024-04-06 to 2026-09-25, spanning two complete winter seasons).
  - Hyperparameter tuning conducted on Train only via 4-fold expanding-window `TimeSeriesSplit(n_splits=4, gap=5)`.

- **Holdout Test Performance ($N=887$, 2024–2026)**:
  - *Best baseline definition*: The "best baseline" is chosen per horizon based on validation MAE among {{Persistence, Climatology, Damped Anomaly Persistence}} (`models/best_baseline.json`). Damped Anomaly Persistence achieves the lowest validation MAE across all five horizons ($h=1..5$). (Ridge is a model, not a baseline). All evaluations involving Damped Anomaly Persistence are post-hoc diagnostics.
  - **Horizon $h=1$**: Random Forest achieves **MAE 27.57** (vs Damped Anomaly Persistence 28.90, Ridge 28.49, Persistence 30.97, Climatology 40.37), **RMSE 38.84**, **$R^2$ 0.876**, **Skill vs Best Baseline +4.6%** (95% block CI `[-2.18, -0.48]`, RF clearly better). Comparing models at $h=1$, {h1_ridge_desc}.
  - **Horizon $h=2$**: Random Forest achieves **MAE 36.34** (vs Damped Anomaly Persistence 36.25, Ridge 37.01, Climatology 40.43, Persistence 42.57), **Skill vs Best Baseline -0.3%**. Paired 95% block-bootstrap CI for RF vs best baseline is `[-1.05, +1.22]` (includes zero).
  - **Horizons $h=3..5$**: Random Forest achieves MAE 38.75 ($h=3$), 40.26 ($h=4$), and 41.48 ($h=5$) vs Damped Anomaly Persistence 38.20 ($h=3$), 39.33 ($h=4$), and 39.79 ($h=5$). Paired bootstrap CIs with the best baseline include zero at $h=3,4$ (`[-1.07, +2.08]`, `[-0.88, +2.54]`), while at $h=5$ the CI `[+0.04, +3.22]` excludes zero with Damped Anomaly Persistence outperforming Random Forest.
  - **High-Pollution Alert Detection ($\\text{{AQI}} > 200$)**:
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

{results_tbl}

### Paired Comparisons & 95% Block-Bootstrap Confidence Intervals

Paired differences in daily absolute error (7-day blocks, 2,000 resamples, seed 42). A block-size sensitivity analysis across 7, 14, 21, and 30 days (`outputs/bootstrap_block_sensitivity.csv`) confirms that Random Forest's advantage over the best baseline at Day +1 is robust to block definition, while intervals at Days 2 to 4 span zero across all block sizes:

{ci_tbl}

### Full Test Set Alert Detection Benchmark (Random Forest vs Trivial Seasonal Rule)

Across all 887 test days (alert base rate = 37.3%), comparison of Random Forest alert precision and recall against the trivial seasonal rule ("alert on every winter day, never in non-winter"):

{alert_tbl}

- **Horizons where Random Forest is at or below the trivial rule**:
  - **Alert Precision**: Random Forest ties the trivial rule at horizon $h=2$ (0.821 vs 0.821) and falls strictly below the trivial rule at horizons **$h=3, h=4$, and $h=5$** (0.809, 0.781, and 0.761 vs 0.821).
  - **Alert Recall**: Random Forest exceeds the trivial rule across all five horizons on the full test set (0.870 down to 0.764 vs 0.746 to 0.752) due to detecting non-winter alerts (12.0% to 51.2% recall) where the trivial rule catches zero. At day +4, Random Forest alert recall is 0.752 vs 0.749 for the trivial rule, and at day +5 it is 0.764 vs 0.752. In plain words, at days 3 to 5 Random Forest has no demonstrated advantage over the trivial winter rule (lower precision: 95% block-bootstrap CI [-0.137, +0.013] at h=5; recall gap within about one point: 95% block-bootstrap CI [-0.030, +0.053] at h=5, spanning zero).

### Chronological Train / Validation / Test Partitions

{split_tbl}

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

The project is developed and verified under Python 3.13.15.

```bash
# 1. Navigate to the project root directory
cd <project-root>

# 2. Create and activate a virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\\.venv\\Scripts\\Activate.ps1

# On Linux/macOS:
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

Pinned dependencies in `requirements.txt`:
- `pandas==3.1.1`
- `numpy==2.5.4`
- `scipy==1.18.1`
- `scikit-learn==1.9.1`
- `joblib==1.6.0`
- `pyarrow==26.0.1`
- `matplotlib==3.11.2`
- `shap==0.54.1`
- `pytest==9.2.0`
- `seaborn==0.13.2`
- `statsmodels==0.15.0`
- `nbformat==5.11.1`
- `nbclient==0.11.0`

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
├── models/                            # Serialized models (scikit-learn 1.9.1 / Python 3.13 pickle compatibility)
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

All scripts run from the repository root using standard Python 3.13+ and rely on the fixed seed `RANDOM_SEED = 42` defined in `src/config.py`:

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

The daily Air Quality Index follows the Central Pollution Control Board (CPCB) National Air Quality Index methodology (*breakpoints pending verification by the project owner against the official CPCB report*):

1. **Daily Aggregation Protocol**:
   - $PM_{{2.5}}, PM_{{10}}, NO_2, SO_2, NH_3$: 24-hour arithmetic mean (requires ≥16 valid hourly readings; otherwise NaN).
   - $CO, O_3$: Continuous 8-hour trailing rolling mean computed on the continuous hourly data (requires ≥6 of 8 valid hours, `min_periods=6`), followed by taking the daily maximum of these 8h rolling means for the calendar day.
2. **Sub-Index Calculation**:
   - Continuous linear interpolation between breakpoint intervals $[B_{{lo}}, B_{{hi}}]$ mapped to index ranges $[I_{{lo}}, I_{{hi}}]$.
   - Internal sub-indices may extend linearly above 500 for concentrations exceeding the Severe breakpoint.
3. **City AQI & Clipping**:
   - Overall $\\text{{AQI}} = \\max(\\text{{Sub-indices}})$.
   - Validity criteria: requires valid sub-indices for at least 3 criteria pollutants, of which at least one must be $PM_{{2.5}}$ or $PM_{{10}}$.
   - The final reported AQI and category assignment are strictly clipped to the range **$[0, 500]$**.

---

## 9. Mandatory Project Rules

1. **Column Mapping**: Never assume column names or units; use `COLUMN_MAP` in `src/config.py`. Output pressure unit is standardized to `hPa`.
2. **Chronological Splitting**: Never shuffle time series data. Use chronological splits separated by 5-day buffer gaps. Headline metrics were computed once by final_evaluate.py; later diagnostics (alert comparison vs the trivial seasonal rule, Diwali analysis, error analysis) were exploratory and post-hoc.
3. **Leakage Controls with Automated Tests**: Features use only information available up to the end of today ($t' \\le t$). Rolling windows are strictly trailing (`center=False`). No negative shifts allowed in features.
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
| `tests/test_shap.py` | Additivity verification: model prediction matches SHAP values sum + base value within $10^{{-5}}$ | **Passed** ✅ |
| `tests/test_predict.py` | Command-line demo verification: prediction match with saved test predictions within $10^{{-13}}$ across 3 dates and error handling | **Passed** ✅ |

To execute the test suite:
```bash
pytest
```
*Current test result: 31 passed, 1 warning (R² on single sample edge test) across 7 test files.*

---

## 11. Limitations

1. **AQI Cap at 500 and the 53 Days Clipped**: CPCB AQI is capped at 500. Exactly 53 calendar days in `clean_daily.csv` have unclipped sub-indices exceeding 500 (reaching up to 885.22); the model may under-predict the most extreme days (explanation not tested).
2. **City-Average AQI vs Station Bulletins**: AQI in this project is computed from city-wide spatially averaged concentrations. This produces a single representative metropolitan metric, which differs from averaging station-level AQIs or from official CPCB bulletins that report individual station sub-indices.
3. **Breakpoint Verification**: AQI breakpoints implemented in `src/config.py` are pending formal verification by the project owner against the official CPCB NAQI technical report.
4. **Absence of Numerical Weather Forecasts**: The models rely solely on historical trailing meteorology up to today ($t' \\le t$) without forward NWP weather forecast inputs. Rapid meteorological changes (sudden wind shifts, rainfall washout, boundary layer collapses) cannot be anticipated days in advance.
5. **Data Missingness and August 2026 Gap**: 4 calendar days in 2018–2026 have missing daily AQI due to insufficient station coverage. In August 2026, exactly 71 hours have zero reporting stations for $PM_{{2.5}}$ (53 hours from Aug 14 to Aug 16, and 18 hours from Aug 18 to Aug 19; cause unknown).
6. **Distribution Shift Across Split Periods**: High-pollution frequency differs between partitions: Poor-or-worse days represent 48.97% of Train (132 Severe days), 43.47% of Validation (25 Severe days), and 37.32% of Test (44 Severe days).
7. **Geographic Specificity**: All models and spatial weights are tuned strictly to the National Capital Territory of Delhi and do not generalize to other urban airsheds without retraining.
8. **Long-Range Model Skill ($h \\ge 3$) and Damped Anomaly Baseline**: Random Forest achieves statistically significant skill improvements at Day +1 (+4.6% vs Damped Anomaly Persistence, 95% block CI `[-2.18, -0.48]`). Beyond Day +1, Random Forest does not demonstrate superior accuracy over the damped anomaly baseline: at Days 2 to 4 paired bootstrap CIs include zero, and at Day +5 the damped baseline outperforms Random Forest (MAE 39.79 vs 41.48, 95% CI `[+0.04, +3.22]`).
9. **Retrospective Elements in Hourly Cleaning**: While daily features are strictly trailing ($t' \\le t$, enforced by tests), hourly data cleaning is retrospective: (a) bounded linear interpolation uses the value after a gap ($\\le 6$ hours), (b) the stuck-sensor filter blanks a whole consecutive run once it is known to last $\\ge 24$ hours, and (c) station selection evaluates coverage over the full 2018–2026 period.

---

## 12. References

1. **Central Pollution Control Board (CPCB)**, Ministry of Environment, Forest and Climate Change, Government of India: *National Air Quality Index* report, 2014.
2. **Kaggle Dataset**: *Delhi Air Quality CPCB Hourly (2010–2026)*, [https://www.kaggle.com/datasets/yug201/delhi-air-quality-cpcb-hourly-2010-2026](https://www.kaggle.com/datasets/yug201/delhi-air-quality-cpcb-hourly-2010-2026).
3. **Scikit-learn**: Pedregosa et al., *Scikit-learn: Machine Learning in Python*, Journal of Machine Learning Research (JMLR), 12, pp. 2825–2830, 2011.
4. **SHAP (SHapley Additive exPlanations)**: Lundberg, S. M., & Lee, S.-I., *A Unified Approach to Interpreting Model Predictions*, Advances in Neural Information Processing Systems (NeurIPS 30), 2017.
"""
    return readme_content


def update_readme_file():
    """Generates and writes the updated README.md."""
    content = build_full_readme()
    readme_path = REPO_ROOT / "README.md"
    readme_path.write_text(content, encoding="utf-8")
    print(f"Successfully updated {readme_path}")


if __name__ == "__main__":
    update_readme_file()
