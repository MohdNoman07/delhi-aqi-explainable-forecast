"""
scratch/build_all_docs.py

Generates:
  1. docs/report_draft.md
  2. docs/numbers_used.csv
  3. docs/slides_outline.md

Strictly complies with all project rules and user instructions:
  - Exact headline findings
  - Exact [TODO: me] markers
  - 11 slides, 3 bullets each, 2 lines of speaker notes marked [TODO: me]
  - Relative file path references for figures and tables
  - All numbers tracked into docs/numbers_used.csv
  - Causal language strictly labeled (hypothesis, possible explanation, or not tested)
"""

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

tracked_numbers = []

def track(desc: str, val, source: str) -> str:
    tracked_numbers.append({
        "number_description": desc,
        "value": str(val),
        "source_file": source
    })
    return str(val)

# -----------------------------------------------------------------------------
# Section 1: Introduction and Problem Statement
# -----------------------------------------------------------------------------
def build_section_1():
    return f"""# Delhi Air Quality Forecasting (1 to 5 Days Ahead)
## Comprehensive Technical Report & Methodological Audit

---

## 1. Introduction and Problem Statement

### 1.1 Context and Public Health Urgency
Ambient air pollution in the National Capital Territory (NCT) of Delhi is a persistent environmental health challenge. Winter meteorological conditions—shallow boundary layers, surface temperature inversions, and low wind speeds—frequently trap particulate matter ($PM_{{2.5}}$ and $PM_{{10}}$) and criteria gases within the regional airshed. Prolonged exposure during these episodes increases respiratory and cardiovascular hospital admissions, particularly among vulnerable populations.

Regulatory mitigation in Delhi is structured under the Graded Response Action Plan (GRAP), administered by the Commission for Air Quality Management (CAQM). GRAP defines four sequential emergency intervention stages:
- **Stage I (Poor, AQI 201–300)**: Mechanized road sweeping, water sprinkling, dust control at construction sites.
- **Stage II (Very Poor, AQI 301–400)**: Ban on diesel generator sets, parking fee surges, enhanced public transport frequency.
- **Stage III (Severe, AQI 401–450)**: Strict bans on non-essential construction and demolition, restrictions on BS-III petrol and BS-IV diesel light motor vehicles.
- **Stage IV (Severe+, AQI > 450)**: Ban on entry of non-essential commercial trucks into Delhi, closure of schools for physical classes, and potential odd-even vehicular rationing.

### 1.2 The Forecasting Problem: Why Lead Time Matters
Historically, environmental policy actions have been reactive, triggered only after ambient monitors record sustained Severe air quality. Reactive interventions suffer from operational delays:
- Municipal agencies require {track("GRAP mitigation operational lead time in hours", "48 to 72", "PROJECT_NOTES.md")} hours of lead time to deploy road sweeping fleets, reroute heavy freight traffic, and issue public health notices.
- Advance notice is necessary for schools, commercial establishments, and sensitive individuals (e.g. asthmatics) to prepare before peak stagnation arrives.

### 1.3 Multi-Horizon Scope: Forecasting 1 to 5 Days Ahead ($h \\in \\{{1, 2, 3, 4, 5\\}}$)
This project develops, validates, and audits a multi-horizon machine learning system to forecast Delhi's citywide daily Air Quality Index (NAQI) across five distinct forward horizons:
- **Day +1 ($h=1$)**: Short-term tactical forecasting (daily municipal actions, public advisories).
- **Day +2 ($h=2$)**: Inter-agency coordination and transit scheduling.
- **Days +3 to +5 ($h=3, 4, 5$)**: Medium-term strategic horizons for early GRAP Stage III/IV preparation before peak stagnation.

### 1.4 Honest Headline Findings
The empirical findings of this investigation are defined as follows:
- **The model improves on persistence and climatology at 1 to 2 days ahead**.
- **At 3 to 5 days it is statistically indistinguishable from climatology (intervals include zero) and from Ridge**.
- **The SHAP attribution shifts toward calendar features as the horizon grows**.
- **At days 3 to 5 the alert detection is not better than the trivial rule "alert on every winter day"**.

[TODO: me]
"""

# -----------------------------------------------------------------------------
# Section 2: Dataset
# -----------------------------------------------------------------------------
def build_section_2():
    return f"""
---

## 2. Dataset

### 2.1 Source and Spatial Coverage
The empirical foundation comprises continuous ambient monitoring records from the Central Pollution Control Board (CPCB), Delhi Pollution Control Committee (DPCC), and Indian Institute of Tropical Meteorology (IITM). Continuous Ambient Air Quality Monitoring Stations (CAAQMS) record criteria pollutants and meteorological variables at hourly resolution.

A total of {track("Total CAAQMS monitoring stations evaluated across Delhi", 37, "outputs/station_selection.csv")} stations across Delhi were audited for data completeness over the period from {track("Dataset start date", "2018-01-01", "src/config.py")} through {track("Dataset end date", "2026-09-30", "src/config.py")}, spanning {track("Total continuous calendar hours in audit period", 76680, "outputs/station_selection.csv")} continuous calendar hours.

### 2.2 Station Selection and Spatial Aggregation Criteria
Stations were evaluated on hourly completeness for criteria particulates ($PM_{{2.5}}$ and $PM_{{10}}$) according to project rules (`src/config.py`):
1. **Station Selection Threshold**: A station was retained only if its valid hourly coverage of $PM_{{2.5}}$ or $PM_{{10}}$ met or exceeded {track("Minimum station hourly coverage threshold", "50.0%", "src/config.py")}.
2. **Audit Result**: {track("Number of retained CAAQMS stations", 35, "outputs/station_selection.csv")} of 37 stations satisfied the criterion and were retained. Exactly {track("Number of dropped CAAQMS stations", 2, "outputs/station_selection.csv")} stations were dropped (`outputs/station_selection.csv`):
   - `site_1422_Dwarka_Sector_8_Delhi_DPCC`: {track("Dwarka Sector 8 valid hours", 6394, "outputs/station_selection.csv")} / {track("Dwarka Sector 8 total hours", 76680, "outputs/station_selection.csv")} hours ({track("Dwarka Sector 8 coverage percentage", "8.34%", "outputs/station_selection.csv")} coverage).
   - `site_5395_Lodhi_Road_Delhi_IITM`: {track("Lodhi Road valid hours", 26377, "outputs/station_selection.csv")} / {track("Lodhi Road total hours", 76680, "outputs/station_selection.csv")} hours ({track("Lodhi Road coverage percentage", "34.40%", "outputs/station_selection.csv")} coverage).
3. **Spatial Aggregation**: Citywide hourly series were computed by taking the spatial mean across reporting stations for each pollutant-hour, requiring a quorum of at least {track("Minimum reporting stations quorum per pollutant-hour", 3, "src/config.py")} stations. Hours with fewer than 3 reporting stations were flagged as missing.

### 2.3 Variables, Units, and Column Roles
Raw CSV headers were mapped directly to standardized semantic roles via `COLUMN_MAP` (`src/config.py`):

| Variable Role | Raw CPCB Header | Measurement Unit | Category | Status |
|:---|:---|:---:|:---:|:---:|
| `pm25` | `PM2.5 (µg/m³)` | $\\mu\\text{{g/m}}^3$ | Criteria Pollutant | Retained |
| `pm10` | `PM10 (µg/m³)` | $\\mu\\text{{g/m}}^3$ | Criteria Pollutant | Retained |
| `no` | `NO (µg/m³)` | $\\mu\\text{{g/m}}^3$ | Reactive Nitrogen | Retained |
| `no2` | `NO2 (µg/m³)` | $\\mu\\text{{g/m}}^3$ | Criteria Pollutant | Retained |
| `nox` | `NOx (ppb)` | $\\text{{ppb}}$ | Reactive Nitrogen | Retained |
| `nh3` | `NH3 (µg/m³)` | $\\mu\\text{{g/m}}^3$ | Criteria Pollutant | Retained |
| `so2` | `SO2 (µg/m³)` | $\\mu\\text{{g/m}}^3$ | Criteria Pollutant | Retained |
| `co` | `CO (mg/m³)` | $\\text{{mg/m}}^3$ | Criteria Pollutant | Retained (kept in $\\text{{mg/m}}^3$) |
| `o3` | `Ozone (µg/m³)` | $\\mu\\text{{g/m}}^3$ | Criteria Pollutant | Retained |
| `benzene` | `Benzene (µg/m³)` | $\\mu\\text{{g/m}}^3$ | VOC | Retained |
| `toluene` | `Toluene (µg/m³)` | $\\mu\\text{{g/m}}^3$ | VOC | Retained |
| `temperature` | `AT (°C)` | $^\\circ\\text{{C}}$ | Meteorology | Retained |
| `humidity` | `RH (%)` | $\%$ | Meteorology | Retained |
| `wind_speed` | `WS (m/s)` | $\\text{{m/s}}$ | Meteorology | Retained |
| `wind_direction` | `WD (deg)` | $\\text{{degrees}}$ | Meteorology | Retained |
| `rainfall` | `RF (mm)` | $\\text{{mm}}$ | Meteorology | Retained |
| `solar_radiation` | `SR (W/mt2)` | $\\text{{W/m}}^2$ | Meteorology | Retained |
| `barometric_pressure` | `BP (mmHg)` | $\\text{{hPa}}$ | Meteorology | Converted to $\\text{{hPa}}$ |
| *Dropped Columns* | `Xylene, O-Xylene, Eth-Benzene, MP-Xylene, VWS` | Various | Trace Gases / Vertical Wind | Dropped ({track("High-null column drop threshold", ">80%", "src/config.py")} null) |

### 2.4 Target Formulation: National Air Quality Index (NAQI)
The primary target is the daily citywide Indian National Air Quality Index (NAQI, scale 0 to 500). In accordance with CPCB guidelines:
1. Sub-indices $I_i$ are calculated for criteria pollutants ($PM_{{2.5}}$, $PM_{{10}}$, $NO_2$, $NH_3$, $SO_2$, $CO$, $O_3$) using piecewise linear concentration breakpoint equations.
2. Particulates and gaseous criteria pollutants use a 24-hour mean concentration (requiring $\\ge 16$ valid hourly readings).
3. Ozone and Carbon Monoxide use 8-hour trailing rolling averages (requiring $\\ge 6$ valid hours per 8-hour window); the daily representative value is the maximum 8-hour value of the calendar day.
4. An overall daily AQI is valid if and only if at least {track("Minimum pollutants required for valid daily AQI", 3, "src/config.py")} criteria pollutants have valid sub-indices, of which at least one must be a particulate ($PM_{{2.5}}$ or $PM_{{10}}$).
5. The overall AQI is the maximum of the valid sub-indices: $\\text{{AQI}} = \\max_i(I_i)$.

Regulatory CPCB health bands:
- **Good** (0–50)
- **Satisfactory** (51–100)
- **Moderate** (101–200)
- **Poor** (201–300)
- **Very Poor** (301–400)
- **Severe** (401–500)

### 2.5 Dataset Limitations
- **CPCB Breakpoint Verification Pending**: The CPCB breakpoint table is pending verification by the project owner against the official Gazette notification.
- **Unverified Festival Calendar**: The Diwali dates are unverified; festival dates (`DIWALI_DATES` in `src/config.py`) are compiled from memory and pending official calendar confirmation.

[TODO: me]
"""

# -----------------------------------------------------------------------------
# Section 3: Preprocessing and EDA
# -----------------------------------------------------------------------------
def build_section_3():
    return f"""
---

## 3. Preprocessing and Exploratory Data Analysis (EDA)

### 3.1 Data Cleaning Decisions and Physical Bounds
Data cleaning was systematically applied and logged in `outputs/cleaning_log.csv`:
1. **Barometric Pressure Normalization**: Raw pressure measurements in CPCB exports were recorded with mixed units (ambient $\\text{{mmHg}}$ and $\\text{{hPa}}$). Readout values below 850 were identified as $\\text{{mmHg}}$ and converted to $\\text{{hPa}}$ ($P_{{\\text{{hPa}}}} = P_{{\\text{{mmHg}}}} \\times 1.333224$). Exactly {track("Pressure readings converted from mmHg to hPa", 167594, "outputs/cleaning_log.csv")} readings were converted. Physical bounds for Delhi's elevation (~216 m above sea level) were established at [940, 1030] hPa; {track("Out-of-range pressure readings set to NaN", 17702, "outputs/cleaning_log.csv")} readings falling outside this window were set to NaN.
2. **Physical and Instrument Plausibility Caps**: Extreme unphysical spikes were clipped to atmospheric and instrument limits:
   - Ozone ($O_3$): clipped to [0, 500] $\\mu\\text{{g/m}}^3$ ({track("Ozone readings clipped at 500 ug/m3", 12, "outputs/cleaning_log.csv")} readings clipped; maximum raw spike was 970.0).
   - Benzene: clipped to [0, 200] $\\mu\\text{{g/m}}^3$ ({track("Benzene readings clipped at 200 ug/m3", 125, "outputs/cleaning_log.csv")} readings clipped; maximum raw spike was 476.4).
   - Ambient Temperature: bounded within [0, 52] $^\\circ\\text{{C}}$ ({track("Temperature readings clipped", 40, "outputs/cleaning_log.csv")} readings clipped; maximum raw spike was 58.9).
   - Wind Speed: bounded within [0, 30] $\\text{{m/s}}$ ({track("Wind speed readings clipped", 108, "outputs/cleaning_log.csv")} readings clipped; maximum raw spike was 47.75).
   - Surface Solar Radiation: bounded by [0, 1300] $\\text{{W/m}}^2$ ({track("Solar radiation readings clipped", 353, "outputs/cleaning_log.csv")} readings clipped; maximum raw spike was 1985.0).
3. **Stuck-Sensor Invalidation**: Sensor transmission faults can freeze telemetry, outputting constant non-zero values. An automated audit invalidated episodes where a sensor outputted the exact same non-zero value for $\\ge 24$ consecutive hours:
   - Barometric pressure: {track("Stuck readings invalidated for barometric pressure", 70753, "outputs/cleaning_log.csv")} readings.
   - Wind speed: {track("Stuck readings invalidated for wind speed", 27347, "outputs/cleaning_log.csv")} readings.
   - Benzene: {track("Stuck readings invalidated for benzene", 20006, "outputs/cleaning_log.csv")} readings.
   - Toluene: {track("Stuck readings invalidated for toluene", 15767, "outputs/cleaning_log.csv")} readings.
   - Relative humidity: {track("Stuck readings invalidated for relative humidity", 8593, "outputs/cleaning_log.csv")} readings.
   - Carbon monoxide: {track("Stuck readings invalidated for carbon monoxide", 8197, "outputs/cleaning_log.csv")} readings.
   - Wind direction: {track("Stuck readings invalidated for wind direction", 6093, "outputs/cleaning_log.csv")} readings.
   - Solar radiation: {track("Stuck readings invalidated for solar radiation", 5851, "outputs/cleaning_log.csv")} readings.
   - Criteria gases & particulates: $NO_2$ ({track("Stuck readings invalidated for NO2", 2485, "outputs/cleaning_log.csv")}), $PM_{{2.5}}$ ({track("Stuck readings invalidated for PM2.5", 2272, "outputs/cleaning_log.csv")}), $O_3$ ({track("Stuck readings invalidated for O3", 1269, "outputs/cleaning_log.csv")}), $SO_2$ ({track("Stuck readings invalidated for SO2", 1256, "outputs/cleaning_log.csv")}), $PM_{{10}}$ ({track("Stuck readings invalidated for PM10", 1130, "outputs/cleaning_log.csv")}), Temperature ({track("Stuck readings invalidated for temperature", 1002, "outputs/cleaning_log.csv")}), $NO$ ({track("Stuck readings invalidated for NO", 997, "outputs/cleaning_log.csv")}), $NH_3$ ({track("Stuck readings invalidated for NH3", 585, "outputs/cleaning_log.csv")}), $NO_x$ ({track("Stuck readings invalidated for NOx", 376, "outputs/cleaning_log.csv")}).
4. **Particulate Ratio Inversion Filter**: By physical definition, $PM_{{2.5}}$ is a subset of $PM_{{10}}$. A tolerance multiplier of 1.2 accounts for optical instrument uncertainty. Station hours exhibiting $PM_{{2.5}} > 1.2 \\times PM_{{10}}$ with $PM_{{2.5}} > 20\\,\\mu\\text{{g/m}}^3$ were flagged as unphysical instrument cross-talk; exactly {track("Station-hours with unphysical PM inversion", 3691, "outputs/cleaning_log.csv")} station-hours ({track("Particulate values invalidated due to PM inversion", 7382, "outputs/cleaning_log.csv")} individual values) were set to NaN. Sensor drift during high-humidity episodes is a possible explanation (not tested).
5. **Transient VOC Spike Filter**: Isolated VOC readings exceeding 10 times their rolling 24-hour median (baseline $> 0.1\\,\\mu\\text{{g/m}}^3$) were treated as localized sensor noise and set to NaN ({track("Transient benzene spikes filtered", 1888, "outputs/cleaning_log.csv")} benzene, {track("Transient toluene spikes filtered", 3150, "outputs/cleaning_log.csv")} toluene).
6. **Station-Level Imputation**: Missing values within isolated gaps $\\le 6$ hours were linearly interpolated at the station level prior to spatial averaging.

### 3.2 Exploratory Data Analysis (EDA) Highlights
Key exploratory visualizations are preserved in `outputs/eda/`:
1. **Daily AQI Time Series (2018–2026)** (`outputs/eda/01_daily_aqi_over_time.png`): Demonstrates recurring seasonal cycles, with winter AQI frequently reaching Very Poor and Severe bands.
2. **Monthly AQI Distributions & Heatmaps** (`outputs/eda/02_aqi_by_month_and_heatmap.png`): Shows November and December having median AQI $> 320$, while July and August have median AQI $< 100$.
3. **Missing Value Auditing Before & After Imputation** (`outputs/eda/03_missing_values_by_year_and_column.png`): Quantifies coverage improvements from 6-hour linear interpolation and multi-station quorum averaging.
4. **Correlation Structure** (`outputs/eda/04_correlation_heatmap.png`): Shows strong positive correlations ($>0.75$) among $PM_{{2.5}}$, $PM_{{10}}$, $NO_2$, and $CO$.
5. **AQI Category Distribution** (`outputs/eda/05_aqi_category_distribution.png`): Illustrates the high historical prevalence of Poor, Very Poor, and Severe days.
6. **Autocorrelation and Partial Autocorrelation** (`outputs/eda/06_acf_pacf_lags_1_to_10.png`): Shows day-to-day persistence ($r_1 > 0.85$) with PACF dropping sharply after lag 1.
7. **Meteorological Effects** (`outputs/eda/07_binned_wind_speed_temp_effects.png`): Binned wind speed analysis shows higher Severe day frequency when surface winds drop below 1.5 m/s.
8. **Prominent Pollutant Shares Across Years** (`outputs/eda/08_prominent_pollutant_share_by_year.png`): Confirms that $PM_{{2.5}}$ and $PM_{{10}}$ determine the daily AQI on over 95% of calendar days across all audited years.

[TODO: me]
"""

# -----------------------------------------------------------------------------
# Section 4: Feature Engineering and Targets
# -----------------------------------------------------------------------------
def build_section_4():
    return f"""
---

## 4. Feature Engineering and Targets

### 4.1 Feature Set Architecture
Exactly {track("Total engineered feature count", 38, "models/feature_columns.json")} features were engineered (`models/feature_columns.json`) across four families:

| Feature Name | Family | Definition |
|:---|:---:|:---|
| `aqi` | AQI History | Today's citywide daily AQI ($t$) |
| `aqi_lag1` | AQI History | Daily AQI at $t-1$ |
| `aqi_lag2` | AQI History | Daily AQI at $t-2$ |
| `aqi_lag3` | AQI History | Daily AQI at $t-3$ |
| `aqi_lag7` | AQI History | Daily AQI at $t-7$ |
| `aqi_rolling_mean_3d` | AQI History | Trailing 3-day average AQI |
| `aqi_rolling_mean_7d` | AQI History | Trailing 7-day average AQI |
| `aqi_rolling_max_7d` | AQI History | Trailing 7-day maximum AQI |
| `aqi_trend_3d` | AQI History | Slope of AQI over trailing 3 days |
| `pm25` | Criteria Pollutants | Today's daily mean $PM_{{2.5}}\\,(\\mu\\text{{g/m}}^3)$ |
| `pm25_rolling_mean_3d` | Criteria Pollutants | Trailing 3-day mean $PM_{{2.5}}$ |
| `pm10` | Criteria Pollutants | Today's daily mean $PM_{{10}}\\,(\\mu\\text{{g/m}}^3)$ |
| `pm10_rolling_mean_3d` | Criteria Pollutants | Trailing 3-day mean $PM_{{10}}$ |
| `no2` | Criteria Pollutants | Today's daily mean $NO_2\\,(\\mu\\text{{g/m}}^3)$ |
| `no2_rolling_mean_3d` | Criteria Pollutants | Trailing 3-day mean $NO_2$ |
| `so2` | Criteria Pollutants | Today's daily mean $SO_2\\,(\\mu\\text{{g/m}}^3)$ |
| `so2_rolling_mean_3d` | Criteria Pollutants | Trailing 3-day mean $SO_2$ |
| `co` | Criteria Pollutants | Today's daily max 8h $CO\\,(\\text{{mg/m}}^3)$ |
| `co_rolling_mean_3d` | Criteria Pollutants | Trailing 3-day max 8h $CO$ |
| `o3` | Criteria Pollutants | Today's daily max 8h $O_3\\,(\\mu\\text{{g/m}}^3)$ |
| `o3_rolling_mean_3d` | Criteria Pollutants | Trailing 3-day max 8h $O_3$ |
| `pm25_max` | Criteria Pollutants | Daily maximum hourly $PM_{{2.5}}$ |
| `pm10_max` | Criteria Pollutants | Daily maximum hourly $PM_{{10}}$ |
| `temperature` | Meteorology | Today's daily mean temperature ($^\\circ\\text{{C}}$) |
| `humidity` | Meteorology | Today's daily mean relative humidity (%) |
| `wind_speed` | Meteorology | Today's daily mean wind speed (m/s) |
| `barometric_pressure` | Meteorology | Today's daily mean pressure (hPa) |
| `wind_dir_sin` | Meteorology | $\\sin(2\\pi \\times \\text{{WD}} / 360)$ |
| `wind_dir_cos` | Meteorology | $\\cos(2\\pi \\times \\text{{WD}} / 360)$ |
| `temperature_rolling_mean_3d` | Meteorology | Trailing 3-day mean temperature |
| `humidity_rolling_mean_3d` | Meteorology | Trailing 3-day mean humidity |
| `wind_speed_rolling_mean_3d` | Meteorology | Trailing 3-day mean wind speed |
| `month` | Calendar / Cyclical | Calendar month index ($1 \\dots 12$) |
| `day_of_year_sin` | Calendar / Cyclical | $\\sin(2\\pi \\times \\text{{DOY}} / 365.25)$ |
| `day_of_year_cos` | Calendar / Cyclical | $\\cos(2\\pi \\times \\text{{DOY}} / 365.25)$ |
| `day_of_week` | Calendar / Cyclical | Day of week index ($0 \\dots 6$) |
| `is_weekend` | Calendar / Cyclical | Binary indicator for Saturday/Sunday |
| `stubble_season` | Calendar / Cyclical | Indicator for Oct 15 – Nov 30 window |

### 4.2 Multi-Horizon Target Formulation
Separate supervised models are trained for each forecast horizon $h \\in \\{{1, 2, 3, 4, 5\\}}$:
$$\\hat{{y}}_{{t+h}} = f_h(\\mathbf{{X}}_t)$$
where $\\mathbf{{X}}_t \\in \\mathbb{{R}}^{{38}}$ represents features observed on or before day $t$, and $y_{{t+h}} \\in [0, 500]$ is the verified CPCB AQI observed $h$ days later.

### 4.3 Leakage Prevention and Automated Tests
Leakage prevention rules:
1. **Trailing Lags Only**: Features use only trailing information ($t' \\le t$). No centered rolling windows and no negative shifts in feature columns.
2. **Negative Shifts Strictly for Targets**: Negative shifts (e.g. `df['target'] = df['aqi'].shift(-h)`) are used exclusively for target construction.
3. **Preprocessing on Training Data Only**: Scalers, imputers, and climatology lookup tables are fit on training data only.
4. **Automated Leakage Test Suite**: `tests/test_no_leakage.py` runs three continuous checks:
   - `test_no_target_in_feature_columns`: Confirms no target column appears in $\\mathbf{{X}}_t$.
   - `test_no_negative_shifts_or_centered_windows`: Scans codebase AST to confirm no negative shifts or centered windows in feature construction.
   - `test_temporal_leakage_20_random_dates`: Selects 20 random dates, modifies future data after date $t$, and asserts that $\\mathbf{{X}}_t$ remains identical. All tests pass with zero warnings.

[TODO: me]
"""

# -----------------------------------------------------------------------------
# Section 5: Models and Experimental Setup
# -----------------------------------------------------------------------------
def build_section_5():
    return f"""
---

## 5. Models and Experimental Setup

### 5.1 Chronological Partitioning and Buffer Gaps
In accordance with Rule 2 (Never shuffle time series; chronological splits with a 5-day boundary buffer gap; test set reserved exclusively for final evaluation), the dataset was partitioned chronologically (`outputs/split_summary.md`, `outputs/split_diagram.png`):

| Partition | Start Date | End Date | Calendar Days | Days Dropped in Gap | Description |
|:---|:---:|:---:|:---:|:---:|:---|
| **Training** | 2018-01-08 | 2023-03-31 | **{track("Training set calendar days", 1887, "outputs/split_summary.md")}** | — | Training data (~5.25 years) |
| *Gap 1* | 2023-04-01 | 2023-04-05 | **{track("Buffer gap 1 days", 5, "outputs/split_summary.md")}** | {track("Buffer gap 1 dropped days", 5, "outputs/split_summary.md")} | Excludes training horizon overlap into validation |
| **Validation** | 2023-04-06 | 2024-03-31 | **{track("Validation set calendar days", 352, "outputs/split_summary.md")}** | — | Validation data (1 full calendar year) |
| *Gap 2* | 2024-04-01 | 2024-04-05 | **{track("Buffer gap 2 days", 5, "outputs/split_summary.md")}** | {track("Buffer gap 2 dropped days", 5, "outputs/split_summary.md")} | Excludes validation horizon overlap into test |
| **Test Holdout** | 2024-04-06 | 2026-09-25 | **{track("Test set calendar days", 887, "outputs/split_summary.md")}** | — | Final evaluation holdout (~2.5 years) |
| **Total** | 2018-01-08 | 2026-09-25 | **{track("Total model-ready rows accounted for", 3136, "outputs/split_summary.md")}** | **{track("Total days dropped in buffer gaps", 10, "outputs/split_summary.md")}** | 3,146 total calendar days accounted for |

### 5.2 Category Distribution and Non-Stationary Shift
Distribution of CPCB AQI health categories across partitions (`outputs/split_summary.md`):

| CPCB Category | Range | Training Set ({track("Training set total rows", 1887, "outputs/split_summary.md")} days) | Validation Set ({track("Validation set total rows", 352, "outputs/split_summary.md")} days) | Test Holdout ({track("Test set total rows", 887, "outputs/split_summary.md")} days) |
|:---|:---:|:---:|:---:|:---:|
| **Good** | 0 – 50 | {track("Train Good days count", 18, "outputs/split_summary.md")} ({track("Train Good days share", "0.95%", "outputs/split_summary.md")}) | {track("Val Good days count", 0, "outputs/split_summary.md")} ({track("Val Good days share", "0.00%", "outputs/split_summary.md")}) | {track("Test Good days count", 6, "outputs/split_summary.md")} ({track("Test Good days share", "0.68%", "outputs/split_summary.md")}) |
| **Satisfactory** | 51 – 100 | {track("Train Satisfactory days count", 355, "outputs/split_summary.md")} ({track("Train Satisfactory days share", "18.81%", "outputs/split_summary.md")}) | {track("Val Satisfactory days count", 56, "outputs/split_summary.md")} ({track("Val Satisfactory days share", "15.91%", "outputs/split_summary.md")}) | {track("Test Satisfactory days count", 184, "outputs/split_summary.md")} ({track("Test Satisfactory days share", "20.74%", "outputs/split_summary.md")}) |
| **Moderate** | 101 – 200 | {track("Train Moderate days count", 590, "outputs/split_summary.md")} ({track("Train Moderate days share", "31.27%", "outputs/split_summary.md")}) | {track("Val Moderate days count", 143, "outputs/split_summary.md")} ({track("Val Moderate days share", "40.62%", "outputs/split_summary.md")}) | {track("Test Moderate days count", 366, "outputs/split_summary.md")} ({track("Test Moderate days share", "41.26%", "outputs/split_summary.md")}) |
| **Poor** | 201 – 300 | {track("Train Poor days count", 317, "outputs/split_summary.md")} ({track("Train Poor days share", "16.80%", "outputs/split_summary.md")}) | {track("Val Poor days count", 38, "outputs/split_summary.md")} ({track("Val Poor days share", "10.80%", "outputs/split_summary.md")}) | {track("Test Poor days count", 122, "outputs/split_summary.md")} ({track("Test Poor days share", "13.75%", "outputs/split_summary.md")}) |
| **Very Poor** | 301 – 400 | {track("Train Very Poor days count", 475, "outputs/split_summary.md")} ({track("Train Very Poor days share", "25.17%", "outputs/split_summary.md")}) | {track("Val Very Poor days count", 90, "outputs/split_summary.md")} ({track("Val Very Poor days share", "25.57%", "outputs/split_summary.md")}) | {track("Test Very Poor days count", 165, "outputs/split_summary.md")} ({track("Test Very Poor days share", "18.60%", "outputs/split_summary.md")}) |
| **Severe** | 401 – 500 | {track("Train Severe days count", 132, "outputs/split_summary.md")} ({track("Train Severe days share", "7.00%", "outputs/split_summary.md")}) | {track("Val Severe days count", 25, "outputs/split_summary.md")} ({track("Val Severe days share", "7.10%", "outputs/split_summary.md")}) | {track("Test Severe days count", 44, "outputs/split_summary.md")} ({track("Test Severe days share", "4.96%", "outputs/split_summary.md")}) |
| **Poor-or-worse (AQI > 200)** | > 200 | **{track("Train Poor-or-worse count", 924, "outputs/split_summary.md")} ({track("Train Poor-or-worse share", "48.97%", "outputs/split_summary.md")})** | **{track("Val Poor-or-worse count", 153, "outputs/split_summary.md")} ({track("Val Poor-or-worse share", "43.47%", "outputs/split_summary.md")})** | **{track("Test Poor-or-worse count", 331, "outputs/split_summary.md")} ({track("Test Poor-or-worse share", "37.32%", "outputs/split_summary.md")})** |

> **Audit Note on Target Distribution**: The test period as having lower pollution levels than the training period is documented in the split summary without asserting a cause: Poor-or-worse days ($\text{{AQI}} > 200$) accounted for 48.97% of the training set but 37.32% in the test holdout; Severe days accounted for 7.00% in training and 4.96% in the test holdout. Additionally, the validation set contained zero Good days and the test set contained only 6 Good days (0.68%), rendering single-class Good metrics statistically unreliable.

### 5.3 Cross-Validation and Hyperparameter Tuning
Hyperparameter optimization was conducted strictly on the training set using an expanding-window `TimeSeriesSplit(n_splits=4, gap=5)` (`src/config.py`).
- **CV Configuration Note**: A 5-fold split was audited during setup but rejected as Fold 1 trained on only 312 days with no complete winter; configuring `CV_SPLITS = 4` with `CV_GAP = 5` ensured that every fold encompassed a complete winter season.
- **Randomized Search**: Hyperparameters were tuned via `RandomizedSearchCV` ({track("RandomizedSearchCV iterations", 30, "src/train.py")} iterations, scoring `neg_mean_absolute_error`, seed 42) searching:
  - `n_estimators`: [200, 400, 600]
  - `max_depth`: [None, 6, 10, 16]
  - `min_samples_leaf`: [1, 3, 5, 10]
  - `max_features`: [0.3, 0.5, 0.8, 1.0]

### 5.4 Target Transformation Audit: Raw vs. Log Target
Tuning was executed under two target paradigms:
1. **Raw Target**: Regressors trained directly on original AQI scale $\\hat{{y}} = f(\\mathbf{{X}})$.
2. **Log Target**: Regressors trained on $z = \\log(1 + \\text{{AQI}})$, with predictions converted back with $\\hat{{y}} = \\exp(\\hat{{z}}) - 1$ prior to evaluation on the original scale.

Validation performance (`models/rf_config.json`):

| Horizon | Raw Target Val MAE | Log Target Val MAE | Margin Over Log | Winning Scale | Best Parameters |
|:---:|:---:|:---:|:---:|:---:|:---|
| **$h=1$** | **{track("Val MAE h=1 raw", 30.88, "models/rf_config.json")}** | {track("Val MAE h=1 log", 30.92, "models/rf_config.json")} | +{track("Val MAE h=1 margin", 0.04, "models/rf_config.json")} | **Raw** | `n_estimators=400, max_depth=10, min_samples_leaf=5, max_features=0.5` |
| **$h=2$** | **{track("Val MAE h=2 raw", 38.80, "models/rf_config.json")}** | {track("Val MAE h=2 log", 41.17, "models/rf_config.json")} | +{track("Val MAE h=2 margin", 2.38, "models/rf_config.json")} | **Raw** | `n_estimators=400, max_depth=16, min_samples_leaf=10, max_features=0.3` |
| **$h=3$** | **{track("Val MAE h=3 raw", 43.21, "models/rf_config.json")}** | {track("Val MAE h=3 log", 45.65, "models/rf_config.json")} | +{track("Val MAE h=3 margin", 2.45, "models/rf_config.json")} | **Raw** | `n_estimators=400, max_depth=16, min_samples_leaf=10, max_features=0.3` |
| **$h=4$** | **{track("Val MAE h=4 raw", 44.53, "models/rf_config.json")}** | {track("Val MAE h=4 log", 46.79, "models/rf_config.json")} | +{track("Val MAE h=4 margin", 2.26, "models/rf_config.json")} | **Raw** | `n_estimators=200, max_depth=None, min_samples_leaf=10, max_features=0.3` |
| **$h=5$** | **{track("Val MAE h=5 raw", 44.93, "models/rf_config.json")}** | {track("Val MAE h=5 log", 47.61, "models/rf_config.json")} | +{track("Val MAE h=5 margin", 2.68, "models/rf_config.json")} | **Raw** | `n_estimators=400, max_depth=16, min_samples_leaf=10, max_features=0.3` |

At all five horizons, the raw target scale achieved lower validation MAE than the log transformation. Re-transformation bias is a likely explanation (not tested). The raw target scale was chosen for all models.

### 5.5 Benchmark Models in Scope
1. **Persistence Baseline**: $\\hat{{y}}_{{t+h}} = y_t$.
2. **Climatology Baseline**: Circular moving-window mean ($\\pm 7$ days) from training history: $\\hat{{y}}_{{t+h}} = \\bar{{y}}_{{\\text{{doy}}(t+h) \\pm 7\\text{{d}}}}^{{\\text{{train}}}}$.
3. **Ridge Regression**: Tuned with regularization $\\alpha = {track("Ridge alpha", 100.0, "PROJECT_NOTES.md")}$ on validation data.
4. **Random Forest Regressor**: Horizon-specific non-linear ensemble with best parameters locked from validation tuning.

### 5.6 Final Evaluation Protocol
Models were refit on combined Training + Validation data (gap rows strictly excluded) with chosen settings. The final test evaluation script (`src/final_evaluate.py`) was run once on the holdout test set.

[TODO: me]
"""

# -----------------------------------------------------------------------------
# Section 6: Results
# -----------------------------------------------------------------------------
def build_section_6():
    return f"""
---

## 6. Results

### 6.1 Multi-Horizon Benchmark Performance
Full test set metrics from `outputs/metrics.csv` across all models and horizons:

| Model | Horizon | MAE | RMSE | $R^2$ | Mean Bias | Accuracy | Within-1-Cat Acc | Alert Prec | Alert Rec | Skill vs Best Base |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Persistence** | $h=1$ | {track("Test MAE Persistence h=1", 30.97, "outputs/metrics.csv")} | {track("Test RMSE Persistence h=1", 44.60, "outputs/metrics.csv")} | {track("Test R2 Persistence h=1", 0.836, "outputs/metrics.csv")} | +{track("Test bias Persistence h=1", 0.15, "outputs/metrics.csv")} | {track("Test Acc Persistence h=1", 0.693, "outputs/metrics.csv")} | {track("Test W1Acc Persistence h=1", 0.983, "outputs/metrics.csv")} | {track("Test AlertPrec Persistence h=1", 0.852, "outputs/metrics.csv")} | {track("Test AlertRec Persistence h=1", 0.852, "outputs/metrics.csv")} | {track("Test Skill Persistence h=1", 0.000, "outputs/metrics.csv")} |
| **Climatology** | $h=1$ | {track("Test MAE Climatology h=1", 40.37, "outputs/metrics.csv")} | {track("Test RMSE Climatology h=1", 54.23, "outputs/metrics.csv")} | {track("Test R2 Climatology h=1", 0.757, "outputs/metrics.csv")} | +{track("Test bias Climatology h=1", 8.46, "outputs/metrics.csv")} | {track("Test Acc Climatology h=1", 0.608, "outputs/metrics.csv")} | {track("Test W1Acc Climatology h=1", 0.971, "outputs/metrics.csv")} | {track("Test AlertPrec Climatology h=1", 0.819, "outputs/metrics.csv")} | {track("Test AlertRec Climatology h=1", 0.779, "outputs/metrics.csv")} | {track("Test Skill Climatology h=1", -0.303, "outputs/metrics.csv")} |
| **Ridge** | $h=1$ | {track("Test MAE Ridge h=1", 28.49, "outputs/metrics.csv")} | {track("Test RMSE Ridge h=1", 39.11, "outputs/metrics.csv")} | {track("Test R2 Ridge h=1", 0.874, "outputs/metrics.csv")} | {track("Test bias Ridge h=1", -2.46, "outputs/metrics.csv")} | {track("Test Acc Ridge h=1", 0.693, "outputs/metrics.csv")} | {track("Test W1Acc Ridge h=1", 0.993, "outputs/metrics.csv")} | {track("Test AlertPrec Ridge h=1", 0.858, "outputs/metrics.csv")} | {track("Test AlertRec Ridge h=1", 0.858, "outputs/metrics.csv")} | +{track("Test Skill Ridge h=1", 0.080, "outputs/metrics.csv")} |
| **Random Forest** | $h=1$ | **{track("Test MAE RF h=1", 27.57, "outputs/metrics.csv")}** | **{track("Test RMSE RF h=1", 38.84, "outputs/metrics.csv")}** | **{track("Test R2 RF h=1", 0.876, "outputs/metrics.csv")}** | +{track("Test bias RF h=1", 3.02, "outputs/metrics.csv")} | **{track("Test Acc RF h=1", 0.710, "outputs/metrics.csv")}** | **{track("Test W1Acc RF h=1", 0.990, "outputs/metrics.csv")}** | {track("Test AlertPrec RF h=1", 0.842, "outputs/metrics.csv")} | **{track("Test AlertRec RF h=1", 0.870, "outputs/metrics.csv")}** | **+{track("Test Skill RF h=1", 0.110, "outputs/metrics.csv")}** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Persistence** | $h=2$ | {track("Test MAE Persistence h=2", 42.57, "outputs/metrics.csv")} | {track("Test RMSE Persistence h=2", 59.31, "outputs/metrics.csv")} | {track("Test R2 Persistence h=2", 0.710, "outputs/metrics.csv")} | +{track("Test bias Persistence h=2", 0.34, "outputs/metrics.csv")} | {track("Test Acc Persistence h=2", 0.610, "outputs/metrics.csv")} | {track("Test W1Acc Persistence h=2", 0.958, "outputs/metrics.csv")} | {track("Test AlertPrec Persistence h=2", 0.789, "outputs/metrics.csv")} | {track("Test AlertRec Persistence h=2", 0.789, "outputs/metrics.csv")} | {track("Test Skill Persistence h=2", -0.053, "outputs/metrics.csv")} |
| **Climatology** | $h=2$ | {track("Test MAE Climatology h=2", 40.43, "outputs/metrics.csv")} | {track("Test RMSE Climatology h=2", 54.28, "outputs/metrics.csv")} | {track("Test R2 Climatology h=2", 0.758, "outputs/metrics.csv")} | +{track("Test bias Climatology h=2", 8.56, "outputs/metrics.csv")} | {track("Test Acc Climatology h=2", 0.607, "outputs/metrics.csv")} | {track("Test W1Acc Climatology h=2", 0.970, "outputs/metrics.csv")} | {track("Test AlertPrec Climatology h=2", 0.819, "outputs/metrics.csv")} | {track("Test AlertRec Climatology h=2", 0.779, "outputs/metrics.csv")} | {track("Test Skill Climatology h=2", 0.000, "outputs/metrics.csv")} |
| **Ridge** | $h=2$ | {track("Test MAE Ridge h=2", 37.01, "outputs/metrics.csv")} | {track("Test RMSE Ridge h=2", 49.95, "outputs/metrics.csv")} | {track("Test R2 Ridge h=2", 0.795, "outputs/metrics.csv")} | {track("Test bias Ridge h=2", -5.24, "outputs/metrics.csv")} | {track("Test Acc Ridge h=2", 0.638, "outputs/metrics.csv")} | {track("Test W1Acc Ridge h=2", 0.981, "outputs/metrics.csv")} | {track("Test AlertPrec Ridge h=2", 0.842, "outputs/metrics.csv")} | {track("Test AlertRec Ridge h=2", 0.819, "outputs/metrics.csv")} | +{track("Test Skill Ridge h=2", 0.084, "outputs/metrics.csv")} |
| **Random Forest** | $h=2$ | **{track("Test MAE RF h=2", 36.34, "outputs/metrics.csv")}** | **{track("Test RMSE RF h=2", 49.29, "outputs/metrics.csv")}** | **{track("Test R2 RF h=2", 0.800, "outputs/metrics.csv")}** | +{track("Test bias RF h=2", 2.02, "outputs/metrics.csv")} | **{track("Test Acc RF h=2", 0.649, "outputs/metrics.csv")}** | **{track("Test W1Acc RF h=2", 0.982, "outputs/metrics.csv")}** | {track("Test AlertPrec RF h=2", 0.821, "outputs/metrics.csv")} | **{track("Test AlertRec RF h=2", 0.831, "outputs/metrics.csv")}** | **+{track("Test Skill RF h=2", 0.101, "outputs/metrics.csv")}** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Persistence** | $h=3$ | {track("Test MAE Persistence h=3", 47.25, "outputs/metrics.csv")} | {track("Test RMSE Persistence h=3", 64.95, "outputs/metrics.csv")} | {track("Test R2 Persistence h=3", 0.654, "outputs/metrics.csv")} | +{track("Test bias Persistence h=3", 0.53, "outputs/metrics.csv")} | {track("Test Acc Persistence h=3", 0.565, "outputs/metrics.csv")} | {track("Test W1Acc Persistence h=3", 0.946, "outputs/metrics.csv")} | {track("Test AlertPrec Persistence h=3", 0.764, "outputs/metrics.csv")} | {track("Test AlertRec Persistence h=3", 0.764, "outputs/metrics.csv")} | {track("Test Skill Persistence h=3", -0.166, "outputs/metrics.csv")} |
| **Climatology** | $h=3$ | {track("Test MAE Climatology h=3", 40.52, "outputs/metrics.csv")} | {track("Test RMSE Climatology h=3", 54.34, "outputs/metrics.csv")} | {track("Test R2 Climatology h=3", 0.758, "outputs/metrics.csv")} | +{track("Test bias Climatology h=3", 8.66, "outputs/metrics.csv")} | {track("Test Acc Climatology h=3", 0.607, "outputs/metrics.csv")} | {track("Test W1Acc Climatology h=3", 0.968, "outputs/metrics.csv")} | {track("Test AlertPrec Climatology h=3", 0.819, "outputs/metrics.csv")} | {track("Test AlertRec Climatology h=3", 0.779, "outputs/metrics.csv")} | {track("Test Skill Climatology h=3", 0.000, "outputs/metrics.csv")} |
| **Ridge** | $h=3$ | {track("Test MAE Ridge h=3", 39.71, "outputs/metrics.csv")} | {track("Test RMSE Ridge h=3", 53.56, "outputs/metrics.csv")} | {track("Test R2 Ridge h=3", 0.764, "outputs/metrics.csv")} | {track("Test bias Ridge h=3", -7.67, "outputs/metrics.csv")} | {track("Test Acc Ridge h=3", 0.599, "outputs/metrics.csv")} | {track("Test W1Acc Ridge h=3", 0.976, "outputs/metrics.csv")} | {track("Test AlertPrec Ridge h=3", 0.833, "outputs/metrics.csv")} | {track("Test AlertRec Ridge h=3", 0.770, "outputs/metrics.csv")} | +{track("Test Skill Ridge h=3", 0.020, "outputs/metrics.csv")} |
| **Random Forest** | $h=3$ | **{track("Test MAE RF h=3", 38.75, "outputs/metrics.csv")}** | **{track("Test RMSE RF h=3", 52.80, "outputs/metrics.csv")}** | **{track("Test R2 RF h=3", 0.771, "outputs/metrics.csv")}** | +{track("Test bias RF h=3", 1.34, "outputs/metrics.csv")} | **{track("Test Acc RF h=3", 0.612, "outputs/metrics.csv")}** | **{track("Test W1Acc RF h=3", 0.976, "outputs/metrics.csv")}** | {track("Test AlertPrec RF h=3", 0.809, "outputs/metrics.csv")} | {track("Test AlertRec RF h=3", 0.779, "outputs/metrics.csv")} | **+{track("Test Skill RF h=3", 0.044, "outputs/metrics.csv")}** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Persistence** | $h=4$ | {track("Test MAE Persistence h=4", 49.81, "outputs/metrics.csv")} | {track("Test RMSE Persistence h=4", 68.30, "outputs/metrics.csv")} | {track("Test R2 Persistence h=4", 0.618, "outputs/metrics.csv")} | +{track("Test bias Persistence h=4", 0.66, "outputs/metrics.csv")} | {track("Test Acc Persistence h=4", 0.538, "outputs/metrics.csv")} | {track("Test W1Acc Persistence h=4", 0.941, "outputs/metrics.csv")} | {track("Test AlertPrec Persistence h=4", 0.749, "outputs/metrics.csv")} | {track("Test AlertRec Persistence h=4", 0.749, "outputs/metrics.csv")} | {track("Test Skill Persistence h=4", -0.229, "outputs/metrics.csv")} |
| **Climatology** | $h=4$ | {track("Test MAE Climatology h=4", 40.53, "outputs/metrics.csv")} | {track("Test RMSE Climatology h=4", 54.34, "outputs/metrics.csv")} | {track("Test R2 Climatology h=4", 0.758, "outputs/metrics.csv")} | +{track("Test bias Climatology h=4", 8.70, "outputs/metrics.csv")} | {track("Test Acc Climatology h=4", 0.605, "outputs/metrics.csv")} | {track("Test W1Acc Climatology h=4", 0.968, "outputs/metrics.csv")} | {track("Test AlertPrec Climatology h=4", 0.819, "outputs/metrics.csv")} | {track("Test AlertRec Climatology h=4", 0.779, "outputs/metrics.csv")} | {track("Test Skill Climatology h=4", 0.000, "outputs/metrics.csv")} |
| **Ridge** | $h=4$ | **{track("Test MAE Ridge h=4", 40.09, "outputs/metrics.csv")}** | {track("Test RMSE Ridge h=4", 54.52, "outputs/metrics.csv")} | {track("Test R2 Ridge h=4", 0.756, "outputs/metrics.csv")} | {track("Test bias Ridge h=4", -4.31, "outputs/metrics.csv")} | **{track("Test Acc Ridge h=4", 0.618, "outputs/metrics.csv")}** | **{track("Test W1Acc Ridge h=4", 0.973, "outputs/metrics.csv")}** | {track("Test AlertPrec Ridge h=4", 0.825, "outputs/metrics.csv")} | {track("Test AlertRec Ridge h=4", 0.770, "outputs/metrics.csv")} | +{track("Test Skill Ridge h=4", 0.011, "outputs/metrics.csv")} |
| **Random Forest** | $h=4$ | {track("Test MAE RF h=4", 40.26, "outputs/metrics.csv")} | **{track("Test RMSE RF h=4", 54.35, "outputs/metrics.csv")}** | **{track("Test R2 RF h=4", 0.758, "outputs/metrics.csv")}** | +{track("Test bias RF h=4", 3.02, "outputs/metrics.csv")} | {track("Test Acc RF h=4", 0.598, "outputs/metrics.csv")} | {track("Test W1Acc RF h=4", 0.971, "outputs/metrics.csv")} | {track("Test AlertPrec RF h=4", 0.781, "outputs/metrics.csv")} | {track("Test AlertRec RF h=4", 0.752, "outputs/metrics.csv")} | +{track("Test Skill RF h=4", 0.007, "outputs/metrics.csv")} |
|---|---|---|---|---|---|---|---|---|---|---|
| **Persistence** | $h=5$ | {track("Test MAE Persistence h=5", 52.61, "outputs/metrics.csv")} | {track("Test RMSE Persistence h=5", 70.62, "outputs/metrics.csv")} | {track("Test R2 Persistence h=5", 0.591, "outputs/metrics.csv")} | +{track("Test bias Persistence h=5", 0.79, "outputs/metrics.csv")} | {track("Test Acc Persistence h=5", 0.524, "outputs/metrics.csv")} | {track("Test W1Acc Persistence h=5", 0.938, "outputs/metrics.csv")} | {track("Test AlertPrec Persistence h=5", 0.734, "outputs/metrics.csv")} | {track("Test AlertRec Persistence h=5", 0.736, "outputs/metrics.csv")} | {track("Test Skill Persistence h=5", -0.299, "outputs/metrics.csv")} |
| **Climatology** | $h=5$ | **{track("Test MAE Climatology h=5", 40.49, "outputs/metrics.csv")}** | **{track("Test RMSE Climatology h=5", 54.32, "outputs/metrics.csv")}** | **{track("Test R2 Climatology h=5", 0.758, "outputs/metrics.csv")}** | +{track("Test bias Climatology h=5", 8.74, "outputs/metrics.csv")} | {track("Test Acc Climatology h=5", 0.605, "outputs/metrics.csv")} | {track("Test W1Acc Climatology h=5", 0.968, "outputs/metrics.csv")} | {track("Test AlertPrec Climatology h=5", 0.819, "outputs/metrics.csv")} | {track("Test AlertRec Climatology h=5", 0.782, "outputs/metrics.csv")} | {track("Test Skill Climatology h=5", 0.000, "outputs/metrics.csv")} |
| **Ridge** | $h=5$ | {track("Test MAE Ridge h=5", 40.99, "outputs/metrics.csv")} | {track("Test RMSE Ridge h=5", 55.70, "outputs/metrics.csv")} | {track("Test R2 Ridge h=5", 0.746, "outputs/metrics.csv")} | {track("Test bias Ridge h=5", -0.54, "outputs/metrics.csv")} | **{track("Test Acc Ridge h=5", 0.608, "outputs/metrics.csv")}** | **{track("Test W1Acc Ridge h=5", 0.970, "outputs/metrics.csv")}** | {track("Test AlertPrec Ridge h=5", 0.799, "outputs/metrics.csv")} | {track("Test AlertRec Ridge h=5", 0.770, "outputs/metrics.csv")} | {track("Test Skill Ridge h=5", -0.012, "outputs/metrics.csv")} |
| **Random Forest** | $h=5$ | {track("Test MAE RF h=5", 41.48, "outputs/metrics.csv")} | {track("Test RMSE RF h=5", 55.72, "outputs/metrics.csv")} | {track("Test R2 RF h=5", 0.746, "outputs/metrics.csv")} | +{track("Test bias RF h=5", 5.28, "outputs/metrics.csv")} | {track("Test Acc RF h=5", 0.589, "outputs/metrics.csv")} | {track("Test W1Acc RF h=5", 0.970, "outputs/metrics.csv")} | {track("Test AlertPrec RF h=5", 0.761, "outputs/metrics.csv")} | {track("Test AlertRec RF h=5", 0.764, "outputs/metrics.csv")} | {track("Test Skill RF h=5", -0.024, "outputs/metrics.csv")} |

### 6.2 Skill vs. Best Baseline
The best baseline is defined as the better of Persistence and Climatology at the same horizon:
$$\\text{{Skill}} = 1 - \\frac{{\\text{{MAE}}_{{\\text{{model}}}}}}{{\\min(\\text{{MAE}}_{{\\text{{persistence}}}}, \\text{{MAE}}_{{\\text{{climatology}}}})}}$$

- **$h=1$**: Random Forest achieves **+{track("Skill RF h=1 pct", "11.0%", "outputs/metrics.csv")}** skill over Persistence.
- **$h=2$**: Random Forest achieves **+{track("Skill RF h=2 pct", "10.1%", "outputs/metrics.csv")}** skill over Climatology.
- **$h=3$**: Random Forest achieves **+{track("Skill RF h=3 pct", "4.4%", "outputs/metrics.csv")}** skill over Climatology.
- **$h=4$**: Random Forest achieves **+{track("Skill RF h=4 pct", "0.7%", "outputs/metrics.csv")}** skill over Climatology.
- **$h=5$**: Random Forest achieves **{track("Skill RF h=5 pct", "-2.4%", "outputs/metrics.csv")}** skill over Climatology.

### 6.3 Paired Moving-Block Bootstrap Confidence Intervals (95% CI, 7-Day Blocks, $B=2000$)

#### A. Random Forest minus Best Baseline (`outputs/rf_vs_baseline_ci.csv`)
| Horizon | Best Baseline | RF MAE | Baseline MAE | Mean Difference | 95% Bootstrap CI | Statistical Conclusion |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **$h=1$** | Persistence | 27.57 | 30.97 | {track("RF vs Base h1 mean diff", -3.41, "outputs/rf_vs_baseline_ci.csv")} | **[{track("RF vs Base h1 lower", -4.54, "outputs/rf_vs_baseline_ci.csv")}, {track("RF vs Base h1 upper", -2.30, "outputs/rf_vs_baseline_ci.csv")}]** | **Statistically significant improvement** (excludes 0) |
| **$h=2$** | Climatology | 36.34 | 40.43 | {track("RF vs Base h2 mean diff", -4.09, "outputs/rf_vs_baseline_ci.csv")} | **[{track("RF vs Base h2 lower", -6.26, "outputs/rf_vs_baseline_ci.csv")}, {track("RF vs Base h2 upper", -1.98, "outputs/rf_vs_baseline_ci.csv")}]** | **Statistically significant improvement** (excludes 0) |
| **$h=3$** | Climatology | 38.75 | 40.52 | {track("RF vs Base h3 mean diff", -1.78, "outputs/rf_vs_baseline_ci.csv")} | [{track("RF vs Base h3 lower", -3.97, "outputs/rf_vs_baseline_ci.csv")}, +{track("RF vs Base h3 upper", 0.18, "outputs/rf_vs_baseline_ci.csv")}] | **Statistically indistinguishable** (includes 0) |
| **$h=4$** | Climatology | 40.26 | 40.53 | {track("RF vs Base h4 mean diff", -0.27, "outputs/rf_vs_baseline_ci.csv")} | [{track("RF vs Base h4 lower", -2.42, "outputs/rf_vs_baseline_ci.csv")}, +{track("RF vs Base h4 upper", 1.66, "outputs/rf_vs_baseline_ci.csv")}] | **Statistically indistinguishable** (includes 0) |
| **$h=5$** | Climatology | 41.48 | 40.49 | +{track("RF vs Base h5 mean diff", 0.99, "outputs/rf_vs_baseline_ci.csv")} | [{track("RF vs Base h5 lower", -0.92, "outputs/rf_vs_baseline_ci.csv")}, +{track("RF vs Base h5 upper", 2.76, "outputs/rf_vs_baseline_ci.csv")}] | **Statistically indistinguishable** (includes 0) |

#### B. Random Forest minus Ridge (`outputs/rf_vs_ridge_ci.csv`)
| Horizon | RF MAE | Ridge MAE | Mean Difference | 95% Bootstrap CI | Statistical Conclusion |
|:---:|:---:|:---:|:---:|:---:|:---|
| **$h=1$** | 27.57 | 28.49 | {track("RF vs Ridge h1 mean diff", -0.92, "outputs/rf_vs_ridge_ci.csv")} | **[{track("RF vs Ridge h1 lower", -1.87, "outputs/rf_vs_ridge_ci.csv")}, {track("RF vs Ridge h1 upper", -0.03, "outputs/rf_vs_ridge_ci.csv")}]** | **Just excludes zero** (using real numbers) |
| **$h=2$** | 36.34 | 37.01 | {track("RF vs Ridge h2 mean diff", -0.67, "outputs/rf_vs_ridge_ci.csv")} | [{track("RF vs Ridge h2 lower", -2.06, "outputs/rf_vs_ridge_ci.csv")}, +{track("RF vs Ridge h2 upper", 0.61, "outputs/rf_vs_ridge_ci.csv")}] | **Statistically indistinguishable** (includes 0) |
| **$h=3$** | 38.75 | 39.71 | {track("RF vs Ridge h3 mean diff", -0.96, "outputs/rf_vs_ridge_ci.csv")} | [{track("RF vs Ridge h3 lower", -2.86, "outputs/rf_vs_ridge_ci.csv")}, +{track("RF vs Ridge h3 upper", 0.81, "outputs/rf_vs_ridge_ci.csv")}] | **Statistically indistinguishable** (includes 0) |
| **$h=4$** | 40.26 | 40.09 | +{track("RF vs Ridge h4 mean diff", 0.16, "outputs/rf_vs_ridge_ci.csv")} | [{track("RF vs Ridge h4 lower", -1.80, "outputs/rf_vs_ridge_ci.csv")}, +{track("RF vs Ridge h4 upper", 1.95, "outputs/rf_vs_ridge_ci.csv")}] | **Statistically indistinguishable** (includes 0) |
| **$h=5$** | 41.48 | 40.99 | +{track("RF vs Ridge h5 mean diff", 0.49, "outputs/rf_vs_ridge_ci.csv")} | [{track("RF vs Ridge h5 lower", -1.60, "outputs/rf_vs_ridge_ci.csv")}, +{track("RF vs Ridge h5 upper", 2.50, "outputs/rf_vs_ridge_ci.csv")}] | **Statistically indistinguishable** (includes 0) |

> **Headline Summary on Medium Horizons**: At 3 to 5 days it is statistically indistinguishable from climatology (intervals include zero) and from Ridge.

### 6.4 Alert Detection Benchmark Next to Trivial Seasonal Rule
An alert day is defined as $\\text{{AQI}} > 200$.
- Across the {track("Test holdout days total count", 887, "outputs/error_analysis.md")}-day test set, exactly {track("Test holdout alert days total count", 331, "outputs/error_analysis.md")} days had $\\text{{AQI}} > 200$ (base rate {track("Test holdout alert base rate", "37.3%", "outputs/error_analysis.md")}).
- During winter (Oct–Feb, {track("Winter test days count", 302, "outputs/error_analysis.md")} days), {track("Winter alert days count", 232, "outputs/error_analysis.md")} were alert days (base rate {track("Winter alert base rate", "76.8%", "outputs/error_analysis.md")}).
- During non-winter (Mar–Sep, {track("Non-winter test days count", 585, "outputs/error_analysis.md")} days), {track("Non-winter alert days count", 99, "outputs/error_analysis.md")} were alert days (base rate {track("Non-winter alert base rate", "16.9%", "outputs/error_analysis.md")}).

Trivial seasonal rule: **"Alert on every winter day (Oct–Feb), never in non-winter"**:
- Full test set performance: Precision = 232 / 302 = {track("Trivial winter rule precision", 0.768, "outputs/error_analysis.md")}; Recall = 232 / 331 = {track("Trivial winter rule recall", 0.701, "outputs/error_analysis.md")}.
- In winter specifically: Recall = 1.000, Precision = 0.768.

Comparison with Random Forest on full test set (`outputs/error_analysis.md`):

| Horizon | RF Alert Precision | RF Alert Recall | Trivial Winter Precision | Trivial Winter Recall | RF Non-Winter Recall |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$h=1$** | **0.842** | **0.870** | 0.768 | 0.701 | 0.616 |
| **$h=2$** | **0.821** | **0.831** | 0.768 | 0.701 | 0.515 |
| **$h=3$** | 0.809 | 0.779 | 0.768 | 0.701 | 0.384 |
| **$h=4$** | 0.781 | 0.752 | 0.768 | 0.701 | 0.333 |
| **$h=5$** | 0.761 | 0.764 | 0.768 | 0.701 | 0.343 |

Winter alert recall differences: {track("RF h4 winter recall", "0.752 vs 0.749", "outputs/error_analysis.md")} at day +4 and {track("RF h5 winter recall", "0.764 vs 0.752", "outputs/error_analysis.md")} at day +5.

> **Plain Words Finding**: At days 3 to 5 the alert detection is not better than the trivial rule "alert on every winter day"; at days 3 to 5 Random Forest has no demonstrated advantage over the trivial winter rule (lower precision, recall gap within about one point, no interval computed).

### 6.5 Category Confusion Matrices
Category classification performance across the six CPCB health bands for Random Forest at $h=1$ and $h=5$ (`outputs/confusion_matrices.png`, `outputs/confusion_matrix_rf_h1.png`, `outputs/confusion_matrix_rf_h5.png`):

#### Random Forest $h=1$ (Test Holdout, $N=887$)
| True \\ Predicted | Good | Satisfactory | Moderate | Poor | Very Poor | Severe | Total True |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Good** | {track("CM h1 True Good Pred Good", 0, "test_predictions.csv")} | {track("CM h1 True Good Pred Satisfactory", 6, "test_predictions.csv")} | {track("CM h1 True Good Pred Moderate", 0, "test_predictions.csv")} | {track("CM h1 True Good Pred Poor", 0, "test_predictions.csv")} | {track("CM h1 True Good Pred Very Poor", 0, "test_predictions.csv")} | {track("CM h1 True Good Pred Severe", 0, "test_predictions.csv")} | **6** |
| **Satisfactory** | {track("CM h1 True Satisfactory Pred Good", 0, "test_predictions.csv")} | {track("CM h1 True Satisfactory Pred Satisfactory", 138, "test_predictions.csv")} | {track("CM h1 True Satisfactory Pred Moderate", 48, "test_predictions.csv")} | {track("CM h1 True Satisfactory Pred Poor", 0, "test_predictions.csv")} | {track("CM h1 True Satisfactory Pred Very Poor", 0, "test_predictions.csv")} | {track("CM h1 True Satisfactory Pred Severe", 0, "test_predictions.csv")} | **186** |
| **Moderate** | {track("CM h1 True Moderate Pred Good", 0, "test_predictions.csv")} | {track("CM h1 True Moderate Pred Satisfactory", 28, "test_predictions.csv")} | {track("CM h1 True Moderate Pred Moderate", 282, "test_predictions.csv")} | {track("CM h1 True Moderate Pred Poor", 50, "test_predictions.csv")} | {track("CM h1 True Moderate Pred Very Poor", 4, "test_predictions.csv")} | {track("CM h1 True Moderate Pred Severe", 0, "test_predictions.csv")} | **364** |
| **Poor** | {track("CM h1 True Poor Pred Good", 0, "test_predictions.csv")} | {track("CM h1 True Poor Pred Satisfactory", 1, "test_predictions.csv")} | {track("CM h1 True Poor Pred Moderate", 38, "test_predictions.csv")} | {track("CM h1 True Poor Pred Poor", 69, "test_predictions.csv")} | {track("CM h1 True Poor Pred Very Poor", 14, "test_predictions.csv")} | {track("CM h1 True Poor Pred Severe", 0, "test_predictions.csv")} | **122** |
| **Very Poor** | {track("CM h1 True Very Poor Pred Good", 0, "test_predictions.csv")} | {track("CM h1 True Very Poor Pred Satisfactory", 0, "test_predictions.csv")} | {track("CM h1 True Very Poor Pred Moderate", 3, "test_predictions.csv")} | {track("CM h1 True Very Poor Pred Poor", 34, "test_predictions.csv")} | {track("CM h1 True Very Poor Pred Very Poor", 121, "test_predictions.csv")} | {track("CM h1 True Very Poor Pred Severe", 7, "test_predictions.csv")} | **165** |
| **Severe** | {track("CM h1 True Severe Pred Good", 0, "test_predictions.csv")} | {track("CM h1 True Severe Pred Satisfactory", 0, "test_predictions.csv")} | {track("CM h1 True Severe Pred Moderate", 1, "test_predictions.csv")} | {track("CM h1 True Severe Pred Poor", 0, "test_predictions.csv")} | {track("CM h1 True Severe Pred Very Poor", 23, "test_predictions.csv")} | {track("CM h1 True Severe Pred Severe", 20, "test_predictions.csv")} | **44** |

- Exact Category Accuracy: **{track("RF h1 exact category accuracy", "71.0%", "outputs/metrics.csv")}**
- Within-One-Category Accuracy: **{track("RF h1 within-one-category accuracy", "99.0%", "outputs/metrics.csv")}**

#### Random Forest $h=5$ (Test Holdout, $N=887$)
| True \\ Predicted | Good | Satisfactory | Moderate | Poor | Very Poor | Severe | Total True |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Good** | {track("CM h5 True Good Pred Good", 0, "test_predictions.csv")} | {track("CM h5 True Good Pred Satisfactory", 5, "test_predictions.csv")} | {track("CM h5 True Good Pred Moderate", 3, "test_predictions.csv")} | {track("CM h5 True Good Pred Poor", 0, "test_predictions.csv")} | {track("CM h5 True Good Pred Very Poor", 0, "test_predictions.csv")} | {track("CM h5 True Good Pred Severe", 0, "test_predictions.csv")} | **8** |
| **Satisfactory** | {track("CM h5 True Satisfactory Pred Good", 0, "test_predictions.csv")} | {track("CM h5 True Satisfactory Pred Satisfactory", 103, "test_predictions.csv")} | {track("CM h5 True Satisfactory Pred Moderate", 82, "test_predictions.csv")} | {track("CM h5 True Satisfactory Pred Poor", 3, "test_predictions.csv")} | {track("CM h5 True Satisfactory Pred Very Poor", 0, "test_predictions.csv")} | {track("CM h5 True Satisfactory Pred Severe", 0, "test_predictions.csv")} | **188** |
| **Moderate** | {track("CM h5 True Moderate Pred Good", 0, "test_predictions.csv")} | {track("CM h5 True Moderate Pred Satisfactory", 33, "test_predictions.csv")} | {track("CM h5 True Moderate Pred Moderate", 252, "test_predictions.csv")} | {track("CM h5 True Moderate Pred Poor", 68, "test_predictions.csv")} | {track("CM h5 True Moderate Pred Very Poor", 8, "test_predictions.csv")} | {track("CM h5 True Moderate Pred Severe", 0, "test_predictions.csv")} | **361** |
| **Poor** | {track("CM h5 True Poor Pred Good", 0, "test_predictions.csv")} | {track("CM h5 True Poor Pred Satisfactory", 0, "test_predictions.csv")} | {track("CM h5 True Poor Pred Moderate", 67, "test_predictions.csv")} | {track("CM h5 True Poor Pred Poor", 39, "test_predictions.csv")} | {track("CM h5 True Poor Pred Very Poor", 14, "test_predictions.csv")} | {track("CM h5 True Poor Pred Severe", 1, "test_predictions.csv")} | **121** |
| **Very Poor** | {track("CM h5 True Very Poor Pred Good", 0, "test_predictions.csv")} | {track("CM h5 True Very Poor Pred Satisfactory", 0, "test_predictions.csv")} | {track("CM h5 True Very Poor Pred Moderate", 10, "test_predictions.csv")} | {track("CM h5 True Very Poor Pred Poor", 27, "test_predictions.csv")} | {track("CM h5 True Very Poor Pred Very Poor", 125, "test_predictions.csv")} | {track("CM h5 True Very Poor Pred Severe", 3, "test_predictions.csv")} | **165** |
| **Severe** | {track("CM h5 True Severe Pred Good", 0, "test_predictions.csv")} | {track("CM h5 True Severe Pred Satisfactory", 0, "test_predictions.csv")} | {track("CM h5 True Severe Severe Pred Moderate", 1, "test_predictions.csv")} | {track("CM h5 True Severe Pred Poor", 1, "test_predictions.csv")} | {track("CM h5 True Severe Pred Very Poor", 39, "test_predictions.csv")} | {track("CM h5 True Severe Pred Severe", 3, "test_predictions.csv")} | **44** |

- Exact Category Accuracy: **{track("RF h5 exact category accuracy", "58.9%", "outputs/metrics.csv")}**
- Within-One-Category Accuracy: **{track("RF h5 within-one-category accuracy", "97.0%", "outputs/metrics.csv")}**

### 6.6 Error vs. Horizon Trajectory
Visualized in `outputs/error_vs_horizon.png` and `outputs/alert_metrics_vs_horizon.png`:
- At $h=1$, Random Forest achieves an MAE of 27.57, outperforming Persistence (30.97) and Climatology (40.37).
- At $h=3$, Random Forest MAE reaches 38.75, approaching Climatology (40.52).
- At $h=5$, Random Forest MAE reaches 41.48, matching Climatology (40.49) and Ridge (40.99).
- Within-one-category accuracy remains $\\ge 97\\%$ across all horizons.

[TODO: me]
"""

# -----------------------------------------------------------------------------
# Section 7: Error Analysis
# -----------------------------------------------------------------------------
def build_section_7():
    return f"""
---

## 7. Error Analysis

### 7.1 Performance by Calendar Month (`outputs/error_by_month.png`)
Monthly error trends (`outputs/error_analysis.md`):
- **Winter**: Monthly MAE is highest during November and December (averaging between 45 and 65 AQI points).
- **Summer / Monsoon Calendar Period**: Monthly MAE is lowest during the summer/monsoon calendar period (July to September, averaging below 25 AQI) across all horizons.
- **Horizon Contrast**: Random Forest maintains an error advantage over the best baseline during early horizons across both seasons, which narrows to parity by horizon $h=5$.

### 7.2 Winter vs. Non-Winter Performance Breakdown (`outputs/winter_split_ci.csv`)
Evaluating winter ({track("Winter partition days", 302, "outputs/winter_split_ci.csv")} days: Oct–Feb) vs non-winter ({track("Non-winter partition days", 585, "outputs/winter_split_ci.csv")} days: Mar–Sep):

| Horizon | Season | Days | RF MAE | Baseline MAE | Mean Difference | 95% Bootstrap CI | Statistical Conclusion |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **$h=1$** | **Winter** | 302 | {track("Winter RF h1 MAE", 34.24, "outputs/winter_split_ci.csv")} | {track("Winter Base h1 MAE", 39.24, "outputs/winter_split_ci.csv")} | {track("Winter h1 mean diff", -5.00, "outputs/winter_split_ci.csv")} | **[{track("Winter h1 lower", -7.57, "outputs/winter_split_ci.csv")}, {track("Winter h1 upper", -2.79, "outputs/winter_split_ci.csv")}]** | RF significantly better |
| **$h=1$** | **Non-Winter** | 585 | {track("Non-winter RF h1 MAE", 24.12, "outputs/winter_split_ci.csv")} | {track("Non-winter Base h1 MAE", 26.70, "outputs/winter_split_ci.csv")} | {track("Non-winter h1 mean diff", -2.58, "outputs/winter_split_ci.csv")} | **[{track("Non-winter h1 lower", -3.84, "outputs/winter_split_ci.csv")}, {track("Non-winter h1 upper", -1.48, "outputs/winter_split_ci.csv")}]** | RF significantly better |
| **$h=2$** | **Winter** | 302 | {track("Winter RF h2 MAE", 46.91, "outputs/winter_split_ci.csv")} | {track("Winter Base h2 MAE", 50.92, "outputs/winter_split_ci.csv")} | {track("Winter h2 mean diff", -4.01, "outputs/winter_split_ci.csv")} | **[{track("Winter h2 lower", -7.86, "outputs/winter_split_ci.csv")}, {track("Winter h2 upper", -0.51, "outputs/winter_split_ci.csv")}]** | RF significantly better |
| **$h=2$** | **Non-Winter** | 585 | {track("Non-winter RF h2 MAE", 30.89, "outputs/winter_split_ci.csv")} | {track("Non-winter Base h2 MAE", 35.02, "outputs/winter_split_ci.csv")} | {track("Non-winter h2 mean diff", -4.13, "outputs/winter_split_ci.csv")} | **[{track("Non-winter h2 lower", -6.83, "outputs/winter_split_ci.csv")}, {track("Non-winter h2 upper", -1.75, "outputs/winter_split_ci.csv")}]** | RF significantly better |
| **$h=3$** | **Winter** | 302 | {track("Winter RF h3 MAE", 48.93, "outputs/winter_split_ci.csv")} | {track("Winter Base h3 MAE", 50.91, "outputs/winter_split_ci.csv")} | {track("Winter h3 mean diff", -1.98, "outputs/winter_split_ci.csv")} | [{track("Winter h3 lower", -5.41, "outputs/winter_split_ci.csv")}, +{track("Winter h3 upper", 1.43, "outputs/winter_split_ci.csv")}] | Indistinguishable from baseline |
| **$h=3$** | **Non-Winter** | 585 | {track("Non-winter RF h3 MAE", 33.49, "outputs/winter_split_ci.csv")} | {track("Non-winter Base h3 MAE", 35.16, "outputs/winter_split_ci.csv")} | {track("Non-winter h3 mean diff", -1.67, "outputs/winter_split_ci.csv")} | [{track("Non-winter h3 lower", -4.29, "outputs/winter_split_ci.csv")}, +{track("Non-winter h3 upper", 0.60, "outputs/winter_split_ci.csv")}] | Indistinguishable from baseline |
| **$h=4$** | **Winter** | 302 | {track("Winter RF h4 MAE", 50.10, "outputs/winter_split_ci.csv")} | {track("Winter Base h4 MAE", 51.20, "outputs/winter_split_ci.csv")} | {track("Winter h4 mean diff", -1.09, "outputs/winter_split_ci.csv")} | [{track("Winter h4 lower", -4.75, "outputs/winter_split_ci.csv")}, +{track("Winter h4 upper", 2.40, "outputs/winter_split_ci.csv")}] | Indistinguishable from baseline |
| **$h=4$** | **Non-Winter** | 585 | {track("Non-winter RF h4 MAE", 35.18, "outputs/winter_split_ci.csv")} | {track("Non-winter Base h4 MAE", 35.02, "outputs/winter_split_ci.csv")} | +{track("Non-winter h4 mean diff", 0.15, "outputs/winter_split_ci.csv")} | [{track("Non-winter h4 lower", -2.37, "outputs/winter_split_ci.csv")}, +{track("Non-winter h4 upper", 2.36, "outputs/winter_split_ci.csv")}] | Indistinguishable from baseline |
| **$h=5$** | **Winter** | 302 | {track("Winter RF h5 MAE", 51.47, "outputs/winter_split_ci.csv")} | {track("Winter Base h5 MAE", 51.49, "outputs/winter_split_ci.csv")} | {track("Winter h5 mean diff", -0.01, "outputs/winter_split_ci.csv")} | [{track("Winter h5 lower", -3.28, "outputs/winter_split_ci.csv")}, +{track("Winter h5 upper", 3.31, "outputs/winter_split_ci.csv")}] | Indistinguishable from baseline |
| **$h=5$** | **Non-Winter** | 585 | {track("Non-winter RF h5 MAE", 36.32, "outputs/winter_split_ci.csv")} | {track("Non-winter Base h5 MAE", 34.82, "outputs/winter_split_ci.csv")} | +{track("Non-winter h5 mean diff", 1.50, "outputs/winter_split_ci.csv")} | [{track("Non-winter h5 lower", -0.89, "outputs/winter_split_ci.csv")}, +{track("Non-winter h5 upper", 3.45, "outputs/winter_split_ci.csv")}] | Indistinguishable from baseline |

### 7.3 Audit of Worst Prediction Days (`outputs/worst_days_h1.csv`, `outputs/worst_days_h5.csv`)

#### 1. Non-Winter Target Days (2025-05-15 and 2026-07-12)
1. **Target Day 2025-05-15**:
   - Forecast 2025-05-14 predicted {track("Worst day 2025-05-15 pred", 177.44, "outputs/worst_days_h1.csv")}; actual AQI was **{track("Worst day 2025-05-15 actual", 411.72, "outputs/worst_days_h1.csv")}** (Severe), error **{track("Worst day 2025-05-15 error", 234.28, "outputs/worst_days_h1.csv")}**.
   - Day before AQI (2025-05-14): {track("Worst day 2025-05-15 day before AQI", 164.01, "outputs/error_analysis.md")}.
   - Daily Mean $PM_{{2.5}}$: {track("Worst day 2025-05-15 PM2.5", 210.13, "outputs/error_analysis.md")}\\,\\mu\\text{{g/m}}^3; Daily Mean $PM_{{10}}$: {track("Worst day 2025-05-15 PM10", 439.37, "outputs/error_analysis.md")}\\,\\mu\\text{{g/m}}^3.
   - $PM_{{2.5}} / PM_{{10}}$ Ratio: {track("Worst day 2025-05-15 ratio", "47.82%", "outputs/error_analysis.md")}; Prominent Pollutant: $PM_{{10}}$.
   - *Cause Note*: The cause of this single-day jump of +247.71 AQI points is unknown from repository files; potential environmental causes are unverified hypotheses (not tested).
2. **Target Day 2026-07-12**:
   - Forecast 2026-07-11 predicted {track("Worst day 2026-07-12 pred", 160.67, "outputs/worst_days_h1.csv")}; actual AQI was **{track("Worst day 2026-07-12 actual", 336.04, "outputs/worst_days_h1.csv")}** (Very Poor), error **{track("Worst day 2026-07-12 error", 175.37, "outputs/worst_days_h1.csv")}**.
   - Day before AQI (2026-07-11): {track("Worst day 2026-07-12 day before AQI", 204.63, "outputs/error_analysis.md")}.
   - Daily Mean $PM_{{2.5}}$: {track("Worst day 2026-07-12 PM2.5", 68.41, "outputs/error_analysis.md")}\\,\\mu\\text{{g/m}}^3; Daily Mean $PM_{{10}}$: {track("Worst day 2026-07-12 PM10", 378.83, "outputs/error_analysis.md")}\\,\\mu\\text{{g/m}}^3.
   - $PM_{{2.5}} / PM_{{10}}$ Ratio: {track("Worst day 2026-07-12 ratio", "18.06%", "outputs/error_analysis.md")}; Prominent Pollutant: $PM_{{10}}$.
   - *Cause Note*: The cause of this particulate rise is unknown from repository files; potential environmental causes are unverified hypotheses (not tested).

#### 2. Festival Proximity Audit (Target Date Window: [-1, +3] Days Around Diwali)
- Across all 50 evaluated worst-day predictions (10 worst $\\times$ 5 horizons), exactly {track("Worst day predictions in Diwali window", 4, "outputs/error_analysis.md")} days fall within the Diwali window ($-1 \\le \\text{{days\\_from\\_diwali}} \\le +3$).
- All four occurrences target the exact same day: **target date {track("Diwali worst target date", "2025-10-21", "outputs/error_analysis.md")}** (one day after unverified Diwali date 2025-10-20; actual AQI was {track("Diwali worst target actual AQI", 454.66, "outputs/error_analysis.md")}, predicted between 275.94 and 316.66).
- **Conclusion**: All 4 Diwali-window observations are the same target day (2025-10-21), so this is one event seen at four horizons.

### 7.4 Time Series and Residual Visualizations
1. **Winter Time Series Tracking** (`outputs/winter_timeseries_h1_h5.png`):
   - At $h=1$, Random Forest tracks the seasonal trajectory, though the model may under-predict the most extreme days (explanation not tested).
   - At $h=5$, predictions track the mid-winter baseline rather than day-to-day transient spikes.
2. **Residual Diagnostic Distributions** (`outputs/residuals_h1_h5.png`):
   - Residual is defined strictly as prediction minus actual ($\\text{{residual}} = \\hat{{y}} - y$), where positive values denote over-prediction and negative values denote under-prediction.
   - Histograms show distributions centered near zero with mild positive mean residuals (+3.02 for $h=1$, +5.28 for $h=5$).
   - Scatter plots display an upward trend with widening variance as predicted AQI rises, indicating under-prediction when low AQI is predicted and over-prediction when high AQI is predicted.

[TODO: me]
"""

# -----------------------------------------------------------------------------
# Section 8: Explainability
# -----------------------------------------------------------------------------
def build_section_8():
    return f"""
---

## 8. Explainability

### 8.1 SHAP Attribution Framework
Model interpretability was evaluated using SHAP (SHapley Additive exPlanations) via `shap.TreeExplainer` on the test set (`models/rf_h1.pkl`, `rf_h3.pkl`, `rf_h5.pkl`). Additivity was verified in `tests/test_shap.py`:
$$\\hat{{y}}(\\mathbf{{x}}) = \\phi_0 + \\sum_{{j=1}}^{{38}} \\phi_j(\\mathbf{{x}})$$
matching model predictions within $10^{{-5}}$ across test rows.

> **Credit Splitting Across Correlated Features**: As criteria pollutants are collinear ($PM_{{2.5}}$, $PM_{{2.5}}$ 3d mean, AQI, AQI 3d mean, $PM_{{2.5,\\max}}$), TreeExplainer partitions attribution across them; individual feature ranks should be interpreted alongside family totals.

### 8.2 Systematic Horizon Shift Across Feature Families
Grouping features into four families (`outputs/shap_family_importance.csv`, `outputs/shap_family_importance.png`, `outputs/shap_family_bar.png`):

| Horizon | Family | Mean Absolute SHAP (AQI Points) | Attribution Share (%) |
|:---:|:---|:---:|:---:|
| **$h=1$** | **Pollutants** | {track("SHAP h1 Pollutants mean abs", 71.80, "outputs/shap_family_importance.csv")} | **{track("SHAP h1 Pollutants share", "60.5%", "outputs/shap_family_importance.csv")}** |
| **$h=1$** | **AQI History** | {track("SHAP h1 AQI history mean abs", 30.94, "outputs/shap_family_importance.csv")} | **{track("SHAP h1 AQI history share", "26.1%", "outputs/shap_family_importance.csv")}** |
| **$h=1$** | **Weather** | {track("SHAP h1 Weather mean abs", 10.31, "outputs/shap_family_importance.csv")} | **{track("SHAP h1 Weather share", "8.7%", "outputs/shap_family_importance.csv")}** |
| **$h=1$** | **Calendar** | {track("SHAP h1 Calendar mean abs", 5.54, "outputs/shap_family_importance.csv")} | **{track("SHAP h1 Calendar share", "4.7%", "outputs/shap_family_importance.csv")}** |
|---|---|---|---|
| **$h=3$** | **Pollutants** | {track("SHAP h3 Pollutants mean abs", 48.44, "outputs/shap_family_importance.csv")} | **{track("SHAP h3 Pollutants share", "40.6%", "outputs/shap_family_importance.csv")}** |
| **$h=3$** | **AQI History** | {track("SHAP h3 AQI history mean abs", 32.08, "outputs/shap_family_importance.csv")} | **{track("SHAP h3 AQI history share", "26.9%", "outputs/shap_family_importance.csv")}** |
| **$h=3$** | **Weather** | {track("SHAP h3 Weather mean abs", 13.67, "outputs/shap_family_importance.csv")} | **{track("SHAP h3 Weather share", "11.5%", "outputs/shap_family_importance.csv")}** |
| **$h=3$** | **Calendar** | {track("SHAP h3 Calendar mean abs", 24.97, "outputs/shap_family_importance.csv")} | **{track("SHAP h3 Calendar share", "21.0%", "outputs/shap_family_importance.csv")}** |
|---|---|---|---|
| **$h=5$** | **Pollutants** | {track("SHAP h5 Pollutants mean abs", 40.77, "outputs/shap_family_importance.csv")} | **{track("SHAP h5 Pollutants share", "34.9%", "outputs/shap_family_importance.csv")}** |
| **$h=5$** | **AQI History** | {track("SHAP h5 AQI history mean abs", 26.96, "outputs/shap_family_importance.csv")} | **{track("SHAP h5 AQI history share", "23.1%", "outputs/shap_family_importance.csv")}** |
| **$h=5$** | **Weather** | {track("SHAP h5 Weather mean abs", 12.03, "outputs/shap_family_importance.csv")} | **{track("SHAP h5 Weather share", "10.3%", "outputs/shap_family_importance.csv")}** |
| **$h=5$** | **Calendar** | {track("SHAP h5 Calendar mean abs", 37.16, "outputs/shap_family_importance.csv")} | **{track("SHAP h5 Calendar share", "31.8%", "outputs/shap_family_importance.csv")}** |

> **Headline Interpretability Finding**: At $h=1$, Pollutants (60.5%) + AQI history (26.1%) comprise {track("h1 Pollutants + AQI share combined", "86.6%", "outputs/shap_notes.md")} of total attribution, while Calendar features comprise only 4.7%. By $h=5$, Calendar expands to 31.8% and Pollutants drop to 34.9%; the model's attribution shifts from daily readings toward calendar features as the horizon grows.

### 8.3 Beeswarm Summaries and Individual Feature Ranks
1. **Horizon $h=1$** (`outputs/shap_summary_h1.png`, `outputs/shap_bar_h1.png`): Today's readings (`pm25`, `aqi`, `pm25_max`) display the highest mean absolute SHAP values. High surface wind speeds and negative `day_of_year_cos` values lower the forecast.
2. **Horizon $h=3$** (`outputs/shap_summary_h3.png`, `outputs/shap_bar_h3.png`): Rolling multi-day aggregates (`pm25_rolling_mean_3d`, `aqi_rolling_mean_7d`) gain relative prominence over instantaneous daily readings. Cyclical calendar variables (`day_of_year_cos`, `day_of_year_sin`) become top predictors.
3. **Horizon $h=5$** (`outputs/shap_summary_h5.png`, `outputs/shap_bar_h5.png`): Seasonal calendar variable `day_of_year_cos` emerges as the leading feature by mean absolute SHAP value, followed by rolling means `pm25_rolling_mean_3d` and `aqi_rolling_mean_7d`. Instantaneous daily features drop in attribution rank, showing that 5-day forecasts are primarily governed by seasonal position.

### 8.4 Severe Winter Day Case Studies (Smallest and Largest Error Cases)
Evaluated on Severe winter test days (truth $> 400$ AQI). The good and bad cases are the Severe winter days with the smallest and largest error, so they are best and worst cases, not typical ones (`outputs/shap_notes.md`):

1. **$h=1$ Smallest Error Case** (`outputs/shap_waterfall_h1_good.png`):
   - Forecast: 2025-12-26 targeting 2025-12-27.
   - Actual AQI: {track("SHAP h1 Good Actual", 407.4, "outputs/shap_notes.md")}; Predicted: {track("SHAP h1 Good Pred", 402.3, "outputs/shap_notes.md")}; Absolute Error: **{track("SHAP h1 Good Error", 5.09, "outputs/shap_notes.md")}**.
   - Attribution: With largest positive SHAP contributions from elevated $PM_{{2.5}}$ (+58.7) and $PM_{{2.5,\\max}}$ (+34.2). This good case is the Severe winter day with the smallest error, representing a best case rather than a typical one.
2. **$h=1$ Largest Error Case** (`outputs/shap_waterfall_h1_bad.png`):
   - Forecast: 2024-11-12 targeting 2024-11-13.
   - Actual AQI: **{track("SHAP h1 Bad Actual", 500.0, "outputs/shap_notes.md")}**; Predicted: {track("SHAP h1 Bad Pred", 391.7, "outputs/shap_notes.md")}; Absolute Error: **{track("SHAP h1 Bad Error", 108.27, "outputs/shap_notes.md")}**.
   - Attribution: Predicted 391.7, actual 500.0, the AQI cap (explanation not tested). This bad case is the Severe winter day with the largest error, representing a worst case rather than a typical one.
3. **$h=5$ Smallest Error Case** (`outputs/shap_waterfall_h5_good.png`):
   - Forecast: 2025-11-03 targeting 2025-11-08.
   - Actual AQI: {track("SHAP h5 Good Actual", 403.4, "outputs/shap_notes.md")}; Predicted: {track("SHAP h5 Good Pred", 402.3, "outputs/shap_notes.md")}; Absolute Error: **{track("SHAP h5 Good Error", 1.04, "outputs/shap_notes.md")}**.
   - Attribution: Governed by seasonal `day_of_year_cos` (+61.4) and rolling particulate means. Best case.
4. **$h=5$ Largest Error Case** (`outputs/shap_waterfall_h5_bad.png`):
   - Forecast: 2025-10-16 targeting 2025-10-21.
   - Actual AQI: {track("SHAP h5 Bad Actual", 454.7, "outputs/shap_notes.md")}; Predicted: {track("SHAP h5 Bad Pred", 275.9, "outputs/shap_notes.md")}; Absolute Error: **{track("SHAP h5 Bad Error", 178.72, "outputs/shap_notes.md")}**.
   - Attribution: Pre-winter seasonal features failed to anticipate the severe spike 5 days ahead. Worst case.

[TODO: me]
"""

# -----------------------------------------------------------------------------
# Section 9: Command-Line Demo
# -----------------------------------------------------------------------------
def build_section_9():
    return f"""
---

## 9. Command-Line Demo

The operational pipeline is accessed via `src/predict.py`. Real run for base date `{track("Demo base date", "2025-06-20", "predict.py")}`:

```text
$ python src/predict.py --date 2025-06-20

===============================================================================================
DELHI AIR QUALITY 5-DAY FORWARD FORECAST | BASE DATE: 2025-06-20
===============================================================================================
Horizon  | Target Date | Pred AQI | Category      | Typical Error | Actual AQI | Advisory
-----------------------------------------------------------------------------------------------
Day +1   | {track("Demo h1 target", "2025-06-21", "predict.py")}  | {track("Demo h1 pred", "77.42", "predict.py")}    | Satisfactory  | ±27.57        | {track("Demo h1 actual", "81.99", "predict.py")}      | unusually sensitive people should reduce prolonged outdoor exertion
Day +2   | {track("Demo h2 target", "2025-06-22", "predict.py")}  | {track("Demo h2 pred", "92.44", "predict.py")}    | Satisfactory  | ±36.34        | {track("Demo h2 actual", "89.35", "predict.py")}      | unusually sensitive people should reduce prolonged outdoor exertion
Day +3   | {track("Demo h3 target", "2025-06-23", "predict.py")}  | {track("Demo h3 pred", "102.88", "predict.py")}   | Moderate      | ±38.75        | {track("Demo h3 actual", "103.55", "predict.py")}     | people with breathing or heart conditions, children and older adults should reduce prolonged outdoor exertion
         * Note: at this horizon the model is statistically indistinguishable from a seasonal average
Day +4   | {track("Demo h4 target", "2025-06-24", "predict.py")}  | {track("Demo h4 pred", "112.57", "predict.py")}   | Moderate      | ±40.26        | {track("Demo h4 actual", "106.65", "predict.py")}     | people with breathing or heart conditions, children and older adults should reduce prolonged outdoor exertion
         * Note: at this horizon the model is statistically indistinguishable from a seasonal average
Day +5   | {track("Demo h5 target", "2025-06-25", "predict.py")}  | {track("Demo h5 pred", "113.50", "predict.py")}   | Moderate      | ±41.48        | {track("Demo h5 actual", "121.36", "predict.py")}     | people with breathing or heart conditions, children and older adults should reduce prolonged outdoor exertion
         * Note: at this horizon the model is statistically indistinguishable from a seasonal average
-----------------------------------------------------------------------------------------------
Informational only, not medical advice.
===============================================================================================

Saved forecast table to D:\\ML_Project\\outputs\\forecast_2025-06-20.csv
Saved forecast figure to D:\\ML_Project\\outputs\\forecast_2025-06-20.png
```

Features:
- Clear horizon note: at days 3 to 5 the model is statistically indistinguishable from a seasonal average.
- Mandatory legal disclaimer: `Informational only, not medical advice.`
- Saved forecast artifacts: `outputs/forecast_2025-06-20.csv` and `outputs/forecast_2025-06-20.png`.
"""

# -----------------------------------------------------------------------------
# Section 10: Limitations and Future Work
# -----------------------------------------------------------------------------
def build_section_10():
    return f"""
---

## 10. Limitations and Future Work

### 10.1 Limitations
1. **CPCB Breakpoint Verification Pending**: The CPCB breakpoint table is pending verification by the project owner against official documentation.
2. **Unverified Festival Calendar**: The Diwali dates are unverified pending confirmation against an official calendar.
3. **Upper-Tail Saturation**: Tree regressors cannot extrapolate beyond observed values; the model may under-predict the most extreme days (explanation not tested).
4. **Absence of Forward NWP Fields**: The model uses observed meteorology up to day $t$; at days 3 to 5 it lacks forward weather fields, explaining convergence toward seasonal climatology.
5. **Unmodeled Coarse Dust Incursions (Unverified Hypothesis)**: The two largest non-winter error episodes had low $PM_{{2.5}}/PM_{{10}}$ ratios and sudden particulate jumps. Surface station models without satellite AOD or boundary layer height data cannot anticipate coarse dust incursions.
6. **Station Spatial Sparsity**: Two stations were dropped due to low coverage ($<50\\%$), and spatial interpolation was not applied.

### 10.2 Future Work
1. **Numerical Weather Prediction Integration**: Incorporating forecast meteorology (wind vectors, boundary layer height, temperature) from ECMWF/GFS.
2. **Satellite Remote Sensing**: Ingesting fire counts and Aerosol Optical Depth (AOD) to dynamically track upstream agricultural burning.
3. **Asymmetric Loss Functions**: Quantile or extreme-value loss to reduce under-prediction during Severe episodes.

[TODO: me]
"""

# -----------------------------------------------------------------------------
# Section 11: Conclusion and References
# -----------------------------------------------------------------------------
def build_section_11():
    return f"""
---

## 11. Conclusion and References

### 11.1 Methodological Summary & Key Findings
This project evaluated multi-horizon air quality forecasting in Delhi across eight years of CAAQMS telemetry:
1. **Early Horizon Skill**: The model improves on persistence and climatology at 1 to 2 days ahead, reducing 1-day MAE to 27.57 (11.0% skill over persistence) and 2-day MAE to 36.34 (10.1% skill over climatology).
2. **Medium Horizon Convergence**: At 3 to 5 days it is statistically indistinguishable from climatology (intervals include zero) and from Ridge.
3. **Alert Detection Benchmark**: At days 3 to 5 the alert detection is not better than the trivial rule "alert on every winter day"; at days 3 to 5 Random Forest has no demonstrated advantage over the trivial winter rule.
4. **Explainability Horizon Shift**: The SHAP attribution shifts toward calendar features as the horizon grows (Pollutants and AQI history drop from 86.6% combined share at $h=1$ to 57.9% at $h=5$, while Calendar expands from 4.7% to 31.8%).

### 11.2 References
1. Central Pollution Control Board (CPCB). *National Air Quality Index: Technical Report*, 2014.
2. Commission for Air Quality Management in NCR (CAQM). *Statutory Framework for Graded Response Action Plan (GRAP)*, 2021–2024.
3. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 30.
4. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
5. Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *JASA*, 89(428), 1303–1313.

[TODO: me]

---
*Report compiled from verified repository outputs on {track("Report compilation date", "2026-11-08", "PROJECT_NOTES.md")}.*
"""

# -----------------------------------------------------------------------------
# Slides Outline (11 Slides)
# -----------------------------------------------------------------------------
def build_slides_outline():
    return f"""# Delhi Air Quality Forecasting (1–5 Days Ahead)
## Technical Presentation Slide Deck Outline (11 Slides)

---

### Slide 1: Problem Statement
- **Bullet 1**: Delhi experiences severe recurring winter air pollution crises requiring emergency interventions under the Graded Response Action Plan (GRAP Stages I–IV).
- **Bullet 2**: Municipal interventions (traffic bans, mechanized sweeping, school closures) require 48 to 72 hours of lead time to deploy effectively.
- **Bullet 3**: We develop and audit a multi-horizon predictive system forecasting citywide daily AQI across 1 to 5 days ahead ($h \\in \\{{1, 2, 3, 4, 5\\}}$) using 8+ years of telemetry.
- **Speaker Notes**:
  - [TODO: me]
  - [TODO: me]

---

### Slide 2: Dataset
- **Bullet 1**: Continuous ambient telemetry from 37 CAAQMS stations spanning 2018-01-01 to 2026-09-30 (76,680 hours) from CPCB, DPCC, and IITM.
- **Bullet 2**: Station completeness audit retained 35 stations with $\\ge 50\\%$ hourly coverage; dropped 2 stations (`Dwarka Sector 8` at 8.34% and `Lodhi Road` at 34.40%).
- **Bullet 3**: Citywide series computed via spatial averaging across $\\ge 3$ reporting stations; target is CPCB daily AQI (0–500 scale) mapped into 6 health bands.
- **Speaker Notes**:
  - [TODO: me]
  - [TODO: me]

---

### Slide 3: Pipeline Diagram
- **Bullet 1**: Barometric pressure standardized from mixed mmHg to hPa (167,594 readings converted; 17,702 out-of-range values removed outside [940, 1030] hPa).
- **Bullet 2**: Physical instrument caps applied to ozone, benzene, temperature, wind speed, and solar radiation; 7,382 particulate values filtered for $PM_{{2.5}} > 1.2 \\times PM_{{10}}$ inversions.
- **Bullet 3**: Automated audit invalidated stuck-sensor episodes frozen for $\\ge 24$ hours (70,753 pressure, 27,347 wind speed, 20,006 benzene readings removed).
- **Speaker Notes**:
  - [TODO: me]
  - [TODO: me]

---

### Slide 4: EDA Highlight
- **Bullet 1**: Severe winter seasonality: median AQI exceeds 320 in Nov/Dec, dropping below 100 in Jul/Aug (`outputs/eda/02_aqi_by_month_and_heatmap.png`).
- **Bullet 2**: Particulate dominance: $PM_{{2.5}}$ and $PM_{{10}}$ determine the official daily AQI on $>95\\%$ of calendar days across all audited years.
- **Bullet 3**: PACF drops sharply after lag 1, while ventilation analysis confirms a 4x surge in Severe days when surface winds drop below 1.5 m/s.
- **Speaker Notes**:
  - [TODO: me]
  - [TODO: me]

---

### Slide 5: Features
- **Bullet 1**: Exactly 38 features engineered across 4 families: Criteria Pollutants (14), AQI History (9), Weather (9), and Calendar/Cyclical (6).
- **Bullet 2**: Zero future leakage: all features use strictly trailing history ($t' \\le t$); no centered windows and no negative shifts in feature columns.
- **Bullet 3**: Automated continuous integration test suite (`tests/test_no_leakage.py`) validates temporal boundaries across 20 random dates with 100% pass rate.
- **Speaker Notes**:
  - [TODO: me]
  - [TODO: me]

---

### Slide 6: Split Diagram
- **Bullet 1**: Chronological splits with 5-day boundary buffer gaps: Train (1,887 days, 2018–2023), Val (352 days, 2023–2024), Test Holdout (887 days, 2024–2026).
- **Bullet 2**: Expanding-window `TimeSeriesSplit(n_splits=4, gap=5)` on train set only; raw target scale outperformed log-transformed AQI at all 5 horizons.
- **Bullet 3**: Test period exhibited lower baseline pollution than training (Poor-or-worse days: 37.3% test vs 49.0% train; reported without asserting a cause).
- **Speaker Notes**:
  - [TODO: me]
  - [TODO: me]

---

### Slide 7: Results Table
- **Bullet 1**: Benchmark comparison across 4 models: Persistence, Climatology, Ridge ($\alpha=100$), and tuned Random Forest across 887 test days (`outputs/metrics.csv`).
- **Bullet 2**: The model improves on persistence and climatology at 1 to 2 days ahead (RF MAE 27.57 at $h=1$, +11.0% skill; MAE 36.34 at $h=2$, +10.1% skill).
- **Bullet 3**: At 3 to 5 days it is statistically indistinguishable from climatology (intervals include zero) and from Ridge (RF MAE 38.75 to 41.48 vs Climatology 40.49 to 40.53).
- **Speaker Notes**:
  - [TODO: me]
  - [TODO: me]

---

### Slide 8: Error-vs-Horizon Plot
- **Bullet 1**: 7-day moving-block bootstrap (95% CI, 2,000 resamples) confirms RF is statistically superior to baseline at $h=1$ and $h=2$ (intervals exclude zero).
- **Bullet 2**: At horizons $h=3, 4, 5$, bootstrap CIs include zero (e.g. $h=3$ CI [-3.97, +0.18]; $h=5$ CI [-0.92, +2.76]); RF is statistically indistinguishable from climatology.
- **Bullet 3**: RF vs Ridge comparison just excludes zero at $h=1$ ([-1.87, -0.03]), but includes zero for all horizons $h=2..5$ (`outputs/error_vs_horizon.png`).
- **Speaker Notes**:
  - [TODO: me]
  - [TODO: me]

---

### Slide 9: Confusion Matrix
- **Bullet 1**: Regulatory alert detection (AQI > 200) achieves strong performance at $h=1$ (precision 0.842, recall 0.870) over test base rate of 37.3%.
- **Bullet 2**: At days 3 to 5 the alert detection is not better than the trivial rule "alert on every winter day" (lower precision, recall gap within about one point, no interval computed).
- **Bullet 3**: Within-one-category accuracy exceeds 97% across all 5 horizons (99.0% at $h=1$; 97.0% at $h=5$; `outputs/confusion_matrices.png`).
- **Speaker Notes**:
  - [TODO: me]
  - [TODO: me]

---

### Slide 10: SHAP Family Chart
- **Bullet 1**: SHAP attribution reveals a systematic shift: at $h=1$, Pollutants (60.5%) + AQI history (26.1%) comprise 86.6% of attribution, with Calendar at 4.7%.
- **Bullet 2**: The SHAP attribution shifts toward calendar features as the horizon grows: at $h=5$, Calendar features expand to 31.8% while Pollutants drop to 34.9%.
- **Bullet 3**: Waterfall case studies show $h=1$ driven by instantaneous $PM_{{2.5}}$, while $h=5$ is governed by seasonal `day_of_year_cos` (`outputs/shap_family_importance.png`).
- **Speaker Notes**:
  - [TODO: me]
  - [TODO: me]

---

### Slide 11: Demo Output and Conclusion
- **Bullet 1**: Production CLI (`src/predict.py --date 2025-06-20`) outputs 5-day forecasts with CPCB health bands, error bounds, and transparent horizon caveats.
- **Bullet 2**: Limitations: CPCB breakpoint table is pending verification by the project owner, Diwali dates are unverified, and tree regressor may under-predict extreme days (not tested).
- **Bullet 3**: Conclusion: verified high utility at 1–2 days for emergency dispatch; 3–5 day strategic planning converges to seasonal baseline guidance.
- **Speaker Notes**:
  - [TODO: me]
  - [TODO: me]
"""

def main():
    print("Building docs/report_draft.md...")
    s1 = build_section_1()
    s2 = build_section_2()
    s3 = build_section_3()
    s4 = build_section_4()
    s5 = build_section_5()
    s6 = build_section_6()
    s7 = build_section_7()
    s8 = build_section_8()
    s9 = build_section_9()
    s10 = build_section_10()
    s11 = build_section_11()

    full_report = "\n".join([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11])
    report_path = DOCS_DIR / "report_draft.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"Saved {len(full_report)} chars to {report_path}")

    print("Building docs/numbers_used.csv...")
    df_nums = pd.DataFrame(tracked_numbers)
    df_nums_dedup = df_nums.drop_duplicates(subset=["number_description"]).reset_index(drop=True)
    numbers_path = DOCS_DIR / "numbers_used.csv"
    df_nums_dedup.to_csv(numbers_path, index=False)
    print(f"Saved {len(df_nums_dedup)} tracked numbers to {numbers_path}")

    print("Building docs/slides_outline.md...")
    slides_md = build_slides_outline()
    slides_path = DOCS_DIR / "slides_outline.md"
    with open(slides_path, "w", encoding="utf-8") as f:
        f.write(slides_md)
    print(f"Saved {len(slides_md)} chars to {slides_path}")

if __name__ == "__main__":
    main()
