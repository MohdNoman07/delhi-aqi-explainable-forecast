# Delhi Air Quality Forecasting (1 to 5 Days Ahead)
## Comprehensive Technical Report & Methodological Audit

---

## 1. Introduction and Problem Statement

### 1.1 Context and Public Health Urgency
Ambient air pollution in the National Capital Territory (NCT) of Delhi is a persistent environmental health challenge. Winter meteorological conditions—shallow boundary layers, surface temperature inversions, and low wind speeds—frequently trap particulate matter ($PM_{2.5}$ and $PM_{10}$) and criteria gases within the regional airshed. Prolonged exposure during these episodes increases respiratory and cardiovascular hospital admissions, particularly among vulnerable populations.

Regulatory mitigation in Delhi is structured under the Graded Response Action Plan (GRAP), administered by the Commission for Air Quality Management (CAQM). GRAP defines four sequential emergency intervention stages:
- **Stage I (Poor, AQI 201–300)**: Mechanized road sweeping, water sprinkling, dust control at construction sites.
- **Stage II (Very Poor, AQI 301–400)**: Ban on diesel generator sets, parking fee surges, enhanced public transport frequency.
- **Stage III (Severe, AQI 401–450)**: Strict bans on non-essential construction and demolition, restrictions on BS-III petrol and BS-IV diesel light motor vehicles.
- **Stage IV (Severe+, AQI > 450)**: Ban on entry of non-essential commercial trucks into Delhi, closure of schools for physical classes, and potential odd-even vehicular rationing.

### 1.2 The Forecasting Problem: Why Lead Time Matters
Historically, environmental policy actions have been reactive, triggered only after ambient monitors record sustained Severe air quality. Reactive interventions suffer from operational delays:
- Municipal agencies require 48 to 72 hours of lead time to deploy road sweeping fleets, reroute heavy freight traffic, and issue public health notices.
- Advance notice is necessary for schools, commercial establishments, and sensitive individuals (e.g. asthmatics) to prepare before peak stagnation arrives.

### 1.3 Multi-Horizon Scope: Forecasting 1 to 5 Days Ahead ($h \in \{1, 2, 3, 4, 5\}$)
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


---

## 2. Dataset

### 2.1 Source and Spatial Coverage
The empirical foundation comprises continuous ambient monitoring records from the Central Pollution Control Board (CPCB), Delhi Pollution Control Committee (DPCC), and Indian Institute of Tropical Meteorology (IITM). Continuous Ambient Air Quality Monitoring Stations (CAAQMS) record criteria pollutants and meteorological variables at hourly resolution.

A total of 37 stations across Delhi were audited for data completeness over the period from 2018-01-01 through 2026-09-30, spanning 76680 continuous calendar hours.

### 2.2 Station Selection and Spatial Aggregation Criteria
Stations were evaluated on hourly completeness for criteria particulates ($PM_{2.5}$ and $PM_{10}$) according to project rules (`src/config.py`):
1. **Station Selection Threshold**: A station was retained only if its valid hourly coverage of $PM_{2.5}$ or $PM_{10}$ met or exceeded 50.0%.
2. **Audit Result**: 35 of 37 stations satisfied the criterion and were retained. Exactly 2 stations were dropped (`outputs/station_selection.csv`):
   - `site_1422_Dwarka_Sector_8_Delhi_DPCC`: 6394 / 76680 hours (8.34% coverage).
   - `site_5395_Lodhi_Road_Delhi_IITM`: 26377 / 76680 hours (34.40% coverage).
3. **Spatial Aggregation**: Citywide hourly series were computed by taking the spatial mean across reporting stations for each pollutant-hour, requiring a quorum of at least 3 stations. Hours with fewer than 3 reporting stations were flagged as missing.

### 2.3 Variables, Units, and Column Roles
Raw CSV headers were mapped directly to standardized semantic roles via `COLUMN_MAP` (`src/config.py`):

| Variable Role | Raw CPCB Header | Measurement Unit | Category | Status |
|:---|:---|:---:|:---:|:---:|
| `pm25` | `PM2.5 (µg/m³)` | $\mu\text{g/m}^3$ | Criteria Pollutant | Retained |
| `pm10` | `PM10 (µg/m³)` | $\mu\text{g/m}^3$ | Criteria Pollutant | Retained |
| `no` | `NO (µg/m³)` | $\mu\text{g/m}^3$ | Reactive Nitrogen | Retained |
| `no2` | `NO2 (µg/m³)` | $\mu\text{g/m}^3$ | Criteria Pollutant | Retained |
| `nox` | `NOx (ppb)` | $\text{ppb}$ | Reactive Nitrogen | Retained |
| `nh3` | `NH3 (µg/m³)` | $\mu\text{g/m}^3$ | Criteria Pollutant | Retained |
| `so2` | `SO2 (µg/m³)` | $\mu\text{g/m}^3$ | Criteria Pollutant | Retained |
| `co` | `CO (mg/m³)` | $\text{mg/m}^3$ | Criteria Pollutant | Retained (kept in $\text{mg/m}^3$) |
| `o3` | `Ozone (µg/m³)` | $\mu\text{g/m}^3$ | Criteria Pollutant | Retained |
| `benzene` | `Benzene (µg/m³)` | $\mu\text{g/m}^3$ | VOC | Retained |
| `toluene` | `Toluene (µg/m³)` | $\mu\text{g/m}^3$ | VOC | Retained |
| `temperature` | `AT (°C)` | $^\circ\text{C}$ | Meteorology | Retained |
| `humidity` | `RH (%)` | $\%$ | Meteorology | Retained |
| `wind_speed` | `WS (m/s)` | $\text{m/s}$ | Meteorology | Retained |
| `wind_direction` | `WD (deg)` | $\text{degrees}$ | Meteorology | Retained |
| `rainfall` | `RF (mm)` | $\text{mm}$ | Meteorology | Retained |
| `solar_radiation` | `SR (W/mt2)` | $\text{W/m}^2$ | Meteorology | Retained |
| `barometric_pressure` | `BP (mmHg)` | $\text{hPa}$ | Meteorology | Converted to $\text{hPa}$ |
| *Dropped Columns* | `Xylene, O-Xylene, Eth-Benzene, MP-Xylene, VWS` | Various | Trace Gases / Vertical Wind | Dropped (>80% null) |

### 2.4 Target Formulation: National Air Quality Index (NAQI)
The primary target is the daily citywide Indian National Air Quality Index (NAQI, scale 0 to 500). In accordance with CPCB guidelines:
1. Sub-indices $I_i$ are calculated for criteria pollutants ($PM_{2.5}$, $PM_{10}$, $NO_2$, $NH_3$, $SO_2$, $CO$, $O_3$) using piecewise linear concentration breakpoint equations.
2. Particulates and gaseous criteria pollutants use a 24-hour mean concentration (requiring $\ge 16$ valid hourly readings).
3. Ozone and Carbon Monoxide use 8-hour trailing rolling averages (requiring $\ge 6$ valid hours per 8-hour window); the daily representative value is the maximum 8-hour value of the calendar day.
4. An overall daily AQI is valid if and only if at least 3 criteria pollutants have valid sub-indices, of which at least one must be a particulate ($PM_{2.5}$ or $PM_{10}$).
5. The overall AQI is the maximum of the valid sub-indices: $\text{AQI} = \max_i(I_i)$.

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


---

## 3. Preprocessing and Exploratory Data Analysis (EDA)

### 3.1 Data Cleaning Decisions and Physical Bounds
Data cleaning was systematically applied and logged in `outputs/cleaning_log.csv`:
1. **Barometric Pressure Normalization**: Raw pressure measurements in CPCB exports were recorded with mixed units (ambient $\text{mmHg}$ and $\text{hPa}$). Readout values below 850 were identified as $\text{mmHg}$ and converted to $\text{hPa}$ ($P_{\text{hPa}} = P_{\text{mmHg}} \times 1.333224$). Exactly 167594 readings were converted. Physical bounds for Delhi's elevation (~216 m above sea level) were established at [940, 1030] hPa; 17702 readings falling outside this window were set to NaN.
2. **Physical and Instrument Plausibility Caps**: Extreme unphysical spikes were clipped to atmospheric and instrument limits:
   - Ozone ($O_3$): clipped to [0, 500] $\mu\text{g/m}^3$ (12 readings clipped; maximum raw spike was 970.0).
   - Benzene: clipped to [0, 200] $\mu\text{g/m}^3$ (125 readings clipped; maximum raw spike was 476.4).
   - Ambient Temperature: bounded within [0, 52] $^\circ\text{C}$ (40 readings clipped; maximum raw spike was 58.9).
   - Wind Speed: bounded within [0, 30] $\text{m/s}$ (108 readings clipped; maximum raw spike was 47.75).
   - Surface Solar Radiation: bounded by [0, 1300] $\text{W/m}^2$ (353 readings clipped; maximum raw spike was 1985.0).
3. **Stuck-Sensor Invalidation**: Sensor transmission faults can freeze telemetry, outputting constant non-zero values. An automated audit invalidated episodes where a sensor outputted the exact same non-zero value for $\ge 24$ consecutive hours:
   - Barometric pressure: 70753 readings.
   - Wind speed: 27347 readings.
   - Benzene: 20006 readings.
   - Toluene: 15767 readings.
   - Relative humidity: 8593 readings.
   - Carbon monoxide: 8197 readings.
   - Wind direction: 6093 readings.
   - Solar radiation: 5851 readings.
   - Criteria gases & particulates: $NO_2$ (2485), $PM_{2.5}$ (2272), $O_3$ (1269), $SO_2$ (1256), $PM_{10}$ (1130), Temperature (1002), $NO$ (997), $NH_3$ (585), $NO_x$ (376).
4. **Particulate Ratio Inversion Filter**: By physical definition, $PM_{2.5}$ is a subset of $PM_{10}$. A tolerance multiplier of 1.2 accounts for optical instrument uncertainty. Station hours exhibiting $PM_{2.5} > 1.2 \times PM_{10}$ with $PM_{2.5} > 20\,\mu\text{g/m}^3$ were flagged as unphysical instrument cross-talk; exactly 3691 station-hours (7382 individual values) were set to NaN. Sensor drift during high-humidity episodes is a possible explanation (not tested).
5. **Transient VOC Spike Filter**: Isolated VOC readings exceeding 10 times their rolling 24-hour median (baseline $> 0.1\,\mu\text{g/m}^3$) were treated as localized sensor noise and set to NaN (1888 benzene, 3150 toluene).
6. **Station-Level Imputation**: Missing values within isolated gaps $\le 6$ hours were linearly interpolated at the station level prior to spatial averaging.

### 3.2 Exploratory Data Analysis (EDA) Highlights
Key exploratory visualizations are preserved in `outputs/eda/`:
1. **Daily AQI Time Series (2018–2026)** (`outputs/eda/01_daily_aqi_over_time.png`): Demonstrates recurring seasonal cycles, with winter AQI frequently reaching Very Poor and Severe bands.
2. **Monthly AQI Distributions & Heatmaps** (`outputs/eda/02_aqi_by_month_and_heatmap.png`): Shows November and December having median AQI $> 320$, while July and August have median AQI $< 100$.
3. **Missing Value Auditing Before & After Imputation** (`outputs/eda/03_missing_values_by_year_and_column.png`): Quantifies coverage improvements from 6-hour linear interpolation and multi-station quorum averaging.
4. **Correlation Structure** (`outputs/eda/04_correlation_heatmap.png`): Shows strong positive correlations ($>0.75$) among $PM_{2.5}$, $PM_{10}$, $NO_2$, and $CO$.
5. **AQI Category Distribution** (`outputs/eda/05_aqi_category_distribution.png`): Illustrates the high historical prevalence of Poor, Very Poor, and Severe days.
6. **Autocorrelation and Partial Autocorrelation** (`outputs/eda/06_acf_pacf_lags_1_to_10.png`): Shows day-to-day persistence ($r_1 > 0.85$) with PACF dropping sharply after lag 1.
7. **Meteorological Effects** (`outputs/eda/07_binned_wind_speed_temp_effects.png`): Binned wind speed analysis shows higher Severe day frequency when surface winds drop below 1.5 m/s.
8. **Prominent Pollutant Shares Across Years** (`outputs/eda/08_prominent_pollutant_share_by_year.png`): Confirms that $PM_{2.5}$ and $PM_{10}$ determine the daily AQI on over 95% of calendar days across all audited years.

[TODO: me]


---

## 4. Feature Engineering and Targets

### 4.1 Feature Set Architecture
Exactly 38 features were engineered (`models/feature_columns.json`) across four families:

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
| `pm25` | Criteria Pollutants | Today's daily mean $PM_{2.5}\,(\mu\text{g/m}^3)$ |
| `pm25_rolling_mean_3d` | Criteria Pollutants | Trailing 3-day mean $PM_{2.5}$ |
| `pm10` | Criteria Pollutants | Today's daily mean $PM_{10}\,(\mu\text{g/m}^3)$ |
| `pm10_rolling_mean_3d` | Criteria Pollutants | Trailing 3-day mean $PM_{10}$ |
| `no2` | Criteria Pollutants | Today's daily mean $NO_2\,(\mu\text{g/m}^3)$ |
| `no2_rolling_mean_3d` | Criteria Pollutants | Trailing 3-day mean $NO_2$ |
| `so2` | Criteria Pollutants | Today's daily mean $SO_2\,(\mu\text{g/m}^3)$ |
| `so2_rolling_mean_3d` | Criteria Pollutants | Trailing 3-day mean $SO_2$ |
| `co` | Criteria Pollutants | Today's daily max 8h $CO\,(\text{mg/m}^3)$ |
| `co_rolling_mean_3d` | Criteria Pollutants | Trailing 3-day max 8h $CO$ |
| `o3` | Criteria Pollutants | Today's daily max 8h $O_3\,(\mu\text{g/m}^3)$ |
| `o3_rolling_mean_3d` | Criteria Pollutants | Trailing 3-day max 8h $O_3$ |
| `pm25_max` | Criteria Pollutants | Daily maximum hourly $PM_{2.5}$ |
| `pm10_max` | Criteria Pollutants | Daily maximum hourly $PM_{10}$ |
| `temperature` | Meteorology | Today's daily mean temperature ($^\circ\text{C}$) |
| `humidity` | Meteorology | Today's daily mean relative humidity (%) |
| `wind_speed` | Meteorology | Today's daily mean wind speed (m/s) |
| `barometric_pressure` | Meteorology | Today's daily mean pressure (hPa) |
| `wind_dir_sin` | Meteorology | $\sin(2\pi \times \text{WD} / 360)$ |
| `wind_dir_cos` | Meteorology | $\cos(2\pi \times \text{WD} / 360)$ |
| `temperature_rolling_mean_3d` | Meteorology | Trailing 3-day mean temperature |
| `humidity_rolling_mean_3d` | Meteorology | Trailing 3-day mean humidity |
| `wind_speed_rolling_mean_3d` | Meteorology | Trailing 3-day mean wind speed |
| `month` | Calendar / Cyclical | Calendar month index ($1 \dots 12$) |
| `day_of_year_sin` | Calendar / Cyclical | $\sin(2\pi \times \text{DOY} / 365.25)$ |
| `day_of_year_cos` | Calendar / Cyclical | $\cos(2\pi \times \text{DOY} / 365.25)$ |
| `day_of_week` | Calendar / Cyclical | Day of week index ($0 \dots 6$) |
| `is_weekend` | Calendar / Cyclical | Binary indicator for Saturday/Sunday |
| `stubble_season` | Calendar / Cyclical | Indicator for Oct 15 – Nov 30 window |

### 4.2 Multi-Horizon Target Formulation
Separate supervised models are trained for each forecast horizon $h \in \{1, 2, 3, 4, 5\}$:
$$\hat{y}_{t+h} = f_h(\mathbf{X}_t)$$
where $\mathbf{X}_t \in \mathbb{R}^{38}$ represents features observed on or before day $t$, and $y_{t+h} \in [0, 500]$ is the verified CPCB AQI observed $h$ days later.

### 4.3 Leakage Prevention and Automated Tests
Leakage prevention rules:
1. **Trailing Lags Only**: Features use only trailing information ($t' \le t$). No centered rolling windows and no negative shifts in feature columns.
2. **Negative Shifts Strictly for Targets**: Negative shifts (e.g. `df['target'] = df['aqi'].shift(-h)`) are used exclusively for target construction.
3. **Preprocessing on Training Data Only**: Scalers, imputers, and climatology lookup tables are fit on training data only.
4. **Automated Leakage Test Suite**: `tests/test_no_leakage.py` runs three continuous checks:
   - `test_no_target_in_feature_columns`: Confirms no target column appears in $\mathbf{X}_t$.
   - `test_no_negative_shifts_or_centered_windows`: Scans codebase AST to confirm no negative shifts or centered windows in feature construction.
   - `test_temporal_leakage_20_random_dates`: Selects 20 random dates, modifies future data after date $t$, and asserts that $\mathbf{X}_t$ remains identical. All tests pass with zero warnings.

[TODO: me]


---

## 5. Models and Experimental Setup

### 5.1 Chronological Partitioning and Buffer Gaps
In accordance with Rule 2 (Never shuffle time series; chronological splits with a 5-day boundary buffer gap; test set reserved exclusively for final evaluation), the dataset was partitioned chronologically (`outputs/split_summary.md`, `outputs/split_diagram.png`):

| Partition | Start Date | End Date | Calendar Days | Days Dropped in Gap | Description |
|:---|:---:|:---:|:---:|:---:|:---|
| **Training** | 2018-01-08 | 2023-03-31 | **1887** | — | Training data (~5.25 years) |
| *Gap 1* | 2023-04-01 | 2023-04-05 | **5** | 5 | Excludes training horizon overlap into validation |
| **Validation** | 2023-04-06 | 2024-03-31 | **352** | — | Validation data (1 full calendar year) |
| *Gap 2* | 2024-04-01 | 2024-04-05 | **5** | 5 | Excludes validation horizon overlap into test |
| **Test Holdout** | 2024-04-06 | 2026-09-25 | **887** | — | Final evaluation holdout (~2.5 years) |
| **Total** | 2018-01-08 | 2026-09-25 | **3136** | **10** | 3,146 total calendar days accounted for |

### 5.2 Category Distribution and Non-Stationary Shift
Distribution of CPCB AQI health categories across partitions (`outputs/split_summary.md`):

| CPCB Category | Range | Training Set (1887 days) | Validation Set (352 days) | Test Holdout (887 days) |
|:---|:---:|:---:|:---:|:---:|
| **Good** | 0 – 50 | 18 (0.95%) | 0 (0.00%) | 6 (0.68%) |
| **Satisfactory** | 51 – 100 | 355 (18.81%) | 56 (15.91%) | 184 (20.74%) |
| **Moderate** | 101 – 200 | 590 (31.27%) | 143 (40.62%) | 366 (41.26%) |
| **Poor** | 201 – 300 | 317 (16.80%) | 38 (10.80%) | 122 (13.75%) |
| **Very Poor** | 301 – 400 | 475 (25.17%) | 90 (25.57%) | 165 (18.60%) |
| **Severe** | 401 – 500 | 132 (7.00%) | 25 (7.10%) | 44 (4.96%) |
| **Poor-or-worse (AQI > 200)** | > 200 | **924 (48.97%)** | **153 (43.47%)** | **331 (37.32%)** |

> **Audit Note on Target Distribution**: The test period as having lower pollution levels than the training period is documented in the split summary without asserting a cause: Poor-or-worse days ($\text{AQI} > 200$) accounted for 48.97% of the training set but 37.32% in the test holdout; Severe days accounted for 7.00% in training and 4.96% in the test holdout. Additionally, the validation set contained zero Good days and the test set contained only 6 Good days (0.68%), rendering single-class Good metrics statistically unreliable.

### 5.3 Cross-Validation and Hyperparameter Tuning
Hyperparameter optimization was conducted strictly on the training set using an expanding-window `TimeSeriesSplit(n_splits=4, gap=5)` (`src/config.py`).
- **CV Configuration Note**: A 5-fold split was audited during setup but rejected as Fold 1 trained on only 312 days with no complete winter; configuring `CV_SPLITS = 4` with `CV_GAP = 5` ensured that every fold encompassed a complete winter season.
- **Randomized Search**: Hyperparameters were tuned via `RandomizedSearchCV` (30 iterations, scoring `neg_mean_absolute_error`, seed 42) searching:
  - `n_estimators`: [200, 400, 600]
  - `max_depth`: [None, 6, 10, 16]
  - `min_samples_leaf`: [1, 3, 5, 10]
  - `max_features`: [0.3, 0.5, 0.8, 1.0]

### 5.4 Target Transformation Audit: Raw vs. Log Target
Tuning was executed under two target paradigms:
1. **Raw Target**: Regressors trained directly on original AQI scale $\hat{y} = f(\mathbf{X})$.
2. **Log Target**: Regressors trained on $z = \log(1 + \text{AQI})$, with predictions converted back with $\hat{y} = \exp(\hat{z}) - 1$ prior to evaluation on the original scale.

Validation performance (`models/rf_config.json`):

| Horizon | Raw Target Val MAE | Log Target Val MAE | Margin Over Log | Winning Scale | Best Parameters |
|:---:|:---:|:---:|:---:|:---:|:---|
| **$h=1$** | **30.88** | 30.92 | +0.04 | **Raw** | `n_estimators=400, max_depth=10, min_samples_leaf=5, max_features=0.5` |
| **$h=2$** | **38.8** | 41.17 | +2.38 | **Raw** | `n_estimators=400, max_depth=16, min_samples_leaf=10, max_features=0.3` |
| **$h=3$** | **43.21** | 45.65 | +2.45 | **Raw** | `n_estimators=400, max_depth=16, min_samples_leaf=10, max_features=0.3` |
| **$h=4$** | **44.53** | 46.79 | +2.26 | **Raw** | `n_estimators=200, max_depth=None, min_samples_leaf=10, max_features=0.3` |
| **$h=5$** | **44.93** | 47.61 | +2.68 | **Raw** | `n_estimators=400, max_depth=16, min_samples_leaf=10, max_features=0.3` |

At all five horizons, the raw target scale achieved lower validation MAE than the log transformation. Re-transformation bias is a likely explanation (not tested). The raw target scale was chosen for all models.

### 5.5 Benchmark Models in Scope
1. **Persistence Baseline**: $\hat{y}_{t+h} = y_t$.
2. **Climatology Baseline**: Circular moving-window mean ($\pm 7$ days) from training history: $\hat{y}_{t+h} = \bar{y}_{\text{doy}(t+h) \pm 7\text{d}}^{\text{train}}$.
3. **Ridge Regression**: Tuned with regularization $\alpha = 100.0$ on validation data.
4. **Random Forest Regressor**: Horizon-specific non-linear ensemble with best parameters locked from validation tuning.

### 5.6 Final Evaluation Protocol
Models were refit on combined Training + Validation data (gap rows strictly excluded) with chosen settings. The final test evaluation script (`src/final_evaluate.py`) was run once on the holdout test set.

[TODO: me]


---

## 6. Results

### 6.1 Multi-Horizon Benchmark Performance
Full test set metrics from `outputs/metrics.csv` across all models and horizons:

| Model | Horizon | MAE | RMSE | $R^2$ | Mean Bias | Accuracy | Within-1-Cat Acc | Alert Prec | Alert Rec | Skill vs Best Base |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Persistence** | $h=1$ | 30.97 | 44.6 | 0.836 | +0.15 | 0.693 | 0.983 | 0.852 | 0.852 | 0.0 |
| **Climatology** | $h=1$ | 40.37 | 54.23 | 0.757 | +8.46 | 0.608 | 0.971 | 0.819 | 0.779 | -0.303 |
| **Ridge** | $h=1$ | 28.49 | 39.11 | 0.874 | -2.46 | 0.693 | 0.993 | 0.858 | 0.858 | +0.08 |
| **Random Forest** | $h=1$ | **27.57** | **38.84** | **0.876** | +3.02 | **0.71** | **0.99** | 0.842 | **0.87** | **+0.11** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Persistence** | $h=2$ | 42.57 | 59.31 | 0.71 | +0.34 | 0.61 | 0.958 | 0.789 | 0.789 | -0.053 |
| **Climatology** | $h=2$ | 40.43 | 54.28 | 0.758 | +8.56 | 0.607 | 0.97 | 0.819 | 0.779 | 0.0 |
| **Ridge** | $h=2$ | 37.01 | 49.95 | 0.795 | -5.24 | 0.638 | 0.981 | 0.842 | 0.819 | +0.084 |
| **Random Forest** | $h=2$ | **36.34** | **49.29** | **0.8** | +2.02 | **0.649** | **0.982** | 0.821 | **0.831** | **+0.101** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Persistence** | $h=3$ | 47.25 | 64.95 | 0.654 | +0.53 | 0.565 | 0.946 | 0.764 | 0.764 | -0.166 |
| **Climatology** | $h=3$ | 40.52 | 54.34 | 0.758 | +8.66 | 0.607 | 0.968 | 0.819 | 0.779 | 0.0 |
| **Ridge** | $h=3$ | 39.71 | 53.56 | 0.764 | -7.67 | 0.599 | 0.976 | 0.833 | 0.77 | +0.02 |
| **Random Forest** | $h=3$ | **38.75** | **52.8** | **0.771** | +1.34 | **0.612** | **0.976** | 0.809 | 0.779 | **+0.044** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Persistence** | $h=4$ | 49.81 | 68.3 | 0.618 | +0.66 | 0.538 | 0.941 | 0.749 | 0.749 | -0.229 |
| **Climatology** | $h=4$ | 40.53 | 54.34 | 0.758 | +8.7 | 0.605 | 0.968 | 0.819 | 0.779 | 0.0 |
| **Ridge** | $h=4$ | **40.09** | 54.52 | 0.756 | -4.31 | **0.618** | **0.973** | 0.825 | 0.77 | +0.011 |
| **Random Forest** | $h=4$ | 40.26 | **54.35** | **0.758** | +3.02 | 0.598 | 0.971 | 0.781 | 0.752 | +0.007 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Persistence** | $h=5$ | 52.61 | 70.62 | 0.591 | +0.79 | 0.524 | 0.938 | 0.734 | 0.736 | -0.299 |
| **Climatology** | $h=5$ | **40.49** | **54.32** | **0.758** | +8.74 | 0.605 | 0.968 | 0.819 | 0.782 | 0.0 |
| **Ridge** | $h=5$ | 40.99 | 55.7 | 0.746 | -0.54 | **0.608** | **0.97** | 0.799 | 0.77 | -0.012 |
| **Random Forest** | $h=5$ | 41.48 | 55.72 | 0.746 | +5.28 | 0.589 | 0.97 | 0.761 | 0.764 | -0.024 |

### 6.2 Skill vs. Best Baseline
The best baseline is defined as the better of Persistence and Climatology at the same horizon:
$$\text{Skill} = 1 - \frac{\text{MAE}_{\text{model}}}{\min(\text{MAE}_{\text{persistence}}, \text{MAE}_{\text{climatology}})}$$

- **$h=1$**: Random Forest achieves **+11.0%** skill over Persistence.
- **$h=2$**: Random Forest achieves **+10.1%** skill over Climatology.
- **$h=3$**: Random Forest achieves **+4.4%** skill over Climatology.
- **$h=4$**: Random Forest achieves **+0.7%** skill over Climatology.
- **$h=5$**: Random Forest achieves **-2.4%** skill over Climatology.

### 6.3 Paired Moving-Block Bootstrap Confidence Intervals (95% CI, 7-Day Blocks, $B=2000$)

#### A. Random Forest minus Best Baseline (`outputs/rf_vs_baseline_ci.csv`)
| Horizon | Best Baseline | RF MAE | Baseline MAE | Mean Difference | 95% Bootstrap CI | Statistical Conclusion |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **$h=1$** | Persistence | 27.57 | 30.97 | -3.41 | **[-4.54, -2.3]** | **Statistically significant improvement** (excludes 0) |
| **$h=2$** | Climatology | 36.34 | 40.43 | -4.09 | **[-6.26, -1.98]** | **Statistically significant improvement** (excludes 0) |
| **$h=3$** | Climatology | 38.75 | 40.52 | -1.78 | [-3.97, +0.18] | **Statistically indistinguishable** (includes 0) |
| **$h=4$** | Climatology | 40.26 | 40.53 | -0.27 | [-2.42, +1.66] | **Statistically indistinguishable** (includes 0) |
| **$h=5$** | Climatology | 41.48 | 40.49 | +0.99 | [-0.92, +2.76] | **Statistically indistinguishable** (includes 0) |

#### B. Random Forest minus Ridge (`outputs/rf_vs_ridge_ci.csv`)
| Horizon | RF MAE | Ridge MAE | Mean Difference | 95% Bootstrap CI | Statistical Conclusion |
|:---:|:---:|:---:|:---:|:---:|:---|
| **$h=1$** | 27.57 | 28.49 | -0.92 | **[-1.87, -0.03]** | **Just excludes zero** (using real numbers) |
| **$h=2$** | 36.34 | 37.01 | -0.67 | [-2.06, +0.61] | **Statistically indistinguishable** (includes 0) |
| **$h=3$** | 38.75 | 39.71 | -0.96 | [-2.86, +0.81] | **Statistically indistinguishable** (includes 0) |
| **$h=4$** | 40.26 | 40.09 | +0.16 | [-1.8, +1.95] | **Statistically indistinguishable** (includes 0) |
| **$h=5$** | 41.48 | 40.99 | +0.49 | [-1.6, +2.5] | **Statistically indistinguishable** (includes 0) |

> **Headline Summary on Medium Horizons**: At 3 to 5 days it is statistically indistinguishable from climatology (intervals include zero) and from Ridge.

### 6.4 Alert Detection Benchmark Next to Trivial Seasonal Rule
An alert day is defined as $\text{AQI} > 200$.
- Across the 887-day test set, exactly 331 days had $\text{AQI} > 200$ (base rate 37.3%).
- During winter (Oct–Feb, 302 days), 232 were alert days (base rate 76.8%).
- During non-winter (Mar–Sep, 585 days), 99 were alert days (base rate 16.9%).

Trivial seasonal rule: **"Alert on every winter day (Oct–Feb), never in non-winter"**:
- Full test set performance: Precision = 232 / 302 = 0.768; Recall = 232 / 331 = 0.701.
- In winter specifically: Recall = 1.000, Precision = 0.768.

Comparison with Random Forest on full test set (`outputs/error_analysis.md`):

| Horizon | RF Alert Precision | RF Alert Recall | Trivial Winter Precision | Trivial Winter Recall | RF Non-Winter Recall |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$h=1$** | **0.842** | **0.870** | 0.768 | 0.701 | 0.616 |
| **$h=2$** | **0.821** | **0.831** | 0.768 | 0.701 | 0.515 |
| **$h=3$** | 0.809 | 0.779 | 0.768 | 0.701 | 0.384 |
| **$h=4$** | 0.781 | 0.752 | 0.768 | 0.701 | 0.333 |
| **$h=5$** | 0.761 | 0.764 | 0.768 | 0.701 | 0.343 |

Winter alert recall differences: 0.752 vs 0.749 at day +4 and 0.764 vs 0.752 at day +5.

> **Plain Words Finding**: At days 3 to 5 the alert detection is not better than the trivial rule "alert on every winter day"; at days 3 to 5 Random Forest has no demonstrated advantage over the trivial winter rule (lower precision, recall gap within about one point, no interval computed).

### 6.5 Category Confusion Matrices
Category classification performance across the six CPCB health bands for Random Forest at $h=1$ and $h=5$ (`outputs/confusion_matrices.png`, `outputs/confusion_matrix_rf_h1.png`, `outputs/confusion_matrix_rf_h5.png`):

#### Random Forest $h=1$ (Test Holdout, $N=887$)
| True \ Predicted | Good | Satisfactory | Moderate | Poor | Very Poor | Severe | Total True |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Good** | 0 | 6 | 0 | 0 | 0 | 0 | **6** |
| **Satisfactory** | 0 | 138 | 48 | 0 | 0 | 0 | **186** |
| **Moderate** | 0 | 28 | 282 | 50 | 4 | 0 | **364** |
| **Poor** | 0 | 1 | 38 | 69 | 14 | 0 | **122** |
| **Very Poor** | 0 | 0 | 3 | 34 | 121 | 7 | **165** |
| **Severe** | 0 | 0 | 1 | 0 | 23 | 20 | **44** |

- Exact Category Accuracy: **71.0%**
- Within-One-Category Accuracy: **99.0%**

#### Random Forest $h=5$ (Test Holdout, $N=887$)
| True \ Predicted | Good | Satisfactory | Moderate | Poor | Very Poor | Severe | Total True |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Good** | 0 | 5 | 3 | 0 | 0 | 0 | **8** |
| **Satisfactory** | 0 | 103 | 82 | 3 | 0 | 0 | **188** |
| **Moderate** | 0 | 33 | 252 | 68 | 8 | 0 | **361** |
| **Poor** | 0 | 0 | 67 | 39 | 14 | 1 | **121** |
| **Very Poor** | 0 | 0 | 10 | 27 | 125 | 3 | **165** |
| **Severe** | 0 | 0 | 1 | 1 | 39 | 3 | **44** |

- Exact Category Accuracy: **58.9%**
- Within-One-Category Accuracy: **97.0%**

### 6.6 Error vs. Horizon Trajectory
Visualized in `outputs/error_vs_horizon.png` and `outputs/alert_metrics_vs_horizon.png`:
- At $h=1$, Random Forest achieves an MAE of 27.57, outperforming Persistence (30.97) and Climatology (40.37).
- At $h=3$, Random Forest MAE reaches 38.75, approaching Climatology (40.52).
- At $h=5$, Random Forest MAE reaches 41.48, matching Climatology (40.49) and Ridge (40.99).
- Within-one-category accuracy remains $\ge 97\%$ across all horizons.

[TODO: me]


---

## 7. Error Analysis

### 7.1 Performance by Calendar Month (`outputs/error_by_month.png`)
Monthly error trends (`outputs/error_analysis.md`):
- **Winter**: Monthly MAE is highest during November and December (averaging between 45 and 65 AQI points).
- **Summer / Monsoon Calendar Period**: Monthly MAE is lowest during the summer/monsoon calendar period (July to September, averaging below 25 AQI) across all horizons.
- **Horizon Contrast**: Random Forest maintains an error advantage over the best baseline during early horizons across both seasons, which narrows to parity by horizon $h=5$.

### 7.2 Winter vs. Non-Winter Performance Breakdown (`outputs/winter_split_ci.csv`)
Evaluating winter (302 days: Oct–Feb) vs non-winter (585 days: Mar–Sep):

| Horizon | Season | Days | RF MAE | Baseline MAE | Mean Difference | 95% Bootstrap CI | Statistical Conclusion |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **$h=1$** | **Winter** | 302 | 34.24 | 39.24 | -5.0 | **[-7.57, -2.79]** | RF significantly better |
| **$h=1$** | **Non-Winter** | 585 | 24.12 | 26.7 | -2.58 | **[-3.84, -1.48]** | RF significantly better |
| **$h=2$** | **Winter** | 302 | 46.91 | 50.92 | -4.01 | **[-7.86, -0.51]** | RF significantly better |
| **$h=2$** | **Non-Winter** | 585 | 30.89 | 35.02 | -4.13 | **[-6.83, -1.75]** | RF significantly better |
| **$h=3$** | **Winter** | 302 | 48.93 | 50.91 | -1.98 | [-5.41, +1.43] | Indistinguishable from baseline |
| **$h=3$** | **Non-Winter** | 585 | 33.49 | 35.16 | -1.67 | [-4.29, +0.6] | Indistinguishable from baseline |
| **$h=4$** | **Winter** | 302 | 50.1 | 51.2 | -1.09 | [-4.75, +2.4] | Indistinguishable from baseline |
| **$h=4$** | **Non-Winter** | 585 | 35.18 | 35.02 | +0.15 | [-2.37, +2.36] | Indistinguishable from baseline |
| **$h=5$** | **Winter** | 302 | 51.47 | 51.49 | -0.01 | [-3.28, +3.31] | Indistinguishable from baseline |
| **$h=5$** | **Non-Winter** | 585 | 36.32 | 34.82 | +1.5 | [-0.89, +3.45] | Indistinguishable from baseline |

### 7.3 Audit of Worst Prediction Days (`outputs/worst_days_h1.csv`, `outputs/worst_days_h5.csv`)

#### 1. Non-Winter Target Days (2025-05-15 and 2026-07-12)
1. **Target Day 2025-05-15**:
   - Forecast 2025-05-14 predicted 177.44; actual AQI was **411.72** (Severe), error **234.28**.
   - Day before AQI (2025-05-14): 164.01.
   - Daily Mean $PM_{2.5}$: 210.13\,\mu\text{g/m}^3; Daily Mean $PM_{10}$: 439.37\,\mu\text{g/m}^3.
   - $PM_{2.5} / PM_{10}$ Ratio: 47.82%; Prominent Pollutant: $PM_{10}$.
   - *Cause Note*: The cause of this single-day jump of +247.71 AQI points is unknown from repository files; potential environmental causes are unverified hypotheses (not tested).
2. **Target Day 2026-07-12**:
   - Forecast 2026-07-11 predicted 160.67; actual AQI was **336.04** (Very Poor), error **175.37**.
   - Day before AQI (2026-07-11): 204.63.
   - Daily Mean $PM_{2.5}$: 68.41\,\mu\text{g/m}^3; Daily Mean $PM_{10}$: 378.83\,\mu\text{g/m}^3.
   - $PM_{2.5} / PM_{10}$ Ratio: 18.06%; Prominent Pollutant: $PM_{10}$.
   - *Cause Note*: The cause of this particulate rise is unknown from repository files; potential environmental causes are unverified hypotheses (not tested).

#### 2. Festival Proximity Audit (Target Date Window: [-1, +3] Days Around Diwali)
- Across all 50 evaluated worst-day predictions (10 worst $\times$ 5 horizons), exactly 4 days fall within the Diwali window ($-1 \le \text{days\_from\_diwali} \le +3$).
- All four occurrences target the exact same day: **target date 2025-10-21** (one day after unverified Diwali date 2025-10-20; actual AQI was 454.66, predicted between 275.94 and 316.66).
- **Conclusion**: All 4 Diwali-window observations are the same target day (2025-10-21), so this is one event seen at four horizons.

### 7.4 Time Series and Residual Visualizations
1. **Winter Time Series Tracking** (`outputs/winter_timeseries_h1_h5.png`):
   - At $h=1$, Random Forest tracks the seasonal trajectory, though the model may under-predict the most extreme days (explanation not tested).
   - At $h=5$, predictions track the mid-winter baseline rather than day-to-day transient spikes.
2. **Residual Diagnostic Distributions** (`outputs/residuals_h1_h5.png`):
   - Residual is defined strictly as prediction minus actual ($\text{residual} = \hat{y} - y$), where positive values denote over-prediction and negative values denote under-prediction.
   - Histograms show distributions centered near zero with mild positive mean residuals (+3.02 for $h=1$, +5.28 for $h=5$).
   - Scatter plots display an upward trend with widening variance as predicted AQI rises, indicating under-prediction when low AQI is predicted and over-prediction when high AQI is predicted.

[TODO: me]


---

## 8. Explainability

### 8.1 SHAP Attribution Framework
Model interpretability was evaluated using SHAP (SHapley Additive exPlanations) via `shap.TreeExplainer` on the test set (`models/rf_h1.pkl`, `rf_h3.pkl`, `rf_h5.pkl`). Additivity was verified in `tests/test_shap.py`:
$$\hat{y}(\mathbf{x}) = \phi_0 + \sum_{j=1}^{38} \phi_j(\mathbf{x})$$
matching model predictions within $10^{-5}$ across test rows.

> **Credit Splitting Across Correlated Features**: As criteria pollutants are collinear ($PM_{2.5}$, $PM_{2.5}$ 3d mean, AQI, AQI 3d mean, $PM_{2.5,\max}$), TreeExplainer partitions attribution across them; individual feature ranks should be interpreted alongside family totals.

### 8.2 Systematic Horizon Shift Across Feature Families
Grouping features into four families (`outputs/shap_family_importance.csv`, `outputs/shap_family_importance.png`, `outputs/shap_family_bar.png`):

| Horizon | Family | Mean Absolute SHAP (AQI Points) | Attribution Share (%) |
|:---:|:---|:---:|:---:|
| **$h=1$** | **Pollutants** | 59.92 | **61.6%** |
| **$h=1$** | **AQI History** | 27.96 | **28.7%** |
| **$h=1$** | **Weather** | 5.66 | **5.8%** |
| **$h=1$** | **Calendar** | 3.73 | **3.8%** |
|---|---|---|---|
| **$h=3$** | **Pollutants** | 38.63 | **40.9%** |
| **$h=3$** | **AQI History** | 28.79 | **30.5%** |
| **$h=3$** | **Weather** | 8.26 | **8.8%** |
| **$h=3$** | **Calendar** | 18.65 | **19.8%** |
|---|---|---|---|
| **$h=5$** | **Pollutants** | 33.13 | **35.7%** |
| **$h=5$** | **AQI History** | 23.99 | **25.8%** |
| **$h=5$** | **Weather** | 7.25 | **7.8%** |
| **$h=5$** | **Calendar** | 28.54 | **30.7%** |

> **Headline Interpretability Finding**: At $h=1$, Pollutants (61.6%) + AQI history (28.7%) comprise 90.3% of total attribution, while Calendar features comprise only 3.8%. By $h=5$, Calendar expands to 30.7% and Pollutants drop to 35.7%; the model's attribution shifts from daily readings toward calendar features as the horizon grows.

### 8.3 Beeswarm Summaries and Individual Feature Ranks
1. **Horizon $h=1$** (`outputs/shap_summary_h1.png`, `outputs/shap_bar_h1.png`): Today's readings (`pm25`, `aqi`, `pm25_max`) display the highest mean absolute SHAP values. High surface wind speeds and negative `day_of_year_cos` values lower the forecast.
2. **Horizon $h=3$** (`outputs/shap_summary_h3.png`, `outputs/shap_bar_h3.png`): Rolling multi-day aggregates (`pm25_rolling_mean_3d`, `aqi_rolling_mean_7d`) gain relative prominence over instantaneous daily readings. Cyclical calendar variables (`day_of_year_cos`, `day_of_year_sin`) become top predictors.
3. **Horizon $h=5$** (`outputs/shap_summary_h5.png`, `outputs/shap_bar_h5.png`): Seasonal calendar variable `day_of_year_cos` emerges as the leading feature by mean absolute SHAP value, followed by rolling means `pm25_rolling_mean_3d` and `aqi_rolling_mean_7d`. Instantaneous daily features drop in attribution rank, showing that 5-day forecasts are primarily governed by seasonal position.

### 8.4 Severe Winter Day Case Studies (Smallest and Largest Error Cases)
Evaluated on Severe winter test days (truth $> 400$ AQI). The good and bad cases are the Severe winter days with the smallest and largest error, so they are best and worst cases, not typical ones (`outputs/shap_notes.md`):

1. **$h=1$ Smallest Error Case** (`outputs/shap_waterfall_h1_good.png`):
   - Forecast: 2025-12-26 targeting 2025-12-27.
   - Actual AQI: 407.4; Predicted: 402.3; Absolute Error: **5.09**.
   - Attribution: With largest positive SHAP contributions from elevated $PM_{2.5}$ (+58.7) and $PM_{2.5,\max}$ (+34.2). This good case is the Severe winter day with the smallest error, representing a best case rather than a typical one.
2. **$h=1$ Largest Error Case** (`outputs/shap_waterfall_h1_bad.png`):
   - Forecast: 2024-11-12 targeting 2024-11-13.
   - Actual AQI: **500.0**; Predicted: 391.7; Absolute Error: **108.27**.
   - Attribution: Predicted 391.7, actual 500.0, the AQI cap (explanation not tested). This bad case is the Severe winter day with the largest error, representing a worst case rather than a typical one.
3. **$h=5$ Smallest Error Case** (`outputs/shap_waterfall_h5_good.png`):
   - Forecast: 2025-11-03 targeting 2025-11-08.
   - Actual AQI: 403.4; Predicted: 402.3; Absolute Error: **1.04**.
   - Attribution: Governed by seasonal `day_of_year_cos` (+61.4) and rolling particulate means. Best case.
4. **$h=5$ Largest Error Case** (`outputs/shap_waterfall_h5_bad.png`):
   - Forecast: 2025-10-16 targeting 2025-10-21.
   - Actual AQI: 454.7; Predicted: 275.9; Absolute Error: **178.72**.
   - Attribution: Pre-winter seasonal features failed to anticipate the severe spike 5 days ahead. Worst case.

[TODO: me]


---

## 9. Command-Line Demo

The operational pipeline is accessed via `src/predict.py`. Real run for base date `2025-06-20`:

```text
$ python src/predict.py --date 2025-06-20

===============================================================================================
DELHI AIR QUALITY 5-DAY FORWARD FORECAST | BASE DATE: 2025-06-20
===============================================================================================
Horizon  | Target Date | Pred AQI | Category      | Typical Error | Actual AQI | Advisory
-----------------------------------------------------------------------------------------------
Day +1   | 2025-06-21  | 77.42    | Satisfactory  | ±27.57        | 81.99      | unusually sensitive people should reduce prolonged outdoor exertion
Day +2   | 2025-06-22  | 92.44    | Satisfactory  | ±36.34        | 89.35      | unusually sensitive people should reduce prolonged outdoor exertion
Day +3   | 2025-06-23  | 102.88   | Moderate      | ±38.75        | 103.55     | people with breathing or heart conditions, children and older adults should reduce prolonged outdoor exertion
         * Note: at this horizon the model is statistically indistinguishable from a seasonal average
Day +4   | 2025-06-24  | 112.57   | Moderate      | ±40.26        | 106.65     | people with breathing or heart conditions, children and older adults should reduce prolonged outdoor exertion
         * Note: at this horizon the model is statistically indistinguishable from a seasonal average
Day +5   | 2025-06-25  | 113.50   | Moderate      | ±41.48        | 121.36     | people with breathing or heart conditions, children and older adults should reduce prolonged outdoor exertion
         * Note: at this horizon the model is statistically indistinguishable from a seasonal average
-----------------------------------------------------------------------------------------------
Informational only, not medical advice.
===============================================================================================

Saved forecast table to D:\ML_Project\outputs\forecast_2025-06-20.csv
Saved forecast figure to D:\ML_Project\outputs\forecast_2025-06-20.png
```

Features:
- Clear horizon note: at days 3 to 5 the model is statistically indistinguishable from a seasonal average.
- Mandatory legal disclaimer: `Informational only, not medical advice.`
- Saved forecast artifacts: `outputs/forecast_2025-06-20.csv` and `outputs/forecast_2025-06-20.png`.


---

## 10. Limitations and Future Work

### 10.1 Limitations
1. **CPCB Breakpoint Verification Pending**: The CPCB breakpoint table is pending verification by the project owner against official documentation.
2. **Unverified Festival Calendar**: The Diwali dates are unverified pending confirmation against an official calendar.
3. **Upper-Tail Saturation**: Tree regressors cannot extrapolate beyond observed values; the model may under-predict the most extreme days (explanation not tested).
4. **Absence of Forward NWP Fields**: The model uses observed meteorology up to day $t$; at days 3 to 5 it lacks forward weather fields, explaining convergence toward seasonal climatology.
5. **Unmodeled Coarse Dust Incursions (Unverified Hypothesis)**: The two largest non-winter error episodes had low $PM_{2.5}/PM_{10}$ ratios and sudden particulate jumps. Surface station models without satellite AOD or boundary layer height data cannot anticipate coarse dust incursions.
6. **Station Spatial Sparsity**: Two stations were dropped due to low coverage ($<50\%$), and spatial interpolation was not applied.

### 10.2 Future Work
1. **Numerical Weather Prediction Integration**: Incorporating forecast meteorology (wind vectors, boundary layer height, temperature) from ECMWF/GFS.
2. **Satellite Remote Sensing**: Ingesting fire counts and Aerosol Optical Depth (AOD) to dynamically track upstream agricultural burning.
3. **Asymmetric Loss Functions**: Quantile or extreme-value loss to reduce under-prediction during Severe episodes.

[TODO: me]


---

## 11. Conclusion and References

### 11.1 Methodological Summary & Key Findings
This project evaluated multi-horizon air quality forecasting in Delhi across eight years of CAAQMS telemetry:
1. **Early Horizon Skill**: The model improves on persistence and climatology at 1 to 2 days ahead, reducing 1-day MAE to 27.57 (11.0% skill over persistence) and 2-day MAE to 36.34 (10.1% skill over climatology).
2. **Medium Horizon Convergence**: At 3 to 5 days it is statistically indistinguishable from climatology (intervals include zero) and from Ridge.
3. **Alert Detection Benchmark**: At days 3 to 5 the alert detection is not better than the trivial rule "alert on every winter day"; at days 3 to 5 Random Forest has no demonstrated advantage over the trivial winter rule.
4. **Explainability Horizon Shift**: The SHAP attribution shifts toward calendar features as the horizon grows (Pollutants and AQI history drop from 90.3% combined share at $h=1$ to 61.5% at $h=5$, while Calendar expands from 3.8% to 30.7%).

### 11.2 References
1. Central Pollution Control Board (CPCB). *National Air Quality Index: Technical Report*, 2014.
2. Commission for Air Quality Management in NCR (CAQM). *Statutory Framework for Graded Response Action Plan (GRAP)*, 2021–2024.
3. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 30.
4. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
5. Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *JASA*, 89(428), 1303–1313.

[TODO: me]

---
*Report compiled from verified repository outputs on 2026-11-08.*
