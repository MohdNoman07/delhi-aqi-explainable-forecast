# Delhi Air Quality Forecasting (1–5 Days Ahead)
## Technical Presentation Slide Deck Outline (11 Slides)

---

### Slide 1: Problem Statement
- **Bullet 1**: Delhi experiences severe recurring winter air pollution crises requiring emergency interventions under the Graded Response Action Plan (GRAP Stages I–IV).
- **Bullet 2**: Municipal interventions (traffic bans, mechanized sweeping, school closures) require 48 to 72 hours of lead time to deploy effectively.
- **Bullet 3**: We develop and audit a multi-horizon predictive system forecasting citywide daily AQI across 1 to 5 days ahead ($h \in \{1, 2, 3, 4, 5\}$) using 8+ years of telemetry.
- **Speaker Notes**:
  - [TODO: me]
  - [TODO: me]

---

### Slide 2: Dataset
- **Bullet 1**: Continuous ambient telemetry from 37 CAAQMS stations spanning 2018-01-01 to 2026-09-30 (76,680 hours) from CPCB, DPCC, and IITM.
- **Bullet 2**: Station completeness audit retained 35 stations with $\ge 50\%$ hourly coverage; dropped 2 stations (`Dwarka Sector 8` at 8.34% and `Lodhi Road` at 34.40%).
- **Bullet 3**: Citywide series computed via spatial averaging across $\ge 3$ reporting stations; target is CPCB daily AQI (0–500 scale) mapped into 6 health bands.
- **Speaker Notes**:
  - [TODO: me]
  - [TODO: me]

---

### Slide 3: Pipeline Diagram
- **Bullet 1**: Barometric pressure standardized from mixed mmHg to hPa (167,594 readings converted; 17,702 out-of-range values removed outside [940, 1030] hPa).
- **Bullet 2**: Physical instrument caps applied to ozone, benzene, temperature, wind speed, and solar radiation; 7,382 particulate values filtered for $PM_{2.5} > 1.2 \times PM_{10}$ inversions.
- **Bullet 3**: Automated audit invalidated stuck-sensor episodes frozen for $\ge 24$ hours (70,753 pressure, 27,347 wind speed, 20,006 benzene readings removed).
- **Speaker Notes**:
  - [TODO: me]
  - [TODO: me]

---

### Slide 4: EDA Highlight
- **Bullet 1**: Severe winter seasonality: median AQI exceeds 320 in Nov/Dec, dropping below 100 in Jul/Aug (`outputs/eda/02_aqi_by_month_and_heatmap.png`).
- **Bullet 2**: Particulate dominance: $PM_{2.5}$ and $PM_{10}$ determine the official daily AQI on $>95\%$ of calendar days across all audited years.
- **Bullet 3**: PACF drops sharply after lag 1, while ventilation analysis confirms a 4x surge in Severe days when surface winds drop below 1.5 m/s.
- **Speaker Notes**:
  - [TODO: me]
  - [TODO: me]

---

### Slide 5: Features
- **Bullet 1**: Exactly 38 features engineered across 4 families: Criteria Pollutants (14), AQI History (9), Weather (9), and Calendar/Cyclical (6).
- **Bullet 2**: Zero future leakage: all features use strictly trailing history ($t' \le t$); no centered windows and no negative shifts in feature columns.
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
- **Bullet 1**: Benchmark comparison across 4 models: Persistence, Climatology, Ridge ($lpha=100$), and tuned Random Forest across 887 test days (`outputs/metrics.csv`).
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
- **Bullet 1**: SHAP attribution reveals a systematic shift: at $h=1$, Pollutants (61.6%) + AQI history (28.7%) comprise 90.3% of attribution, with Calendar at 3.8%.
- **Bullet 2**: The SHAP attribution shifts toward calendar features as the horizon grows: at $h=5$, Calendar features expand to 30.7% while Pollutants drop to 35.7%.
- **Bullet 3**: Waterfall case studies show $h=1$ driven by instantaneous $PM_{2.5}$, while $h=5$ is governed by seasonal `day_of_year_cos` (`outputs/shap_family_importance.png`).
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
