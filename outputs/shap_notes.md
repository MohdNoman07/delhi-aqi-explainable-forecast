# SHAP Feature Attribution Notes — Delhi Air Quality Forecasting

SHAP (SHapley Additive exPlanations) values computed on test set features (`models/rf_h1.pkl`, `rf_h3.pkl`, `rf_h5.pkl`, raw targets) using `shap.TreeExplainer`.

> [!NOTE]
> **Credit Splitting Across Correlated Features**: SHAP distributes attribution across mutually correlated features (e.g. `pm25`, `pm25_rolling_mean_3d`, `aqi`, `aqi_rolling_mean_3d`, `pm25_max`). Individual feature rankings reflect this partition, whereas the grouped family importance aggregates these collinear contributions into cohesive thematic blocks.

---

## 1. Feature Family Importance Across Horizons

| Horizon | Family | Total Mean |SHAP| (AQI Points) | Share of Total (%) |
|:---:|:---|:---:|:---:|
| h=1 | AQI history | 27.96 | 28.7% |
| h=1 | Pollutants | 59.92 | 61.6% |
| h=1 | Weather | 5.66 | 5.8% |
| h=1 | Calendar | 3.73 | 3.8% |
| h=3 | AQI history | 28.79 | 30.5% |
| h=3 | Pollutants | 38.63 | 40.9% |
| h=3 | Weather | 8.26 | 8.8% |
| h=3 | Calendar | 18.65 | 19.8% |
| h=5 | AQI history | 23.99 | 25.8% |
| h=5 | Pollutants | 33.13 | 35.7% |
| h=5 | Weather | 7.25 | 7.8% |
| h=5 | Calendar | 28.54 | 30.7% |

### Figure Interpretation: `outputs/shap_family_importance.png` (Stacked Bar Chart)
> The stacked bar chart shows that Pollutants and AQI history account for the largest share of attribution at horizon h=1 (~90.3% combined), but their combined share declines to ~61.5% by horizon h=5.
> Conversely, Calendar features (representing seasonal day-of-year cycles) increase their share from ~3.8% at h=1 to ~30.7% at h=5; the model's attribution shifts from daily readings toward calendar features as the horizon grows.

---

## 2. Beeswarm Summary & Bar Importance Plots

### Horizon h=1: `outputs/shap_summary_h1.png` and `outputs/shap_bar_h1.png`
> At horizon h=1, `pm25`, `aqi`, and `pm25_max` display the highest mean absolute SHAP values, with elevated concentrations (red points) driving sharp positive revisions above the baseline expectation.
> Favorable dispersion conditions, marked by elevated wind speed and higher negative `day_of_year_cos` values, consistently exert negative SHAP contributions that lower the forecast.

### Horizon h=3: `outputs/shap_summary_h3.png` and `outputs/shap_bar_h3.png`
> At horizon h=3, rolling multi-day aggregates (`pm25_rolling_mean_3d` and `aqi_rolling_mean_7d`) gain relative prominence over instantaneous daily readings.
> Calendar cyclical components (`day_of_year_cos` and `day_of_year_sin`) become dominant top-tier predictors, rivaling particulate concentrations in attribution magnitude.

### Horizon h=5: `outputs/shap_summary_h5.png` and `outputs/shap_bar_h5.png`
> At horizon h=5, the seasonal calendar variable `day_of_year_cos` emerges as the leading feature by mean absolute SHAP value, followed by multi-day rolling means `pm25_rolling_mean_3d` and `aqi_rolling_mean_7d`.
> Instantaneous daily features like today's `pm25` and `aqi` drop substantially in attribution rank, showing that 5-day forecasts are primarily governed by seasonal position rather than transient daily spikes.

---

## 3. Case Studies: Severe Winter Days Waterfall Plots

Evaluation of individual predictions on Severe winter test days (truth > 400 AQI). Note that the good and bad cases are the Severe winter days with the smallest and largest error, so they are best and worst cases, not typical ones:

| Horizon | Case | Forecast Date | Target Date | Actual AQI | Predicted AQI | Absolute Error | Plot File |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| h=1 | Good | 2025-12-26 | 2025-12-27 | 407.4 | 402.3 | 5.09 | `shap_waterfall_h1_good.png` |
| h=1 | Bad | 2024-11-12 | 2024-11-13 | 500.0 | 391.7 | 108.27 | `shap_waterfall_h1_bad.png` |
| h=5 | Good | 2025-11-03 | 2025-11-08 | 403.4 | 402.3 | 1.04 | `shap_waterfall_h5_good.png` |
| h=5 | Bad | 2025-10-16 | 2025-10-21 | 454.7 | 275.9 | 178.72 | `shap_waterfall_h5_bad.png` |

### Figure Interpretation: `outputs/shap_waterfall_h1_good.png` (h=1 Smallest Error: 2025-12-26)
> For the forecast on 2025-12-26 targeting 2025-12-27 (actual 407.4, predicted 402.3, error 5.1), elevated `pm25` (215.1) and `pm25_max` (273.7) provide strong positive attributions of +58.7 and +34.2 AQI points above the baseline. This good case is the Severe winter day with the smallest error, representing a best case rather than a typical one.

### Figure Interpretation: `outputs/shap_waterfall_h1_bad.png` (h=1 Largest Error: 2024-11-12)
> For the forecast on 2024-11-12 targeting 2024-11-13: predicted 391.7, actual 500.0, the AQI cap (explanation not tested). This bad case is the Severe winter day with the largest error, representing a worst case rather than a typical one.

### Figure Interpretation: `outputs/shap_waterfall_h5_good.png` (h=5 Smallest Error: 2025-11-03)
> For the 5-day forecast on 2025-11-03 targeting 2025-11-08 (actual 403.4, predicted 402.3, error 1.0), the seasonal feature `day_of_year_cos` (+61.4 points) and particulate rolling means push the prediction into the Severe band. This good case is the Severe winter day with the smallest error, representing a best case rather than a typical one.

### Figure Interpretation: `outputs/shap_waterfall_h5_bad.png` (h=5 Largest Error: 2025-10-16)
> For the 5-day forecast on 2025-10-16 targeting 2025-10-21: forecast 2025-10-16, target 2025-10-21, actual 454.7, predicted 275.9, error 178.7. This bad case is the Severe winter day with the largest error, representing a worst case rather than a typical one.

---
*Report generated: 2026-11-08 23:51:49*
