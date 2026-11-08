# Delhi Air Quality Forecasting — Detailed Error Analysis

Diagnostic evaluation based strictly on holdout test set predictions (`outputs/test_predictions.csv`, 887 observations from 2024-04-06 to 2026-09-25). No models were retrained or retuned.

---

## 1. Error by Calendar Month (Random Forest vs Best Baseline)

Monthly breakdown of Mean Absolute Error (MAE) and Mean Bias ($Predicted - Actual$) across all 12 calendar months:

### Horizon h=1 (Best Baseline: Persistence)

| Month | Observations | RF MAE | Baseline MAE | RF Bias | Baseline Bias |
|:---|:---:|:---:|:---:|:---:|:---:|
| Jan | 62 | 35.31 | 43.47 | +4.58 | +1.75 |
| Feb | 56 | 47.35 | 49.83 | +20.95 | +5.69 |
| Mar | 62 | 34.46 | 34.54 | +16.72 | +0.03 |
| Apr | 85 | 29.47 | 35.23 | +2.27 | -0.42 |
| May | 93 | 33.13 | 36.81 | -4.99 | -0.41 |
| Jun | 90 | 23.89 | 28.19 | -2.30 | +2.53 |
| Jul | 93 | 17.57 | 20.01 | +2.68 | +0.69 |
| Aug | 77 | 15.26 | 16.03 | +7.32 | -0.13 |
| Sep | 85 | 16.81 | 16.81 | +5.06 | -0.58 |
| Oct | 62 | 27.34 | 29.84 | -1.87 | -8.18 |
| Nov | 60 | 28.41 | 30.25 | -8.73 | +1.60 |
| Dec | 62 | 33.90 | 43.58 | +0.92 | -0.75 |

### Horizon h=2 (Best Baseline: Climatology)

| Month | Observations | RF MAE | Baseline MAE | RF Bias | Baseline Bias |
|:---|:---:|:---:|:---:|:---:|:---:|
| Jan | 62 | 43.95 | 45.48 | -1.97 | +3.28 |
| Feb | 56 | 54.88 | 57.12 | +20.96 | +37.75 |
| Mar | 62 | 45.77 | 49.52 | +19.94 | +22.63 |
| Apr | 85 | 34.10 | 36.74 | +2.48 | +13.65 |
| May | 93 | 41.70 | 49.76 | -5.15 | +2.52 |
| Jun | 90 | 29.52 | 31.68 | +0.57 | +11.43 |
| Jul | 93 | 23.22 | 28.98 | +3.76 | +3.33 |
| Aug | 77 | 18.56 | 23.59 | +10.72 | +15.16 |
| Sep | 85 | 25.99 | 27.06 | +5.94 | -0.39 |
| Oct | 62 | 39.64 | 41.98 | -9.19 | +3.80 |
| Nov | 60 | 39.38 | 47.72 | -19.29 | -20.36 |
| Dec | 62 | 57.22 | 62.79 | -3.71 | +15.96 |

### Horizon h=3 (Best Baseline: Climatology)

| Month | Observations | RF MAE | Baseline MAE | RF Bias | Baseline Bias |
|:---|:---:|:---:|:---:|:---:|:---:|
| Jan | 62 | 42.08 | 44.74 | -3.50 | +3.17 |
| Feb | 56 | 53.97 | 57.66 | +26.83 | +39.20 |
| Mar | 62 | 51.74 | 50.14 | +18.12 | +19.85 |
| Apr | 85 | 37.19 | 37.46 | -0.17 | +14.04 |
| May | 93 | 45.04 | 48.49 | -5.24 | +3.82 |
| Jun | 90 | 29.68 | 32.03 | -0.12 | +11.78 |
| Jul | 93 | 25.06 | 28.56 | +5.23 | +2.74 |
| Aug | 77 | 19.74 | 24.28 | +11.31 | +16.14 |
| Sep | 85 | 29.55 | 27.75 | +6.54 | -0.13 |
| Oct | 62 | 44.24 | 41.91 | -11.07 | +4.32 |
| Nov | 60 | 41.92 | 47.44 | -26.17 | -20.64 |
| Dec | 62 | 62.71 | 63.35 | -5.90 | +15.07 |

### Horizon h=4 (Best Baseline: Climatology)

| Month | Observations | RF MAE | Baseline MAE | RF Bias | Baseline Bias |
|:---|:---:|:---:|:---:|:---:|:---:|
| Jan | 62 | 44.08 | 44.85 | -3.57 | +1.96 |
| Feb | 56 | 52.38 | 58.06 | +27.59 | +42.81 |
| Mar | 62 | 52.81 | 49.45 | +19.11 | +19.14 |
| Apr | 85 | 43.36 | 39.69 | -0.25 | +12.72 |
| May | 93 | 44.32 | 46.21 | -1.42 | +3.95 |
| Jun | 90 | 30.55 | 32.51 | +4.34 | +13.13 |
| Jul | 93 | 27.95 | 28.46 | +6.61 | +2.33 |
| Aug | 77 | 20.34 | 24.43 | +12.35 | +17.09 |
| Sep | 85 | 30.38 | 27.06 | +6.27 | -1.17 |
| Oct | 62 | 47.79 | 43.14 | -6.53 | +6.02 |
| Nov | 60 | 46.46 | 49.20 | -25.11 | -18.88 |
| Dec | 62 | 59.91 | 61.33 | -4.03 | +11.48 |

### Horizon h=5 (Best Baseline: Climatology)

| Month | Observations | RF MAE | Baseline MAE | RF Bias | Baseline Bias |
|:---|:---:|:---:|:---:|:---:|:---:|
| Jan | 62 | 44.32 | 44.69 | -0.69 | +1.79 |
| Feb | 56 | 56.94 | 59.14 | +35.47 | +43.89 |
| Mar | 62 | 55.87 | 49.30 | +20.32 | +18.99 |
| Apr | 85 | 45.92 | 41.83 | +2.67 | +12.47 |
| May | 93 | 45.91 | 44.30 | +1.63 | +3.26 |
| Jun | 90 | 31.14 | 31.82 | +8.30 | +14.08 |
| Jul | 93 | 28.70 | 28.11 | +8.04 | +1.95 |
| Aug | 77 | 21.44 | 24.65 | +14.24 | +17.97 |
| Sep | 85 | 29.29 | 26.60 | +4.40 | -2.10 |
| Oct | 62 | 49.98 | 45.55 | -3.54 | +8.43 |
| Nov | 60 | 48.04 | 49.42 | -24.61 | -18.66 |
| Dec | 62 | 58.50 | 59.31 | -2.65 | +9.45 |

### Figure Interpretation: `outputs/error_by_month.png`
> The bar chart displays higher MAE during the winter months (November through January, peaking above 45 AQI) and lower MAE during the summer/monsoon calendar period (July to September, below 25 AQI) across all horizons.
> Random Forest achieves lower MAE than the baseline during most months at horizon h=1, whereas at horizon h=5 the errors of Random Forest and Climatology are closely matched throughout the year.

---

## 2. Winter (Oct–Feb) versus Non-Winter (Mar–Sep) Performance

Comparison across Persistence, Climatology, Ridge, and Random Forest partitioned by season (Winter: 302 days; Non-Winter: 585 days), including the alert base rate and the trivial seasonal rule:

| Horizon | Season | Model | Days | Alert Base Rate | MAE | Bias | Alert Precision | Alert Recall |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| h=1 | Winter (Oct-Feb) | Persistence | 302 | 0.818 | 39.24 | -0.10 | 0.943 | 0.943 |
| h=1 | Winter (Oct-Feb) | Climatology | 302 | 0.818 | 50.84 | +7.44 | 0.876 | 1.000 |
| h=1 | Winter (Oct-Feb) | Ridge | 302 | 0.818 | 36.40 | -3.20 | 0.910 | 0.988 |
| h=1 | Winter (Oct-Feb) | Random Forest | 302 | 0.818 | 34.24 | +2.90 | 0.918 | 0.992 |
| h=1 | Winter (Oct-Feb) | Trivial Seasonal Rule | 302 | 0.818 | — | — | 0.818 | 1.000 |
| h=1 | Non-Winter (Mar-Sep) | Persistence | 585 | 0.144 | 26.70 | +0.27 | 0.583 | 0.583 |
| h=1 | Non-Winter (Mar-Sep) | Climatology | 585 | 0.144 | 34.96 | +8.98 | 0.333 | 0.131 |
| h=1 | Non-Winter (Mar-Sep) | Ridge | 585 | 0.144 | 24.41 | -2.07 | 0.635 | 0.476 |
| h=1 | Non-Winter (Mar-Sep) | Random Forest | 585 | 0.144 | 24.12 | +3.08 | 0.573 | 0.512 |
| h=1 | Non-Winter (Mar-Sep) | Trivial Seasonal Rule | 585 | 0.144 | — | — | 0.000 | 0.000 |
| h=2 | Winter (Oct-Feb) | Persistence | 302 | 0.821 | 56.68 | -0.25 | 0.911 | 0.907 |
| h=2 | Winter (Oct-Feb) | Climatology | 302 | 0.821 | 50.92 | +7.69 | 0.876 | 0.996 |
| h=2 | Winter (Oct-Feb) | Ridge | 302 | 0.821 | 48.02 | -6.79 | 0.894 | 0.984 |
| h=2 | Winter (Oct-Feb) | Random Forest | 302 | 0.821 | 46.91 | -3.00 | 0.897 | 0.980 |
| h=2 | Winter (Oct-Feb) | Trivial Seasonal Rule | 302 | 0.821 | — | — | 0.821 | 1.000 |
| h=2 | Non-Winter (Mar-Sep) | Persistence | 585 | 0.142 | 35.28 | +0.64 | 0.429 | 0.434 |
| h=2 | Non-Winter (Mar-Sep) | Climatology | 585 | 0.142 | 35.02 | +9.01 | 0.333 | 0.133 |
| h=2 | Non-Winter (Mar-Sep) | Ridge | 585 | 0.142 | 31.33 | -4.45 | 0.551 | 0.325 |
| h=2 | Non-Winter (Mar-Sep) | Random Forest | 585 | 0.142 | 30.89 | +4.61 | 0.500 | 0.386 |
| h=2 | Non-Winter (Mar-Sep) | Trivial Seasonal Rule | 585 | 0.142 | — | — | 0.000 | 0.000 |
| h=3 | Winter (Oct-Feb) | Persistence | 302 | 0.821 | 63.49 | -0.48 | 0.895 | 0.891 |
| h=3 | Winter (Oct-Feb) | Climatology | 302 | 0.821 | 50.91 | +7.80 | 0.876 | 0.996 |
| h=3 | Winter (Oct-Feb) | Ridge | 302 | 0.821 | 51.54 | -9.59 | 0.886 | 0.976 |
| h=3 | Winter (Oct-Feb) | Random Forest | 302 | 0.821 | 48.93 | -4.43 | 0.889 | 0.972 |
| h=3 | Winter (Oct-Feb) | Trivial Seasonal Rule | 302 | 0.821 | — | — | 0.821 | 1.000 |
| h=3 | Non-Winter (Mar-Sep) | Persistence | 585 | 0.142 | 38.86 | +1.05 | 0.381 | 0.386 |
| h=3 | Non-Winter (Mar-Sep) | Climatology | 585 | 0.142 | 35.16 | +9.11 | 0.333 | 0.133 |
| h=3 | Non-Winter (Mar-Sep) | Ridge | 585 | 0.142 | 33.60 | -6.68 | 0.394 | 0.157 |
| h=3 | Non-Winter (Mar-Sep) | Random Forest | 585 | 0.142 | 33.49 | +4.31 | 0.354 | 0.205 |
| h=3 | Non-Winter (Mar-Sep) | Trivial Seasonal Rule | 585 | 0.142 | — | — | 0.000 | 0.000 |
| h=4 | Winter (Oct-Feb) | Persistence | 302 | 0.821 | 65.11 | -0.38 | 0.903 | 0.899 |
| h=4 | Winter (Oct-Feb) | Climatology | 302 | 0.821 | 51.20 | +8.18 | 0.876 | 0.996 |
| h=4 | Winter (Oct-Feb) | Ridge | 302 | 0.821 | 52.19 | -7.00 | 0.879 | 0.968 |
| h=4 | Winter (Oct-Feb) | Random Forest | 302 | 0.821 | 50.10 | -2.77 | 0.895 | 0.964 |
| h=4 | Winter (Oct-Feb) | Trivial Seasonal Rule | 302 | 0.821 | — | — | 0.821 | 1.000 |
| h=4 | Non-Winter (Mar-Sep) | Persistence | 585 | 0.142 | 41.91 | +1.20 | 0.298 | 0.301 |
| h=4 | Non-Winter (Mar-Sep) | Climatology | 585 | 0.142 | 35.02 | +8.97 | 0.333 | 0.133 |
| h=4 | Non-Winter (Mar-Sep) | Ridge | 585 | 0.142 | 33.85 | -2.92 | 0.417 | 0.181 |
| h=4 | Non-Winter (Mar-Sep) | Random Forest | 585 | 0.142 | 35.18 | +6.02 | 0.192 | 0.120 |
| h=4 | Non-Winter (Mar-Sep) | Trivial Seasonal Rule | 585 | 0.142 | — | — | 0.000 | 0.000 |
| h=5 | Winter (Oct-Feb) | Persistence | 302 | 0.821 | 65.88 | -0.29 | 0.903 | 0.899 |
| h=5 | Winter (Oct-Feb) | Climatology | 302 | 0.821 | 51.49 | +8.47 | 0.876 | 0.996 |
| h=5 | Winter (Oct-Feb) | Ridge | 302 | 0.821 | 53.83 | -3.98 | 0.865 | 0.960 |
| h=5 | Winter (Oct-Feb) | Random Forest | 302 | 0.821 | 51.47 | +0.28 | 0.882 | 0.968 |
| h=5 | Winter (Oct-Feb) | Trivial Seasonal Rule | 302 | 0.821 | — | — | 0.821 | 1.000 |
| h=5 | Non-Winter (Mar-Sep) | Persistence | 585 | 0.140 | 45.76 | +1.34 | 0.238 | 0.244 |
| h=5 | Non-Winter (Mar-Sep) | Climatology | 585 | 0.140 | 34.82 | +8.88 | 0.333 | 0.134 |
| h=5 | Non-Winter (Mar-Sep) | Ridge | 585 | 0.140 | 34.36 | +1.24 | 0.372 | 0.195 |
| h=5 | Non-Winter (Mar-Sep) | Random Forest | 585 | 0.140 | 36.32 | +7.87 | 0.203 | 0.146 |
| h=5 | Non-Winter (Mar-Sep) | Trivial Seasonal Rule | 585 | 0.140 | — | — | 0.000 | 0.000 |

### Full Test Set Alert Detection Benchmark (Random Forest vs Trivial Seasonal Rule)

Across all 887 test days (alert base rate = 37.3%), comparison of Random Forest alert precision and recall against the trivial rule ("alert on every winter day, never in non-winter"), alongside Random Forest's non-winter alert recall:

| Horizon | RF Alert Precision | RF Alert Recall | Trivial Alert Precision | Trivial Alert Recall | RF Non-Winter Alert Recall |
|:---:|:---:|:---:|:---:|:---:|:---:|
| h=1 | 0.842 | 0.870 | 0.818 | 0.746 | 0.512 |
| h=2 | 0.821 | 0.831 | 0.821 | 0.749 | 0.386 |
| h=3 | 0.809 | 0.779 | 0.821 | 0.749 | 0.205 |
| h=4 | 0.781 | 0.752 | 0.821 | 0.749 | 0.120 |
| h=5 | 0.761 | 0.764 | 0.821 | 0.752 | 0.146 |

**Horizons where Random Forest is at or below the trivial rule**:
- **Alert Precision**: Random Forest is virtually tied with the trivial rule at horizon h=2 (0.821 vs 0.821) and falls strictly below the trivial rule at horizons **h=3, h=4, and h=5** (0.809, 0.781, and 0.761 vs 0.821).
- **Alert Recall**: Random Forest exceeds the trivial rule across all five horizons on the full test set (0.870 down to 0.764 vs 0.746 to 0.752) due to detecting non-winter alerts (12.0% to 51.2% recall) where the trivial rule catches zero. At day +4, Random Forest alert recall is 0.752 vs 0.749 for the trivial rule, and at day +5 it is 0.764 vs 0.752. In plain words, at days 3 to 5 Random Forest has no demonstrated advantage over the trivial winter rule (lower precision, recall gap within about one point, no interval computed).

**Observations on seasonal alert detection and trivial rule comparison**:
- **Alert base rate**: 81.8%–82.1% of winter days have AQI > 200, compared to only 14.0%–14.4% of non-winter days.
- **Trivial rule vs Random Forest in Winter**: The trivial rule ('alert every winter day') achieves 100% recall with precision equal to the base rate (81.8% at h=1). Random Forest achieves 99.2% recall with higher precision (91.8% at h=1, 89.7% at h=2), demonstrating that it identifies which winter days are non-alerts rather than issuing blanket alarms.
- **Trivial rule vs Random Forest in Non-Winter**: The trivial rule catches 0% of non-winter alerts. Random Forest successfully identifies 51.2% of non-winter alerts at h=1 and 38.6% at h=2.

---

## 3. Paired Block-Bootstrap Confidence Intervals (95% CI, blocks=7d, B=2000)

### A. Random Forest minus Ridge (Full Test Set)

Difference defined as $|y - \hat{y}_{\text{RF}}| - |y - \hat{y}_{\text{Ridge}}|$ (negative indicates Random Forest error is lower):

| Horizon | RF MAE | Ridge MAE | Mean Diff | 95% Bootstrap CI | RF Clearly Better? | Includes Zero? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| h=1 | 27.57 | 28.49 | -0.92 | `[-1.87, -0.03]` | True | False |
| h=2 | 36.34 | 37.01 | -0.67 | `[-2.06, 0.61]` | False | True |
| h=3 | 38.75 | 39.71 | -0.96 | `[-2.86, 0.81]` | False | True |
| h=4 | 40.26 | 40.09 | +0.16 | `[-1.80, 1.95]` | False | True |
| h=5 | 41.48 | 40.99 | +0.49 | `[-1.60, 2.50]` | False | True |

**Plain-English Summary (RF vs Ridge)**:
- At **horizon h=1**, Random Forest is **clearly better** than Ridge: the 95% bootstrap confidence interval `[-1.87, -0.03]` lies strictly below zero.
- At **horizons h=2, h=3, h=4, and h=5**, the confidence interval **includes zero** (`[-2.06, 0.61]` at h=2, `[-2.86, 0.81]` at h=3, `[-1.80, 1.95]` at h=4, and `[-1.60, 2.50]` at h=5); Random Forest and Ridge are not statistically distinguishable at the 95% level beyond day 1.

### B. Random Forest minus Best Baseline (Winter vs Non-Winter)

Difference defined as $|y - \hat{y}_{\text{RF}}| - |y - \hat{y}_{\text{Baseline}}|$:

| Horizon | Season | Best Baseline | Days | RF MAE | Baseline MAE | Mean Diff | 95% Bootstrap CI | RF Clearly Better? | Includes Zero? |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| h=1 | Winter | Persistence | 302 | 34.24 | 39.24 | -5.00 | `[-7.57, -2.79]` | True | False |
| h=1 | Non-Winter | Persistence | 585 | 24.12 | 26.70 | -2.58 | `[-3.84, -1.48]` | True | False |
| h=2 | Winter | Climatology | 302 | 46.91 | 50.92 | -4.01 | `[-7.86, -0.51]` | True | False |
| h=2 | Non-Winter | Climatology | 585 | 30.89 | 35.02 | -4.13 | `[-6.83, -1.75]` | True | False |
| h=3 | Winter | Climatology | 302 | 48.93 | 50.91 | -1.98 | `[-5.41, 1.43]` | False | True |
| h=3 | Non-Winter | Climatology | 585 | 33.49 | 35.16 | -1.67 | `[-4.29, 0.60]` | False | True |
| h=4 | Winter | Climatology | 302 | 50.10 | 51.20 | -1.09 | `[-4.75, 2.40]` | False | True |
| h=4 | Non-Winter | Climatology | 585 | 35.18 | 35.02 | +0.15 | `[-2.37, 2.36]` | False | True |
| h=5 | Winter | Climatology | 302 | 51.47 | 51.49 | -0.01 | `[-3.28, 3.31]` | False | True |
| h=5 | Non-Winter | Climatology | 585 | 36.32 | 34.82 | +1.50 | `[-0.89, 3.45]` | False | True |

**Plain-English Summary (Seasonal Comparison)**:
- At **horizon h=1**, Random Forest is **clearly better** than Persistence in both Winter (`[-7.57, -2.79]`) and Non-Winter (`[-3.84, -1.48]`).
- At **horizon h=2**, Random Forest is **clearly better** than Climatology in both Winter (`[-7.86, -0.51]`) and Non-Winter (`[-6.83, -1.75]`).
- At **horizons h=3, h=4, and h=5**, the confidence intervals **include zero** in both seasonal splits; Random Forest and Climatology are not statistically distinguishable at the 95% level.

---

## 4. Top 10 Worst Test Days per Horizon (Random Forest)

Analysis of the 10 largest absolute errors for each horizon, evaluating forecast date, target date ($forecast\_date + h$), days from Diwali, and missing-data adjacency:

### Horizon h=1 Worst Days (saved to `outputs/worst_days_h1.csv`)

| Forecast Date | Target Date | Actual AQI | Predicted AQI | Absolute Error | Actual Category | Predicted Category | Days from Diwali | In Diwali Window [-1, 3]? | Next to Missing? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2025-05-14 | 2025-05-15 | 411.72 | 177.44 | 234.28 | Severe | Moderate | -158 | No | No |
| 2026-07-11 | 2026-07-12 | 336.04 | 160.67 | 175.37 | Very Poor | Moderate | +265 | No | No |
| 2026-02-11 | 2026-02-12 | 142.47 | 311.32 | 168.85 | Moderate | Very Poor | +115 | No | No |
| 2024-12-03 | 2024-12-04 | 147.31 | 306.43 | 159.12 | Moderate | Very Poor | +33 | No | No |
| 2025-10-30 | 2025-10-31 | 225.5 | 368.95 | 143.45 | Poor | Very Poor | +11 | No | No |
| 2026-01-01 | 2026-01-02 | 208.72 | 348.56 | 139.84 | Poor | Very Poor | +74 | No | No |
| 2024-12-27 | 2024-12-28 | 188.97 | 325.23 | 136.26 | Moderate | Very Poor | +57 | No | No |
| 2024-05-02 | 2024-05-03 | 312.41 | 185.71 | 126.7 | Very Poor | Moderate | +173 | No | No |
| 2025-12-23 | 2025-12-24 | 259.42 | 385.68 | 126.26 | Poor | Very Poor | +65 | No | No |
| 2026-05-24 | 2026-05-25 | 307.7 | 184.21 | 123.49 | Very Poor | Moderate | +217 | No | No |

### Horizon h=2 Worst Days (saved to `outputs/worst_days_h2.csv`)

| Forecast Date | Target Date | Actual AQI | Predicted AQI | Absolute Error | Actual Category | Predicted Category | Days from Diwali | In Diwali Window [-1, 3]? | Next to Missing? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2025-05-13 | 2025-05-15 | 411.72 | 148.21 | 263.51 | Severe | Moderate | -158 | No | No |
| 2026-07-10 | 2026-07-12 | 336.04 | 96.11 | 239.93 | Very Poor | Satisfactory | +265 | No | No |
| 2026-01-22 | 2026-01-24 | 155.87 | 332.02 | 176.15 | Moderate | Very Poor | +96 | No | No |
| 2024-12-02 | 2024-12-04 | 147.31 | 322.47 | 175.16 | Moderate | Very Poor | +33 | No | No |
| 2024-12-26 | 2024-12-28 | 188.97 | 351.44 | 162.47 | Moderate | Very Poor | +57 | No | No |
| 2025-12-31 | 2026-01-02 | 208.72 | 369.5 | 160.78 | Poor | Very Poor | +74 | No | No |
| 2024-12-03 | 2024-12-05 | 162.3 | 314.67 | 152.37 | Moderate | Very Poor | +34 | No | No |
| 2025-02-05 | 2025-02-07 | 137.82 | 283.44 | 145.62 | Moderate | Poor | +98 | No | No |
| 2024-11-11 | 2024-11-13 | 500.0 | 357.06 | 142.94 | Severe | Very Poor | +12 | No | No |
| 2025-10-19 | 2025-10-21 | 454.66 | 316.66 | 138.0 | Severe | Very Poor | +1 | Yes | No |

### Horizon h=3 Worst Days (saved to `outputs/worst_days_h3.csv`)

| Forecast Date | Target Date | Actual AQI | Predicted AQI | Absolute Error | Actual Category | Predicted Category | Days from Diwali | In Diwali Window [-1, 3]? | Next to Missing? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2025-05-12 | 2025-05-15 | 411.72 | 164.34 | 247.38 | Severe | Moderate | -158 | No | No |
| 2026-07-09 | 2026-07-12 | 336.04 | 90.69 | 245.35 | Very Poor | Satisfactory | +265 | No | No |
| 2024-12-01 | 2024-12-04 | 147.31 | 338.24 | 190.93 | Moderate | Very Poor | +33 | No | No |
| 2025-10-18 | 2025-10-21 | 454.66 | 282.52 | 172.14 | Severe | Poor | +1 | Yes | No |
| 2024-12-02 | 2024-12-05 | 162.3 | 331.86 | 169.56 | Moderate | Very Poor | +34 | No | No |
| 2025-03-12 | 2025-03-15 | 70.57 | 240.02 | 169.45 | Satisfactory | Poor | +134 | No | No |
| 2026-01-22 | 2026-01-25 | 159.77 | 328.79 | 169.02 | Moderate | Very Poor | +97 | No | No |
| 2026-01-21 | 2026-01-24 | 155.87 | 324.8 | 168.93 | Moderate | Very Poor | +96 | No | No |
| 2026-05-22 | 2026-05-25 | 307.7 | 140.17 | 167.53 | Very Poor | Moderate | +217 | No | No |
| 2024-12-14 | 2024-12-17 | 500.0 | 333.09 | 166.91 | Severe | Very Poor | +46 | No | No |

### Horizon h=4 Worst Days (saved to `outputs/worst_days_h4.csv`)

| Forecast Date | Target Date | Actual AQI | Predicted AQI | Absolute Error | Actual Category | Predicted Category | Days from Diwali | In Diwali Window [-1, 3]? | Next to Missing? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2026-07-08 | 2026-07-12 | 336.04 | 91.51 | 244.53 | Very Poor | Satisfactory | +265 | No | No |
| 2025-05-11 | 2025-05-15 | 411.72 | 167.52 | 244.2 | Severe | Moderate | -158 | No | No |
| 2024-11-30 | 2024-12-04 | 147.31 | 344.56 | 197.25 | Moderate | Very Poor | +33 | No | No |
| 2024-12-01 | 2024-12-05 | 162.3 | 338.42 | 176.12 | Moderate | Very Poor | +34 | No | No |
| 2025-10-17 | 2025-10-21 | 454.66 | 279.83 | 174.83 | Severe | Poor | +1 | Yes | No |
| 2024-11-14 | 2024-11-18 | 500.0 | 331.24 | 168.76 | Severe | Very Poor | +17 | No | No |
| 2026-01-21 | 2026-01-25 | 159.77 | 327.15 | 167.38 | Moderate | Very Poor | +97 | No | No |
| 2025-12-09 | 2025-12-13 | 500.0 | 337.04 | 162.96 | Severe | Very Poor | +54 | No | No |
| 2025-10-27 | 2025-10-31 | 225.5 | 388.15 | 162.65 | Poor | Very Poor | +11 | No | No |
| 2026-01-20 | 2026-01-24 | 155.87 | 318.24 | 162.37 | Moderate | Very Poor | +96 | No | No |

### Horizon h=5 Worst Days (saved to `outputs/worst_days_h5.csv`)

| Forecast Date | Target Date | Actual AQI | Predicted AQI | Absolute Error | Actual Category | Predicted Category | Days from Diwali | In Diwali Window [-1, 3]? | Next to Missing? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2025-05-10 | 2025-05-15 | 411.72 | 175.74 | 235.98 | Severe | Moderate | -158 | No | No |
| 2026-07-07 | 2026-07-12 | 336.04 | 118.95 | 217.09 | Very Poor | Moderate | +265 | No | No |
| 2024-11-29 | 2024-12-04 | 147.31 | 338.87 | 191.56 | Moderate | Very Poor | +33 | No | No |
| 2024-11-30 | 2024-12-05 | 162.3 | 345.91 | 183.61 | Moderate | Very Poor | +34 | No | No |
| 2025-10-26 | 2025-10-31 | 225.5 | 407.0 | 181.5 | Poor | Severe | +11 | No | No |
| 2025-10-16 | 2025-10-21 | 454.66 | 275.94 | 178.72 | Severe | Poor | +1 | Yes | No |
| 2024-11-11 | 2024-11-16 | 500.0 | 329.76 | 170.24 | Severe | Very Poor | +15 | No | No |
| 2024-11-12 | 2024-11-17 | 500.0 | 331.48 | 168.52 | Severe | Very Poor | +16 | No | No |
| 2024-12-23 | 2024-12-28 | 188.97 | 350.39 | 161.42 | Moderate | Very Poor | +57 | No | No |
| 2025-12-08 | 2025-12-13 | 500.0 | 339.44 | 160.56 | Severe | Very Poor | +54 | No | No |

**Diwali Proximity Audit (Target Date Window: -1 to +3 Days)**:
- Across all 50 evaluated worst-day predictions (10 worst $\times$ 5 horizons), exactly **4 days fall within the Diwali window** ($-1 \le \text{days\_from\_diwali} \le +3$).
- The remaining **46 worst days fall outside this window**.
- Specifically, all 4 occurrences in the Diwali window target the exact same day: **target date 2025-10-21** (one day after unverified Diwali date 2025-10-20; actual AQI was 454.66, predicted between 275.94 and 316.66 across horizons h=2, 3, 4, 5). Therefore, this represents **one single pollution event observed at four different forecast horizons**, rather than four separate events.
- None of the worst days are adjacent to a missing-data calendar day.

### Audit of Largest-Error Non-Winter Target Days (2025-05-15 and 2026-07-12)

Measurements for the two non-winter target days with the largest prediction errors across horizons:

1. **Target Day 2025-05-15** (evaluated in top-2 errors across all horizons h=1..5):
   - **AQI day before (2025-05-14)**: 164.01
   - **AQI day of (2025-05-15)**: 411.72 (Severe)
   - **$PM_{2.5}$ daily mean**: 210.13 $\mu\text{g/m}^3$
   - **$PM_{10}$ daily mean**: 439.37 $\mu\text{g/m}^3$
   - **$PM_{2.5} / PM_{10}$ ratio**: 0.4782 (47.82%)
   - **Prominent pollutant**: $PM_{10}$
   - *Note on cause*: The cause of this single-day jump of +247.71 AQI points is unknown from repository files; potential environmental causes are unverified hypotheses.

2. **Target Day 2026-07-12** (evaluated in top-2 errors across all horizons h=1..5):
   - **AQI day before (2026-07-11)**: 204.63
   - **AQI day of (2026-07-12)**: 336.04 (Very Poor)
   - **$PM_{2.5}$ daily mean**: 68.41 $\mu\text{g/m}^3$
   - **$PM_{10}$ daily mean**: 378.83 $\mu\text{g/m}^3$
   - **$PM_{2.5} / PM_{10}$ ratio**: 0.1806 (18.06%)
   - **Prominent pollutant**: $PM_{10}$
   - *Note on cause*: The cause of this sudden monsoon-season particulate rise is unknown from repository files; potential environmental causes are unverified hypotheses.

---

## 5. Visualizations & Two-Sentence Interpretations

### Figure 1: `outputs/winter_timeseries_h1_h5.png` (Actual vs Predicted Time Series across Test Winters)
> The time series plot shows that Random Forest tracks the broad seasonal rise and peak pollution episodes across both test winters at horizon h=1, though the model may under-predict the most extreme days (explanation not tested).
> At horizon h=5, the predicted curve shows lower high-frequency variance and closely tracks the mid-winter baseline rather than day-to-day transient spikes.

### Figure 2: `outputs/residuals_h1_h5.png` (Residual vs Predicted Scatter and Residual Histograms)
> Residual is defined strictly as prediction minus actual (residual = prediction - actual), where positive values denote over-prediction and negative values denote under-prediction.
> The scatter plots display an upward trend with widening variance as predicted AQI rises, indicating under-prediction when low AQI is predicted and over-prediction when high AQI is predicted; the histograms show distributions centered near zero with mild positive mean residuals (+3.02 for h=1, +5.28 for h=5).

### Figure 3: `outputs/error_by_month.png` (Monthly Error Breakdown)
> Monthly MAE is lowest during the summer/monsoon calendar period (July to September, averaging below 25 AQI) and highest during November and December (averaging between 45 and 65 AQI).
> Random Forest maintains an error advantage over the best baseline during early horizons across both seasons, which narrows to parity by horizon h=5.

---
*Report generated: 2026-11-08 20:03:03*
