# Chronological Train / Validation / Test Split Summary

## 1. Split Definition & Boundary Integrity

Strict adherence to **Rule 2** (Never shuffle time series; chronological splits with a 5-day buffer gap; test set reserved exclusively for final evaluation).

| Partition | Start Date | End Date | Row Count | Days Dropped in Buffer Gap | Description |
|:---|:---:|:---:|:---:|:---:|:---|
| **Train** | `2018-01-08` | `2023-03-31` | **1887** | — | Training data up to `2023-03-31` (~5.25 years) |
| *Gap 1* | `2023-04-01` | `2023-04-05` | **5** | **5** | 5-day chronological buffer between Train and Val |
| **Validation** | `2023-04-06` | `2024-03-31` | **352** | — | Validation data up to `2024-03-31` (1 full calendar year) |
| *Gap 2* | `2024-04-01` | `2024-04-05` | **5** | **5** | 5-day chronological buffer between Val and Test |
| **Test** | `2024-04-06` | `2026-09-25` | **887** | — | Final evaluation holdout (~2.5 years, through `2026-09-25`) |
| **Total** | `2018-01-08` | `2026-09-25` | **3136** | **10** | All calendar rows accounted for |

---

## 2. AQI Category & High-Pollution Day Distribution

### Class Balance Across Partitions

| AQI Category | Range | Train Count (Share) | Validation Count (Share) | Test Count (Share) |
|:---|:---:|:---:|:---:|:---:|
| **Good** | 0 – 50 | 18 (0.95%) | 0 (0.00%) | 6 (0.68%) |
| **Satisfactory** | 51 – 100 | 355 (18.81%) | 56 (15.91%) | 184 (20.74%) |
| **Moderate** | 101 – 200 | 590 (31.27%) | 143 (40.62%) | 366 (41.26%) |
| **Poor** | 201 – 300 | 317 (16.80%) | 38 (10.80%) | 122 (13.75%) |
| **Very Poor** | 301 – 400 | 475 (25.17%) | 90 (25.57%) | 165 (18.60%) |
| **Severe** | 401 – 500 | 132 (7.00%) | 25 (7.10%) | 44 (4.96%) |
| **Total** | — | **1887** | **352** | **887** |

### High Pollution Episodes

| Metric | Condition | Train | Validation | Test |
|:---|:---:|:---:|:---:|:---:|
| **Poor-or-worse Days** | $\text{AQI} > 200$ | **924** (48.97%) | **153** (43.47%) | **331** (37.32%) |
| **Severe Days** | $\text{AQI} > 400$ | **132** (7.00%) | **25** (7.10%) | **44** (4.96%) |

---

## 3. Winter Season Completeness Audit

Delhi pollution displays strong seasonality, with peak pollution episodes concentrated in winter (October through February).

- **Validation Set Winter Audit**:
  - Required Winter: **October 2023 through February 2024**
  - Present Months: `2023-10, 2023-11, 2023-12, 2024-01, 2024-02`
  - Status: **Confirmed complete** (all 5 winter months present with zero missing months).
- **Test Set Winter Audit**:
  - Required Winter 1: **October 2024 through February 2025**
    - Present Months: `2024-10, 2024-11, 2024-12, 2025-01, 2025-02`
    - Status: **Confirmed complete** (all 5 winter months present).
  - Required Winter 2: **October 2025 through February 2026**
    - Present Months: `2025-10, 2025-11, 2025-12, 2026-01, 2026-02`
    - Status: **Confirmed complete** (all 5 winter months present).
  - Overall Test Set Winter Status: **Confirmed complete** (all 10 winter months across both seasons present with zero missing months).

---

## 4. Expanding-Window Cross-Validation on Training Set

Configured with `TimeSeriesSplit(n_splits=5, gap=5)` on the training set only (strictly preserving temporal ordering and 5-day boundary gaps; no test set data accessed).

| Fold | Training Window | Train Days | Validation Window | Val Days | Validation Poor+ (AQI > 200) | Validation Severe (AQI > 400) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | `2018-01-08` to `2018-11-22` | 312 | `2018-11-28` to `2019-10-11` | 314 | **143** (45.54%) | **28** |
| **Fold 2** | `2018-01-08` to `2019-10-06` | 626 | `2019-10-12` to `2020-08-20` | 314 | **132** (42.04%) | **25** |
| **Fold 3** | `2018-01-08` to `2020-08-15` | 940 | `2020-08-21` to `2021-07-11` | 314 | **166** (52.87%) | **26** |
| **Fold 4** | `2018-01-08` to `2021-07-06` | 1254 | `2021-07-12` to `2022-05-21` | 314 | **172** (54.78%) | **25** |
| **Fold 5** | `2018-01-08` to `2022-05-16` | 1568 | `2022-05-22` to `2023-03-31` | 314 | **151** (48.09%) | **11** |

Every validation fold contains at least 132 Poor-or-worse days (42.0% – 54.8% prevalence), providing robust statistical power for hyperparameter tuning and model validation across multiple seasonal cycles.
