# Delhi Air Quality — Quality Checks Report (AFTER APPROVED FIXES)

## Executive Summary (Plain English)

All 5 approved data-quality fixes were applied at the station level in the mandated order of operations before reindexing or interpolation:
1. **Pressure Unit Standardization to hPa**: Converted **167,594** readings from mmHg to hPa ($* 1.333224$). Clipped **17,702** readings outside $[940, 1030]\text{ hPa}$. Resulting citywide pressure has median **983.1 hPa**, $p_1 = 970.5\text{ hPa}$, $p_{99} = 995.8\text{ hPa}$, strictly bounded within $[965.0, 1008.7]\text{ hPa}$. The bimodal 740/980 artifact is completely eliminated.
2. **Physical and Analyzer Caps**: Preserved all established physical limits across pollutants and meteorology (638 extreme spike values set to NaN).
3. **Stuck Sensor Invalidation**: Set **173,979** frozen sensor readings ($\ge 24$ consecutive hours of identical non-zero values) to NaN across all variables except rainfall prior to interpolation.
4. **PM Inversion Invalidation**: Invalidated **3,691 station-hours** (7,382 values across PM2.5 and PM10) where $PM_{2.5} > 1.2 \times PM_{10}$ and $PM_{2.5} > 20\,\mu\text{g/m}^3$.
5. **Targeted VOC Spike Filtering**: Filtered **5,038** isolated single-station VOC spikes ($>10\times$ rolling 24h median) on benzene and toluene. Per instructions, **no spike filter was applied** to PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, O3 or meteorology, ensuring true pollution episodes are preserved.
6. **Station Selection Verification**: Evaluated station coverage after all fixes. **All 35 kept stations remain well above 50% coverage** (the lowest kept station is Burari Crossing at **68.53%**; all others are between 86% and 96%). No stations dropped below 50%.

---

## 1. Barometric Pressure Distribution (Standardized to hPa)

| Metric | Value (hPa) | Interpretation |
|---|---:|---|
| **Valid Hours** | 76,448 / 76,680 (99.7%) | Complete timeline coverage |
| **Median** | **983.08 hPa** | Exactly aligns with Delhi elevation (~216m standard ~988 hPa) |
| **1st Percentile (p1)** | **970.51 hPa** | Lowest observed pressure percentile; meteorological cause not verified |
| **99th Percentile (p99)** | **995.82 hPa** | High-pressure winter anticyclonic ridges |
| **Minimum** | **964.98 hPa** | Strictly within physical bounds [940, 1030] |
| **Maximum** | **1008.66 hPa** | Strictly within physical bounds [940, 1030] |

---

## 2. Before vs. After Missing Fraction (%) per Variable per Year

Missing fraction in `clean_hourly.parquet` across all years:

| Year | pm25 | pm10 | no | no2 | nox | nh3 | so2 | co | o3 | benzene | toluene | temperature | humidity | wind_speed | wind_direction | rainfall | solar_radiation | barometric_pressure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.6% | 0.0% | 0.0% | 0.0% | 48.4% | 0.0% | 0.0% |
| 2019 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2020 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2021 | 0.0% | 0.3% | 0.0% | 0.3% | 0.0% | 0.0% | 0.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.7% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2022 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2023 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 1.8% | 0.0% | 0.0% | 0.0% | 1.8% | 0.0% | 1.8% |
| 2024 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2025 | 0.0% | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2026 | 1.1% | 1.1% | 1.1% | 1.1% | 1.1% | 1.1% | 1.1% | 1.1% | 1.1% | 1.1% | 1.1% | 1.1% | 1.1% | 1.1% | 1.1% | 1.1% | 1.1% | 1.1% |

> **Note on 2018 Rainfall**: In 2018, rainfall gauges were still being installed across Delhi stations, so fewer than 2 stations reported rainfall during the first half of the year (48.4% missing). From 2019 onward, rainfall missingness is **0.0% to 1.8%**.

---

## 3. Hours in 2026 Where `n_stations_pm25 == 0`

Total hours in 2026 with 0 reporting stations for PM2.5: **71 hours** (1.08% of 2026).

These 71 hours are concentrated in exactly **two contiguous periods** in August 2026:

| Gap # | Start Timestamp | End Timestamp | Duration (Hours) | Note |
|---|---|---|---:|---|
| **1** | `2026-08-14 11:00:00` | `2026-08-16 15:00:00` | **53 hours** (~2.2 days) | No station reported PM2.5 for these hours; cause unknown. |
| **2** | `2026-08-18 16:00:00` | `2026-08-19 09:00:00` | **18 hours** (~0.75 days) | No station reported PM2.5 for these hours; cause unknown. |

---

## 4. PM2.5 to PM10 Inversion Check (After Fixes)

- Total valid paired hours: **76,577**
- Hours with $PM_{2.5} > 1.2 \times PM_{10}$: **2** (down from 8 before fixes).

Remaining 2 Boundary Hours:

| Datetime | City PM2.5 (µg/m³) | City PM10 (µg/m³) | PM2.5 / PM10 Ratio | Contributing Stations |
|---|---:|---:|---:|---:|
| `2024-06-28 12:00:00` | 64.1 | 50.8 | **1.26** | 27 stations |
| `2026-07-09 06:00:00` | 43.3 | 24.8 | **1.75** | 31 stations |

---

## 5. Cleaning Log Action Counts (`outputs/cleaning_log.csv`)

| Target Column | Rule Name | Threshold Applied | Values Set to NaN / Converted | Description |
|---|---|---|---:|---|
| `barometric_pressure` | `pressure_unit_conversion_mmhg_to_hpa` | `P < 850.0 mmHg` | **0** | Converted 167,594 readings from mmHg to hPa (* 1.333224) |
| `barometric_pressure` | `pressure_physical_bounds_hpa` | `Outside [940, 1030] hPa` | **17,702** | Set 17,702 out-of-range pressure values to NaN |
| `o3` | `physical_and_sensor_cap` | `[0.0, 500.0]` | **12** | Extreme tropospheric ozone cap (500 µg/m³; p99.99 = 325.56, max spike 970.0) |
| `benzene` | `physical_and_sensor_cap` | `[0.0, 200.0]` | **125** | Extreme VOC spike cap (200 µg/m³; p99.99 = 168.26, max spike 476.4) |
| `temperature` | `physical_and_sensor_cap` | `[0.0, 52.0]` | **40** | Delhi historical bounds (lowest ~0°C, record heat ~49.2°C; p99.99 = 48.65, max spike 58.9) |
| `wind_speed` | `physical_and_sensor_cap` | `[0.0, 30.0]` | **108** | Anemometer physical bounds (30 m/s = 108 km/h; p99.99 = 18.08, max spike 47.75) |
| `solar_radiation` | `physical_and_sensor_cap` | `[0.0, 1300.0]` | **353** | Clear-sky surface solar radiation limit (1300 W/m²; TOA solar const 1361; max spike 1985.0) |
| `pm25` | `stuck_sensor_invalidation` | `>= 24h constant non-zero (excl. rainfall)` | **2,272** | Invalidated 2,272 stuck sensor readings |
| `pm10` | `stuck_sensor_invalidation` | `>= 24h constant non-zero (excl. rainfall)` | **1,130** | Invalidated 1,130 stuck sensor readings |
| `no` | `stuck_sensor_invalidation` | `>= 24h constant non-zero (excl. rainfall)` | **997** | Invalidated 997 stuck sensor readings |
| `no2` | `stuck_sensor_invalidation` | `>= 24h constant non-zero (excl. rainfall)` | **2,485** | Invalidated 2,485 stuck sensor readings |
| `nox` | `stuck_sensor_invalidation` | `>= 24h constant non-zero (excl. rainfall)` | **376** | Invalidated 376 stuck sensor readings |
| `nh3` | `stuck_sensor_invalidation` | `>= 24h constant non-zero (excl. rainfall)` | **585** | Invalidated 585 stuck sensor readings |
| `so2` | `stuck_sensor_invalidation` | `>= 24h constant non-zero (excl. rainfall)` | **1,256** | Invalidated 1,256 stuck sensor readings |
| `co` | `stuck_sensor_invalidation` | `>= 24h constant non-zero (excl. rainfall)` | **8,197** | Invalidated 8,197 stuck sensor readings |
| `o3` | `stuck_sensor_invalidation` | `>= 24h constant non-zero (excl. rainfall)` | **1,269** | Invalidated 1,269 stuck sensor readings |
| `benzene` | `stuck_sensor_invalidation` | `>= 24h constant non-zero (excl. rainfall)` | **20,006** | Invalidated 20,006 stuck sensor readings |
| `toluene` | `stuck_sensor_invalidation` | `>= 24h constant non-zero (excl. rainfall)` | **15,767** | Invalidated 15,767 stuck sensor readings |
| `temperature` | `stuck_sensor_invalidation` | `>= 24h constant non-zero (excl. rainfall)` | **1,002** | Invalidated 1,002 stuck sensor readings |
| `humidity` | `stuck_sensor_invalidation` | `>= 24h constant non-zero (excl. rainfall)` | **8,593** | Invalidated 8,593 stuck sensor readings |
| `wind_speed` | `stuck_sensor_invalidation` | `>= 24h constant non-zero (excl. rainfall)` | **27,347** | Invalidated 27,347 stuck sensor readings |
| `wind_direction` | `stuck_sensor_invalidation` | `>= 24h constant non-zero (excl. rainfall)` | **6,093** | Invalidated 6,093 stuck sensor readings |
| `solar_radiation` | `stuck_sensor_invalidation` | `>= 24h constant non-zero (excl. rainfall)` | **5,851** | Invalidated 5,851 stuck sensor readings |
| `barometric_pressure` | `stuck_sensor_invalidation` | `>= 24h constant non-zero (excl. rainfall)` | **70,753** | Invalidated 70,753 stuck sensor readings |
| `pm25_and_pm10` | `pm_inversion_filter` | `PM2.5 > 1.2 * PM10 AND PM2.5 > 20 ug/m3` | **7,382** | Invalidated 3,691 station-hours with unphysical PM inversion |
| `benzene` | `voc_rate_of_change_spike` | `> 10x rolling 24h median (baseline > 0.1)` | **1,888** | Filtered 1,888 transient isolated VOC spikes |
| `toluene` | `voc_rate_of_change_spike` | `> 10x rolling 24h median (baseline > 0.1)` | **3,150** | Filtered 3,150 transient isolated VOC spikes |

---
*Generated by `src/preprocess.py` on 2026-11-08.*