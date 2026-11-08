# Delhi Air Quality — Data Inspection Report

## Plain-English Summary

The dataset contains **3,636,360** hourly observation rows across **37** monitoring stations in Delhi, spanning **2010-01-01** to **2026-09-30 23:00**.

Files are organised as one CSV per station-year inside year folders (2010–2026). Each CSV has the same 25 columns (Timestamp + 24 measurement fields). All files share identical column names and order.

Station coverage grows from 10 in 2010–2016 to 36–37 from 2017 onward. The last year (2026) is **partial** (data ends 2026-09-30).

**No AQI column** exists in the raw data — it must be computed.

CO is reported in **mg/m³** (milligrams), all other pollutants in **µg/m³** (micrograms). Text placeholder `"NA"` is used for missing values in newer files; earlier files use empty cells.


## Decisions You Need to Make

1. **Single station vs. city-wide aggregate**: Use all stations and average/aggregate spatially, or pick one or a few 'reference' stations with best coverage?

2. **Years to include**: 2010–2016 have only 10 stations (and some of those are header-only). Include them for long-term trend, or start from 2017 when 36 stations are available?

3. **2026 partial year**: Include partial 2026 for near-real-time analysis, or exclude to avoid incomplete-year bias?

4. **VOCs and rare met columns**: Keep benzene/toluene/xylene/ethylbenzene/mp-xylene (high null rates) or drop them to simplify?

5. **CO unit normalisation**: Convert CO to µg/m³ (×1000) for consistency, or keep mg/m³?

6. **AQI computation**: Compute sub-indices and overall AQI using India's NAQI formula? If so, which breakpoints?

7. **Duplicate-timestamp handling**: Remove exact duplicates, average them, or flag and keep?

8. **Imputation strategy**: Forward-fill short gaps, interpolate, model-based, or leave as NaN?


## Columns, Dtypes, and Roles

| # | Raw Column | Pandas Dtype | Role | Unit |
|---|---|---|---|---|
| 1 | `Timestamp` | str | datetime | datetime |
| 2 | `PM2.5 (µg/m³)` | float64 | pm25 | µg/m³ |
| 3 | `PM10 (µg/m³)` | float64 | pm10 | µg/m³ |
| 4 | `NO (µg/m³)` | float64 | no | µg/m³ |
| 5 | `NO2 (µg/m³)` | float64 | no2 | µg/m³ |
| 6 | `NOx (ppb)` | float64 | nox | ppb |
| 7 | `NH3 (µg/m³)` | float64 | nh3 | µg/m³ |
| 8 | `SO2 (µg/m³)` | float64 | so2 | µg/m³ |
| 9 | `CO (mg/m³)` | float64 | co | mg/m³ |
| 10 | `Ozone (µg/m³)` | float64 | o3 | µg/m³ |
| 11 | `Benzene (µg/m³)` | float64 | benzene | µg/m³ |
| 12 | `Toluene (µg/m³)` | float64 | toluene | µg/m³ |
| 13 | `Xylene (µg/m³)` | float64 | xylene | µg/m³ |
| 14 | `O Xylene (µg/m³)` | float64 | o_xylene | µg/m³ |
| 15 | `Eth-Benzene (µg/m³)` | float64 | eth_benzene | µg/m³ |
| 16 | `MP-Xylene (µg/m³)` | float64 | mp_xylene | µg/m³ |
| 17 | `AT (°C)` | float64 | temperature | °C |
| 18 | `RH (%)` | float64 | humidity | % |
| 19 | `WS (m/s)` | float64 | wind_speed | m/s |
| 20 | `WD (deg)` | float64 | wind_direction | deg |
| 21 | `RF (mm)` | float64 | rainfall | mm |
| 22 | `TOT-RF (mm)` | float64 | total_rainfall | mm |
| 23 | `SR (W/mt2)` | float64 | solar_radiation | W/mt2 |
| 24 | `BP (mmHg)` | float64 | barometric_pressure | mmHg |
| 25 | `VWS (m/s)` | float64 | vertical_wind_speed | m/s |

## Datetime Range & Frequency

- **Column**: `Timestamp`
- **Min**: 2010-01-01 00:00:00
- **Max**: 2026-09-30 23:00:00
- **Median row interval** (sampled): 0 days 01:00:00  → **hourly**
- **Duplicate timestamps** (across all files, within-file): 0
- **Last year (2026) partial**: YES

## Station & Row Coverage

- **Total unique stations**: 37

| Year | Files | Rows |
|------|-------|------|
| 2010 | 10 | 52,560 |
| 2011 | 10 | 70,200 |
| 2012 | 10 | 81,984 |
| 2013 | 10 | 87,600 |
| 2014 | 10 | 86,184 |
| 2015 | 10 | 87,600 |
| 2016 | 10 | 87,840 |
| 2017 | 36 | 315,360 |
| 2018 | 36 | 315,360 |
| 2019 | 36 | 315,360 |
| 2020 | 36 | 316,224 |
| 2021 | 36 | 315,360 |
| 2022 | 36 | 315,360 |
| 2023 | 36 | 315,360 |
| 2024 | 36 | 316,224 |
| 2025 | 36 | 315,360 |
| 2026 | 37 | 242,424 |

## Null Fraction per Column (Overall)

| Column | Role | Null % |
|--------|------|--------|
| `Timestamp` | datetime | 0.0% |
| `PM2.5 (µg/m³)` | pm25 | 24.1% |
| `PM10 (µg/m³)` | pm10 | 30.2% |
| `NO (µg/m³)` | no | 21.9% |
| `NO2 (µg/m³)` | no2 | 20.6% |
| `NOx (ppb)` | nox | 21.2% |
| `NH3 (µg/m³)` | nh3 | 38.5% |
| `SO2 (µg/m³)` | so2 | 33.0% |
| `CO (mg/m³)` | co | 27.3% |
| `Ozone (µg/m³)` | o3 | 23.5% |
| `Benzene (µg/m³)` | benzene | 40.8% |
| `Toluene (µg/m³)` | toluene | 43.5% |
| `Xylene (µg/m³)` | xylene | 92.7% |
| `O Xylene (µg/m³)` | o_xylene | 94.9% |
| `Eth-Benzene (µg/m³)` | eth_benzene | 87.3% |
| `MP-Xylene (µg/m³)` | mp_xylene | 86.0% |
| `AT (°C)` | temperature | 49.1% |
| `RH (%)` | humidity | 34.0% |
| `WS (m/s)` | wind_speed | 36.4% |
| `WD (deg)` | wind_direction | 34.8% |
| `RF (mm)` | rainfall | 59.5% |
| `TOT-RF (mm)` | total_rainfall | 14.5% |
| `SR (W/mt2)` | solar_radiation | 35.9% |
| `BP (mmHg)` | barometric_pressure | 45.0% |
| `VWS (m/s)` | vertical_wind_speed | 87.1% |

## Null % per Year (Key Columns)

| Year | pm25 | pm10 | no2 | co | o3 | temperature | humidity |
|------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| 2010 | 88.5% | 100.0% | 28.4% | 93.7% | 39.6% | 92.4% | 45.3% |
| 2011 | 77.2% | 100.0% | 34.2% | 85.9% | 39.1% | 85.3% | 40.2% |
| 2012 | 62.6% | 100.0% | 44.8% | 73.2% | 44.1% | 75.1% | 40.7% |
| 2013 | 78.9% | 100.0% | 58.1% | 90.3% | 65.5% | 93.1% | 64.9% |
| 2014 | 87.1% | 100.0% | 63.0% | 99.1% | 72.1% | 100.0% | 71.9% |
| 2015 | 56.7% | 75.2% | 44.5% | 73.5% | 48.1% | 73.9% | 54.1% |
| 2016 | 43.6% | 67.3% | 42.4% | 66.7% | 54.0% | 68.1% | 43.6% |
| 2017 | 77.2% | 87.9% | 78.2% | 83.5% | 86.1% | 88.1% | 81.5% |
| 2018 | 20.1% | 29.0% | 21.4% | 21.4% | 25.0% | 45.8% | 34.9% |
| 2019 | 7.1% | 15.6% | 7.2% | 8.4% | 10.0% | 37.4% | 23.9% |
| 2020 | 10.7% | 14.9% | 10.7% | 11.0% | 13.5% | 38.5% | 24.6% |
| 2021 | 9.3% | 11.2% | 9.7% | 10.1% | 9.5% | 38.2% | 22.5% |
| 2022 | 6.5% | 6.2% | 8.2% | 8.2% | 7.2% | 37.5% | 23.0% |
| 2023 | 7.9% | 8.2% | 8.0% | 9.7% | 8.5% | 38.7% | 26.9% |
| 2024 | 6.2% | 5.3% | 4.4% | 5.8% | 6.7% | 35.6% | 22.8% |
| 2025 | 4.5% | 4.5% | 3.0% | 7.0% | 6.1% | 34.6% | 21.6% |
| 2026 | 8.2% | 7.6% | 6.3% | 6.6% | 6.7% | 32.4% | 24.6% |

## Numeric Column Stats (Min / Max / Mean)

| Column | Role | Min | Max | Mean | Count |
|--------|------|-----|-----|------|-------|
| `PM2.5 (µg/m³)` | pm25 | 0 | 1000 | 105.2 | 2,760,818 |
| `PM10 (µg/m³)` | pm10 | 0 | 1000 | 209.2 | 2,537,521 |
| `NO (µg/m³)` | no | 0 | 500 | 31.29 | 2,840,560 |
| `NO2 (µg/m³)` | no2 | 0 | 500 | 43.48 | 2,887,914 |
| `NOx (ppb)` | nox | 0 | 500 | 52.98 | 2,863,737 |
| `NH3 (µg/m³)` | nh3 | 0 | 498.9 | 40.48 | 2,234,563 |
| `SO2 (µg/m³)` | so2 | 0 | 200 | 13.27 | 2,436,830 |
| `CO (mg/m³)` | co | 0 | 50 | 1.398 | 2,642,317 |
| `Ozone (µg/m³)` | o3 | 0 | 970 | 32.94 | 2,782,506 |
| `Benzene (µg/m³)` | benzene | 0 | 490 | 3.099 | 2,151,766 |
| `Toluene (µg/m³)` | toluene | 0 | 500 | 20.83 | 2,052,850 |
| `Xylene (µg/m³)` | xylene | 0 | 344.1 | 2.13 | 266,216 |
| `O Xylene (µg/m³)` | o_xylene | 0 | 2165 | 6.996 | 184,100 |
| `Eth-Benzene (µg/m³)` | eth_benzene | 0 | 2030 | 4.787 | 461,447 |
| `MP-Xylene (µg/m³)` | mp_xylene | 0 | 2164 | 5.344 | 507,416 |
| `AT (°C)` | temperature | 0 | 58.9 | 25.45 | 1,850,972 |
| `RH (%)` | humidity | 0.01 | 100 | 61.24 | 2,398,296 |
| `WS (m/s)` | wind_speed | 0 | 47.75 | 1.085 | 2,314,041 |
| `WD (deg)` | wind_direction | 0 | 360 | 181.6 | 2,370,969 |
| `RF (mm)` | rainfall | 0 | 50 | 0.02368 | 1,471,440 |
| `TOT-RF (mm)` | total_rainfall | -0.6 | 1.884e+05 | 11.79 | 3,110,587 |
| `SR (W/mt2)` | solar_radiation | 0 | 1995 | 127.7 | 2,331,584 |
| `BP (mmHg)` | barometric_pressure | 700 | 1099 | 949.8 | 2,000,879 |
| `VWS (m/s)` | vertical_wind_speed | -12.02 | 19.93 | 0.1493 | 469,649 |

## Text Placeholders in Numeric Columns

None found — all non-null values coerce to numeric.


## Negative Values in Pollutant Columns

No negative pollutant values found.


## Humidity Range Check

- Values > 100%: **0**
- Values < 0%:   **0**

## AQI Column

**Not present** in raw data. Must be computed during cleaning/feature-engineering.

## Pollutant Units

| Pollutant | Raw Column | Unit |
|-----------|-----------|------|
| pm25 | `PM2.5 (µg/m³)` | µg/m³ |
| pm10 | `PM10 (µg/m³)` | µg/m³ |
| no | `NO (µg/m³)` | µg/m³ |
| no2 | `NO2 (µg/m³)` | µg/m³ |
| nox | `NOx (ppb)` | ppb |
| nh3 | `NH3 (µg/m³)` | µg/m³ |
| so2 | `SO2 (µg/m³)` | µg/m³ |
| co | `CO (mg/m³)` | mg/m³ |
| o3 | `Ozone (µg/m³)` | µg/m³ |
| benzene | `Benzene (µg/m³)` | µg/m³ |
| toluene | `Toluene (µg/m³)` | µg/m³ |
| xylene | `Xylene (µg/m³)` | µg/m³ |
| o_xylene | `O Xylene (µg/m³)` | µg/m³ |
| eth_benzene | `Eth-Benzene (µg/m³)` | µg/m³ |
| mp_xylene | `MP-Xylene (µg/m³)` | µg/m³ |

> ⚠ **CO is in mg/m³** while every other gas/particle pollutant is in µg/m³. Decide whether to convert CO to µg/m³ (×1000) before modelling.

## Rows per Station per Year (sample)

Showing first 5 stations alphabetically.

| Station | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| site_103_CRRI_Mathura_Road_Delhi_IITM | — | — | — | — | — | — | — | 8760 | 8760 | 8760 | 8784 | 8760 | 8760 | 8760 | 8784 | 8760 | 6552 |
| site_104_Burari_Crossing_Delhi_IITM | — | — | — | — | — | — | — | 8760 | 8760 | 8760 | 8784 | 8760 | 8760 | 8760 | 8784 | 8760 | 6552 |
| site_105_North_Campus_DU_Delhi_IITM | — | — | — | — | — | — | — | 8760 | 8760 | 8760 | 8784 | 8760 | 8760 | 8760 | 8784 | 8760 | 6552 |
| site_106_IGI_Airport_(T3)_Delhi_IITM | — | — | — | — | — | — | — | 8760 | 8760 | 8760 | 8784 | 8760 | 8760 | 8760 | 8784 | 8760 | 6552 |
| site_107_Pusa_Delhi_IITM | — | — | — | — | — | — | — | 8760 | 8760 | 8760 | 8784 | 8760 | 8760 | 8760 | 8784 | 8760 | 6552 |
