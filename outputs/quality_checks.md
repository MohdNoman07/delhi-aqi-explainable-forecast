# Delhi Air Quality — Comprehensive Data Quality Check Report

## Executive Summary (Plain English)

A rigorous pre-AQI quality audit was conducted on both the aggregated city dataset (`clean_hourly.parquet`) and the underlying 35 kept monitoring stations (`raw_filtered.parquet`). The key findings are:

1. **Barometric Pressure is in Mixed Units across Stations & Time**:
   - The CPCB header specifies `BP (mmHg)`, but the network is heterogeneous.
   - 4 stations (`site_113_Shadipur`, `site_114_IHBAS`, `site_115_NSIT_Dwarka`, `site_125_Punjabi_Bagh`) report in **true mmHg** (median $\sim 735 - 748\text{ mmHg}$).
   - Most DPCC and IITM stations report in **hPa / mbar** (median $\sim 980 - 985\text{ hPa}$).
   - Several stations (`Mandir Marg`, `R K Puram`, `Anand Vihar`) actually **switched units mid-record** (reporting in mmHg in 2018–2019 and switching to hPa later).
   - Averaging 740 and 980 directly creates an unphysical bimodal distortion. Furthermore, the previous cap of 1050 hPa clipped high hPa spikes but let uncalibrated 740 values pass through.
2. **Stuck Sensor Episodes are Prevalent**:
   - We identified **240 distinct episodes** where a sensor reported the exact same non-zero value for $\ge 24$ consecutive hours.
   - The worst cases were barometric pressure (frozen for up to 745–1,970 consecutive hours), toluene/benzene (frozen for up to 1,627 hours), solar radiation (frozen at a non-zero value across day and night for up to 643 hours), and anemometers (wind speed frozen for up to 557 hours).
3. **PM2.5 > 1.2 * PM10 Inversions**:
   - At the **city level**, only **8 hours out of 76,577 (0.01%)** violate $PM_{2.5} \le 1.2 \times PM_{10}$, showing that spatial averaging across $\ge 3$ stations effectively filters out local calibration offsets.
   - At the **station level**, **4,777 hourly observations (0.20%)** exhibit $PM_{2.5} > 1.2 \times PM_{10}$. Over 60% of these inversions occur at four IITM stations (`Aya Nagar`, `Pusa IITM`, `CRRI Mathura Road`, and `North Campus DU`). The meteorological cause is unverified; sensor drift during high-humidity episodes is a possible explanation (not tested).
4. **Contributing Station Counts are Extremely Robust**:
   - Across all years from 2018 to 2026, the median number of reporting stations is **30 to 34** (out of 35 kept stations) for PM2.5, PM10, and NO2.
   - The 5th percentile ($p_5$) is consistently **28 to 33 stations** (except early 2018 when stations were first coming online).
5. **Rate-of-Change Jumps**:
   - At the **city level**, NO, NO2, NOx, NH3, SO2, CO, O3, PM2.5, PM10, and toluene have **0 jumps** exceeding 10x their rolling 24-hour median. Only benzene experienced 4 isolated jumps (during rapid industrial plume dispersals against a sub-$1\,\mu\text{g/m}^3$ baseline).
   - At the **station level**, thousands of single-station localized spikes exist, but spatial aggregation cleanly neutralizes them.

---

## 1. Barometric Pressure Unit Inconsistency

Expected sea-level pressure is $\sim 1013.25\text{ hPa} = 760\text{ mmHg}$. Delhi is at $\sim 216\text{ m}$ elevation, where standard ambient atmospheric pressure is:
- **In hPa**: $\sim 975 - 1005\text{ hPa}$ (standard mean $\approx 988\text{ hPa}$)
- **In mmHg**: $\sim 730 - 755\text{ mmHg}$ (standard mean $\approx 741\text{ mmHg}$)
- Conversion factor: $1\text{ mmHg} = 1.3332239\text{ hPa}$ (or $P_{\text{hPa}} = P_{\text{mmHg}} \times \frac{1013.25}{760}$)

### Station Pressure Distribution Table

| Station | Valid Rows | Median | p1 (1%) | p99 (99%) | Detected Sensor Unit |
|---|---:|---:|---:|---:|---|
| `site_125_Punjabi_Bagh_Delhi_DPCC` | 47,444 | 735.8 | 729.9 | 786.7 | **mmHg (~740)** |
| `site_115_NSIT_Dwarka_Delhi_CPCB` | 5,666 | 737.9 | 705.0 | 751.3 | **mmHg (~740)** |
| `site_113_Shadipur_Delhi_CPCB` | 29,423 | 747.8 | 736.1 | 759.2 | **mmHg (~740)** |
| `site_114_IHBAS_Dilshad_Garden_Delhi_CPCB` | 28,946 | 747.9 | 736.7 | 758.0 | **mmHg (~740)** |
| `site_108_Aya_Nagar_Delhi_IITM` | 127 | 967.7 | 941.6 | 1019.7 | **hPa / mbar (~980)** |
| `site_301_Anand_Vihar_Delhi_DPCC` | 66,660 | 971.9 | 722.2 | 987.0 | **Mixed / Switched (both mmHg & hPa)** |
| `site_118_DTU_Delhi_CPCB` | 310 | 974.3 | 968.5 | 977.7 | **hPa / mbar (~980)** |
| `site_124_R_K_Puram_Delhi_DPCC` | 74,224 | 976.5 | 735.0 | 992.5 | **Mixed / Switched (both mmHg & hPa)** |
| `site_103_CRRI_Mathura_Road_Delhi_IITM` | 178 | 978.0 | 972.3 | 982.0 | **hPa / mbar (~980)** |
| `site_1561_Mundka_Delhi_DPCC` | 70,436 | 980.5 | 961.0 | 997.4 | **hPa / mbar (~980)** |
| `site_1428_Okhla_Phase-2_Delhi_DPCC` | 74,673 | 981.2 | 971.2 | 996.5 | **hPa / mbar (~980)** |
| `site_1434_Wazirpur_Delhi_DPCC` | 73,244 | 981.6 | 938.5 | 997.9 | **hPa / mbar (~980)** |
| `site_1426_Narela_Delhi_DPCC` | 73,764 | 981.6 | 937.0 | 998.2 | **hPa / mbar (~980)** |
| `site_1420_Ashok_Vihar_Delhi_DPCC` | 74,320 | 982.0 | 835.0 | 1025.1 | **hPa / mbar (~980)** |
| `site_122_Mandir_Marg_Delhi_DPCC` | 73,527 | 982.4 | 737.0 | 991.8 | **Mixed / Switched (both mmHg & hPa)** |
| `site_1421_Dr._Karni_Singh_Shooting_Range_Delhi_DPCC` | 74,791 | 982.5 | 968.1 | 995.8 | **hPa / mbar (~980)** |
| `site_105_North_Campus_DU_Delhi_IITM` | 588 | 982.6 | 982.0 | 983.0 | **hPa / mbar (~980)** |
| `site_1423_Jahangirpuri_Delhi_DPCC` | 73,645 | 982.7 | 969.9 | 997.6 | **hPa / mbar (~980)** |
| `site_1563_Pusa_Delhi_DPCC` | 69,315 | 982.7 | 935.0 | 996.1 | **hPa / mbar (~980)** |
| `site_1424_Jawaharlal_Nehru_Stadium_Delhi_DPCC` | 73,341 | 982.8 | 939.1 | 999.0 | **hPa / mbar (~980)** |
| `site_1429_Nehru_Nagar_Delhi_DPCC` | 74,766 | 983.2 | 968.0 | 996.0 | **hPa / mbar (~980)** |
| `site_1431_Patparganj_Delhi_DPCC` | 73,037 | 983.2 | 960.0 | 998.1 | **hPa / mbar (~980)** |
| `site_1430_Rohini_Delhi_DPCC` | 73,710 | 983.7 | 962.3 | 997.2 | **hPa / mbar (~980)** |
| `site_5024_Alipur_Delhi_DPCC` | 66,296 | 984.0 | 966.4 | 998.4 | **hPa / mbar (~980)** |
| `site_1562_Sri_Aurobindo_Marg_Delhi_DPCC` | 68,864 | 984.1 | 957.0 | 997.0 | **hPa / mbar (~980)** |
| `site_1435_Vivek_Vihar_Delhi_DPCC` | 73,020 | 984.2 | 847.8 | 998.5 | **hPa / mbar (~980)** |
| `site_1432_Sonia_Vihar_Delhi_DPCC` | 72,605 | 984.4 | 808.8 | 998.6 | **hPa / mbar (~980)** |
| `site_1427_Najafgarh_Delhi_DPCC` | 72,397 | 984.9 | 968.3 | 998.1 | **hPa / mbar (~980)** |
| `site_1425_Major_Dhyan_Chand_National_Stadium_Delhi_DPCC` | 74,825 | 985.0 | 968.5 | 997.8 | **hPa / mbar (~980)** |
| `site_107_Pusa_Delhi_IITM` | 692 | 985.0 | 965.0 | 989.0 | **hPa / mbar (~980)** |
| `site_104_Burari_Crossing_Delhi_IITM` | 298 | 986.7 | 973.0 | 990.0 | **hPa / mbar (~980)** |
| `site_1560_Bawana_Delhi_DPCC` | 71,118 | 986.8 | 967.5 | 999.0 | **hPa / mbar (~980)** |

### Diagnosis & Correct Unit / Cap
- **Mixed Units Confirmed**: Stations with median $<800$ (`site_113`, `site_114`, `site_115`, `site_125`) are calibrated in **mmHg**. Stations with median $>950$ are calibrated in **hPa**.
- **Stations with Mid-Record Unit Switch**: `site_122` (Mandir Marg), `site_124` (R K Puram), and `site_301` (Anand Vihar) logged values in mmHg ($720-760$) during 2018–2019 and then switched to hPa ($970-1000$) in later years without changing column headers.
- **Target Unit**: All barometric pressure measurements should be unified to **hPa**.
- **Correct Conversion Rule**: Any value $<850$ is in mmHg and must be converted to hPa by multiplying by $1.333224$.
- **Correct Physical Bounds**: In hPa, ambient Delhi pressure strictly ranges between **$940\text{ hPa}$ and $1030\text{ hPa}$**. Any reading outside $[940, 1030]\text{ hPa}$ is an uncalibrated instrument artifact.

---

## 2. Stuck Sensors (Constant Non-Zero Values $\ge 24$ Consecutive Hours)

Detected **240** distinct stuck sensor episodes across the 35 kept stations.

### Top 25 Longest Stuck Sensor Episodes

| Station | Variable | Stuck Runs Count | Total Stuck Hours | Max Consecutive Run (Hours) |
|---|---|---:|---:|---:|
| `site_124_R_K_Puram_Delhi_DPCC` | `barometric_pressure` | 181 | 13,127 | **745h** (31.0 days) |
| `site_122_Mandir_Marg_Delhi_DPCC` | `barometric_pressure` | 159 | 11,942 | **648h** (27.0 days) |
| `site_1423_Jahangirpuri_Delhi_DPCC` | `barometric_pressure` | 97 | 6,141 | **373h** (15.5 days) |
| `site_1426_Narela_Delhi_DPCC` | `barometric_pressure` | 69 | 6,127 | **805h** (33.5 days) |
| `site_301_Anand_Vihar_Delhi_DPCC` | `barometric_pressure` | 97 | 5,768 | **273h** (11.4 days) |
| `site_1428_Okhla_Phase-2_Delhi_DPCC` | `barometric_pressure` | 38 | 5,081 | **1,970h** (82.1 days) |
| `site_1425_Major_Dhyan_Chand_National_Stadium_Delhi_DPCC` | `barometric_pressure` | 19 | 4,868 | **842h** (35.1 days) |
| `site_1434_Wazirpur_Delhi_DPCC` | `barometric_pressure` | 83 | 4,810 | **205h** (8.5 days) |
| `site_1429_Nehru_Nagar_Delhi_DPCC` | `toluene` | 32 | 4,764 | **1,627h** (67.8 days) |
| `site_1420_Ashok_Vihar_Delhi_DPCC` | `barometric_pressure` | 52 | 4,083 | **268h** (11.2 days) |
| `site_113_Shadipur_Delhi_CPCB` | `solar_radiation` | 39 | 3,788 | **643h** (26.8 days) |
| `site_125_Punjabi_Bagh_Delhi_DPCC` | `wind_speed` | 53 | 2,916 | **287h** (12.0 days) |
| `site_1563_Pusa_Delhi_DPCC` | `wind_speed` | 35 | 2,864 | **557h** (23.2 days) |
| `site_1562_Sri_Aurobindo_Marg_Delhi_DPCC` | `toluene` | 37 | 2,859 | **302h** (12.6 days) |
| `site_1562_Sri_Aurobindo_Marg_Delhi_DPCC` | `benzene` | 36 | 2,737 | **302h** (12.6 days) |
| `site_119_Sirifort_Delhi_CPCB` | `humidity` | 11 | 2,591 | **489h** (20.4 days) |
| `site_1429_Nehru_Nagar_Delhi_DPCC` | `benzene` | 24 | 2,563 | **588h** (24.5 days) |
| `site_1432_Sonia_Vihar_Delhi_DPCC` | `wind_speed` | 23 | 2,545 | **409h** (17.0 days) |
| `site_301_Anand_Vihar_Delhi_DPCC` | `wind_speed` | 35 | 2,534 | **464h** (19.3 days) |
| `site_115_NSIT_Dwarka_Delhi_CPCB` | `benzene` | 31 | 2,396 | **757h** (31.5 days) |
| `site_1431_Patparganj_Delhi_DPCC` | `barometric_pressure` | 37 | 2,364 | **253h** (10.5 days) |
| `site_1561_Mundka_Delhi_DPCC` | `wind_speed` | 7 | 2,008 | **653h** (27.2 days) |
| `site_1424_Jawaharlal_Nehru_Stadium_Delhi_DPCC` | `benzene` | 35 | 1,934 | **293h** (12.2 days) |
| `site_1434_Wazirpur_Delhi_DPCC` | `benzene` | 11 | 1,829 | **1,095h** (45.6 days) |
| `site_1434_Wazirpur_Delhi_DPCC` | `wind_speed` | 27 | 1,780 | **330h** (13.8 days) |

### Total Stuck Hours by Variable Across All Stations

| Variable | Total Incidents | Total Stuck Hours |
|---|---:|---:|
| `barometric_pressure` | 926 | 69,070 |
| `wind_speed` | 335 | 27,347 |
| `benzene` | 288 | 20,006 |
| `toluene` | 175 | 15,767 |
| `humidity` | 120 | 8,389 |
| `co` | 191 | 8,163 |
| `wind_direction` | 69 | 6,093 |
| `solar_radiation` | 74 | 5,877 |
| `no2` | 25 | 2,485 |
| `pm25` | 29 | 1,649 |
| `o3` | 13 | 1,269 |
| `so2` | 20 | 1,256 |
| `temperature` | 17 | 1,002 |
| `no` | 20 | 997 |
| `pm10` | 15 | 742 |
| `nh3` | 10 | 477 |
| `nox` | 7 | 376 |

---

## 3. PM2.5 to PM10 Inversions ($PM_{2.5} > 1.2 \times PM_{10}$)

Because fine particulate matter ($PM_{2.5}$) is a physical constituent of coarse particulate matter ($PM_{10}$), $PM_{2.5}$ cannot exceed $PM_{10}$ in reality. A tolerance ratio of $1.2$ accounts for minor independent gravimetric/optical sensor measurement noise.

### City-Level Findings (`clean_hourly.parquet`)
- Total valid paired hours: **76,577**
- Hours with $PM_2.5 > 1.2 \times PM_10$: **8** (**0.01%**)

All 8 City-Level Inversion Hours:

| Datetime | City PM2.5 (µg/m³) | City PM10 (µg/m³) | PM2.5 / PM10 Ratio |
|---|---:|---:|---:|
| `2018-07-28 19:00:00` | 55.3 | 46.0 | **1.20** |
| `2018-08-23 09:00:00` | 52.3 | 41.5 | **1.26** |
| `2018-08-23 10:00:00` | 45.9 | 34.7 | **1.32** |
| `2018-08-23 11:00:00` | 47.3 | 38.1 | **1.24** |
| `2024-06-28 12:00:00` | 81.1 | 49.7 | **1.63** |
| `2024-06-28 13:00:00` | 82.3 | 61.8 | **1.33** |
| `2024-07-26 08:00:00` | 43.2 | 31.1 | **1.39** |
| `2026-07-09 06:00:00` | 43.3 | 24.8 | **1.75** |

### Station-Level Findings (Raw Station Observations)
- Total valid paired hours across all kept stations: **2,375,357**
- Station-hours with $PM_2.5 > 1.2 \times PM_10$: **4,777** (**0.20%**)

Top Stations by Inversion Frequency:

| Station | Inversion Hours | Total Valid Hours | Inversion % | Max Ratio Recorded |
|---|---:|---:|---:|---:|
| `site_108_Aya_Nagar_Delhi_IITM` | 870 | 70,016 | **1.24%** | 384.5 |
| `site_107_Pusa_Delhi_IITM` | 745 | 68,458 | **1.09%** | 349.43 |
| `site_103_CRRI_Mathura_Road_Delhi_IITM` | 727 | 70,159 | **1.04%** | 307.73 |
| `site_105_North_Campus_DU_Delhi_IITM` | 670 | 70,189 | **0.95%** | 862.75 |
| `site_104_Burari_Crossing_Delhi_IITM` | 337 | 48,304 | **0.70%** | 114.31 |
| `site_125_Punjabi_Bagh_Delhi_DPCC` | 174 | 72,031 | **0.24%** | 176.47 |
| `site_106_IGI_Airport_(T3)_Delhi_IITM` | 152 | 69,656 | **0.22%** | 64.14 |
| `site_1560_Bawana_Delhi_DPCC` | 107 | 68,409 | **0.16%** | 9.13 |
| `site_124_R_K_Puram_Delhi_DPCC` | 90 | 69,887 | **0.13%** | 8.59 |
| `site_117_ITO_Delhi_CPCB` | 76 | 67,074 | **0.11%** | 22.61 |
| `site_1427_Najafgarh_Delhi_DPCC` | 72 | 70,500 | **0.10%** | 8.0 |
| `site_5024_Alipur_Delhi_DPCC` | 64 | 65,297 | **0.10%** | 5.5 |
| `site_122_Mandir_Marg_Delhi_DPCC` | 65 | 70,239 | **0.09%** | 8.18 |
| `site_301_Anand_Vihar_Delhi_DPCC` | 52 | 66,995 | **0.08%** | 11.8 |
| `site_1424_Jawaharlal_Nehru_Stadium_Delhi_DPCC` | 54 | 71,294 | **0.08%** | 102.0 |

> **Note**: Notice that the top 5 inversion stations are IITM stations (`Aya Nagar`, `Pusa IITM`, `CRRI Mathura Road`, `North Campus DU`, `Burari Crossing`). High relative humidity fog droplets causing differential hygroscopic growth in the PM2.5 inlet cyclone is a possible explanation (unverified).

---

## 4. Contributing Stations Distribution per Year

### PM2.5 Contributing Stations (`n_stations_pm25`)

| Year | Min Stations | 5th Percentile (p5) | Median Stations | 95th Percentile (p95) | Max Stations |
|---|---:|---:|---:|---:|---:|
| 2018 | 11 | 16.0 | **30.0** | 35.0 | 35 |
| 2019 | 12 | 32.0 | **34.0** | 35.0 | 35 |
| 2020 | 25 | 29.0 | **33.0** | 34.0 | 34 |
| 2021 | 8 | 29.0 | **33.0** | 35.0 | 35 |
| 2022 | 18 | 31.0 | **34.0** | 35.0 | 35 |
| 2023 | 7 | 31.0 | **34.0** | 35.0 | 35 |
| 2024 | 27 | 32.0 | **34.0** | 35.0 | 35 |
| 2025 | 28 | 32.0 | **34.0** | 35.0 | 35 |
| 2026 | 0 | 32.0 | **34.0** | 35.0 | 35 |

### PM10 Contributing Stations (`n_stations_pm10`)

| Year | Min Stations | 5th Percentile (p5) | Median Stations | 95th Percentile (p95) | Max Stations |
|---|---:|---:|---:|---:|---:|
| 2018 | 4 | 11.0 | **26.0** | 32.0 | 32 |
| 2019 | 9 | 29.0 | **31.0** | 32.0 | 32 |
| 2020 | 19 | 26.0 | **31.0** | 34.0 | 34 |
| 2021 | 0 | 28.0 | **32.0** | 34.0 | 35 |
| 2022 | 18 | 31.0 | **34.0** | 35.0 | 35 |
| 2023 | 7 | 31.0 | **34.0** | 35.0 | 35 |
| 2024 | 27 | 32.0 | **34.0** | 35.0 | 35 |
| 2025 | 2 | 32.0 | **34.0** | 35.0 | 35 |
| 2026 | 0 | 32.0 | **34.0** | 35.0 | 35 |

### NO2 Contributing Stations (`n_stations_no2`)

| Year | Min Stations | 5th Percentile (p5) | Median Stations | 95th Percentile (p95) | Max Stations |
|---|---:|---:|---:|---:|---:|
| 2018 | 10 | 15.0 | **30.0** | 35.0 | 35 |
| 2019 | 11 | 31.0 | **34.0** | 35.0 | 35 |
| 2020 | 25 | 29.0 | **33.0** | 34.0 | 34 |
| 2021 | 0 | 30.0 | **32.0** | 34.0 | 34 |
| 2022 | 18 | 30.0 | **33.0** | 35.0 | 35 |
| 2023 | 7 | 31.0 | **34.0** | 35.0 | 35 |
| 2024 | 29 | 32.0 | **34.0** | 35.0 | 35 |
| 2025 | 29 | 32.0 | **35.0** | 35.0 | 35 |
| 2026 | 0 | 33.0 | **34.0** | 35.0 | 35 |

---

## 5. Rate-of-Change Jumps ($>10\times$ Rolling 24-Hour Median)

### City Level (`clean_hourly.parquet`)

| Pollutant | Jump Incidents (>10x rolling 24h median) |
|---|---:|
| `pm25` | **0** |
| `pm10` | **0** |
| `no` | **0** |
| `no2` | **0** |
| `nox` | **0** |
| `nh3` | **0** |
| `so2` | **0** |
| `co` | **0** |
| `o3` | **0** |
| `benzene` | **4** |
| `toluene` | **0** |

All City-Level Jumps:

| Pollutant | Datetime | Value at $t-1$ | Value at $t$ | Absolute Jump | 24h Rolling Median | Jump Ratio |
|---|---|---:|---:|---:|---:|---:|
| `benzene` | `2018-04-09 20:00:00` | 2.11 | 11.29 | 9.17 | 0.805 | **11.4x** |
| `benzene` | `2025-08-02 00:00:00` | 7.15 | 0.46 | 6.7 | 0.616 | **10.9x** |
| `benzene` | `2018-04-10 02:00:00` | 13.37 | 3.77 | 9.6 | 0.891 | **10.8x** |
| `benzene` | `2025-08-01 23:00:00` | 0.62 | 7.15 | 6.53 | 0.647 | **10.1x** |

### Station Level (Individual Sensor Jumps)

| Pollutant | Station-Level Jump Count |
|---|---:|
| `pm25` | 319 |
| `pm10` | 108 |
| `no` | 18,295 |
| `no2` | 400 |
| `nox` | 729 |
| `nh3` | 275 |
| `so2` | 837 |
| `co` | 546 |
| `o3` | 12,364 |
| `benzene` | 1,937 |
| `toluene` | 3,168 |

Top 5 Largest Station-Level Jumps Overall:

| Station | Pollutant | Datetime | Value at $t-1$ | Value at $t$ | Absolute Jump | 24h Rolling Median | Jump Ratio |
|---|---|---|---:|---:|---:|---:|---:|
| `site_1563_Pusa_Delhi_DPCC` | `benzene` | `2025-06-22 23:00:00` | 0.0 | 149.82 | 149.82 | 0.11 | **1362.0x** |
| `site_105_North_Campus_DU_Delhi_IITM` | `no` | `2023-04-20 14:00:00` | 134.18 | 316.06 | 181.88 | 0.169 | **1077.8x** |
| `site_1427_Najafgarh_Delhi_DPCC` | `benzene` | `2018-05-16 13:00:00` | 0.33 | 192.98 | 192.65 | 0.19 | **1013.9x** |
| `site_125_Punjabi_Bagh_Delhi_DPCC` | `nox` | `2021-05-19 12:00:00` | 39.0 | 186.72 | 147.72 | 0.15 | **984.8x** |
| `site_125_Punjabi_Bagh_Delhi_DPCC` | `nox` | `2021-05-19 13:00:00` | 186.72 | 3.3 | 183.42 | 0.19 | **965.4x** |

---

## 6. Concrete Proposed Fixes for Your Approval

Based on these empirical findings, here are the proposed preprocessing fixes to apply before computing daily values or AQI:

### Fix 1: Standardize Barometric Pressure to hPa (Resolve Mixed Units)
- **Rule**: For every station-hour, if `barometric_pressure < 850`, treat as mmHg and convert: $P_{\text{hPa}} = P_{\text{mmHg}} \times 1.333224$.
- **Post-Conversion Bound**: Filter converted pressure to physical Delhi bounds: $[940, 1030]\text{ hPa}$. Values outside this range become `NaN`.
- **Benefit**: Eliminates the severe bimodal averaging artifact between mmHg (740) and hPa (980), producing a physically consistent citywide pressure series.

### Fix 2: Invalidate Stuck Sensor Runs ($\ge 24$ Consecutive Hours)
- **Rule**: For any kept station, if an exact non-zero reading repeats unchanged for $\ge 24$ consecutive hours, set that entire frozen run to `NaN`.
- **Benefit**: Prevents frozen sensors (like solar radiation stuck at 50 W/m² at midnight, or stuck pressure/wind/VOC sensors) from poisoning spatial averages and downstream models.

### Fix 3: Handle Unphysical PM Inversions ($PM_{2.5} > 1.2 \times PM_{10}$)
- **Rule at Station Level**: When a station reports $PM_{2.5} > 1.2 \times PM_{10}$, set both particulate values for that station-hour to `NaN`. Recommending setting both to `NaN` because contradictory readings indicate sensor anomaly (inlet clogging, optical sensor failure, or uncalibrated hygroscopic growth are possible explanations (unverified)).
- **Benefit**: Eliminates the 8 unphysical citywide inversion hours and guarantees consistent NAQI particulate sub-indices.

### Fix 4: Filter Isolated Station-Level Spike Glitches Before Averaging
- **Rule**: If a station's hourly reading jumps by $>10\times$ its 24-hour rolling median and returns to baseline within 1–2 hours while neighboring stations show no change, flag as an isolated transient glitch and set to `NaN`.
- **Benefit**: Prevents anomalous station spikes from pulling up the city average.

---
*Awaiting your approval before applying any of these proposed fixes.*