"""
Central configuration for the Delhi Air Quality project.

COLUMN_MAP maps raw CSV column names → semantic roles.
All raw column names match the CPCB export headers exactly (units included).
"""

from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"

# ── Column mapping: raw header → role ─────────────────────────────────────────
# Roles used downstream: datetime, station (derived from filename), pm25, pm10,
# no, no2, nox, nh3, so2, co, o3, benzene, toluene, xylene, o_xylene,
# eth_benzene, mp_xylene, temperature, humidity, wind_speed, wind_direction,
# rainfall, total_rainfall, solar_radiation, barometric_pressure,
# vertical_wind_speed
#
# NOTE: There is no AQI column in the raw data — it must be computed if needed.
# NOTE: CO is in mg/m³ while most other pollutants are in µg/m³.

COLUMN_MAP = {
    # ── datetime ──────────────────────────────────────────────────────────────
    "Timestamp":            "datetime",

    # ── criteria pollutants ───────────────────────────────────────────────────
    "PM2.5 (µg/m³)":       "pm25",
    "PM10 (µg/m³)":        "pm10",
    "NO (µg/m³)":          "no",
    "NO2 (µg/m³)":         "no2",
    "NOx (ppb)":           "nox",
    "NH3 (µg/m³)":         "nh3",
    "SO2 (µg/m³)":         "so2",
    "CO (mg/m³)":          "co",         # ⚠ milligrams, not micrograms
    "Ozone (µg/m³)":       "o3",

    # ── VOCs ──────────────────────────────────────────────────────────────────
    "Benzene (µg/m³)":     "benzene",
    "Toluene (µg/m³)":     "toluene",
    "Xylene (µg/m³)":      "xylene",
    "O Xylene (µg/m³)":    "o_xylene",
    "Eth-Benzene (µg/m³)": "eth_benzene",
    "MP-Xylene (µg/m³)":   "mp_xylene",

    # ── meteorology ───────────────────────────────────────────────────────────
    "AT (°C)":             "temperature",
    "RH (%)":              "humidity",
    "WS (m/s)":            "wind_speed",
    "WD (deg)":            "wind_direction",
    "RF (mm)":             "rainfall",
    "TOT-RF (mm)":         "total_rainfall",
    "SR (W/mt2)":          "solar_radiation",
    "BP (mmHg)":           "barometric_pressure",  # Output unit standardized to hPa (converted from raw mmHg if < 850)
    "VWS (m/s)":           "vertical_wind_speed",
}

# Reverse lookup: role → raw column name
ROLE_TO_RAW = {v: k for k, v in COLUMN_MAP.items()}

# Columns that are pollutants (for range / negative checks)
POLLUTANT_ROLES = [
    "pm25", "pm10", "no", "no2", "nox", "nh3", "so2", "co", "o3",
    "benzene", "toluene", "xylene", "o_xylene", "eth_benzene", "mp_xylene",
]

# Station identity is NOT a CSV column; it is derived from the filename.
# Format: site_<id>_<station_name>_<agency>_1Hr.csv

# ── Processed data paths ──────────────────────────────────────────────────────
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ── Cleaning decisions (confirmed) ───────────────────────────────────────────
CLEANING = {
    "year_range":                 (2018, 2026),   # 2018-01-01 onward
    "include_partial_year":       True,           # include partial 2026 (through Sep 30)
    "station_min_coverage":       0.50,           # ≥ 50% hourly coverage of PM2.5 or PM10
    "station_min_reporting":      3,              # ≥ 3 stations reporting per pollutant-hour
    "high_null_threshold":        0.80,           # drop columns >80% null over 2018–2026
    "keep_co_in_mg":              True,           # keep CO in mg/m³
    "compute_aqi":                True,           # compute India NAQI
    "max_interp_gap_hours":       6,              # linear interpolation inside gaps <= 6h per station
}

# Columns to drop (>80% null from 2018–2026 inspection)
DROP_COLS_HIGH_NULL = [
    "Xylene (µg/m³)",
    "O Xylene (µg/m³)",
    "Eth-Benzene (µg/m³)",
    "MP-Xylene (µg/m³)",
    "VWS (m/s)",
]

# ── Chronological Split Configuration ─────────────────────────────────────────
TRAIN_END = "2023-03-31"
VAL_END   = "2024-03-31"
GAP_DAYS  = 5

# ── Cross-Validation Configuration ───────────────────────────────────────────
CV_SPLITS = 4
CV_GAP    = 5

# ── Festival & Event Dates (Unverified — pending verification against an official calendar by project owner) ──
DIWALI_DATES = [
    "2018-11-07",
    "2019-10-27",
    "2020-11-14",
    "2021-11-04",
    "2022-10-24",
    "2023-11-12",
    "2024-11-01",
    "2025-10-20",
]




