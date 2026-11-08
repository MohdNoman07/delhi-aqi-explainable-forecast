"""
src/daily_aggregation.py

Aggregates clean hourly citywide data (data/processed/clean_hourly.parquet)
into a complete daily calendar table (data/processed/clean_daily.csv)
and computes CPCB-style NAQI (breakpoints pending verification by the project owner against the official CPCB report).

Follows project rules and CPCB-style NAQI guidelines:
- Naive IST timestamps, day is the calendar day in the file's time.
- Calendar timeline is complete (2018-01-01 to 2026-09-30). Days with insufficient data remain as NaN rows.
- Daily PM2.5, PM10, NO2, SO2, NH3 = 24-hour mean (requires >= 16 valid hours).
- CO and O3 = 8-hour trailing rolling mean computed on the HOURLY data (requires >= 6 of 8 valid hours),
  then the daily maximum of these 8h rolling means.
- Wind direction = circular vector mean (arctan2(mean(sin), mean(cos))), plus sin and cos components.
- AQI computed from city-average concentrations, not from station AQIs.
- Requires at least 3 pollutants with a valid sub-index, and at least one must be PM2.5 or PM10.
- Saves:
    - data/processed/clean_daily.csv
    - outputs/daily_coverage.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.compute_aqi import sub_index, compute_aqi_from_subs, get_aqi_category, AQI_POLLUTANTS


def run_daily_aggregation():
    hourly_path = REPO_ROOT / "data" / "processed" / "clean_hourly.parquet"
    if not hourly_path.exists():
        raise FileNotFoundError(f"Hourly dataset not found at {hourly_path}")

    print(f"Loading {hourly_path} ...")
    df = pd.read_parquet(hourly_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # 1. Compute 8-hour trailing rolling means on HOURLY data for CO and O3
    # Requires at least 6 of 8 valid hours (min_periods=6)
    df["co_8h"] = df["co"].rolling(window=8, min_periods=6).mean()
    df["o3_8h"] = df["o3"].rolling(window=8, min_periods=6).mean()

    # 2. Hourly unit vector components for wind direction
    # wind_direction is in degrees [0, 360)
    rad = np.radians(df["wind_direction"])
    df["wind_sin"] = np.sin(rad)
    df["wind_cos"] = np.cos(rad)

    df["date"] = df["datetime"].dt.date

    # Criteria pollutants and meteorological variables
    pollutants_24h = ["pm25", "pm10", "no", "no2", "nox", "nh3", "so2", "co", "o3", "benzene", "toluene"]
    weather_vars = ["temperature", "humidity", "wind_speed", "solar_radiation", "barometric_pressure", "rainfall"]

    daily_records = []
    print("Aggregating daily records across 2018-2026...")

    for d, g in df.groupby("date"):
        row = {"date": str(d)}
        n_hours = len(g)

        # Pollutant 24-h means (require >= 16 valid hours)
        for p in pollutants_24h:
            valid_cnt = int(g[p].notna().sum())
            row[f"valid_hours_{p}"] = valid_cnt
            if valid_cnt >= 16:
                row[p] = float(g[p].mean())
            else:
                row[p] = np.nan

        # Particulate daily maxima
        row["pm25_max"] = float(g["pm25"].max()) if g["pm25"].notna().any() else np.nan
        row["pm10_max"] = float(g["pm10"].max()) if g["pm10"].notna().any() else np.nan

        # CO and O3 8-hour trailing rolling max
        row["co_8h_max"] = float(g["co_8h"].max()) if g["co_8h"].notna().any() else np.nan
        row["o3_8h_max"] = float(g["o3_8h"].max()) if g["o3_8h"].notna().any() else np.nan

        # Meteorological daily means (require >= 16 valid hours)
        for w in weather_vars:
            valid_cnt = int(g[w].notna().sum())
            if valid_cnt >= 16:
                row[w] = float(g[w].mean())
            else:
                row[w] = np.nan

        # Wind direction vector mean
        valid_wind = g["wind_sin"].notna() & g["wind_cos"].notna()
        if valid_wind.sum() >= 16:
            mean_sin = float(g.loc[valid_wind, "wind_sin"].mean())
            mean_cos = float(g.loc[valid_wind, "wind_cos"].mean())
            vec_deg = float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0)
            row["wind_direction"] = round(vec_deg, 2)
            row["wind_dir_sin"] = round(mean_sin, 4)
            row["wind_dir_cos"] = round(mean_cos, 4)
        else:
            row["wind_direction"] = np.nan
            row["wind_dir_sin"] = np.nan
            row["wind_dir_cos"] = np.nan

        # Station count statistics for PM2.5
        row["n_stations_pm25_min"] = int(g["n_stations_pm25"].min())
        row["n_stations_pm25_mean"] = round(float(g["n_stations_pm25"].mean()), 2)

        # 3. NAQI Sub-indices calculation
        # 24h concentrations for PM2.5, PM10, NO2, SO2, NH3; 8h max for CO and O3
        subs = {
            "pm25": sub_index("pm25", row["pm25"]),
            "pm10": sub_index("pm10", row["pm10"]),
            "no2": sub_index("no2", row["no2"]),
            "so2": sub_index("so2", row["so2"]),
            "nh3": sub_index("nh3", row["nh3"]),
            "co": sub_index("co", row["co_8h_max"]),
            "o3": sub_index("o3", row["o3_8h_max"]),
        }

        for p, s in subs.items():
            row[f"sub_index_{p}"] = round(s, 2) if (s is not None and not np.isnan(s)) else np.nan

        aqi_val, prom_pol = compute_aqi_from_subs(subs)
        row["aqi"] = round(aqi_val, 2) if (aqi_val is not None and not np.isnan(aqi_val)) else np.nan
        row["prominent_pollutant"] = prom_pol
        row["aqi_category"] = get_aqi_category(aqi_val)

        daily_records.append(row)

    daily_df = pd.DataFrame(daily_records)
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    # Save to data/processed/clean_daily.csv
    out_csv = REPO_ROOT / "data" / "processed" / "clean_daily.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(out_csv, index=False)
    print(f"Saved daily dataset to {out_csv} ({len(daily_df)} rows)")

    # Also save parquet version for fast downstream usage
    out_parquet = REPO_ROOT / "data" / "processed" / "clean_daily.parquet"
    daily_df.to_parquet(out_parquet, index=False)
    print(f"Saved daily dataset to {out_parquet}")

    # 4. Generate outputs/daily_coverage.csv (valid days per year and month)
    daily_df["year"] = daily_df["date"].dt.year
    daily_df["month"] = daily_df["date"].dt.month

    cov = daily_df.groupby(["year", "month"]).agg(
        total_days=("date", "count"),
        valid_aqi_days=("aqi", lambda s: int(s.notna().sum())),
        missing_aqi_days=("aqi", lambda s: int(s.isna().sum())),
        valid_pm25_days=("pm25", lambda s: int(s.notna().sum())),
        valid_pm10_days=("pm10", lambda s: int(s.notna().sum())),
    ).reset_index()

    cov["aqi_coverage_pct"] = round(100.0 * cov["valid_aqi_days"] / cov["total_days"], 2)

    cov_csv = REPO_ROOT / "outputs" / "daily_coverage.csv"
    cov_csv.parent.mkdir(parents=True, exist_ok=True)
    cov.to_csv(cov_csv, index=False)
    print(f"Saved daily coverage to {cov_csv}")

    # Summary prints
    print("\n" + "="*80)
    print("DAILY AQI SUMMARY REPORT")
    print("="*80)
    print(f"Total calendar days: {len(daily_df)}")
    print(f"Days with valid AQI: {daily_df['aqi'].notna().sum()} ({100.0*daily_df['aqi'].notna().mean():.2f}%)")
    print(f"Days missing AQI:    {daily_df['aqi'].isna().sum()}")
    print("\nAQI Category counts overall:")
    print(daily_df["aqi_category"].value_counts(dropna=False).to_string())

    print("\nAQI Category counts per year:")
    cat_by_year = pd.crosstab(daily_df["year"], daily_df["aqi_category"].fillna("Missing"), margins=True)
    print(cat_by_year.to_string())

    print("\nDays with AQI > 400 (Severe) per year:")
    severe_by_year = daily_df[daily_df["aqi"] > 400].groupby("year")["date"].count()
    print(severe_by_year.to_string())
    print(f"Total days with AQI > 400: {(daily_df['aqi'] > 400).sum()}")

    return daily_df


if __name__ == "__main__":
    run_daily_aggregation()
