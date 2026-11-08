"""
src/features.py

Constructs feature matrix and forward multi-horizon targets for Delhi air quality forecasting.
Input: data/processed/clean_daily.csv
Outputs:
  - data/processed/model_ready.csv
  - models/feature_columns.json

Follows Project Rules:
  - Strict chronological order on a complete daily calendar.
  - Targets are forward shifted by -h days (never used as features).
  - Features use only information available up to the end of today (t).
  - All rolling windows are trailing only (no center=True, no negative shifts in features).
  - 3-day rolling: min_periods=3. 7-day rolling: min_periods=5.
  - Drops rows with any NaN in features or targets (no imputation).
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import RANDOM_SEED


def compute_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all feature columns from a daily calendar dataframe.
    Assumes df_daily has a complete daily datetime index or 'date' column.
    Uses strictly information up to date t (no future information).
    """
    df = df_daily.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    
    # Ensure sorted by date
    df = df.sort_index()

    feat = pd.DataFrame(index=df.index)

    # ── 1. AQI features ────────────────────────────────────────────────────────
    feat["aqi"] = df["aqi"]
    feat["aqi_lag1"] = df["aqi"].shift(1)
    feat["aqi_lag2"] = df["aqi"].shift(2)
    feat["aqi_lag3"] = df["aqi"].shift(3)
    feat["aqi_lag7"] = df["aqi"].shift(7)
    feat["aqi_rolling_mean_3d"] = df["aqi"].rolling(window=3, min_periods=3).mean()
    feat["aqi_rolling_mean_7d"] = df["aqi"].rolling(window=7, min_periods=5).mean()
    feat["aqi_rolling_max_7d"] = df["aqi"].rolling(window=7, min_periods=5).max()
    feat["aqi_trend_3d"] = df["aqi"] - df["aqi"].shift(3)

    # ── 2. Pollutant features ──────────────────────────────────────────────────
    # PM2.5, PM10, NO2, SO2, CO, O3 today, 3-day rolling means of each, daily max of PM2.5 and PM10
    pollutants = ["pm25", "pm10", "no2", "so2", "co", "o3"]
    for p in pollutants:
        feat[p] = df[p]
        feat[f"{p}_rolling_mean_3d"] = df[p].rolling(window=3, min_periods=3).mean()

    feat["pm25_max"] = df["pm25_max"]
    feat["pm10_max"] = df["pm10_max"]

    # ── 3. Weather features ────────────────────────────────────────────────────
    # temperature, humidity, wind speed, barometric pressure (hPa), wind_dir_sin, wind_dir_cos today,
    # and 3-day rolling means of temperature, humidity and wind speed
    weather_vars = ["temperature", "humidity", "wind_speed", "barometric_pressure", "wind_dir_sin", "wind_dir_cos"]
    for w in weather_vars:
        feat[w] = df[w]

    for w in ["temperature", "humidity", "wind_speed"]:
        feat[f"{w}_rolling_mean_3d"] = df[w].rolling(window=3, min_periods=3).mean()

    # ── 4. Calendar features ───────────────────────────────────────────────────
    dates = feat.index
    feat["month"] = dates.month
    doy = dates.day_of_year
    feat["day_of_year_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    feat["day_of_year_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
    feat["day_of_week"] = dates.day_of_week
    feat["is_weekend"] = (dates.day_of_week >= 5).astype(int)
    # stubble-season flag for Oct 15 to Nov 15 (date only)
    feat["stubble_season"] = (
        ((dates.month == 10) & (dates.day >= 15)) |
        ((dates.month == 11) & (dates.day <= 15))
    ).astype(int)

    return feat


def build_feature_row_for_date(df_daily: pd.DataFrame, target_date: str | pd.Timestamp) -> pd.DataFrame:
    """
    Builds the exact 38-feature row for a specified date from clean_daily history.
    Uses strictly historical trailing observations (t' <= target_date).
    Raises ValueError if date is missing or lacks sufficient history (requires >= 7 trailing days).
    """
    df = df_daily.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])

    t_dt = pd.to_datetime(target_date)
    min_date = df["date"].min()
    max_date = df["date"].max()

    if t_dt > max_date:
        raise ValueError(f"Date {t_dt.strftime('%Y-%m-%d')} is beyond available daily data (max: {max_date.strftime('%Y-%m-%d')}).")
    if t_dt < min_date + pd.Timedelta(7, unit="D"):
        raise ValueError(f"Insufficient history: {t_dt.strftime('%Y-%m-%d')} requires at least 7 days of trailing history.")
    if t_dt not in set(df["date"]):
        raise ValueError(f"Date {t_dt.strftime('%Y-%m-%d')} is missing from daily observations.")

    # Reindex up to t_dt on a complete daily calendar
    full_calendar = pd.date_range(min_date, t_dt, freq="D")
    daily_full = df.set_index("date").reindex(full_calendar)
    daily_full.index.name = "date"

    # Compute features on history
    all_feats = compute_features(daily_full)
    feat_row = all_feats.loc[[t_dt]]

    # Check for NaNs
    if feat_row.isna().any(axis=1).iloc[0]:
        nan_cols = feat_row.columns[feat_row.isna().iloc[0]].tolist()
        raise ValueError(f"Features for {t_dt.strftime('%Y-%m-%d')} contain NaNs in columns: {nan_cols}")

    return feat_row


def run_feature_engineering():
    daily_path = REPO_ROOT / "data" / "processed" / "clean_daily.csv"
    if not daily_path.exists():
        raise FileNotFoundError(f"Clean daily data not found at {daily_path}")

    print(f"Loading {daily_path}...")
    raw_daily = pd.read_csv(daily_path)
    raw_daily["date"] = pd.to_datetime(raw_daily["date"])

    # Reindex to complete daily calendar
    full_idx = pd.date_range("2018-01-01", "2026-09-30", freq="D")
    daily_full = raw_daily.set_index("date").reindex(full_idx)
    daily_full.index.name = "date"

    print(f"Complete calendar range: {daily_full.index.min().date()} to {daily_full.index.max().date()} ({len(daily_full)} days)")

    # 1. Compute Targets: target_h1 ... target_h5 = AQI shifted by -h days (never used as features)
    targets = pd.DataFrame(index=daily_full.index)
    for h in range(1, 6):
        targets[f"target_h{h}"] = daily_full["aqi"].shift(-h)

    target_cols = [f"target_h{h}" for h in range(1, 6)]

    # 2. Compute Features
    features = compute_features(daily_full)
    feature_cols = features.columns.tolist()

    print("\n" + "="*80)
    print(f"FINAL LIST OF FEATURE NAMES ({len(feature_cols)} features):")
    print("="*80)
    for idx, f_name in enumerate(feature_cols, 1):
        print(f"  {idx:2d}. {f_name}")

    # Combine into single dataframe
    combined = pd.concat([features, targets], axis=1).reset_index()

    # 3. Analyze rows lost and drop rows with NaN in any feature or target
    cols_to_check = feature_cols + target_cols
    nan_mask = combined[cols_to_check].isna().any(axis=1)
    nan_rows = combined[nan_mask].copy()

    # Breakdown reasons
    lag_window = nan_rows[nan_rows["date"] <= "2018-01-07"]
    target_window = nan_rows[nan_rows["date"] >= "2026-09-26"]
    mid_missing = nan_rows[(nan_rows["date"] > "2018-01-07") & (nan_rows["date"] < "2026-09-26")]

    print("\n" + "="*80)
    print("ROW REMOVAL AUDIT (NO IMPUTATION):")
    print("="*80)
    print(f"Total calendar days:       {len(combined)}")
    print(f"Rows dropped due to NaNs:   {len(nan_rows)} ({len(nan_rows)/len(combined)*100:.2f}%)")
    print(f"Final model-ready rows:    {len(combined) - len(nan_rows)} ({(len(combined)-len(nan_rows))/len(combined)*100:.2f}%)")
    print("\nSplit by reason:")
    print(f"  1. Lag window (start of series): {len(lag_window)} rows ({lag_window['date'].min().date()} to {lag_window['date'].max().date()})")
    print(f"     Reason: Requires 7 trailing days of AQI history (aqi_lag7, aqi_rolling_mean_7d).")
    print(f"  2. Target window (end of series): {len(target_window)} rows ({target_window['date'].min().date()} to {target_window['date'].max().date()})")
    print(f"     Reason: Requires 5 forward days of AQI target (target_h5).")
    print(f"  3. Mid-series missing data ripple: {len(mid_missing)} rows ({mid_missing['date'].min().date()} to {mid_missing['date'].max().date()})")
    print(f"     Reason: Missing criteria pollutant/AQI days and their forward-lag / backward-target ripples.")

    clean_df = combined[~nan_mask].reset_index(drop=True)
    clean_df["date"] = clean_df["date"].dt.strftime("%Y-%m-%d")

    # 4. Save outputs
    out_csv = REPO_ROOT / "data" / "processed" / "model_ready.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(out_csv, index=False)
    print(f"\nSaved model-ready dataset to {out_csv} ({len(clean_df)} rows)")

    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    feat_json = models_dir / "feature_columns.json"
    with open(feat_json, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Saved feature columns list to {feat_json}")

    # Yearly distribution
    clean_df["year"] = pd.to_datetime(clean_df["date"]).dt.year
    print("\n" + "="*80)
    print("MODEL-READY ROWS PER YEAR:")
    print("="*80)
    print(clean_df.groupby("year")["date"].count().to_string())

    return clean_df, feature_cols


if __name__ == "__main__":
    run_feature_engineering()
