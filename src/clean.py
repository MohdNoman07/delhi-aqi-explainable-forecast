#!/usr/bin/env python3
"""
clean.py  –  Clean and standardise raw CPCB air quality data.

Steps:
  1. Load files for selected years (default: 2015–2026).
  2. Map column names to semantic roles (see config.COLUMN_MAP).
  3. Derive `station` from filename.
  4. Convert text 'NA' to NaN.
  5. Drop columns with >80% missing data (VOCs, VWS, etc.).
  6. Convert CO from mg/m³ → µg/m³ (×1000) for uniform units.
  7. Handle short missing-value gaps with forward-fill (≤ 3 hours per station).
  8. Compute India NAQI sub-indices and overall AQI per row.
  9. Save cleaned dataset to data/processed/delhi_aq_clean.parquet (and .csv.gz sample).
"""

import os, re, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RAW_DATA_DIR, PROCESSED_DIR, COLUMN_MAP,
    CLEANING, DROP_COLS_HIGH_NULL, ROLE_TO_RAW
)
from compute_aqi import sub_index, compute_aqi_from_subs, AQI_POLLUTANTS

warnings.filterwarnings("ignore")

# ── helpers ───────────────────────────────────────────────────────────────────

def station_from_path(p: Path) -> str:
    return p.stem.replace("_1Hr", "")

def safe_load_csv(p: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(p, na_values=["NA", "None", "null", "nan", "NaN", "-", ""],
                         keep_default_na=True, low_memory=False)
        return df
    except Exception:
        return pd.DataFrame()


def clean_single_station_year(p: Path, yr: int) -> pd.DataFrame:
    df = safe_load_csv(p)
    if df.empty or len(df) == 0:
        return pd.DataFrame()

    # Drop high-null columns
    if CLEANING["drop_high_null_cols"]:
        to_drop = [c for c in DROP_COLS_HIGH_NULL if c in df.columns]
        df = df.drop(columns=to_drop)

    # Rename to semantic roles
    rename_dict = {col: COLUMN_MAP[col] for col in df.columns if col in COLUMN_MAP}
    df = df.rename(columns=rename_dict)

    # Add metadata
    df["station"] = station_from_path(p)
    df["year"] = yr

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    # Convert CO: keep original mg/m³ as `co_mg` and add `co_ug` (µg/m³)
    if "co" in df.columns:
        df["co_mg"] = df["co"]
        if CLEANING["co_to_ug"]:
            df["co_ug"] = df["co"] * 1000.0

    # Sort & forward-fill short gaps per station (≤ 3 hours)
    gap = CLEANING.get("ffill_max_gap_hours", 0)
    if gap > 0:
        # Only fill numeric measurement columns, not metadata
        num_cols = [c for c in df.columns if c not in ["datetime", "station", "year"]]
        df[num_cols] = df[num_cols].ffill(limit=gap)

    return df


def add_aqi_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute NAQI sub-indices and overall AQI for the dataframe."""
    print("  Calculating AQI sub-indices...")

    # Calculate sub-indices vectorized or via apply
    for p in AQI_POLLUTANTS:
        if p == "co" and "co_mg" in df.columns:
            # CO sub-index needs mg/m³
            df[f"sub_idx_{p}"] = df["co_mg"].apply(lambda v: sub_index("co", v))
        elif p in df.columns:
            df[f"sub_idx_{p}"] = df[p].apply(lambda v: sub_index(p, v))

    sub_cols = [f"sub_idx_{p}" for p in AQI_POLLUTANTS if f"sub_idx_{p}" in df.columns]

    print("  Computing overall AQI...")
    # Vectorized max across sub-indices, requiring >= 3 valid
    sub_df = df[sub_cols]
    valid_count = sub_df.notna().sum(axis=1)
    aqi_series = sub_df.max(axis=1)
    aqi_series[valid_count < 3] = np.nan

    df["aqi"] = aqi_series
    df["aqi_pollutant_count"] = valid_count

    # Category
    def aqi_category(val):
        if np.isnan(val): return np.nan
        if val <= 50: return "Good"
        if val <= 100: return "Satisfactory"
        if val <= 200: return "Moderate"
        if val <= 300: return "Poor"
        if val <= 400: return "Very Poor"
        return "Severe"

    df["aqi_category"] = df["aqi"].apply(aqi_category)
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    min_yr, max_yr = CLEANING["year_range"]

    print("=" * 60)
    print(f"CLEANING PIPELINE: Delhi AQ ({min_yr}–{max_yr})")
    print("=" * 60)

    all_dfs = []
    total_files = 0

    for yr in range(min_yr, max_yr + 1):
        folder = RAW_DATA_DIR / str(yr)
        if not folder.is_dir():
            continue
        files = sorted(folder.glob("*.csv"))
        print(f"  Loading {yr}: {len(files)} files...")
        for p in files:
            sdf = clean_single_station_year(p, yr)
            if not sdf.empty:
                all_dfs.append(sdf)
                total_files += 1

    print(f"\n  Concatenating {len(all_dfs)} station-year dataframes...")
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  Combined rows: {len(combined):,}")

    if CLEANING["compute_aqi"]:
        combined = add_aqi_columns(combined)

    # Save to parquet
    out_parquet = PROCESSED_DIR / "delhi_aq_clean.parquet"
    print(f"\n  Saving parquet to {out_parquet}...")
    combined.to_parquet(out_parquet, index=False, compression="snappy")

    # Save small CSV sample (first 50,000 rows) for easy inspection
    out_sample = PROCESSED_DIR / "sample_50k.csv"
    combined.head(50000).to_csv(out_sample, index=False)
    print(f"  Saved sample CSV to {out_sample}")

    # Summary
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print(f"  Total cleaned rows:    {len(combined):,}")
    print(f"  Unique stations:       {combined['station'].nunique()}")
    print(f"  Date range:            {combined['datetime'].min()} → {combined['datetime'].max()}")
    print(f"  Columns:               {list(combined.columns)}")
    print(f"  AQI coverage:          {100 * combined['aqi'].notna().mean():.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
