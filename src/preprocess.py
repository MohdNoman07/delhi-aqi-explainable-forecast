#!/usr/bin/env python3
"""
preprocess.py  –  Clean and preprocess hourly Delhi air quality data (2018–2026).

Exact order of operations at the STATION level, before any reindexing or interpolation:
  1. Pressure: if P < 850, convert to hPa by multiplying by 1.333224; then set
     values outside [940, 1030] hPa to NaN.
  2. Physical and sensor caps (as before: non-numeric to NaN, negative concentrations to NaN,
     humidity outside [0, 100] to NaN, instrument analyzer ceilings).
  3. Stuck sensors: set a run to NaN if the exact same non-zero value repeats for
     24 or more consecutive hours. Applied to all variables except rainfall. Done on raw
     station series before reindexing or interpolation.
  4. PM inversions: when PM2.5 > 1.2 * PM10 AND PM2.5 > 20 ug/m3, set both PM2.5 and PM10
     to NaN for that station-hour.
  5. Spike filter: applied ONLY to benzene and toluene (>10x rolling 24h median).
     NOT applied to PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, O3 or meteorology,
     preserving real pollution episodes.
  6. Station selection verification: recompute coverage after these fixes and report
     whether any kept station falls below 50%.
  7. Bounded linear interpolation per kept station: reindex to complete hourly timeline
     and interpolate linearly only inside gaps of at most 6 consecutive hours
     (limit=6, limit_area=inside). Longer gaps stay NaN.
  8. Spatial averaging across kept stations:
     - Pollutants: require >= 3 reporting stations, otherwise NaN.
     - Meteorology: require >= 2 reporting stations, otherwise NaN.
     - Wind direction: circular vector mean (arctan2(sum(sin), sum(cos))).
     - Store n_stations_<variable> columns.
  9. Save data/processed/clean_hourly.parquet and log all removal counts to outputs/cleaning_log.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Ensure project root on sys.path ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RAW_DATA_DIR, PROCESSED_DIR, OUTPUT_DIR,
    COLUMN_MAP, RANDOM_SEED
)
from load_raw import load_raw_data

TOTAL_PERIOD_HOURS = 76680  # Exactly 2018-01-01 00:00:00 to 2026-09-30 23:00:00

# ── Documented physical & sensor caps justified by distribution ────────────────
CAPS = {
    # Criteria & gas pollutants (µg/m³ except CO in mg/m³, NOx in ppb)
    "pm25": {"min": 0.0, "max": 1000.0, "desc": "CPCB analyzer ceiling (1000 µg/m³; p99.99 = 968.0)"},
    "pm10": {"min": 0.0, "max": 1000.0, "desc": "CPCB analyzer ceiling (1000 µg/m³; p99.99 = 1000.0)"},
    "no":   {"min": 0.0, "max": 500.0,  "desc": "CPCB chemiluminescence ceiling (500 µg/m³; p99.99 = 494.3)"},
    "no2":  {"min": 0.0, "max": 500.0,  "desc": "CPCB chemiluminescence ceiling (500 µg/m³; p99.99 = 450.2)"},
    "nox":  {"min": 0.0, "max": 500.0,  "desc": "CPCB analyzer ceiling (500 ppb; p99.99 = 495.8)"},
    "nh3":  {"min": 0.0, "max": 500.0,  "desc": "CPCB analyzer ceiling (500 µg/m³; p99.99 = 411.4)"},
    "so2":  {"min": 0.0, "max": 200.0,  "desc": "CPCB UV fluorescence ceiling (200 µg/m³; p99.99 = 171.5)"},
    "co":   {"min": 0.0, "max": 50.0,   "desc": "NAQI severe breakpoint & NDIR ceiling (50 mg/m³; p99.99 = 23.45)"},
    "o3":   {"min": 0.0, "max": 500.0,  "desc": "Extreme tropospheric ozone cap (500 µg/m³; p99.99 = 325.56, max spike 970.0)"},
    "benzene": {"min": 0.0, "max": 200.0, "desc": "Extreme VOC spike cap (200 µg/m³; p99.99 = 168.26, max spike 476.4)"},
    "toluene": {"min": 0.0, "max": 500.0, "desc": "VOC analyzer ceiling (500 µg/m³; p99.99 = 490.46)"},

    # Meteorology
    "temperature": {"min": 0.0, "max": 52.0, "desc": "Delhi historical bounds (lowest ~0°C, record heat ~49.2°C; p99.99 = 48.65, max spike 58.9)"},
    "humidity":    {"min": 0.0, "max": 100.0, "desc": "Physical relative humidity limits [0, 100%]"},
    "wind_speed":  {"min": 0.0, "max": 30.0, "desc": "Anemometer physical bounds (30 m/s = 108 km/h; p99.99 = 18.08, max spike 47.75)"},
    "wind_direction": {"min": 0.0, "max": 360.0, "desc": "Compass degree circle [0, 360°]"},
    "rainfall":    {"min": 0.0, "max": 150.0, "desc": "Hourly rainfall physical limit (150 mm/h cloudburst ceiling; max in data = 50.0)"},
    "solar_radiation": {"min": 0.0, "max": 1300.0, "desc": "Clear-sky surface solar radiation limit (1300 W/m²; TOA solar const 1361; max spike 1985.0)"},
    "barometric_pressure": {"min": 940.0, "max": 1030.0, "desc": "Standard ambient atmospheric pressure at Delhi elevation ~216m in hPa [940, 1030]"},
}

POLLUTANT_VARS = ["pm25", "pm10", "no", "no2", "nox", "nh3", "so2", "co", "o3", "benzene", "toluene"]
MET_VARS = ["temperature", "humidity", "wind_speed", "wind_direction", "rainfall", "solar_radiation", "barometric_pressure"]


def invalidate_stuck_runs(series: pd.Series, min_run: int = 24) -> tuple:
    """
    Find runs of constant non-zero values >= min_run hours and set them to NaN.
    Returns (cleaned_series, count_invalidated).
    """
    valid = series.notna() & (series != 0)
    if not valid.any():
        return series, 0

    change = (series != series.shift(1)) | (~valid) | (~valid.shift(1, fill_value=False))
    group_id = change.cumsum()
    group_sizes = series.groupby(group_id).transform("count")
    stuck = valid & (group_sizes >= min_run)
    stuck_cnt = int(stuck.sum())
    if stuck_cnt > 0:
        return series.mask(stuck), stuck_cnt
    return series, 0


def interpolate_bounded_gaps(series: pd.Series, max_gap: int = 6) -> pd.Series:
    """
    Interpolate linearly ONLY inside gaps of at most `max_gap` consecutive hours.
    Longer gaps and leading/trailing boundary NaNs remain NaN.
    """
    is_na = series.isna()
    if not is_na.any() or is_na.all():
        return series

    blocks = (~is_na).cumsum()
    gap_sizes = is_na.groupby(blocks).transform("sum")

    interp = series.interpolate(method="linear", limit_area="inside")
    return series.where(~is_na, interp.where(gap_sizes <= max_gap))


def circular_mean_degrees(angles_deg: np.ndarray) -> float:
    """Compute circular vector mean of angles in degrees."""
    valid = angles_deg[~np.isnan(angles_deg)]
    if len(valid) == 0:
        return np.nan
    rads = np.radians(valid)
    sin_sum = np.sum(np.sin(rads))
    cos_sum = np.sum(np.cos(rads))
    if np.isclose(sin_sum, 0.0) and np.isclose(cos_sum, 0.0):
        return np.nan
    mean_deg = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360.0
    return mean_deg


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DELHI AIR QUALITY PREPROCESSING PIPELINE (2018-01-01 to 2026-09-30)")
    print("=" * 80)

    # ── Load raw concatenated data ────────────────────────────────────────────
    raw_path = PROCESSED_DIR / "raw_filtered.parquet"
    if raw_path.exists():
        print(f"Loading cached filtered data from {raw_path}...")
        df = pd.read_parquet(raw_path)
    else:
        df = load_raw_data()

    initial_row_count = len(df)
    print(f"\nInitial row count: {initial_row_count:,}")

    # Parse datetime & sort by station and time
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["station", "datetime"]).reset_index(drop=True)

    # Drop exact duplicate rows within same station & timestamp
    dup_mask = df.duplicated(subset=["station", "datetime"], keep="first")
    dup_count = int(dup_mask.sum())
    df = df[~dup_mask].reset_index(drop=True)
    print(f"  Exact duplicates removed (same station & timestamp): {dup_count:,}")
    print(f"  Rows remaining: {len(df):,}")

    cleaning_logs = []
    meas_cols = [c for c in df.columns if c not in ["datetime", "station"]]

    # Convert non-numeric text placeholders to NaN
    for col in meas_cols:
        s_num = pd.to_numeric(df[col], errors="coerce")
        text_nan_cnt = int(df[col].notna().sum() - s_num.notna().sum())
        if text_nan_cnt > 0:
            cleaning_logs.append({
                "column": col,
                "rule": "non_numeric_placeholder",
                "threshold": "coerce_to_numeric",
                "count_set_to_nan": text_nan_cnt,
                "description": "Non-numeric placeholder string converted to NaN",
            })
        df[col] = s_num

    # ── Drop columns >80% null and TOT-RF ───────────────────────────────────
    null_pcts = df[meas_cols].isna().mean()
    high_null_cols = null_pcts[null_pcts > 0.80].index.tolist()
    print("\nColumns >80% null over 2018–2026:")
    for col in high_null_cols:
        print(f"  - {col}: {null_pcts[col]*100:.1f}% null -> DROPPED")

    print("Checking TOT-RF (total_rainfall):")
    print("  - Raw TOT-RF exhibits sensor saturation jumps up to 188,400 mm and 8191.5 mm.")
    print("  - Dropping total_rainfall (TOT-RF); retaining rainfall (RF) as the hourly measure.")

    cols_to_drop = list(set(high_null_cols + ["total_rainfall"]))
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # ── STATION-LEVEL ORDER OF OPERATIONS (Steps 1 to 5) ──────────────────────
    print("\n" + "=" * 80)
    print("APPLYING STATION-LEVEL FIXES (Steps 1 to 5 in exact order)")
    print("=" * 80)

    # Load previously kept stations (35 stations)
    st_sel_prev = pd.read_csv(OUTPUT_DIR / "station_selection.csv")
    kept_stations = st_sel_prev[st_sel_prev["kept"]]["station"].tolist()
    print(f"Kept stations list: {len(kept_stations)} stations")

    total_pressure_converted = 0
    total_pressure_clipped = 0
    total_caps_applied = {c: 0 for c in df.columns if c in CAPS}
    total_stuck_invalidated = {c: 0 for c in POLLUTANT_VARS + MET_VARS if c in df.columns}
    total_pm_inversions = 0
    total_voc_spikes = {"benzene": 0, "toluene": 0}

    station_dfs_processed = []

    for st, st_df in df.groupby("station"):
        st_sub = st_df.sort_values("datetime").copy()

        # ── 1. Pressure standardization ───────────────────────────────────────
        if "barometric_pressure" in st_sub.columns:
            # If P < 850, convert from mmHg to hPa (* 1.333224)
            p_mask = (st_sub["barometric_pressure"] > 0) & (st_sub["barometric_pressure"] < 850.0)
            p_conv_cnt = int(p_mask.sum())
            if p_conv_cnt > 0:
                st_sub.loc[p_mask, "barometric_pressure"] = st_sub.loc[p_mask, "barometric_pressure"] * 1.333224
                total_pressure_converted += p_conv_cnt

            # Set values outside [940, 1030] hPa to NaN
            p_clip_mask = (st_sub["barometric_pressure"] < 940.0) | (st_sub["barometric_pressure"] > 1030.0)
            p_clip_cnt = int(p_clip_mask.sum())
            if p_clip_cnt > 0:
                st_sub.loc[p_clip_mask, "barometric_pressure"] = np.nan
                total_pressure_clipped += p_clip_cnt

        # ── 2. Physical & Sensor Caps ─────────────────────────────────────────
        for col in st_sub.columns:
            if col in CAPS and col != "barometric_pressure":  # pressure handled above
                cfg = CAPS[col]
                invalid_mask = (st_sub[col] < cfg["min"]) | (st_sub[col] > cfg["max"])
                inv_cnt = int(invalid_mask.sum())
                if inv_cnt > 0:
                    st_sub.loc[invalid_mask, col] = np.nan
                    total_caps_applied[col] += inv_cnt

        # ── 3. Stuck Sensors (>= 24 consecutive hours, all vars EXCEPT rainfall) ─
        for col in total_stuck_invalidated.keys():
            if col in st_sub.columns and col != "rainfall":
                cleaned_col, stuck_cnt = invalidate_stuck_runs(st_sub[col], min_run=24)
                st_sub[col] = cleaned_col
                total_stuck_invalidated[col] += stuck_cnt

        # ── 4. PM Inversions: PM2.5 > 1.2 * PM10 AND PM2.5 > 20 ug/m3 ──────────
        if "pm25" in st_sub.columns and "pm10" in st_sub.columns:
            inv_mask = (
                st_sub["pm25"].notna() &
                st_sub["pm10"].notna() &
                (st_sub["pm10"] > 0) &
                (st_sub["pm25"] > 1.2 * st_sub["pm10"]) &
                (st_sub["pm25"] > 20.0)
            )
            inv_cnt = int(inv_mask.sum())
            if inv_cnt > 0:
                st_sub.loc[inv_mask, ["pm25", "pm10"]] = np.nan
                total_pm_inversions += inv_cnt

        # ── 5. Spike Filter: ONLY Benzene and Toluene (>10x rolling 24h median) ─
        # NO spike filter on PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, O3 or meteorology!
        for voc in ["benzene", "toluene"]:
            if voc in st_sub.columns:
                diff = st_sub[voc].diff().abs()
                roll_med = st_sub[voc].rolling(24, min_periods=6).median()
                spike_mask = (diff > 10 * roll_med) & (roll_med > 0.1) & diff.notna() & roll_med.notna()
                spk_cnt = int(spike_mask.sum())
                if spk_cnt > 0:
                    st_sub.loc[spike_mask, voc] = np.nan
                    total_voc_spikes[voc] += spk_cnt

        station_dfs_processed.append(st_sub)

    df_cleaned_station = pd.concat(station_dfs_processed, ignore_index=True)

    # ── Log cleaning counts ───────────────────────────────────────────────────
    cleaning_logs.append({
        "column": "barometric_pressure",
        "rule": "pressure_unit_conversion_mmhg_to_hpa",
        "threshold": "P < 850.0 mmHg",
        "count_set_to_nan": 0,
        "description": f"Converted {total_pressure_converted:,} readings from mmHg to hPa (* 1.333224)",
    })
    cleaning_logs.append({
        "column": "barometric_pressure",
        "rule": "pressure_physical_bounds_hpa",
        "threshold": "Outside [940, 1030] hPa",
        "count_set_to_nan": total_pressure_clipped,
        "description": f"Set {total_pressure_clipped:,} out-of-range pressure values to NaN",
    })
    print(f"Step 1: Converted {total_pressure_converted:,} pressure readings from mmHg to hPa; clipped {total_pressure_clipped:,} outside [940, 1030] hPa.")

    for col, cnt in total_caps_applied.items():
        if cnt > 0:
            cleaning_logs.append({
                "column": col,
                "rule": "physical_and_sensor_cap",
                "threshold": f"[{CAPS[col]['min']}, {CAPS[col]['max']}]",
                "count_set_to_nan": cnt,
                "description": CAPS[col]["desc"],
            })
    print(f"Step 2: Applied physical & analyzer caps ({sum(total_caps_applied.values()):,} values set to NaN).")

    for col, cnt in total_stuck_invalidated.items():
        if cnt > 0:
            cleaning_logs.append({
                "column": col,
                "rule": "stuck_sensor_invalidation",
                "threshold": ">= 24h constant non-zero (excl. rainfall)",
                "count_set_to_nan": cnt,
                "description": f"Invalidated {cnt:,} stuck sensor readings",
            })
    print(f"Step 3: Invalidated {sum(total_stuck_invalidated.values()):,} stuck sensor readings (>=24h, excl. rainfall).")

    cleaning_logs.append({
        "column": "pm25_and_pm10",
        "rule": "pm_inversion_filter",
        "threshold": "PM2.5 > 1.2 * PM10 AND PM2.5 > 20 ug/m3",
        "count_set_to_nan": total_pm_inversions * 2,
        "description": f"Invalidated {total_pm_inversions:,} station-hours with unphysical PM inversion",
    })
    print(f"Step 4: Invalidated {total_pm_inversions:,} station-hours with PM2.5 > 1.2 * PM10 and PM2.5 > 20 µg/m³.")

    for voc, cnt in total_voc_spikes.items():
        if cnt > 0:
            cleaning_logs.append({
                "column": voc,
                "rule": "voc_rate_of_change_spike",
                "threshold": "> 10x rolling 24h median (baseline > 0.1)",
                "count_set_to_nan": cnt,
                "description": f"Filtered {cnt:,} transient isolated VOC spikes",
            })
    print(f"Step 5: Filtered {sum(total_voc_spikes.values()):,} isolated VOC spikes on benzene/toluene (NO spike filter on criteria pollutants or meteorology).")

    cleaning_log_df = pd.DataFrame(cleaning_logs)
    cleaning_log_df.to_csv(OUTPUT_DIR / "cleaning_log.csv", index=False)
    print(f"Updated cleaning log written to {OUTPUT_DIR / 'cleaning_log.csv'}")

    # ── Station Selection Coverage Recheck ────────────────────────────────────
    print("\n" + "=" * 80)
    print("STATION SELECTION COVERAGE RECHECK AFTER ALL FIXES")
    print("=" * 80)
    station_cov_after = []
    for st in sorted(df_cleaned_station["station"].unique()):
        st_sub = df_cleaned_station[df_cleaned_station["station"] == st]
        pm25_valid = int(st_sub["pm25"].notna().sum()) if "pm25" in st_sub.columns else 0
        pm10_valid = int(st_sub["pm10"].notna().sum()) if "pm10" in st_sub.columns else 0
        either_valid = int(((st_sub["pm25"].notna()) | (st_sub["pm10"].notna())).sum())
        cov = either_valid / TOTAL_PERIOD_HOURS
        was_kept = st in kept_stations
        is_above_50 = cov >= 0.50

        station_cov_after.append({
            "station": st,
            "valid_pm_hours_after": either_valid,
            "coverage_after": round(cov, 4),
            "was_kept": was_kept,
            "above_50_after": is_above_50,
        })

    cov_after_df = pd.DataFrame(station_cov_after)
    dropped_now = cov_after_df[cov_after_df["was_kept"] & (~cov_after_df["above_50_after"])]
    print(f"Kept stations evaluated: {len(kept_stations)}")
    if len(dropped_now) == 0:
        print("✓ All 35 kept stations REMAIN above 50% coverage after all fixes!")
        print(f"  Lowest coverage among kept stations: {cov_after_df[cov_after_df['was_kept']]['coverage_after'].min()*100:.2f}% ({cov_after_df[cov_after_df['was_kept']].sort_values('coverage_after').iloc[0]['station']})")
    else:
        print(f"⚠ Kept stations that dropped below 50%: {dropped_now['station'].tolist()}")

    # Filter to kept stations for final aggregation
    df_kept = df_cleaned_station[df_cleaned_station["station"].isin(kept_stations)].copy()

    # ── Step 6: Timeline Reindexing & 6-Hour Bounded Interpolation ─────────────
    print("\n" + "=" * 80)
    print("STEP 6: TIMELINE REINDEXING & 6-HOUR BOUNDED INTERPOLATION")
    print("=" * 80)
    full_hourly_idx = pd.date_range("2018-01-01 00:00:00", "2026-09-30 23:00:00", freq="1h", name="datetime")
    station_dfs = []
    interpolated_counts = {c: 0 for c in POLLUTANT_VARS + MET_VARS if c in df_kept.columns}

    for st in kept_stations:
        st_sub = df_kept[df_kept["station"] == st].set_index("datetime")
        st_reindexed = st_sub.reindex(full_hourly_idx)
        st_reindexed["station"] = st

        for col in interpolated_counts.keys():
            if col in st_reindexed.columns:
                before_na = int(st_reindexed[col].isna().sum())
                st_reindexed[col] = interpolate_bounded_gaps(st_reindexed[col], max_gap=6)
                after_na = int(st_reindexed[col].isna().sum())
                interpolated_counts[col] += (before_na - after_na)

        station_dfs.append(st_reindexed.reset_index())

    df_interpolated = pd.concat(station_dfs, ignore_index=True)
    print(f"Total values interpolated across all 35 stations (limit=6, limit_area=inside):")
    for col, cnt in interpolated_counts.items():
        print(f"  - {col:20s}: {cnt:,} values filled")

    # ── Step 7: Spatial Hourly Averaging Across Kept Stations ──────────────────
    print("\n" + "=" * 80)
    print("STEP 7: SPATIAL HOURLY AVERAGING ACROSS KEPT STATIONS")
    print("=" * 80)
    var_cols = [c for c in POLLUTANT_VARS + MET_VARS if c in df_interpolated.columns]
    grouped = df_interpolated.groupby("datetime")

    agg_dict = {}
    scalar_vars = [c for c in var_cols if c != "wind_direction"]
    for c in scalar_vars:
        agg_dict[f"{c}_mean"] = grouped[c].mean()
        agg_dict[f"n_stations_{c}"] = grouped[c].count()

    # Wind direction: circular vector mean
    if "wind_direction" in var_cols:
        wd_rad = np.radians(df_interpolated["wind_direction"])
        df_interpolated["_wd_sin"] = np.sin(wd_rad)
        df_interpolated["_wd_cos"] = np.cos(wd_rad)

        sin_sum = df_interpolated.groupby("datetime")["_wd_sin"].sum()
        cos_sum = df_interpolated.groupby("datetime")["_wd_cos"].sum()
        wd_count = df_interpolated.groupby("datetime")["wind_direction"].count()

        wd_mean_deg = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360.0
        wd_mean_deg[wd_count == 0] = np.nan
        wd_mean_deg[(np.isclose(sin_sum, 0.0)) & (np.isclose(cos_sum, 0.0))] = np.nan

        agg_dict["wind_direction_mean"] = wd_mean_deg
        agg_dict["n_stations_wind_direction"] = wd_count

    clean_hourly = pd.DataFrame(agg_dict, index=full_hourly_idx).reset_index()
    clean_hourly = clean_hourly.rename(columns={"index": "datetime"})

    # Apply reporting thresholds
    for p in POLLUTANT_VARS:
        if f"{p}_mean" in clean_hourly.columns:
            mask_insufficient = clean_hourly[f"n_stations_{p}"] < 3
            clean_hourly.loc[mask_insufficient, f"{p}_mean"] = np.nan
            clean_hourly = clean_hourly.rename(columns={f"{p}_mean": p})

    for m in MET_VARS:
        if f"{m}_mean" in clean_hourly.columns:
            mask_insufficient = clean_hourly[f"n_stations_{m}"] < 2
            clean_hourly.loc[mask_insufficient, f"{m}_mean"] = np.nan
            clean_hourly = clean_hourly.rename(columns={f"{m}_mean": m})

    n_station_cols = sorted([c for c in clean_hourly.columns if c.startswith("n_stations_")])
    main_cols = [c for c in POLLUTANT_VARS + MET_VARS if c in clean_hourly.columns]
    final_cols = ["datetime"] + main_cols + n_station_cols
    clean_hourly = clean_hourly[final_cols]

    # Save regenerated parquet
    out_clean_path = PROCESSED_DIR / "clean_hourly.parquet"
    clean_hourly.to_parquet(out_clean_path, index=False)
    print(f"✓ Saved regenerated clean hourly dataset to {out_clean_path}")

    out_clean_sample = PROCESSED_DIR / "clean_hourly_sample.csv"
    clean_hourly.head(1000).to_csv(out_clean_sample, index=False)
    print(f"✓ Saved sample preview to {out_clean_sample}")

    print("\n" + "=" * 80)
    print("HOURLY DATA CLEANING COMPLETE (ALL APPROVED RULES APPLIED)")
    print("=" * 80)
    print(f"Clean hourly timeline rows: {len(clean_hourly):,} (continuous 1h steps)")
    print(f"Date range:                {clean_hourly['datetime'].min()} to {clean_hourly['datetime'].max()}")
    print("=" * 80)

    return clean_hourly


if __name__ == "__main__":
    main()
