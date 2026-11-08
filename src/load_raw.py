#!/usr/bin/env python3
"""
load_raw.py  –  Read raw CPCB air quality station-year CSVs.

Responsibilities:
  1. Inspect every station-year CSV (432 files) across all year folders (2010–2026) under data/raw/.
  2. Extract station identifier from the filename pattern: site_<id>_<station>_<agency>_1Hr.csv.
  3. Rename columns using COLUMN_MAP from src/config.py.
  4. Log the row count per file and total row counts to outputs/load_raw_log.csv.
  5. Filter and retain only observations in the window: 2018-01-01 00:00:00 to 2026-09-30 23:00:00.
  6. Save concatenated filtered data to data/processed/raw_filtered.parquet.
  7. Never modify raw files.
"""

import os
import re
import sys
from pathlib import Path
import pandas as pd

# ── Ensure project root is on sys.path ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DATA_DIR, PROCESSED_DIR, OUTPUT_DIR, COLUMN_MAP


def extract_station(filepath: Path) -> str:
    """Extract station identity from filename pattern site_<id>_<station>_<agency>_1Hr.csv."""
    stem = filepath.stem
    if stem.endswith("_1Hr"):
        return stem[:-4]
    return stem


def load_raw_data() -> pd.DataFrame:
    """
    Reads every station-year CSV under data/raw/, logs counts, and returns
    a concatenated DataFrame filtered to 2018-01-01 through 2026-09-30.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    year_folders = sorted([
        d for d in RAW_DATA_DIR.iterdir()
        if d.is_dir() and re.fullmatch(r"\d{4}", d.name)
    ])

    file_logs = []
    filtered_dfs = []
    total_raw_rows = 0

    print("=" * 70)
    print("STEP 1: Reading and logging all 432 station-year CSVs from data/raw/")
    print("=" * 70)

    for ydir in year_folders:
        yr = int(ydir.name)
        csv_files = sorted(ydir.glob("*.csv"))
        for p in csv_files:
            station_name = extract_station(p)
            try:
                # Fast row count check
                with open(p, "rb") as f:
                    # Count non-empty lines minus header
                    raw_lines = sum(1 for line in f if line.strip())
                    raw_row_count = max(0, raw_lines - 1)
            except Exception:
                raw_row_count = 0

            total_raw_rows += raw_row_count

            # If within 2018-2026, load and filter
            filtered_count = 0
            if 2018 <= yr <= 2026:
                try:
                    df = pd.read_csv(
                        p,
                        na_values=["NA", "None", "null", "nan", "NaN", "-", ""],
                        keep_default_na=True,
                        low_memory=False,
                    )
                    if not df.empty:
                        # Rename columns according to COLUMN_MAP
                        rename_dict = {c: COLUMN_MAP[c] for c in df.columns if c in COLUMN_MAP}
                        df = df.rename(columns=rename_dict)
                        df["station"] = station_name

                        # Filter by datetime window
                        if "datetime" in df.columns:
                            dt_col = pd.to_datetime(df["datetime"], errors="coerce")
                            mask = (dt_col >= "2018-01-01 00:00:00") & (dt_col <= "2026-09-30 23:00:00")
                            df = df[mask].copy()
                            filtered_count = len(df)
                            if filtered_count > 0:
                                filtered_dfs.append(df)
                except Exception as e:
                    print(f"  ⚠ Error reading {p.name}: {e}")

            file_logs.append({
                "year": yr,
                "file": p.name,
                "station": station_name,
                "raw_rows": raw_row_count,
                "filtered_rows_2018_2026": filtered_count,
            })

    log_df = pd.DataFrame(file_logs)
    log_csv_path = OUTPUT_DIR / "load_raw_log.csv"
    log_df.to_csv(log_csv_path, index=False)
    print(f"✓ Processed {len(log_df)} files.")
    print(f"✓ Total raw rows across all 432 files (2010–2026): {total_raw_rows:,}")
    print(f"✓ Saved per-file row counts to {log_csv_path}")

    print("\nConcatenating 2018–2026 station files...")
    combined_df = pd.concat(filtered_dfs, ignore_index=True)
    print(f"✓ Retained rows (2018-01-01 to 2026-09-30): {len(combined_df):,}")
    print(f"✓ Unique stations in 2018–2026: {combined_df['station'].nunique()}")

    out_parquet = PROCESSED_DIR / "raw_filtered.parquet"
    combined_df.to_parquet(out_parquet, index=False)
    print(f"✓ Saved intermediate dataset to {out_parquet}")

    return combined_df


if __name__ == "__main__":
    load_raw_data()
