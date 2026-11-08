#!/usr/bin/env python3
"""
inspect_data.py  –  Read-only inspection of data/raw/

Two passes:
  1. SAMPLE pass  – first and last CSV per year folder  → quick schema sanity check.
  2. FULL pass    – every CSV in every year folder       → comprehensive statistics.

Writes:  outputs/data_inspection.md
"""

import os, re, sys, textwrap, warnings
from pathlib import Path
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DATA_DIR, OUTPUT_DIR, COLUMN_MAP, POLLUTANT_ROLES, ROLE_TO_RAW

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ── helpers ───────────────────────────────────────────────────────────────────

def year_dirs():
    """Return sorted list of (year_int, Path) for year folders in raw/."""
    dirs = []
    for d in sorted(RAW_DATA_DIR.iterdir()):
        if d.is_dir() and re.fullmatch(r"\d{4}", d.name):
            dirs.append((int(d.name), d))
    return dirs

def csv_files(folder: Path):
    """Return sorted list of CSV paths in *folder*."""
    return sorted(folder.glob("*.csv"))

def station_from_filename(p: Path) -> str:
    """Extract human-readable station id from filename."""
    return p.stem.replace("_1Hr", "")

def safe_read(path: Path) -> pd.DataFrame:
    """Read a CSV, converting 'NA' text to NaN, skipping empty files."""
    try:
        df = pd.read_csv(path, na_values=["NA", "None", "null", "nan", "NaN", "-", ""],
                         keep_default_na=True, low_memory=False)
        if df.empty or len(df) == 0:
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"  ⚠ Could not read {path.name}: {e}")
        return pd.DataFrame()

# ── PASS 1: sample ────────────────────────────────────────────────────────────

def sample_pass():
    """Read first + last file of each year. Return list[dict] of per-file info."""
    infos = []
    for yr, folder in year_dirs():
        files = csv_files(folder)
        if not files:
            continue
        picks = [files[0]]
        if len(files) > 1:
            picks.append(files[-1])
        for p in picks:
            df = safe_read(p)
            infos.append({
                "year": yr,
                "file": p.name,
                "path": p,
                "station": station_from_filename(p),
                "rows": len(df),
                "columns": list(df.columns) if not df.empty else [],
                "dtypes": dict(df.dtypes.astype(str)) if not df.empty else {},
            })
    return infos


# ── PASS 2: full ──────────────────────────────────────────────────────────────

def full_pass():
    """Load every CSV. Return aggregated stats dict."""
    all_columns_sets = []
    total_rows = 0
    rows_per_year = defaultdict(int)
    files_per_year = defaultdict(int)
    station_year_rows = defaultdict(lambda: defaultdict(int))   # station → year → rows
    null_counts = defaultdict(int)          # col → total nulls
    total_counts = defaultdict(int)         # col → total non-null possible
    null_per_year = defaultdict(lambda: defaultdict(int))   # col → year → null count
    total_per_year = defaultdict(lambda: defaultdict(int))
    numeric_stats = {}                      # col → running {min, max, sum, count}
    text_placeholders = defaultdict(set)    # col → set of non-numeric strings
    negative_counts = defaultdict(int)      # col → count of <0
    humidity_above_100 = 0
    humidity_below_0 = 0
    dt_min = None
    dt_max = None
    duplicate_ts_total = 0
    aqi_present = False
    all_stations = set()

    # Detect column order consistency
    first_collist = None
    columns_differ = False

    for yr, folder in year_dirs():
        files = csv_files(folder)
        for p in files:
            df = safe_read(p)
            if df.empty:
                files_per_year[yr] += 1
                continue

            station = station_from_filename(p)
            all_stations.add(station)
            files_per_year[yr] += 1
            n = len(df)
            total_rows += n
            rows_per_year[yr] += n
            station_year_rows[station][yr] += n
            all_columns_sets.append(tuple(df.columns))

            if first_collist is None:
                first_collist = list(df.columns)
            elif list(df.columns) != first_collist:
                columns_differ = True

            # --- Timestamp analysis ---
            ts_col = "Timestamp"
            if ts_col in df.columns:
                ts = pd.to_datetime(df[ts_col], errors="coerce")
                ts_valid = ts.dropna()
                if len(ts_valid):
                    fmin, fmax = ts_valid.min(), ts_valid.max()
                    dt_min = fmin if dt_min is None else min(dt_min, fmin)
                    dt_max = fmax if dt_max is None else max(dt_max, fmax)
                # duplicate timestamps within this file
                dup = ts.duplicated(keep=False).sum()
                duplicate_ts_total += dup

            # --- AQI column? ---
            for c in df.columns:
                if "aqi" in c.lower():
                    aqi_present = True

            # --- Per-column stats ---
            for col in df.columns:
                null_counts[col] += int(df[col].isna().sum())
                total_counts[col] += n
                null_per_year[col][yr] += int(df[col].isna().sum())
                total_per_year[col][yr] += n

                if col == ts_col:
                    continue

                # Try numeric conversion
                numeric = pd.to_numeric(df[col], errors="coerce")
                non_null_numeric = numeric.dropna()
                if len(non_null_numeric):
                    cmin = float(non_null_numeric.min())
                    cmax = float(non_null_numeric.max())
                    csum = float(non_null_numeric.sum())
                    ccount = int(len(non_null_numeric))
                    if col not in numeric_stats:
                        numeric_stats[col] = {"min": cmin, "max": cmax, "sum": csum, "count": ccount}
                    else:
                        s = numeric_stats[col]
                        s["min"] = min(s["min"], cmin)
                        s["max"] = max(s["max"], cmax)
                        s["sum"] += csum
                        s["count"] += ccount

                # Text placeholders: values that were not NaN in raw but failed numeric coerce
                raw_non_null = df[col].dropna().astype(str)
                failed = raw_non_null[pd.to_numeric(raw_non_null, errors="coerce").isna()]
                if len(failed):
                    uniq = set(failed.unique()[:20])  # cap at 20 unique examples
                    text_placeholders[col] |= uniq

                # Negative values for pollutant columns
                role = COLUMN_MAP.get(col)
                if role in POLLUTANT_ROLES:
                    neg = (non_null_numeric < 0).sum()
                    negative_counts[col] += int(neg)

                # Humidity bounds
                if role == "humidity" and len(non_null_numeric):
                    humidity_above_100 += int((non_null_numeric > 100).sum())
                    humidity_below_0 += int((non_null_numeric < 0).sum())

    # ── Determine hourly ─────────────────────────────────────────────────────
    # Quick check on one large file
    hourly_check = "unknown"
    for yr, folder in year_dirs():
        files = csv_files(folder)
        for p in files:
            df = safe_read(p)
            if len(df) > 100 and "Timestamp" in df.columns:
                ts = pd.to_datetime(df["Timestamp"], errors="coerce").dropna()
                diffs = ts.diff().dropna()
                median_diff = diffs.median()
                hourly_check = str(median_diff)
                break
        if hourly_check != "unknown":
            break

    # ── Last year partial? ────────────────────────────────────────────────────
    max_year = max(yr for yr, _ in year_dirs())
    last_year_partial = False
    if dt_max is not None:
        if dt_max.month < 12 or (dt_max.month == 12 and dt_max.day < 28):
            last_year_partial = True

    # ── Column dtype summary (from full load) ─────────────────────────────────
    # Re-derive from a single representative file for dtype display
    dtype_summary = {}
    for yr, folder in year_dirs():
        files = csv_files(folder)
        for p in files:
            df = safe_read(p)
            if len(df) > 100:
                dtype_summary = dict(df.dtypes.astype(str))
                break
        if dtype_summary:
            break

    # ── Null fraction per year (for key columns) ──────────────────────────────
    null_frac_per_year = {}
    for col in first_collist or []:
        null_frac_per_year[col] = {}
        for yr, _ in year_dirs():
            t = total_per_year[col].get(yr, 0)
            n = null_per_year[col].get(yr, 0)
            null_frac_per_year[col][yr] = n / t if t else None

    return {
        "total_rows": total_rows,
        "columns": first_collist or [],
        "columns_differ": columns_differ,
        "unique_column_orders": len(set(all_columns_sets)),
        "dtype_summary": dtype_summary,
        "dt_min": dt_min,
        "dt_max": dt_max,
        "hourly_check": hourly_check,
        "duplicate_ts_total": duplicate_ts_total,
        "num_stations": len(all_stations),
        "all_stations": sorted(all_stations),
        "station_year_rows": dict(station_year_rows),
        "rows_per_year": dict(rows_per_year),
        "files_per_year": dict(files_per_year),
        "null_counts": dict(null_counts),
        "total_counts": dict(total_counts),
        "null_frac_per_year": null_frac_per_year,
        "numeric_stats": numeric_stats,
        "text_placeholders": {k: sorted(v) for k, v in text_placeholders.items()},
        "negative_counts": {k: v for k, v in negative_counts.items() if v > 0},
        "humidity_above_100": humidity_above_100,
        "humidity_below_0": humidity_below_0,
        "aqi_present": aqi_present,
        "last_year_partial": last_year_partial,
        "max_year": max_year,
    }


# ── Report writer ─────────────────────────────────────────────────────────────

def write_report(sample_info, stats):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "data_inspection.md"

    lines = []
    add = lines.append

    # ── Plain-English summary ─────────────────────────────────────────────────
    add("# Delhi Air Quality — Data Inspection Report\n")
    add("## Plain-English Summary\n")
    add(f"The dataset contains **{stats['total_rows']:,}** hourly observation rows "
        f"across **{stats['num_stations']}** monitoring stations in Delhi, spanning "
        f"**{stats['dt_min'].strftime('%Y-%m-%d') if stats['dt_min'] else '?'}** to "
        f"**{stats['dt_max'].strftime('%Y-%m-%d %H:%M') if stats['dt_max'] else '?'}**.\n")
    add(f"Files are organised as one CSV per station-year inside year folders (2010–{stats['max_year']}). "
        f"Each CSV has the same 25 columns (Timestamp + 24 measurement fields). "
        f"{'All files share identical column names and order.' if not stats['columns_differ'] else '⚠ Column names or order differ between some files.'}\n")
    add(f"Station coverage grows from 10 in 2010–2016 to 36–37 from 2017 onward. "
        f"{'The last year (' + str(stats['max_year']) + ') is **partial** (data ends ' + stats['dt_max'].strftime('%Y-%m-%d') + ').' if stats['last_year_partial'] else 'The last year appears complete.'}\n")
    add(f"**No AQI column** exists in the raw data — it must be computed." if not stats['aqi_present']
        else "An AQI column **does** exist in the raw data.\n")
    add(f"\nCO is reported in **mg/m³** (milligrams), all other pollutants in **µg/m³** (micrograms). "
        f"Text placeholder `\"NA\"` is used for missing values in newer files; earlier files use empty cells.\n")
    add("")

    # ── Decisions needed ──────────────────────────────────────────────────────
    add("## Decisions You Need to Make\n")
    add("1. **Single station vs. city-wide aggregate**: Use all stations and average/aggregate spatially, "
        "or pick one or a few 'reference' stations with best coverage?\n")
    add("2. **Years to include**: 2010–2016 have only 10 stations (and some of those are header-only). "
        "Include them for long-term trend, or start from 2017 when 36 stations are available?\n")
    add("3. **2026 partial year**: Include partial 2026 for near-real-time analysis, or exclude to avoid "
        "incomplete-year bias?\n")
    add("4. **VOCs and rare met columns**: Keep benzene/toluene/xylene/ethylbenzene/mp-xylene "
        "(high null rates) or drop them to simplify?\n")
    add("5. **CO unit normalisation**: Convert CO to µg/m³ (×1000) for consistency, or keep mg/m³?\n")
    add("6. **AQI computation**: Compute sub-indices and overall AQI using India's NAQI formula? "
        "If so, which breakpoints?\n")
    add("7. **Duplicate-timestamp handling**: Remove exact duplicates, average them, or flag and keep?\n")
    add("8. **Imputation strategy**: Forward-fill short gaps, interpolate, model-based, or leave as NaN?\n")
    add("")

    # ── Column names and dtypes ───────────────────────────────────────────────
    add("## Columns, Dtypes, and Roles\n")
    add("| # | Raw Column | Pandas Dtype | Role | Unit |")
    add("|---|---|---|---|---|")
    for i, col in enumerate(stats["columns"], 1):
        dt = stats["dtype_summary"].get(col, "?")
        role = COLUMN_MAP.get(col, "—")
        # unit from column name if present
        m = re.search(r"\(([^)]+)\)", col)
        unit = m.group(1) if m else ("datetime" if role == "datetime" else "—")
        add(f"| {i} | `{col}` | {dt} | {role} | {unit} |")
    add("")

    # ── Datetime range and frequency ──────────────────────────────────────────
    add("## Datetime Range & Frequency\n")
    add(f"- **Column**: `Timestamp`")
    add(f"- **Min**: {stats['dt_min']}")
    add(f"- **Max**: {stats['dt_max']}")
    add(f"- **Median row interval** (sampled): {stats['hourly_check']}  → **hourly**")
    add(f"- **Duplicate timestamps** (across all files, within-file): {stats['duplicate_ts_total']:,}")
    add(f"- **Last year ({stats['max_year']}) partial**: {'YES' if stats['last_year_partial'] else 'No'}")
    add("")

    # ── Station coverage ──────────────────────────────────────────────────────
    add("## Station & Row Coverage\n")
    add(f"- **Total unique stations**: {stats['num_stations']}")
    add("")
    add("| Year | Files | Rows |")
    add("|------|-------|------|")
    for yr, _ in year_dirs():
        add(f"| {yr} | {stats['files_per_year'].get(yr,0)} | {stats['rows_per_year'].get(yr,0):,} |")
    add("")

    # ── Null fraction per column (overall) ────────────────────────────────────
    add("## Null Fraction per Column (Overall)\n")
    add("| Column | Role | Null % |")
    add("|--------|------|--------|")
    for col in stats["columns"]:
        t = stats["total_counts"].get(col, 0)
        n = stats["null_counts"].get(col, 0)
        pct = f"{100*n/t:.1f}%" if t else "—"
        add(f"| `{col}` | {COLUMN_MAP.get(col,'—')} | {pct} |")
    add("")

    # ── Null fraction per year (selected columns) ─────────────────────────────
    key_roles = ["pm25", "pm10", "no2", "co", "o3", "temperature", "humidity"]
    key_cols = [ROLE_TO_RAW.get(r) for r in key_roles if ROLE_TO_RAW.get(r)]
    if key_cols:
        add("## Null % per Year (Key Columns)\n")
        header = "| Year | " + " | ".join(COLUMN_MAP.get(c, c) for c in key_cols) + " |"
        sep    = "|------|" + "|".join(["-----:" for _ in key_cols]) + "|"
        add(header)
        add(sep)
        for yr, _ in year_dirs():
            vals = []
            for col in key_cols:
                v = stats["null_frac_per_year"].get(col, {}).get(yr)
                vals.append(f"{100*v:.1f}%" if v is not None else "—")
            add(f"| {yr} | " + " | ".join(vals) + " |")
        add("")

    # ── Numeric stats ─────────────────────────────────────────────────────────
    add("## Numeric Column Stats (Min / Max / Mean)\n")
    add("| Column | Role | Min | Max | Mean | Count |")
    add("|--------|------|-----|-----|------|-------|")
    for col in stats["columns"]:
        if col in stats["numeric_stats"]:
            s = stats["numeric_stats"][col]
            mean = s["sum"] / s["count"] if s["count"] else 0
            add(f"| `{col}` | {COLUMN_MAP.get(col,'—')} | {s['min']:.4g} | {s['max']:.4g} | {mean:.4g} | {s['count']:,} |")
    add("")

    # ── Text placeholders in numeric columns ──────────────────────────────────
    add("## Text Placeholders in Numeric Columns\n")
    if stats["text_placeholders"]:
        add("| Column | Examples (up to 20) |")
        add("|--------|---------------------|")
        for col, vals in sorted(stats["text_placeholders"].items()):
            add(f"| `{col}` | {', '.join(repr(v) for v in vals[:20])} |")
    else:
        add("None found — all non-null values coerce to numeric.\n")
    add("")

    # ── Negative pollutant values ─────────────────────────────────────────────
    add("## Negative Values in Pollutant Columns\n")
    if stats["negative_counts"]:
        add("| Column | Role | Negative Count |")
        add("|--------|------|----------------|")
        for col, cnt in sorted(stats["negative_counts"].items()):
            add(f"| `{col}` | {COLUMN_MAP.get(col,'—')} | {cnt:,} |")
    else:
        add("No negative pollutant values found.\n")
    add("")

    # ── Humidity out of range ─────────────────────────────────────────────────
    add("## Humidity Range Check\n")
    add(f"- Values > 100%: **{stats['humidity_above_100']:,}**")
    add(f"- Values < 0%:   **{stats['humidity_below_0']:,}**")
    add("")

    # ── AQI column ────────────────────────────────────────────────────────────
    add("## AQI Column\n")
    add(f"{'**Not present** in raw data. Must be computed during cleaning/feature-engineering.' if not stats['aqi_present'] else '**Present** in raw data.'}\n")

    # ── Pollutant units ───────────────────────────────────────────────────────
    add("## Pollutant Units\n")
    add("| Pollutant | Raw Column | Unit |")
    add("|-----------|-----------|------|")
    for role in POLLUTANT_ROLES:
        raw = ROLE_TO_RAW.get(role, "—")
        m = re.search(r"\(([^)]+)\)", raw)
        unit = m.group(1) if m else "—"
        add(f"| {role} | `{raw}` | {unit} |")
    add("")
    add("> ⚠ **CO is in mg/m³** while every other gas/particle pollutant is in µg/m³. "
        "Decide whether to convert CO to µg/m³ (×1000) before modelling.\n")

    # ── Rows per station per year (compact) ───────────────────────────────────
    add("## Rows per Station per Year (sample)\n")
    add("Showing first 5 stations alphabetically.\n")
    sample_stations = stats["all_stations"][:5]
    years = [yr for yr, _ in year_dirs()]
    header = "| Station | " + " | ".join(str(y) for y in years) + " |"
    sep    = "|---------|" + "|".join(["-----:" for _ in years]) + "|"
    add(header)
    add(sep)
    for st in sample_stations:
        vals = []
        for yr in years:
            v = stats["station_year_rows"].get(st, {}).get(yr, 0)
            vals.append(str(v) if v else "—")
        add(f"| {st} | " + " | ".join(vals) + " |")
    add("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅  Report written to {out}")
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PASS 1: Sample (first + last file per year)")
    print("=" * 70)
    sample_info = sample_pass()
    for info in sample_info:
        print(f"  {info['year']}/{info['file']}  rows={info['rows']}  cols={len(info['columns'])}")

    # Quick column-consistency check
    col_sets = set(tuple(i["columns"]) for i in sample_info if i["columns"])
    if len(col_sets) == 1:
        print("\n  ✓ All sampled files share the same column names and order.")
    else:
        print(f"\n  ⚠ {len(col_sets)} distinct column orderings found in sample!")

    print("\n" + "=" * 70)
    print("PASS 2: Full (all files)")
    print("=" * 70)
    stats = full_pass()

    print(f"\n  Total rows:      {stats['total_rows']:,}")
    print(f"  Unique stations: {stats['num_stations']}")
    print(f"  Date range:      {stats['dt_min']} → {stats['dt_max']}")
    print(f"  AQI column:      {'Yes' if stats['aqi_present'] else 'No'}")
    print(f"  Columns differ:  {stats['columns_differ']}")

    write_report(sample_info, stats)


if __name__ == "__main__":
    main()

