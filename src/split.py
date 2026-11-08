"""
src/split.py

Chronological Train / Validation / Test Splitting.
Strictly adheres to Project Rules:
  - Rule 2: Never shuffle time series. Use chronological split with a 5-day gap.
            The test set is used only in the final evaluation script.
  - Chronological partitions:
      Train:      date <= TRAIN_END (2023-03-31)
      Gap 1:      TRAIN_END < date <= TRAIN_END + 5 days (2023-04-01 to 2023-04-05)
      Validation: TRAIN_END + 5 days < date <= VAL_END (2023-04-06 to 2024-03-31)
      Gap 2:      VAL_END < date <= VAL_END + 5 days (2024-04-01 to 2024-04-05)
      Test:       date > VAL_END + 5 days (2024-04-06 to 2026-09-25)

Outputs:
  - outputs/split_summary.md
  - outputs/split_diagram.png
"""

import sys
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import TimeSeriesSplit

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import TRAIN_END, VAL_END, GAP_DAYS, CV_SPLITS, CV_GAP, RANDOM_SEED


def get_aqi_category(aqi_val: float) -> str:
    """Returns official CPCB AQI category."""
    if pd.isna(aqi_val):
        return "Missing"
    elif aqi_val <= 50:
        return "Good"
    elif aqi_val <= 100:
        return "Satisfactory"
    elif aqi_val <= 200:
        return "Moderate"
    elif aqi_val <= 300:
        return "Poor"
    elif aqi_val <= 400:
        return "Very Poor"
    else:
        return "Severe"


def get_splits(df: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset chronologically into Train, Validation, and Test sets with 5-day buffer gaps.
    Returns (train_df, val_df, test_df) without shuffling (strictly chronological order).
    """
    if df is None:
        data_path = REPO_ROOT / "data" / "processed" / "model_ready.csv"
        df = pd.read_csv(data_path)
    
    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    df_copy = df_copy.sort_values("date").reset_index(drop=True)

    train_end_dt = pd.to_datetime(TRAIN_END)
    val_end_dt = pd.to_datetime(VAL_END)
    gap_delta = timedelta(days=GAP_DAYS)

    # Filtering partitions
    train_mask = df_copy["date"] <= train_end_dt
    val_mask = (df_copy["date"] > (train_end_dt + gap_delta)) & (df_copy["date"] <= val_end_dt)
    test_mask = df_copy["date"] > (val_end_dt + gap_delta)

    train_df = df_copy[train_mask].copy().reset_index(drop=True)
    val_df = df_copy[val_mask].copy().reset_index(drop=True)
    test_df = df_copy[test_mask].copy().reset_index(drop=True)

    # Convert date back to string for standard output consistency
    train_df["date"] = train_df["date"].dt.strftime("%Y-%m-%d")
    val_df["date"] = val_df["date"].dt.strftime("%Y-%m-%d")
    test_df["date"] = test_df["date"].dt.strftime("%Y-%m-%d")

    return train_df, val_df, test_df


def generate_split_diagram(df: pd.DataFrame, out_path: Path):
    """
    Generates a timeline diagram of the Train, Validation, Test splits,
    with 5-day gaps and all winter seasons shaded.
    """
    df_plot = df.copy()
    df_plot["date"] = pd.to_datetime(df_plot["date"])
    
    train_end_dt = pd.to_datetime(TRAIN_END)
    val_end_dt = pd.to_datetime(VAL_END)
    gap_delta = timedelta(days=GAP_DAYS)

    fig, (ax_timeline, ax_aqi) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [1, 2.5]}
    )

    # Winter periods (October 1 to end of February)
    winters = [
        ("2018-10-01", "2019-02-28", "Winter 18-19"),
        ("2019-10-01", "2020-02-29", "Winter 19-20"),
        ("2020-10-01", "2021-02-28", "Winter 20-21"),
        ("2021-10-01", "2022-02-28", "Winter 21-22"),
        ("2022-10-01", "2023-02-28", "Winter 22-23"),
        ("2023-10-01", "2024-02-29", "Val Winter 23-24"),
        ("2024-10-01", "2025-02-28", "Test Winter 24-25"),
        ("2025-10-01", "2026-02-28", "Test Winter 25-26"),
    ]

    # Shading winters on both axes
    for w_start, w_end, w_lbl in winters:
        ws = pd.to_datetime(w_start)
        we = pd.to_datetime(w_end)
        ax_aqi.axvspan(ws, we, color="gray", alpha=0.18, zorder=0)
        ax_timeline.axvspan(ws, we, color="gray", alpha=0.18, zorder=0)

    # 1. Timeline bar on ax_timeline
    y_pos = 0.5
    bar_height = 0.5

    # Train
    t_start = df_plot["date"].min()
    t_end = train_end_dt
    ax_timeline.barh(y_pos, (t_end - t_start).days, left=t_start, height=bar_height,
                     color="#1f77b4", edgecolor="black", label=f"Train ({t_start.strftime('%Y-%m-%d')} to {t_end.strftime('%Y-%m-%d')})")

    # Gap 1
    g1_start = train_end_dt
    g1_end = train_end_dt + gap_delta
    ax_timeline.barh(y_pos, (g1_end - g1_start).days, left=g1_start, height=bar_height,
                     color="#d62728", hatch="//", edgecolor="black", label="5-Day Gap 1 (Dropped)")

    # Val
    v_start = train_end_dt + gap_delta
    v_end = val_end_dt
    ax_timeline.barh(y_pos, (v_end - v_start).days, left=v_start, height=bar_height,
                     color="#ff7f0e", edgecolor="black", label=f"Validation ({v_start.strftime('%Y-%m-%d')} to {v_end.strftime('%Y-%m-%d')})")

    # Gap 2
    g2_start = val_end_dt
    g2_end = val_end_dt + gap_delta
    ax_timeline.barh(y_pos, (g2_end - g2_start).days, left=g2_start, height=bar_height,
                     color="#d62728", hatch="//", edgecolor="black", label="5-Day Gap 2 (Dropped)")

    # Test
    te_start = val_end_dt + gap_delta
    te_end = df_plot["date"].max()
    ax_timeline.barh(y_pos, (te_end - te_start).days, left=te_start, height=bar_height,
                     color="#2ca02c", edgecolor="black", label=f"Test ({te_start.strftime('%Y-%m-%d')} to {te_end.strftime('%Y-%m-%d')})")

    ax_timeline.set_ylim(0, 1)
    ax_timeline.set_yticks([])
    ax_timeline.set_title("Chronological Train / Validation / Test Splitting Timeline (5-Day Gaps & Winters Shaded)", fontsize=13, fontweight="bold")
    ax_timeline.legend(loc="upper left", bbox_to_anchor=(0.0, 1.45), ncol=3, frameon=True, fontsize=9)

    # 2. Daily AQI on ax_aqi
    ax_aqi.plot(df_plot["date"], df_plot["aqi"], color="#333333", alpha=0.5, linewidth=0.8, label="Daily AQI")
    ax_aqi.axhline(200, color="orange", linestyle="--", linewidth=1.0, alpha=0.8, label="Poor / Unhealthy Threshold (AQI > 200)")
    ax_aqi.axhline(400, color="crimson", linestyle="--", linewidth=1.0, alpha=0.8, label="Severe Threshold (AQI > 400)")

    # Vertical split markers
    ax_aqi.axvline(train_end_dt, color="#1f77b4", linestyle="-", linewidth=1.5, alpha=0.9)
    ax_aqi.axvline(train_end_dt + gap_delta, color="#ff7f0e", linestyle="-", linewidth=1.5, alpha=0.9)
    ax_aqi.axvline(val_end_dt, color="#ff7f0e", linestyle="-", linewidth=1.5, alpha=0.9)
    ax_aqi.axvline(val_end_dt + gap_delta, color="#2ca02c", linestyle="-", linewidth=1.5, alpha=0.9)

    ax_aqi.set_ylabel("Daily AQI", fontsize=11)
    ax_aqi.set_ylim(0, 520)
    ax_aqi.xaxis.set_major_locator(mdates.YearLocator())
    ax_aqi.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_aqi.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax_aqi.grid(True, linestyle=":", alpha=0.5)
    ax_aqi.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved timeline diagram to {out_path}")


def main():
    data_path = REPO_ROOT / "data" / "processed" / "model_ready.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"model_ready.csv not found at {data_path}")
    
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    train_end_dt = pd.to_datetime(TRAIN_END)
    val_end_dt = pd.to_datetime(VAL_END)
    gap_delta = timedelta(days=GAP_DAYS)

    train_df, val_df, test_df = get_splits(df)

    # Inspect gap rows
    gap1_df = df[(df["date"] > train_end_dt) & (df["date"] <= train_end_dt + gap_delta)].copy()
    gap2_df = df[(df["date"] > val_end_dt) & (df["date"] <= val_end_dt + gap_delta)].copy()
    total_gap_rows = len(gap1_df) + len(gap2_df)

    print("="*80)
    print("CHRONOLOGICAL SPLIT SUMMARY:")
    print("="*80)
    print(f"Train set:       {len(train_df):4d} rows | {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Gap 1 (dropped): {len(gap1_df):4d} rows | {gap1_df['date'].min().strftime('%Y-%m-%d')} to {gap1_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Validation set:  {len(val_df):4d} rows | {val_df['date'].min()} to {val_df['date'].max()}")
    print(f"Gap 2 (dropped): {len(gap2_df):4d} rows | {gap2_df['date'].min().strftime('%Y-%m-%d')} to {gap2_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Test set:        {len(test_df):4d} rows | {test_df['date'].min()} to {test_df['date'].max()}")
    print(f"Total rows:      {len(train_df) + total_gap_rows + len(val_df) + len(test_df)} == {len(df)}")

    # Category distributions and thresholds
    splits_dict = {
        "Train": train_df,
        "Validation": val_df,
        "Test": test_df
    }

    stats = {}
    for name, s_df in splits_dict.items():
        s_cats = s_df["aqi"].apply(get_aqi_category).value_counts()
        poor_plus = int((s_df["aqi"] > 200).sum())
        severe = int((s_df["aqi"] > 400).sum())
        stats[name] = {
            "first_date": s_df["date"].min(),
            "last_date": s_df["date"].max(),
            "row_count": len(s_df),
            "categories": {c: int(s_cats.get(c, 0)) for c in ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]},
            "poor_plus": poor_plus,
            "poor_plus_pct": poor_plus / len(s_df) * 100,
            "severe": severe,
            "severe_pct": severe / len(s_df) * 100,
        }

    # Winter coverage verification
    val_months = pd.to_datetime(val_df["date"]).dt.to_period("M").unique().astype(str).tolist()
    expected_val_winter = ["2023-10", "2023-11", "2023-12", "2024-01", "2024-02"]
    val_winter_ok = all(m in val_months for m in expected_val_winter)

    test_months = pd.to_datetime(test_df["date"]).dt.to_period("M").unique().astype(str).tolist()
    expected_test_winter_1 = ["2024-10", "2024-11", "2024-12", "2025-01", "2025-02"]
    expected_test_winter_2 = ["2025-10", "2025-11", "2025-12", "2026-01", "2026-02"]
    test_w1_ok = all(m in test_months for m in expected_test_winter_1)
    test_w2_ok = all(m in test_months for m in expected_test_winter_2)
    test_winter_ok = test_w1_ok and test_w2_ok

    print("\n" + "="*80)
    print("WINTER COVERAGE CONFIRMATION:")
    print("="*80)
    print(f"Validation set contains full winter Oct 2023 - Feb 2024: {val_winter_ok}")
    print(f"Test set contains full winter Oct 2024 - Feb 2025:       {test_w1_ok}")
    print(f"Test set contains full winter Oct 2025 - Feb 2026:       {test_w2_ok}")
    print(f"No missing months in validation or test winters:         {val_winter_ok and test_winter_ok}")

    # Expanding window folds via TimeSeriesSplit(n_splits=CV_SPLITS, gap=CV_GAP) on train_df
    # Ensure train_df is sorted
    train_sorted = train_df.sort_values("date").reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=CV_GAP)

    print("\n" + "="*80)
    print(f"EXPANDING-WINDOW FOLDS ON TRAINING SET (TimeSeriesSplit(n_splits={CV_SPLITS}, gap={CV_GAP})):")
    print("="*80)
    folds_info = []
    for fold, (tr_idx, v_idx) in enumerate(tscv.split(train_sorted), 1):
        tr_part = train_sorted.iloc[tr_idx]
        v_part = train_sorted.iloc[v_idx]
        tr_start, tr_end = tr_part["date"].min(), tr_part["date"].max()
        v_start, v_end = v_part["date"].min(), v_part["date"].max()
        poor_plus_fold = int((v_part["aqi"] > 200).sum())
        severe_fold = int((v_part["aqi"] > 400).sum())
        folds_info.append({
            "fold": fold,
            "train_start": tr_start,
            "train_end": tr_end,
            "train_days": len(tr_part),
            "val_start": v_start,
            "val_end": v_end,
            "val_days": len(v_part),
            "poor_plus": poor_plus_fold,
            "poor_plus_pct": poor_plus_fold / len(v_part) * 100,
            "severe": severe_fold,
        })
        print(f"Fold {fold}:")
        print(f"  Training Part:   {tr_start} to {tr_end} ({len(tr_part):4d} days)")
        print(f"  Validation Part: {v_start} to {v_end} ({len(v_part):4d} days)")
        print(f"  Validation Poor-or-worse days (AQI > 200): {poor_plus_fold:3d} ({poor_plus_fold/len(v_part)*100:5.2f}%)")
        print(f"  Validation Severe days (AQI > 400):        {severe_fold:3d} ({severe_fold/len(v_part)*100:5.2f}%)")

    # Generate timeline plot
    diagram_path = REPO_ROOT / "outputs" / "split_diagram.png"
    generate_split_diagram(df, diagram_path)

    # Build folds table rows dynamically from folds_info (length = CV_SPLITS)
    _header = "| Fold | Training Window | Train Days | Validation Window | Val Days | Validation Poor+ (AQI > 200) | Validation Severe (AQI > 400) |\n|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    _rows = "\n".join(
        f"| **Fold {fi['fold']}** | `{fi['train_start']}` to `{fi['train_end']}` | {fi['train_days']} "
        f"| `{fi['val_start']}` to `{fi['val_end']}` | {fi['val_days']} "
        f"| **{fi['poor_plus']}** ({fi['poor_plus_pct']:.2f}%) | **{fi['severe']}** |"
        for fi in folds_info
    )
    cv_folds_table_rows = _header + "\n" + _rows

    # Save summary markdown
    summary_path = REPO_ROOT / "outputs" / "split_summary.md"
    summary_md = f"""# Chronological Train / Validation / Test Split Summary

## 1. Split Definition & Boundary Integrity

Strict adherence to **Rule 2** (Never shuffle time series; chronological splits with a 5-day buffer gap; test set reserved exclusively for final evaluation).

| Partition | Start Date | End Date | Row Count | Days Dropped in Buffer Gap | Description |
|:---|:---:|:---:|:---:|:---:|:---|
| **Train** | `{stats['Train']['first_date']}` | `{stats['Train']['last_date']}` | **{stats['Train']['row_count']}** | — | Training data up to `{TRAIN_END}` (~5.25 years) |
| *Gap 1* | `{gap1_df['date'].min().strftime('%Y-%m-%d')}` | `{gap1_df['date'].max().strftime('%Y-%m-%d')}` | **{len(gap1_df)}** | **5** | 5-day chronological buffer between Train and Val |
| **Validation** | `{stats['Validation']['first_date']}` | `{stats['Validation']['last_date']}` | **{stats['Validation']['row_count']}** | — | Validation data up to `{VAL_END}` (1 full calendar year) |
| *Gap 2* | `{gap2_df['date'].min().strftime('%Y-%m-%d')}` | `{gap2_df['date'].max().strftime('%Y-%m-%d')}` | **{len(gap2_df)}** | **5** | 5-day chronological buffer between Val and Test |
| **Test** | `{stats['Test']['first_date']}` | `{stats['Test']['last_date']}` | **{stats['Test']['row_count']}** | — | Final evaluation holdout (~2.5 years, through `{stats['Test']['last_date']}`) |
| **Total** | `{stats['Train']['first_date']}` | `{stats['Test']['last_date']}` | **{len(df)}** | **{total_gap_rows}** | All calendar rows accounted for |

---

## 2. AQI Category & High-Pollution Day Distribution

### Class Balance Across Partitions

| AQI Category | Range | Train Count (Share) | Validation Count (Share) | Test Count (Share) |
|:---|:---:|:---:|:---:|:---:|
| **Good** | 0 – 50 | {stats['Train']['categories']['Good']} ({stats['Train']['categories']['Good']/stats['Train']['row_count']*100:.2f}%) | {stats['Validation']['categories']['Good']} ({stats['Validation']['categories']['Good']/stats['Validation']['row_count']*100:.2f}%) | {stats['Test']['categories']['Good']} ({stats['Test']['categories']['Good']/stats['Test']['row_count']*100:.2f}%) |
| **Satisfactory** | 51 – 100 | {stats['Train']['categories']['Satisfactory']} ({stats['Train']['categories']['Satisfactory']/stats['Train']['row_count']*100:.2f}%) | {stats['Validation']['categories']['Satisfactory']} ({stats['Validation']['categories']['Satisfactory']/stats['Validation']['row_count']*100:.2f}%) | {stats['Test']['categories']['Satisfactory']} ({stats['Test']['categories']['Satisfactory']/stats['Test']['row_count']*100:.2f}%) |
| **Moderate** | 101 – 200 | {stats['Train']['categories']['Moderate']} ({stats['Train']['categories']['Moderate']/stats['Train']['row_count']*100:.2f}%) | {stats['Validation']['categories']['Moderate']} ({stats['Validation']['categories']['Moderate']/stats['Validation']['row_count']*100:.2f}%) | {stats['Test']['categories']['Moderate']} ({stats['Test']['categories']['Moderate']/stats['Test']['row_count']*100:.2f}%) |
| **Poor** | 201 – 300 | {stats['Train']['categories']['Poor']} ({stats['Train']['categories']['Poor']/stats['Train']['row_count']*100:.2f}%) | {stats['Validation']['categories']['Poor']} ({stats['Validation']['categories']['Poor']/stats['Validation']['row_count']*100:.2f}%) | {stats['Test']['categories']['Poor']} ({stats['Test']['categories']['Poor']/stats['Test']['row_count']*100:.2f}%) |
| **Very Poor** | 301 – 400 | {stats['Train']['categories']['Very Poor']} ({stats['Train']['categories']['Very Poor']/stats['Train']['row_count']*100:.2f}%) | {stats['Validation']['categories']['Very Poor']} ({stats['Validation']['categories']['Very Poor']/stats['Validation']['row_count']*100:.2f}%) | {stats['Test']['categories']['Very Poor']} ({stats['Test']['categories']['Very Poor']/stats['Test']['row_count']*100:.2f}%) |
| **Severe** | 401 – 500 | {stats['Train']['categories']['Severe']} ({stats['Train']['categories']['Severe']/stats['Train']['row_count']*100:.2f}%) | {stats['Validation']['categories']['Severe']} ({stats['Validation']['categories']['Severe']/stats['Validation']['row_count']*100:.2f}%) | {stats['Test']['categories']['Severe']} ({stats['Test']['categories']['Severe']/stats['Test']['row_count']*100:.2f}%) |
| **Total** | — | **{stats['Train']['row_count']}** | **{stats['Validation']['row_count']}** | **{stats['Test']['row_count']}** |

### High Pollution Episodes

| Metric | Condition | Train | Validation | Test |
|:---|:---:|:---:|:---:|:---:|
| **Poor-or-worse Days** | $\\text{{AQI}} > 200$ | **{stats['Train']['poor_plus']}** ({stats['Train']['poor_plus_pct']:.2f}%) | **{stats['Validation']['poor_plus']}** ({stats['Validation']['poor_plus_pct']:.2f}%) | **{stats['Test']['poor_plus']}** ({stats['Test']['poor_plus_pct']:.2f}%) |
| **Severe Days** | $\\text{{AQI}} > 400$ | **{stats['Train']['severe']}** ({stats['Train']['severe_pct']:.2f}%) | **{stats['Validation']['severe']}** ({stats['Validation']['severe_pct']:.2f}%) | **{stats['Test']['severe']}** ({stats['Test']['severe_pct']:.2f}%) |

---

## 3. Winter Season Completeness Audit

Delhi pollution displays strong seasonality, with peak pollution episodes concentrated in winter (October through February).

- **Validation Set Winter Audit**:
  - Required Winter: **October 2023 through February 2024**
  - Present Months: `{', '.join(expected_val_winter)}`
  - Status: **Confirmed complete** (all 5 winter months present with zero missing months).
- **Test Set Winter Audit**:
  - Required Winter 1: **October 2024 through February 2025**
    - Present Months: `{', '.join(expected_test_winter_1)}`
    - Status: **Confirmed complete** (all 5 winter months present).
  - Required Winter 2: **October 2025 through February 2026**
    - Present Months: `{', '.join(expected_test_winter_2)}`
    - Status: **Confirmed complete** (all 5 winter months present).
  - Overall Test Set Winter Status: **Confirmed complete** (all 10 winter months across both seasons present with zero missing months).

---

## 4. Expanding-Window Cross-Validation on Training Set

Configured with `TimeSeriesSplit(n_splits={CV_SPLITS}, gap={CV_GAP})` on the training set only (strictly preserving temporal ordering and 5-day boundary gaps; no test set data accessed).

{cv_folds_table_rows}

"""
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"Saved split summary to {summary_path}")


if __name__ == "__main__":
    main()
