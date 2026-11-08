"""
scratch/test_error_analysis.py
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
from datetime import timedelta
import numpy as np
import pandas as pd

from src.config import DIWALI_DATES
from src.evaluate import get_aqi_category_6

test_preds = pd.read_csv("outputs/test_predictions.csv")
test_preds["date"] = pd.to_datetime(test_preds["date"])
diwali_dts = [pd.to_datetime(d) for d in DIWALI_DATES]

# Check missing AQI days in clean_daily
clean_daily = pd.read_csv("data/processed/clean_daily.csv")
clean_daily["date"] = pd.to_datetime(clean_daily["date"])
missing_aqi_days = set(clean_daily[clean_daily["aqi"].isna()]["date"])

# Check calendar days completely absent from clean_daily
full_calendar = set(pd.date_range("2018-01-01", "2026-09-30", freq="D"))
absent_days = full_calendar - set(clean_daily["date"])
all_missing_days = missing_aqi_days.union(absent_days)

print(f"Total missing or absent AQI days across 2018-2026: {len(all_missing_days)}")
print(f"Missing days in 2024-2026: {[d.strftime('%Y-%m-%d') for d in sorted(all_missing_days) if d.year >= 2024]}")

# Function to check adjacent to missing day
def is_next_to_missing(dt):
    prev_d = dt - timedelta(days=1)
    next_d = dt + timedelta(days=1)
    return (prev_d in all_missing_days) or (next_d in all_missing_days)

def is_within_3d_diwali(dt):
    return any(abs((dt - d_dt).days) <= 3 for d_dt in diwali_dts)

# Check worst 10 days for each horizon
print("\n" + "="*80)
print("10 WORST DAYS PER HORIZON FOR RANDOM FOREST:")
print("="*80)

total_near_diwali = 0
total_not_near_diwali = 0

for h in range(1, 6):
    df_h = test_preds[test_preds["horizon"] == h].copy()
    df_h["abs_err"] = (df_h["truth"] - df_h["pred_random_forest"]).abs()
    worst10 = df_h.sort_values(by="abs_err", ascending=False).head(10).reset_index(drop=True)
    
    print(f"\n--- Horizon h={h} Worst 10 Days ---")
    for idx, r in worst10.iterrows():
        d = r["date"]
        near_diwali = is_within_3d_diwali(d)
        next_missing = is_next_to_missing(d)
        act_cat = get_aqi_category_6(r["truth"])
        pred_cat = get_aqi_category_6(r["pred_random_forest"])
        
        if near_diwali:
            total_near_diwali += 1
        else:
            total_not_near_diwali += 1
            
        print(f"  {idx+1:2d}. {d.strftime('%Y-%m-%d')} | Actual: {r['truth']:6.2f} ({act_cat:10s}) | Pred: {r['pred_random_forest']:6.2f} ({pred_cat:10s}) | Err: {r['abs_err']:6.2f} | Near Diwali: {str(near_diwali):5s} | Next to Missing: {str(next_missing):5s}")

print(f"\nTotal across all 50 worst days (10 x 5 horizons): {total_near_diwali} near Diwali, {total_not_near_diwali} not near Diwali.")
