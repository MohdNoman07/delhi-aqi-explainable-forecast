import pandas as pd
from datetime import timedelta
import sys
from pathlib import Path
REPO_ROOT = Path("D:/ML_Project")
sys.path.insert(0, str(REPO_ROOT))
from src.config import DIWALI_DATES
from src.evaluate import get_aqi_category_6

test_preds = pd.read_csv(REPO_ROOT / "outputs" / "test_predictions.csv")
daily_path = REPO_ROOT / "data" / "processed" / "clean_daily.csv"
clean_daily = pd.read_csv(daily_path)
clean_daily["date"] = pd.to_datetime(clean_daily["date"])
missing_aqi_days = set(clean_daily[clean_daily["aqi"].isna()]["date"])
full_cal = set(pd.date_range("2018-01-01", "2026-09-30", freq="D"))
absent_days = full_cal - set(clean_daily["date"])
all_missing_days = missing_aqi_days.union(absent_days)

def is_next_to_missing(dt):
    prev_d = dt - timedelta(days=1)
    next_d = dt + timedelta(days=1)
    return (prev_d in all_missing_days) or (next_d in all_missing_days)

diwali_dts = [pd.to_datetime(d) for d in DIWALI_DATES]

total_in_window = 0
total_outside_window = 0
in_window_days_by_h = {}

for h in range(1, 6):
    df_h = test_preds[test_preds["horizon"] == h].copy()
    df_h["abs_err"] = (df_h["truth"] - df_h["pred_random_forest"]).abs()
    worst10 = df_h.sort_values(by="abs_err", ascending=False).head(10).reset_index(drop=True)
    
    rows = []
    in_window_days_by_h[h] = []
    for idx, r in worst10.iterrows():
        f_dt = pd.to_datetime(r["date"])
        t_dt = f_dt + timedelta(days=h)
        
        # days from nearest Diwali date
        diffs = [(t_dt - d_dt).days for d_dt in diwali_dts]
        days_from_diwali = min(diffs, key=abs)
        
        # Window: from 1 day before to 3 days after Diwali (-1 <= days_from_diwali <= 3)
        in_window = (-1 <= days_from_diwali <= 3)
        if in_window:
            total_in_window += 1
            in_window_days_by_h[h].append((t_dt.strftime("%Y-%m-%d"), days_from_diwali, r['abs_err']))
        else:
            total_outside_window += 1
            
        next_m = is_next_to_missing(t_dt)
        
        rows.append({
            "forecast_date": f_dt.strftime("%Y-%m-%d"),
            "target_date": t_dt.strftime("%Y-%m-%d"),
            "actual": round(float(r["truth"]), 2),
            "predicted": round(float(r["pred_random_forest"]), 2),
            "error": round(float(r["abs_err"]), 2),
            "actual_category": get_aqi_category_6(r["truth"]),
            "predicted_category": get_aqi_category_6(r["pred_random_forest"]),
            "days_from_diwali": days_from_diwali,
            "within_diwali_window": "Yes" if in_window else "No",
            "next_to_missing_data_day": "Yes" if next_m else "No",
        })
    df_w = pd.DataFrame(rows)
    print(f"\n--- Horizon h={h} worst days (top 3) ---")
    print(df_w[['forecast_date', 'target_date', 'actual', 'predicted', 'error', 'days_from_diwali', 'within_diwali_window']].head(3))
    print(f"Horizon h={h}: {len(in_window_days_by_h[h])} days in window [-1, 3]: {in_window_days_by_h[h]}")

print(f"\nTotal in window [-1, 3]: {total_in_window}")
print(f"Total outside window: {total_outside_window}")
