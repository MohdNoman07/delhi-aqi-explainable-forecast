import sys
from pathlib import Path
sys.path.insert(0, str(Path("D:/ML_Project")))
import pandas as pd
import numpy as np
from src.features import compute_features

daily = pd.read_csv("data/processed/clean_daily.csv")
daily["date"] = pd.to_datetime(daily["date"])

# Complete calendar reindexing as in features.py
full_calendar = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
daily_full = daily.set_index("date").reindex(full_calendar)
daily_full.index.name = "date"

# Pick 3 dates
test_dates = ["2024-11-15", "2025-06-20", "2025-10-19"]
model_ready = pd.read_csv("data/processed/model_ready.csv").set_index("date")

for dt_str in test_dates:
    dt = pd.to_datetime(dt_str)
    # History up to dt
    hist = daily_full.loc[:dt]
    f = compute_features(hist)
    f_row = f.loc[dt]
    mr_row = model_ready.loc[dt_str]
    
    # Compare common feature columns
    feat_cols = [c for c in f_row.index if c in mr_row.index]
    diffs = np.abs(f_row[feat_cols].values.astype(float) - mr_row[feat_cols].values.astype(float))
    print(f"Date {dt_str}: max feature difference = {np.max(diffs)}")
