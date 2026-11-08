import sys
from pathlib import Path
REPO_ROOT = Path("D:/ML_Project")
sys.path.insert(0, str(REPO_ROOT))

import json
import joblib
import pandas as pd
import numpy as np
from src.features import compute_features

daily = pd.read_csv(REPO_ROOT / "data/processed/clean_daily.csv")
daily["date"] = pd.to_datetime(daily["date"])
full_calendar = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
daily_full = daily.set_index("date").reindex(full_calendar)
daily_full.index.name = "date"

with open(REPO_ROOT / "models/feature_columns.json", "r") as f:
    feat_cols = json.load(f)

rf_models = {h: joblib.load(REPO_ROOT / f"models/rf_h{h}.pkl") for h in range(1, 6)}
test_preds = pd.read_csv(REPO_ROOT / "outputs/test_predictions.csv")

test_dates = ["2024-11-15", "2025-06-20", "2025-10-19"]

for dt_str in test_dates:
    dt = pd.to_datetime(dt_str)
    hist = daily_full.loc[:dt]
    f = compute_features(hist)
    x_row = f.loc[[dt], feat_cols]
    
    print(f"\nDate: {dt_str}")
    for h in range(1, 6):
        pred = float(np.clip(rf_models[h].predict(x_row)[0], 0.0, 500.0))
        # Look up in test_predictions.csv
        sub = test_preds[(test_preds["date"] == dt_str) & (test_preds["horizon"] == h)]
        expected = float(sub["pred_random_forest"].iloc[0])
        diff = abs(pred - expected)
        print(f"  h={h}: Pred={pred:.4f}, Expected={expected:.4f}, Diff={diff:.2e}")
        assert diff < 1e-5
print("\nAll 3 test dates match test_predictions.csv within 1e-5!")
