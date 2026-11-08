import sys
from pathlib import Path
REPO_ROOT = Path("D:/ML_Project")
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from src.config import RANDOM_SEED
from src.split import get_splits
from src.evaluate import get_aqi_category_6

train_df, val_df, test_df = get_splits()
with open(REPO_ROOT / "models" / "feature_columns.json", "r") as f:
    import json
    feature_cols = json.load(f)

test_preds = pd.read_csv(REPO_ROOT / "outputs" / "test_predictions.csv")
test_preds["date"] = pd.to_datetime(test_preds["date"])
test_preds["month"] = test_preds["date"].dt.month
test_preds["is_winter"] = test_preds["month"].isin([10, 11, 12, 1, 2])

for h in [1, 5]:
    df_h = test_preds[test_preds["horizon"] == h].copy()
    # Severe winter days: is_winter == True and truth >= 401 (or category == Severe)
    severe_winter = df_h[df_h["is_winter"] & (df_h["truth"] > 400.0)].copy()
    severe_winter["abs_err"] = (severe_winter["truth"] - severe_winter["pred_random_forest"]).abs()
    
    smallest_err = severe_winter.sort_values("abs_err").iloc[0]
    largest_err = severe_winter.sort_values("abs_err", ascending=False).iloc[0]
    
    print(f"\n--- Horizon h={h} Severe Winter Days ---")
    print(f"Total Severe winter days: {len(severe_winter)}")
    print(f"Smallest error day: date={smallest_err['date'].date()}, truth={smallest_err['truth']:.2f}, pred={smallest_err['pred_random_forest']:.2f}, err={smallest_err['abs_err']:.2f}")
    print(f"Largest error day: date={largest_err['date'].date()}, truth={largest_err['truth']:.2f}, pred={largest_err['pred_random_forest']:.2f}, err={largest_err['abs_err']:.2f}")
