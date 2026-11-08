import sys
from pathlib import Path
REPO_ROOT = Path("D:/ML_Project")
sys.path.insert(0, str(REPO_ROOT))

import json
import joblib
import shap
import pandas as pd
import numpy as np

from src.split import get_splits

train_df, val_df, test_df = get_splits()
with open(REPO_ROOT / "models" / "feature_columns.json", "r") as f:
    feature_cols = json.load(f)

FEATURE_FAMILIES = {
    "AQI history": [
        "aqi", "aqi_lag1", "aqi_lag2", "aqi_lag3", "aqi_lag7",
        "aqi_rolling_mean_3d", "aqi_rolling_mean_7d", "aqi_rolling_max_7d", "aqi_trend_3d"
    ],
    "Pollutants": [
        "pm25", "pm25_rolling_mean_3d", "pm10", "pm10_rolling_mean_3d",
        "no2", "no2_rolling_mean_3d", "so2", "so2_rolling_mean_3d",
        "co", "co_rolling_mean_3d", "o3", "o3_rolling_mean_3d",
        "pm25_max", "pm10_max"
    ],
    "Weather": [
        "temperature", "humidity", "wind_speed", "barometric_pressure",
        "wind_dir_sin", "wind_dir_cos",
        "temperature_rolling_mean_3d", "humidity_rolling_mean_3d", "wind_speed_rolling_mean_3d"
    ],
    "Calendar": [
        "month", "day_of_year_sin", "day_of_year_cos",
        "day_of_week", "is_weekend", "stubble_season"
    ]
}

# Verify partition of all 38 features
all_fam = []
for k, v in FEATURE_FAMILIES.items():
    all_fam.extend(v)
assert set(all_fam) == set(feature_cols)
assert len(all_fam) == len(feature_cols) == 38
print("Feature family partition verified: exactly 38 features partitioned into 4 families.")
