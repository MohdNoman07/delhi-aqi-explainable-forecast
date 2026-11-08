"""
scratch/test_train_logic.py
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.split import get_splits
from src.config import CV_SPLITS, CV_GAP, RANDOM_SEED
from src.evaluate import evaluate, per_class_report, skill_vs_persistence

train_df, val_df, test_df = get_splits()
with open("models/feature_columns.json") as f:
    feat_cols = json.load(f)

X_train = train_df[feat_cols]
X_val = val_df[feat_cols]

# Climatology helper
def to_doy365(dt):
    m, d = dt.month, dt.day
    if m == 2 and d == 29:
        d = 28
    days_before = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    return days_before[m - 1] + (d - 1)

train_dt = pd.to_datetime(train_df["date"])
train_doy = train_dt.apply(to_doy365).values
train_aqi = train_df["aqi"].values

climatology_map = {}
for doy in range(365):
    diff = np.abs(train_doy - doy)
    circ_diff = np.minimum(diff, 365 - diff)
    climatology_map[doy] = float(np.mean(train_aqi[circ_diff <= 7]))

cv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=CV_GAP)

val_dt = pd.to_datetime(val_df["date"])

print("Testing all 3 baselines across horizons 1..5:")
for h in range(1, 6):
    y_true = val_df[f"target_h{h}"].values
    
    # 1. Persistence
    pred_pers = val_df["aqi"].values
    met_pers = evaluate(y_true, pred_pers)
    
    # 2. Climatology
    target_dts = val_dt + timedelta(days=h)
    pred_clim = np.array([climatology_map[to_doy365(t_dt)] for t_dt in target_dts])
    met_clim = evaluate(y_true, pred_clim)
    
    # 3. Ridge
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(random_state=RANDOM_SEED))
    ])
    grid = GridSearchCV(pipe, {"ridge__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}, cv=cv, scoring="neg_mean_absolute_error")
    grid.fit(X_train, train_df[f"target_h{h}"])
    pred_ridge = grid.predict(X_val)
    met_ridge = evaluate(y_true, pred_ridge)
    
    skill_clim = skill_vs_persistence(met_clim["MAE"], met_pers["MAE"])
    skill_ridge = skill_vs_persistence(met_ridge["MAE"], met_pers["MAE"])
    
    print(f"\n--- Horizon h={h} ---")
    print(f"  Persistence: MAE={met_pers['MAE']:.2f}, Bias={met_pers['bias']:.2f}, Within1={met_pers['within_one_category_accuracy']:.3f}, AlertRec={met_pers['alert_recall']:.3f}, Skill=0.000")
    print(f"  Climatology: MAE={met_clim['MAE']:.2f}, Bias={met_clim['bias']:.2f}, Within1={met_clim['within_one_category_accuracy']:.3f}, AlertRec={met_clim['alert_recall']:.3f}, Skill={skill_clim:.3f}")
    print(f"  Ridge (a={grid.best_params_['ridge__alpha']}): MAE={met_ridge['MAE']:.2f}, Bias={met_ridge['bias']:.2f}, Within1={met_ridge['within_one_category_accuracy']:.3f}, AlertRec={met_ridge['alert_recall']:.3f}, Skill={skill_ridge:.3f}")
