import sys
from pathlib import Path
REPO_ROOT = Path("D:/ML_Project")
sys.path.insert(0, str(REPO_ROOT))

import json
import joblib
import shap
import time
import pandas as pd

from src.split import get_splits

train_df, val_df, test_df = get_splits()
with open(REPO_ROOT / "models" / "feature_columns.json", "r") as f:
    feature_cols = json.load(f)

X_test = test_df[feature_cols]

t0 = time.time()
rf_h1 = joblib.load(REPO_ROOT / "models" / "rf_h1.pkl")
explainer = shap.TreeExplainer(rf_h1)
shap_values = explainer(X_test)
t1 = time.time()
print(f"Computed SHAP on 887 rows in {t1 - t0:.2f} seconds")
