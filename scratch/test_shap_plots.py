import sys
from pathlib import Path
REPO_ROOT = Path("D:/ML_Project")
sys.path.insert(0, str(REPO_ROOT))

import json
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd

from src.split import get_splits

train_df, val_df, test_df = get_splits()
with open(REPO_ROOT / "models" / "feature_columns.json", "r") as f:
    feature_cols = json.load(f)

X_test = test_df[feature_cols].iloc[:50]
rf_h1 = joblib.load(REPO_ROOT / "models" / "rf_h1.pkl")
explainer = shap.TreeExplainer(rf_h1)
shap_values = explainer(X_test)

# Test beeswarm
plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_values, max_display=10, show=False)
plt.title("Test Beeswarm h=1", fontsize=12)
plt.savefig(REPO_ROOT / "scratch" / "test_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

# Test bar
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values, max_display=10, show=False)
plt.title("Test Bar h=1", fontsize=12)
plt.savefig(REPO_ROOT / "scratch" / "test_bar.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved test plots successfully!")
