import sys
from pathlib import Path
REPO_ROOT = Path("D:/ML_Project")
sys.path.insert(0, str(REPO_ROOT))

import json
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.split import get_splits

train_df, val_df, test_df = get_splits()
with open(REPO_ROOT / "models" / "feature_columns.json", "r") as f:
    feature_cols = json.load(f)

rf_h1 = joblib.load(REPO_ROOT / "models" / "rf_h1.pkl")
explainer = shap.TreeExplainer(rf_h1)

X_test = test_df[feature_cols]
print(f"X_test shape: {X_test.shape}")

# Test explanation on 10 rows
X_sample = X_test.iloc[:10]
shap_values = explainer(X_sample)

print("shap_values shape:", shap_values.shape)
print("base_values:", shap_values.base_values[:3])
print("values sum:", shap_values.values.sum(axis=1)[:3])
preds = rf_h1.predict(X_sample)
print("preds:", preds[:3])
reconstructed = shap_values.base_values + shap_values.values.sum(axis=1)
print("reconstructed:", reconstructed[:3])
max_diff = np.max(np.abs(preds - reconstructed))
print("max_diff between model prediction and SHAP sum:", max_diff)

# Test waterfall plot
plt.figure()
shap.plots.waterfall(shap_values[0], show=False)
plt.tight_layout()
plt.savefig(REPO_ROOT / "scratch" / "test_waterfall.png", dpi=150)
plt.close()
print("Saved test waterfall plot successfully!")
