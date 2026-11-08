"""
scratch/test_bootstrap_comparisons.py
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.final_evaluate import block_bootstrap_ci

test_preds = pd.read_csv("outputs/test_predictions.csv")
test_preds["date"] = pd.to_datetime(test_preds["date"])
test_preds["month"] = test_preds["date"].dt.month
test_preds["is_winter"] = test_preds["month"].isin([10, 11, 12, 1, 2])

print("=== 1. Random Forest vs Ridge (Full Test Set) ===")
rf_ridge_records = []
for h in range(1, 6):
    df_h = test_preds[test_preds["horizon"] == h].copy()
    err_rf = (df_h["truth"] - df_h["pred_random_forest"]).abs().values
    err_ridge = (df_h["truth"] - df_h["pred_ridge"]).abs().values
    diff = err_rf - err_ridge # negative means RF is better
    
    mean_d, lo, hi = block_bootstrap_ci(diff, block_size=7, n_boot=2000, seed=42)
    rf_better = hi < 0.0
    includes_0 = (lo <= 0.0 <= hi)
    rf_ridge_records.append({
        "horizon": h,
        "rf_mae": float(np.mean(err_rf)),
        "ridge_mae": float(np.mean(err_ridge)),
        "mean_diff_mae": mean_d,
        "ci_lower_95": lo,
        "ci_upper_95": hi,
        "rf_clearly_better": rf_better,
        "includes_zero": includes_0
    })
    print(f"h={h}: RF MAE={np.mean(err_rf):.2f}, Ridge MAE={np.mean(err_ridge):.2f} | Diff={mean_d:.2f} [{lo:.2f}, {hi:.2f}] | RF better: {rf_better} | Zero in CI: {includes_0}")

print("\n=== 2. Random Forest vs Best Baseline (Winter vs Non-Winter) ===")
# Best baseline: h=1 Persistence, h=2..5 Climatology
winter_split_records = []
for h in range(1, 6):
    df_h = test_preds[test_preds["horizon"] == h].copy()
    base_col = "pred_persistence" if h == 1 else "pred_climatology"
    base_name = "Persistence" if h == 1 else "Climatology"
    
    for season, mask in [("Winter", df_h["is_winter"]), ("Non-Winter", ~df_h["is_winter"])]:
        sub = df_h[mask].copy()
        err_rf = (sub["truth"] - sub["pred_random_forest"]).abs().values
        err_base = (sub["truth"] - sub[base_col]).abs().values
        diff = err_rf - err_base
        
        mean_d, lo, hi = block_bootstrap_ci(diff, block_size=7, n_boot=2000, seed=42)
        rf_better = hi < 0.0
        includes_0 = (lo <= 0.0 <= hi)
        winter_split_records.append({
            "horizon": h,
            "season": season,
            "best_baseline": base_name,
            "days_count": len(sub),
            "rf_mae": float(np.mean(err_rf)),
            "baseline_mae": float(np.mean(err_base)),
            "mean_diff_mae": mean_d,
            "ci_lower_95": lo,
            "ci_upper_95": hi,
            "rf_clearly_better": rf_better,
            "includes_zero": includes_0
        })
        print(f"h={h} ({season:10s}, N={len(sub)}): RF={np.mean(err_rf):.2f}, Base({base_name})={np.mean(err_base):.2f} | Diff={mean_d:.2f} [{lo:.2f}, {hi:.2f}] | RF better: {rf_better} | Zero in CI: {includes_0}")
