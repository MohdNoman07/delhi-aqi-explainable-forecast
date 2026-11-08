import sys
from pathlib import Path
sys.path.insert(0, str(Path("D:/ML_Project")))
import pandas as pd
import numpy as np

test_preds = pd.read_csv("outputs/test_predictions.csv")
test_preds["date"] = pd.to_datetime(test_preds["date"])
test_preds["month"] = test_preds["date"].dt.month
test_preds["is_winter"] = test_preds["month"].isin([10, 11, 12, 1, 2])

print("--- Alert Base Rate and Trivial Rule Performance ---")
for h in range(1, 6):
    df_h = test_preds[test_preds["horizon"] == h].copy()
    winter_mask = df_h["is_winter"]
    non_winter_mask = ~df_h["is_winter"]
    
    y_true = df_h["truth"].values
    actual_alerts = (y_true > 200)
    
    n_winter = winter_mask.sum()
    n_non_winter = non_winter_mask.sum()
    
    winter_alerts = (df_h.loc[winter_mask, "truth"] > 200).sum()
    non_winter_alerts = (df_h.loc[non_winter_mask, "truth"] > 200).sum()
    
    winter_base_rate = winter_alerts / n_winter
    non_winter_base_rate = non_winter_alerts / n_non_winter
    overall_base_rate = actual_alerts.sum() / len(df_h)
    
    # Trivial rule: alert on every winter day, never in non-winter
    # Across entire test set:
    # Pred alert = winter_mask
    tp = (winter_mask & actual_alerts).sum()
    fp = (winter_mask & ~actual_alerts).sum()
    fn = (non_winter_mask & actual_alerts).sum()
    
    trivial_prec_overall = tp / (tp + fp) if (tp + fp) > 0 else 0
    trivial_rec_overall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nHorizon h={h}:")
    print(f"  Winter (N={n_winter}): {winter_alerts} alerts, base rate = {winter_base_rate:.4f} ({winter_base_rate*100:.2f}%)")
    print(f"  Non-Winter (N={n_non_winter}): {non_winter_alerts} alerts, base rate = {non_winter_base_rate:.4f} ({non_winter_base_rate*100:.2f}%)")
    print(f"  Overall: {actual_alerts.sum()} alerts, base rate = {overall_base_rate:.4f}")
    print(f"  Trivial rule (Overall): Alert Precision = {trivial_prec_overall:.4f}, Alert Recall = {trivial_rec_overall:.4f}")
    print(f"  Trivial rule in Winter: Alert Precision = {winter_base_rate:.4f}, Alert Recall = 1.0000")
    print(f"  Trivial rule in Non-Winter: Alert Precision = 0.0000, Alert Recall = 0.0000")
