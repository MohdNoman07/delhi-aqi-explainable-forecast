import pandas as pd
import numpy as np

test_preds = pd.read_csv("outputs/test_predictions.csv")
test_preds["date"] = pd.to_datetime(test_preds["date"])
test_preds["month"] = test_preds["date"].dt.month
test_preds["is_winter"] = test_preds["month"].isin([10, 11, 12, 1, 2])

rows = []
for h in range(1, 6):
    df_h = test_preds[test_preds["horizon"] == h].copy()
    y_true = df_h["truth"].values
    y_pred_rf = df_h["pred_random_forest"].values
    
    true_alerts = (y_true > 200)
    pred_alerts_rf = (y_pred_rf > 200)
    
    # Random Forest full test set
    tp_rf = (true_alerts & pred_alerts_rf).sum()
    fp_rf = (~true_alerts & pred_alerts_rf).sum()
    fn_rf = (true_alerts & ~pred_alerts_rf).sum()
    rf_prec = tp_rf / (tp_rf + fp_rf)
    rf_rec = tp_rf / (tp_rf + fn_rf)
    
    # Trivial rule across full test set: alert on every winter day, never in non-winter
    # pred_alerts_triv = df_h["is_winter"]
    pred_alerts_triv = df_h["is_winter"].values
    tp_triv = (true_alerts & pred_alerts_triv).sum()
    fp_triv = (~true_alerts & pred_alerts_triv).sum()
    fn_triv = (true_alerts & ~pred_alerts_triv).sum()
    triv_prec = tp_triv / (tp_triv + fp_triv)
    triv_rec = tp_triv / (tp_triv + fn_triv)
    
    # Random Forest non-winter alert recall
    non_winter_mask = ~df_h["is_winter"].values
    nw_true_alerts = true_alerts & non_winter_mask
    nw_pred_rf = pred_alerts_rf & non_winter_mask
    tp_nw = (nw_true_alerts & nw_pred_rf).sum()
    fn_nw = (nw_true_alerts & ~nw_pred_rf).sum()
    rf_nw_rec = tp_nw / (tp_nw + fn_nw) if (tp_nw + fn_nw) > 0 else 0.0
    
    rows.append({
        "horizon": h,
        "rf_alert_precision": rf_prec,
        "rf_alert_recall": rf_rec,
        "trivial_alert_precision": triv_prec,
        "trivial_alert_recall": triv_rec,
        "rf_non_winter_alert_recall": rf_nw_rec,
    })

res_df = pd.DataFrame(rows)
print(res_df.to_string(index=False))
