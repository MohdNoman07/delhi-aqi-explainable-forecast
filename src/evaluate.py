"""
src/evaluate.py

Evaluation metrics for Delhi Air Quality forecasting models.
Provides:
  - evaluate(y_true, y_pred): comprehensive regression and classification metrics
  - per_class_report(y_true, y_pred): precision, recall, F1, support per AQI category
  - skill_vs_persistence(mae_model, mae_persistence): fractional skill score
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

AQI_CATEGORIES_6 = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
AQI_CATEGORIES_5 = ["Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def get_aqi_category_6(aqi_val: float) -> str:
    """
    Standard 6-class CPCB NAQI category.
    Boundaries:
      Good:          0 - 50
      Satisfactory: 51 - 100
      Moderate:    101 - 200
      Poor:        201 - 300
      Very Poor:   301 - 400
      Severe:      401 - 500
    """
    if aqi_val <= 50.0:
        return "Good"
    elif aqi_val <= 100.0:
        return "Satisfactory"
    elif aqi_val <= 200.0:
        return "Moderate"
    elif aqi_val <= 300.0:
        return "Poor"
    elif aqi_val <= 400.0:
        return "Very Poor"
    else:
        return "Severe"


def get_aqi_category_5(aqi_val: float) -> str:
    """
    5-class category where Good (0-50) is merged into Satisfactory (0-100).
    """
    if aqi_val <= 100.0:
        return "Satisfactory"
    elif aqi_val <= 200.0:
        return "Moderate"
    elif aqi_val <= 300.0:
        return "Poor"
    elif aqi_val <= 400.0:
        return "Very Poor"
    else:
        return "Severe"


def get_category_index_6(cat_name: str) -> int:
    return AQI_CATEGORIES_6.index(cat_name)


def evaluate(y_true, y_pred) -> dict:
    """
    Computes regression, classification, alert, and 5-class macro metrics.
    Automatically clips y_pred to [0, 500].
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.clip(np.asarray(y_pred, dtype=float), 0.0, 500.0)

    # 1. Regression Metrics
    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))
    bias = float(np.mean(y_p - y_t))

    # 2. 6-Class Categorization
    cats_true_6 = [get_aqi_category_6(val) for val in y_t]
    cats_pred_6 = [get_aqi_category_6(val) for val in y_p]

    acc = float(accuracy_score(cats_true_6, cats_pred_6))

    idx_true_6 = np.array([get_category_index_6(c) for c in cats_true_6])
    idx_pred_6 = np.array([get_category_index_6(c) for c in cats_pred_6])
    within_one = float(np.mean(np.abs(idx_pred_6 - idx_true_6) <= 1))

    macro_prec = float(precision_score(cats_true_6, cats_pred_6, labels=AQI_CATEGORIES_6, average="macro", zero_division=0))
    macro_rec  = float(recall_score(cats_true_6, cats_pred_6, labels=AQI_CATEGORIES_6, average="macro", zero_division=0))
    macro_f1   = float(f1_score(cats_true_6, cats_pred_6, labels=AQI_CATEGORIES_6, average="macro", zero_division=0))

    weighted_prec = float(precision_score(cats_true_6, cats_pred_6, labels=AQI_CATEGORIES_6, average="weighted", zero_division=0))
    weighted_rec  = float(recall_score(cats_true_6, cats_pred_6, labels=AQI_CATEGORIES_6, average="weighted", zero_division=0))
    weighted_f1   = float(f1_score(cats_true_6, cats_pred_6, labels=AQI_CATEGORIES_6, average="weighted", zero_division=0))

    # 3. Alert Metrics (Alert: AQI > 200, i.e., Poor, Very Poor, Severe)
    alert_true = (y_t > 200.0).astype(int)
    alert_pred = (y_p > 200.0).astype(int)

    alert_prec = float(precision_score(alert_true, alert_pred, pos_label=1, zero_division=0))
    alert_rec  = float(recall_score(alert_true, alert_pred, pos_label=1, zero_division=0))

    # 4. 5-Class Macro Metrics (Good merged into Satisfactory)
    cats_true_5 = [get_aqi_category_5(val) for val in y_t]
    cats_pred_5 = [get_aqi_category_5(val) for val in y_p]

    macro_prec_5class = float(precision_score(cats_true_5, cats_pred_5, labels=AQI_CATEGORIES_5, average="macro", zero_division=0))
    macro_rec_5class  = float(recall_score(cats_true_5, cats_pred_5, labels=AQI_CATEGORIES_5, average="macro", zero_division=0))
    macro_f1_5class   = float(f1_score(cats_true_5, cats_pred_5, labels=AQI_CATEGORIES_5, average="macro", zero_division=0))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "bias": bias,
        "accuracy": acc,
        "within_one_category_accuracy": within_one,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_prec,
        "weighted_recall": weighted_rec,
        "weighted_f1": weighted_f1,
        "alert_precision": alert_prec,
        "alert_recall": alert_rec,
        "macro_precision_5class": macro_prec_5class,
        "macro_recall_5class": macro_rec_5class,
        "macro_f1_5class": macro_f1_5class,
    }


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    """
    Returns DataFrame with precision, recall, F1, and support for each of the 6 AQI classes.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.clip(np.asarray(y_pred, dtype=float), 0.0, 500.0)

    cats_true = [get_aqi_category_6(val) for val in y_t]
    cats_pred = [get_aqi_category_6(val) for val in y_p]

    records = []
    for cat in AQI_CATEGORIES_6:
        support = sum(1 for c in cats_true if c == cat)
        b_true = [1 if c == cat else 0 for c in cats_true]
        b_pred = [1 if c == cat else 0 for c in cats_pred]

        prec = float(precision_score(b_true, b_pred, pos_label=1, zero_division=0))
        rec  = float(recall_score(b_true, b_pred, pos_label=1, zero_division=0))
        f1   = float(f1_score(b_true, b_pred, pos_label=1, zero_division=0))

        records.append({
            "category": cat,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
        })

    return pd.DataFrame(records)


def skill_vs_persistence(mae_model: float, mae_persistence: float) -> float:
    """Computes fractional skill score relative to persistence: 1 - mae_model / mae_persistence."""
    if mae_persistence == 0.0:
        return 0.0
    return float(1.0 - (mae_model / mae_persistence))


def skill_vs_best_baseline(mae_model: float, mae_persistence: float, mae_climatology: float) -> float:
    """
    Computes fractional skill score relative to the best baseline (min of persistence and climatology):
    1 - MAE_model / min(MAE_persistence, MAE_climatology).
    """
    best_base = min(mae_persistence, mae_climatology)
    if best_base == 0.0:
        return 0.0
    return float(1.0 - (mae_model / best_base))

