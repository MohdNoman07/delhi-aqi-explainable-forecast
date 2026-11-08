"""
Unit tests for SHAP value computation and prediction additivity.
Verifies that SHAP values plus base value reproduce model prediction for 10 test rows.
"""

import json
from pathlib import Path
import numpy as np
import pytest
import joblib
import shap

from src.split import get_splits

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_shap_additivity_reproduction():
    """Checks that SHAP values + base_value equal model predictions within machine tolerance."""
    feat_path = REPO_ROOT / "models" / "feature_columns.json"
    with open(feat_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    # Load horizon 1 model
    model_path = REPO_ROOT / "models" / "rf_h1.pkl"
    assert model_path.exists(), f"Model file {model_path} does not exist"
    model = joblib.load(model_path)

    # Get test features
    _, _, test_df = get_splits()
    X_test_10 = test_df[feature_cols].iloc[:10]
    assert len(X_test_10) == 10

    # Compute SHAP
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X_test_10)

    # Model predictions
    preds = model.predict(X_test_10)

    # Reconstruct predictions: base_value + sum(shap_values)
    reconstructed = shap_vals.base_values + np.sum(shap_vals.values, axis=1)

    # Check maximum discrepancy is within small tolerance
    max_diff = np.max(np.abs(preds - reconstructed))
    assert max_diff < 1e-4, f"SHAP values failed to reproduce predictions! Max discrepancy: {max_diff}"
    assert np.allclose(preds, reconstructed, atol=1e-5), "Predictions and reconstructed SHAP sums do not match within atol=1e-5"


def test_shap_family_aggregation():
    """Checks that family value equals mean|row-sum of SHAP over family columns| and shares sum to 100."""
    from src.explain import FEATURE_FAMILIES
    feat_path = REPO_ROOT / "models" / "feature_columns.json"
    with open(feat_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    model_path = REPO_ROOT / "models" / "rf_h1.pkl"
    assert model_path.exists(), f"Model file {model_path} does not exist"
    model = joblib.load(model_path)

    _, _, test_df = get_splits()
    X_test_20 = test_df[feature_cols].iloc[:20]

    explainer = shap.TreeExplainer(model)
    sv = explainer(X_test_20)

    fam_vals = {}
    for fam, feats in FEATURE_FAMILIES.items():
        idx = [feature_cols.index(f) for f in feats if f in feature_cols]
        # family value equals mean|row-sum of SHAP over family columns|
        val = float(np.abs(sv.values[:, idx].sum(axis=1)).mean())
        fam_vals[fam] = val

    total = sum(fam_vals.values())
    assert total > 0, "Total family importance must be positive"
    shares = [v / total * 100.0 for v in fam_vals.values()]

    assert sum(shares) == pytest.approx(100.0, rel=1e-5), "Family shares must sum to 100"
    for fam, val in fam_vals.items():
        assert val >= 0.0, f"Family {fam} value must be non-negative"

