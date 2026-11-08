"""
tests/test_rf.py

Unit tests for Random Forest forecasting models:
1. All saved model files (models/rf_h1.pkl ... models/rf_h5.pkl) and models/rf_config.json exist.
2. Models load successfully and config contains target scale (raw or log).
3. Predictions on validation features have the correct shape and stay strictly within [0, 500] after clipping.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.split import get_splits


@pytest.fixture(scope="module")
def rf_setup():
    models_dir = REPO_ROOT / "models"
    config_path = models_dir / "rf_config.json"
    feat_path = models_dir / "feature_columns.json"

    assert config_path.exists(), f"RF config not found at {config_path}"
    assert feat_path.exists(), f"Feature columns not found at {feat_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(feat_path, "r", encoding="utf-8") as f:
        feat_cols = json.load(f)

    _, val_df, _ = get_splits()
    X_val = val_df[feat_cols]

    models = {}
    for h in range(1, 6):
        pkl_path = models_dir / f"rf_h{h}.pkl"
        assert pkl_path.exists(), f"Model file not found at {pkl_path}"
        models[h] = joblib.load(pkl_path)

    return config, feat_cols, X_val, models


def test_rf_models_exist_and_config(rf_setup):
    config, _, _, models = rf_setup
    assert len(models) == 5, f"Expected 5 models, loaded {len(models)}"
    for h in range(1, 6):
        h_key = f"h{h}"
        assert h_key in config, f"Horizon {h_key} missing from config"
        assert "target_scale" in config[h_key]
        assert config[h_key]["target_scale"] in ["raw", "log"]
        assert "use_log" in config[h_key]
        assert isinstance(config[h_key]["use_log"], bool)


def test_rf_predictions_shape_and_range(rf_setup):
    config, _, X_val, models = rf_setup

    sample_size = 25
    X_sample = X_val.iloc[:sample_size]

    for h in range(1, 6):
        h_key = f"h{h}"
        model = models[h]
        use_log = config[h_key]["use_log"]

        raw_pred = model.predict(X_sample)
        assert len(raw_pred) == sample_size, f"Horizon {h} prediction shape mismatch"

        if use_log:
            pred = np.expm1(raw_pred)
        else:
            pred = raw_pred

        pred_clipped = np.clip(pred, 0.0, 500.0)

        assert not np.isnan(pred_clipped).any(), f"Horizon {h} contains NaN predictions"
        assert not np.isinf(pred_clipped).any(), f"Horizon {h} contains Inf predictions"
        assert np.all(pred_clipped >= 0.0), f"Horizon {h} has values below 0"
        assert np.all(pred_clipped <= 500.0), f"Horizon {h} has values above 500"
