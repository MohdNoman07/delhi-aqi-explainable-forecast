"""
tests/test_no_leakage.py

Verification tests to ensure strictly zero data leakage in feature engineering:
1. No column starting with 'target_' in feature list.
2. No negative shifts or centered rolling windows in compute_features.
3. For 20 random dates t (seed 42), features computed using only data up to and including t
   must exactly match the stored model-ready features for date t.
"""

import sys
import json
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features import compute_features
from src.config import RANDOM_SEED


def test_no_target_in_feature_columns():
    """Ensure no target column is present in the feature definitions."""
    feat_json = REPO_ROOT / "models" / "feature_columns.json"
    assert feat_json.exists(), f"Feature columns file not found at {feat_json}"
    
    with open(feat_json, "r", encoding="utf-8") as f:
        features = json.load(f)
    
    assert len(features) == 38, f"Expected 38 features, got {len(features)}"
    for col in features:
        assert not col.startswith("target_"), f"Target column '{col}' found in feature list!"
        assert "target" not in col.lower(), f"Suspicious column '{col}' found in feature list!"


def test_no_negative_shifts_or_centered_windows():
    """Ensure compute_features does not use negative shifts or centered rolling windows."""
    src = inspect.getsource(compute_features)
    
    # Check for negative shift
    assert ".shift(-" not in src, "Found negative shift (.shift(-...)) in compute_features!"
    assert "center=True" not in src, "Found center=True in compute_features rolling calls!"


def test_temporal_leakage_20_random_dates():
    """
    Sample 20 random dates t from model_ready.csv using RANDOM_SEED.
    Recompute features using ONLY data up to and including t (no future data available).
    Assert that the recomputed features for date t exactly match model_ready.csv.
    """
    model_ready_path = REPO_ROOT / "data" / "processed" / "model_ready.csv"
    daily_path = REPO_ROOT / "data" / "processed" / "clean_daily.csv"
    feat_json = REPO_ROOT / "models" / "feature_columns.json"
    
    assert model_ready_path.exists(), f"Model ready data not found at {model_ready_path}"
    assert daily_path.exists(), f"Clean daily data not found at {daily_path}"
    
    df_model = pd.read_csv(model_ready_path)
    df_daily = pd.read_csv(daily_path)
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    
    with open(feat_json, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    
    # Sample 20 random dates with fixed seed 42
    rng = np.random.default_rng(RANDOM_SEED)
    all_dates = df_model["date"].values
    sample_dates = rng.choice(all_dates, size=20, replace=False)
    
    for date_str in sample_dates:
        t = pd.to_datetime(date_str)
        # Strictly truncate daily data to <= t
        daily_past = df_daily[df_daily["date"] <= t].copy()
        
        # Complete daily calendar up to t
        full_idx_past = pd.date_range("2018-01-01", t, freq="D")
        daily_past_full = daily_past.set_index("date").reindex(full_idx_past)
        daily_past_full.index.name = "date"
        
        # Compute features using only past data
        feats_past = compute_features(daily_past_full)
        recomputed_t = feats_past.loc[t, feature_cols].astype(float)
        
        # Expected features from model_ready.csv for date t
        expected_t = df_model.loc[df_model["date"] == date_str, feature_cols].iloc[0].astype(float)
        
        pd.testing.assert_series_equal(
            recomputed_t,
            expected_t,
            check_names=False,
            check_exact=False,
            rtol=1e-5,
            atol=1e-5,
            obj=f"Feature mismatch on date {date_str}"
        )
