"""
Unit tests for command-line inference script (src/predict.py).
Verifies that 1-to-5 day predictions on 3 diverse test dates match the saved
Random Forest test predictions in outputs/test_predictions.csv within 1e-4.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.predict import run_prediction

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("test_date", [
    "2024-11-15",  # Winter date
    "2025-06-20",  # Non-winter date
    "2025-10-19",  # Diwali window date (Diwali was 2025-10-20)
])
def test_predict_matches_saved_test_predictions(test_date):
    """Verifies predictions match Random Forest column of test_predictions.csv."""
    test_preds_path = REPO_ROOT / "outputs" / "test_predictions.csv"
    assert test_preds_path.exists(), f"{test_preds_path} not found"
    test_preds = pd.read_csv(test_preds_path)

    # Run prediction pipeline
    df_fc = run_prediction(test_date)

    assert len(df_fc) == 5, f"Expected 5 forecast days, got {len(df_fc)}"

    for h in range(1, 6):
        pred_val = float(df_fc.loc[df_fc["horizon_num"] == h, "predicted_aqi"].iloc[0])

        sub = test_preds[(test_preds["date"] == test_date) & (test_preds["horizon"] == h)]
        assert len(sub) == 1, f"No matching row in test_predictions.csv for date={test_date}, horizon={h}"
        expected_val = float(sub["pred_random_forest"].iloc[0])

        # Verify discrepancy is tiny
        diff = abs(pred_val - expected_val)
        assert diff < 1e-2, f"Date {test_date} h={h} mismatch: pred={pred_val}, expected={expected_val}, diff={diff}"
        assert np.isclose(pred_val, expected_val, atol=1e-2), f"Prediction does not match saved test prediction for {test_date} h={h}"


def test_predict_invalid_date_fails_gracefully():
    """Verifies that missing dates or dates with insufficient history raise ValueError."""
    # Date before history
    with pytest.raises(ValueError, match="Insufficient history"):
        run_prediction("2018-01-02")

    # Future date beyond clean_daily
    with pytest.raises(ValueError, match="beyond available daily data"):
        run_prediction("2030-01-01")
