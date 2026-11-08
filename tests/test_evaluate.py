"""
tests/test_evaluate.py

Unit tests for src/evaluate.py:
1. Perfect predictions give MAE 0, RMSE 0, R2 1, bias 0, and accuracy 1.
2. Constant offset gives the exact expected MAE and bias.
3. Category boundaries 50 / 51 / 200 / 201 map correctly.
4. Skill score vs persistence behaves correctly.
5. Per-class report contains all categories and correct supports.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import (
    evaluate,
    per_class_report,
    skill_vs_persistence,
    skill_vs_best_baseline,
    get_aqi_category_6,
    get_aqi_category_5,
)


def test_perfect_predictions():
    y_true = np.array([35.0, 75.0, 150.0, 250.0, 350.0, 450.0])
    y_pred = np.array([35.0, 75.0, 150.0, 250.0, 350.0, 450.0])

    metrics = evaluate(y_true, y_pred)
    assert metrics["MAE"] == pytest.approx(0.0)
    assert metrics["RMSE"] == pytest.approx(0.0)
    assert metrics["R2"] == pytest.approx(1.0)
    assert metrics["bias"] == pytest.approx(0.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["within_one_category_accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["alert_precision"] == pytest.approx(1.0)
    assert metrics["alert_recall"] == pytest.approx(1.0)
    assert metrics["macro_f1_5class"] == pytest.approx(1.0)


def test_constant_offset():
    y_true = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    offset = 12.5
    y_pred = y_true + offset

    metrics = evaluate(y_true, y_pred)
    assert metrics["MAE"] == pytest.approx(offset)
    assert metrics["bias"] == pytest.approx(offset)


def test_category_boundary_mapping():
    # 50 -> Good, 51 -> Satisfactory
    assert get_aqi_category_6(50.0) == "Good"
    assert get_aqi_category_6(51.0) == "Satisfactory"

    # 100 -> Satisfactory, 101 -> Moderate
    assert get_aqi_category_6(100.0) == "Satisfactory"
    assert get_aqi_category_6(101.0) == "Moderate"

    # 200 -> Moderate, 201 -> Poor
    assert get_aqi_category_6(200.0) == "Moderate"
    assert get_aqi_category_6(201.0) == "Poor"

    # 300 -> Poor, 301 -> Very Poor
    assert get_aqi_category_6(300.0) == "Poor"
    assert get_aqi_category_6(301.0) == "Very Poor"

    # 400 -> Very Poor, 401 -> Severe
    assert get_aqi_category_6(400.0) == "Very Poor"
    assert get_aqi_category_6(401.0) == "Severe"


def test_clipping_above_500():
    y_true = np.array([450.0])
    y_pred = np.array([600.0])  # Should be clipped to 500

    metrics = evaluate(y_true, y_pred)
    assert metrics["bias"] == pytest.approx(50.0)
    assert metrics["MAE"] == pytest.approx(50.0)


def test_skill_vs_persistence():
    assert skill_vs_persistence(mae_model=30.0, mae_persistence=40.0) == pytest.approx(0.25)
    assert skill_vs_persistence(mae_model=50.0, mae_persistence=40.0) == pytest.approx(-0.25)
    assert skill_vs_persistence(mae_model=40.0, mae_persistence=40.0) == pytest.approx(0.0)


def test_skill_vs_best_baseline():
    # If persistence is 50 and climatology is 40, best baseline is 40
    # Model MAE = 30 -> skill = 1 - 30/40 = 0.25
    assert skill_vs_best_baseline(mae_model=30.0, mae_persistence=50.0, mae_climatology=40.0) == pytest.approx(0.25)
    # If persistence is 30 and climatology is 40, best baseline is 30
    # Model MAE = 36 -> skill = 1 - 36/30 = -0.20
    assert skill_vs_best_baseline(mae_model=36.0, mae_persistence=30.0, mae_climatology=40.0) == pytest.approx(-0.20)



def test_per_class_report():
    y_true = [30.0, 80.0, 150.0, 250.0, 350.0, 450.0]
    y_pred = [30.0, 80.0, 150.0, 250.0, 350.0, 450.0]

    rep = per_class_report(y_true, y_pred)
    assert len(rep) == 6
    assert set(rep["category"]) == {"Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"}
    assert all(rep["support"] == 1)
    assert all(rep["f1"] == 1.0)


def test_damped_persistence_rho0_equals_climatology():
    """Verify that when rho=0, damped anomaly persistence predictions equal climatology."""
    import pandas as pd
    from src.train import compute_damped_persistence

    dates = pd.date_range("2020-01-01", periods=30)
    clim_map = {d: 120.0 for d in range(365)}

    # When target equals climatology, target anomaly is 0 -> rho = 0
    train_df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "aqi": np.linspace(80.0, 160.0, 30),
        "target_h1": np.full(30, 120.0)  # equals climatology, so target anomaly is identically 0
    })
    eval_df = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=10).strftime("%Y-%m-%d"),
        "aqi": np.linspace(50.0, 300.0, 10)
    })

    pred, rho = compute_damped_persistence(train_df, eval_df, clim_map, h=1)
    assert rho == pytest.approx(0.0)

    # With rho=0, predictions must equal climatology for target dates (120.0)
    expected_clim = np.full(10, 120.0)
    np.testing.assert_allclose(pred, expected_clim)


def test_damped_persistence_no_test_dates_in_fit():
    """Verify that the fit uses only the dates it is given (no test dates in the fit)."""
    import pandas as pd
    from src.train import compute_damped_persistence

    dates_train = pd.date_range("2020-01-01", periods=40)
    train_df = pd.DataFrame({
        "date": dates_train.strftime("%Y-%m-%d"),
        "aqi": np.sin(np.linspace(0, 5, 40)) * 50 + 150,
        "target_h1": np.sin(np.linspace(0.1, 5.1, 40)) * 50 + 150
    })
    clim_map = {d: 150.0 for d in range(365)}

    # Eval dataset 1
    eval_df1 = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=15).strftime("%Y-%m-%d"),
        "aqi": np.full(15, 200.0)
    })
    pred1, rho1 = compute_damped_persistence(train_df, eval_df1, clim_map, h=1)

    # Eval dataset 2 with completely different dates and AQI
    eval_df2 = pd.DataFrame({
        "date": pd.date_range("2025-06-01", periods=20).strftime("%Y-%m-%d"),
        "aqi": np.full(20, 50.0)
    })
    pred2, rho2 = compute_damped_persistence(train_df, eval_df2, clim_map, h=1)

    # rho must be strictly identical regardless of eval_df
    assert rho1 == pytest.approx(rho2)
    assert rho1 != 0.0
