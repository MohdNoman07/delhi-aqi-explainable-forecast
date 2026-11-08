"""
Unit tests for CPCB-style NAQI calculation.
"""

import math
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.compute_aqi import sub_index, compute_aqi_from_subs, get_aqi_category


def test_hand_checked_sub_indices():
    # PM2.5 = 30 gives 50
    assert math.isclose(sub_index("pm25", 30.0), 50.0, rel_tol=1e-5)
    
    # PM2.5 = 90 gives 200
    assert math.isclose(sub_index("pm25", 90.0), 200.0, rel_tol=1e-5)
    
    # PM10 = 100 gives 100
    assert math.isclose(sub_index("pm10", 100.0), 100.0, rel_tol=1e-5)


def test_severe_band():
    # PM2.5 = 250 gives 400
    assert math.isclose(sub_index("pm25", 250.0), 400.0, rel_tol=1e-5)
    
    # PM2.5 = 380 gives 500
    assert math.isclose(sub_index("pm25", 380.0), 500.0, rel_tol=1e-5)
    
    # PM10 = 430 gives 400
    assert math.isclose(sub_index("pm10", 430.0), 400.0, rel_tol=1e-5)
    
    # CO = 34 gives 400
    assert math.isclose(sub_index("co", 34.0), 400.0, rel_tol=1e-5)


def test_pm25_500_clipping():
    # PM2.5 = 500 sub-index is computed > 500 internally
    si_500 = sub_index("pm25", 500.0)
    assert si_500 > 500.0
    
    # But final AQI gives 500 after clipping
    subs = {"pm25": si_500, "no2": 50.0, "co": 50.0}
    aqi, prominent = compute_aqi_from_subs(subs)
    assert aqi == 500.0
    assert prominent == "pm25"
    assert get_aqi_category(aqi) == "Severe"


def test_two_pollutants_gives_nan():
    # Day with only 2 pollutants gives NaN
    subs = {"pm25": 50.0, "pm10": 100.0}
    aqi, prominent = compute_aqi_from_subs(subs)
    assert math.isnan(aqi)
    assert prominent is None


def test_three_pollutants_without_pm_gives_nan():
    # 3 pollutants available, but neither PM2.5 nor PM10 present -> must give NaN
    subs = {"no2": 60.0, "so2": 45.0, "co": 80.0}
    aqi, prominent = compute_aqi_from_subs(subs)
    assert math.isnan(aqi)
    assert prominent is None


def test_valid_three_pollutants_with_pm():
    # 3 pollutants including PM2.5 -> valid AQI is max sub-index (clipped to 500)
    subs = {"pm25": 150.0, "no2": 60.0, "co": 80.0}
    aqi, prominent = compute_aqi_from_subs(subs)
    assert math.isclose(aqi, 150.0, rel_tol=1e-5)
    assert prominent == "pm25"


def test_aqi_categories():
    assert get_aqi_category(30.0) == "Good"
    assert get_aqi_category(50.0) == "Good"
    assert get_aqi_category(51.0) == "Satisfactory"
    assert get_aqi_category(100.0) == "Satisfactory"
    assert get_aqi_category(150.0) == "Moderate"
    assert get_aqi_category(200.0) == "Moderate"
    assert get_aqi_category(250.0) == "Poor"
    assert get_aqi_category(300.0) == "Poor"
    assert get_aqi_category(350.0) == "Very Poor"
    assert get_aqi_category(400.0) == "Very Poor"
    assert get_aqi_category(401.0) == "Severe"
    assert get_aqi_category(500.0) == "Severe"
    assert get_aqi_category(float("nan")) is None
