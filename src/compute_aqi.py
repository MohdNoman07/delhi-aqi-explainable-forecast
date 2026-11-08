"""
compute_aqi.py  –  CPCB-style NAQI (breakpoints pending verification by the project owner against the official CPCB report).

Sub-indices are computed for 7 criteria pollutants: PM2.5, PM10, NO2, SO2, CO, O3, NH3.

Calculation Rules:
  1. AQI is calculated only if data of at least 3 pollutants are available.
  2. At least one of the available pollutants MUST be either PM2.5 or PM10.
  3. Prominent pollutant = pollutant with the highest sub-index.
  4. Final AQI is clipped to the range 0-500 (sub-indices may be computed above 500 internally,
     but the final AQI and category use the clipped value).

Concentrations:
  - CO is in mg/m³ (8-hour trailing rolling mean, daily maximum).
  - O3 is in µg/m³ (8-hour trailing rolling mean, daily maximum).
  - PM2.5, PM10, NO2, SO2, NH3 are in µg/m³ (24-hour daily mean, requiring >=16 valid hours).
"""

import math
from typing import Dict, Optional, Tuple
import numpy as np

# ── Breakpoint Tables (B_lo, B_hi, I_lo, I_hi) ────────────────────────────────
# All bands are continuous floats to avoid gaps at integer boundaries.
# Breakpoints pending verification by the project owner against the official CPCB report.

_BREAKPOINTS = {
    "pm25": [       # 24-h avg, µg/m³
        (0.0,    30.0,    0.0,   50.0),   # Good
        (30.0,   60.0,   50.0,  100.0),   # Satisfactory
        (60.0,   90.0,  100.0,  200.0),   # Moderate
        (90.0,  120.0,  200.0,  300.0),   # Poor
        (120.0, 250.0,  300.0,  400.0),   # Very Poor
        (250.0, 380.0,  400.0,  500.0),   # Severe (250+)
    ],
    "pm10": [       # 24-h avg, µg/m³
        (0.0,    50.0,    0.0,   50.0),   # Good
        (50.0,  100.0,   50.0,  100.0),   # Satisfactory
        (100.0, 250.0,  100.0,  200.0),   # Moderate
        (250.0, 350.0,  200.0,  300.0),   # Poor
        (350.0, 430.0,  300.0,  400.0),   # Very Poor
        (430.0, 510.0,  400.0,  500.0),   # Severe (430+)
    ],
    "no2": [        # 24-h avg, µg/m³
        (0.0,    40.0,    0.0,   50.0),   # Good
        (40.0,   80.0,   50.0,  100.0),   # Satisfactory
        (80.0,  180.0,  100.0,  200.0),   # Moderate
        (180.0, 280.0,  200.0,  300.0),   # Poor
        (280.0, 400.0,  300.0,  400.0),   # Very Poor
        (400.0, 520.0,  400.0,  500.0),   # Severe (400+)
    ],
    "so2": [        # 24-h avg, µg/m³
        (0.0,    40.0,    0.0,   50.0),   # Good
        (40.0,   80.0,   50.0,  100.0),   # Satisfactory
        (80.0,  380.0,  100.0,  200.0),   # Moderate
        (380.0, 800.0,  200.0,  300.0),   # Poor
        (800.0, 1600.0, 300.0,  400.0),   # Very Poor
        (1600.0, 2400.0, 400.0, 500.0),  # Severe (1600+)
    ],
    "co": [         # 8-h max avg, mg/m³
        (0.0,    1.0,    0.0,   50.0),    # Good
        (1.0,    2.0,   50.0,  100.0),    # Satisfactory
        (2.0,   10.0,  100.0,  200.0),    # Moderate
        (10.0,  17.0,  200.0,  300.0),    # Poor
        (17.0,  34.0,  300.0,  400.0),    # Very Poor
        (34.0,  51.0,  400.0,  500.0),    # Severe (34+)
    ],
    "o3": [         # 8-h max avg, µg/m³
        (0.0,    50.0,    0.0,   50.0),    # Good
        (50.0,  100.0,   50.0,  100.0),    # Satisfactory
        (100.0, 168.0,  100.0,  200.0),   # Moderate
        (168.0, 208.0,  200.0,  300.0),   # Poor
        (208.0, 748.0,  300.0,  400.0),   # Very Poor
        (748.0, 1288.0, 400.0,  500.0),   # Severe (748+)
    ],
    "nh3": [        # 24-h avg, µg/m³
        (0.0,    200.0,   0.0,   50.0),   # Good
        (200.0,  400.0,  50.0,  100.0),   # Satisfactory
        (400.0,  800.0, 100.0,  200.0),   # Moderate
        (800.0, 1200.0, 200.0,  300.0),   # Poor
        (1200.0, 1800.0, 300.0, 400.0),   # Very Poor
        (1800.0, 2400.0, 400.0, 500.0),   # Severe (1800+)
    ],
}

AQI_POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3"]


def sub_index(pollutant: str, concentration: float) -> float:
    """
    Compute the CPCB-style sub-index for `pollutant` at given `concentration`.
    Returns NaN if concentration is missing or negative.
    Sub-indices may be computed above 500 internally by linear extrapolation
    using the Severe band slope.
    """
    if pollutant not in _BREAKPOINTS:
        return float("nan")
    if concentration is None or (isinstance(concentration, float) and math.isnan(concentration)):
        return float("nan")
    if concentration < 0:
        return float("nan")

    for b_lo, b_hi, i_lo, i_hi in _BREAKPOINTS[pollutant]:
        if b_lo <= concentration <= b_hi:
            if b_hi == b_lo:
                return i_lo
            return i_lo + (concentration - b_lo) * (i_hi - i_lo) / (b_hi - b_lo)

    # Beyond upper breakpoint of Severe: extrapolate using the Severe band slope
    b_lo, b_hi, i_lo, i_hi = _BREAKPOINTS[pollutant][-1]
    return i_lo + (concentration - b_lo) * (i_hi - i_lo) / (b_hi - b_lo)


def compute_aqi_from_subs(sub_indices: Dict[str, float]) -> Tuple[float, Optional[str]]:
    """
    Given {pollutant: sub_index, ...}, returns (aqi, prominent_pollutant).
    Requires at least 3 valid pollutants, and at least one must be PM2.5 or PM10.
    Otherwise returns (NaN, None).
    
    The sub-index may be computed above 500 internally, but the final AQI
    is clipped to the range 0-500.
    """
    valid = {k: v for k, v in sub_indices.items()
             if v is not None and not math.isnan(v)}
    
    has_pm = ("pm25" in valid) or ("pm10" in valid)
    if len(valid) < 3 or not has_pm:
        return (float("nan"), None)

    best = max(valid, key=valid.get)
    raw_aqi = valid[best]
    # Clip final AQI to [0, 500]
    clipped_aqi = min(500.0, max(0.0, raw_aqi))
    return (clipped_aqi, best)


def get_aqi_category(aqi_val: float) -> Optional[str]:
    """Return CPCB-style NAQI category label for a clipped AQI value (0-500)."""
    if aqi_val is None or math.isnan(aqi_val):
        return None
    if aqi_val <= 50:
        return "Good"
    if aqi_val <= 100:
        return "Satisfactory"
    if aqi_val <= 200:
        return "Moderate"
    if aqi_val <= 300:
        return "Poor"
    if aqi_val <= 400:
        return "Very Poor"
    return "Severe"
