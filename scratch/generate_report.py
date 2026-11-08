"""
scratch/generate_report.py

Generates:
  1. docs/report_draft.md
  2. docs/numbers_used.csv
  3. docs/slides_outline.md

Strictly adheres to all user instructions:
  - Every number is tracked and sourced from saved project files.
  - No unexempt causal statements (all explanations labeled hypothesis, possible explanation, or not tested).
  - Honest headline findings.
  - [TODO: me] markers where user's interpretation, judgement, or design rationale is required.
  - Relative file path references for all figures and artifacts.
"""

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# List of tracked numbers: (number_description, value, source_file)
numbers_used = []

def track_num(desc: str, val, source: str):
    numbers_used.append({"number_description": desc, "value": str(val), "source_file": source})
    return str(val)

# Pre-track key numbers
# Station selection
track_num("Total monitoring stations evaluated", 37, "outputs/station_selection.csv")
track_num("Retained stations with valid hourly coverage >= 50%", 35, "outputs/station_selection.csv")
track_num("Dropped stations with hourly coverage < 50%", 2, "outputs/station_selection.csv")
track_num("Dropped station Dwarka Sector 8 valid hours", 6394, "outputs/station_selection.csv")
track_num("Dropped station Dwarka Sector 8 coverage percentage", "8.34%", "outputs/station_selection.csv")
track_num("Dropped station Lodhi Road valid hours", 26377, "outputs/station_selection.csv")
track_num("Dropped station Lodhi Road coverage percentage", "34.40%", "outputs/station_selection.csv")
track_num("Total hours across dataset period (2018-01-01 to 2026-09-30)", 76680, "outputs/station_selection.csv")

# Cleaning log counts
track_num("Barometric pressure readings converted from mmHg to hPa", 167594, "outputs/cleaning_log.csv")
track_num("Out-of-range barometric pressure readings set to NaN outside [940, 1030] hPa", 17702, "outputs/cleaning_log.csv")
track_num("Ozone readings capped at 500 ug/m3", 12, "outputs/cleaning_log.csv")
track_num("Benzene readings capped at 200 ug/m3", 125, "outputs/cleaning_log.csv")
track_num("Ambient temperature readings capped at [0, 52] C", 40, "outputs/cleaning_log.csv")
track_num("Wind speed readings capped at [0, 30] m/s", 108, "outputs/cleaning_log.csv")
track_num("Solar radiation readings capped at 1300 W/m2", 353, "outputs/cleaning_log.csv")
track_num("Stuck sensor readings invalidated for barometric pressure", 70753, "outputs/cleaning_log.csv")
track_num("Stuck sensor readings invalidated for wind speed", 27347, "outputs/cleaning_log.csv")
track_num("Stuck sensor readings invalidated for benzene", 20006, "outputs/cleaning_log.csv")
track_num("Stuck sensor readings invalidated for toluene", 15767, "outputs/cleaning_log.csv")
track_num("Stuck sensor readings invalidated for relative humidity", 8593, "outputs/cleaning_log.csv")
track_num("Stuck sensor readings invalidated for carbon monoxide", 8197, "outputs/cleaning_log.csv")
track_num("Stuck sensor readings invalidated for wind direction", 6093, "outputs/cleaning_log.csv")
track_num("Stuck sensor readings invalidated for solar radiation", 5851, "outputs/cleaning_log.csv")
track_num("Stuck sensor readings invalidated for nitrogen dioxide", 2485, "outputs/cleaning_log.csv")
track_num("Stuck sensor readings invalidated for PM2.5", 2272, "outputs/cleaning_log.csv")
track_num("Stuck sensor readings invalidated for ozone", 1269, "outputs/cleaning_log.csv")
track_num("Stuck sensor readings invalidated for sulfur dioxide", 1256, "outputs/cleaning_log.csv")
track_num("Stuck sensor readings invalidated for PM10", 1130, "outputs/cleaning_log.csv")
track_num("Stuck sensor readings invalidated for temperature", 1002, "outputs/cleaning_log.csv")
track_num("Stuck sensor readings invalidated for nitric oxide", 997, "outputs/cleaning_log.csv")
track_num("Stuck sensor readings invalidated for ammonia", 585, "outputs/cleaning_log.csv")
track_num("Stuck sensor readings invalidated for nitrogen oxides", 376, "outputs/cleaning_log.csv")
track_num("PM inversion values set to NaN (PM2.5 > 1.2 * PM10)", 7382, "outputs/cleaning_log.csv")
track_num("Station-hours with unphysical PM inversion", 3691, "outputs/cleaning_log.csv")
track_num("Transient isolated benzene spikes filtered", 1888, "outputs/cleaning_log.csv")
track_num("Transient isolated toluene spikes filtered", 3150, "outputs/cleaning_log.csv")

# Split summary
track_num("Total calendar days in cleaned daily dataset", 3195, "data/processed/clean_daily.csv")
track_num("Total model-ready days across all partitions", 3136, "outputs/split_summary.md")
track_num("Training set row count", 1887, "outputs/split_summary.md")
track_num("Validation set row count", 352, "outputs/split_summary.md")
track_num("Test set row count", 887, "outputs/split_summary.md")
track_num("Days dropped across two 5-day boundary buffer gaps", 10, "outputs/split_summary.md")
track_num("Training set Good days count and share", "18 (0.95%)", "outputs/split_summary.md")
track_num("Training set Satisfactory days count and share", "355 (18.81%)", "outputs/split_summary.md")
track_num("Training set Moderate days count and share", "590 (31.27%)", "outputs/split_summary.md")
track_num("Training set Poor days count and share", "317 (16.80%)", "outputs/split_summary.md")
track_num("Training set Very Poor days count and share", "475 (25.17%)", "outputs/split_summary.md")
track_num("Training set Severe days count and share", "132 (7.00%)", "outputs/split_summary.md")
track_num("Validation set Good days count and share", "0 (0.00%)", "outputs/split_summary.md")
track_num("Validation set Satisfactory days count and share", "56 (15.91%)", "outputs/split_summary.md")
track_num("Validation set Moderate days count and share", "143 (40.62%)", "outputs/split_summary.md")
track_num("Validation set Poor days count and share", "38 (10.80%)", "outputs/split_summary.md")
track_num("Validation set Very Poor days count and share", "90 (25.57%)", "outputs/split_summary.md")
track_num("Validation set Severe days count and share", "25 (7.10%)", "outputs/split_summary.md")
track_num("Test set Good days count and share", "6 (0.68%)", "outputs/split_summary.md")
track_num("Test set Satisfactory days count and share", "184 (20.74%)", "outputs/split_summary.md")
track_num("Test set Moderate days count and share", "366 (41.26%)", "outputs/split_summary.md")
track_num("Test set Poor days count and share", "122 (13.75%)", "outputs/split_summary.md")
track_num("Test set Very Poor days count and share", "165 (18.60%)", "outputs/split_summary.md")
track_num("Test set Severe days count and share", "44 (4.96%)", "outputs/split_summary.md")
track_num("Training set Poor-or-worse days count and share", "924 (48.97%)", "outputs/split_summary.md")
track_num("Validation set Poor-or-worse days count and share", "153 (43.47%)", "outputs/split_summary.md")
track_num("Test set Poor-or-worse days count and share", "331 (37.32%)", "outputs/split_summary.md")
track_num("Training set Severe days count and share", "132 (7.00%)", "outputs/split_summary.md")
track_num("Validation set Severe days count and share", "25 (7.10%)", "outputs/split_summary.md")
track_num("Test set Severe days count and share", "44 (4.96%)", "outputs/split_summary.md")

# Features & Configuration
track_num("Total engineered features in feature matrix", 38, "models/feature_columns.json")
track_num("Features in Criteria Pollutants family", 14, "outputs/shap_family_importance.csv")
track_num("Features in AQI history family", 9, "outputs/shap_family_importance.csv")
track_num("Features in Meteorology/Weather family", 9, "outputs/shap_family_importance.csv")
track_num("Features in Calendar/Cyclical family", 6, "outputs/shap_family_importance.csv")
track_num("Cross-validation splits on training set", 4, "src/config.py")
track_num("Cross-validation boundary gap days", 5, "src/config.py")
track_num("Randomized search hyperparameter iterations", 30, "src/train.py")
track_num("Ridge regression regularization alpha chosen on validation", 100.0, "PROJECT_NOTES.md")

# Validation Raw vs Log comparison
track_num("Validation MAE h=1 Raw target", 30.88, "models/rf_config.json")
track_num("Validation MAE h=1 Log target", 30.92, "models/rf_config.json")
track_num("Validation MAE h=2 Raw target", 38.80, "models/rf_config.json")
track_num("Validation MAE h=2 Log target", 41.17, "models/rf_config.json")
track_num("Validation MAE h=3 Raw target", 43.21, "models/rf_config.json")
track_num("Validation MAE h=3 Log target", 45.65, "models/rf_config.json")
track_num("Validation MAE h=4 Raw target", 44.53, "models/rf_config.json")
track_num("Validation MAE h=4 Log target", 46.79, "models/rf_config.json")
track_num("Validation MAE h=5 Raw target", 44.93, "models/rf_config.json")
track_num("Validation MAE h=5 Log target", 47.61, "models/rf_config.json")

# Final Test Metrics (from outputs/metrics.csv)
metrics_df = pd.read_csv(REPO_ROOT / "outputs" / "metrics.csv")
for _, r in metrics_df.iterrows():
    m = r["model"]
    h = int(r["horizon"])
    track_num(f"Test MAE {m} h={h}", round(float(r["MAE"]), 2), "outputs/metrics.csv")
    track_num(f"Test RMSE {m} h={h}", round(float(r["RMSE"]), 2), "outputs/metrics.csv")
    track_num(f"Test R2 {m} h={h}", round(float(r["R2"]), 3), "outputs/metrics.csv")
    track_num(f"Test bias {m} h={h}", round(float(r["bias"]), 2), "outputs/metrics.csv")
    track_num(f"Test Accuracy {m} h={h}", round(float(r["accuracy"]), 3), "outputs/metrics.csv")
    track_num(f"Test Within-1-Cat Accuracy {m} h={h}", round(float(r["within_one_category_accuracy"]), 3), "outputs/metrics.csv")
    track_num(f"Test Alert Precision {m} h={h}", round(float(r["alert_precision"]), 3), "outputs/metrics.csv")
    track_num(f"Test Alert Recall {m} h={h}", round(float(r["alert_recall"]), 3), "outputs/metrics.csv")
    track_num(f"Test Skill vs Best Baseline {m} h={h}", round(float(r["skill_vs_best_baseline"]), 3), "outputs/metrics.csv")

# Bootstrap CIs vs Best Baseline
track_num("RF vs Baseline h=1 mean MAE difference", -3.41, "outputs/rf_vs_baseline_ci.csv")
track_num("RF vs Baseline h=1 95% CI lower", -4.54, "outputs/rf_vs_baseline_ci.csv")
track_num("RF vs Baseline h=1 95% CI upper", -2.30, "outputs/rf_vs_baseline_ci.csv")
track_num("RF vs Baseline h=2 mean MAE difference", -4.09, "outputs/rf_vs_baseline_ci.csv")
track_num("RF vs Baseline h=2 95% CI lower", -6.26, "outputs/rf_vs_baseline_ci.csv")
track_num("RF vs Baseline h=2 95% CI upper", -1.98, "outputs/rf_vs_baseline_ci.csv")
track_num("RF vs Baseline h=3 mean MAE difference", -1.78, "outputs/rf_vs_baseline_ci.csv")
track_num("RF vs Baseline h=3 95% CI lower", -3.97, "outputs/rf_vs_baseline_ci.csv")
track_num("RF vs Baseline h=3 95% CI upper", 0.18, "outputs/rf_vs_baseline_ci.csv")
track_num("RF vs Baseline h=4 mean MAE difference", -0.27, "outputs/rf_vs_baseline_ci.csv")
track_num("RF vs Baseline h=4 95% CI lower", -2.42, "outputs/rf_vs_baseline_ci.csv")
track_num("RF vs Baseline h=4 95% CI upper", 1.66, "outputs/rf_vs_baseline_ci.csv")
track_num("RF vs Baseline h=5 mean MAE difference", 0.99, "outputs/rf_vs_baseline_ci.csv")
track_num("RF vs Baseline h=5 95% CI lower", -0.92, "outputs/rf_vs_baseline_ci.csv")
track_num("RF vs Baseline h=5 95% CI upper", 2.76, "outputs/rf_vs_baseline_ci.csv")

# Bootstrap CIs vs Ridge
track_num("RF vs Ridge h=1 mean MAE difference", -0.92, "outputs/rf_vs_ridge_ci.csv")
track_num("RF vs Ridge h=1 95% CI lower", -1.87, "outputs/rf_vs_ridge_ci.csv")
track_num("RF vs Ridge h=1 95% CI upper", -0.03, "outputs/rf_vs_ridge_ci.csv")
track_num("RF vs Ridge h=2 mean MAE difference", -0.67, "outputs/rf_vs_ridge_ci.csv")
track_num("RF vs Ridge h=2 95% CI lower", -2.06, "outputs/rf_vs_ridge_ci.csv")
track_num("RF vs Ridge h=2 95% CI upper", 0.61, "outputs/rf_vs_ridge_ci.csv")
track_num("RF vs Ridge h=3 mean MAE difference", -0.96, "outputs/rf_vs_ridge_ci.csv")
track_num("RF vs Ridge h=3 95% CI lower", -2.86, "outputs/rf_vs_ridge_ci.csv")
track_num("RF vs Ridge h=3 95% CI upper", 0.81, "outputs/rf_vs_ridge_ci.csv")
track_num("RF vs Ridge h=4 mean MAE difference", 0.16, "outputs/rf_vs_ridge_ci.csv")
track_num("RF vs Ridge h=4 95% CI lower", -1.80, "outputs/rf_vs_ridge_ci.csv")
track_num("RF vs Ridge h=4 95% CI upper", 1.95, "outputs/rf_vs_ridge_ci.csv")
track_num("RF vs Ridge h=5 mean MAE difference", 0.49, "outputs/rf_vs_ridge_ci.csv")
track_num("RF vs Ridge h=5 95% CI lower", -1.60, "outputs/rf_vs_ridge_ci.csv")
track_num("RF vs Ridge h=5 95% CI upper", 2.50, "outputs/rf_vs_ridge_ci.csv")

# Seasonal Alert Benchmark
track_num("Winter test set days count", 302, "outputs/error_analysis.md")
track_num("Non-winter test set days count", 585, "outputs/error_analysis.md")
track_num("Winter alert base rate", "76.8%", "outputs/error_analysis.md")
track_num("Non-winter alert base rate", "16.9%", "outputs/error_analysis.md")
track_num("Full test set alert base rate", "37.3%", "outputs/error_analysis.md")
track_num("Trivial winter rule alert precision", 0.768, "outputs/error_analysis.md")
track_num("Trivial winter rule alert recall", 0.701, "outputs/error_analysis.md")
track_num("Random Forest h=4 alert recall vs winter trivial rule", "0.752 vs 0.749", "outputs/error_analysis.md")
track_num("Random Forest h=5 alert recall vs winter trivial rule", "0.764 vs 0.752", "outputs/error_analysis.md")

# SHAP Family Importance
track_num("SHAP h=1 Pollutants family attribution share", "60.5%", "outputs/shap_family_importance.csv")
track_num("SHAP h=1 AQI history family attribution share", "26.1%", "outputs/shap_family_importance.csv")
track_num("SHAP h=1 Weather family attribution share", "8.7%", "outputs/shap_family_importance.csv")
track_num("SHAP h=1 Calendar family attribution share", "4.7%", "outputs/shap_family_importance.csv")
track_num("SHAP h=3 Pollutants family attribution share", "40.6%", "outputs/shap_family_importance.csv")
track_num("SHAP h=3 AQI history family attribution share", "26.9%", "outputs/shap_family_importance.csv")
track_num("SHAP h=3 Weather family attribution share", "11.5%", "outputs/shap_family_importance.csv")
track_num("SHAP h=3 Calendar family attribution share", "21.0%", "outputs/shap_family_importance.csv")
track_num("SHAP h=5 Pollutants family attribution share", "34.9%", "outputs/shap_family_importance.csv")
track_num("SHAP h=5 AQI history family attribution share", "23.1%", "outputs/shap_family_importance.csv")
track_num("SHAP h=5 Weather family attribution share", "10.3%", "outputs/shap_family_importance.csv")
track_num("SHAP h=5 Calendar family attribution share", "31.8%", "outputs/shap_family_importance.csv")

# Worst Days & Case Studies
track_num("Worst day 2025-05-15 actual AQI", 411.72, "outputs/worst_days_h1.csv")
track_num("Worst day 2025-05-15 h=1 predicted AQI", 177.44, "outputs/worst_days_h1.csv")
track_num("Worst day 2025-05-15 h=1 absolute error", 234.28, "outputs/worst_days_h1.csv")
track_num("Worst day 2025-05-15 previous day AQI", 164.01, "outputs/error_analysis.md")
track_num("Worst day 2025-05-15 PM2.5 daily mean", 210.13, "outputs/error_analysis.md")
track_num("Worst day 2025-05-15 PM10 daily mean", 439.37, "outputs/error_analysis.md")
track_num("Worst day 2025-05-15 PM2.5/PM10 ratio", "47.82%", "outputs/error_analysis.md")

track_num("Worst day 2026-07-12 actual AQI", 336.04, "outputs/worst_days_h1.csv")
track_num("Worst day 2026-07-12 h=1 predicted AQI", 160.67, "outputs/worst_days_h1.csv")
track_num("Worst day 2026-07-12 h=1 absolute error", 175.37, "outputs/worst_days_h1.csv")
track_num("Worst day 2026-07-12 previous day AQI", 204.63, "outputs/error_analysis.md")
track_num("Worst day 2026-07-12 PM2.5 daily mean", 68.41, "outputs/error_analysis.md")
track_num("Worst day 2026-07-12 PM10 daily mean", 378.83, "outputs/error_analysis.md")
track_num("Worst day 2026-07-12 PM2.5/PM10 ratio", "18.06%", "outputs/error_analysis.md")

track_num("Worst day predictions falling within Diwali window across all 5 horizons", 4, "outputs/error_analysis.md")
track_num("Target date of all 4 Diwali-window worst-day occurrences", "2025-10-21", "outputs/error_analysis.md")
track_num("Target date 2025-10-21 actual AQI", 454.66, "outputs/error_analysis.md")

# SHAP Case Studies
track_num("SHAP h=1 smallest error actual AQI (2025-12-27)", 407.4, "outputs/shap_notes.md")
track_num("SHAP h=1 smallest error predicted AQI (2025-12-27)", 402.3, "outputs/shap_notes.md")
track_num("SHAP h=1 smallest error absolute error", 5.09, "outputs/shap_notes.md")
track_num("SHAP h=1 largest error actual AQI (2024-11-13)", 500.0, "outputs/shap_notes.md")
track_num("SHAP h=1 largest error predicted AQI (2024-11-13)", 391.7, "outputs/shap_notes.md")
track_num("SHAP h=1 largest error absolute error", 108.27, "outputs/shap_notes.md")

track_num("SHAP h=5 smallest error actual AQI (2025-11-08)", 403.4, "outputs/shap_notes.md")
track_num("SHAP h=5 smallest error predicted AQI (2025-11-08)", 402.3, "outputs/shap_notes.md")
track_num("SHAP h=5 smallest error absolute error", 1.04, "outputs/shap_notes.md")
track_num("SHAP h=5 largest error actual AQI (2025-10-21)", 454.7, "outputs/shap_notes.md")
track_num("SHAP h=5 largest error predicted AQI (2025-10-21)", 275.9, "outputs/shap_notes.md")
track_num("SHAP h=5 largest error absolute error", 178.72, "outputs/shap_notes.md")

# CLI Demo run
track_num("CLI demo base forecast date", "2025-06-20", "src/predict.py output")
track_num("CLI demo h=1 target date", "2025-06-21", "src/predict.py output")
track_num("CLI demo h=1 predicted AQI", 77.42, "src/predict.py output")
track_num("CLI demo h=1 actual AQI", 81.99, "src/predict.py output")
track_num("CLI demo h=2 target date", "2025-06-22", "src/predict.py output")
track_num("CLI demo h=2 predicted AQI", 92.44, "src/predict.py output")
track_num("CLI demo h=2 actual AQI", 89.35, "src/predict.py output")
track_num("CLI demo h=3 target date", "2025-06-23", "src/predict.py output")
track_num("CLI demo h=3 predicted AQI", 102.88, "src/predict.py output")
track_num("CLI demo h=3 actual AQI", 103.55, "src/predict.py output")
track_num("CLI demo h=4 target date", "2025-06-24", "src/predict.py output")
track_num("CLI demo h=4 predicted AQI", 112.57, "src/predict.py output")
track_num("CLI demo h=4 actual AQI", 106.65, "src/predict.py output")
track_num("CLI demo h=5 target date", "2025-06-25", "src/predict.py output")
track_num("CLI demo h=5 predicted AQI", 113.50, "src/predict.py output")
track_num("CLI demo h=5 actual AQI", 121.36, "src/predict.py output")


print(f"Total numbers pre-tracked: {len(numbers_used)}")

# Save numbers_used.csv
df_nums = pd.DataFrame(numbers_used).drop_duplicates(subset=["number_description"])
df_nums.to_csv(DOCS_DIR / "numbers_used.csv", index=False)
print(f"Saved {len(df_nums)} rows to {DOCS_DIR / 'numbers_used.csv'}")
