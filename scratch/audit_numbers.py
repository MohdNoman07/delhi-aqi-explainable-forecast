import pandas as pd
import re

metrics_df = pd.read_csv("outputs/metrics.csv")
ridge_ci_df = pd.read_csv("outputs/rf_vs_ridge_ci.csv")
base_ci_df = pd.read_csv("outputs/rf_vs_baseline_ci.csv")
station_df = pd.read_csv("outputs/station_selection.csv")

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

mismatches = []
# 1. Check metrics table
for _, r in metrics_df.iterrows():
    h = int(r["horizon"])
    m = r["model"]
    mae_str = f"{r['MAE']:.2f}"
    rmse_str = f"{r['RMSE']:.2f}"
    r2_str = f"{r['R2']:.3f}"
    bias_str = f"{r['bias']:+.2f}"
    acc1_str = f"{r['within_one_category_accuracy']*100:.1f}%"
    prec_str = f"{r['alert_precision']:.3f}"
    rec_str = f"{r['alert_recall']:.3f}"
    
    m_escaped = r"\*\*" + m + r"\*\*" if m == "Random Forest" else m
    pat = rf"\|\s*Day \+{h}\s*\|\s*{m_escaped}\s*\|\s*{mae_str}\s*\|\s*{rmse_str}\s*\|\s*{r2_str}\s*\|\s*{re.escape(bias_str)}\s*\|\s*{acc1_str}\s*\|\s*{prec_str}\s*\|\s*{rec_str}"
    if not re.search(pat, readme):
        mismatches.append(f"Metrics table row missing/mismatch: h={h}, model={m}")

# 2. Check CI table
for _, r in base_ci_df.iterrows():
    h = int(r["horizon"])
    rf_mae = f"{r['rf_mae']:.2f}"
    bb = r["best_baseline"]
    bb_mae = f"{r['best_baseline_mae']:.2f}"
    diff_base = f"{r['mean_diff_mae']:+.2f}"
    ci_base_str = f"[{r['ci_lower_95']:+.2f}, {r['ci_upper_95']:+.2f}]"
    
    expected_substr = f"| Day +{h} | {rf_mae} | {bb} | {bb_mae} | {diff_base} {ci_base_str} |"
    if expected_substr not in readme:
        mismatches.append(f"Base CI table row missing/mismatch: h={h}, expected '{expected_substr}'")

for _, r in ridge_ci_df.iterrows():
    h = int(r["horizon"])
    r_mae = f"{r['ridge_mae']:.2f}"
    diff_r = f"{r['mean_diff_mae']:+.2f}"
    ci_r_str = f"[{r['ci_lower_95']:+.2f}, {r['ci_upper_95']:+.2f}]"
    expected_substr = f"{r_mae} | {diff_r} {ci_r_str} |"
    if expected_substr not in readme:
        mismatches.append(f"Ridge CI table row missing/mismatch: h={h}, expected '{expected_substr}'")

# 3. Check Station numbers
for _, r in station_df[~station_df["kept"]].iterrows():
    pct = f"{r['coverage']*100:.2f}%"
    v_hours = f"{int(r['valid_pm_hours']):,}"
    t_hours = f"{int(r['total_period_hours']):,}"
    if pct not in readme or v_hours not in readme or t_hours not in readme:
        mismatches.append(f"Station {r['station']} details not found exactly in README.md")

print(f"Total mismatches found: {len(mismatches)}")
for m in mismatches:
    print(" -", m)
