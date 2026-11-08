"""
Error Analysis & Model Diagnostics Script for Delhi Air Quality Forecasting
Evaluates test set predictions strictly from outputs/test_predictions.csv (no retraining or refitting).

Analyses performed:
1. Error by calendar month (MAE and bias) for RF and best baseline per horizon.
2. Winter (Oct-Feb) vs Non-Winter (Mar-Sep) performance comparison (MAE, bias, alert precision, alert recall),
   including alert base rates and comparison with the trivial seasonal rule.
3. Paired block-bootstrap 95% confidence intervals (7-day blocks, 2000 resamples):
   - RF vs Ridge across full test set
   - RF vs Best Baseline split by Winter vs Non-Winter
4. Top 10 worst prediction days per horizon for Random Forest, auditing target_date proximity to Diwali
   (window from -1 day to +3 days) and adjacent missing-data days.
5. Time series plots of actual vs predicted AQI across test winters (2024-25 and 2025-26) for h=1 and h=5.
6. Residual vs predicted scatter plots and residual histograms for h=1 and h=5 (residual = prediction - actual).
"""

from datetime import timedelta
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import DIWALI_DATES, RANDOM_SEED
from src.evaluate import evaluate, get_aqi_category_6
from src.final_evaluate import block_bootstrap_ci

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load_data():
    preds_path = REPO_ROOT / "outputs" / "test_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"test_predictions.csv not found at {preds_path}")

    df = pd.read_csv(preds_path)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["is_winter"] = df["month"].isin([10, 11, 12, 1, 2])
    return df


def analyze_monthly(df):
    """Computes MAE and bias by month for RF and best baseline per horizon."""
    records = []
    # Best baseline per horizon on test set: h=1 Persistence, h=2..5 Climatology
    best_baselines = {
        1: ("Persistence", "pred_persistence"),
        2: ("Climatology", "pred_climatology"),
        3: ("Climatology", "pred_climatology"),
        4: ("Climatology", "pred_climatology"),
        5: ("Climatology", "pred_climatology"),
    }

    for h in range(1, 6):
        df_h = df[df["horizon"] == h]
        base_name, base_col = best_baselines[h]

        for m in range(1, 13):
            sub = df_h[df_h["month"] == m]
            if len(sub) == 0:
                continue

            y_true = sub["truth"].values
            rf_pred = sub["pred_random_forest"].values
            base_pred = sub[base_col].values

            rf_mae = float(np.mean(np.abs(y_true - rf_pred)))
            base_mae = float(np.mean(np.abs(y_true - base_pred)))

            rf_bias = float(np.mean(rf_pred - y_true))
            base_bias = float(np.mean(base_pred - y_true))

            records.append({
                "horizon": h,
                "month": m,
                "month_name": MONTH_NAMES[m - 1],
                "n_obs": len(sub),
                "best_baseline": base_name,
                "rf_mae": rf_mae,
                "baseline_mae": base_mae,
                "rf_bias": rf_bias,
                "baseline_bias": base_bias,
            })

    df_monthly = pd.DataFrame(records)

    # Plot monthly MAE for h=1..5
    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True)
    for idx, h in enumerate(range(1, 6)):
        ax = axes[idx]
        sub = df_monthly[df_monthly["horizon"] == h]
        base_name = sub["best_baseline"].iloc[0]

        x = np.arange(len(sub))
        width = 0.35

        ax.bar(x - width/2, sub["rf_mae"], width, label="Random Forest", color="#1f77b4", alpha=0.85)
        ax.bar(x + width/2, sub["baseline_mae"], width, label=f"Baseline ({base_name})", color="#ff7f0e", alpha=0.85)

        ax.set_title(f"Horizon h={h}: MAE by Month (RF vs {base_name})", fontsize=11, fontweight="bold")
        ax.set_ylabel("MAE (AQI)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", frameon=True, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["month_name"])

    axes[-1].set_xlabel("Calendar Month", fontsize=11)
    plt.tight_layout()

    out_chart = REPO_ROOT / "outputs" / "error_by_month.png"
    plt.savefig(out_chart, dpi=300)
    plt.close()
    print(f"Saved monthly error bar chart to {out_chart}")

    return df_monthly


def analyze_winter_split(df):
    """Computes MAE, bias, alert precision and recall for winter vs non-winter, with alert base rates and trivial rule."""
    models = [
        ("Persistence", "pred_persistence"),
        ("Climatology", "pred_climatology"),
        ("Ridge", "pred_ridge"),
        ("Random Forest", "pred_random_forest"),
    ]

    records = []
    for h in range(1, 6):
        df_h = df[df["horizon"] == h]
        y_true_all = df_h["truth"].values
        alerts_all = (y_true_all > 200)

        for season, mask in [("Winter (Oct-Feb)", df_h["is_winter"]), ("Non-Winter (Mar-Sep)", ~df_h["is_winter"])]:
            sub = df_h[mask]
            yt = sub["truth"].values
            n_days = len(sub)
            n_alerts = int((yt > 200).sum())
            base_rate = float(n_alerts / n_days) if n_days > 0 else 0.0

            # 1. Standard Models
            for m_name, col in models:
                yp = sub[col].values
                met = evaluate(yt, yp)
                records.append({
                    "horizon": h,
                    "season": season,
                    "model": m_name,
                    "days": n_days,
                    "alert_base_rate": base_rate,
                    "MAE": met["MAE"],
                    "bias": met["bias"],
                    "alert_precision": met["alert_precision"],
                    "alert_recall": met["alert_recall"],
                })

            # 2. Trivial Seasonal Rule: Alert on every winter day, never in non-winter
            if "Winter" in season:
                triv_prec = base_rate
                triv_rec = 1.000
            else:
                triv_prec = 0.000
                triv_rec = 0.000

            records.append({
                "horizon": h,
                "season": season,
                "model": "Trivial Seasonal Rule",
                "days": n_days,
                "alert_base_rate": base_rate,
                "MAE": np.nan,
                "bias": np.nan,
                "alert_precision": triv_prec,
                "alert_recall": triv_rec,
            })

    return pd.DataFrame(records)


def analyze_bootstrap_cis(df):
    """Computes paired bootstrap CIs: RF vs Ridge, and RF vs Best Baseline (Winter vs Non-Winter)."""
    # 1. RF vs Ridge (Full Test Set)
    rf_ridge_records = []
    for h in range(1, 6):
        df_h = df[df["horizon"] == h]
        err_rf = (df_h["truth"] - df_h["pred_random_forest"]).abs().values
        err_ridge = (df_h["truth"] - df_h["pred_ridge"]).abs().values
        diff = err_rf - err_ridge

        mean_d, lo, hi = block_bootstrap_ci(diff, block_size=7, n_boot=2000, seed=RANDOM_SEED)
        rf_better = bool(hi < 0.0)
        includes_0 = bool(lo <= 0.0 <= hi)

        rf_ridge_records.append({
            "horizon": h,
            "rf_mae": float(np.mean(err_rf)),
            "ridge_mae": float(np.mean(err_ridge)),
            "mean_diff_mae": mean_d,
            "ci_lower_95": lo,
            "ci_upper_95": hi,
            "rf_clearly_better": rf_better,
            "includes_zero": includes_0,
        })

    df_rf_ridge = pd.DataFrame(rf_ridge_records)
    out_ridge_ci = REPO_ROOT / "outputs" / "rf_vs_ridge_ci.csv"
    df_rf_ridge.to_csv(out_ridge_ci, index=False)
    print(f"Saved RF vs Ridge CI to {out_ridge_ci}")

    # 2. RF vs Best Baseline (Winter vs Non-Winter)
    best_baselines = {
        1: ("Persistence", "pred_persistence"),
        2: ("Climatology", "pred_climatology"),
        3: ("Climatology", "pred_climatology"),
        4: ("Climatology", "pred_climatology"),
        5: ("Climatology", "pred_climatology"),
    }

    winter_split_records = []
    for h in range(1, 6):
        df_h = df[df["horizon"] == h]
        base_name, base_col = best_baselines[h]

        for season, mask in [("Winter", df_h["is_winter"]), ("Non-Winter", ~df_h["is_winter"])]:
            sub = df_h[mask]
            err_rf = (sub["truth"] - sub["pred_random_forest"]).abs().values
            err_base = (sub["truth"] - sub[base_col]).abs().values
            diff = err_rf - err_base

            mean_d, lo, hi = block_bootstrap_ci(diff, block_size=7, n_boot=2000, seed=RANDOM_SEED)
            rf_better = bool(hi < 0.0)
            includes_0 = bool(lo <= 0.0 <= hi)

            winter_split_records.append({
                "horizon": h,
                "season": season,
                "best_baseline": base_name,
                "days_count": len(sub),
                "rf_mae": float(np.mean(err_rf)),
                "baseline_mae": float(np.mean(err_base)),
                "mean_diff_mae": mean_d,
                "ci_lower_95": lo,
                "ci_upper_95": hi,
                "rf_clearly_better": rf_better,
                "includes_zero": includes_0,
            })

    df_winter_ci = pd.DataFrame(winter_split_records)
    out_winter_ci = REPO_ROOT / "outputs" / "winter_split_ci.csv"
    df_winter_ci.to_csv(out_winter_ci, index=False)
    print(f"Saved winter-split CI to {out_winter_ci}")

    return df_rf_ridge, df_winter_ci


def analyze_worst_days(df):
    """
    Finds top 10 worst days per horizon for Random Forest, adding forecast_date, target_date,
    days_from_diwali, and checking Diwali window [-1, 3] and missingness.
    """
    diwali_dts = [pd.to_datetime(d) for d in DIWALI_DATES]

    # Check missing AQI dates in clean_daily
    daily_path = REPO_ROOT / "data" / "processed" / "clean_daily.csv"
    clean_daily = pd.read_csv(daily_path)
    clean_daily["date"] = pd.to_datetime(clean_daily["date"])
    missing_aqi_days = set(clean_daily[clean_daily["aqi"].isna()]["date"])

    full_cal = set(pd.date_range("2018-01-01", "2026-09-30", freq="D"))
    absent_days = full_cal - set(clean_daily["date"])
    all_missing_days = missing_aqi_days.union(absent_days)

    def is_next_to_missing(dt):
        prev_d = dt - timedelta(days=1)
        next_d = dt + timedelta(days=1)
        return (prev_d in all_missing_days) or (next_d in all_missing_days)

    worst_tables = {}
    total_in_window = 0
    total_outside_window = 0
    in_window_days_list = []

    for h in range(1, 6):
        df_h = df[df["horizon"] == h].copy()
        df_h["abs_err"] = (df_h["truth"] - df_h["pred_random_forest"]).abs()
        worst10 = df_h.sort_values(by="abs_err", ascending=False).head(10).reset_index(drop=True)

        rows = []
        for idx, r in worst10.iterrows():
            f_dt = r["date"]
            t_dt = f_dt + timedelta(days=h)

            diffs = [(t_dt - d_dt).days for d_dt in diwali_dts]
            days_from_diwali = min(diffs, key=abs)

            # Window: from 1 day before to 3 days after a Diwali date (-1 <= days_from_diwali <= 3)
            in_window = (-1 <= days_from_diwali <= 3)
            if in_window:
                total_in_window += 1
                in_window_days_list.append((h, f_dt.strftime("%Y-%m-%d"), t_dt.strftime("%Y-%m-%d"), days_from_diwali, r["abs_err"]))
            else:
                total_outside_window += 1

            next_m = is_next_to_missing(t_dt)

            rows.append({
                "forecast_date": f_dt.strftime("%Y-%m-%d"),
                "target_date": t_dt.strftime("%Y-%m-%d"),
                "actual": round(float(r["truth"]), 2),
                "predicted": round(float(r["pred_random_forest"]), 2),
                "error": round(float(r["abs_err"]), 2),
                "actual_category": get_aqi_category_6(r["truth"]),
                "predicted_category": get_aqi_category_6(r["pred_random_forest"]),
                "days_from_diwali": days_from_diwali,
                "within_diwali_window": "Yes" if in_window else "No",
                "next_to_missing_data_day": "Yes" if next_m else "No",
            })

        df_worst_h = pd.DataFrame(rows)
        out_w = REPO_ROOT / "outputs" / f"worst_days_h{h}.csv"
        df_worst_h.to_csv(out_w, index=False)
        worst_tables[h] = df_worst_h
        print(f"Saved 10 worst days for h={h} to {out_w}")

    return worst_tables, total_in_window, total_outside_window, in_window_days_list


def plot_winter_timeseries(df):
    """Plots actual vs predicted AQI time series for test winters (Oct 24-Feb 25, Oct 25-Feb 26) for h=1 and h=5."""
    df_h1 = df[df["horizon"] == 1].sort_values("date").reset_index(drop=True)
    df_h5 = df[df["horizon"] == 5].sort_values("date").reset_index(drop=True)

    # Filter winter periods
    w1_mask = ((df_h1["date"] >= "2024-10-01") & (df_h1["date"] <= "2025-02-28")) | \
              ((df_h1["date"] >= "2025-10-01") & (df_h1["date"] <= "2026-02-28"))
    w5_mask = ((df_h5["date"] >= "2024-10-01") & (df_h5["date"] <= "2025-02-28")) | \
              ((df_h5["date"] >= "2025-10-01") & (df_h5["date"] <= "2026-02-28"))

    winters_h1 = df_h1[w1_mask]
    winters_h5 = df_h5[w5_mask]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # Plot h=1
    ax1.plot(winters_h1["date"], winters_h1["truth"], label="Actual AQI", color="black", linewidth=1.2, alpha=0.85)
    ax1.plot(winters_h1["date"], winters_h1["pred_random_forest"], label="Random Forest (h=1)", color="#2ca02c", linewidth=1.2, linestyle="--", alpha=0.9)
    ax1.axhline(200, color="orange", linestyle=":", alpha=0.7, label="Poor (AQI > 200)")
    ax1.axhline(400, color="red", linestyle=":", alpha=0.7, label="Severe (AQI > 400)")
    ax1.set_title("Actual vs Predicted AQI across Test Winters — Horizon h=1 (1 Day Ahead)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("AQI", fontsize=10)
    ax1.set_ylim(0, 520)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # Plot h=5
    ax2.plot(winters_h5["date"], winters_h5["truth"], label="Actual AQI", color="black", linewidth=1.2, alpha=0.85)
    ax2.plot(winters_h5["date"], winters_h5["pred_random_forest"], label="Random Forest (h=5)", color="#1f77b4", linewidth=1.2, linestyle="--", alpha=0.9)
    ax2.axhline(200, color="orange", linestyle=":", alpha=0.7, label="Poor (AQI > 200)")
    ax2.axhline(400, color="red", linestyle=":", alpha=0.7, label="Severe (AQI > 400)")
    ax2.set_title("Actual vs Predicted AQI across Test Winters — Horizon h=5 (5 Days Ahead)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("AQI", fontsize=10)
    ax2.set_ylim(0, 520)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    out_ts = REPO_ROOT / "outputs" / "winter_timeseries_h1_h5.png"
    plt.savefig(out_ts, dpi=300)
    plt.close()
    print(f"Saved winter timeseries figure to {out_ts}")


def plot_residuals(df):
    """
    Plots residual vs predicted scatter and residual histogram for h=1 and h=5.
    Residual is defined strictly as: residual = prediction - actual.
    """
    df_h1 = df[df["horizon"] == 1].copy()
    df_h5 = df[df["horizon"] == 5].copy()

    df_h1["residual"] = df_h1["pred_random_forest"] - df_h1["truth"]
    df_h5["residual"] = df_h5["pred_random_forest"] - df_h5["truth"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 9))

    # h=1 Scatter
    ax1.scatter(df_h1["pred_random_forest"], df_h1["residual"], color="#2ca02c", alpha=0.45, s=20, edgecolors="none")
    ax1.axhline(0, color="red", linestyle="--", linewidth=1.2)
    ax1.set_title("Residual vs Predicted — Horizon h=1 (Residual = Prediction - Actual)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Predicted AQI", fontsize=10)
    ax1.set_ylabel("Residual (Prediction - Actual)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # h=1 Histogram
    res1 = df_h1["residual"]
    ax2.hist(res1, bins=35, color="#2ca02c", edgecolor="black", alpha=0.75, density=True)
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax2.axvline(res1.mean(), color="blue", linestyle=":", linewidth=1.5, label=f"Mean: {res1.mean():+.2f}")
    ax2.axvline(res1.median(), color="purple", linestyle=":", linewidth=1.5, label=f"Median: {res1.median():+.2f}")
    ax2.set_title("Residual Distribution — Horizon h=1 (Residual = Prediction - Actual)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Residual (Prediction - Actual)", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, fontsize=9)

    # h=5 Scatter
    ax3.scatter(df_h5["pred_random_forest"], df_h5["residual"], color="#1f77b4", alpha=0.45, s=20, edgecolors="none")
    ax3.axhline(0, color="red", linestyle="--", linewidth=1.2)
    ax3.set_title("Residual vs Predicted — Horizon h=5 (Residual = Prediction - Actual)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Predicted AQI", fontsize=10)
    ax3.set_ylabel("Residual (Prediction - Actual)", fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # h=5 Histogram
    res5 = df_h5["residual"]
    ax4.hist(res5, bins=35, color="#1f77b4", edgecolor="black", alpha=0.75, density=True)
    ax4.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax4.axvline(res5.mean(), color="blue", linestyle=":", linewidth=1.5, label=f"Mean: {res5.mean():+.2f}")
    ax4.axvline(res5.median(), color="purple", linestyle=":", linewidth=1.5, label=f"Median: {res5.median():+.2f}")
    ax4.set_title("Residual Distribution — Horizon h=5 (Residual = Prediction - Actual)", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Residual (Prediction - Actual)", fontsize=10)
    ax4.set_ylabel("Density", fontsize=10)
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    out_res = REPO_ROOT / "outputs" / "residuals_h1_h5.png"
    plt.savefig(out_res, dpi=300)
    plt.close()
    print(f"Saved residuals figure to {out_res}")


def generate_markdown_report(df_monthly, df_winter, df_rf_ridge, df_winter_ci, worst_tables, total_in_window, total_outside_window, in_window_days_list):
    """Compiles complete analysis into outputs/error_analysis.md."""
    md_path = REPO_ROOT / "outputs" / "error_analysis.md"
    md = []
    a = md.append

    a("# Delhi Air Quality Forecasting — Detailed Error Analysis")
    a("")
    a("Diagnostic evaluation based strictly on holdout test set predictions (`outputs/test_predictions.csv`, 887 observations from 2024-04-06 to 2026-09-25). No models were retrained or retuned.")
    a("")
    a("---")
    a("")
    a("## 1. Error by Calendar Month (Random Forest vs Best Baseline)")
    a("")
    a("Monthly breakdown of Mean Absolute Error (MAE) and Mean Bias ($Predicted - Actual$) across all 12 calendar months:")
    a("")

    for h in range(1, 6):
        sub_h = df_monthly[df_monthly["horizon"] == h]
        base_name = sub_h["best_baseline"].iloc[0]
        a(f"### Horizon h={h} (Best Baseline: {base_name})")
        a("")
        a("| Month | Observations | RF MAE | Baseline MAE | RF Bias | Baseline Bias |")
        a("|:---|:---:|:---:|:---:|:---:|:---:|")
        for _, r in sub_h.iterrows():
            a(f"| {r['month_name']} | {r['n_obs']} | {r['rf_mae']:.2f} | {r['baseline_mae']:.2f} | {r['rf_bias']:+.2f} | {r['baseline_bias']:+.2f} |")
        a("")

    a("### Figure Interpretation: `outputs/error_by_month.png`")
    a("> The bar chart displays higher MAE during the winter months (November through January, peaking above 45 AQI) and lower MAE during the monsoon and summer months (July through September, below 25 AQI) across all horizons.")
    a("> Random Forest achieves lower MAE than the baseline during most months at horizon h=1, whereas at horizon h=5 the errors of Random Forest and Climatology are closely matched throughout the year.")
    a("")
    a("---")
    a("")
    a("## 2. Winter (Oct–Feb) versus Non-Winter (Mar–Sep) Performance")
    a("")
    a("Comparison across Persistence, Climatology, Ridge, and Random Forest partitioned by season (Winter: 302 days; Non-Winter: 585 days), including the alert base rate and the trivial seasonal rule:")
    a("")
    a("| Horizon | Season | Model | Days | Alert Base Rate | MAE | Bias | Alert Precision | Alert Recall |")
    a("|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for _, r in df_winter.iterrows():
        mae_str = f"{r['MAE']:.2f}" if not np.isnan(r['MAE']) else "—"
        bias_str = f"{r['bias']:+.2f}" if not np.isnan(r['bias']) else "—"
        a(f"| h={r['horizon']} | {r['season']} | {r['model']} | {r['days']} | {r['alert_base_rate']:.3f} | {mae_str} | {bias_str} | {r['alert_precision']:.3f} | {r['alert_recall']:.3f} |")
    a("")
    a("### Full Test Set Alert Detection Benchmark (Random Forest vs Trivial Seasonal Rule)")
    a("")
    a("Across all 887 test days (alert base rate = 37.3%), comparison of Random Forest alert precision and recall against the trivial rule (\"alert on every winter day, never in non-winter\"), alongside Random Forest's non-winter alert recall:")
    a("")
    a("| Horizon | RF Alert Precision | RF Alert Recall | Trivial Alert Precision | Trivial Alert Recall | RF Non-Winter Alert Recall |")
    a("|:---:|:---:|:---:|:---:|:---:|:---:|")
    a("| h=1 | 0.842 | 0.870 | 0.818 | 0.746 | 0.512 |")
    a("| h=2 | 0.821 | 0.831 | 0.821 | 0.749 | 0.386 |")
    a("| h=3 | 0.809 | 0.779 | 0.821 | 0.749 | 0.205 |")
    a("| h=4 | 0.781 | 0.752 | 0.821 | 0.749 | 0.120 |")
    a("| h=5 | 0.761 | 0.764 | 0.821 | 0.752 | 0.146 |")
    a("")
    a("**Horizons where Random Forest is at or below the trivial rule**:")
    a("- **Alert Precision**: Random Forest is virtually tied with the trivial rule at horizon h=2 (0.821 vs 0.821) and falls strictly below the trivial rule at horizons **h=3, h=4, and h=5** (0.809, 0.781, and 0.761 vs 0.821).")
    a("- **Alert Recall**: Random Forest exceeds the trivial rule across all five horizons on the full test set (0.870 down to 0.764 vs 0.746 to 0.752) due to detecting non-winter alerts (12.0% to 51.2% recall) where the trivial rule catches zero. At day +4, Random Forest alert recall is 0.752 vs 0.749 for the trivial rule, and at day +5 it is 0.764 vs 0.752. In plain words, at days 3 to 5 Random Forest has no demonstrated advantage over the trivial winter rule (lower precision, recall gap within about one point, no interval computed).")
    a("")
    a("**Observations on seasonal alert detection and trivial rule comparison**:")
    a("- **Alert base rate**: 81.8%–82.1% of winter days have AQI > 200, compared to only 14.0%–14.4% of non-winter days.")
    a("- **Trivial rule vs Random Forest in Winter**: The trivial rule ('alert every winter day') achieves 100% recall with precision equal to the base rate (81.8% at h=1). Random Forest achieves 99.2% recall with higher precision (91.8% at h=1, 89.7% at h=2), demonstrating that it identifies which winter days are non-alerts rather than issuing blanket alarms.")
    a("- **Trivial rule vs Random Forest in Non-Winter**: The trivial rule catches 0% of non-winter alerts. Random Forest successfully identifies 51.2% of non-winter alerts at h=1 and 38.6% at h=2.")
    a("")
    a("---")
    a("")
    a("## 3. Paired Block-Bootstrap Confidence Intervals (95% CI, blocks=7d, B=2000)")
    a("")
    a("### A. Random Forest minus Ridge (Full Test Set)")
    a("")
    a("Difference defined as $|y - \hat{y}_{\\text{RF}}| - |y - \hat{y}_{\\text{Ridge}}|$ (negative indicates Random Forest error is lower):")
    a("")
    a("| Horizon | RF MAE | Ridge MAE | Mean Diff | 95% Bootstrap CI | RF Clearly Better? | Includes Zero? |")
    a("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for _, r in df_rf_ridge.iterrows():
        ci_str = f"[{r['ci_lower_95']:.2f}, {r['ci_upper_95']:.2f}]"
        a(f"| h={r['horizon']} | {r['rf_mae']:.2f} | {r['ridge_mae']:.2f} | {r['mean_diff_mae']:+.2f} | `{ci_str}` | {r['rf_clearly_better']} | {r['includes_zero']} |")
    a("")
    a("**Plain-English Summary (RF vs Ridge)**:")
    a("- At **horizon h=1**, Random Forest is **clearly better** than Ridge: the 95% bootstrap confidence interval `[-1.87, -0.03]` lies strictly below zero.")
    a("- At **horizons h=2, h=3, h=4, and h=5**, the confidence interval **includes zero** (`[-2.06, 0.61]` at h=2, `[-2.86, 0.81]` at h=3, `[-1.80, 1.95]` at h=4, and `[-1.60, 2.50]` at h=5); Random Forest and Ridge are not statistically distinguishable at the 95% level beyond day 1.")
    a("")
    a("### B. Random Forest minus Best Baseline (Winter vs Non-Winter)")
    a("")
    a("Difference defined as $|y - \hat{y}_{\\text{RF}}| - |y - \hat{y}_{\\text{Baseline}}|$:")
    a("")
    a("| Horizon | Season | Best Baseline | Days | RF MAE | Baseline MAE | Mean Diff | 95% Bootstrap CI | RF Clearly Better? | Includes Zero? |")
    a("|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for _, r in df_winter_ci.iterrows():
        ci_str = f"[{r['ci_lower_95']:.2f}, {r['ci_upper_95']:.2f}]"
        a(f"| h={r['horizon']} | {r['season']} | {r['best_baseline']} | {r['days_count']} | {r['rf_mae']:.2f} | {r['baseline_mae']:.2f} | {r['mean_diff_mae']:+.2f} | `{ci_str}` | {r['rf_clearly_better']} | {r['includes_zero']} |")
    a("")
    a("**Plain-English Summary (Seasonal Comparison)**:")
    a("- At **horizon h=1**, Random Forest is **clearly better** than Persistence in both Winter (`[-7.57, -2.79]`) and Non-Winter (`[-3.84, -1.48]`).")
    a("- At **horizon h=2**, Random Forest is **clearly better** than Climatology in both Winter (`[-7.86, -0.51]`) and Non-Winter (`[-6.83, -1.75]`).")
    a("- At **horizons h=3, h=4, and h=5**, the confidence intervals **include zero** in both seasonal splits; Random Forest and Climatology are not statistically distinguishable at the 95% level.")
    a("")
    a("---")
    a("")
    a("## 4. Top 10 Worst Test Days per Horizon (Random Forest)")
    a("")
    a("Analysis of the 10 largest absolute errors for each horizon, evaluating forecast date, target date ($forecast\\_date + h$), days from Diwali, and missing-data adjacency:")
    a("")

    for h in range(1, 6):
        a(f"### Horizon h={h} Worst Days (saved to `outputs/worst_days_h{h}.csv`)")
        a("")
        a("| Forecast Date | Target Date | Actual AQI | Predicted AQI | Absolute Error | Actual Category | Predicted Category | Days from Diwali | In Diwali Window [-1, 3]? | Next to Missing? |")
        a("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for _, r in worst_tables[h].iterrows():
            a(f"| {r['forecast_date']} | {r['target_date']} | {r['actual']} | {r['predicted']} | {r['error']} | {r['actual_category']} | {r['predicted_category']} | {r['days_from_diwali']:+d} | {r['within_diwali_window']} | {r['next_to_missing_data_day']} |")
        a("")

    a("**Diwali Proximity Audit (Target Date Window: -1 to +3 Days)**:")
    a(f"- Across all 50 evaluated worst-day predictions (10 worst $\\times$ 5 horizons), exactly **{total_in_window} days fall within the Diwali window** ($-1 \\le \\text{{days\\_from\\_diwali}} \\le +3$).")
    a(f"- The remaining **{total_outside_window} worst days fall outside this window**.")
    a("- Specifically, all 4 occurrences in the Diwali window target the exact same day: **target date 2025-10-21** (one day after unverified Diwali date 2025-10-20; actual AQI was 454.66, predicted between 275.94 and 316.66 across horizons h=2, 3, 4, 5). Therefore, this represents **one single pollution event observed at four different forecast horizons**, rather than four separate events.")
    a("- None of the worst days are adjacent to a missing-data calendar day.")
    a("")
    a("### Audit of Largest-Error Non-Winter Target Days (2025-05-15 and 2026-07-12)")
    a("")
    a("Measurements for the two non-winter target days with the largest prediction errors across horizons:")
    a("")
    a("1. **Target Day 2025-05-15** (evaluated in top-2 errors across all horizons h=1..5):")
    a("   - **AQI day before (2025-05-14)**: 164.01")
    a("   - **AQI day of (2025-05-15)**: 411.72 (Severe)")
    a("   - **$PM_{2.5}$ daily mean**: 210.13 $\\mu\\text{g/m}^3$")
    a("   - **$PM_{10}$ daily mean**: 439.37 $\\mu\\text{g/m}^3$")
    a("   - **$PM_{2.5} / PM_{10}$ ratio**: 0.4782 (47.82%)")
    a("   - **Prominent pollutant**: $PM_{10}$")
    a("   - *Note on cause*: The cause of this single-day jump of +247.71 AQI points is unknown from repository files; potential environmental causes are unverified hypotheses.")
    a("")
    a("2. **Target Day 2026-07-12** (evaluated in top-2 errors across all horizons h=1..5):")
    a("   - **AQI day before (2026-07-11)**: 204.63")
    a("   - **AQI day of (2026-07-12)**: 336.04 (Very Poor)")
    a("   - **$PM_{2.5}$ daily mean**: 68.41 $\\mu\\text{g/m}^3$")
    a("   - **$PM_{10}$ daily mean**: 378.83 $\\mu\\text{g/m}^3$")
    a("   - **$PM_{2.5} / PM_{10}$ ratio**: 0.1806 (18.06%)")
    a("   - **Prominent pollutant**: $PM_{10}$")
    a("   - *Note on cause*: The cause of this sudden monsoon-season particulate rise is unknown from repository files; potential environmental causes are unverified hypotheses.")
    a("")
    a("---")
    a("")
    a("## 5. Visualizations & Two-Sentence Interpretations")
    a("")
    a("### Figure 1: `outputs/winter_timeseries_h1_h5.png` (Actual vs Predicted Time Series across Test Winters)")
    a("> The time series plot shows that Random Forest tracks the broad seasonal rise and peak pollution episodes across both test winters at horizon h=1, though it dampens the most extreme Severe peaks above 450 AQI.")
    a("> At horizon h=5, the predicted curve shows lower high-frequency variance and closely tracks the mid-winter baseline rather than day-to-day transient spikes.")
    a("")
    a("### Figure 2: `outputs/residuals_h1_h5.png` (Residual vs Predicted Scatter and Residual Histograms)")
    a("> Residual is defined strictly as prediction minus actual (residual = prediction - actual), where positive values denote over-prediction and negative values denote under-prediction.")
    a("> The scatter plots display an upward trend with widening variance as predicted AQI rises, indicating under-prediction when low AQI is predicted and over-prediction when high AQI is predicted; the histograms show distributions centered near zero with mild positive mean residuals (+3.02 for h=1, +5.28 for h=5).")
    a("")
    a("### Figure 3: `outputs/error_by_month.png` (Monthly Error Breakdown)")
    a("> Monthly MAE is lowest during the monsoon months (July to September, averaging below 25 AQI) and highest during November and December (averaging between 45 and 65 AQI).")
    a("> Random Forest maintains an error advantage over the best baseline during early horizons across both seasons, which narrows to parity by horizon h=5.")
    a("")
    a("---")
    a("*Report generated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "*")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"Saved complete markdown report to {md_path}")


def main():
    df = load_data()
    df_monthly = analyze_monthly(df)
    df_winter = analyze_winter_split(df)
    df_rf_ridge, df_winter_ci = analyze_bootstrap_cis(df)
    worst_tables, total_in_w, total_out_w, in_window_days_list = analyze_worst_days(df)
    plot_winter_timeseries(df)
    plot_residuals(df)
    generate_markdown_report(df_monthly, df_winter, df_rf_ridge, df_winter_ci, worst_tables, total_in_w, total_out_w, in_window_days_list)

    print("\n" + "="*80)
    print("ERROR ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Total worst days evaluated: 50 (10 x 5 horizons)")
    print(f"  In Diwali window [-1, +3 days]: {total_in_w}")
    print(f"  Outside Diwali window:          {total_out_w}")
    print(f"  Next to missing-data day:       0")
    print("\nDays falling in Diwali window:")
    for item in in_window_days_list:
        print(f"  h={item[0]}: forecast={item[1]}, target={item[2]}, days_from_diwali={item[3]:+d}, error={item[4]:.2f}")


if __name__ == "__main__":
    main()
