"""
src/predict.py

Command-line multi-horizon AQI prediction demo for Delhi air quality forecasting.
Usage:
    python src/predict.py --date YYYY-MM-DD
    python src/predict.py --latest

Loads final Random Forest models (models/rf_h1.pkl ... rf_h5.pkl) and feature columns.
Extracts features using src/features.py (build_feature_row_for_date) strictly from history.
Never retrains or refits models.
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import get_aqi_category_6
from src.features import build_feature_row_for_date

# Health Advisories per CPCB AQI Category
ADVISORIES = {
    "Good": "normal outdoor activity",
    "Satisfactory": "unusually sensitive people should reduce prolonged outdoor exertion",
    "Moderate": "people with breathing or heart conditions, children and older adults should reduce prolonged outdoor exertion",
    "Poor": "avoid prolonged outdoor exertion",
    "Very Poor": "avoid outdoor activity where possible",
    "Severe": "stay indoors and avoid exertion",
}

# Standard Category Color Scheme
CATEGORY_COLORS = {
    "Good": "#00e400",
    "Satisfactory": "#92d050",
    "Moderate": "#ffff00",
    "Poor": "#ff7e00",
    "Very Poor": "#ff0000",
    "Severe": "#7e0023",
}


def load_resources():
    """Loads models, feature definitions, and test evaluation metrics."""
    models_dir = REPO_ROOT / "models"
    feat_path = models_dir / "feature_columns.json"
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature columns JSON not found at {feat_path}")
    with open(feat_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    models = {}
    for h in range(1, 6):
        m_path = models_dir / f"rf_h{h}.pkl"
        if not m_path.exists():
            raise FileNotFoundError(f"Trained model not found at {m_path}")
        models[h] = joblib.load(m_path)

    # Read typical errors from outputs/metrics.csv
    metrics_path = REPO_ROOT / "outputs" / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}")
    metrics_df = pd.read_csv(metrics_path)
    rf_metrics = metrics_df[metrics_df["model"] == "Random Forest"].set_index("horizon")
    typical_maes = {h: float(rf_metrics.loc[h, "MAE"]) for h in range(1, 6)}

    # Read statistical equivalence note from outputs/rf_vs_baseline_ci.csv
    ci_path = REPO_ROOT / "outputs" / "rf_vs_baseline_ci.csv"
    ci_notes = {}
    if ci_path.exists():
        ci_df = pd.read_csv(ci_path).set_index("horizon")
        for h in range(1, 6):
            if h in ci_df.index and bool(ci_df.loc[h, "includes_zero"]):
                ci_notes[h] = "at this horizon the model is statistically indistinguishable from a seasonal average"

    return feature_cols, models, typical_maes, ci_notes


def run_prediction(date_str: str):
    """Generates 1 to 5 day forecasts for date_str."""
    daily_path = REPO_ROOT / "data" / "processed" / "clean_daily.csv"
    if not daily_path.exists():
        raise FileNotFoundError(f"clean_daily.csv not found at {daily_path}")
    daily_df = pd.read_csv(daily_path)
    daily_df["date"] = pd.to_datetime(daily_df["date"])

    t_dt = pd.to_datetime(date_str)
    date_str_fmt = t_dt.strftime("%Y-%m-%d")

    # Build feature vector
    feat_row = build_feature_row_for_date(daily_df, t_dt)
    feature_cols, models, typical_maes, ci_notes = load_resources()

    X_in = feat_row[feature_cols]

    # Look up actuals in clean_daily
    daily_map = daily_df.set_index("date")["aqi"].to_dict()

    forecast_records = []
    any_poor_or_worse = False

    for h in range(1, 6):
        target_dt = t_dt + timedelta(days=h)
        target_str = target_dt.strftime("%Y-%m-%d")

        raw_pred = float(models[h].predict(X_in)[0])
        pred_clipped = float(np.clip(raw_pred, 0.0, 500.0))
        cat = get_aqi_category_6(pred_clipped)
        advisory = ADVISORIES[cat]
        typical_err = typical_maes[h]

        if pred_clipped > 200.0:
            any_poor_or_worse = True

        actual_aqi = daily_map.get(target_dt, np.nan)
        actual_val_str = f"{actual_aqi:.2f}" if not np.isnan(actual_aqi) else "N/A"

        note = ci_notes.get(h, "")

        forecast_records.append({
            "horizon": f"Day +{h}",
            "horizon_num": h,
            "forecast_date": date_str_fmt,
            "target_date": target_str,
            "predicted_aqi": round(pred_clipped, 2),
            "category": cat,
            "advisory": advisory,
            "typical_mae": round(typical_err, 2),
            "actual_aqi": actual_val_str,
            "actual_aqi_num": actual_aqi,
            "note": note,
        })

    df_fc = pd.DataFrame(forecast_records)

    # ── Console Output ──────────────────────────────────────────────────────────
    print("\n" + "="*95)
    print(f"DELHI AIR QUALITY 5-DAY FORWARD FORECAST | BASE DATE: {date_str_fmt}")
    print("="*95)

    if any_poor_or_worse:
        alert_days = [r["horizon"] for r in forecast_records if r["predicted_aqi"] > 200.0]
        print(f"⚠️  ALERT: High pollution episode forecasted (AQI > 200, Poor or worse) on {', '.join(alert_days)}!")
        print("-" * 95)

    header = f"{'Horizon':<8} | {'Target Date':<11} | {'Pred AQI':<8} | {'Category':<13} | {'Typical Error':<13} | {'Actual AQI':<10} | {'Advisory'}"
    print(header)
    print("-" * 95)
    for r in forecast_records:
        row_str = (
            f"{r['horizon']:<8} | {r['target_date']:<11} | {r['predicted_aqi']:<8.2f} | "
            f"{r['category']:<13} | ±{r['typical_mae']:<12.2f} | {r['actual_aqi']:<10} | {r['advisory']}"
        )
        print(row_str)
        if r["note"]:
            print(f"         * Note: {r['note']}")

    print("-" * 95)
    print("Informational only, not medical advice.")
    print("=" * 95)

    # ── Save Outputs ────────────────────────────────────────────────────────────
    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save CSV
    out_csv = out_dir / f"forecast_{date_str_fmt}.csv"
    save_cols = ["horizon", "forecast_date", "target_date", "predicted_aqi", "category", "typical_mae", "actual_aqi", "advisory", "note"]
    df_fc[save_cols].to_csv(out_csv, index=False)
    print(f"\nSaved forecast table to {out_csv}")

    # 2. Save PNG Plot (14 days past actuals + 5 days forecast)
    plot_forecast(daily_df, t_dt, df_fc, out_dir / f"forecast_{date_str_fmt}.png")

    return df_fc


def plot_forecast(daily_df: pd.DataFrame, base_dt: pd.Timestamp, df_fc: pd.DataFrame, out_path: Path):
    """Plots previous 14 days of actual AQI followed by 5 days of forecast."""
    # Past 14 days history
    start_hist = base_dt - timedelta(days=14)
    hist_mask = (daily_df["date"] >= start_hist) & (daily_df["date"] <= base_dt)
    hist_df = daily_df[hist_mask].copy().sort_values("date").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 6))

    # Plot past actuals
    hist_dates = [d.strftime("%m-%d") for d in hist_df["date"]]
    hist_aqis = hist_df["aqi"].values
    hist_colors = [CATEGORY_COLORS.get(get_aqi_category_6(a) if not np.isnan(a) else "Moderate", "#999999") for a in hist_aqis]

    bars_hist = ax.bar(hist_dates, hist_aqis, color=hist_colors, alpha=0.65, edgecolor="black", linewidth=0.8, label="Historical Actual AQI")

    # Plot forecast days
    fc_dates = [pd.to_datetime(r["target_date"]).strftime("%m-%d") + f"\n(+{r['horizon_num']}d)" for _, r in df_fc.iterrows()]
    fc_aqis = df_fc["predicted_aqi"].values
    fc_colors = [CATEGORY_COLORS.get(r["category"], "#ff7e00") for _, r in df_fc.iterrows()]

    bars_fc = ax.bar(
        fc_dates, fc_aqis,
        color=fc_colors, alpha=0.95, edgecolor="black", linewidth=1.5, hatch="//",
        label="Predicted Forecast (Day +1..+5)"
    )

    # Threshold lines
    ax.axhline(100, color="#92d050", linestyle=":", alpha=0.7, label="Satisfactory (100)")
    ax.axhline(200, color="#ff7e00", linestyle="--", linewidth=1.1, alpha=0.8, label="Poor Alert (200)")
    ax.axhline(300, color="#ff0000", linestyle=":", alpha=0.7, label="Very Poor (300)")
    ax.axhline(400, color="#7e0023", linestyle="--", linewidth=1.2, alpha=0.8, label="Severe (400)")

    # Annotate forecast values on top of bars
    for bar in bars_fc:
        height = bar.get_height()
        ax.annotate(
            f"{height:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold"
        )

    base_str = base_dt.strftime("%Y-%m-%d")
    ax.set_title(f"Delhi AQI Forecast: Base Date {base_str} (Past 14 Days History + 5-Day Forward Prediction)", fontsize=12, fontweight="bold")
    ax.set_ylabel("AQI", fontsize=11)
    ax.set_xlabel("Date (Month-Day)", fontsize=11)
    ax.set_ylim(0, max(520, max(hist_aqis.max() if len(hist_aqis) else 0, fc_aqis.max()) + 30))
    ax.grid(True, linestyle=":", alpha=0.55)

    # Clean legend
    handles, labels = ax.get_legend_handles_labels()
    # deduplicate
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved forecast figure to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-horizon Delhi AQI forecasting demo.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", type=str, help="Forecast base date in YYYY-MM-DD format.")
    group.add_argument("--latest", action="store_true", help="Use the latest available date in clean_daily.csv.")

    args = parser.parse_args()

    daily_path = REPO_ROOT / "data" / "processed" / "clean_daily.csv"
    if not daily_path.exists():
        raise FileNotFoundError(f"clean_daily.csv not found at {daily_path}")
    daily_df = pd.read_csv(daily_path)

    if args.latest:
        target_date = daily_df["date"].max()
        print(f"Selecting latest date from daily observations: {target_date}")
    else:
        target_date = args.date

    run_prediction(target_date)


if __name__ == "__main__":
    main()
