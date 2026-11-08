"""
src/final_evaluate.py

Final Test Set Evaluation for Delhi Air Quality multi-horizon forecasting (h = 1..5).
Strictly adheres to Project Rules:
  - Rule 2: Test set is evaluated ONLY here.
  - Rule 4: Models refit on TRAIN + VALIDATION (gap rows excluded) with already chosen settings.
  - Rule 5: Random seed 42 used for all models and bootstrap resampling.
  - Rule 7: Never invent numbers; all reported numbers come from saved outputs.

Models in scope:
  1. Persistence
  2. Climatology (recomputed from Train + Validation)
  3. Ridge (alpha=100.0 as chosen on validation)
  4. Random Forest (parameters and target scale chosen from validation rf_config.json)

Outputs:
  - outputs/metrics.csv
  - outputs/per_class_metrics.csv
  - outputs/test_predictions.csv
  - outputs/confusion_matrix_rf_h1.png
  - outputs/confusion_matrix_rf_h5.png
  - outputs/confusion_matrices.png
  - outputs/error_vs_horizon.png
  - outputs/alert_metrics_vs_horizon.png
  - outputs/rf_vs_baseline_ci.csv
  - outputs/test_run_log.txt
  - models/rf_h1.pkl ... models/rf_h5.pkl
  - models/ridge_h1.pkl ... models/ridge_h5.pkl
  - models/climatology_train_val.json
"""

import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.split import get_splits
from src.config import RANDOM_SEED
from src.evaluate import (
    evaluate,
    per_class_report,
    skill_vs_persistence,
    skill_vs_best_baseline,
    get_aqi_category_6,
    AQI_CATEGORIES_6,
)


def to_doy365(dt: pd.Timestamp) -> int:
    """Maps date to day-of-year index 0..364, mapping Feb 29 to Feb 28."""
    m, d = dt.month, dt.day
    if m == 2 and d == 29:
        d = 28
    days_before = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    return days_before[m - 1] + (d - 1)


def build_climatology_lookup(df: pd.DataFrame) -> dict[int, float]:
    """
    Computes climatology for each day-of-year (0..364) as the mean AQI of given dataset
    within a +/- 7 day circular window.
    """
    df_dt = pd.to_datetime(df["date"])
    doy = df_dt.apply(to_doy365).values
    aqi = df["aqi"].values

    climatology_map = {}
    for d in range(365):
        diff = np.abs(doy - d)
        circ_diff = np.minimum(diff, 365 - diff)
        climatology_map[d] = float(np.mean(aqi[circ_diff <= 7]))
    return climatology_map


def block_bootstrap_ci(diffs: np.ndarray, block_size: int = 7, n_boot: int = 2000, seed: int = RANDOM_SEED) -> tuple[float, float, float]:
    """
    Computes 95% moving-block bootstrap confidence interval for mean paired differences.
    """
    rng = np.random.default_rng(seed)
    n = len(diffs)
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size

    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        resampled = np.concatenate([diffs[s:s + block_size] for s in starts])[:n]
        boot_means[b] = np.mean(resampled)

    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
    return float(np.mean(diffs)), float(ci_lower), float(ci_upper)


def main():
    parser = argparse.ArgumentParser(description="Final Test Evaluation for Delhi Air Quality Forecasting")
    parser.add_argument("--no-refit", action="store_true", help="Skip refitting on Train+Val; use existing models fit on Train only")
    args = parser.parse_args()

    out_dir = REPO_ROOT / "outputs"
    models_dir = REPO_ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_df, val_df, test_df = get_splits()
    with open(models_dir / "feature_columns.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    with open(models_dir / "rf_config.json", "r", encoding="utf-8") as f:
        rf_config = json.load(f)

    # Combined Train + Validation (gap rows stay excluded)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    X_train_val = train_val_df[feature_cols]
    X_test = test_df[feature_cols]
    test_dates = pd.to_datetime(test_df["date"])

    refit_mode = not args.no_refit
    run_timestamp = datetime.now().isoformat()

    print("="*95)
    print(f"FINAL TEST EVALUATION (Refit Mode: {'ENABLED (Train+Val)' if refit_mode else 'DISABLED (Train only)'})")
    print(f"Test Set Range: {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df)} days)")
    print("="*95)

    # Record run log
    log_lines = [
        f"Final Test Evaluation Run Log",
        f"Timestamp: {run_timestamp}",
        f"Refit on Train+Val: {refit_mode}",
        f"Train rows: {len(train_df)}, Val rows: {len(val_df)}, Combined Train+Val rows: {len(train_val_df)}",
        f"Test rows: {len(test_df)} ({test_df['date'].min()} to {test_df['date'].max()})",
        f"Chosen Settings:",
        f"  Ridge: alpha=100.0 (chosen by TimeSeriesSplit CV on train)",
    ]
    for h in range(1, 6):
        h_cfg = rf_config[f"h{h}"]
        log_lines.append(f"  Random Forest h={h}: target_scale={h_cfg['target_scale']}, params={h_cfg['best_params']}")
    with open(out_dir / "test_run_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    # Climatology lookup
    if refit_mode:
        climatology_map = build_climatology_lookup(train_val_df)
    else:
        climatology_map = build_climatology_lookup(train_df)
    with open(models_dir / "climatology_train_val.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in climatology_map.items()}, f, indent=2)

    # Fit / Load Models
    ridge_models = {}
    rf_models = {}

    for h in range(1, 6):
        target_col = f"target_h{h}"
        y_train_val = train_val_df[target_col].values
        h_cfg = rf_config[f"h{h}"]
        best_rf_params = h_cfg["best_params"]
        use_log = h_cfg["use_log"]

        if refit_mode:
            # 1. Refit Ridge (alpha=100.0)
            ridge_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=100.0, random_state=RANDOM_SEED))
            ])
            ridge_pipe.fit(X_train_val, y_train_val)
            ridge_models[h] = ridge_pipe
            joblib.dump(ridge_pipe, models_dir / f"ridge_h{h}.pkl")

            # 2. Refit Random Forest
            rf_model = RandomForestRegressor(**best_rf_params, random_state=RANDOM_SEED, n_jobs=-1)
            if use_log:
                rf_model.fit(X_train_val, np.log1p(y_train_val))
            else:
                rf_model.fit(X_train_val, y_train_val)
            rf_models[h] = rf_model
            joblib.dump(rf_model, models_dir / f"rf_h{h}.pkl")
        else:
            ridge_pipe = joblib.load(models_dir / f"ridge_h{h}.pkl") if (models_dir / f"ridge_h{h}.pkl").exists() else None
            if ridge_pipe is None:
                # If not saved previously, fit on train only
                ridge_pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=100.0, random_state=RANDOM_SEED))
                ])
                ridge_pipe.fit(train_df[feature_cols], train_df[target_col].values)
            ridge_models[h] = ridge_pipe

            rf_model = joblib.load(models_dir / f"rf_h{h}.pkl")
            rf_models[h] = rf_model

    # 2. Predict Test Set & Compute Metrics
    all_metrics = []
    all_per_class = []
    test_pred_records = []
    rf_confusion_data = {}
    paired_ci_records = []

    compact_results = []

    for h in range(1, 6):
        target_col = f"target_h{h}"
        y_true_h = test_df[target_col].values
        use_log = rf_config[f"h{h}"]["use_log"]

        # Predictions
        # 1. Persistence
        pred_pers = np.clip(test_df["aqi"].values, 0.0, 500.0)

        # 2. Climatology
        target_dts = test_dates + timedelta(days=h)
        pred_clim = np.clip(np.array([climatology_map[to_doy365(td)] for td in target_dts]), 0.0, 500.0)

        # 3. Ridge
        pred_ridge = np.clip(ridge_models[h].predict(X_test), 0.0, 500.0)

        # 4. Random Forest
        raw_rf_pred = rf_models[h].predict(X_test)
        if use_log:
            pred_rf = np.clip(np.expm1(raw_rf_pred), 0.0, 500.0)
        else:
            pred_rf = np.clip(raw_rf_pred, 0.0, 500.0)

        # Evaluate all models
        m_pers = evaluate(y_true_h, pred_pers)
        m_clim = evaluate(y_true_h, pred_clim)
        m_ridge = evaluate(y_true_h, pred_ridge)
        m_rf = evaluate(y_true_h, pred_rf)

        # Per-class reports
        for m_name, preds in [("Persistence", pred_pers), ("Climatology", pred_clim), ("Ridge", pred_ridge), ("Random Forest", pred_rf)]:
            rep = per_class_report(y_true_h, preds)
            rep["model"] = m_name
            rep["horizon"] = h
            all_per_class.append(rep)

        # Base MAEs
        p_mae = m_pers["MAE"]
        c_mae = m_clim["MAE"]
        r_mae = m_ridge["MAE"]
        rf_mae = m_rf["MAE"]

        best_base_mae = min(p_mae, c_mae)

        # Skills
        for m_name, met in [("Persistence", m_pers), ("Climatology", m_clim), ("Ridge", m_ridge), ("Random Forest", m_rf)]:
            cur_mae = met["MAE"]
            s_pers = skill_vs_persistence(cur_mae, p_mae)
            s_best = skill_vs_best_baseline(cur_mae, p_mae, c_mae)
            all_metrics.append({
                "model": m_name,
                "horizon": h,
                **met,
                "skill_vs_persistence": s_pers,
                "skill_vs_best_baseline": s_best,
            })
            compact_results.append({
                "model": m_name,
                "horizon": f"h={h}",
                "MAE": cur_mae,
                "bias": met["bias"],
                "within_one_acc": met["within_one_category_accuracy"],
                "alert_prec": met["alert_precision"],
                "alert_rec": met["alert_recall"],
                "skill_pers": s_pers,
                "skill_best": s_best,
            })

        # Save predictions per date for horizon h
        for d_str, yt, pp, pc, pr, pf in zip(test_df["date"], y_true_h, pred_pers, pred_clim, pred_ridge, pred_rf):
            test_pred_records.append({
                "date": d_str,
                "horizon": h,
                "truth": float(yt),
                "pred_persistence": float(pp),
                "pred_climatology": float(pc),
                "pred_ridge": float(pr),
                "pred_random_forest": float(pf),
            })

        # Store confusion data for RF
        true_cats = [get_aqi_category_6(y) for y in y_true_h]
        pred_cats = [get_aqi_category_6(p) for p in pred_rf]
        rf_confusion_data[h] = (true_cats, pred_cats)

        # 6. Paired Comparison vs Best Baseline
        # Determine best baseline name
        if p_mae <= c_mae:
            best_base_name = "Persistence"
            best_base_preds = pred_pers
            best_b_mae = p_mae
        else:
            best_base_name = "Climatology"
            best_base_preds = pred_clim
            best_b_mae = c_mae

        err_rf = np.abs(y_true_h - pred_rf)
        err_base = np.abs(y_true_h - best_base_preds)
        diff_err = err_rf - err_base  # negative means RF has lower error

        mean_diff, ci_lo, ci_hi = block_bootstrap_ci(diff_err, block_size=7, n_boot=2000, seed=RANDOM_SEED)
        rf_clearly_better = bool(ci_hi < 0.0)
        includes_zero = bool(ci_lo <= 0.0 <= ci_hi)

        paired_ci_records.append({
            "horizon": h,
            "best_baseline": best_base_name,
            "rf_mae": rf_mae,
            "best_baseline_mae": best_b_mae,
            "mean_diff_mae": mean_diff,
            "ci_lower_95": ci_lo,
            "ci_upper_95": ci_hi,
            "rf_clearly_better": rf_clearly_better,
            "includes_zero": includes_zero,
        })

    # Save Tables
    # outputs/metrics.csv
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(out_dir / "metrics.csv", index=False)
    print(f"\nSaved test metrics to {out_dir / 'metrics.csv'}")

    # outputs/per_class_metrics.csv
    df_per_class = pd.concat(all_per_class, ignore_index=True)
    cols_order = ["model", "horizon", "category", "precision", "recall", "f1", "support"]
    df_per_class = df_per_class[cols_order]
    df_per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)
    print(f"Saved per-class metrics to {out_dir / 'per_class_metrics.csv'}")

    # outputs/test_predictions.csv
    df_preds = pd.DataFrame(test_pred_records)
    df_preds.to_csv(out_dir / "test_predictions.csv", index=False)
    print(f"Saved test predictions to {out_dir / 'test_predictions.csv'}")

    # outputs/rf_vs_baseline_ci.csv
    df_ci = pd.DataFrame(paired_ci_records)
    df_ci.to_csv(out_dir / "rf_vs_baseline_ci.csv", index=False)
    print(f"Saved paired bootstrap CI to {out_dir / 'rf_vs_baseline_ci.csv'}")

    # 4. Confusion Matrices
    plot_confusion_matrices(rf_confusion_data, out_dir)

    # 5. Error & Alert vs Horizon plots
    plot_error_and_alert_curves(df_metrics, out_dir)

    # Print Compact Tables
    print("\n" + "="*115)
    print("FINAL TEST EVALUATION RESULTS TABLE (h = 1..5)")
    print("="*115)
    print(f"{'Model':16s} | {'Horizon':7s} | {'MAE':7s} | {'Bias':7s} | {'Within-1 Acc':12s} | {'Alert Prec':10s} | {'Alert Rec':10s} | {'Skill vs Pers':13s} | {'Skill vs Best':13s}")
    print("-" * 115)
    for row in compact_results:
        print(f"{row['model']:16s} | {row['horizon']:7s} | {row['MAE']:7.2f} | {row['bias']:7.2f} | {row['within_one_acc']:12.3f} | {row['alert_prec']:10.3f} | {row['alert_rec']:10.3f} | {row['skill_pers']:13.3f} | {row['skill_best']:13.3f}")
    print("="*115)

    print("\n" + "="*115)
    print("PAIRED BLOCK-BOOTSTRAP COMPARISON: RANDOM FOREST vs BEST BASELINE (95% CI, blocks=7d, B=2000)")
    print("="*115)
    print(f"{'Horizon':7s} | {'Best Baseline':14s} | {'RF MAE':7s} | {'Base MAE':8s} | {'Mean Diff':10s} | {'95% CI':20s} | {'RF Clearly Better':18s} | {'Includes 0':10s}")
    print("-" * 115)
    for _, r in df_ci.iterrows():
        ci_str = f"[{r['ci_lower_95']:.2f}, {r['ci_upper_95']:.2f}]"
        print(f"h={r['horizon']}     | {r['best_baseline']:14s} | {r['rf_mae']:7.2f} | {r['best_baseline_mae']:8.2f} | {r['mean_diff_mae']:10.2f} | {ci_str:20s} | {str(r['rf_clearly_better']):18s} | {str(r['includes_zero']):10s}")
    print("="*115)


def plot_single_cm(true_cats, pred_cats, title, out_path):
    """Plots and saves normalized confusion matrix with raw counts."""
    cm = confusion_matrix(true_cats, pred_cats, labels=AQI_CATEGORIES_6)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    # Annotations: count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(len(AQI_CATEGORIES_6)):
        for j in range(len(AQI_CATEGORIES_6)):
            cnt = cm[i, j]
            pct = cm_norm[i, j] * 100
            annot[i, j] = f"{cnt}\n({pct:.1f}%)"

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=AQI_CATEGORIES_6,
        yticklabels=AQI_CATEGORIES_6,
        cbar=True,
        vmin=0.0,
        vmax=1.0,
        ax=ax
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted AQI Category", fontsize=11)
    ax.set_ylabel("True AQI Category", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {out_path}")


def plot_confusion_matrices(rf_confusion_data, out_dir):
    """Plots separate CMs for h=1 and h=5, and all 5 in one composite figure."""
    # Separate h=1
    plot_single_cm(
        rf_confusion_data[1][0], rf_confusion_data[1][1],
        "Random Forest Confusion Matrix — Horizon h=1 (Day Ahead)",
        out_dir / "confusion_matrix_rf_h1.png"
    )

    # Separate h=5
    plot_single_cm(
        rf_confusion_data[5][0], rf_confusion_data[5][1],
        "Random Forest Confusion Matrix — Horizon h=5 (5 Days Ahead)",
        out_dir / "confusion_matrix_rf_h5.png"
    )

    # All five horizons in one figure (2 rows, 3 cols)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    for h in range(1, 6):
        ax = axes_flat[h - 1]
        t_c, p_c = rf_confusion_data[h]
        cm = confusion_matrix(t_c, p_c, labels=AQI_CATEGORIES_6)
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

        annot = np.empty_like(cm, dtype=object)
        for i in range(len(AQI_CATEGORIES_6)):
            for j in range(len(AQI_CATEGORIES_6)):
                cnt = cm[i, j]
                pct = cm_norm[i, j] * 100
                annot[i, j] = f"{cnt}\n({pct:.0f}%)"

        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=AQI_CATEGORIES_6,
            yticklabels=AQI_CATEGORIES_6,
            cbar=False,
            vmin=0.0,
            vmax=1.0,
            ax=ax
        )
        ax.set_title(f"Random Forest — Horizon h={h}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Category", fontsize=9)
        ax.set_ylabel("True Category", fontsize=9)
        ax.tick_params(axis='x', rotation=45)

    # Hide 6th empty subplot
    axes_flat[5].axis("off")

    fig.suptitle("Random Forest Confusion Matrices across All Horizons h=1..5 (Test Set)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    cm_all_path = out_dir / "confusion_matrices.png"
    plt.savefig(cm_all_path, dpi=300)
    plt.close()
    print(f"Saved all confusion matrices to {cm_all_path}")


def plot_error_and_alert_curves(df_metrics, out_dir):
    """Plots MAE vs horizon and alert precision/recall vs horizon."""
    horizons = [1, 2, 3, 4, 5]
    models = ["Persistence", "Climatology", "Ridge", "Random Forest"]
    colors = {"Persistence": "#7f7f7f", "Climatology": "#17becf", "Ridge": "#1f77b4", "Random Forest": "#2ca02c"}
    markers = {"Persistence": "o", "Climatology": "s", "Ridge": "^", "Random Forest": "D"}

    # 1. Error vs Horizon (MAE)
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        m_df = df_metrics[df_metrics["model"] == m].sort_values("horizon")
        ax.plot(m_df["horizon"], m_df["MAE"], marker=markers[m], color=colors[m], label=m, linewidth=1.8, markersize=7)

    ax.set_title("Test Set MAE by Forecast Horizon (h = 1..5)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Forecast Horizon (Days)", fontsize=11)
    ax.set_ylabel("Mean Absolute Error (AQI)", fontsize=11)
    ax.set_xticks(horizons)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    mae_path = out_dir / "error_vs_horizon.png"
    plt.savefig(mae_path, dpi=300)
    plt.close()
    print(f"Saved error vs horizon figure to {mae_path}")

    # 2. Alert Metrics vs Horizon
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for m in models:
        m_df = df_metrics[df_metrics["model"] == m].sort_values("horizon")
        ax1.plot(m_df["horizon"], m_df["alert_recall"], marker=markers[m], color=colors[m], label=m, linewidth=1.8, markersize=7)
        ax2.plot(m_df["horizon"], m_df["alert_precision"], marker=markers[m], color=colors[m], label=m, linewidth=1.8, markersize=7)

    ax1.set_title("Alert Recall (AQI > 200) by Horizon", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Forecast Horizon (Days)", fontsize=10)
    ax1.set_ylabel("Alert Recall", fontsize=10)
    ax1.set_xticks(horizons)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, fontsize=9)

    ax2.set_title("Alert Precision (AQI > 200) by Horizon", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Forecast Horizon (Days)", fontsize=10)
    ax2.set_ylabel("Alert Precision", fontsize=10)
    ax2.set_xticks(horizons)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, fontsize=9)

    fig.suptitle("Test Set Alert Metrics by Forecast Horizon (AQI > 200)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    alert_path = out_dir / "alert_metrics_vs_horizon.png"
    plt.savefig(alert_path, dpi=300)
    plt.close()
    print(f"Saved alert metrics vs horizon figure to {alert_path}")


if __name__ == "__main__":
    main()
