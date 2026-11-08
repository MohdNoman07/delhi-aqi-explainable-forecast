"""
src/final_evaluate.py

Final Holdout Evaluation Script for Delhi Air Quality Forecasting (h = 1..5).
Strictly adheres to Project Rules:
  - Rule 2: Evaluates on HOLDOUT TEST set (2024-04-06 to 2026-09-25) strictly once.
  - Retrains winning models (Random Forest and Ridge) on combined Train + Validation data
    (buffer gaps strictly excluded). Climatology lookup is refit on Train + Validation.
  - Rule 5: Seed 42 used for all refits, predictions, and bootstrap evaluations.

Models Evaluated:
  1. Persistence
  2. Climatology (lookup table refit on Train+Val)
  3. Damped Anomaly Persistence (rho refit on Train+Val)
  4. Ridge (alpha chosen from validation val_metrics_baselines.csv)
  5. Random Forest (parameters and target scale chosen from validation rf_config.json)

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
  - outputs/bootstrap_block_sensitivity.csv
  - outputs/alert_ci.csv
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

    # Load best baseline configuration chosen on VALIDATION set
    best_baseline_file = models_dir / "best_baseline.json"
    if not best_baseline_file.exists():
        raise FileNotFoundError(f"{best_baseline_file} not found. Run python src/train.py first.")
    with open(best_baseline_file, "r", encoding="utf-8") as f:
        best_baseline_cfg = json.load(f)

    # Combined Train + Validation (gap rows stay excluded)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    X_train_val = train_val_df[feature_cols]

    X_test = test_df[feature_cols]
    test_dates = pd.to_datetime(test_df["date"])

    refit_mode = not args.no_refit

    log_lines = [
        "="*80,
        f"DELHI AQI FINAL TEST EVALUATION RUN LOG — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*80,
        f"Refit on Train+Val: {refit_mode}",
        f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}",
        f"Combined Train+Val shape: {train_val_df.shape}",
        f"Test date range: {test_df['date'].min()} to {test_df['date'].max()} (N = {len(test_df)})",
        f"Number of features: {len(feature_cols)}",
        "Hyperparameters loaded from models/rf_config.json and models/best_baseline.json:",
    ]
    for h in range(1, 6):
        cfg = rf_config[f"h{h}"]
        b_name = best_baseline_cfg[str(h)]["best_baseline_name"]
        b_val_mae = best_baseline_cfg[str(h)]["best_baseline_val_mae"]
        log_lines.append(
            f"  Horizon h={h}: RF scale={cfg['target_scale']} (use_log={cfg['use_log']}), "
            f"best baseline on validation={b_name} (val MAE={b_val_mae:.2f}), "
            f"params={cfg['best_params']}"
        )

    # Climatology lookup
    if refit_mode:
        climatology_map = build_climatology_lookup(train_val_df)
    else:
        climatology_map = build_climatology_lookup(train_df)
    with open(models_dir / "climatology_train_val.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in climatology_map.items()}, f, indent=2)

    # Damped Anomaly Persistence: fit rho on train_val (or train)
    fit_df = train_val_df if refit_mode else train_df
    fit_dates = pd.to_datetime(fit_df["date"])
    fit_doy = np.array([to_doy365(d) for d in fit_dates])
    fit_clim = np.array([climatology_map[d] for d in fit_doy])
    fit_anom = fit_df["aqi"].values - fit_clim

    damped_rho_models = {}
    for h in range(1, 6):
        fit_target_doy = np.array([to_doy365(d) for d in fit_dates + timedelta(days=h)])
        fit_target_clim = np.array([climatology_map[d] for d in fit_target_doy])
        fit_target_anom = fit_df[f"target_h{h}"].values - fit_target_clim
        denom = np.sum(fit_anom ** 2)
        damped_rho_models[h] = float(np.sum(fit_anom * fit_target_anom) / denom) if denom > 0 else 0.0

    # Read Ridge chosen alpha from validation baselines
    val_base_path = REPO_ROOT / "outputs" / "val_metrics_baselines.csv"
    val_base_df = pd.read_csv(val_base_path) if val_base_path.exists() else None

    # Fit / Load Models
    ridge_models = {}
    rf_models = {}

    for h in range(1, 6):
        target_col = f"target_h{h}"
        y_train_val = train_val_df[target_col].values
        h_cfg = rf_config[f"h{h}"]
        best_rf_params = h_cfg["best_params"]
        use_log = h_cfg["use_log"]

        if val_base_df is not None:
            r_sub = val_base_df[(val_base_df["model"] == "Ridge") & (val_base_df["horizon"] == h)]
            alpha_h = float(r_sub["best_alpha"].iloc[0]) if not r_sub.empty and "best_alpha" in r_sub.columns else 100.0
        else:
            alpha_h = 100.0

        if refit_mode:
            # TODO: Ridge receives raw integer month and day_of_week — do not change feature list
            ridge_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha_h, random_state=RANDOM_SEED))
            ])
            ridge_pipe.fit(X_train_val, y_train_val)
            ridge_models[h] = ridge_pipe
            joblib.dump(ridge_pipe, models_dir / f"ridge_h{h}.pkl")

            # Refit Random Forest
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
                # TODO: Ridge receives raw integer month and day_of_week — do not change feature list
                ridge_pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha_h, random_state=RANDOM_SEED))
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
    rf_preds_by_h = {}
    diff_err_by_h = {}

    compact_results = []

    test_doy = np.array([to_doy365(d) for d in test_dates])
    test_clim_base = np.array([climatology_map[d] for d in test_doy])
    test_anom = test_df["aqi"].values - test_clim_base

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

        # 3. Damped Anomaly Persistence
        pred_damped = np.clip(pred_clim + damped_rho_models[h] * test_anom, 0.0, 500.0)

        # 4. Ridge
        pred_ridge = np.clip(ridge_models[h].predict(X_test), 0.0, 500.0)

        # 5. Random Forest
        raw_rf_pred = rf_models[h].predict(X_test)
        if use_log:
            pred_rf = np.clip(np.expm1(raw_rf_pred), 0.0, 500.0)
        else:
            pred_rf = np.clip(raw_rf_pred, 0.0, 500.0)
        rf_preds_by_h[h] = pred_rf

        # Evaluate all models
        m_pers = evaluate(y_true_h, pred_pers)
        m_clim = evaluate(y_true_h, pred_clim)
        m_dp = evaluate(y_true_h, pred_damped)
        m_ridge = evaluate(y_true_h, pred_ridge)
        m_rf = evaluate(y_true_h, pred_rf)

        # Per-class reports
        for m_name, preds in [
            ("Persistence", pred_pers),
            ("Climatology", pred_clim),
            ("Damped Anomaly Persistence", pred_damped),
            ("Ridge", pred_ridge),
            ("Random Forest", pred_rf),
        ]:
            rep = per_class_report(y_true_h, preds)
            rep["model"] = m_name
            rep["horizon"] = h
            all_per_class.append(rep)

        # Base MAEs
        p_mae = m_pers["MAE"]
        c_mae = m_clim["MAE"]
        dp_mae = m_dp["MAE"]
        r_mae = m_ridge["MAE"]
        rf_mae = m_rf["MAE"]

        # Fixed selection chosen on VALIDATION set
        chosen_base_name = best_baseline_cfg[str(h)]["best_baseline_name"]
        if chosen_base_name == "Damped Anomaly Persistence":
            best_base_preds = pred_damped
            best_base_mae = dp_mae
        elif chosen_base_name == "Persistence":
            best_base_preds = pred_pers
            best_base_mae = p_mae
        else:
            best_base_preds = pred_clim
            best_base_mae = c_mae

        # Skills
        for m_name, met in [
            ("Persistence", m_pers),
            ("Climatology", m_clim),
            ("Damped Anomaly Persistence", m_dp),
            ("Ridge", m_ridge),
            ("Random Forest", m_rf),
        ]:
            cur_mae = met["MAE"]
            s_pers = skill_vs_persistence(cur_mae, p_mae)
            s_best = skill_vs_best_baseline(cur_mae, best_base_mae)
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
                "accuracy": met["accuracy"],
                "within_one_acc": met["within_one_category_accuracy"],
                "alert_prec": met["alert_precision"],
                "alert_rec": met["alert_recall"],
                "skill_pers": s_pers,
                "skill_best": s_best,
            })

        # Save predictions per date for horizon h
        for d_str, yt, pp, pc, pdp, pr, pf in zip(
            test_df["date"], y_true_h, pred_pers, pred_clim, pred_damped, pred_ridge, pred_rf
        ):
            test_pred_records.append({
                "date": d_str,
                "horizon": h,
                "truth": float(yt),
                "pred_persistence": float(pp),
                "pred_climatology": float(pc),
                "pred_damped_persistence": float(pdp),
                "pred_ridge": float(pr),
                "pred_random_forest": float(pf),
            })

        # Store confusion data for RF
        true_cats = [get_aqi_category_6(y) for y in y_true_h]
        pred_cats = [get_aqi_category_6(p) for p in pred_rf]
        rf_confusion_data[h] = (true_cats, pred_cats)

        # 6. Paired Comparison vs Best Baseline
        err_rf = np.abs(y_true_h - pred_rf)
        err_base = np.abs(y_true_h - best_base_preds)
        diff_err = err_rf - err_base  # negative means RF has lower error
        diff_err_by_h[h] = (diff_err, chosen_base_name)

        mean_diff, ci_lo, ci_hi = block_bootstrap_ci(diff_err, block_size=7, n_boot=2000, seed=RANDOM_SEED)
        rf_clearly_better = bool(ci_hi < 0.0)
        includes_zero = bool(ci_lo <= 0.0 <= ci_hi)

        paired_ci_records.append({
            "horizon": h,
            "best_baseline": chosen_base_name,
            "rf_mae": rf_mae,
            "best_baseline_mae": best_base_mae,
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

    # outputs/bootstrap_block_sensitivity.csv (blocks 7, 14, 21, 30 days)
    sensitivity_records = []
    for h in range(1, 6):
        diff_err_h, b_name = diff_err_by_h[h]
        for bs in [7, 14, 21, 30]:
            md, lo, hi = block_bootstrap_ci(diff_err_h, block_size=bs, n_boot=2000, seed=RANDOM_SEED)
            sensitivity_records.append({
                "horizon": h,
                "best_baseline": b_name,
                "block_size_days": bs,
                "mean_diff_mae": md,
                "ci_lower_95": lo,
                "ci_upper_95": hi,
                "rf_clearly_better": bool(hi < 0.0),
                "includes_zero": bool(lo <= 0.0 <= hi)
            })
    df_sens = pd.DataFrame(sensitivity_records)
    df_sens.to_csv(out_dir / "bootstrap_block_sensitivity.csv", index=False)
    print(f"Saved bootstrap block sensitivity table to {out_dir / 'bootstrap_block_sensitivity.csv'}")

    # outputs/alert_ci.csv (block bootstrap CI for RF vs trivial seasonal rule)
    alert_ci_records = []
    test_dates_dt = pd.to_datetime(test_df["date"])
    is_winter = test_dates_dt.dt.month.isin([10, 11, 12, 1, 2]).values
    n_test = len(test_df)
    block_size = 7
    n_blocks = int(np.ceil(n_test / block_size))

    for h in range(1, 6):
        y_true = test_df[f"target_h{h}"].values
        y_rf = rf_preds_by_h[h]

        rng = np.random.default_rng(RANDOM_SEED)
        prec_diffs = []
        rec_diffs = []
        rf_precs = []
        triv_precs = []
        rf_recs = []
        triv_recs = []

        for _ in range(2000):
            starts = rng.integers(0, n_test - block_size + 1, size=n_blocks)
            sample_idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n_test]

            yt_s = y_true[sample_idx]
            yrf_s = y_rf[sample_idx]
            w_s = is_winter[sample_idx]

            true_alert = yt_s > 200
            rf_alert = yrf_s > 200
            triv_alert = w_s

            rf_p = np.sum(true_alert & rf_alert) / np.sum(rf_alert) if np.sum(rf_alert) > 0 else 0.0
            tr_p = np.sum(true_alert & triv_alert) / np.sum(triv_alert) if np.sum(triv_alert) > 0 else 0.0

            total_true = np.sum(true_alert)
            rf_r = np.sum(true_alert & rf_alert) / total_true if total_true > 0 else 0.0
            tr_r = np.sum(true_alert & triv_alert) / total_true if total_true > 0 else 0.0

            rf_precs.append(rf_p)
            triv_precs.append(tr_p)
            rf_recs.append(rf_r)
            triv_recs.append(tr_r)
            prec_diffs.append(rf_p - tr_p)
            rec_diffs.append(rf_r - tr_r)

        p_lo, p_hi = np.percentile(prec_diffs, [2.5, 97.5])
        r_lo, r_hi = np.percentile(rec_diffs, [2.5, 97.5])

        alert_ci_records.append({
            "horizon": h,
            "rf_alert_prec": float(np.mean(rf_precs)),
            "trivial_alert_prec": float(np.mean(triv_precs)),
            "prec_diff_mean": float(np.mean(prec_diffs)),
            "prec_ci_lower_95": float(p_lo),
            "prec_ci_upper_95": float(p_hi),
            "rf_alert_rec": float(np.mean(rf_recs)),
            "trivial_alert_rec": float(np.mean(triv_recs)),
            "rec_diff_mean": float(np.mean(rec_diffs)),
            "rec_ci_lower_95": float(r_lo),
            "rec_ci_upper_95": float(r_hi),
        })
    df_alert_ci = pd.DataFrame(alert_ci_records)
    df_alert_ci.to_csv(out_dir / "alert_ci.csv", index=False)
    print(f"Saved alert CI table to {out_dir / 'alert_ci.csv'}")

    # 4. Confusion Matrices
    plot_confusion_matrices(rf_confusion_data, out_dir)

    # 5. Error & Alert vs Horizon plots
    plot_error_and_alert_curves(df_metrics, out_dir)

    # Print Compact Tables
    print("\n" + "="*115)
    print("FINAL TEST EVALUATION RESULTS TABLE (h = 1..5)")
    print("="*115)
    print(f"{'Model':28s} | {'Horizon':7s} | {'MAE':7s} | {'Bias':7s} | {'Accuracy':9s} | {'Alert Prec':10s} | {'Alert Rec':10s} | {'Skill vs Pers':13s} | {'Skill vs Best':13s}")
    print("-" * 115)
    for row in compact_results:
        print(f"{row['model']:28s} | {row['horizon']:7s} | {row['MAE']:7.2f} | {row['bias']:7.2f} | {row['accuracy']:9.3f} | {row['alert_prec']:10.3f} | {row['alert_rec']:10.3f} | {row['skill_pers']:13.3f} | {row['skill_best']:13.3f}")
    print("="*115)

    print("\n" + "="*115)
    print("PAIRED BLOCK-BOOTSTRAP COMPARISON: RANDOM FOREST vs BEST BASELINE (95% CI, blocks=7d, B=2000)")
    print("="*115)
    print(f"{'Horizon':7s} | {'Best Baseline':28s} | {'RF MAE':7s} | {'Base MAE':8s} | {'Mean Diff':10s} | {'95% CI':20s} | {'RF Clearly Better':18s} | {'Includes 0':10s}")
    print("-" * 115)
    for _, r in df_ci.iterrows():
        ci_str = f"[{r['ci_lower_95']:.2f}, {r['ci_upper_95']:.2f}]"
        print(f"h={r['horizon']}     | {r['best_baseline']:28s} | {r['rf_mae']:7.2f} | {r['best_baseline_mae']:8.2f} | {r['mean_diff_mae']:10.2f} | {ci_str:20s} | {str(r['rf_clearly_better']):18s} | {str(r['includes_zero']):10s}")
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
    plot_single_cm(
        rf_confusion_data[1][0], rf_confusion_data[1][1],
        "Random Forest Confusion Matrix — Horizon h=1 (Day Ahead)",
        out_dir / "confusion_matrix_rf_h1.png"
    )

    plot_single_cm(
        rf_confusion_data[5][0], rf_confusion_data[5][1],
        "Random Forest Confusion Matrix — Horizon h=5 (5 Days Ahead)",
        out_dir / "confusion_matrix_rf_h5.png"
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    for idx, h in enumerate(range(1, 6)):
        ax = axes_flat[idx]
        true_cats, pred_cats = rf_confusion_data[h]
        cm = confusion_matrix(true_cats, pred_cats, labels=AQI_CATEGORIES_6)
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

        annot = np.empty_like(cm, dtype=object)
        for i in range(len(AQI_CATEGORIES_6)):
            for j in range(len(AQI_CATEGORIES_6)):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.0f}%)"

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
            ax=ax,
            annot_kws={"size": 8}
        )
        ax.set_title(f"Horizon h={h} ({h} Day{'s' if h > 1 else ''} Ahead)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Category", fontsize=9)
        ax.set_ylabel("True Category", fontsize=9)
        ax.tick_params(labelsize=8)

    axes_flat[5].axis("off")
    fig.suptitle("Random Forest Confusion Matrices Across Horizons (h=1..5)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    cm_all_path = out_dir / "confusion_matrices.png"
    plt.savefig(cm_all_path, dpi=300)
    plt.close()
    print(f"Saved all confusion matrices to {cm_all_path}")


def plot_error_and_alert_curves(df_metrics, out_dir):
    """Plots MAE vs horizon and alert precision/recall vs horizon."""
    horizons = [1, 2, 3, 4, 5]
    models = ["Persistence", "Climatology", "Damped Anomaly Persistence", "Ridge", "Random Forest"]
    colors = {
        "Persistence": "#7f7f7f",
        "Climatology": "#17becf",
        "Damped Anomaly Persistence": "#9467bd",
        "Ridge": "#1f77b4",
        "Random Forest": "#2ca02c",
    }
    markers = {
        "Persistence": "o",
        "Climatology": "s",
        "Damped Anomaly Persistence": "v",
        "Ridge": "^",
        "Random Forest": "D",
    }

    # 1. Error vs Horizon (MAE)
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        m_df = df_metrics[df_metrics["model"] == m].sort_values("horizon")
        if not m_df.empty:
            ax.plot(m_df["horizon"], m_df["MAE"], marker=markers[m], color=colors[m], label=m, linewidth=1.8, markersize=7)

    ax.set_title("Test Set MAE by Forecast Horizon (h = 1..5)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Forecast Horizon (Days)", fontsize=11)
    ax.set_ylabel("Mean Absolute Error (AQI)", fontsize=11)
    ax.set_xticks(horizons)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    mae_path = out_dir / "error_vs_horizon.png"
    plt.savefig(mae_path, dpi=300)
    plt.close()
    print(f"Saved error vs horizon figure to {mae_path}")

    # 2. Alert Metrics vs Horizon
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for m in models:
        m_df = df_metrics[df_metrics["model"] == m].sort_values("horizon")
        if not m_df.empty:
            ax1.plot(m_df["horizon"], m_df["alert_recall"], marker=markers[m], color=colors[m], label=m, linewidth=1.8, markersize=7)
            ax2.plot(m_df["horizon"], m_df["alert_precision"], marker=markers[m], color=colors[m], label=m, linewidth=1.8, markersize=7)

    ax1.set_title("Alert Recall (AQI > 200) by Horizon", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Forecast Horizon (Days)", fontsize=10)
    ax1.set_ylabel("Alert Recall", fontsize=10)
    ax1.set_xticks(horizons)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, fontsize=8)

    ax2.set_title("Alert Precision (AQI > 200) by Horizon", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Forecast Horizon (Days)", fontsize=10)
    ax2.set_ylabel("Alert Precision", fontsize=10)
    ax2.set_xticks(horizons)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, fontsize=8)

    fig.suptitle("Test Set Alert Metrics by Forecast Horizon (AQI > 200)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    alert_path = out_dir / "alert_metrics_vs_horizon.png"
    plt.savefig(alert_path, dpi=300)
    plt.close()
    print(f"Saved alert metrics vs horizon figure to {alert_path}")


if __name__ == "__main__":
    main()
