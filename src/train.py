"""
src/train.py

Trains and evaluates forecasting models on Delhi Air Quality multi-horizon forecasting (h = 1..5).
Strictly adheres to Project Rules:
  - Rule 2: Evaluates on VALIDATION set only. Test set is NEVER touched here.
  - Rule 4: Scalers and models fit on training data only.
  - Rule 5: Random seed 42 used for all models.

Models:
  1. Baselines: Persistence, Climatology, Ridge
  2. Random Forest Regressor: tuned via RandomizedSearchCV (n_iter=30, scoring='neg_mean_absolute_error',
     cv=TimeSeriesSplit(CV_SPLITS, CV_GAP)) on TRAIN set only, evaluated for raw AQI and log1p(AQI).

Outputs:
  - outputs/val_metrics_baselines.csv
  - outputs/val_per_class_baselines.csv
  - outputs/val_predictions_baselines.csv
  - outputs/rf_best_params.json
  - outputs/val_metrics_rf.csv
  - outputs/val_per_class_rf.csv
  - outputs/val_predictions_rf.csv
  - outputs/rf_cv_results.csv
  - outputs/rf_feature_importance.png
  - models/rf_h1.pkl ... models/rf_h5.pkl
  - models/rf_config.json
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.split import get_splits
from src.config import CV_SPLITS, CV_GAP, RANDOM_SEED
from src.evaluate import (
    evaluate,
    per_class_report,
    skill_vs_persistence,
    skill_vs_best_baseline,
)


def to_doy365(dt: pd.Timestamp) -> int:
    """Maps date to day-of-year index 0..364, mapping Feb 29 to Feb 28."""
    m, d = dt.month, dt.day
    if m == 2 and d == 29:
        d = 28
    days_before = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    return days_before[m - 1] + (d - 1)


def build_climatology_lookup(train_df: pd.DataFrame) -> dict[int, float]:
    """
    Computes climatology for each day-of-year (0..364) as the mean AQI of TRAIN set
    within a +/- 7 day circular window. Uses training data only.
    """
    train_dt = pd.to_datetime(train_df["date"])
    train_doy = train_dt.apply(to_doy365).values
    train_aqi = train_df["aqi"].values

    climatology_map = {}
    for doy in range(365):
        diff = np.abs(train_doy - doy)
        circ_diff = np.minimum(diff, 365 - diff)
        climatology_map[doy] = float(np.mean(train_aqi[circ_diff <= 7]))
    return climatology_map


def compute_damped_persistence(train_df: pd.DataFrame, eval_df: pd.DataFrame, climatology_map: dict[int, float], h: int) -> tuple[np.ndarray, float]:
    """
    Computes damped anomaly persistence baseline for horizon h.
    Anomaly rho_h is fit strictly on TRAIN data via OLS through the origin:
      a_{t+h} = rho_h * a_t + eps
    Predictions for eval_df are then:
      y_hat_{t+h} = clip(C(doy(t+h)) + rho_h * (y_t - C(doy(t))), 0, 500)
    """
    train_dates = pd.to_datetime(train_df["date"])
    train_doy = np.array([to_doy365(d) for d in train_dates])
    train_clim = np.array([climatology_map[d] for d in train_doy])
    train_anom = train_df["aqi"].values - train_clim

    train_target_doy = np.array([to_doy365(d) for d in train_dates + timedelta(days=h)])
    train_target_clim = np.array([climatology_map[d] for d in train_target_doy])
    train_target_anom = train_df[f"target_h{h}"].values - train_target_clim

    denom = np.sum(train_anom ** 2)
    rho_h = float(np.sum(train_anom * train_target_anom) / denom) if denom > 0 else 0.0

    eval_dates = pd.to_datetime(eval_df["date"])
    eval_doy = np.array([to_doy365(d) for d in eval_dates])
    eval_clim = np.array([climatology_map[d] for d in eval_doy])
    eval_anom = eval_df["aqi"].values - eval_clim

    eval_target_doy = np.array([to_doy365(d) for d in eval_dates + timedelta(days=h)])
    eval_target_clim = np.array([climatology_map[d] for d in eval_target_doy])

    pred = np.clip(eval_target_clim + rho_h * eval_anom, 0.0, 500.0)
    return pred, rho_h


def run_baselines(train_df, val_df, feature_cols, climatology_map, cv):
    """Evaluates Persistence, Climatology, and Ridge on Validation set for h=1..5."""
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    val_dates = pd.to_datetime(val_df["date"])

    ridge_param_grid = {"ridge__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}

    all_metrics = []
    all_per_class = []
    val_preds_records = []
    compact_summary = []
    baseline_maes = {}

    print("\n" + "="*105)
    print("RUNNING MULTI-HORIZON BASELINES ON VALIDATION SET (h = 1..5)")
    print("="*105)

    best_baseline_records = {}

    for h in range(1, 6):
        target_col = f"target_h{h}"
        y_train_h = train_df[target_col].values
        y_val_h = val_df[target_col].values

        # 1. Persistence
        pred_pers = val_df["aqi"].values
        met_pers = evaluate(y_val_h, pred_pers)
        rep_pers = per_class_report(y_val_h, pred_pers)
        rep_pers["model"] = "Persistence"
        rep_pers["horizon"] = h
        all_per_class.append(rep_pers)

        # 2. Climatology
        target_dates_h = val_dates + timedelta(days=h)
        pred_clim = np.array([climatology_map[to_doy365(t_dt)] for t_dt in target_dates_h])
        met_clim = evaluate(y_val_h, pred_clim)
        rep_clim = per_class_report(y_val_h, pred_clim)
        rep_clim["model"] = "Climatology"
        rep_clim["horizon"] = h
        all_per_class.append(rep_clim)

        # 3. Damped Anomaly Persistence
        pred_damped, rho_h = compute_damped_persistence(train_df, val_df, climatology_map, h)
        met_dp = evaluate(y_val_h, pred_damped)
        rep_dp = per_class_report(y_val_h, pred_damped)
        rep_dp["model"] = "Damped Anomaly Persistence"
        rep_dp["horizon"] = h
        all_per_class.append(rep_dp)

        # 4. Ridge
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(random_state=RANDOM_SEED))
        ])
        grid = GridSearchCV(pipe, ridge_param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
        grid.fit(X_train, y_train_h)
        best_alpha = grid.best_params_["ridge__alpha"]

        pred_ridge = grid.predict(X_val)
        met_ridge = evaluate(y_val_h, pred_ridge)
        rep_ridge = per_class_report(y_val_h, pred_ridge)
        rep_ridge["model"] = "Ridge"
        rep_ridge["horizon"] = h
        all_per_class.append(rep_ridge)

        # Candidate baseline selection (Ridge is a model, not a baseline)
        p_mae = met_pers["MAE"]
        c_mae = met_clim["MAE"]
        dp_mae = met_dp["MAE"]
        r_mae = met_ridge["MAE"]

        candidates = [
            ("Persistence", p_mae),
            ("Climatology", c_mae),
            ("Damped Anomaly Persistence", dp_mae),
        ]
        best_cand_name, best_b_mae = min(candidates, key=lambda x: x[1])

        best_baseline_records[str(h)] = {
            "best_baseline_name": best_cand_name,
            "best_baseline_val_mae": float(best_b_mae),
            "rho_h": float(rho_h)
        }

        baseline_maes[h] = {
            "persistence": p_mae,
            "climatology": c_mae,
            "damped_persistence": dp_mae,
            "best_baseline": best_cand_name,
            "best_baseline_mae": best_b_mae
        }

        # Skills
        skill_pers_pers = 0.0
        skill_pers_best = skill_vs_best_baseline(p_mae, best_b_mae)
        skill_clim_pers = skill_vs_persistence(c_mae, p_mae)
        skill_clim_best = skill_vs_best_baseline(c_mae, best_b_mae)
        skill_dp_pers = skill_vs_persistence(dp_mae, p_mae)
        skill_dp_best = skill_vs_best_baseline(dp_mae, best_b_mae)
        skill_ridge_pers = skill_vs_persistence(r_mae, p_mae)
        skill_ridge_best = skill_vs_best_baseline(r_mae, best_b_mae)

        met_pers_row = {
            "model": "Persistence", "horizon": h, **met_pers,
            "skill_vs_persistence": skill_pers_pers, "skill_vs_best_baseline": skill_pers_best
        }
        met_clim_row = {
            "model": "Climatology", "horizon": h, **met_clim,
            "skill_vs_persistence": skill_clim_pers, "skill_vs_best_baseline": skill_clim_best
        }
        met_dp_row = {
            "model": "Damped Anomaly Persistence", "horizon": h, "best_alpha": np.nan, **met_dp,
            "skill_vs_persistence": skill_dp_pers, "skill_vs_best_baseline": skill_dp_best
        }
        met_ridge_row = {
            "model": "Ridge", "horizon": h, "best_alpha": best_alpha, **met_ridge,
            "skill_vs_persistence": skill_ridge_pers, "skill_vs_best_baseline": skill_ridge_best
        }
        all_metrics.extend([met_pers_row, met_clim_row, met_dp_row, met_ridge_row])

        compact_summary.append({
            "model": "Persistence", "horizon": f"h={h}", "MAE": p_mae, "bias": met_pers["bias"],
            "within_one_acc": met_pers["within_one_category_accuracy"],
            "alert_prec": met_pers["alert_precision"], "alert_recall": met_pers["alert_recall"],
            "skill_pers": skill_pers_pers, "skill_best": skill_pers_best
        })
        compact_summary.append({
            "model": "Climatology", "horizon": f"h={h}", "MAE": c_mae, "bias": met_clim["bias"],
            "within_one_acc": met_clim["within_one_category_accuracy"],
            "alert_prec": met_clim["alert_precision"], "alert_recall": met_clim["alert_recall"],
            "skill_pers": skill_clim_pers, "skill_best": skill_clim_best
        })
        compact_summary.append({
            "model": "Damped Anomaly Pers", "horizon": f"h={h}", "MAE": dp_mae, "bias": met_dp["bias"],
            "within_one_acc": met_dp["within_one_category_accuracy"],
            "alert_prec": met_dp["alert_precision"], "alert_recall": met_dp["alert_recall"],
            "skill_pers": skill_dp_pers, "skill_best": skill_dp_best
        })
        compact_summary.append({
            "model": f"Ridge (a={best_alpha})", "horizon": f"h={h}", "MAE": r_mae, "bias": met_ridge["bias"],
            "within_one_acc": met_ridge["within_one_category_accuracy"],
            "alert_prec": met_ridge["alert_precision"], "alert_recall": met_ridge["alert_recall"],
            "skill_pers": skill_ridge_pers, "skill_best": skill_ridge_best
        })

        for d_str, yt, pp, pc, pdp, pr in zip(val_df["date"], y_val_h, pred_pers, pred_clim, pred_damped, pred_ridge):
            val_preds_records.append({
                "date": d_str,
                "horizon": h,
                "truth": float(yt),
                "pred_persistence": float(np.clip(pp, 0.0, 500.0)),
                "pred_climatology": float(np.clip(pc, 0.0, 500.0)),
                "pred_damped_persistence": float(np.clip(pdp, 0.0, 500.0)),
                "pred_ridge": float(np.clip(pr, 0.0, 500.0)),
            })

    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_metrics = pd.DataFrame(all_metrics)
    metrics_path = out_dir / "val_metrics_baselines.csv"
    df_metrics.to_csv(metrics_path, index=False)
    print(f"Saved baseline metrics to {metrics_path}")

    df_per_class = pd.concat(all_per_class, ignore_index=True)
    per_class_path = out_dir / "val_per_class_baselines.csv"
    df_per_class.to_csv(per_class_path, index=False)
    print(f"Saved baseline per-class report to {per_class_path}")

    df_preds = pd.DataFrame(val_preds_records)
    preds_path = out_dir / "val_predictions_baselines.csv"
    df_preds.to_csv(preds_path, index=False)
    print(f"Saved baseline predictions to {preds_path}")

    # Save models/best_baseline.json
    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_baseline_path = models_dir / "best_baseline.json"
    with open(best_baseline_path, "w", encoding="utf-8") as f:
        json.dump(best_baseline_records, f, indent=2)
    print(f"Saved best baseline validation choices to {best_baseline_path}")

    df_compact = pd.DataFrame(compact_summary)
    print("\n" + "="*115)
    print("COMPACT BASELINE COMPARISON ON VALIDATION SET (h = 1..5)")
    print("="*115)
    print(f"{'Model':24s} | {'Horizon':7s} | {'MAE':7s} | {'Bias':7s} | {'Within-1 Acc':12s} | {'Alert Prec':10s} | {'Alert Rec':10s} | {'Skill vs Pers':13s} | {'Skill vs Best':13s}")
    print("-" * 115)
    for _, row in df_compact.iterrows():
        print(f"{row['model']:24s} | {row['horizon']:7s} | {row['MAE']:7.2f} | {row['bias']:7.2f} | {row['within_one_acc']:12.3f} | {row['alert_prec']:10.3f} | {row['alert_recall']:10.3f} | {row['skill_pers']:13.3f} | {row['skill_best']:13.3f}")
    print("="*115)

    return baseline_maes


def run_random_forest(train_df, val_df, feature_cols, baseline_maes, cv):
    """
    Trains and tunes RandomForestRegressor for raw and log1p targets across horizons h=1..5.
    Evaluates on validation set, compares with baselines, saves models and plots.
    """
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]

    rf_param_dist = {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 6, 10, 16],
        "min_samples_leaf": [1, 3, 5, 10],
        "max_features": [0.3, 0.5, 0.8, 1.0],
    }

    all_rf_metrics = []
    all_rf_per_class = []
    val_preds_rf_records = []
    cv_results_records = []
    best_params_dict = {}
    rf_config_dict = {}

    winning_models = {}

    start_total_time = time.time()

    print("\n" + "="*105)
    print("TUNING AND EVALUATING RANDOM FOREST ON VALIDATION SET (h = 1..5, raw vs log target)")
    print("="*105)

    for h in range(1, 6):
        h_start = time.time()
        target_col = f"target_h{h}"
        y_train_raw = train_df[target_col].values
        y_train_log = np.log1p(y_train_raw)
        y_val_raw = val_df[target_col].values

        p_mae = baseline_maes[h]["persistence"]
        c_mae = baseline_maes[h]["climatology"]
        best_base_mae = min(p_mae, c_mae)

        # ── 1. Tune RF Raw Target ────────────────────────────────────────────
        print(f"\n[Horizon h={h}] Tuning RF on RAW target AQI (30 iterations)...")
        rf_raw = RandomForestRegressor(random_state=RANDOM_SEED)
        search_raw = RandomizedSearchCV(
            rf_raw,
            rf_param_dist,
            n_iter=30,
            scoring="neg_mean_absolute_error",
            cv=cv,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        search_raw.fit(X_train, y_train_raw)
        best_model_raw = search_raw.best_estimator_
        best_params_raw = search_raw.best_params_
        best_idx_raw = search_raw.best_index_
        mean_cv_raw = -search_raw.cv_results_["mean_test_score"][best_idx_raw]
        std_cv_raw = search_raw.cv_results_["std_test_score"][best_idx_raw]

        cv_results_records.append({
            "horizon": h,
            "target_scale": "raw",
            "mean_cv_mae": mean_cv_raw,
            "std_cv_mae": std_cv_raw,
            "best_params": json.dumps(best_params_raw)
        })

        pred_val_raw = best_model_raw.predict(X_val)
        met_raw = evaluate(y_val_raw, pred_val_raw)
        rep_raw = per_class_report(y_val_raw, pred_val_raw)
        rep_raw["model"] = "rf_raw"
        rep_raw["horizon"] = h
        all_rf_per_class.append(rep_raw)

        skill_pers_raw = skill_vs_persistence(met_raw["MAE"], p_mae)
        skill_best_raw = skill_vs_best_baseline(met_raw["MAE"], p_mae, c_mae)
        all_rf_metrics.append({
            "model": "rf_raw",
            "horizon": h,
            **met_raw,
            "skill_vs_persistence": skill_pers_raw,
            "skill_vs_best_baseline": skill_best_raw
        })

        # ── 2. Tune RF Log1p Target ──────────────────────────────────────────
        print(f"[Horizon h={h}] Tuning RF on LOG1P target AQI (30 iterations)...")
        rf_log = RandomForestRegressor(random_state=RANDOM_SEED)
        search_log = RandomizedSearchCV(
            rf_log,
            rf_param_dist,
            n_iter=30,
            scoring="neg_mean_absolute_error",
            cv=cv,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        search_log.fit(X_train, y_train_log)
        best_model_log = search_log.best_estimator_
        best_params_log = search_log.best_params_
        best_idx_log = search_log.best_index_
        mean_cv_log = -search_log.cv_results_["mean_test_score"][best_idx_log]
        std_cv_log = search_log.cv_results_["std_test_score"][best_idx_log]

        cv_results_records.append({
            "horizon": h,
            "target_scale": "log",
            "mean_cv_mae": mean_cv_log,
            "std_cv_mae": std_cv_log,
            "best_params": json.dumps(best_params_log)
        })

        pred_val_log_raw = best_model_log.predict(X_val)
        pred_val_log = np.expm1(pred_val_log_raw)
        met_log = evaluate(y_val_raw, pred_val_log)
        rep_log = per_class_report(y_val_raw, pred_val_log)
        rep_log["model"] = "rf_log"
        rep_log["horizon"] = h
        all_rf_per_class.append(rep_log)

        skill_pers_log = skill_vs_persistence(met_log["MAE"], p_mae)
        skill_best_log = skill_vs_best_baseline(met_log["MAE"], p_mae, c_mae)
        all_rf_metrics.append({
            "model": "rf_log",
            "horizon": h,
            **met_log,
            "skill_vs_persistence": skill_pers_log,
            "skill_vs_best_baseline": skill_best_log
        })

        # Record best params
        best_params_dict[f"h{h}"] = {
            "raw": best_params_raw,
            "log": best_params_log
        }

        # ── 3. Select Winning Target Scale on Validation Set ─────────────────
        if met_raw["MAE"] <= met_log["MAE"]:
            winner_scale = "raw"
            winning_model = best_model_raw
            winning_params = best_params_raw
            winning_mae = met_raw["MAE"]
            winning_pred = pred_val_raw
            winning_skill_best = skill_best_raw
            margin = met_log["MAE"] - met_raw["MAE"]
        else:
            winner_scale = "log"
            winning_model = best_model_log
            winning_params = best_params_log
            winning_mae = met_log["MAE"]
            winning_pred = pred_val_log
            winning_skill_best = skill_best_log
            margin = met_raw["MAE"] - met_log["MAE"]

        winning_models[h] = (winning_model, winner_scale)
        rf_config_dict[f"h{h}"] = {
            "target_scale": winner_scale,
            "use_log": bool(winner_scale == "log"),
            "val_mae": winning_mae,
            "raw_val_mae": met_raw["MAE"],
            "log_val_mae": met_log["MAE"],
            "margin_over_runner_up": margin,
            "skill_vs_best_baseline": winning_skill_best,
            "beats_climatology": bool(winning_mae < c_mae),
            "best_params": winning_params
        }

        # Save predictions per date for horizon h
        for d_str, yt, pr, pl, pw in zip(val_df["date"], y_val_raw, pred_val_raw, pred_val_log, winning_pred):
            val_preds_rf_records.append({
                "date": d_str,
                "horizon": h,
                "truth": float(yt),
                "pred_rf_raw": float(np.clip(pr, 0.0, 500.0)),
                "pred_rf_log": float(np.clip(pl, 0.0, 500.0)),
                "pred_rf_winning": float(np.clip(pw, 0.0, 500.0)),
            })

        h_elapsed = time.time() - h_start
        print(f"[Horizon h={h} complete in {h_elapsed:.1f}s] Winner: {winner_scale} (MAE={winning_mae:.2f} vs {met_log['MAE'] if winner_scale=='raw' else met_raw['MAE']:.2f}, margin={margin:.2f} AQI). Climatology MAE={c_mae:.2f}. Beats Climatology: {winning_mae < c_mae}")

    total_elapsed = time.time() - start_total_time
    print(f"\nTotal Random Forest tuning runtime: {total_elapsed:.1f}s ({total_elapsed/60:.2f} min)")

    # Save outputs
    out_dir = REPO_ROOT / "outputs"
    models_dir = REPO_ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Best params json
    with open(out_dir / "rf_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params_dict, f, indent=2)
    print(f"Saved best params to {out_dir / 'rf_best_params.json'}")

    # 2. Validation metrics
    df_rf_metrics = pd.DataFrame(all_rf_metrics)
    metrics_path = out_dir / "val_metrics_rf.csv"
    df_rf_metrics.to_csv(metrics_path, index=False)
    print(f"Saved RF validation metrics to {metrics_path}")

    # 3. Per class report
    df_rf_per_class = pd.concat(all_rf_per_class, ignore_index=True)
    cols_order = ["model", "horizon", "category", "precision", "recall", "f1", "support"]
    df_rf_per_class = df_rf_per_class[cols_order]
    per_class_path = out_dir / "val_per_class_rf.csv"
    df_rf_per_class.to_csv(per_class_path, index=False)
    print(f"Saved RF per-class report to {per_class_path}")

    # 4. Predictions
    df_rf_preds = pd.DataFrame(val_preds_rf_records)
    preds_path = out_dir / "val_predictions_rf.csv"
    df_rf_preds.to_csv(preds_path, index=False)
    print(f"Saved RF validation predictions to {preds_path}")

    # 5. CV results
    df_cv = pd.DataFrame(cv_results_records)
    cv_path = out_dir / "rf_cv_results.csv"
    df_cv.to_csv(cv_path, index=False)
    print(f"Saved RF CV results to {cv_path}")

    # 6. Save winning models and config
    for h in range(1, 6):
        w_model, _ = winning_models[h]
        joblib.dump(w_model, models_dir / f"rf_h{h}.pkl")
    print(f"Saved winning models to models/rf_h1.pkl ... models/rf_h5.pkl")

    with open(models_dir / "rf_config.json", "w", encoding="utf-8") as f:
        json.dump(rf_config_dict, f, indent=2)
    print(f"Saved RF config to models/rf_config.json")

    # 7. Feature importances bar chart for h=1 and h=5
    plot_rf_feature_importances(winning_models[1][0], winning_models[5][0], feature_cols, out_dir / "rf_feature_importance.png")

    # Print summary comparison table
    print("\n" + "="*115)
    print("RANDOM FOREST VALIDATION PERFORMANCE vs BASELINES (h = 1..5)")
    print("="*115)
    print(f"{'Model':16s} | {'Horizon':7s} | {'MAE':7s} | {'Bias':7s} | {'Within-1 Acc':12s} | {'Alert Prec':10s} | {'Alert Rec':10s} | {'Skill vs Pers':13s} | {'Skill vs Best':13s}")
    print("-" * 115)
    for _, row in df_rf_metrics.iterrows():
        print(f"{row['model']:16s} | h={row['horizon']}     | {row['MAE']:7.2f} | {row['bias']:7.2f} | {row['within_one_category_accuracy']:12.3f} | {row['alert_precision']:10.3f} | {row['alert_recall']:10.3f} | {row['skill_vs_persistence']:13.3f} | {row['skill_vs_best_baseline']:13.3f}")
    print("="*115)


def plot_rf_feature_importances(model_h1, model_h5, feature_names, out_path):
    """Plots horizontal bar charts of top 15 feature importances for h=1 and h=5."""
    imp_h1 = pd.Series(model_h1.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    imp_h5 = pd.Series(model_h5.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharex=False)

    imp_h1.sort_values(ascending=True).plot(kind="barh", ax=ax1, color="#1f77b4", edgecolor="black")
    ax1.set_title("Top 15 Feature Importances — Horizon h=1 (Day Ahead)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Gini Feature Importance", fontsize=10)
    ax1.grid(axis="x", linestyle=":", alpha=0.6)

    imp_h5.sort_values(ascending=True).plot(kind="barh", ax=ax2, color="#2ca02c", edgecolor="black")
    ax2.set_title("Top 15 Feature Importances — Horizon h=5 (5 Days Ahead)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Gini Feature Importance", fontsize=10)
    ax2.grid(axis="x", linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved feature importance diagram to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train baselines and Random Forest models.")
    parser.add_argument("--baselines-only", action="store_true", help="Run validation baselines and generate models/best_baseline.json without tuning RF.")
    args = parser.parse_args()

    train_df, val_df, _ = get_splits()
    with open(REPO_ROOT / "models" / "feature_columns.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    climatology_map = build_climatology_lookup(train_df)
    cv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=CV_GAP)

    baseline_maes = run_baselines(train_df, val_df, feature_cols, climatology_map, cv)
    if not args.baselines_only:
        run_random_forest(train_df, val_df, feature_cols, baseline_maes, cv)


if __name__ == "__main__":
    main()
