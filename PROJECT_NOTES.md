# PROJECT NOTES — Delhi Air Quality Forecasting

## Project Rules (MANDATORY — NEVER REMOVE)

1. **Never assume column names or units; use `COLUMN_MAP` in `src/config.py`.**
2. **Never shuffle time series.** Use a chronological split with a 5-day gap. The test set is used only in the final evaluation script.
3. **Features use only information up to the end of today.** Rolling windows are trailing only (`center=False`). No negative shifts in features. Daily features are strictly trailing (enforced by automated tests), while hourly cleaning decisions (such as bounded linear interpolation $\le 6$h and stuck-sensor invalidation) are retrospective across the dataset.
4. **Imputers, scalers, and climatology lookup tables are fit on training data only.**
5. **Use random seed 42 from `config.py` everywhere.**
6. **Every script runs from the repo root**, saves outputs to `outputs/` or `data/processed/`, and logs how many rows or values it removed.
7. **Never invent numbers;** every number in the report must come from a saved file.
8. **After each task, summarise what you did, what you found and anything you were unsure about, then wait for my next instruction.**

---

## Pipeline Execution Run Order

To reproduce the pipeline from raw CPCB telemetry to final evaluation:

1. **Station Selection & Hourly Cleaning**:
   - `python src/clean.py`
   - Inputs: `data/raw/*.csv`
   - Outputs: `outputs/station_selection.csv`, `outputs/cleaning_log.csv`, `data/processed/clean_hourly.parquet`
2. **Daily Aggregation & CPCB NAQI Computation**:
   - `python src/daily_aggregation.py`
   - Inputs: `data/processed/clean_hourly.parquet`
   - Outputs: `data/processed/clean_daily.csv`, `outputs/daily_coverage.csv`, `outputs/prominent_pollutant.csv`
3. **Feature Engineering & Targets**:
   - `python src/features.py`
   - Inputs: `data/processed/clean_daily.csv`
   - Outputs: `data/processed/model_ready.csv`, `models/feature_columns.json`
4. **Chronological Splitting & Diagnostic Folds**:
   - `python src/split.py`
   - Inputs: `data/processed/model_ready.csv`
   - Outputs: `outputs/split_summary.md`, `outputs/split_diagram.png`
   - Note: Split configuration uses `TRAIN_END = 2023-03-31`, `VAL_END = 2024-03-31`, `GAP_DAYS = 5`, `CV_SPLITS = 4`, `CV_GAP = 5`.
5. **Model Training & Hyperparameter Tuning on Validation**:
   - `python src/train.py`
   - Inputs: `data/processed/model_ready.csv`, `models/feature_columns.json`
   - Outputs: `models/rf_config.json`, `models/best_baseline.json`, `outputs/val_metrics_baselines.csv`, `outputs/val_metrics_rf.csv`
6. **Final Holdout Evaluation**:
   - `python src/final_evaluate.py`
   - Inputs: `data/processed/model_ready.csv`, `models/rf_config.json`, `models/best_baseline.json`
   - Outputs: `outputs/metrics.csv`, `outputs/test_predictions.csv`, `outputs/per_class_metrics.csv`, `outputs/rf_vs_baseline_ci.csv`, `outputs/rf_vs_ridge_ci.csv`, `outputs/bootstrap_block_sensitivity.csv`, `outputs/alert_ci.csv`
7. **Error Analysis & Diagnostics**:
   - `python src/error_analysis.py`
   - Inputs: `outputs/test_predictions.csv`
   - Outputs: `outputs/error_analysis.md`, `outputs/error_by_month.png`, `outputs/winter_timeseries_h1_h5.png`, `outputs/residuals_h1_h5.png`, `outputs/worst_days_h*.csv`
8. **Explainability & Feature Attribution**:
   - `python src/explain.py`
   - Inputs: `data/processed/model_ready.csv`, `models/rf_h*.pkl`
   - Outputs: `outputs/shap_family_importance.csv`, `outputs/shap_family_importance.png`, `outputs/shap_summary_h*.png`, `outputs/shap_bar_h*.png`, `outputs/shap_waterfall_*.png`, `outputs/shap_notes.md`
9. **Interactive CLI Inference Demo**:
   - `python src/predict.py --latest` (or `--date YYYY-MM-DD`)

---

## Numbers Ledger & Documentation Pointers

- **All numerical citations**: Sourced directly from saved artifacts cataloged in `docs/numbers_used.csv`.
- **Comprehensive Technical Report**: `docs/report_draft.md`.
- **Slide Presentation Outline**: `docs/slides_outline.md`.
