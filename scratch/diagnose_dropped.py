"""
scratch/diagnose_dropped.py
Diagnostic on the 47 mid-series dropped rows from model_ready.csv
"""

import sys
import json
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features import compute_features

daily = pd.read_csv('data/processed/clean_daily.csv')
daily['date'] = pd.to_datetime(daily['date'])
full_idx = pd.date_range('2018-01-01', '2026-09-30', freq='D')
daily_full = daily.set_index('date').reindex(full_idx)
daily_full.index.name = 'date'

targets = pd.DataFrame(index=daily_full.index)
for h in range(1, 6):
    targets[f'target_h{h}'] = daily_full['aqi'].shift(-h)

features = compute_features(daily_full)
with open('models/feature_columns.json') as f:
    feature_cols = json.load(f)
target_cols = [f'target_h{h}' for h in range(1, 6)]
cols_to_check = feature_cols + target_cols

combined = pd.concat([features, targets], axis=1).reset_index()
nan_mask = combined[cols_to_check].isna().any(axis=1)
nan_rows = combined[nan_mask].copy()

mid_missing = nan_rows[(nan_rows['date'] > '2018-01-07') & (nan_rows['date'] < '2026-09-26')]
print(f'Total mid-series dropped rows: {len(mid_missing)}')

# Examine all feature columns and target columns that had NaNs
col_summary = []
for col in cols_to_check:
    dates_with_nan = mid_missing[mid_missing[col].isna()]['date'].dt.strftime('%Y-%m-%d').tolist()
    if len(dates_with_nan) > 0:
        is_feature = col in feature_cols
        col_summary.append({
            'column': col,
            'column_type': 'feature' if is_feature else 'target',
            'dropped_row_count': len(dates_with_nan),
            'dates': '; '.join(dates_with_nan)
        })

df_diag = pd.DataFrame(col_summary)
df_diag = df_diag.sort_values(by=['column_type', 'dropped_row_count'], ascending=[True, False]).reset_index(drop=True)

out_path = Path('outputs/dropped_rows_by_column.csv')
out_path.parent.mkdir(parents=True, exist_ok=True)
df_diag.to_csv(out_path, index=False)
print(f"Saved diagnostic to {out_path}")

print("\n" + "="*80)
print("FEATURE COLUMNS WITH NaNs (Task A):")
print("="*80)
feat_only = df_diag[df_diag['column_type'] == 'feature']
for _, r in feat_only.iterrows():
    print(f"  {r['column']:30s} | {r['dropped_row_count']:2d} rows | Dates: {r['dates']}")

print("\n" + "="*80)
print("TARGET COLUMNS WITH NaNs (Forward Target Ripple):")
print("="*80)
target_only = df_diag[df_diag['column_type'] == 'target']
for _, r in target_only.iterrows():
    print(f"  {r['column']:30s} | {r['dropped_row_count']:2d} rows | Dates: {r['dates']}")
