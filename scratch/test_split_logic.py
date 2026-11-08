"""
scratch/test_split_logic.py
"""

import pandas as pd
from datetime import timedelta
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv('data/processed/model_ready.csv')
df['date'] = pd.to_datetime(df['date'])

train_end = pd.to_datetime('2023-03-31')
val_end = pd.to_datetime('2024-03-31')
gap = timedelta(days=5)

train = df[df['date'] <= train_end].copy()
gap1 = df[(df['date'] > train_end) & (df['date'] <= train_end + gap)].copy()
val = df[(df['date'] > train_end + gap) & (df['date'] <= val_end)].copy()
gap2 = df[(df['date'] > val_end) & (df['date'] <= val_end + gap)].copy()
test = df[df['date'] > val_end + gap].copy()

print(f"Train: {len(train)} rows | {train['date'].min().date()} to {train['date'].max().date()}")
print(f"Gap 1: {len(gap1)} rows | {gap1['date'].min().date()} to {gap1['date'].max().date()}")
print(f"Val:   {len(val)} rows | {val['date'].min().date()} to {val['date'].max().date()}")
print(f"Gap 2: {len(gap2)} rows | {gap2['date'].min().date()} to {gap2['date'].max().date()}")
print(f"Test:  {len(test)} rows | {test['date'].min().date()} to {test['date'].max().date()}")
print(f"Total: {len(train) + len(gap1) + len(val) + len(gap2) + len(test)} == {len(df)}")

def aqi_cat(aqi_val):
    if aqi_val <= 50:
        return "Good"
    elif aqi_val <= 100:
        return "Satisfactory"
    elif aqi_val <= 200:
        return "Moderate"
    elif aqi_val <= 300:
        return "Poor"
    elif aqi_val <= 400:
        return "Very Poor"
    else:
        return "Severe"

for name, s_df in [("Train", train), ("Validation", val), ("Test", test)]:
    cats = s_df['aqi'].apply(aqi_cat).value_counts()
    poor_plus = (s_df['aqi'] > 200).sum()
    severe = (s_df['aqi'] > 400).sum()
    print(f"\n--- {name} (N={len(s_df)}) ---")
    print(f"Date range: {s_df['date'].min().date()} to {s_df['date'].max().date()}")
    for c in ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]:
        cnt = cats.get(c, 0)
        pct = cnt / len(s_df) * 100
        print(f"  {c:15s}: {cnt:4d} ({pct:5.2f}%)")
    print(f"  Poor-or-worse (AQI > 200): {poor_plus:4d} ({poor_plus/len(s_df)*100:5.2f}%)")
    print(f"  Severe (AQI > 400):        {severe:4d} ({severe/len(s_df)*100:5.2f}%)")

# Check winter coverage
print("\n=== Winter Coverage Check ===")
# Val winter: Oct 2023 to Feb 2024
val_months = val['date'].dt.to_period('M').unique().astype(str).tolist()
print("Val months:", val_months)
expected_val_winter = ["2023-10", "2023-11", "2023-12", "2024-01", "2024-02"]
has_all_val_winter = all(m in val_months for m in expected_val_winter)
print(f"Val has full winter Oct 2023 - Feb 2024: {has_all_val_winter}")

# Test winters: Oct 2024 to Feb 2025, and Oct 2025 to Feb 2026
test_months = test['date'].dt.to_period('M').unique().astype(str).tolist()
print("Test months:", test_months)
expected_test_winter_1 = ["2024-10", "2024-11", "2024-12", "2025-01", "2025-02"]
expected_test_winter_2 = ["2025-10", "2025-11", "2025-12", "2026-01", "2026-02"]
has_test_w1 = all(m in test_months for m in expected_test_winter_1)
has_test_w2 = all(m in test_months for m in expected_test_winter_2)
print(f"Test has full winter Oct 2024 - Feb 2025: {has_test_w1}")
print(f"Test has full winter Oct 2025 - Feb 2026: {has_test_w2}")

# Check TimeSeriesSplit(n_splits=5, gap=5) on train
tscv = TimeSeriesSplit(n_splits=5, gap=5)
print("\n=== TimeSeriesSplit(n_splits=5, gap=5) on Train ===")
for fold, (tr_idx, val_idx) in enumerate(tscv.split(train), 1):
    tr_part = train.iloc[tr_idx]
    val_part = train.iloc[val_idx]
    poor_worse = (val_part['aqi'] > 200).sum()
    severe = (val_part['aqi'] > 400).sum()
    print(f"Fold {fold}:")
    print(f"  Train:      {tr_part['date'].min().date()} to {tr_part['date'].max().date()} ({len(tr_part):4d} days)")
    print(f"  Gap:        {(val_part['date'].min() - tr_part['date'].max()).days} days difference")
    print(f"  Validation: {val_part['date'].min().date()} to {val_part['date'].max().date()} ({len(val_part):4d} days)")
    print(f"  Validation Poor-or-worse days: {poor_worse} ({poor_worse/len(val_part)*100:.2f}%)")
    print(f"  Validation Severe days:        {severe} ({severe/len(val_part)*100:.2f}%)")
