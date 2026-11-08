import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv('data/processed/model_ready.csv')
train = df[pd.to_datetime(df['date']) <= pd.to_datetime('2023-03-31')].reset_index(drop=True)

tscv = TimeSeriesSplit(n_splits=4, gap=5)
print('TimeSeriesSplit(n_splits=4, gap=5) on Train (1887 days):')
for fold, (tr_idx, val_idx) in enumerate(tscv.split(train), 1):
    tr_part = train.iloc[tr_idx]
    val_part = train.iloc[val_idx]
    poor = (val_part['aqi'] > 200).sum()
    severe = (val_part['aqi'] > 400).sum()
    print(f"Fold {fold}: Train {tr_part['date'].min()} to {tr_part['date'].max()} ({len(tr_part)} days) | Val {val_part['date'].min()} to {val_part['date'].max()} ({len(val_part)} days) | Poor+: {poor} | Severe: {severe}")
