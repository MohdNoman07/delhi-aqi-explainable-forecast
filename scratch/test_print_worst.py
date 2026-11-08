import sys
from pathlib import Path
sys.path.insert(0, str(Path("D:/ML_Project")))
import pandas as pd
from datetime import timedelta
from src.config import DIWALI_DATES
diwali_dts = [pd.to_datetime(d) for d in DIWALI_DATES]
test_preds = pd.read_csv('outputs/test_predictions.csv')
for h in range(1, 6):
    df_h = test_preds[test_preds['horizon'] == h].copy()
    df_h['abs_err'] = (df_h['truth'] - df_h['pred_random_forest']).abs()
    w10 = df_h.sort_values(by='abs_err', ascending=False).head(10)
    print(f'=== h={h} ===')
    for _, r in w10.iterrows():
        f_dt = pd.to_datetime(r['date'])
        t_dt = f_dt + timedelta(days=h)
        diffs = [(t_dt - d).days for d in diwali_dts]
        df_diw = min(diffs, key=abs)
        print(f"  f={f_dt.date()} -> t={t_dt.date()} (truth={r['truth']:.1f}, pred={r['pred_random_forest']:.1f}, err={r['abs_err']:.1f}) days_diwali={df_diw:+d}")
