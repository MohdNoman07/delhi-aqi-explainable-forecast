import pandas as pd
import numpy as np

daily = pd.read_csv('data/processed/clean_daily.csv')
hourly = pd.read_parquet('data/processed/clean_hourly.parquet')
hourly['date'] = pd.to_datetime(hourly['datetime']).dt.strftime('%Y-%m-%d')

test_preds = pd.read_csv('outputs/test_predictions.csv')

print("--- In test_predictions.csv ---")
for h in [1, 2, 3, 4, 5]:
    df_h = test_preds[test_preds['horizon'] == h]
    df_h = df_h.copy()
    df_h['abs_err'] = (df_h['truth'] - df_h['pred_random_forest']).abs()
    top2 = df_h.sort_values(by='abs_err', ascending=False).head(2)
    print(f"\nTop 2 errors for h={h}:")
    for _, r in top2.iterrows():
        f_date = pd.to_datetime(r['date'])
        t_date = f_date + pd.Timedelta(days=h)
        print(f"  forecast_date={f_date.strftime('%Y-%m-%d')}, target_date={t_date.strftime('%Y-%m-%d')}, truth={r['truth']:.2f}, pred={r['pred_random_forest']:.2f}, err={r['abs_err']:.2f}")

print("\n--- Inspecting dates in clean_daily and clean_hourly ---")
for target_str in ['2025-05-14', '2025-05-15', '2026-07-11', '2026-07-12']:
    d_row = daily[daily['date'] == target_str]
    if not d_row.empty:
        r = d_row.iloc[0]
        print(f"\nDate: {target_str}")
        print(f"  clean_daily: AQI={r.get('aqi')}, PM2.5={r.get('pm25')}, PM10={r.get('pm10')}, Prominent={r.get('prominent_pollutant')}")
    
    h_sub = hourly[hourly['date'] == target_str]
    if not h_sub.empty:
        valid_pm25 = h_sub['pm25'].notna().sum()
        valid_pm10 = h_sub['pm10'].notna().sum()
        n_st = h_sub['n_stations_pm25']
        print(f"  clean_hourly: valid_hours={len(h_sub)}, valid_pm25_hours={valid_pm25}, valid_pm10_hours={valid_pm10}, n_stations_pm25 mean={n_st.mean():.2f}, min={n_st.min()}, max={n_st.max()}")
