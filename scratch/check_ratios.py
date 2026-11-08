import pandas as pd
daily = pd.read_csv('data/processed/clean_daily.csv').set_index('date')
for dt, prev in [('2025-05-15', '2025-05-14'), ('2026-07-12', '2026-07-11')]:
    r = daily.loc[dt]
    r_prev = daily.loc[prev]
    pm25, pm10 = r['pm25'], r['pm10']
    ratio = pm25 / pm10
    print(f"Target Day: {dt}")
    print(f"  AQI Day Before ({prev}): {r_prev['aqi']:.2f}")
    print(f"  AQI Day Of ({dt}): {r['aqi']:.2f}")
    print(f"  PM2.5: {pm25:.2f}, PM10: {pm10:.2f}")
    print(f"  PM2.5 / PM10 ratio: {ratio:.4f} ({ratio*100:.2f}%)")
