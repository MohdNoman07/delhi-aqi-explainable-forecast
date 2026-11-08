"""
src/generate_eda.py

Generates all EDA plots for Delhi air quality daily dataset and saves each figure
to outputs/eda/. Also saves top 15 AQI days to outputs/top_aqi_days.csv.
"""

import os
os.environ["MPLCONFIGDIR"] = "C:/Users/Noman/AppData/Local/Temp/mpl"

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf

REPO_ROOT = Path(__file__).resolve().parent.parent
EDA_DIR = REPO_ROOT / "outputs" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# Set global plotting style
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def generate_all_eda():
    # 1. Load clean daily data
    daily_csv = REPO_ROOT / "data" / "processed" / "clean_daily.csv"
    hourly_parquet = REPO_ROOT / "data" / "processed" / "clean_hourly.parquet"
    prom_csv = REPO_ROOT / "outputs" / "prominent_pollutant.csv"

    print(f"Loading {daily_csv}...")
    df = pd.read_csv(daily_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")

    # -------------------------------------------------------------------------
    # Plot 1: Daily AQI over time with 30-day trailing rolling mean
    # -------------------------------------------------------------------------
    print("Generating Plot 1: Daily AQI over time...")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(df["date"], df["aqi"], color="slategray", s=8, alpha=0.4, label="Daily AQI")
    df["rolling_30d"] = df["aqi"].rolling(window=30, min_periods=15).mean()
    ax.plot(df["date"], df["rolling_30d"], color="darkred", lw=2, label="30-Day Trailing Rolling Mean")

    bands = [
        (50, "Good (50)", "green"),
        (100, "Satisfactory (100)", "limegreen"),
        (200, "Moderate (200)", "goldenrod"),
        (300, "Poor (300)", "darkorange"),
        (400, "Very Poor (400)", "red"),
        (500, "Severe (500)", "darkred"),
    ]
    for y_val, lbl, c in bands:
        ax.axhline(y_val, color=c, linestyle="--", alpha=0.45, lw=1)

    ax.set_ylim(0, 520)
    ax.set_ylabel("CPCB-Style AQI (0–500)")
    ax.set_title("Daily CPCB-Style AQI (2018–2026) with 30-Day Trailing Rolling Mean", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "01_daily_aqi_over_time.png", dpi=150)
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 2: Average AQI by month (all years) and heatmap by year/month
    # -------------------------------------------------------------------------
    print("Generating Plot 2: Monthly AQI and heatmap...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [1, 1.2]})

    # Monthly bar chart
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_stats = df.groupby("month_name")["aqi"].agg(["mean", "std", "count"]).reindex(month_order)
    monthly_stats["sem"] = monthly_stats["std"] / np.sqrt(monthly_stats["count"])

    ax1.bar(monthly_stats.index, monthly_stats["mean"], yerr=monthly_stats["sem"], color="steelblue", capsize=4, alpha=0.85, edgecolor="black")
    ax1.set_ylabel("Mean AQI")
    ax1.set_title("Average Daily AQI by Month (All Years 2018–2026)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 420)
    for i, v in enumerate(monthly_stats["mean"]):
        ax1.text(i, v + 8, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

    # Heatmap of month per year
    pivot_monthly = df.pivot_table(index="year", columns="month", values="aqi", aggfunc="mean")
    pivot_monthly.columns = month_order
    sns.heatmap(pivot_monthly, cmap="YlOrRd", annot=True, fmt=".1f", cbar_kws={"label": "Monthly Mean AQI"}, ax=ax2, linewidths=0.5)
    ax2.set_title("Monthly Mean AQI by Year and Month", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Year")

    plt.tight_layout()
    plt.savefig(EDA_DIR / "02_aqi_by_month_and_heatmap.png", dpi=150)
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 3: Missing values per year and per column (daily table)
    # -------------------------------------------------------------------------
    print("Generating Plot 3: Missing values per year and per column...")
    eval_cols = [
        "pm25", "pm10", "no", "no2", "nox", "nh3", "so2", "co", "o3", "benzene", "toluene",
        "temperature", "humidity", "wind_speed", "solar_radiation", "barometric_pressure", "rainfall", "aqi"
    ]
    missing_pct_by_year = df.groupby("year")[eval_cols].apply(lambda g: (g.isna().mean() * 100).round(2))

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(missing_pct_by_year, annot=True, fmt=".1f", cmap="Blues", cbar_kws={"label": "Missing Percentage (%)"}, ax=ax, linewidths=0.5)
    ax.set_title("Missing Values Percentage per Year and Column (Daily Table)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Year")
    ax.set_xlabel("Measurement Variable")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "03_missing_values_by_year_and_column.png", dpi=150)
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 4: Correlation heatmap of pollutants, weather and AQI
    # -------------------------------------------------------------------------
    print("Generating Plot 4: Correlation heatmap...")
    corr_cols = [
        "aqi", "pm25", "pm10", "no", "no2", "nox", "nh3", "so2", "co", "o3",
        "temperature", "humidity", "wind_speed", "solar_radiation", "barometric_pressure", "rainfall"
    ]
    corr_mat = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, cbar_kws={"label": "Pearson Correlation (r)"}, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap: Pollutants, Weather Variables, and AQI", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "04_correlation_heatmap.png", dpi=150)
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 5: AQI category distribution overall and per year
    # -------------------------------------------------------------------------
    print("Generating Plot 5: Category distribution...")
    cat_order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    cat_colors = {
        "Good": "#2ca02c",
        "Satisfactory": "#8c564b",
        "Moderate": "#e377c2",
        "Poor": "#ff7f0e",
        "Very Poor": "#d62728",
        "Severe": "#7f0000"
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.3]})

    # Overall counts
    overall_counts = df["aqi_category"].value_counts().reindex(cat_order).fillna(0)
    bars = ax1.bar(cat_order, overall_counts, color=[cat_colors[c] for c in cat_order], edgecolor="black", alpha=0.85)
    ax1.set_ylabel("Number of Days")
    ax1.set_title("Overall AQI Category Counts (2018–2026)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1250)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 15, f"{int(h)}", ha="center", fontsize=9, fontweight="bold")
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Yearly breakdown stacked percentage
    cat_year = pd.crosstab(df["year"], df["aqi_category"]).reindex(columns=cat_order).fillna(0)
    cat_year_pct = cat_year.div(cat_year.sum(axis=1), axis=0) * 100.0

    bottom = np.zeros(len(cat_year_pct))
    for c in cat_order:
        ax2.bar(cat_year_pct.index.astype(str), cat_year_pct[c], bottom=bottom, label=c, color=cat_colors[c], edgecolor="black", alpha=0.85)
        bottom += cat_year_pct[c].values

    ax2.set_ylabel("Share of Valid Days (%)")
    ax2.set_title("AQI Category Share per Year (2018–2026)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="lower right", bbox_to_anchor=(1.25, 0.2), framealpha=0.9)

    plt.tight_layout()
    plt.savefig(EDA_DIR / "05_aqi_category_distribution.png", dpi=150)
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 6: Autocorrelation & Partial Autocorrelation (lags 1 to 10)
    # -------------------------------------------------------------------------
    print("Generating Plot 6: ACF and PACF...")
    # Missing days kept NaN, not interpolated. Using missing='conservative'
    series_aqi = df["aqi"].dropna()
    lags = list(range(1, 11))
    acf_vals = acf(series_aqi, nlags=10, missing="conservative")[1:11]
    pacf_vals = pacf(series_aqi, nlags=10)[1:11]
    conf_level = 1.96 / np.sqrt(len(series_aqi))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ACF plot
    ax1.stem(lags, acf_vals, basefmt="black")
    ax1.axhline(0, color="black", lw=1)
    ax1.axhline(conf_level, color="blue", linestyle="--", alpha=0.6, label="95% Confidence Band")
    ax1.axhline(-conf_level, color="blue", linestyle="--", alpha=0.6)
    ax1.set_xlabel("Lag (Days)")
    ax1.set_ylabel("Autocorrelation")
    ax1.set_title("Autocorrelation (ACF) of Daily AQI (Lags 1–10)", fontsize=11, fontweight="bold")
    ax1.set_xticks(lags)
    ax1.set_ylim(-0.1, 1.0)
    for x, y in zip(lags, acf_vals):
        ax1.text(x, y + 0.03, f"{y:.2f}", ha="center", fontsize=8)
    ax1.legend(loc="upper right")

    # PACF plot
    ax2.stem(lags, pacf_vals, basefmt="black")
    ax2.axhline(0, color="black", lw=1)
    ax2.axhline(conf_level, color="blue", linestyle="--", alpha=0.6, label="95% Confidence Band")
    ax2.axhline(-conf_level, color="blue", linestyle="--", alpha=0.6)
    ax2.set_xlabel("Lag (Days)")
    ax2.set_ylabel("Partial Autocorrelation")
    ax2.set_title("Partial Autocorrelation (PACF) of Daily AQI (Lags 1–10)", fontsize=11, fontweight="bold")
    ax2.set_xticks(lags)
    ax2.set_ylim(-0.1, 1.0)
    for x, y in zip(lags, pacf_vals):
        ax2.text(x, y + 0.03, f"{y:.2f}", ha="center", fontsize=8)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(EDA_DIR / "06_acf_pacf_lags_1_to_10.png", dpi=150)
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 7: Binned effect of wind speed and temperature on AQI (median & IQR)
    # -------------------------------------------------------------------------
    print("Generating Plot 7: Binned weather effects...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Wind speed bins
    ws_bins = [0, 1.0, 1.5, 2.0, 2.5, 3.0, 10.0]
    ws_labels = ["<1.0", "1.0–1.5", "1.5–2.0", "2.0–2.5", "2.5–3.0", "≥3.0"]
    df["ws_bin"] = pd.cut(df["wind_speed"], bins=ws_bins, labels=ws_labels, right=False)

    ws_agg = df.groupby("ws_bin", observed=True)["aqi"].agg(
        p25=lambda x: x.quantile(0.25),
        median="median",
        p75=lambda x: x.quantile(0.75),
        count="count"
    ).reset_index()

    yerr_lower = ws_agg["median"] - ws_agg["p25"]
    yerr_upper = ws_agg["p75"] - ws_agg["median"]
    ax1.errorbar(ws_agg.index, ws_agg["median"], yerr=[yerr_lower, yerr_upper], fmt="-o", color="navy", ecolor="cornflowerblue", elinewidth=2, capsize=5, capthick=2, markersize=7)
    ax1.set_xticks(ws_agg.index)
    ax1.set_xticklabels(ws_agg["ws_bin"])
    ax1.set_xlabel("Wind Speed Bin (m/s)")
    ax1.set_ylabel("Daily AQI")
    ax1.set_title("Effect of Wind Speed on AQI (Median and IQR)", fontsize=11, fontweight="bold")
    ax1.set_ylim(50, 400)
    for i, r in ws_agg.iterrows():
        ax1.text(i, r["median"] + 12, f"Med: {r['median']:.0f}\n(n={int(r['count'])})", ha="center", fontsize=8)

    # Temperature bins
    temp_bins = [0, 15, 20, 25, 30, 35, 60]
    temp_labels = ["<15", "15–20", "20–25", "25–30", "30–35", "≥35"]
    df["temp_bin"] = pd.cut(df["temperature"], bins=temp_bins, labels=temp_labels, right=False)

    t_agg = df.groupby("temp_bin", observed=True)["aqi"].agg(
        p25=lambda x: x.quantile(0.25),
        median="median",
        p75=lambda x: x.quantile(0.75),
        count="count"
    ).reset_index()

    yerr_t_lower = t_agg["median"] - t_agg["p25"]
    yerr_t_upper = t_agg["p75"] - t_agg["median"]
    ax2.errorbar(t_agg.index, t_agg["median"], yerr=[yerr_t_lower, yerr_t_upper], fmt="-o", color="darkred", ecolor="salmon", elinewidth=2, capsize=5, capthick=2, markersize=7)
    ax2.set_xticks(t_agg.index)
    ax2.set_xticklabels(t_agg["temp_bin"])
    ax2.set_xlabel("Temperature Bin (°C)")
    ax2.set_ylabel("Daily AQI")
    ax2.set_title("Effect of Temperature on AQI (Median and IQR)", fontsize=11, fontweight="bold")
    ax2.set_ylim(50, 400)
    for i, r in t_agg.iterrows():
        ax2.text(i, r["median"] + 12, f"Med: {r['median']:.0f}\n(n={int(r['count'])})", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(EDA_DIR / "07_binned_wind_speed_temp_effects.png", dpi=150)
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 8: Prominent pollutant share per year (from outputs/prominent_pollutant.csv)
    # -------------------------------------------------------------------------
    print("Generating Plot 8: Prominent pollutant share...")
    p_df = pd.read_csv(prom_csv)
    p_df_years = p_df[p_df["year"] != "Overall"].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = p_df_years["year"].astype(str)
    bottom = np.zeros(len(p_df_years))

    colors = {"pm10": "teal", "pm25": "coral", "co": "mediumpurple", "o3": "gold"}
    labels = {"pm10": "PM10", "pm25": "PM2.5", "co": "CO", "o3": "O3"}

    for p in ["pm10", "pm25", "co", "o3"]:
        vals = p_df_years[f"{p}_share_pct"].values
        ax.bar(x, vals, bottom=bottom, label=labels[p], color=colors[p], edgecolor="black", alpha=0.85)
        bottom += vals

    ax.set_ylabel("Share of Valid Days (%)")
    ax.set_title("Prominent Pollutant Share per Year (2018–2026)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", bbox_to_anchor=(1.18, 0.3), framealpha=0.9)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "08_prominent_pollutant_share_by_year.png", dpi=150)
    plt.close()

    # -------------------------------------------------------------------------
    # Section 10: Top AQI Days Table and Hourly Profile for 3 Highest Days
    # -------------------------------------------------------------------------
    print("Generating Section 10: Top AQI days table & hourly plot...")
    sub_cols = ["sub_index_pm25", "sub_index_pm10", "sub_index_no2", "sub_index_so2", "sub_index_nh3", "sub_index_co", "sub_index_o3"]
    df["unclipped_aqi"] = df[sub_cols].max(axis=1)

    top15 = df.sort_values("unclipped_aqi", ascending=False).head(15).copy()

    top_table = pd.DataFrame({
        "date": top15["date"].dt.strftime("%Y-%m-%d"),
        "unclipped_aqi": top15["unclipped_aqi"].round(2),
        "prominent_pollutant": top15["prominent_pollutant"],
        "pm25_mean": top15["pm25"].round(2),
        "pm25_max": top15["pm25_max"].round(2),
        "pm10_mean": top15["pm10"].round(2),
        "pm10_max": top15["pm10_max"].round(2),
        "n_stations_pm25_mean": top15["n_stations_pm25_mean"].round(2),
        "n_stations_pm25_min": top15["n_stations_pm25_min"].astype(int),
        "valid_hours": top15["valid_hours_pm25"].astype(int),
        "fewer_than_10_stations": top15["n_stations_pm25_min"] < 10
    })

    top_csv = REPO_ROOT / "outputs" / "top_aqi_days.csv"
    top_table.to_csv(top_csv, index=False)
    print(f"Saved {top_csv}")

    # Plot hourly city-average PM2.5 and PM10 for 3 highest days
    top3_dates = ["2018-06-13", "2018-06-14", "2024-11-18"]
    print(f"Loading hourly data for top 3 days: {top3_dates}...")
    hdf = pd.read_parquet(hourly_parquet)
    hdf["datetime"] = pd.to_datetime(hdf["datetime"])
    hdf["date_str"] = hdf["datetime"].dt.strftime("%Y-%m-%d")
    hdf["hour"] = hdf["datetime"].dt.hour

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for i, d_str in enumerate(top3_dates):
        sub_h = hdf[hdf["date_str"] == d_str].sort_values("hour")
        ax = axes[i]
        ax.plot(sub_h["hour"], sub_h["pm10"], marker="o", color="teal", lw=2, label="PM10")
        ax.plot(sub_h["hour"], sub_h["pm25"], marker="s", color="coral", lw=2, label="PM2.5")
        
        day_unclipped = top_table.loc[top_table["date"] == d_str, "unclipped_aqi"].values[0]
        prom = top_table.loc[top_table["date"] == d_str, "prominent_pollutant"].values[0]
        ax.set_title(f"{d_str}\n(Unclipped AQI: {day_unclipped:.1f}, Prom: {prom.upper()})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Hour of Day (0–23)")
        if i == 0:
            ax.set_ylabel("Concentration (µg/m³)")
        ax.set_xticks(range(0, 24, 4))
        ax.set_ylim(0, 1000)
        ax.legend(loc="upper right")

    plt.suptitle("Hourly City-Average PM2.5 and PM10 for the 3 Highest Unclipped AQI Days", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "10_hourly_profiles_top_3_days.png", dpi=150)
    plt.close()
    print("All figures successfully generated in outputs/eda/!")


if __name__ == "__main__":
    generate_all_eda()
