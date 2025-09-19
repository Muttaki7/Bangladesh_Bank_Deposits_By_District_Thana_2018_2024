"""
full_deposit_analysis.py

- Adapts to wide-format CSV where year columns (e.g., '2018') are present and each year
  is followed by up to 4 quarterly columns (often 'Unnamed: X').
- Produces CSV and PNG outputs for all requested analyses.

Edit DATA_PATH and optionally SHP_PATH before running.
"""

import os
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# optional imports
try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None

try:
    import geopandas as gpd
except Exception:
    gpd = None

# -----------------------------
# User settings
# -----------------------------
DATA_PATH = r"C:\Users\Muttaki\Desktop\analysis gov\data-resource_2024_06_24_Table-16 Deposit Amount Distributed by District and Thana.csv"
SHP_PATH = r"C:\Users\Muttaki\Desktop\gis\bangladesh_districts.shp"  # optional for maps
OUT_DIR = Path("analysis_outputs")
OUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def safe_save_df(df, filename):
    path = OUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

def safe_save_fig(fig, filename):
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

def gini(array):
    array = np.asarray(array, dtype=float)
    array = array[~np.isnan(array)]
    if array.size == 0:
        return np.nan
    if (array < 0).any():
        array = array - array.min()
    if array.sum() == 0:
        return 0.0
    array = np.sort(array)
    n = array.size
    cum = np.cumsum(array)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n

# -----------------------------
# 1. Load CSV - try common header positions
# -----------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

# try reading with header 4 (your file used header=4 earlier); fallback to header=0
try:
    df_raw = pd.read_csv(DATA_PATH, header=4, low_memory=False)
except Exception:
    df_raw = pd.read_csv(DATA_PATH, header=0, low_memory=False)

print("Columns in dataset:", df_raw.columns.tolist())

# -----------------------------
# 2. Detect id columns (district/thana) and year columns
# -----------------------------
cols = list(df_raw.columns)

# Assume first two cols are district and thana if they are non-numeric names:
id_cols = []
# find left-most non-year columns at start (not pure-digit)
for c in cols:
    if str(c).strip().isdigit():
        break
    id_cols.append(c)
# ensure at least 2 ids; if only 1 found then assume second id missing
if len(id_cols) < 2:
    # fallback: explicitly use first two columns
    id_cols = cols[:2]

print("Identified identifier columns (assumed):", id_cols)

# find columns that are pure years like '2018', '2019' etc.
year_cols = [c for c in cols if str(c).strip().isdigit()]
print("Detected year columns:", year_cols)

if not year_cols:
    raise ValueError("No year-like columns detected. Cannot proceed. Check CSV layout.")

# Build mapping: for each year column, detect following 1-4 columns as quarter breakdowns
# We'll assume the structure is: [id_cols] [2018, Unnamed, Unnamed, Unnamed] [2019, Unnamed, ...] ...
# We'll create a list of (year, [col_q1, col_q2, col_q3, col_q4]) where missing quarters allowed
year_quarter_cols = {}
col_index = {c: i for i, c in enumerate(cols)}
for yc in year_cols:
    i = col_index[yc]
    # look ahead up to 4 columns (i+1 ... i+4) but stop if next is another year
    qcols = []
    for j in range(1,5):
        idx = i + j
        if idx >= len(cols):
            break
        nxt = cols[idx]
        if str(nxt).strip().isdigit():
            break
        qcols.append(nxt)
    year_quarter_cols[str(yc).strip()] = qcols

print("Year -> quarterly columns mapping (detected):")
for y,q in year_quarter_cols.items():
    print(y, "->", q)

# -----------------------------
# 3. Melt wide to long: produce rows at quarter level if quarters exist, else year level
# -----------------------------
records = []
for idx, row in df_raw.iterrows():
    # read identifiers
    ids = { "district": row[id_cols[0]] if id_cols else None,
            "thana": row[id_cols[1]] if len(id_cols)>1 else None }
    # for each year, for each quarter column if present produce record
    for y, qcols in year_quarter_cols.items():
        if qcols:
            # if qcols less than 4, treat them as successive quarters Q1..Qn
            for qi, qc in enumerate(qcols, start=1):
                val = row.get(qc, np.nan)
                # skip NaN / empty strings
                try:
                    val_num = float(str(val).replace(',','').strip()) if pd.notna(val) and str(val).strip()!='' else np.nan
                except Exception:
                    val_num = np.nan
                rec = {
                    "district": ids["district"],
                    "thana": ids["thana"],
                    "year": int(str(y)),
                    "quarter": f"Q{qi}",
                    "deposits": val_num
                }
                records.append(rec)
        else:
            # no quarterly cols -> use the year column value itself
            val = row.get(y, np.nan)
            try:
                val_num = float(str(val).replace(',','').strip()) if pd.notna(val) and str(val).strip()!='' else np.nan
            except Exception:
                val_num = np.nan
            rec = {
                "district": ids["district"],
                "thana": ids["thana"],
                "year": int(str(y)),
                "quarter": None,
                "deposits": val_num
            }
            records.append(rec)

df = pd.DataFrame.from_records(records)
# Clean text columns
df['district'] = df['district'].astype(str).str.strip()
df['thana'] = df['thana'].astype(str).str.strip()
df['deposits'] = pd.to_numeric(df['deposits'], errors='coerce')
df = df.dropna(subset=['deposits']).reset_index(drop=True)
df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)

print(f"Reshaped dataset rows: {len(df)}")
safe_save_df(df, "reshaped_long_deposits.csv")

# -----------------------------
# Basic checks / sample
# -----------------------------
print("\nSample rows after reshape:")
print(df.head(8))

# -----------------------------
# DESCRIPTIVE STATISTICS
# -----------------------------
desc = df['deposits'].describe()
print("\nDescriptive statistics (deposits):\n", desc)
pd.DataFrame(desc).to_csv(OUT_DIR/"descriptive_stats_deposits.csv")

# -----------------------------
# Total deposits by district & thana
# -----------------------------
total_by_dt = df.groupby(['district','thana'], dropna=False)['deposits'].sum().reset_index().sort_values('deposits', ascending=False)
safe_save_df(total_by_dt, "total_deposits_by_district_thana.csv")
print("\nTop 10 district-thana by deposits:")
print(total_by_dt.head(10))

# -----------------------------
# Average quarterly deposits over the years
# -----------------------------
# If quarter is present, compute per-year-per-quarter means; else per-year average
if df['quarter'].notna().any():
    avg_quarterly = df.groupby(['year','quarter'])['deposits'].mean().reset_index().sort_values(['year','quarter'])
    safe_save_df(avg_quarterly, "avg_quarterly_deposits.csv")
    print("\nAverage quarterly deposits sample:")
    print(avg_quarterly.head(8))
else:
    avg_quarterly = df.groupby('year')['deposits'].mean().reset_index()
    safe_save_df(avg_quarterly, "avg_yearly_deposits.csv")

# -----------------------------
# Growth rate of deposits (YoY) national
# -----------------------------
yearly_totals = df.groupby('year')['deposits'].sum().reset_index().sort_values('year')
yearly_totals['growth_rate_%'] = yearly_totals['deposits'].pct_change()*100
safe_save_df(yearly_totals, "yearly_totals_growth.csv")
print("\nYearly totals + YoY growth:")
print(yearly_totals)

# -----------------------------
# Trend Analysis plots
# -----------------------------
sns.set(style="whitegrid")

# Yearly trend
fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(data=yearly_totals, x='year', y='deposits', marker='o', ax=ax)
ax.set_title("Yearly Deposit Trend (Total)")
ax.set_ylabel("Total Deposits")
safe_save_fig(fig, "yearly_deposit_trend.png")

# Quarterly trend: stack quarters by year (if quarters exist)
if df['quarter'].notna().any():
    qsum = df.groupby(['year','quarter'])['deposits'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12,6))
    sns.lineplot(data=qsum, x='quarter', y='deposits', hue='year', marker='o', ax=ax)
    ax.set_title("Quarterly Deposit Trend by Year")
    safe_save_fig(fig, "quarterly_deposit_trend.png")

# -----------------------------
# Identify districts with highest/lowest growth (YoY)
# -----------------------------
district_yearly = df.groupby(['district','year'])['deposits'].sum().reset_index().sort_values(['district','year'])
district_yearly['growth_rate_%'] = district_yearly.groupby('district')['deposits'].pct_change()*100
safe_save_df(district_yearly, "district_yearly_deposits_growth.csv")

latest_year = district_yearly['year'].max()
if not np.isnan(latest_year):
    latest_growth = district_yearly[district_yearly['year'] == latest_year].copy()
    top_growth = latest_growth.sort_values('growth_rate_%', ascending=False).head(10)
    bottom_growth = latest_growth.sort_values('growth_rate_%').head(10)
    safe_save_df(top_growth, "top_growth_districts_latest_year.csv")
    safe_save_df(bottom_growth, "bottom_growth_districts_latest_year.csv")
    print(f"\nTop growth districts in {latest_year}:")
    print(top_growth[['district','deposits','growth_rate_%']].head(10))
    print(f"\nLowest growth districts in {latest_year}:")
    print(bottom_growth[['district','deposits','growth_rate_%']].head(10))

# -----------------------------
# Seasonal patterns: compare Q1 vs Q4 etc.
# -----------------------------
if df['quarter'].notna().any():
    seasonal = df.groupby('quarter')['deposits'].agg(['mean','median','sum','count']).reset_index()
    safe_save_df(seasonal, "seasonal_quarter_summary.csv")
    fig, ax = plt.subplots(figsize=(8,5))
    # order quarters properly if formatted Q1..Q4
    order = sorted(seasonal['quarter'].dropna().unique(), key=lambda x: int(str(x).replace('Q','')) if str(x).upper().startswith('Q') else x)
    sns.barplot(data=seasonal, x='quarter', y='mean', order=order, ax=ax)
    ax.set_title("Average Deposits by Quarter (Mean)")
    safe_save_fig(fig, "seasonal_avg_by_quarter.png")

# -----------------------------
# Comparative Analysis: urban vs rural, divisions, ranking
# -----------------------------
# Try to detect 'area' or 'area_type' or 'division' columns in original df_raw (since id_cols may not include them)
available_cols = [c.lower() for c in df_raw.columns]
col_map = {c.lower(): c for c in df_raw.columns}  # map lowercase -> original

# if the original file contains columns for area_type or division, join them into df
extra_cols = {}
for key in ['area', 'area_type', 'urban_rural', 'division', 'bank_type', 'scheduled_status']:
    if key in available_cols:
        extra_cols[key] = col_map[key]

# join extra info from df_raw to df on district/thana if present
if extra_cols:
    # create a lookup table of first occurrence values per district/thana
    join_cols = [id_cols[0]] + ([id_cols[1]] if len(id_cols)>1 else [])
    lookup = df_raw[[*join_cols, *extra_cols.values()]].drop_duplicates(subset=join_cols)
    # rename join cols to 'district','thana' names used in df
    lookup = lookup.rename(columns={join_cols[0]:'district', join_cols[1]:'thana'} if len(join_cols)>1 else {join_cols[0]:'district'})
    df = df.merge(lookup, on=['district','thana'] if 'thana' in lookup.columns else ['district'], how='left')

# Urban vs Rural
if any(x in df.columns for x in ['area', 'area_type', 'urban_rural']):
    area_col = next((c for c in df.columns if c.lower() in ['area','area_type','urban_rural']), None)
    ur = df.groupby(area_col)['deposits'].sum().reset_index().sort_values('deposits', ascending=False)
    safe_save_df(ur, "urban_vs_rural_totals.csv")
    print("\nUrban vs Rural totals (sample):")
    print(ur.head())

# Division-level aggregates
if 'division' in df.columns:
    div = df.groupby('division')['deposits'].sum().reset_index().sort_values('deposits', ascending=False)
    safe_save_df(div, "division_deposits.csv")
    print("\nDeposits by division (top 10):")
    print(div.head(10))

# Rank districts by total deposits
district_total = df.groupby('district')['deposits'].sum().reset_index().sort_values('deposits', ascending=False)
safe_save_df(district_total, "district_rank_by_total_deposits.csv")
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(data=district_total.head(15), x='deposits', y='district', ax=ax)
ax.set_title("Top 15 Districts by Total Deposits")
safe_save_fig(fig, "top15_districts_total_deposits.png")

# -----------------------------
# Heatmap: district vs quarter (financial inclusion levels)
# -----------------------------
if df['quarter'].notna().any():
    hm = df.pivot_table(values='deposits', index='district', columns='quarter', aggfunc='sum', fill_value=0)
    # Sort districts by total deposits to show hubs vs underbanked
    hm = hm.assign(total=hm.sum(axis=1)).sort_values('total', ascending=False).drop(columns=['total'])
    hm.to_csv(OUT_DIR/"heatmap_table_district_quarter.csv")
    fig, ax = plt.subplots(figsize=(14, max(6, int(len(hm)/6))))
    sns.heatmap(hm, cmap='YlGnBu', ax=ax)
    ax.set_title("Heatmap: Deposits by District (rows) and Quarter (columns)")
    safe_save_fig(fig, "heatmap_district_quarter.png")

# -----------------------------
# Time-Series Forecasting: ARIMA on national quarterly series
# -----------------------------
if ARIMA is None:
    print("\nstatsmodels ARIMA not available -> skipping forecasting. To enable, pip install statsmodels.")
else:
    # Build quarterly index if quarter exists
    if df['quarter'].notna().any():
        qsum = df.groupby(['year','quarter'])['deposits'].sum().reset_index()
        # convert quarter to period start date
        def q_to_ts(yr, q):
            qn = int(str(q).replace('Q','')) if str(q).upper().startswith('Q') else np.nan
            month = (qn-1)*3 + 1 if not np.isnan(qn) else 1
            return pd.Timestamp(year=int(yr), month=int(month), day=1)
        qsum['period'] = qsum.apply(lambda r: q_to_ts(r['year'], r['quarter']), axis=1)
        qsum = qsum.sort_values('period').set_index('period')
        ts = qsum['deposits'].asfreq('QS')  # quarterly start
        ts = ts.fillna(method='ffill')  # simple fill
        # Fit ARIMA(1,1,1)
        try:
            model = ARIMA(ts, order=(1,1,1))
            fit = model.fit()
            steps = 8
            fc = fit.get_forecast(steps=steps)
            fc_df = fc.summary_frame()
            fc_df.to_csv(OUT_DIR/"quarterly_forecast_arima.csv")
            # plot
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(ts.index, ts.values, label='Historical')
            ax.plot(fc_df.index, fc_df['mean'], label='Forecast', color='red')
            ax.fill_between(fc_df.index, fc_df['mean_ci_lower'], fc_df['mean_ci_upper'], color='pink', alpha=0.3)
            ax.legend()
            ax.set_title("ARIMA Forecast - Quarterly Deposits (next 8 quarters)")
            safe_save_fig(fig, "quarterly_forecast_arima.png")
            print("ARIMA forecasting saved.")
        except Exception as e:
            print("ARIMA forecasting failed:", str(e))
    else:
        print("Quarter-level data not present -> cannot build quarterly time series for forecasting.")

# -----------------------------
# Leading & Lagging districts (time-series style)
# -----------------------------
# We already computed district_yearly growth. Let's produce a leader/lagger report over years
leader_report = district_yearly.copy()
# For each year find top/bottom 5 by growth
leader_summary = []
years = sorted(leader_report['year'].unique())
for y in years:
    tmp = leader_report[leader_report['year']==y].dropna(subset=['growth_rate_%'])
    top5 = tmp.sort_values('growth_rate_%', ascending=False).head(5)
    bot5 = tmp.sort_values('growth_rate_%').head(5)
    leader_summary.append({'year': y, 'top5': list(top5['district'].astype(str)), 'bot5': list(bot5['district'].astype(str))})
pd.DataFrame(leader_summary).to_csv(OUT_DIR/"leader_laggard_summary_by_year.csv", index=False)
print("\nLeader/laggard summary saved.")

# -----------------------------
# Inequality: Gini coefficient by latest year and overall
# -----------------------------
latest_year = df['year'].max()
arr_latest = df[df['year']==latest_year].groupby('district')['deposits'].sum().values
gini_latest = gini(arr_latest) if arr_latest.size>0 else np.nan
arr_overall = df.groupby('district')['deposits'].sum().values
gini_overall = gini(arr_overall)
pd.DataFrame([{"year": latest_year, "gini": gini_latest}, {"year": "overall", "gini": gini_overall}]).to_csv(OUT_DIR/"gini_coefficients.csv", index=False)
print(f"\nGini (latest year {latest_year}): {gini_latest:.4f}, overall: {gini_overall:.4f}")

# -----------------------------
# Sector / Bank-wise analysis (if columns detected)
# -----------------------------
# Attempt to detect bank_type or scheduled columns in df_raw, or earlier merged extras
bank_col = next((c for c in df.columns if c.lower() in ['bank_type','bank type','bank']), None)
sched_col = next((c for c in df.columns if c.lower() in ['scheduled_status','scheduled','scheduled_status']), None)

if bank_col is not None:
    bank_totals = df.groupby(bank_col)['deposits'].sum().reset_index().sort_values('deposits', ascending=False)
    safe_save_df(bank_totals, "bank_type_totals.csv")
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(bank_totals['deposits'], labels=bank_totals[bank_col], autopct='%1.1f%%')
    ax.set_title("Deposits by Bank Type")
    safe_save_fig(fig, "bank_type_pie.png")
else:
    print("No bank_type column detected -> skipping bank-type analysis.")

if sched_col is not None:
    sched_totals = df.groupby(sched_col)['deposits'].sum().reset_index()
    safe_save_df(sched_totals, "scheduled_vs_nonscheduled.csv")
    print("Saved scheduled vs non-scheduled totals.")
else:
    print("No scheduled_status column detected -> skipping scheduled vs non-scheduled analysis.")

# -----------------------------
# Geospatial maps (optional)
# -----------------------------
if gpd is None:
    print("\nGeopandas not installed -> skipping geospatial maps. To enable install geopandas and ensure SHP_PATH is set.")
elif not os.path.exists(SHP_PATH):
    print("\nShapefile not found at SHP_PATH -> skipping maps.")
else:
    try:
        gdf = gpd.read_file(SHP_PATH)
        # try to merge on district name - you may need to adjust column names
        possible_name_cols = [c for c in gdf.columns if 'name' in c.lower() or 'district' in c.lower()]
        if not possible_name_cols:
            print("Could not find a district name column in shapefile; available columns:", gdf.columns.tolist())
        else:
            shp_name = possible_name_cols[0]
            # prepare district totals for merging
            dt = district_total.rename(columns={'district': 'district', 'deposits': 'deposits'})
            # try merging (may need manual name standardization)
            merged = gdf.merge(dt, left_on=shp_name, right_on='district', how='left')
            merged['deposits'] = merged['deposits'].fillna(0)
            fig = merged.plot(column='deposits', cmap='OrRd', legend=True, figsize=(12,10)).get_figure()
            safe_save_fig(fig, "deposits_map_districts.png")
    except Exception as e:
        print("Geospatial plotting failed:", str(e))

print("\nALL analyses completed. Outputs in folder:", OUT_DIR.resolve())
# -----------------------------
# Extra Graphs & Heatmaps
# -----------------------------

# 1. Distribution of deposits (Histogram + Boxplot)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['deposits'], bins=50, ax=ax[0], kde=True)
ax[0].set_title("Distribution of Deposits (Histogram)")

sns.boxplot(x=df['deposits'], ax=ax[1])
ax[1].set_title("Distribution of Deposits (Boxplot)")

safe_save_fig(fig, "deposit_distribution.png")

# 2. Correlation heatmap between yearly deposits (district-level)
year_pivot = df.pivot_table(values='deposits', index='district', columns='year', aggfunc='sum', fill_value=0)
corr = year_pivot.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
ax.set_title("Correlation Between Yearly Deposits (District-level)")
safe_save_fig(fig, "yearly_correlation_heatmap.png")

# 3. Division Heatmap (if division exists)
if 'division' in df.columns:
    div_q = df.groupby(['division','year'])['deposits'].sum().reset_index()
    div_pivot = div_q.pivot(index='division', columns='year', values='deposits').fillna(0)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(div_pivot, cmap='YlOrBr', annot=True, fmt=".0f", ax=ax)
    ax.set_title("Heatmap: Deposits by Division and Year")
    safe_save_fig(fig, "division_year_heatmap.png")

# 4. Urban vs Rural barplot
if any(x in df.columns for x in ['area', 'area_type', 'urban_rural']):
    area_col = next((c for c in df.columns if c.lower() in ['area','area_type','urban_rural']), None)
    ur = df.groupby(area_col)['deposits'].sum().reset_index().sort_values('deposits', ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=ur, x=area_col, y='deposits', ax=ax)
    ax.set_title("Urban vs Rural Deposits")
    safe_save_fig(fig, "urban_vs_rural_bar.png")

# 5. Top vs Bottom 10 Districts Heatmap
top10 = district_total.head(10).set_index('district')
bottom10 = district_total.tail(10).set_index('district')
tb = pd.concat([top10, bottom10])
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(tb, cmap="Greens", annot=True, fmt=".0f", ax=ax)
ax.set_title("Top & Bottom 10 Districts by Deposits")
safe_save_fig(fig, "top_bottom10_districts_heatmap.png")
# -----------------------------
# Animated Time-Series Plots (GIFs)
# -----------------------------
try:
    import matplotlib.animation as animation
except ImportError:
    print("matplotlib.animation not available -> skipping GIF animations.")
else:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    # 1. Yearly Total Deposits Animation
    yearly_cumsum = df.groupby(['year','district'])['deposits'].sum().reset_index()
    districts = yearly_cumsum['district'].unique()

    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_xlim(0, len(districts))
    ax.set_ylim(0, yearly_cumsum['deposits'].max()*1.1)
    bars = ax.bar(districts, [0]*len(districts))
    ax.set_title("Yearly Deposits by District (Animated)")
    ax.set_ylabel("Deposits")
    plt.xticks(rotation=90)

    def update(year):
        data = yearly_cumsum[yearly_cumsum['year']==year]
        for i, b in enumerate(bars):
            b.set_height(data.iloc[i]['deposits'] if i < len(data) else 0)
        ax.set_xlabel(f"Year: {year}")
        return bars

    ani = FuncAnimation(fig, update, frames=sorted(yearly_cumsum['year'].unique()), repeat=False)
    ani.save(OUT_DIR/"yearly_deposits_animation.gif", writer=PillowWriter(fps=1))
    plt.close(fig)
    print("Animated GIF saved: yearly_deposits_animation.gif")

    # 2. Quarterly Deposits Animation (if quarter exists)
    if df['quarter'].notna().any():
        qsum = df.groupby(['year','quarter','district'])['deposits'].sum().reset_index()
        quarters = qsum['quarter'].unique()
        years = sorted(qsum['year'].unique())
        fig, ax = plt.subplots(figsize=(12,6))
        ax.set_xlim(0, len(districts))
        ax.set_ylim(0, qsum['deposits'].max()*1.1)
        bars = ax.bar(districts, [0]*len(districts))
        ax.set_title("Quarterly Deposits by District (Animated)")
        ax.set_ylabel("Deposits")
        plt.xticks(rotation=90)

        def update_q(frame):
            year, quarter = frame
            data = qsum[(qsum['year']==year) & (qsum['quarter']==quarter)]
            for i, b in enumerate(bars):
                b.set_height(data.iloc[i]['deposits'] if i < len(data) else 0)
            ax.set_xlabel(f"Year: {year}, Quarter: {quarter}")
            return bars

        frames = [(y,q) for y in years for q in quarters]
        ani_q = FuncAnimation(fig, update_q, frames=frames, repeat=False)
        ani_q.save(OUT_DIR/"quarterly_deposits_animation.gif", writer=PillowWriter(fps=1))
        plt.close(fig)
        print("Animated GIF saved: quarterly_deposits_animation.gif")
