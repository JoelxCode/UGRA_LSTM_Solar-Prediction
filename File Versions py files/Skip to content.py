# Heatmap: full year on X (daily), 30-minute bins on Y (00:00 … 23:30)
# Works with BigData.csv from your LSTM solar project
# Tip: if auto-detection guesses wrong, set TIMESTAMP_COL and VALUE_COL manually.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------
# Config (optional)
# -----------------------
CSV_PATH = Path("BigData.csv")   # change to the correct path if needed
TIMESTAMP_COL = None             # e.g., "timestamp" or "Datetime"; leave None to auto-detect
VALUE_COL = None                 # e.g., "POWER" or "PV_Output"; leave None to auto-detect
AGG = "mean"                     # mean across 30-min bins per day (could be "sum", etc.)
CMAP = "coolwarm"  # was "viridis" — coolwarm goes blue -> red

# -----------------------
# Helpers
# -----------------------
def _guess_datetime_col(df):
    # obvious names first
    candidates = [c for c in df.columns
                  if str(c).lower() in ("timestamp","time","datetime","date_time","date","dt")]
    for c in candidates:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            pass
    # fallback: try every column
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            continue
    raise ValueError("Could not find a parseable datetime column. Set TIMESTAMP_COL explicitly.")

def _guess_value_col(df, ts_col):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    # drop common non-target numeric columns
    drop_like = {"year","month","day","hour","minute","second","epoch","unix"}
    numeric = [c for c in numeric if c != ts_col and str(c).lower() not in drop_like]
    if not numeric:
        # try converting object columns that look numeric
        for c in df.columns:
            if c == ts_col: 
                continue
            try:
                pd.to_numeric(df[c])
                return c
            except Exception:
                pass
        raise ValueError("No numeric target column found. Set VALUE_COL explicitly.")
    return numeric[0]

# -----------------------
# Load & prepare
# -----------------------
df_raw = pd.read_csv(CSV_PATH)

# If the CSV has MONTH/DAY/HOUR/MINUTE columns, build a datetime column
cols_upper = {c.upper() for c in df_raw.columns}
if {"MONTH", "DAY", "HOUR", "MINUTE"}.issubset(cols_upper):
    # preserve original column names (case-insensitive match)
    col_map = {name: next(c for c in df_raw.columns if c.upper() == name)
               for name in ("MONTH", "DAY", "HOUR", "MINUTE")}
    # choose a calendar year (data has no YEAR column) — use 2020 (non-leap) or change as needed
    YEAR_CONST = 2020
    df_raw["__dt"] = pd.to_datetime({
        "year": YEAR_CONST,
        "month": df_raw[col_map["MONTH"]],
        "day": df_raw[col_map["DAY"]],
        "hour": df_raw[col_map["HOUR"]],
        "minute": df_raw[col_map["MINUTE"]],
    }, errors="coerce")
    ts_col = "__dt"
else:
    ts_col = TIMESTAMP_COL or _guess_datetime_col(df_raw)

df = df_raw.copy()
df["__dt"] = pd.to_datetime(df[ts_col], errors="coerce")
df = df.dropna(subset=["__dt"])

# Choose target
val_col = VALUE_COL or _guess_value_col(df, ts_col)

# Round to 30-minute bins and split into date / time-of-day
df["__dt30"] = df["__dt"].dt.floor("30min")
df["__date"]  = df["__dt30"].dt.normalize()            # midnight-truncated datetime (keeps tz naive)
df["__tod"]   = df["__dt30"].dt.strftime("%H:%M")      # 30-min labels

# If dataset spans multiple years, pick the first full year; otherwise use the sole year
years = df["__date"].dt.year.unique()
year = int(sorted(years)[0])
year_mask = (df["__date"] >= pd.Timestamp(year=year, month=1, day=1)) & \
            (df["__date"] <= pd.Timestamp(year=year, month=12, day=31))
df = df.loc[year_mask].copy()

# Ensure every day of the year is present as a column
all_days = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")

# Ensure exactly 48 half-hour labels in order
half_hours = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0,30)]

# Aggregate to 30-min × day
if AGG == "mean":
    g = df.groupby(["__tod", "__date"])[val_col].mean()
elif AGG == "sum":
    g = df.groupby(["__tod", "__date"])[val_col].sum()
else:
    g = df.groupby(["__tod", "__date"])[val_col].agg(AGG)

pivot = g.unstack("__date")

# Reindex to full grid
pivot = pivot.reindex(index=half_hours)
pivot = pivot.reindex(columns=all_days, fill_value=np.nan)

# -----------------------
# Plot
# -----------------------
fig, ax = plt.subplots(figsize=(20, 8))

im = ax.imshow(
    pivot.values,
    aspect="auto",
    origin="lower",
    interpolation="nearest",
    cmap=CMAP
)

# Y-axis: every 30 minutes
ax.set_yticks(np.arange(len(half_hours)))
# To avoid overcrowding, show every 2 hours as labels but keep minor ticks
labels = []
for i, lab in enumerate(half_hours):
    labels.append(lab if lab.endswith(":00") and (i % 4 == 0) else "")
ax.set_yticklabels(labels, fontsize=8)
ax.set_ylabel("Time of Day (30-min bins)")

# X-axis: the entire year (daily columns)
num_days = pivot.shape[1]

# show weekly day ticks to avoid clutter (adjust step to 1 for every day)
day_step = 7
day_positions = np.arange(num_days)
day_labels = [d.strftime("%b %d") for d in all_days]

# set tick positions and vertical labels
ax.set_xticks(day_positions[::day_step])
ax.set_xticklabels(day_labels[::day_step], rotation=90, fontsize=7)
ax.xaxis.set_tick_params(which='major', pad=6)

# Put month labels at first day of each month (keep as light vertical lines)
month_starts = [d for d in all_days if d.day == 1]
month_pos = [all_days.get_loc(d) for d in month_starts]
for pos in month_pos:
    ax.axvline(pos - 0.5, color="k", linewidth=0.3, alpha=0.4)

ax.set_xlabel(f"Days of Year {year}")
title_target = val_col if VALUE_COL else f"{val_col} (auto-detected)"
ax.set_title(f"Heatmap of {title_target} — 30-minute Bins vs. Day of Year {year}")

cbar = plt.colorbar(im, ax=ax, pad=0.02)
cbar.ax.set_ylabel(title_target)

plt.tight_layout()
plt.show()