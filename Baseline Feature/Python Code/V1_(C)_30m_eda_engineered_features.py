"""
=============================================================
EDA — 30-Minute Statistics Summary Features (Selected Variables)
=============================================================
Input : train_30min.csv  +  test_30min.csv
        One row per event. Restricted to the variable set:
        X Frequency, Y Frequency, Hs (m), Tz (s),
        X Velocity_Std_Dev, Y Velocity_Std_Dev,
        X Displacement_Std_Dev, Y Displacement_Std_Dev,
        X Acc Raw_Mean, Y Acc Raw_Mean,
        X Acc Filtered_Std_Dev, Y Acc Filtered_Std_Dev
Output: distribution / correlation / pairplot PNGs
        + descriptive_stats.csv + dispersion_characteristics.csv
        saved to EDA_Eng_Plots_30minute/ beside the input files
=============================================================
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import skew as _skew, kurtosis as _kurt
from statsmodels.tsa.seasonal import STL
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# Chronological 80/20 train/test split produced by
# V1_(A)_30m_Re_Engineered_Feature.py
_CHRONO_DIR = Path(r"C:\Users\N O\Desktop\MASTER\Methodology & Result\Pre-Final Dataset\Dataset")
TRAIN_FILE = _CHRONO_DIR / "train_dataset_30minute_80pct_chronological.csv"
TEST_FILE  = _CHRONO_DIR / "test_dataset_30minute_20pct_chronological.csv"

OUTPUT_DIR = Path(r"C:\Users\N O\Desktop\MASTER\Methodology & Result\Pre-Final Dataset\EDA_Eng_Plots_30minute")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# PLOT STYLE
# ─────────────────────────────────────────────
DARK   = "#FFFFFF"
PANEL  = "#F5F5FA"
GRID   = "#DCDCE6"
TEXT   = "#1A1A2E"
CYAN   = "#0891B2"
AMBER  = "#D97706"
RED    = "#DC2626"
GREEN  = "#059669"
PURPLE = "#7C3AED"
PINK   = "#DB2777"

plt.rcParams.update({
    "figure.facecolor":  DARK,   "axes.facecolor":   PANEL,
    "axes.edgecolor":    GRID,   "axes.labelcolor":  TEXT,
    "axes.titlecolor":   TEXT,   "axes.grid":        True,
    "grid.color":        GRID,   "grid.linewidth":   0.5,
    "xtick.color":       TEXT,   "ytick.color":      TEXT,
    "text.color":        TEXT,   "legend.facecolor": PANEL,
    "legend.edgecolor":  GRID,   "savefig.facecolor":DARK,
    "savefig.dpi":       150,    "font.family":      "DejaVu Sans",
    "font.weight":       "light","axes.titleweight":  "light",
    "axes.labelweight":  "light",
})


def save(fig, name: str):
    fig.savefig(OUTPUT_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


# ─────────────────────────────────────────────
# LOAD  (train first, then test — combined into one df)
# ─────────────────────────────────────────────
print("Loading data …")
df_train = pd.read_csv(TRAIN_FILE, low_memory=False)
df_test  = pd.read_csv(TEST_FILE,  low_memory=False)
df = pd.concat([df_train, df_test], ignore_index=True)
df.columns = df.columns.str.strip()
# Event_No is a plain int in (B)'s chronological output, but was a string
# like "E548_processed" in older sources -- handle both. Extracting digits
# from the stringified column works either way; rows where no digits are
# found (blank/unparsable Event_No) are dropped since they can't be placed
# in the timeline.
event_no = df["Event_No"].astype(str).str.extract(r"(\d+)")[0]
unparsable = event_no.isna()
if unparsable.any():
    print(f"  WARNING: dropping {unparsable.sum()} row(s) with missing/unparsable Event_No")
    df = df[~unparsable].reset_index(drop=True)
    event_no = event_no[~unparsable].reset_index(drop=True)
df["Event_No"] = event_no.astype(int)
df = df.sort_values("Event_No").reset_index(drop=True)
print(f"  train: {len(df_train):,} rows | test: {len(df_test):,} rows | "
      f"combined: {len(df):,} rows | {df['Event_No'].nunique()} events | {df.shape[1]} columns")

# Campaign timing — E571 = 2021-09-28 00:08, ~30-min spacing (one skipped slot at E1011)
CAMPAIGN_ANCHOR_EVENT = 571
CAMPAIGN_ANCHOR_TIME  = pd.Timestamp("2021-09-28 00:08:00")
EVENT_STEP            = pd.Timedelta(minutes=30)
df["Datetime"] = CAMPAIGN_ANCHOR_TIME + (df["Event_No"] - CAMPAIGN_ANCHOR_EVENT) * EVENT_STEP

TARGETS = ["X Frequency", "Y Frequency"]

# Variable set requested for this EDA pass — all plots/CSVs are restricted to these
VARS = [
    ("X Frequency",         CYAN),
    ("Y Frequency",         GREEN),
    ("Hs (m)",              PURPLE),
    ("Tz (s)",              AMBER),
    ("X Velocity_Std_Dev",     CYAN),
    ("Y Velocity_Std_Dev",     GREEN),
    ("X Displacement_Std_Dev", PURPLE),
    ("Y Displacement_Std_Dev", PINK),
    ("X Acc Raw_Mean",         PURPLE),
    ("Y Acc Raw_Mean",         PINK),
    ("X Acc Filtered_Std_Dev", AMBER),
    ("Y Acc Filtered_Std_Dev", RED),
]
VAR_COLS = [c for c, _ in VARS]
FEATURES = [c for c in VAR_COLS if c not in TARGETS]

# Every engineered feature in the dataset (all Min/Max/Mean/Median/Std_Dev per signal
# + Hs/Tz) — used for the frequency-correlation table so nothing is left out
ALL_STAT_COLS  = [c for c in df.columns if c.endswith(("_Min", "_Max", "_Mean", "_Median", "_Std_Dev"))]
WAVE_PARAMS    = ["Hs (m)", "Tz (s)"]
ALL_FEATURES   = ALL_STAT_COLS + WAVE_PARAMS

# STL decomposition — events are ~30-min apart -> 48 events/day
STL_PERIOD = 48
STL_ROBUST = True
STL_SERIES = [
    ("Hs (m)",              PURPLE),
    ("X Acc Filtered_Mean", AMBER),
    ("Y Acc Filtered_Mean", RED),
    ("X Frequency",         CYAN),
    ("Y Frequency",         GREEN),
]

# Bivariate regression grids — feature(s) vs both target frequencies
BIVARIATE_GROUPS = [
    (["Hs (m)"],
     "Significant Wave Height (Hs) vs Natural Frequencies", "hs_vs_frequency", [PURPLE]),
    (["Tz (s)"],
     "Zero-Crossing Period (Tz) vs Natural Frequencies", "tz_vs_frequency", [AMBER]),
    (["X Acc Filtered_Mean", "Y Acc Filtered_Mean"],
     "Mean Filtered Acceleration vs Natural Frequencies", "acc_filtered_mean_vs_frequency", [CYAN, GREEN]),
    (["X Velocity_Mean", "Y Velocity_Mean"],
     "Mean Velocity vs Natural Frequencies", "velocity_mean_vs_frequency", [CYAN, GREEN]),
    (["X Displacement_Mean", "Y Displacement_Mean"],
     "Mean Displacement vs Natural Frequencies", "displacement_mean_vs_frequency", [CYAN, GREEN]),
    (["X Acc Filtered_Std_Dev", "Y Acc Filtered_Std_Dev"],
     "Std Dev Filtered Acceleration vs Natural Frequencies", "acc_filtered_stddev_vs_frequency", [CYAN, GREEN]),
    (["X Velocity_Std_Dev", "Y Velocity_Std_Dev"],
     "Std Dev Velocity vs Natural Frequencies", "velocity_stddev_vs_frequency", [CYAN, GREEN]),
    (["X Displacement_Std_Dev", "Y Displacement_Std_Dev"],
     "Std Dev Displacement vs Natural Frequencies", "displacement_stddev_vs_frequency", [CYAN, GREEN]),
]


# ─────────────────────────────────────────────
# DATA WRANGLING — quick look
# ─────────────────────────────────────────────
print("\nData wrangling …")
df.info()
print(df[VAR_COLS].describe())
print("\nMissing values per column:")
print(df[VAR_COLS].isna().sum()[lambda s: s > 0])


# ══════════════════════════════════════════════════════════════
# PLOT 01 — Dataset Overview
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 6))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
fig.suptitle("Dataset Overview — 30-Minute Statistics Summary (1 row per event)", fontsize=13, y=1.02)

ax0 = fig.add_subplot(gs[0])
train_ids = set(df_train["Event_No"].unique())
test_ids  = set(df_test["Event_No"].unique())
ax0.bar(sorted(train_ids), [1]*len(train_ids), color=CYAN,  alpha=0.85, width=1.2, label="Train")
ax0.bar(sorted(test_ids),  [1]*len(test_ids),  color=AMBER, alpha=0.85, width=1.2, label="Test")
ax0.set_xlabel("Event No"); ax0.set_ylabel("Present (1)")
ax0.set_title("Events in Dataset"); ax0.legend(fontsize=8)

ax1 = fig.add_subplot(gs[1])
split_labels = ["Train", "Test"]
split_vals   = [len(df_train), len(df_test)]
ax1.bar(split_labels, split_vals, color=[CYAN, AMBER], alpha=0.85, width=0.5)
for bar, v in zip(ax1.patches, split_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{v}", ha="center", fontsize=10, color=TEXT)
ax1.set_title("Train / Test Split"); ax1.set_ylabel("Events")

ax2 = fig.add_subplot(gs[2])
total_missing = df[VAR_COLS].isna().sum().sum()
labels  = ["Complete rows", "Missing values"]
vals    = [len(df) - (df[VAR_COLS].isna().any(axis=1).sum()), total_missing]
ax2.bar(labels, vals, color=[GREEN, RED], alpha=0.85, width=0.5)
ax2.set_title("Data Quality (selected variables)"); ax2.set_ylabel("Count")

save(fig, "01_overview.png")


# ══════════════════════════════════════════════════════════════
# PLOTS 02–10 — Distribution (Histogram + KDE + Boxplot) per Variable
# ══════════════════════════════════════════════════════════════
for plot_no, (col, clr) in enumerate(VARS, start=2):
    v = df[col].dropna()
    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(13, 6),
                                           gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle(f"Distribution — {col}", fontsize=13)

    sns.histplot(v, bins=30, kde=True, color=clr, alpha=0.85, edgecolor="none",
                 line_kws={"color": TEXT, "linewidth": 1.8}, ax=ax_hist)
    for q, ls, lbl in zip([0.10, 0.25, 0.50, 0.75, 0.90],
                          [":",  ":",  "--", ":",  ":"],
                          ["P10", "P25", "P50", "P75", "P90"]):
        qv = v.quantile(q)
        ax_hist.axvline(qv, color=TEXT, linewidth=1.0, linestyle=ls, label=f"{lbl}={qv:.4f}")
    ax_hist.set_xlabel(col); ax_hist.set_ylabel("Count")
    ax_hist.set_title("Histogram + KDE", fontsize=11)
    ax_hist.legend(fontsize=8)
    txt = (f"μ = {v.mean():.4f}\nσ = {v.std():.4f}\n"
           f"min = {v.min():.4f}\nmax = {v.max():.4f}\n"
           f"skew = {_skew(v):.3f}")
    ax_hist.text(0.97, 0.68, txt, transform=ax_hist.transAxes, ha="right", fontsize=8,
                 color=TEXT, bbox=dict(facecolor=DARK, edgecolor=GRID, alpha=0.7, boxstyle="round"))

    ax_box.boxplot(v, patch_artist=True,
                   boxprops=dict(facecolor=clr, alpha=0.85, edgecolor=TEXT),
                   medianprops=dict(color=TEXT, linewidth=1.5),
                   whiskerprops=dict(color=TEXT), capprops=dict(color=TEXT),
                   flierprops=dict(marker="o", markersize=4, markerfacecolor=clr,
                                    markeredgecolor=TEXT, alpha=0.5))
    ax_box.set_xticklabels([col], fontsize=8)
    ax_box.set_ylabel(col)
    ax_box.set_title("Boxplot", fontsize=11)

    plt.tight_layout()
    fname = f"{plot_no:02d}_dist_{col.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    save(fig, fname)


_next_no = len(VARS) + 2  # plot numbers continue after the per-variable distribution plots

# ══════════════════════════════════════════════════════════════
# PLOT — Correlation Heatmap (selected variables)
# ══════════════════════════════════════════════════════════════
corr = df[VAR_COLS].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, linecolor=GRID, ax=ax,
            annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix — Selected Variables", fontsize=12)
plt.tight_layout()
save(fig, f"{_next_no:02d}_correlation_heatmap.png")
_next_no += 1


# ══════════════════════════════════════════════════════════════
# PLOT — Feature Correlation Bar vs Each Target (ALL engineered features)
# ══════════════════════════════════════════════════════════════
corr_x = df[ALL_FEATURES + [TARGETS[0]]].corr()[TARGETS[0]].drop(TARGETS[0]).sort_values()
corr_y = df[ALL_FEATURES + [TARGETS[1]]].corr()[TARGETS[1]].drop(TARGETS[1]).sort_values()

R_COL_X, R2_COL_X = f"r_vs_{TARGETS[0]}",  f"R2_vs_{TARGETS[0]}"
R_COL_Y, R2_COL_Y = f"r_vs_{TARGETS[1]}",  f"R2_vs_{TARGETS[1]}"
RANK_COL_X, RANK_COL_Y = f"Rank_vs_{TARGETS[0]}", f"Rank_vs_{TARGETS[1]}"

target_corr_df = pd.DataFrame({
    "Feature": ALL_FEATURES,
    R_COL_X:  [corr_x[c] for c in ALL_FEATURES],
    R2_COL_X: [corr_x[c] ** 2 for c in ALL_FEATURES],
    R_COL_Y:  [corr_y[c] for c in ALL_FEATURES],
    R2_COL_Y: [corr_y[c] ** 2 for c in ALL_FEATURES],
})
# Rank by R² (variance explained), not raw r — a high |r| against only one
# target while the other is weak is NOT a genuinely good relationship, and R²
# (unlike r) makes that explicit. Avg_R2 = mean of R2_vs_X and R2_vs_Y rewards
# features that hold up against BOTH target frequencies consistently.
target_corr_df["Avg_R2"] = target_corr_df[[R2_COL_X, R2_COL_Y]].mean(axis=1)
# Per-target ranks (1 = strongest |r| for that target alone), competition-style
# ties (equal |r| shares the same rank; the next distinct value skips ahead).
target_corr_df[RANK_COL_X] = target_corr_df[R_COL_X].abs().rank(method="min", ascending=False).astype("Int64")
target_corr_df[RANK_COL_Y] = target_corr_df[R_COL_Y].abs().rank(method="min", ascending=False).astype("Int64")

target_corr_df = target_corr_df.sort_values("Avg_R2", ascending=False).reset_index(drop=True)
target_corr_df.insert(0, "Overall_Rank", np.arange(1, len(target_corr_df) + 1))
target_corr_df = target_corr_df.round(4)
target_corr_df.to_csv(OUTPUT_DIR / "target_frequency_correlations.csv", index=False)
print("  ✓ target_frequency_correlations.csv (sorted strongest → weakest by Avg_R2)")

fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(ALL_FEATURES) * 0.28)))
fig.suptitle("Feature Correlations with Target Frequencies — All Engineered Features", fontsize=13)

for ax, corr_s, tgt, clr_pos in [
    (axes[0], corr_x, TARGETS[0], CYAN),
    (axes[1], corr_y, TARGETS[1], GREEN),
]:
    clrs = [clr_pos if v >= 0 else RED for v in corr_s.values]
    bars = ax.barh(corr_s.index, corr_s.values, color=clrs, alpha=0.85)
    ax.axvline(0, color=TEXT, linewidth=0.8)
    for bar, v in zip(bars, corr_s.values):
        ax.text(v + (0.005 if v >= 0 else -0.005),
                bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center",
                ha="left" if v >= 0 else "right",
                fontsize=6, color=TEXT)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_xlabel("Pearson r"); ax.set_title(f"vs {tgt}")

plt.tight_layout()
save(fig, f"{_next_no:02d}_feature_target_correlations.png")
_next_no += 1


# ══════════════════════════════════════════════════════════════
# PLOTS — Bivariate Regression Grids (feature(s) vs both targets, r + R²)
# ══════════════════════════════════════════════════════════════
for feats, title, slug, clrs in BIVARIATE_GROUPS:
    n_rows, n_cols = len(feats), len(TARGETS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5.5 * n_rows), squeeze=False)
    fig.suptitle(title, fontsize=13)

    for i, feat in enumerate(feats):
        clr = clrs[i % len(clrs)]
        for j, tgt in enumerate(TARGETS):
            ax = axes[i][j]
            ax.scatter(df[feat], df[tgt], s=20, alpha=0.55, color=clr)
            m, b = np.polyfit(df[feat].values, df[tgt].values, 1)
            xr = np.linspace(df[feat].min(), df[feat].max(), 200)
            ax.plot(xr, m * xr + b, color=TEXT, linewidth=1.3, linestyle="--")
            r_val = df[[feat, tgt]].corr().iloc[0, 1]
            ax.set_xlabel(feat, fontsize=9); ax.set_ylabel(tgt, fontsize=9)
            ax.set_title(f"{feat}  →  {tgt}\nr = {r_val:.3f}   R² = {r_val**2:.3f}", fontsize=10)

    plt.tight_layout()
    save(fig, f"{_next_no:02d}_{slug}.png")
    _next_no += 1


# ══════════════════════════════════════════════════════════════
# PLOT — Pairplot: selected variables
# ══════════════════════════════════════════════════════════════
pp_df = df[VAR_COLS].copy()

fig, axes = plt.subplots(len(VAR_COLS), len(VAR_COLS), figsize=(18, 18))
fig.suptitle("Pairplot — Selected Variables", fontsize=13, y=1.01)

for i, ci in enumerate(VAR_COLS):
    for j, cj in enumerate(VAR_COLS):
        ax = axes[i, j]
        ax.set_facecolor(PANEL)
        if i == j:
            ax.hist(pp_df[ci], bins=30, color=CYAN, alpha=0.8, edgecolor="none")
        else:
            clr = RED if (ci in TARGETS or cj in TARGETS) else CYAN
            ax.scatter(pp_df[cj], pp_df[ci], s=8, alpha=0.35, color=clr)
        if i == len(VAR_COLS) - 1:
            ax.set_xlabel(cj, fontsize=7, rotation=15, ha="right")
        else:
            ax.set_xticklabels([])
        if j == 0:
            ax.set_ylabel(ci, fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=5)

plt.tight_layout()
save(fig, f"{_next_no:02d}_pairplot_selected_variables.png")
_next_no += 1


# ══════════════════════════════════════════════════════════════
# DESCRIPTIVE STATISTICS — CSV + TABLE
# ══════════════════════════════════════════════════════════════
rows = []
for col in VAR_COLS:
    v = df[col].dropna()
    rows.append({
        "Feature": col,     "Count":  len(v),
        "Mean":   round(v.mean(),   5), "Std":    round(v.std(),         5),
        "Min":    round(v.min(),    5), "P25":    round(v.quantile(0.25),5),
        "Median": round(v.median(), 5), "P75":    round(v.quantile(0.75),5),
        "Max":    round(v.max(),    5),
        "Skew":   round(_skew(v),   3), "Kurt":   round(_kurt(v),        3),
    })
stats_df = pd.DataFrame(rows)
stats_df.to_csv(OUTPUT_DIR / "descriptive_stats.csv", index=False)
print("  ✓ descriptive_stats.csv")

fig, ax = plt.subplots(figsize=(14, len(stats_df) * 0.6 + 1.5))
ax.axis("off")
tbl = ax.table(cellText=stats_df.values.tolist(),
               colLabels=stats_df.columns.tolist(),
               cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor(PANEL if r % 2 == 0 else DARK)
    cell.set_edgecolor(GRID)
    cell.set_text_props(color=TEXT if r > 0 else AMBER)
ax.set_title("Descriptive Statistics — Selected Variables", fontsize=12, pad=12)

plt.tight_layout()
save(fig, f"{_next_no:02d}_descriptive_stats_table.png")
_next_no += 1


# ══════════════════════════════════════════════════════════════
# DISPERSION CHARACTERISTICS — CSV
# ══════════════════════════════════════════════════════════════
disp_rows = []
for col in VAR_COLS:
    v = df[col].dropna()
    mean_v  = v.mean()
    std_v   = v.std()
    q25, q75 = v.quantile(0.25), v.quantile(0.75)
    disp_rows.append({
        "Feature":              col,
        "Range":                round(v.max() - v.min(), 5),
        "Variance":             round(v.var(),            5),
        "Std_Dev":              round(std_v,               5),
        "IQR":                  round(q75 - q25,           5),
        "Coeff_Variation_%":    round((std_v / mean_v) * 100, 3) if mean_v != 0 else np.nan,
        "Mean_Abs_Deviation":   round((v - mean_v).abs().mean(), 5),
        "Std_Error_of_Mean":    round(std_v / np.sqrt(len(v)), 5),
    })
dispersion_df = pd.DataFrame(disp_rows)
dispersion_df.to_csv(OUTPUT_DIR / "dispersion_characteristics.csv", index=False)
print("  ✓ dispersion_characteristics.csv")


# ══════════════════════════════════════════════════════════════
# STL DECOMPOSITION (Trend / Seasonal / Residual) — selected series
# ══════════════════════════════════════════════════════════════
for col, clr in STL_SERIES:
    s = df[col].astype(float)
    if s.isna().any():
        s = s.interpolate(limit_direction="both")

    if len(s) < 2 * STL_PERIOD:
        print(f"  SKIP STL for '{col}': series too short for period={STL_PERIOD} "
              f"(need >= {2 * STL_PERIOD}, have {len(s)})")
        continue

    result = STL(s.values, period=STL_PERIOD, robust=STL_ROBUST).fit()

    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f"STL Decomposition — {col}  (period = {STL_PERIOD} events ≈ 1 day)",
                 fontsize=13, y=1.01)

    panels = [
        ("Observed",        s.values,       clr),
        ("Trend",           result.trend,   TEXT),
        ("Seasonal (daily)", result.seasonal, AMBER),
        ("Residual",        result.resid,   GRID),
    ]
    for ax, (label, values, c) in zip(axes, panels):
        if label == "Residual":
            ax.scatter(df["Datetime"], values, color=c, s=6, alpha=0.6, edgecolor=TEXT, linewidth=0.2)
            ax.axhline(0, color=TEXT, linestyle="--", linewidth=0.8)
        else:
            ax.plot(df["Datetime"], values, color=c, linewidth=1.2)
        ax.set_ylabel(label, fontsize=10)

    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    safe_name = col.lower().replace(" ", "_").replace("(", "").replace(")", "")
    save(fig, f"{_next_no:02d}_stl_{safe_name}.png")

    decomp_df = pd.DataFrame({
        "Event_No":       df["Event_No"].values,
        "Datetime":       df["Datetime"].values,
        f"{col}_observed": s.values,
        f"{col}_trend":    result.trend,
        f"{col}_seasonal": result.seasonal,
        f"{col}_resid":    result.resid,
    })
    decomp_df.to_csv(OUTPUT_DIR / f"{_next_no:02d}_stl_{safe_name}_components.csv", index=False)
    print(f"  ✓ {_next_no:02d}_stl_{safe_name}_components.csv")
    _next_no += 1


print(f"\nDone. All outputs saved to: {OUTPUT_DIR}")
