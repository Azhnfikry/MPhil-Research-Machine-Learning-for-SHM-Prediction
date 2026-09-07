"""
=============================================================
EDA -- 30-Minute (v2) Spectral / Hydrodynamic Engineered Features
=============================================================
Input : engineered_30min_v2_features.csv
        One row per event. Restricted to a representative variable set:
        X Frequency, Y Frequency, Hs (m), Hmax (m), Tz (s),
        Wave_Steepness, Hydro_Interaction, Horizontal_Displacement_RMS,
        X_Acc_Filtered_RMS, Y_Acc_Filtered_RMS,
        X_Acc_Filtered_SpectralCentroid, Y_Acc_Filtered_SpectralCentroid,
        Z_Acc_Raw_SpectralEntropy
Output: distribution / correlation / pairplot / STL PNGs
        + descriptive_stats.csv + dispersion_characteristics.csv
        + target_frequency_correlations.csv (ALL ~108 candidate features)
        saved to 30min_v2_Spectral_Experiment\\Output\\EDA_Plots_30min_v2\\
=============================================================
"""

import warnings; warnings.filterwarnings("ignore")
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
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
_BASE = Path(r"C:\Users\N O\Desktop\MASTER\Methodology & Result\Phase 2 Experiment\TrainTestSplit_30min_v2\Chronological")
ENGINEERED_FILE = _BASE / "Combined_Dataset_V2.csv"  # input CSV with engineered features
OUTPUT_DIR      = _BASE / "EDA_Plots_30min_Final"  # output directory for plots and CSVs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILTER_MIN, FILTER_MAX = 571, 1098   # matches the data-processing script's final filter range

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
    print(f"  \u2713 {name}")


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
print("Loading data …")
if not ENGINEERED_FILE.exists():
    raise SystemExit(
        f"Engineered feature file not found: {ENGINEERED_FILE}\n"
        f"Run 30min_v2_data_processing.py first."
    )
df = pd.read_csv(ENGINEERED_FILE, low_memory=False)
df.columns = df.columns.str.strip()
df = df[(df["Event_No"] >= FILTER_MIN) & (df["Event_No"] <= FILTER_MAX)]
df = df.sort_values("Event_No").reset_index(drop=True)
print(f"  {len(df):,} events | {df.shape[1]} columns (filtered to E{FILTER_MIN}-E{FILTER_MAX})")

# Campaign timing -- E571 = 2021-09-28 00:08, ~30-min spacing (one event per row)
CAMPAIGN_ANCHOR_EVENT = 571
CAMPAIGN_ANCHOR_TIME  = pd.Timestamp("2021-09-28 00:08:00")
EVENT_STEP            = pd.Timedelta(minutes=30)
df["Datetime"] = CAMPAIGN_ANCHOR_TIME + (df["Event_No"] - CAMPAIGN_ANCHOR_EVENT) * EVENT_STEP

TARGETS = ["X Frequency", "Y Frequency"]

# Representative variable set for this EDA pass -- distribution / correlation
# / pairplot are restricted to these; the feature-target correlation table
# below still covers every candidate column.
VARS = [
    ("X Frequency",                       CYAN),
    ("Y Frequency",                       GREEN),
    ("Hs (m)",                            PURPLE),
    ("Tz (s)",                            AMBER),
    ("Wave_Steepness",                    RED),
    ("Hydro_Interaction",                 PURPLE),
    ("Horizontal_Displacement_RMS",       PINK),
    ("X_Acc_Filtered_RMS",                CYAN),
    ("Y_Acc_Filtered_RMS",                GREEN),
    ("X_Acc_Filtered_SpectralCentroid",   AMBER),
    ("Y_Acc_Filtered_SpectralCentroid",   RED),
    ("Z_Acc_Raw_SpectralEntropy",         PURPLE),
    # -- Phase 2 re-engineered features --
    ("Y_Acc_Filtered_PeakFreq1_Refined",  GREEN),   # Rank 2: sub-bin spectral refinement
    ("Y_Acc_Filtered_PeakFreq1_Lag1",     CYAN),    # Rank 1: 1-step temporal lag
    ("Y_Acc_Filtered_PeakFreq1_Diff1",    AMBER),   # Rank 1: first-difference
    ("Y_Acc_Filtered_PeakFreq1_RollStd3", RED),     # Rank 1: 3-step rolling std
    ("Biaxial_PeakFreq_Ratio",            PINK),    # Rank 3: cross-axis peak-freq ratio
    ("SpectralCentroid_to_Peak_Ratio",    PURPLE),  # Rank 4: centroid-to-peak ratio
]
VAR_COLS = [c for c, _ in VARS if c in df.columns]
VARS     = [(c, clr) for c, clr in VARS if c in VAR_COLS]

EXCLUDE_COLS  = ["Event_No", "Source_File", "Datetime"]
ALL_FEATURES  = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in EXCLUDE_COLS + TARGETS]

# STL decomposition -- events are ~30-min apart -> 48 events/day
STL_PERIOD = 48
STL_ROBUST = True
STL_SERIES = [
    ("Hs (m)",              PURPLE),
    ("X_Acc_Filtered_RMS",  AMBER),
    ("Y_Acc_Filtered_RMS",  RED),
    ("X Frequency",         CYAN),
    ("Y Frequency",         GREEN),
]
STL_SERIES = [(c, clr) for c, clr in STL_SERIES if c in df.columns]

# Bivariate regression grids -- feature(s) vs both target frequencies
BIVARIATE_GROUPS = [
    (["Hs (m)"],
     "Significant Wave Height (Hs) vs Natural Frequencies", "hs_vs_frequency", [PURPLE]),
    (["Tz (s)"],
     "Zero-Crossing Period (Tz) vs Natural Frequencies", "tz_vs_frequency", [AMBER]),
    (["X_Acc_Filtered_RMS", "Y_Acc_Filtered_RMS"],
     "RMS Filtered Acceleration vs Natural Frequencies", "acc_filtered_rms_vs_frequency", [CYAN, GREEN]),
    (["X_Velocity_Std", "Y_Velocity_Std"],
     "Std Dev Velocity vs Natural Frequencies", "velocity_stddev_vs_frequency", [CYAN, GREEN]),
    (["X_Displacement_Std", "Y_Displacement_Std"],
     "Std Dev Displacement vs Natural Frequencies", "displacement_stddev_vs_frequency", [CYAN, GREEN]),
    (["X_Acc_Filtered_SpectralCentroid", "Y_Acc_Filtered_SpectralCentroid"],
     "Spectral Centroid (Filtered Acceleration) vs Natural Frequencies",
     "spectral_centroid_vs_frequency", [CYAN, GREEN]),
    (["Wave_Steepness"],
     "Dynamic Wave Steepness vs Natural Frequencies", "wave_steepness_vs_frequency", [RED]),
    (["Hydro_Interaction"],
     "Inertia-Drag Proxy (Hydro Interaction) vs Natural Frequencies",
     "hydro_interaction_vs_frequency", [PINK]),
    (["Y_Acc_Filtered_PeakFreq1_Refined", "X_Acc_Filtered_PeakFreq1_Refined"],
     "Sub-Bin Refined Peak Frequency vs Natural Frequencies",
     "peakfreq_refined_vs_frequency", [GREEN, CYAN]),
    (["Y_Acc_Filtered_PeakFreq1_Lag1", "Y_Acc_Filtered_PeakFreq1_Diff1"],
     "Peak Frequency Lag1 & First-Difference vs Natural Frequencies",
     "peakfreq_lag_diff_vs_frequency", [CYAN, AMBER]),
    (["Biaxial_PeakFreq_Ratio", "SpectralCentroid_to_Peak_Ratio"],
     "Cross-Axis Interaction Ratios vs Natural Frequencies",
     "cross_axis_ratios_vs_frequency", [PINK, PURPLE]),
]
BIVARIATE_GROUPS = [
    (feats, title, slug, clrs) for feats, title, slug, clrs in BIVARIATE_GROUPS
    if all(f in df.columns for f in feats)
]


# ─────────────────────────────────────────────
# DATA WRANGLING -- quick look
# ─────────────────────────────────────────────
print("\nData wrangling …")
df.info()
print(df[VAR_COLS].describe())
print("\nMissing values per column (representative set):")
print(df[VAR_COLS].isna().sum()[lambda s: s > 0])


# ══════════════════════════════════════════════════════════════
# PLOT 01 -- Dataset Overview
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 6))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
fig.suptitle("Dataset Overview -- 30-Minute (v2) Spectral Features (1 row per event)", fontsize=13, y=1.02)

ax0 = fig.add_subplot(gs[0])
ax0.bar(df["Event_No"], [1] * len(df), color=CYAN, alpha=0.85, width=1.2)
ax0.set_xlabel("Event No"); ax0.set_ylabel("Present (1)")
ax0.set_title("Events in Dataset")

ax1 = fig.add_subplot(gs[1])
n_features = len(ALL_FEATURES)
ax1.bar(["Events", "Candidate\nFeatures"], [len(df), n_features],
        color=[CYAN, AMBER], alpha=0.85, width=0.5)
for bar, v in zip(ax1.patches, [len(df), n_features]):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{v}", ha="center", fontsize=10, color=TEXT)
ax1.set_title("Dataset Size"); ax1.set_ylabel("Count")

ax2 = fig.add_subplot(gs[2])
total_missing = df[VAR_COLS].isna().sum().sum()
labels = ["Complete rows", "Missing values"]
vals   = [len(df) - df[VAR_COLS].isna().any(axis=1).sum(), total_missing]
ax2.bar(labels, vals, color=[GREEN, RED], alpha=0.85, width=0.5)
ax2.set_title("Data Quality (representative variables)"); ax2.set_ylabel("Count")

save(fig, "01_overview.png")


# ══════════════════════════════════════════════════════════════
# PLOTS 02+ -- Distribution (Histogram + KDE + Boxplot) per Variable
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
    txt = (f"\u03bc = {v.mean():.4f}\n\u03c3 = {v.std():.4f}\n"
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


_next_no = len(VARS) + 2

# ══════════════════════════════════════════════════════════════
# PLOT -- Correlation Heatmap (representative variables)
# ══════════════════════════════════════════════════════════════
corr = df[VAR_COLS].corr()
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, linecolor=GRID, ax=ax,
            annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix — Representative Variables", fontsize=12)
plt.tight_layout()
save(fig, f"{_next_no:02d}_correlation_heatmap.png")
_next_no += 1


# ══════════════════════════════════════════════════════════════
# PLOT -- Feature Correlation Bar vs Each Target (ALL candidate features)
# ══════════════════════════════════════════════════════════════
corr_x = df[ALL_FEATURES + [TARGETS[0]]].corr()[TARGETS[0]].drop(TARGETS[0]).sort_values()
corr_y = df[ALL_FEATURES + [TARGETS[1]]].corr()[TARGETS[1]].drop(TARGETS[1]).sort_values()

target_corr_df = pd.DataFrame({
    "Feature": ALL_FEATURES,
    f"r_vs_{TARGETS[0]}":  [corr_x[c] for c in ALL_FEATURES],
    f"R2_vs_{TARGETS[0]}": [corr_x[c] ** 2 for c in ALL_FEATURES],
    f"r_vs_{TARGETS[1]}":  [corr_y[c] for c in ALL_FEATURES],
    f"R2_vs_{TARGETS[1]}": [corr_y[c] ** 2 for c in ALL_FEATURES],
})
# Rank best -> worst relationship by R2 (direction-agnostic strength).
# Avg_R2 drives the overall ranking; per-target ranks are also kept since
# a feature can be strong for one target and weak for the other.
target_corr_df["Avg_R2"] = target_corr_df[[f"R2_vs_{TARGETS[0]}", f"R2_vs_{TARGETS[1]}"]].mean(axis=1)
target_corr_df[f"Rank_vs_{TARGETS[0]}"] = target_corr_df[f"R2_vs_{TARGETS[0]}"].rank(ascending=False, method="min").astype(int)
target_corr_df[f"Rank_vs_{TARGETS[1]}"] = target_corr_df[f"R2_vs_{TARGETS[1]}"].rank(ascending=False, method="min").astype(int)
target_corr_df = target_corr_df.sort_values("Avg_R2", ascending=False).reset_index(drop=True)
target_corr_df.insert(0, "Overall_Rank", target_corr_df.index + 1)
target_corr_df = target_corr_df.round(4)
target_corr_df.to_csv(OUTPUT_DIR / "target_frequency_correlations.csv", index=False)
print("  \u2713 target_frequency_correlations.csv")

fig, axes = plt.subplots(1, 2, figsize=(16, max(10, len(ALL_FEATURES) * 0.16)))
fig.suptitle("Feature Correlations with Target Frequencies — All Candidate Features", fontsize=13)

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
                fontsize=5, color=TEXT)
    ax.tick_params(axis="y", labelsize=5)
    ax.set_xlabel("Pearson r"); ax.set_title(f"vs {tgt}")

plt.tight_layout()
save(fig, f"{_next_no:02d}_feature_target_correlations.png")
_next_no += 1


# ══════════════════════════════════════════════════════════════
# PLOTS -- Bivariate Regression Grids (feature(s) vs both targets, r + R²)
# ══════════════════════════════════════════════════════════════
for feats, title, slug, clrs in BIVARIATE_GROUPS:
    n_rows, n_cols = len(feats), len(TARGETS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5.5 * n_rows), squeeze=False)
    fig.suptitle(title, fontsize=13)

    for i, feat in enumerate(feats):
        clr = clrs[i % len(clrs)]
        for j, tgt in enumerate(TARGETS):
            ax = axes[i][j]
            sub = df[[feat, tgt]].dropna()
            ax.scatter(sub[feat], sub[tgt], s=20, alpha=0.55, color=clr)
            m, b = np.polyfit(sub[feat].values, sub[tgt].values, 1)
            xr = np.linspace(sub[feat].min(), sub[feat].max(), 200)
            ax.plot(xr, m * xr + b, color=TEXT, linewidth=1.3, linestyle="--")
            r_val = sub.corr().iloc[0, 1]
            ax.set_xlabel(feat, fontsize=9); ax.set_ylabel(tgt, fontsize=9)
            ax.set_title(f"{feat}  \u2192  {tgt}\nr = {r_val:.3f}   R\u00b2 = {r_val**2:.3f}", fontsize=10)

    plt.tight_layout()
    save(fig, f"{_next_no:02d}_{slug}.png")
    _next_no += 1


# ══════════════════════════════════════════════════════════════
# PLOT -- Pairplot: representative variables
# ══════════════════════════════════════════════════════════════
pp_df = df[VAR_COLS].copy()

fig, axes = plt.subplots(len(VAR_COLS), len(VAR_COLS), figsize=(20, 20))
fig.suptitle("Pairplot — Representative Variables", fontsize=13, y=1.01)

for i, ci in enumerate(VAR_COLS):
    for j, cj in enumerate(VAR_COLS):
        ax = axes[i, j]
        ax.set_facecolor(PANEL)
        if i == j:
            ax.hist(pp_df[ci].dropna(), bins=30, color=CYAN, alpha=0.8, edgecolor="none")
        else:
            clr = RED if (ci in TARGETS or cj in TARGETS) else CYAN
            sub = pp_df[[cj, ci]].dropna()
            ax.scatter(sub[cj], sub[ci], s=8, alpha=0.35, color=clr)
        if i == len(VAR_COLS) - 1:
            ax.set_xlabel(cj, fontsize=6, rotation=25, ha="right")
        else:
            ax.set_xticklabels([])
        if j == 0:
            ax.set_ylabel(ci, fontsize=6)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=5)

plt.tight_layout()
save(fig, f"{_next_no:02d}_pairplot_representative_variables.png")
_next_no += 1


# ══════════════════════════════════════════════════════════════
# DESCRIPTIVE STATISTICS -- CSV + TABLE
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
print("  \u2713 descriptive_stats.csv")

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
ax.set_title("Descriptive Statistics — Representative Variables", fontsize=12, pad=12)

plt.tight_layout()
save(fig, f"{_next_no:02d}_descriptive_stats_table.png")
_next_no += 1


# ══════════════════════════════════════════════════════════════
# DISPERSION CHARACTERISTICS -- CSV
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
print("  \u2713 dispersion_characteristics.csv")


# ══════════════════════════════════════════════════════════════
# STL DECOMPOSITION (Trend / Seasonal / Residual) -- selected series
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
    fig.suptitle(f"STL Decomposition — {col}  (period = {STL_PERIOD} events \u2248 1 day)",
                 fontsize=13, y=1.01)

    panels = [
        ("Observed",         s.values,        clr),
        ("Trend",            result.trend,    TEXT),
        ("Seasonal (daily)", result.seasonal, AMBER),
        ("Residual",         result.resid,    GRID),
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
        "Event_No":        df["Event_No"].values,
        "Datetime":        df["Datetime"].values,
        f"{col}_observed": s.values,
        f"{col}_trend":    result.trend,
        f"{col}_seasonal": result.seasonal,
        f"{col}_resid":    result.resid,
    })
    decomp_df.to_csv(OUTPUT_DIR / f"{_next_no:02d}_stl_{safe_name}_components.csv", index=False)
    print(f"  \u2713 {_next_no:02d}_stl_{safe_name}_components.csv")
    _next_no += 1


print(f"\nDone. All outputs saved to: {OUTPUT_DIR}")
