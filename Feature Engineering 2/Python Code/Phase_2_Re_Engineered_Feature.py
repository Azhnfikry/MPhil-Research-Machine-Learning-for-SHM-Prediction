"""
=============================================================
Phase 2 -- Re-Engineered Feature Pipeline
(built on Stage B: 30-Minute Window Spectral / Hydrodynamic Aggregation)
=============================================================
Input : per-event processed Excel files written by the Stage A script
        (30min_v2_data_processing.py):
            Export NLDP\\Processed_Output\\E{event}_processed.xlsx
        Columns: File Name, Scan Number, Date, Time, X/Y/Z Acc Raw,
        Wave Raw, X/Y Acc Filtered, Wave Filtered, X/Y/Wave Frequency,
        X/Y Velocity, X/Y Displacement -- ~18,000 rows/event @ 10 Hz.

Output: engineered_30min_v2_features.csv (one row per event, base
        spectral PSD / time-domain / hydrodynamic columns + Phase 2's
        re-engineered columns, + X/Y Frequency targets)
        + TrainTestSplit_30min_v2\\Chronological\\ and \\Shuffled\\
          train/test CSVs (80/20, same event pool, two split styles)
        saved to 30min_v2_Spectral_Experiment\\Output\\

Run AFTER Stage A (30min_v2_data_processing.py) has produced the
per-event processed .xlsx files. This script always rebuilds from those
files on every run -- no cache short-circuit -- so changes to Stage A's
filtering/frequency formula are always picked up.

PHASE 2 RE-ENGINEERED FEATURES (added on top of Stage B's base columns,
to address the low chronological-split R2 caused by static per-window
snapshots, feature collinearity, and FFT bin quantization noise):
  Rank 1 -- Temporal lags & moving statistics: 1-/2-step lags and
            3-step rolling mean/std on the key spectral columns, plus
            first-difference (event-to-event change) features.
  Rank 2 -- Sub-bin spectral refinement: parabolic (log-quadratic)
            interpolation across each PSD peak's 3 neighbouring Welch
            bins, smoothing out the discrete FFT frequency-step
            quantization in *_PeakFreq{i} -> *_PeakFreq{i}_Refined.
  Rank 3/4 -- Cross-axis interaction ratios: biaxial peak-frequency
              ratio and spectral centroid-to-peak ratio.
This script only builds and persists the re-engineered feature set (+
chronological/shuffled splits) -- no model training here; run the
existing ML scripts in 30min_v2_Spectral_Experiment\\ML\\ against this
output to evaluate it.
=============================================================
"""

import os
import re
import glob
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import welch, find_peaks
from scipy.stats import skew as _skew, kurtosis as _kurt

# ============================================================
# CONFIG
# ============================================================
# Stage A's OUTPUT_FOLDER -- must match 30min_v2_data_processing.py.
PROCESSED_FOLDER = Path(r"C:\Users\N O\Desktop\MASTER\Methodology & Result\Methodology\Export NLDP\Processed_Output")
OUTPUT_DIR = Path(r"C:\Users\N O\Desktop\MASTER\Methodology & Result\Phase 2 Experiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENGINEERED_FILE = OUTPUT_DIR / "engineered_30min_v2_features.csv"

SPLIT_DIR_CHRONOLOGICAL = OUTPUT_DIR / "TrainTestSplit_30min_v2" / "Chronological"
SPLIT_DIR_SHUFFLED      = OUTPUT_DIR / "TrainTestSplit_30min_v2" / "Shuffled"
TRAIN_FILE_CHRONO   = SPLIT_DIR_CHRONOLOGICAL / "train_dataset_30min_v2_80pct_chronological.csv"
TEST_FILE_CHRONO    = SPLIT_DIR_CHRONOLOGICAL / "test_dataset_30min_v2_20pct_chronological.csv"
TRAIN_FILE_SHUFFLED = SPLIT_DIR_SHUFFLED / "train_dataset_30min_v2_80pct_shuffled.csv"
TEST_FILE_SHUFFLED  = SPLIT_DIR_SHUFFLED / "test_dataset_30min_v2_20pct_shuffled.csv"

ID_START, ID_END       = 571, 1098   # matches Stage A's ID_START/ID_END range

SAMPLING_RATE  = 10   # Hz -- matches Stage A's FS
WINDOW_MINUTES = 30

TARGET_COLS = ["X Frequency", "Y Frequency"]

# All raw signals aggregated for the time-domain / kinetic block.
STAT_SIGNALS = [
    "Wave Raw", "Wave Filtered",
    "X Acc Raw", "X Acc Filtered", "Y Acc Raw", "Y Acc Filtered", "Z Acc Raw",
    "X Displacement", "Y Displacement", "X Velocity", "Y Velocity",
]

# Accelerations put through Welch PSD spectral analysis -- ax, ay, az.
# No "Z Acc Filtered" channel exists in Stage A's output, so Z uses raw.
SPECTRAL_SIGNALS = [
    "X Acc Raw", "X Acc Filtered", "Y Acc Raw", "Y Acc Filtered", "Z Acc Raw",
]
N_SPECTRAL_PEAKS = 3

# ---- Phase 2 re-engineered feature config ---------------------------------
# Rank 1: key spectral columns lagged/rolled/differenced across the
# chronological Event_No sequence (each row is one 30-min window).
LAG_FEATURE_COLS = [
    "Y_Acc_Filtered_PeakFreq1",
    "X_Acc_Filtered_PeakFreq1",
    "Y_Acc_Filtered_SpectralCentroid",
]
ROLLING_STAT_COL   = "Y_Acc_Filtered_PeakFreq1"
ROLLING_WINDOW     = 3
LAG_STEPS          = [1, 2]

# Columns where a literal 0 is a legitimate value (event-to-event change of
# exactly zero, or a ratio that can genuinely be null) -- must be excluded
# from filter_impute_split()'s "0 == missing" sentinel hygiene pass, unlike
# the raw physical spectral/statistical columns.
ZERO_SAFE_SUFFIXES = ("_Diff1",)
ZERO_SAFE_COLS = {"Biaxial_PeakFreq_Ratio", "SpectralCentroid_to_Peak_Ratio"}


# ============================================================
# HELPERS
# ============================================================
def fmt_elapsed(seconds):
    """H:MM:SS (or M:SS under an hour) for printing timing info."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ============================================================
# PER-WINDOW SPECTRAL / TIME-DOMAIN / HYDRODYNAMIC FEATURES
# ============================================================
def compute_wave_zero_upcrossing_tz(wave_filtered, sampling_rate):
    """Mean zero-crossing period (s) of the filtered wave signal, via
    mean-centered upcrossing spacing."""
    s = np.asarray(wave_filtered, dtype=float)
    s = s - np.mean(s)
    upcrossings = np.where((s[:-1] <= 0) & (s[1:] > 0))[0]
    if len(upcrossings) < 2:
        return np.nan
    periods = np.diff(upcrossings) / sampling_rate
    return float(np.mean(periods))


def parabolic_log_refine(f, Pxx, idx):
    """Sub-bin peak refinement (Rank 2): quadratic ("parabolic"/Gaussian)
    interpolation of the log-PSD across the peak bin and its two immediate
    neighbours. Welch's PSD is only resolved to fixed-width bins (f[1]-f[0]),
    so the discrete argmax alone is quantization noise; fitting a parabola
    to log(Pxx) at idx-1/idx/idx+1 and solving for its vertex recovers a
    continuous sub-bin frequency estimate. Falls back to the raw bin
    frequency at the spectrum's edges or on a degenerate (flat) parabola."""
    if idx <= 0 or idx >= len(Pxx) - 1:
        return float(f[idx])

    alpha = np.log(max(Pxx[idx - 1], 1e-300))
    beta  = np.log(max(Pxx[idx],     1e-300))
    gamma = np.log(max(Pxx[idx + 1], 1e-300))

    denom = alpha - 2.0 * beta + gamma
    if denom == 0:
        return float(f[idx])

    # Vertex offset in bins, clipped to +/-0.5 -- the interpolated peak can
    # never cross into a neighbouring bin's own interpolation range.
    p = 0.5 * (alpha - gamma) / denom
    p = float(np.clip(p, -0.5, 0.5))

    bin_width = float(f[1] - f[0])
    return float(f[idx]) + p * bin_width


def spectral_features(signal, sampling_rate, n_peaks=N_SPECTRAL_PEAKS):
    """Welch PSD -> top-N peak frequencies/powers (+ sub-bin refined
    frequencies), spectral centroid, bandwidth, and normalised (0-1
    bounded) spectral entropy."""
    x = np.asarray(signal, dtype=float)
    x = x - np.mean(x)
    nperseg = min(len(x), 1024)
    if nperseg < 8:
        out = {f"PeakFreq{i}": np.nan for i in range(1, n_peaks + 1)}
        out.update({f"PeakFreq{i}_Refined": np.nan for i in range(1, n_peaks + 1)})
        out.update({f"PeakPower{i}": np.nan for i in range(1, n_peaks + 1)})
        out.update(SpectralCentroid=np.nan, SpectralBandwidth=np.nan, SpectralEntropy=np.nan)
        return out

    f, Pxx = welch(x, fs=sampling_rate, nperseg=nperseg)

    peak_idx, _ = find_peaks(Pxx)
    if len(peak_idx) == 0:
        peak_idx = np.array([int(np.argmax(Pxx))])
    order = peak_idx[np.argsort(Pxx[peak_idx])[::-1]][:n_peaks]

    out = {}
    for i in range(n_peaks):
        if i < len(order):
            idx = int(order[i])
            out[f"PeakFreq{i+1}"]          = float(f[idx])
            out[f"PeakFreq{i+1}_Refined"]  = parabolic_log_refine(f, Pxx, idx)
            out[f"PeakPower{i+1}"]         = float(Pxx[idx])
        else:
            out[f"PeakFreq{i+1}"]         = np.nan
            out[f"PeakFreq{i+1}_Refined"] = np.nan
            out[f"PeakPower{i+1}"]        = np.nan

    p_sum = Pxx.sum()
    if p_sum > 0:
        centroid  = float(np.sum(f * Pxx) / p_sum)
        bandwidth = float(np.sqrt(np.sum(Pxx * (f - centroid) ** 2) / p_sum))
        p_norm    = Pxx / p_sum
        p_norm    = p_norm[p_norm > 0]
        entropy   = float(-np.sum(p_norm * np.log2(p_norm)) / np.log2(len(Pxx)))
    else:
        centroid = bandwidth = entropy = np.nan

    out["SpectralCentroid"]  = centroid
    out["SpectralBandwidth"] = bandwidth
    out["SpectralEntropy"]   = entropy
    return out


def compute_window_features(event_df, sampling_rate):
    """One aggregated feature row from a single event's Stage A output --
    spectral + time-domain + hydrodynamic."""
    row = {}

    # ---- Spectral (Welch PSD) ------------------------------------------
    for sig in SPECTRAL_SIGNALS:
        feats  = spectral_features(event_df[sig], sampling_rate)
        prefix = sig.replace(" ", "_")
        for k, v in feats.items():
            row[f"{prefix}_{k}"] = v

    # ---- Time-domain / kinetic indicators --------------------------------
    std_lookup = {}
    for sig in STAT_SIGNALS:
        s   = event_df[sig].astype(float)
        rms = float(np.sqrt(np.mean(s ** 2)))
        std = float(s.std())
        prefix = sig.replace(" ", "_")
        row[f"{prefix}_RMS"]         = rms
        row[f"{prefix}_Std"]         = std
        row[f"{prefix}_Skew"]        = float(_skew(s))
        row[f"{prefix}_Kurtosis"]    = float(_kurt(s))
        row[f"{prefix}_CrestFactor"] = float(np.max(np.abs(s)) / rms) if rms > 0 else np.nan
        std_lookup[sig] = std

    row["Horizontal_Displacement_RMS"] = float(np.sqrt(
        std_lookup["X Displacement"] ** 2 + std_lookup["Y Displacement"] ** 2
    ))

    # ---- Oceanographic / hydrodynamic summaries ---------------------------
    wave_filtered = event_df["Wave Filtered"].astype(float)
    Hs   = float(4.0 * wave_filtered.std())
    Hmax = float(wave_filtered.max() - wave_filtered.min())
    Tz   = compute_wave_zero_upcrossing_tz(wave_filtered, sampling_rate)
    row["Hs (m)"]   = Hs
    row["Hmax (m)"] = Hmax
    row["Tz (s)"]   = Tz
    row["Wave_Steepness"]    = Hs / (Tz ** 2 + 1e-6) if not np.isnan(Tz) else np.nan
    row["Hydro_Interaction"] = Hs * std_lookup["Y Displacement"]

    # ---- Targets ------------------------------------------------------------
    # Stage A already broadcasts one FAMOS-aligned dominant frequency value
    # to every row of the event, so .mean() here just recovers that single
    # scalar -- kept as a mean (not .iloc[0]) so this still works correctly
    # if Stage A's output ever goes back to a per-row varying estimate.
    row["X Frequency"] = float(event_df["X Frequency"].astype(float).mean())
    row["Y Frequency"] = float(event_df["Y Frequency"].astype(float).mean())

    return row


# ============================================================
# PHASE 2 -- RE-ENGINEERED FEATURES (Rank 1, 3, 4)
# Operate on the whole engineered_df (one row per event, chronologically
# ordered by Event_No) -- Rank 2 (sub-bin refinement) is already computed
# per-window above, inside spectral_features().
# ============================================================
def add_temporal_lag_features(df):
    """Rank 1: 1-/2-step lags + 3-step rolling mean/std + first-difference
    on the key spectral columns, computed across the chronological
    Event_No sequence (each row = one 30-min window, so a "lag" here is
    the previous window's value, not a within-window lag)."""
    df = df.sort_values("Event_No").reset_index(drop=True)

    for col in LAG_FEATURE_COLS:
        for step in LAG_STEPS:
            df[f"{col}_Lag{step}"] = df[col].shift(step)
        # First-difference: change vs. the immediately preceding window.
        df[f"{col}_Diff1"] = df[col] - df[col].shift(1)

    df[f"{ROLLING_STAT_COL}_RollMean{ROLLING_WINDOW}"] = (
        df[ROLLING_STAT_COL].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    )
    df[f"{ROLLING_STAT_COL}_RollStd{ROLLING_WINDOW}"] = (
        df[ROLLING_STAT_COL].rolling(window=ROLLING_WINDOW, min_periods=1).std()
    )
    return df


def add_cross_axis_ratio_features(df):
    """Rank 3/4: cross-axis interaction ratios (row-wise, no chronology
    involved -- safe to compute independently on train/test)."""
    df["Biaxial_PeakFreq_Ratio"] = (
        df["X_Acc_Filtered_PeakFreq1"] / (df["Y_Acc_Filtered_PeakFreq1"] + 1e-8)
    )
    df["SpectralCentroid_to_Peak_Ratio"] = (
        df["Y_Acc_Filtered_SpectralCentroid"] / (df["Y_Acc_Filtered_PeakFreq1"] + 1e-8)
    )
    return df


# ============================================================
# COMBINE ALL EVENTS
# ============================================================
def event_id(path):
    m = re.match(r"E(\d+)_processed\.xlsx", os.path.basename(path))
    return int(m.group(1)) if m else None


def extract_event_files():
    files = glob.glob(str(PROCESSED_FOLDER / "E*_processed.xlsx"))
    # Sort by the numeric event ID, not the file path string -- a plain
    # string sort puts "E1000..." before "E571..." (since "1" < "5"
    # lexicographically).
    files = [(event_id(f), f) for f in files]
    files = [(n, f) for n, f in files if n is not None and ID_START <= n <= ID_END]
    files.sort(key=lambda t: t[0])
    return files


def build_engineered_dataset():
    files = extract_event_files()
    if not files:
        print(f"No processed event files found in {PROCESSED_FOLDER} "
              f"(range E{ID_START}-E{ID_END}). Run Stage A first.")
        return None

    print(f"Found {len(files)} processed event files (E{ID_START}-E{ID_END}).")
    rows, skipped = [], []
    t0 = time.time()
    for i, (event_no, xlsx) in enumerate(files):
        try:
            event_df = pd.read_excel(xlsx)
        except Exception as e:
            print(f"  Warning: could not read E{event_no} ({os.path.basename(xlsx)}), skipping: {e}")
            skipped.append((event_no, str(e)))
            continue

        event_df.columns = event_df.columns.str.strip()

        row = compute_window_features(event_df, SAMPLING_RATE)
        row["Source_File"] = f"E{event_no}_processed"
        row["Event_No"]    = event_no
        rows.append(row)

        if i % 20 == 0:
            elapsed = time.time() - t0
            avg_per_event = elapsed / (i + 1)
            eta = avg_per_event * (len(files) - (i + 1))
            print(f"  event {i:,} / {len(files):,} (E{event_no}, "
                  f"{len(event_df):,} raw rows -> 1 window row)  "
                  f"elapsed {fmt_elapsed(elapsed)}  ETA {fmt_elapsed(eta)}")

    build_elapsed = time.time() - t0
    print(f"\nFeature extraction done in {fmt_elapsed(build_elapsed)} "
          f"({len(files)} events).")

    if skipped:
        print(f"\nSkipped {len(skipped)} unreadable file(s): {[f'E{n}' for n, _ in skipped]}")

    if not rows:
        print("No events could be read -- aborting.")
        return None

    engineered_df = pd.DataFrame(rows)

    base_col_count = engineered_df.shape[1]
    engineered_df = add_temporal_lag_features(engineered_df)
    engineered_df = add_cross_axis_ratio_features(engineered_df)
    print(f"\nPhase 2 re-engineered features added: "
          f"{engineered_df.shape[1] - base_col_count} new columns "
          f"({base_col_count} base -> {engineered_df.shape[1]} total).")

    # Event_No / Source_File first, for readability.
    id_cols = ["Event_No", "Source_File"]
    engineered_df = engineered_df[id_cols + [c for c in engineered_df.columns if c not in id_cols]]

    engineered_df.to_csv(ENGINEERED_FILE, index=False)
    print(f"\nSaved: {ENGINEERED_FILE}")
    print(f"Engineered rows: {len(engineered_df):,}  |  Columns: {engineered_df.shape[1]}")

    for tgt in TARGET_COLS:
        n_nan  = engineered_df[tgt].isna().sum()
        n_zero = (engineered_df[tgt] == 0).sum()
        print(f"  {tgt}: NaN={n_nan}, zero={n_zero}")

    return engineered_df


# ============================================================
# FILTER + IMPUTE + SPLIT (chronological + shuffled)
# ============================================================
def filter_impute_split(df):
    t0 = time.time()
    df = df.copy()
    df = df[(df["Event_No"] >= ID_START) & (df["Event_No"] <= ID_END)]
    df = df.sort_values("Event_No").reset_index(drop=True)

    id_cols      = ["Event_No", "Source_File"] + TARGET_COLS
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in id_cols]

    # Sentinel-value hygiene: a literal 0 in a spectral/statistical feature
    # column is not physically meaningful for any of these metrics --
    # treat as missing and impute. Excludes Phase 2's diff/ratio columns
    # (ZERO_SAFE_SUFFIXES / ZERO_SAFE_COLS), where an exact 0 is a
    # legitimate value (e.g. no change vs. the previous window), not a
    # sentinel for missing data.
    zero_hygiene_cols = [
        c for c in feature_cols
        if c not in ZERO_SAFE_COLS and not c.endswith(ZERO_SAFE_SUFFIXES)
    ]
    zero_counts = (df[zero_hygiene_cols] == 0).sum()
    zero_counts = zero_counts[zero_counts > 0]
    if len(zero_counts) > 0:
        print("Columns with literal 0 values (treated as missing):")
        print(zero_counts.sort_values(ascending=False))
        df[zero_hygiene_cols] = df[zero_hygiene_cols].replace(0, np.nan)

    missing_before = df[feature_cols].isna().sum()
    missing_before = missing_before[missing_before > 0]
    if len(missing_before) > 0:
        print("\nMissing values found before imputation:")
        print(missing_before)
        for col in missing_before.index:
            rolling_mean = df[col].shift(1).rolling(window=10, min_periods=1).mean()
            df[col] = df[col].fillna(rolling_mean)
        still_missing = df[feature_cols].isna().sum()
        still_missing = still_missing[still_missing > 0]
        if len(still_missing) > 0:
            print("\nValues with no prior window available (filled with overall column mean):")
            print(still_missing)
            for col in still_missing.index:
                df[col] = df[col].fillna(df[col].mean())
        print(f"\nRemaining missing values after imputation: {df[feature_cols].isna().sum().sum()}")
    else:
        print("No missing values found in feature columns.")

    print(f"\nTotal Samples (E{ID_START}-E{ID_END}): {len(df)}")

    SPLIT_DIR_CHRONOLOGICAL.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR_SHUFFLED.mkdir(parents=True, exist_ok=True)

    split_index = int(len(df) * 0.8)

    # Chronological -- df is already sorted by Event_No, one row per event,
    # so a straight row-count slice can't straddle an event.
    train_chrono = df.iloc[:split_index].reset_index(drop=True)
    test_chrono  = df.iloc[split_index:].reset_index(drop=True)
    train_chrono.to_csv(TRAIN_FILE_CHRONO, index=False)
    test_chrono.to_csv(TEST_FILE_CHRONO, index=False)
    print(f"\nChronological -- Train: {len(train_chrono)} | Test: {len(test_chrono)}")
    print(f"Saved: {TRAIN_FILE_CHRONO}")
    print(f"Saved: {TEST_FILE_CHRONO}")

    # Shuffled -- same cleaned/imputed pool, randomly partitioned instead.
    df_shuffled    = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_shuffled = df_shuffled.iloc[:split_index].reset_index(drop=True)
    test_shuffled  = df_shuffled.iloc[split_index:].reset_index(drop=True)
    train_shuffled.to_csv(TRAIN_FILE_SHUFFLED, index=False)
    test_shuffled.to_csv(TEST_FILE_SHUFFLED, index=False)
    print(f"\nShuffled -- Train: {len(train_shuffled)} | Test: {len(test_shuffled)}")
    print(f"Saved: {TRAIN_FILE_SHUFFLED}")
    print(f"Saved: {TEST_FILE_SHUFFLED}")

    print(f"\nFilter/impute/split done in {fmt_elapsed(time.time() - t0)}.")

    return train_chrono, test_chrono, train_shuffled, test_shuffled


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    script_start_time = time.time()

    print("=" * 60)
    print("PHASE 2: Aggregate Stage A's processed events -> re-engineered 30-minute window features")
    print("=" * 60)
    engineered_df = build_engineered_dataset()

    if engineered_df is None:
        raise SystemExit("Stopping: no processed event files found.")

    print("\n" + "=" * 60)
    print("Filter, impute, and split (chronological + shuffled)")
    print("=" * 60)
    filter_impute_split(engineered_df)

    print("\n" + "=" * 60)
    print(f"TOTAL TIME: {fmt_elapsed(time.time() - script_start_time)}")
    print("=" * 60)
