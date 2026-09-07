# ======================================================
# 30-MINUTE RE-ENGINEERED FEATURE EXPORT
# Reads E{id}_processed.xlsx (10 Hz raw files) from Processed_Output.
# Each file ≈ 18 000 rows = one 30-minute window at 10 Hz.
# Aggregates the entire file into ONE row of engineered features.
#
# Output columns per row (matches train_dataset_30minute_80pct_shuffled.csv):
#   Event_No, Source_File
#   {sensor}_Min/Max/Mean/Median/Std_Dev   for each sensor column
#   X Frequency, Y Frequency                (targets, mean over the window)
#
# Final output: one combined CSV with one row per file.
# ======================================================

import pandas as pd
import numpy as np
from pathlib import Path
import traceback
import time

# ======================================================
# CONFIGURATION
# ======================================================

INPUT_FOLDER   = Path(r"C:\Users\N O\Desktop\MASTER\Methodology & Result\Methodology\Export NLDP\Processed_Output")
OUTPUT_FOLDER  = Path(r"C:\Users\N O\Desktop\MASTER\Methodology & Result\Pre-Final Dataset\Dataset")

ID_START = 571
ID_END   = 1098

OUTPUT_FILE = OUTPUT_FOLDER / "combined_30min_features.csv"

# Chronological train/test split (no shuffling -- rows are already in
# ascending Event_No order, so the first TRAIN_FRACTION of rows become the
# train set and the remainder become the test set).
TRAIN_FRACTION = 0.8
TRAIN_FILE = OUTPUT_FOLDER / "train_dataset_30minute_80pct_chronological.csv"
TEST_FILE  = OUTPUT_FOLDER / "test_dataset_30minute_20pct_chronological.csv"

# Sensor columns to aggregate (5 stats each): Min/Max/Mean/Median/Std_Dev.
SENSOR_COLS = [
    "Wave Raw", "Wave Filtered", "Wave Frequency",
    "X Acc Raw", "X Acc Filtered",
    "Y Acc Raw", "Y Acc Filtered",
    "Z Acc Raw",
    "X Displacement", "Y Displacement",
    "X Velocity", "Y Velocity",
]

# Target columns — mean over the window
TARGET_COLS = ["X Frequency", "Y Frequency"]


# ======================================================
# HELPERS
# ======================================================

def fmt_elapsed(seconds):
    """H:MM:SS (or M:SS under an hour) for printing timing info."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ======================================================
# PROCESS ONE FILE → one-row DataFrame
# ======================================================

def process_file(xlsx_path: Path, file_id: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    df.columns = df.columns.str.strip()

    sensor_present = [c for c in SENSOR_COLS if c in df.columns]
    target_present = [c for c in TARGET_COLS if c in df.columns]

    if not sensor_present and not target_present:
        print(f"  WARN: {file_id} — no usable columns, skipping.")
        return pd.DataFrame()

    row = {"Event_No": file_id, "Source_File": file_id}

    # ── Statistics for each sensor column ─────────────────────────────
    for col in sensor_present:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        row[f"{col}_Min"]     = s.min()    if len(s) else np.nan
        row[f"{col}_Max"]     = s.max()    if len(s) else np.nan
        row[f"{col}_Mean"]    = s.mean()   if len(s) else np.nan
        row[f"{col}_Median"]  = s.median() if len(s) else np.nan
        row[f"{col}_Std_Dev"] = s.std()    if len(s) else np.nan

    # ── Targets — mean over the window ────────────────────────────────
    for col in target_present:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        row[col] = s.mean() if len(s) else np.nan

    return pd.DataFrame([row])


# ======================================================
# CHRONOLOGICAL TRAIN/TEST SPLIT
# No shuffling -- combined is already in ascending Event_No order, so the
# split is a straight head/tail cut at TRAIN_FRACTION.
# ======================================================

def split_chronological(combined: pd.DataFrame, train_fraction: float = TRAIN_FRACTION):
    n_train = int(round(len(combined) * train_fraction))
    train_df = combined.iloc[:n_train]
    test_df  = combined.iloc[n_train:]

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print(f"\n  Chronological split ({train_fraction:.0%} train / {1 - train_fraction:.0%} test):")
    print(f"    Train: {len(train_df):,} rows  ->  {TRAIN_FILE}")
    print(f"    Test : {len(test_df):,} rows  ->  {TEST_FILE}")

    return train_df, test_df


# ======================================================
# MAIN
# ======================================================

def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    ids   = list(range(ID_START, ID_END + 1))
    total = len(ids)
    all_frames = []
    succeeded, skipped, failed = [], [], []

    print(f"\n{'='*65}")
    print(f"  30-min Re-Engineered Feature Export (10 Hz raw files)")
    print(f"  Files : E{ID_START} to E{ID_END}  ({total} IDs)")
    print(f"  Input : {INPUT_FOLDER}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"{'='*65}\n")

    t0 = time.time()

    for i, eid in enumerate(ids, start=1):
        xlsx_path = INPUT_FOLDER / f"E{eid}_processed.xlsx"
        file_id   = f"E{eid}_processed"

        if not xlsx_path.exists():
            skipped.append(eid)
            continue

        try:
            df_out = process_file(xlsx_path, file_id)
            if df_out.empty:
                skipped.append(eid)
            else:
                all_frames.append(df_out)
                succeeded.append(eid)
                elapsed = time.time() - t0
                avg_per_file = elapsed / i
                eta = avg_per_file * (total - i)
                print(f"[{i:>4}/{total}]  OK    E{eid}  "
                      f"elapsed {fmt_elapsed(elapsed)}  ETA {fmt_elapsed(eta)}")
        except Exception:
            failed.append(eid)
            print(f"[{i:>4}/{total}]  FAIL  E{eid}")
            traceback.print_exc()

    # ── Combine and save ───────────────────────────────────────────────
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(OUTPUT_FILE, index=False)
        print(f"\n  Saved {len(combined):,} rows ({len(combined.columns)} cols)"
              f"  ->  {OUTPUT_FILE}")
        split_chronological(combined)
    else:
        print("\n  No data to save.")

    total_time = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  Done in {fmt_elapsed(total_time)}")
    print(f"  Succeeded : {len(succeeded)}")
    print(f"  Skipped   : {len(skipped)}")
    print(f"  Failed    : {len(failed)}   {failed}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
