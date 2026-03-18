from pathlib import Path
import pickle
from collections import defaultdict
import pandas as pd


# =========================
# Paths
# =========================
INPUT_DIR = Path(r"C:\Users\USER\Documents\GitHub\datasets\simbarca\all_agg")
OUTPUT_DIR = Path(r"C:\Users\USER\Documents\GitHub\datasets\simbarca\all_agg_trimmed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Time window function
# =========================
def trim_df_time_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep data from:
        first_timestamp + 15 minutes
    to
        first_timestamp + 2h15 minutes

    So this keeps a 2-hour window starting 15 minutes after the beginning.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected DataFrame with DatetimeIndex.")

    if df.empty:
        return df.copy()

    df = df.sort_index()

    start_time = df.index.min()
    window_start = start_time + pd.Timedelta(minutes=15)
    window_end = window_start + pd.Timedelta(hours=2)

    trimmed = df.loc[(df.index >= window_start) & (df.index < window_end)].copy()
    return trimmed


# =========================
# Process all pickle files
# =========================
pickle_files = sorted(INPUT_DIR.glob("*.pkl"))
print(f"Found {len(pickle_files)} pickle files.")

for pkl_path in pickle_files:
    print(f"\nProcessing: {pkl_path.name}")

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        print(f"  Skipped: top-level object is not a dict/defaultdict, got {type(obj)}")
        continue

    # Preserve defaultdict if needed
    if isinstance(obj, defaultdict):
        trimmed_obj = defaultdict(obj.default_factory)
    else:
        trimmed_obj = {}

    kept_any = False

    for key, value in obj.items():
        if isinstance(value, pd.DataFrame):
            try:
                trimmed_df = trim_df_time_window(value)
                trimmed_obj[key] = trimmed_df
                kept_any = True

                if not trimmed_df.empty:
                    print(
                        f"  {key}: kept {len(trimmed_df)} rows "
                        f"from {trimmed_df.index.min()} to {trimmed_df.index.max()}"
                    )
                else:
                    print(f"  {key}: trimmed result is empty")
            except Exception as e:
                print(f"  {key}: error while trimming -> {e}")
                trimmed_obj[key] = value
        else:
            # keep non-DataFrame entries unchanged
            trimmed_obj[key] = value

    if not kept_any:
        print("  Skipped: no DataFrames found inside this pickle.")
        continue

    output_path = OUTPUT_DIR / pkl_path.name
    with open(output_path, "wb") as f:
        pickle.dump(trimmed_obj, f)

    print(f"  Saved: {output_path}")

print("\nDone.")