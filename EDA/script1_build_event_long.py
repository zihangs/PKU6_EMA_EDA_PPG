# script1_build_event_long.py
# Build event-level long EDA sequences from per-patient CSV files.

import os
import glob
import ast
import numpy as np
import pandas as pd

# ======================= Configuration =======================

EDA_DIR = os.environ.get("EDA_DIR", "./data/eda_aligned")
OUT_DIR = os.environ.get("OUT_DIR", "./data/event_long")

FS = 4.0                       # Sampling rate (Hz)
TIME_COL = None               # If None, use the first column as timestamp
EDA_COL_GUESS = ["eda", "EDA", "EDA_Raw"]  # Candidate EDA column names

# =============================================================

os.makedirs(OUT_DIR, exist_ok=True)

RAW_LONG_NPY = os.path.join(OUT_DIR, "eda_event_raw_long.npy")
LABEL_NPY    = os.path.join(OUT_DIR, "eda_event_labels.npy")
META_CSV     = os.path.join(OUT_DIR, "eda_event_meta_long.csv")


def pick_time_col(df: pd.DataFrame) -> str:
    """Select the timestamp column."""
    if TIME_COL is not None:
        return TIME_COL
    return df.columns[0]


def pick_eda_col(df: pd.DataFrame) -> str:
    """Select the EDA column using heuristics."""
    for c in EDA_COL_GUESS:
        if c in df.columns:
            return c
    # Fallback: use the third column if available
    if len(df.columns) >= 3:
        return df.columns[2]
    raise ValueError("Cannot determine EDA column. Please update EDA_COL_GUESS.")


def eda_is_list_string(series: pd.Series) -> bool:
    """Check whether the EDA column stores list-like strings."""
    sample = series.dropna().astype(str).head(10).tolist()
    return any(s.strip().startswith("[") and s.strip().endswith("]") for s in sample)


def flatten_eda_for_rows(
    df: pd.DataFrame,
    time_col: str,
    eda_col: str,
    fs: float,
):
    """
    Flatten EDA values for a single event group.

    Case 1:
        EDA column is scalar -> directly return numeric series.

    Case 2:
        EDA column contains list-like strings -> parse and concatenate.

    Returns:
        eda_flat: np.ndarray of shape [N_points]
    """
    # Case 1: scalar values
    if not eda_is_list_string(df[eda_col]):
        eda_flat = pd.to_numeric(df[eda_col], errors="coerce").to_numpy()
        return eda_flat

    # Case 2: list-like strings per row
    eda_list = []
    for _, row in df.iterrows():
        txt = str(row[eda_col]).strip()
        try:
            arr = ast.literal_eval(txt)
        except Exception:
            txt2 = txt.strip("[]").replace(";", ",")
            parts = [t for t in txt2.split(",") if t.strip() != ""]
            try:
                arr = [float(t) for t in parts]
            except Exception:
                arr = []

        if len(arr) > 0:
            eda_list.extend(arr)

    return np.array(eda_list, dtype=float)


def map_score_to_label(score):
    """Map raw score to binary label."""
    try:
        v = float(score)
    except Exception:
        return None
    if 0 <= v <= 1:
        return 0
    if 5 <= v <= 9:
        return 1
    return None


def main():
    csv_files = sorted(
        f for f in glob.glob(os.path.join(EDA_DIR, "*.csv"))
        if os.path.basename(f).lower().endswith(".csv")
    )
    print(f"[INFO] Found patient-level EDA files: {len(csv_files)}")

    all_raw = []
    all_labels = []
    meta_rows = []

    total_events = 0
    skipped_score = 0
    skipped_empty = 0

    for path in csv_files:
        base = os.path.basename(path)
        pid = os.path.splitext(base)[0]
        df = pd.read_csv(path)

        if "event_idx" not in df.columns or "raw_score" not in df.columns:
            print(f"  [WARN] {base} missing event_idx/raw_score, skipped.")
            continue

        time_col = pick_time_col(df)
        eda_col  = pick_eda_col(df)

        # Group by event index
        for eidx, sub in df.groupby("event_idx"):
            sub = sub.copy()

            raw_score = pd.to_numeric(sub["raw_score"].iloc[0], errors="coerce")
            label = map_score_to_label(raw_score)
            if label is None:
                skipped_score += 1
                continue

            # Flatten EDA sequence
            eda_flat = flatten_eda_for_rows(sub, time_col, eda_col, fs=FS)

            # Cast to float32
            eda_flat = eda_flat.astype(np.float32)

            # Mark clearly invalid values as NaN
            eda_flat[eda_flat <= -8] = np.nan

            if eda_flat.size == 0 or np.all(~np.isfinite(eda_flat)):
                skipped_empty += 1
                continue

            all_raw.append(eda_flat)
            all_labels.append(int(label))
            total_events += 1

            meta_rows.append({
                "pid": pid,
                "event_idx": int(eidx),
                "raw_score": float(raw_score) if pd.notna(raw_score) else None,
                "label": int(label),
                "n_points": int(len(eda_flat)),
                "duration_sec": float(len(eda_flat) / FS),
            })

    if total_events == 0:
        print("[ERROR] No valid event-level sequences were generated.")
        return

    X_raw_obj = np.array(all_raw, dtype=object)
    y_events  = np.array(all_labels, dtype=np.int64)
    meta_df   = pd.DataFrame(meta_rows).reset_index(drop=True)

    print("\n========== Event-level EDA sequence statistics ==========")
    print(f"Valid events: {len(X_raw_obj)}")
    print(f"Label distribution: {np.bincount(y_events)}")
    lens = [len(x) for x in X_raw_obj]
    print(
        f"Sequence length per event: "
        f"min={int(np.min(lens))}, "
        f"max={int(np.max(lens))}, "
        f"mean={np.mean(lens):.2f}"
    )
    print(
        f"[INFO] Skipped events: "
        f"invalid score={skipped_score}, "
        f"empty/all-NaN={skipped_empty}"
    )

    # Save outputs
    np.save(RAW_LONG_NPY, X_raw_obj)
    np.save(LABEL_NPY,    y_events)
    meta_df.to_csv(META_CSV, index=False, encoding="utf-8-sig")

    print("\n[INFO] Saved outputs:")
    print(f"  Event-level sequences: {RAW_LONG_NPY} (object array, len={len(X_raw_obj)})")
    print(f"  Event-level labels:    {LABEL_NPY}    (shape={y_events.shape})")
    print(f"  Event-level metadata:  {META_CSV}")


if __name__ == "__main__":
    main()
