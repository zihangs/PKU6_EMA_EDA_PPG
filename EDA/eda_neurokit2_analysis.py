import os
import glob
import numpy as np
import pandas as pd
from bisect import bisect_left
from datetime import timedelta

# ======================= Configuration =======================

# Base input directories
EDA_DIR = os.environ.get("EDA_DIR", "./data/eda_raw")
LABEL_PATH = os.environ.get("LABEL_PATH", "./data/labels/label.xlsx")

# Output directories
OUT_EDA_DIR = os.environ.get("OUT_EDA_DIR", "./data/eda_aligned")
OUT_LABEL_PATH = os.environ.get("OUT_LABEL_PATH", "./data/labels/label_filtered.xlsx")
OUT_COUNTS_PATH = os.environ.get("OUT_COUNTS_PATH", "./data/labels/label_counts.xlsx")

# Time window (minutes)
WINDOW_BEFORE_MIN = 10
WINDOW_AFTER_MIN = 10

# Alignment mode:
#   "strict"  -> only use events occurring on the same calendar day as EDA
#   "nearest" -> map event to nearest EDA day (within MAX_DAY_GAP_DAYS)
ALIGN_MODE = "strict"          # "strict" or "nearest"
MAX_DAY_GAP_DAYS = 7            # only used when ALIGN_MODE="nearest"

# Optional timezone offset (hours), e.g., if EDA timestamps are UTC
TIME_OFFSET_HOURS = 0

# Fixed label column indices (0-based)
ID_COL_INDEX    = 0   # patient ID
DATE_COL_INDEX  = 1   # date
TIME_COL_INDEX  = 2   # time
SCORE_COL_INDEX = 16  # score column
# =============================================================


def safe_read_csv(path):
    """Try common encodings when reading CSV."""
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def safe_read_excel(path, sheet=0):
    """Read Excel file with openpyxl."""
    return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")


def normalize_id(x: str) -> str:
    """Normalize patient ID string."""
    return str(x).replace("\u3000", " ").strip()


def clean_dt_series(s: pd.Series) -> pd.Series:
    """
    Normalize date/time strings:
      - normalize whitespace
      - replace separators
      - handle common Chinese date symbols
    """
    s = s.astype(str)
    s = s.str.replace("\u3000", " ", regex=False)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.str.replace(r"[.\-]", "/", regex=True)
    s = s.str.replace("年", "/").str.replace("月", "/").str.replace("日", " ")
    return s


def parse_datetime(series: pd.Series):
    """Parse datetime series for internal alignment."""
    dt = pd.to_datetime(clean_dt_series(series), errors="coerce")
    if TIME_OFFSET_HOURS:
        dt = dt + pd.Timedelta(hours=TIME_OFFSET_HOURS)
    return dt


def attach_event_score_to_eda_preserve_order(
    eda_df: pd.DataFrame,
    tcol: str,
    events: list,
    scores: list,
):
    """
    Keep EDA rows that fall inside any event window (event ± window),
    and attach:
        - event_idx
        - event_dt
        - raw_score

    Original row order and original timestamp strings are preserved.
    Requires eda_df to contain auxiliary column "_dt".
    """
    if len(eda_df) == 0 or len(events) == 0:
        return eda_df.iloc[0:0]

    # Build global mask
    mask = pd.Series(False, index=eda_df.index)
    for evt in events:
        start = evt - timedelta(minutes=WINDOW_BEFORE_MIN)
        end   = evt + timedelta(minutes=WINDOW_AFTER_MIN)
        mask |= ((eda_df["_dt"] >= start) & (eda_df["_dt"] <= end))

    eda_win = eda_df.loc[mask].copy()
    if eda_win.empty:
        return eda_win

    # Assign closest event index for each row
    evt_arr = np.array(events, dtype="datetime64[ns]")

    def closest_event_index(ts: pd.Timestamp) -> int:
        pos = bisect_left(evt_arr, np.datetime64(ts))
        cand = []
        if pos < len(evt_arr):
            cand.append((abs(ts - pd.Timestamp(evt_arr[pos])), pos))
        if pos > 0:
            cand.append((abs(ts - pd.Timestamp(evt_arr[pos - 1])), pos - 1))
        return min(cand, key=lambda x: x[0])[1] if cand else 0

    idx_series = eda_win["_dt"].apply(closest_event_index)
    eda_win["event_idx"] = idx_series
    eda_win["event_dt"]  = idx_series.apply(lambda i: events[i])
    eda_win["raw_score"] = idx_series.apply(lambda i: scores[i])

    # Preserve original order
    eda_win = eda_win.reset_index(drop=True)
    return eda_win


def main():
    os.makedirs(OUT_EDA_DIR, exist_ok=True)

    # 1) Collect patient IDs from EDA filenames
    eda_files = sorted(glob.glob(os.path.join(EDA_DIR, "*.csv")))
    patient_ids = {normalize_id(os.path.splitext(os.path.basename(p))[0]) for p in eda_files}
    print(f"Detected patients: {len(patient_ids)} -> {sorted(list(patient_ids))}")

    # 2) Load label file and filter by patients
    lab = safe_read_excel(LABEL_PATH, sheet=0)
    need_cols = max(ID_COL_INDEX, DATE_COL_INDEX, TIME_COL_INDEX, SCORE_COL_INDEX) + 1
    if lab.shape[1] < need_cols:
        raise ValueError(f"Label file must contain at least {need_cols} columns.")

    id_col    = lab.columns[ID_COL_INDEX]
    date_col  = lab.columns[DATE_COL_INDEX]
    time_col  = lab.columns[TIME_COL_INDEX]
    score_col = lab.columns[SCORE_COL_INDEX]

    lab[id_col] = lab[id_col].apply(normalize_id)
    lab = lab[lab[id_col].isin(patient_ids)].copy()

    lab["_event_dt"]  = parse_datetime(
        clean_dt_series(lab[date_col]) + " " + clean_dt_series(lab[time_col])
    )
    lab["_raw_score"] = pd.to_numeric(lab[score_col], errors="coerce")
    lab = lab[lab["_event_dt"].notna()].copy()
    lab = lab.sort_values([id_col, "_event_dt"]).reset_index(drop=True)

    # Export filtered labels
    lab_out_cols = [id_col, date_col, time_col, score_col, "_raw_score", "_event_dt"]
    lab[lab_out_cols].to_excel(OUT_LABEL_PATH, index=False, engine="openpyxl")
    print(f"Exported filtered labels -> {OUT_LABEL_PATH} ({len(lab)} rows)")

    # Count number of assessments per patient
    counts = lab.groupby(lab[id_col]).size().rename("num_assessments")
    counts.to_excel(OUT_COUNTS_PATH, header=True, engine="openpyxl")
    print(f"Exported assessment counts -> {OUT_COUNTS_PATH}")

    # 3) Process EDA per patient
    kept_files, total_rows = 0, 0

    for eda_path in eda_files:
        pid = normalize_id(os.path.splitext(os.path.basename(eda_path))[0])

        sub_all = lab.loc[lab[id_col] == pid, ["_event_dt", "_raw_score"]].copy()
        events_all = sub_all["_event_dt"].tolist()
        scores_all = sub_all["_raw_score"].tolist()

        eda = safe_read_csv(eda_path)
        if eda.shape[1] < 1:
            print(f"[{pid}] WARNING: invalid EDA file, skipped.")
            continue

        tcol = eda.columns[0]
        eda["_dt"] = parse_datetime(eda[tcol])
        eda = eda[eda["_dt"].notna()].copy()

        # Event alignment
        matched_events, matched_scores = [], []
        if len(events_all) > 0 and len(eda) > 0:
            eda_days = sorted(set(eda["_dt"].dt.normalize().unique()))

            if ALIGN_MODE == "strict":
                eda_date_set = set(d.date() for d in eda_days)
                for e, s in zip(events_all, scores_all):
                    if e.date() in eda_date_set:
                        matched_events.append(e)
                        matched_scores.append(s)
            else:  # nearest
                for e, s in zip(events_all, scores_all):
                    if not eda_days:
                        continue
                    e_day = pd.Timestamp(e.date())
                    best, gap = None, None
                    for d in eda_days:
                        g = abs((d - e_day).days)
                        if gap is None or g < gap:
                            best, gap = d, g
                    if gap is not None and gap <= MAX_DAY_GAP_DAYS:
                        new_e = best + pd.Timedelta(
                            hours=e.hour, minutes=e.minute, seconds=e.second
                        )
                        matched_events.append(new_e)
                        matched_scores.append(s)

        # Attach scores and keep only windowed rows
        eda_out = attach_event_score_to_eda_preserve_order(
            eda, tcol, matched_events, matched_scores
        )

        # Drop auxiliary column before saving
        cols = [c for c in eda_out.columns if c != "_dt"]
        out_path = os.path.join(OUT_EDA_DIR, f"{pid}.csv")
        eda_out[cols].to_csv(out_path, index=False, encoding="utf-8-sig")

        print(
            f"{pid}: matched events {len(matched_events)}/{len(events_all)}, "
            f"kept rows {len(eda_out)} -> {out_path}"
        )
        kept_files += 1
        total_rows += len(eda_out)

    print(f"\nExported EDA files: {kept_files}, total rows kept: {total_rows}")


if __name__ == "__main__":
    main()
