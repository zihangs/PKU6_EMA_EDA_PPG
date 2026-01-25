# -*- coding: utf-8 -*-
"""
Step2 (standalone):
- Load event-level long EDA sequences from eda_event_raw_long.npy (object array)
- Clean / interpolate / bandpass filter (highpass -> lowpass)
- Split SCL/SCR (main.py-style):
    scr = highpass(denoised, cutoff=HIGHPASS_HZ)
    scl = denoised - scr
  (NaN mask is preserved)
- Record quality metrics per event (missing/outlier/SNR) to a CSV

Outputs (saved under BASE_DIR):
- eda_event_raw_long_filt.npy   : filtered long sequence (object array)
- eda_event_scl_long.npy        : SCL long sequence (object array)
- eda_event_scr_long.npy        : SCR long sequence (object array)
- eda_event_quality_long.csv    : quality metrics per event (aligned to meta rows)
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

np.set_printoptions(precision=4, suppress=True)

# ======================= Configuration =======================

BASE_DIR = os.environ.get("BASE_DIR", "./data/event_long")

IN_RAW_LONG_NPY = os.path.join(BASE_DIR, "eda_event_raw_long.npy")
IN_LABEL_NPY    = os.path.join(BASE_DIR, "eda_event_labels.npy")
IN_META_CSV     = os.path.join(BASE_DIR, "eda_event_meta_long.csv")

OUT_RAW_LONG_FILT_NPY = os.path.join(BASE_DIR, "eda_event_raw_long_filt.npy")
OUT_SCL_LONG_NPY      = os.path.join(BASE_DIR, "eda_event_scl_long.npy")
OUT_SCR_LONG_NPY      = os.path.join(BASE_DIR, "eda_event_scr_long.npy")
OUT_QUALITY_CSV       = os.path.join(BASE_DIR, "eda_event_quality_long.csv")

FS = 4.0
LOWPASS_HZ   = 1.0
HIGHPASS_HZ  = 0.05
BUTTER_ORDER = 4

INVALID_LEQ = -8.0  # Values <= INVALID_LEQ are treated as invalid (set to NaN) and counted as outliers

# Quality thresholds (record-only; no early exit)
QUALITY_MISSING_MAX = 0.05
QUALITY_OUTLIER_MAX = 0.02
QUALITY_SNR_MIN_DB  = 18.0
SNR_NOISE_CUTOFF_HZ = 1.0

# ============================================================


def butter_lowpass_filter(x: np.ndarray, fs: float, cutoff: float, order: int = 4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return filtfilt(b, a, x)


def butter_highpass_filter(x: np.ndarray, fs: float, cutoff: float, order: int = 4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="high", analog=False)
    return filtfilt(b, a, x)


def _linear_interp_fill_nan(x: np.ndarray) -> np.ndarray:
    """Fill NaNs by linear interpolation (for filter stability)."""
    x = np.asarray(x, dtype=np.float32)
    mask_nan = ~np.isfinite(x)
    if np.all(mask_nan):
        return x.copy()

    idx = np.arange(len(x))
    valid = ~mask_nan
    x_valid = x[valid]
    idx_valid = idx[valid]

    filled = np.interp(idx, idx_valid, x_valid).astype(np.float32)
    return filled


def compute_quality_metrics(x_raw: np.ndarray, fs: float) -> dict:
    """
    Quality metrics (main.py-like), record-only:
    - missing_ratio: NaN ratio (treat <= INVALID_LEQ as NaN)
    - outlier_ratio: ratio of (x <= INVALID_LEQ)
    - snr_db: estimate noise via highpass(1Hz); SNR = 10*log10(var(signal)/var(noise))
    """
    x = np.asarray(x_raw, dtype=np.float32)

    outlier_ratio = float(np.mean(x <= INVALID_LEQ))

    x2 = x.copy()
    x2[x2 <= INVALID_LEQ] = np.nan
    missing_ratio = float(np.mean(~np.isfinite(x2)))

    valid = x2[np.isfinite(x2)]
    if valid.size < 10:
        snr_db = float("nan")
    else:
        signal_power = float(np.var(valid))
        try:
            filled = _linear_interp_fill_nan(x2)
            nyq = 0.5 * fs
            b, a = butter(4, SNR_NOISE_CUTOFF_HZ / nyq, btype="high", analog=False)
            noise_signal = filtfilt(b, a, filled)
            noise_power = float(np.var(noise_signal[np.isfinite(noise_signal)]))
            if noise_power > 0 and signal_power > 0:
                snr_db = float(10.0 * np.log10(signal_power / noise_power))
            elif noise_power == 0 and signal_power > 0:
                snr_db = float("inf")
            else:
                snr_db = float("nan")
        except Exception:
            d = np.diff(valid)
            noise_std = float(np.std(d)) if d.size > 1 else float("nan")
            sig_std = float(np.std(valid))
            if noise_std > 0 and sig_std > 0:
                snr_db = float(10.0 * np.log10((sig_std ** 2) / (noise_std ** 2)))
            else:
                snr_db = float("nan")

    passed = True
    if missing_ratio >= QUALITY_MISSING_MAX:
        passed = False
    if outlier_ratio >= QUALITY_OUTLIER_MAX:
        passed = False
    if np.isfinite(snr_db) and snr_db < QUALITY_SNR_MIN_DB:
        passed = False

    return {
        "missing_ratio": missing_ratio,
        "outlier_ratio": outlier_ratio,
        "snr_db": snr_db,
        "passed": passed,
    }


def preprocess_and_filter_one(
    x: np.ndarray,
    fs: float,
    low_cutoff: float,
    high_cutoff: float,
    order: int = 4,
) -> np.ndarray:
    """
    Input:
        Event-level long EDA sequence (may contain NaN or invalid values).
    Steps:
        1) Treat <= INVALID_LEQ as invalid -> set NaN
        2) Fill NaN via linear interpolation
        3) Highpass then lowpass (bandpass)
        4) Restore original NaN mask
    Output:
        Filtered sequence with NaNs preserved.
    """
    x = np.asarray(x, dtype=np.float32)

    x[x <= INVALID_LEQ] = np.nan
    mask_nan = ~np.isfinite(x)
    if np.all(mask_nan):
        return x

    x_filled = _linear_interp_fill_nan(x)

    try:
        x_hp = butter_highpass_filter(x_filled, fs=fs, cutoff=high_cutoff, order=order)
    except Exception as e:
        print("[WARN] Highpass filter failed; skipping highpass:", e)
        x_hp = x_filled

    try:
        x_bp = butter_lowpass_filter(x_hp, fs=fs, cutoff=low_cutoff, order=order)
    except Exception as e:
        print("[WARN] Lowpass filter failed; using highpass output only:", e)
        x_bp = x_hp

    x_bp = np.asarray(x_bp, dtype=np.float32)
    x_bp[mask_nan] = np.nan
    return x_bp.astype(np.float32)


def split_scl_scr(
    denoised_with_nan: np.ndarray,
    fs: float,
    scr_highpass_hz: float,
    order: int = 4,
) -> tuple:
    """
    main.py-style SCL/SCR split:
        scr = highpass(denoised)
        scl = denoised - scr
    NaN mask is preserved.
    """
    x = np.asarray(denoised_with_nan, dtype=np.float32)
    mask_nan = ~np.isfinite(x)

    if np.all(mask_nan):
        return x.copy(), x.copy()

    filled = _linear_interp_fill_nan(x)
    try:
        scr = butter_highpass_filter(filled, fs=fs, cutoff=scr_highpass_hz, order=order).astype(np.float32)
    except Exception as e:
        print("[WARN] SCR highpass failed; returning zero SCR:", e)
        scr = np.zeros_like(filled, dtype=np.float32)

    scr[mask_nan] = np.nan
    scl = (x - scr).astype(np.float32)
    return scl, scr


def main():
    print("[INFO] Loading event-level long sequences, labels, and metadata...")
    X_raw_long = np.load(IN_RAW_LONG_NPY, allow_pickle=True)
    y_events   = np.load(IN_LABEL_NPY)
    meta_df    = pd.read_csv(IN_META_CSV)

    assert len(X_raw_long) == len(y_events) == len(meta_df), \
        "raw_long / labels / meta length mismatch. Please check Step1 outputs."

    print(f"[INFO] Num events: {len(X_raw_long)}")
    print(f"[INFO] Label distribution: {np.bincount(y_events)}")
    print(f"[INFO] FS={FS}, band={HIGHPASS_HZ}~{LOWPASS_HZ} Hz, order={BUTTER_ORDER}")

    X_filt_list = []
    X_scl_list  = []
    X_scr_list  = []
    q_rows = []

    skipped_all_nan = 0
    printed_samples = 0
    max_print_samples = 3

    for i, x in enumerate(X_raw_long):
        eid = f"{meta_df.iloc[i].get('pid', 'NA')}_{meta_df.iloc[i].get('event_idx', 'NA')}"
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[INFO] Processing event {i+1}/{len(X_raw_long)} -> {eid}")

        x = np.asarray(x, dtype=np.float32)

        q = compute_quality_metrics(x, fs=FS)

        x_filt = preprocess_and_filter_one(
            x, fs=FS, low_cutoff=LOWPASS_HZ, high_cutoff=HIGHPASS_HZ, order=BUTTER_ORDER
        )

        scl, scr = split_scl_scr(
            denoised_with_nan=x_filt,
            fs=FS,
            scr_highpass_hz=HIGHPASS_HZ,
            order=BUTTER_ORDER,
        )

        if np.all(~np.isfinite(x_filt)):
            skipped_all_nan += 1

        X_filt_list.append(x_filt)
        X_scl_list.append(scl)
        X_scr_list.append(scr)

        q_rows.append({
            "i": i,
            "pid": meta_df.iloc[i].get("pid", None),
            "event_idx": meta_df.iloc[i].get("event_idx", None),
            "label": int(y_events[i]),
            "n_points": int(len(x)),
            **q,
        })

        if printed_samples < max_print_samples:
            label_i = int(y_events[i])
            print(f"\n========== Sample event {i} ==========")
            print(f"Event ID: {eid}, label={label_i}, length={len(x)}")
            print("quality:", q)
            print("raw first 20 (with NaN):")
            print(x[:20])
            print("filtered first 20 (NaN preserved):")
            print(x_filt[:20])
            print("SCL first 20:")
            print(scl[:20])
            print("SCR first 20:")
            print(scr[:20])
            printed_samples += 1

    X_filt_obj = np.array(X_filt_list, dtype=object)
    X_scl_obj  = np.array(X_scl_list, dtype=object)
    X_scr_obj  = np.array(X_scr_list, dtype=object)

    lens = [len(x) for x in X_filt_obj]
    print("\n========== Step2 output statistics ==========")
    print(f"Total events: {len(X_filt_obj)}")
    print(f"Label distribution: {np.bincount(y_events)}")
    print(f"Length: min={int(np.min(lens))}, max={int(np.max(lens))}, mean={np.mean(lens):.2f}")
    print(f"[INFO] All-NaN events (after filtering): {skipped_all_nan}")

    np.save(OUT_RAW_LONG_FILT_NPY, X_filt_obj)
    np.save(OUT_SCL_LONG_NPY,      X_scl_obj)
    np.save(OUT_SCR_LONG_NPY,      X_scr_obj)

    q_df = pd.DataFrame(q_rows)
    q_df.to_csv(OUT_QUALITY_CSV, index=False, encoding="utf-8-sig")

    print(f"\n[INFO] Saved filtered long sequences: {OUT_RAW_LONG_FILT_NPY} (object, len={len(X_filt_obj)})")
    print(f"[INFO] Saved SCL long sequences:      {OUT_SCL_LONG_NPY} (object, len={len(X_scl_obj)})")
    print(f"[INFO] Saved SCR long sequences:      {OUT_SCR_LONG_NPY} (object, len={len(X_scr_obj)})")
    print(f"[INFO] Saved quality metrics CSV:     {OUT_QUALITY_CSV} (rows={len(q_df)})")
    print("[INFO] Labels and metadata remain unchanged.")


if __name__ == "__main__":
    main()
