# script2_filter_event_long_ppg_dedup.py
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch, find_peaks, peak_widths

# ====== 直接复用你上传的两个模块（去重关键点）======
from artifact import detect_motion_artifacts, detect_outliers
from filtering import apply_filters

np.set_printoptions(precision=4, suppress=True)

# ========= 可配置 =========
BASE_DIR = r"/Users/zhushiyao/Desktop/train_set_103_filtered_20_ppg/event_long"

IN_RAW_LONG_NPY = os.path.join(BASE_DIR, "ppg_event_raw_long.npy")
IN_LABEL_NPY    = os.path.join(BASE_DIR, "ppg_event_labels.npy")
IN_META_CSV     = os.path.join(BASE_DIR, "ppg_event_meta_long.csv")

OUT_RAW_LONG_FILT_NPY = os.path.join(BASE_DIR, "ppg_event_raw_long_filt.npy")

# 额外输出（不影响原流程；想要就 True）
EXTRACT_FEATURES = True
OUT_FEATURE_CSV  = os.path.join(BASE_DIR, "ppg_event_features_long.csv")

FS = 25.0
# PPG 心率带通：约 0.4~3 Hz（24~180 bpm）；注意 fs=4Hz Nyquist=2Hz，apply_filters 内部会做保护
LOWPASS_HZ = 3.0
HIGHPASS_HZ = 0.4
FILTER_ORDER = 4

# 伪影/异常检测参数（来自 artifact.py 逻辑）
MOTION_THRESHOLD_FACTOR = 3.48
OUTLIER_WINDOW = 50
OUTLIER_THRESHOLD = 3.0

# 无效值规则（沿用你原脚本 <= -8 的约定；如果 PPG 无效编码不同就在这里改）
INVALID_LEQ = -8.0
NAN_THRESHOLD = 0.5  # NaN 比例阈值，超过此比例的事件将被跳过
# ========================


# ------------------ 去重：统一的线性插值 helper ------------------
def _fill_linear(x: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    对 1D 序列按 valid_mask 做线性插值填补（要求 valid_mask 至少有 1 个 True）
    """
    idx = np.arange(len(x), dtype=np.int64)
    xv = x[valid_mask].astype(np.float32)
    iv = idx[valid_mask]
    return np.interp(idx, iv, xv).astype(np.float32)


def preprocess_and_filter_one_ppg(x: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    输入：一条事件级长 PPG 序列 x（可能含 NaN / 无效值 / 运动伪影 / 异常值）
    输出：滤波后序列（同长），坏点保留 NaN；以及质量统计 qinfo
    """
    x = np.asarray(x, dtype=np.float32)

    # 1) 无效值 -> 删除
    x = x[x > INVALID_LEQ]  # 删除无效值（即小于 INVALID_LEQ 的值）

    # 如果删除后长度为 0，返回空数据
    if len(x) == 0:
        return x, {"nan_frac": 1.0, "motion_frac": 0.0, "outlier_frac": 0.0, "valid_len": 0}

    # 2) 为了做伪影/异常检测：先填补剩余的 NaN
    mask_nan = ~np.isfinite(x)
    valid0 = ~mask_nan
    x_fill0 = _fill_linear(x, valid0)

    # 3) 伪影/异常检测（复用 artifact.py）
    motion_mask = detect_motion_artifacts(x_fill0, threshold_factor=MOTION_THRESHOLD_FACTOR)
    outlier_mask = detect_outliers(x_fill0, window_size=OUTLIER_WINDOW, threshold=OUTLIER_THRESHOLD)

    # 4) 合并坏点：NaN + motion + outlier
    mask_bad = mask_nan | motion_mask | outlier_mask
    if np.all(mask_bad):
        x_all_nan = np.full_like(x, np.nan, dtype=np.float32)
        return x_all_nan, {
            "nan_frac": 1.0,
            "motion_frac": float(np.mean(motion_mask)) if len(x) else 0.0,
            "outlier_frac": float(np.mean(outlier_mask)) if len(x) else 0.0,
            "valid_len": 0
        }

    # 5) 用“好点”再插值一次，用于滤波
    good = ~mask_bad
    x_fill = _fill_linear(x, good)

    # 6) 带通滤波（复用 filtering.py）
    x_filt = apply_filters(
        x_fill,
        fs=FS,
        lowpass_hz=LOWPASS_HZ,
        highpass_hz=HIGHPASS_HZ,
        filter_order=FILTER_ORDER
    ).astype(np.float32)

    # 7) 恢复坏点为 NaN（保持缺失/伪影信息）
    x_filt[mask_bad] = np.nan

    qinfo = {
        "nan_frac": float(np.mean(~np.isfinite(x_filt))),
        "motion_frac": float(np.mean(motion_mask)) if len(x) else 0.0,
        "outlier_frac": float(np.mean(outlier_mask)) if len(x) else 0.0,
        "valid_len": int(np.sum(np.isfinite(x_filt)))
    }
    return x_filt, qinfo


# ------------------ PPG 特征提取（尽可能多，且对 NaN 鲁棒）------------------
def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x[np.isfinite(x)]


def _spectral_feats(x: np.ndarray, fs: float) -> dict:
    v = _finite(x)
    if len(v) < max(8, int(fs) * 2):
        return {}

    nperseg = min(len(v), max(16, int(fs) * 8))
    freqs, pxx = welch(v, fs=fs, nperseg=nperseg, detrend="constant")
    if len(freqs) == 0 or np.all(~np.isfinite(pxx)):
        return {}

    pxx = np.maximum(pxx, 0.0)
    total_power = float(np.trapz(pxx, freqs)) if len(freqs) > 1 else float(np.sum(pxx))

    def band_power(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        if not np.any(m):
            return 0.0
        return float(np.trapz(pxx[m], freqs[m])) if np.sum(m) > 1 else float(np.sum(pxx[m]))

    bp_004_015 = band_power(0.04, 0.15)
    bp_015_04  = band_power(0.15, 0.40)
    bp_04_10   = band_power(0.40, 1.00)
    bp_10_20   = band_power(1.00, 2.00)  # fs=4 -> Nyquist=2

    peak_freq = float(freqs[int(np.argmax(pxx))])
    centroid = float(np.sum(freqs * pxx) / (np.sum(pxx) + 1e-12))
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * pxx) / (np.sum(pxx) + 1e-12)))

    p_norm = pxx / (np.sum(pxx) + 1e-12)
    spec_entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-12)))

    return {
        "psd_total_power": total_power,
        "psd_peak_freq": peak_freq,
        "psd_centroid": centroid,
        "psd_spread": spread,
        "psd_entropy": spec_entropy,
        "bp_0p04_0p15": bp_004_015,
        "bp_0p15_0p40": bp_015_04,
        "bp_0p40_1p00": bp_04_10,
        "bp_1p00_2p00": bp_10_20,
        "bp_ratio_0p15_0p40_to_0p04_0p15": float(bp_015_04 / (bp_004_015 + 1e-12)),
        "bp_ratio_0p40_1p00_to_0p15_0p40": float(bp_04_10 / (bp_015_04 + 1e-12)),
    }


def _peak_hrv_feats(x: np.ndarray, fs: float) -> dict:
    x = np.asarray(x, dtype=np.float32)
    if np.sum(np.isfinite(x)) < max(8, int(fs) * 3):
        return {}

    # 插值后做峰检测（便于稳定）
    valid = np.isfinite(x)
    x_fill = _fill_linear(x, valid)

    min_dist = max(1, int(fs * 0.3))
    amp = np.percentile(x_fill, 95) - np.percentile(x_fill, 5)
    prominence = max(1e-6, 0.1 * amp)

    peaks, _ = find_peaks(x_fill, distance=min_dist, prominence=prominence)
    if len(peaks) < 2:
        return {"n_peaks": int(len(peaks))}

    ibi = np.diff(peaks) / fs
    hr = 60.0 / (ibi + 1e-12)

    sdnn = float(np.std(ibi, ddof=1)) if len(ibi) > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(np.diff(ibi) ** 2))) if len(ibi) > 2 else 0.0
    nn50 = int(np.sum(np.abs(np.diff(ibi)) > 0.05)) if len(ibi) > 2 else 0
    pnn50 = float(nn50 / max(len(ibi) - 1, 1))

    try:
        widths, _, _, _ = peak_widths(x_fill, peaks, rel_height=0.5)
        widths_s = widths / fs
        pw_mean = float(np.mean(widths_s))
        pw_std = float(np.std(widths_s))
    except Exception:
        pw_mean, pw_std = np.nan, np.nan

    trough_amps = []
    for i in range(len(peaks) - 1):
        seg = x_fill[peaks[i]:peaks[i + 1] + 1]
        if len(seg) == 0:
            continue
        trough_amps.append(float(x_fill[peaks[i]] - np.min(seg)))
    trough_amps = np.asarray(trough_amps, dtype=np.float32)
    amp_mean = float(np.mean(trough_amps)) if len(trough_amps) else np.nan
    amp_std = float(np.std(trough_amps)) if len(trough_amps) else np.nan

    dur_s = len(x_fill) / fs
    return {
        "n_peaks": int(len(peaks)),
        "peak_rate_hz": float(len(peaks) / max(dur_s, 1e-12)),
        "ibi_mean_s": float(np.mean(ibi)),
        "ibi_std_s": float(np.std(ibi)),
        "hr_mean_bpm": float(np.mean(hr)),
        "hr_std_bpm": float(np.std(hr)),
        "hr_min_bpm": float(np.min(hr)),
        "hr_max_bpm": float(np.max(hr)),
        "hrv_sdnn_s": sdnn,
        "hrv_rmssd_s": rmssd,
        "hrv_pnn50": pnn50,
        "pulse_width_mean_s": pw_mean,
        "pulse_width_std_s": pw_std,
        "pulse_amp_mean": amp_mean,
        "pulse_amp_std": amp_std,
    }


def extract_ppg_features(x_filt: np.ndarray, fs: float, qinfo: dict) -> dict:
    feats = {}
    x_filt = np.asarray(x_filt, dtype=np.float32)

    feats["len"] = int(len(x_filt))
    feats["valid_len"] = int(np.sum(np.isfinite(x_filt)))
    feats["valid_frac"] = float(feats["valid_len"] / max(feats["len"], 1))

    # 质量特征
    for k, v in (qinfo or {}).items():
        feats[f"q_{k}"] = v

    v = _finite(x_filt)
    if len(v) == 0:
        return feats

    # 时域统计
    feats.update({
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "var": float(np.var(v)),
        "median": float(np.median(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "range": float(np.max(v) - np.min(v)),
        "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
        "mad": float(stats.median_abs_deviation(v, nan_policy="omit")),
        "skew": float(stats.skew(v, nan_policy="omit")),
        "kurtosis": float(stats.kurtosis(v, nan_policy="omit")),
        "rms": float(np.sqrt(np.mean(v ** 2))),
        "energy": float(np.sum(v ** 2)),
        "mean_abs": float(np.mean(np.abs(v))),
    })

    # 导数特征（插值后）
    valid = np.isfinite(x_filt)
    x_fill = _fill_linear(x_filt, valid)
    d1 = np.diff(x_fill)
    d2 = np.diff(d1) if len(d1) > 1 else np.array([], dtype=np.float32)

    if len(d1) > 0:
        feats.update({
            "d1_mean": float(np.mean(d1)),
            "d1_std": float(np.std(d1)),
            "d1_abs_mean": float(np.mean(np.abs(d1))),
            "d1_abs_max": float(np.max(np.abs(d1))),
            "d1_zero_cross_rate": float(np.mean(np.diff(np.sign(d1)) != 0)) if len(d1) > 2 else 0.0,
        })
    if len(d2) > 0:
        feats.update({
            "d2_mean": float(np.mean(d2)),
            "d2_std": float(np.std(d2)),
            "d2_abs_mean": float(np.mean(np.abs(d2))),
            "d2_abs_max": float(np.max(np.abs(d2))),
        })

    # 频域
    feats.update(_spectral_feats(x_filt, fs))
    # 峰/HRV
    feats.update(_peak_hrv_feats(x_filt, fs))

    return feats


def main():
    print("[INFO] 载入事件级长序列 + 标签 + meta...")
    X_raw_long = np.load(IN_RAW_LONG_NPY, allow_pickle=True)
    y_events   = np.load(IN_LABEL_NPY)
    meta_df    = pd.read_csv(IN_META_CSV)

    assert len(X_raw_long) == len(y_events) == len(meta_df), \
        "raw_long / labels / meta 行数不一致，请检查前一步输出。"

    print(f"[INFO] 事件总数 = {len(X_raw_long)}")
    try:
        print(f"[INFO] 标签分布: {np.bincount(y_events.astype(int))}")
    except Exception:
        print("[INFO] 标签分布: (无法 bincount，labels 可能不是从 0 开始的 int)")

    X_filt_list = []
    skipped_all_nan = 0
    feat_rows = []

    printed_samples = 0
    max_print_samples = 3

    for i, x in enumerate(X_raw_long):
        eid = f"{meta_df.iloc[i]['pid']}_{meta_df.iloc[i]['event_idx']}"
        qinfo = {}

        if (i + 1) % 50 == 0 or i == 0:
            print(f"[INFO] 处理事件 {i+1}/{len(X_raw_long)} -> {eid}")

        x = np.asarray(x, dtype=np.float32)
        x_filt, qinfo = preprocess_and_filter_one_ppg(x)

        # 跳过 NaN 比例过高的事件
        if qinfo["nan_frac"] > NAN_THRESHOLD:
            skipped_all_nan += 1
            continue

        X_filt_list.append(x_filt)

        if EXTRACT_FEATURES:
            feats = extract_ppg_features(x_filt, fs=FS, qinfo=qinfo)
            feats["pid"] = meta_df.iloc[i]["pid"]
            feats["event_idx"] = meta_df.iloc[i]["event_idx"]
            feats["event_id"] = eid
            feats["label"] = int(y_events[i]) if np.issubdtype(np.asarray(y_events).dtype, np.integer) else y_events[i]
            feat_rows.append(feats)

        if printed_samples < max_print_samples:
            label_i = int(y_events[i]) if np.issubdtype(np.asarray(y_events).dtype, np.integer) else y_events[i]
            print("\n========== 样例事件 {} ==========".format(i))
            print(f"事件ID: {eid}, label={label_i}, 长度={len(x)}")
            print("原始序列前 20 点（含 NaN）：")
            print(x[:20])
            print("滤波后序列前 20 点（NaN 保留）：")
            print(x_filt[:20])
            print("qinfo：", qinfo)
            printed_samples += 1

    X_filt_obj = np.array(X_filt_list, dtype=object)
    lens = [len(xx) for xx in X_filt_obj]

    print("\n========== 滤波后事件级长序列 统计 ==========")
    print(f"事件总数: {len(X_filt_obj)}")
    print(f"长度: min={int(np.min(lens))}, max={int(np.max(lens))}, mean={np.mean(lens):.2f}")
    print(f"[INFO] 全NaN事件数（滤波后）: {skipped_all_nan}")

    # 保存滤波后的数据
    np.save(OUT_RAW_LONG_FILT_NPY, X_filt_obj)
    print(f"\n[INFO] 已保存滤波后的长序列: {OUT_RAW_LONG_FILT_NPY} (object 数组, len={len(X_filt_obj)})")
    print("[INFO] 标签和 meta 仍使用原来的文件，不需要额外改动。")

    # 保存特征数据
    if EXTRACT_FEATURES:
        feat_df = pd.DataFrame(feat_rows)
        feat_df.to_csv(OUT_FEATURE_CSV, index=False)
    print(f"[INFO] 已保存 PPG 特征: {OUT_FEATURE_CSV} (rows={len(feat_df)}, cols={feat_df.shape[1]})")


if __name__ == "__main__":
    main()

