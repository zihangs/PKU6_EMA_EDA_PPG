# coding: utf-8
"""
script2_filter_event_long.py (Fixed & Robust)

修复说明：
- 修复了 KeyError: 'event_idx' 问题。
- 增加了对 meta_csv 列名的自动兼容（支持 event_idx 或 event_start_idx）。
- 保持所有滤波和清洗逻辑不变。
"""

import os
import numpy as np
import pandas as pd
from scipy import stats, signal

# ================= 配置参数 =================
# 路径配置
BASE_DIR = os.environ.get("OUT_DIR", r"/Users/zhushiyao/Desktop/ppg_code/event_long")

IN_RAW_LONG_NPY = os.path.join(BASE_DIR, "ppg_event_raw_long.npy")
IN_LABEL_NPY    = os.path.join(BASE_DIR, "ppg_event_labels.npy")
IN_META_CSV     = os.path.join(BASE_DIR, "ppg_event_meta_long.csv")

OUT_RAW_LONG_FILT_NPY = os.path.join(BASE_DIR, "ppg_event_raw_long_filt.npy")

# 信号处理参数
FS = 25.0
LOWPASS_HZ = 5.0
HIGHPASS_HZ = 0.5
FILTER_ORDER = 4

# 伪影/异常检测参数
MOTION_THRESHOLD_FACTOR = 3.5
OUTLIER_WINDOW = 50
OUTLIER_THRESHOLD = 3.0
INVALID_LEQ = -8.0
NAN_THRESHOLD = 0.6

np.set_printoptions(precision=4, suppress=True)


# ================= 内置核心功能 =================

def apply_filters(data, fs, lowpass_hz, highpass_hz, filter_order=4):
    """内置带通滤波器"""
    nyq = 0.5 * fs
    low = highpass_hz / nyq
    high = lowpass_hz / nyq
    if high >= 1.0: high = 0.99
    if low <= 0.0: low = 0.001
    b, a = signal.butter(filter_order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data, padtype='odd', padlen=min(len(data)//2, 50))
    return y

def detect_motion_artifacts(x, threshold_factor=3.5):
    """全局伪影检测"""
    if len(x) == 0: return np.array([], dtype=bool)
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-6: return np.zeros_like(x, dtype=bool)
    z_scores = np.abs((x - mean) / std)
    return z_scores > threshold_factor

def detect_outliers(x, window_size=50, threshold=3.0):
    """局部异常检测"""
    if len(x) < window_size:
        return np.zeros_like(x, dtype=bool)
    s = pd.Series(x)
    rolling_mean = s.rolling(window=window_size, center=True, min_periods=1).mean()
    rolling_std = s.rolling(window=window_size, center=True, min_periods=1).std()
    rolling_mean = rolling_mean.fillna(method='bfill').fillna(method='ffill')
    rolling_std = rolling_std.fillna(method='bfill').fillna(method='ffill')
    rolling_std[rolling_std < 1e-6] = 1e-6
    z_local = np.abs((s - rolling_mean) / rolling_std)
    return z_local.values > threshold

def _fill_linear(x: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """线性插值"""
    if len(x) == 0: return x
    if np.all(valid_mask): return x
    if not np.any(valid_mask): return x
    idx = np.arange(len(x))
    xv = x[valid_mask]
    iv = idx[valid_mask]
    return np.interp(idx, iv, xv).astype(np.float32)

def preprocess_and_filter_one_ppg(x: np.ndarray) -> tuple[np.ndarray, dict]:
    """处理单个事件序列"""
    x = np.asarray(x, dtype=np.float32)
    # 1. 移除无效值
    x = x[x > INVALID_LEQ]
    if len(x) < FS:
        return x, {"nan_frac": 1.0, "valid_len": 0, "bad_frac": 1.0}

    # 2. 初始填补
    mask_nan = ~np.isfinite(x)
    x_fill0 = _fill_linear(x, ~mask_nan)

    # 3. 检测
    motion_mask = detect_motion_artifacts(x_fill0, threshold_factor=MOTION_THRESHOLD_FACTOR)
    outlier_mask = detect_outliers(x_fill0, window_size=OUTLIER_WINDOW, threshold=OUTLIER_THRESHOLD)
    mask_bad = mask_nan | motion_mask | outlier_mask
    
    if np.all(mask_bad):
        return np.full_like(x, np.nan), {"nan_frac": 1.0, "valid_len": 0, "bad_frac": 1.0}

    # 4. 二次填补并滤波
    x_clean_input = _fill_linear(x, ~mask_bad)
    try:
        x_filt = apply_filters(x_clean_input, FS, LOWPASS_HZ, HIGHPASS_HZ, FILTER_ORDER)
    except Exception:
        x_filt = x_clean_input
    
    qinfo = {
        "nan_frac": float(np.mean(mask_nan)),
        "bad_frac": float(np.mean(mask_bad)),
        "valid_len": int(len(x) - np.sum(mask_bad))
    }
    return x_filt.astype(np.float32), qinfo


# ================= 主流程 =================

def main():
    print(f"[INFO] 正在读取: {IN_RAW_LONG_NPY}")
    if not os.path.exists(IN_RAW_LONG_NPY):
        print(f"[ERROR] 文件未找到。请先运行 script1 生成数据。")
        return

    X_raw = np.load(IN_RAW_LONG_NPY, allow_pickle=True)
    
    # 智能读取 Meta
    meta_df = None
    if os.path.exists(IN_META_CSV):
        try:
            meta_df = pd.read_csv(IN_META_CSV)
            print(f"[INFO] 成功读取 Meta CSV: {len(meta_df)} rows")
        except Exception as e:
            print(f"[WARN] 读取 Meta CSV 失败: {e}")

    print(f"[INFO] 事件总数 = {len(X_raw)}")

    X_filt_list = []
    skipped_all_nan = 0
    printed_samples = 0
    max_print_samples = 3

    for i, x in enumerate(X_raw):
        # --- 修复核心：更健壮的 ID 生成逻辑 ---
        pid = "unknown"
        eidx = str(i)
        
        if meta_df is not None and i < len(meta_df):
            row = meta_df.iloc[i]
            # 获取 PID
            if 'pid' in row:
                pid = str(row['pid'])
            
            # 获取 Event Index (兼容多种列名)
            if 'event_idx' in row:
                eidx = str(row['event_idx'])
            elif 'event_start_idx' in row: # 兼容标准版 script1
                eidx = str(row['event_start_idx'])
            elif 'event_id' in row:
                eidx = str(row['event_id'])
        
        eid = f"{pid}_{eidx}"
        # -----------------------------------

        if (i + 1) % 100 == 0:
            print(f"[INFO] 处理事件 {i+1}/{len(X_raw)}...", end='\r')

        x = np.asarray(x, dtype=np.float32)
        x_filt, qinfo = preprocess_and_filter_one_ppg(x)

        # 跳过坏数据
        if qinfo["nan_frac"] > NAN_THRESHOLD:
            skipped_all_nan += 1
            X_filt_list.append(x_filt) 
            continue

        X_filt_list.append(x_filt)

        if printed_samples < max_print_samples:
            print(f"\n========== 样例事件 {i} ({eid}) ==========")
            print(f"长度={len(x)}, 坏点比例={qinfo['bad_frac']:.2f}")
            print("原始前10:", x[:10])
            print("滤波前10:", x_filt[:10])
            printed_samples += 1

    print("\n\n========== 处理完成 ==========")
    X_filt_obj = np.array(X_filt_list, dtype=object)
    lens = [len(xx) for xx in X_filt_obj]
    
    print(f"输出形状: {X_filt_obj.shape}")
    print(f"长度统计: Min={np.min(lens):.0f}, Max={np.max(lens):.0f}, Mean={np.mean(lens):.1f}")
    print(f"高缺失率事件(nan>{NAN_THRESHOLD}): {skipped_all_nan}")

    np.save(OUT_RAW_LONG_FILT_NPY, X_filt_obj)
    print(f"[INFO] 已保存滤波后数据: {OUT_RAW_LONG_FILT_NPY}")
    print("[INFO] 下一步 -> 运行 script3 提取特征。")

if __name__ == "__main__":
    main()