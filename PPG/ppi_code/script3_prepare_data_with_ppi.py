# coding: utf-8
"""
从已有的 event_long 数据生成“PPI 版”训练数据（与 labels 对齐）

输入（已存在）：
- event_long/ppg_event_raw_long_filt.npy      # 每个事件一个 1D PPG 序列（25Hz 左右）
- event_long/ppg_event_labels_used.npy        # 每个事件的标签

输出（新生成）：
- event_long/ppi_event_windows.npy            # 每个事件: [n_windows, WIN_LEN]
- event_long/ppi_event_window_features.npy    # 每个事件: [n_windows, feat_dim]

注意：
- 不依赖 ppg_pipeline，也不导入 script3_window_and_extract_features.py（避免其 import-time 执行）。
- “先预处理 PPG -> 转 PPI -> 后续处理”：
  这里假设 ppg_event_raw_long_filt.npy 已经是滤波后的 PPG（你已有该文件），我们再做峰值->PPI。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

try:
    from scipy.signal import find_peaks, welch
except Exception:  # pragma: no cover
    find_peaks = None
    welch = None


BASE_DIR = "/Users/zhushiyao/Desktop/ppi/ppg_code"

IN_PPG_FILT = os.path.join(BASE_DIR, "event_long", "ppg_event_raw_long_filt.npy")
IN_LABELS = os.path.join(BASE_DIR, "event_long", "ppg_event_labels_used.npy")

OUT_PPI_WINDOWS = os.path.join(BASE_DIR, "event_long", "ppi_event_windows.npy")
OUT_PPI_FEATS = os.path.join(BASE_DIR, "event_long", "ppi_event_window_features.npy")

# 你的原工程里常用的 25Hz 配置
FS_PPG = 25.0
FS_PPI_TS = 25.0
WINDOW_SEC = 30
STEP_SEC = 20
WIN_LEN = int(FS_PPI_TS * WINDOW_SEC)  # 750
STEP_LEN = int(FS_PPI_TS * STEP_SEC)  # 500
MIN_WINDOW_LEN = int(FS_PPI_TS * 15)  # 375


@dataclass
class PPIConfig:
    fs_ppg: float = FS_PPG
    fs_ppi_ts: float = FS_PPI_TS
    win_len: int = WIN_LEN
    step_len: int = STEP_LEN
    min_window_len: int = MIN_WINDOW_LEN
    # 峰检测参数
    hr_max: float = 200.0
    prom_k: float = 1.5  # prominence = k * MAD
    # PPI 合理范围（ms）
    ppi_min_ms: float = 300.0
    ppi_max_ms: float = 2000.0


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def detect_peaks_ppg(sig: np.ndarray, fs: float, cfg: PPIConfig) -> np.ndarray:
    """
    峰值检测：
    - 优先用 scipy.find_peaks
    - 若 scipy 不可用则用简化的局部极大（可能差一些）
    """
    x = np.asarray(sig, dtype=np.float32).ravel()
    if len(x) < int(fs * 2):
        return np.array([], dtype=int)

    # 处理 NaN
    if np.any(~np.isfinite(x)):
        m = np.isfinite(x)
        if m.sum() < 10:
            return np.array([], dtype=int)
        idx = np.arange(len(x))
        x = np.interp(idx, idx[m], x[m]).astype(np.float32)

    min_dist = max(1, int((60.0 / cfg.hr_max) * fs))
    prom = max(1e-6, cfg.prom_k * _mad(x))

    if find_peaks is not None:
        peaks, _ = find_peaks(x, distance=min_dist, prominence=prom)
        return peaks.astype(int)

    # fallback：局部极大 + 距离筛
    cand = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    if len(cand) == 0:
        return np.array([], dtype=int)
    # 简单按高度排序，贪心保留
    cand = cand[np.argsort(x[cand])[::-1]]
    kept: List[int] = []
    for p in cand:
        if all(abs(p - k) >= min_dist for k in kept):
            kept.append(int(p))
    kept.sort()
    return np.asarray(kept, dtype=int)


def peaks_to_ppi_ms(peaks: np.ndarray, fs: float, cfg: PPIConfig) -> np.ndarray:
    if len(peaks) < 2:
        return np.array([], dtype=np.float32)
    ppi_ms = np.diff(peaks.astype(np.float32)) * (1000.0 / float(fs))
    m = (ppi_ms >= cfg.ppi_min_ms) & (ppi_ms <= cfg.ppi_max_ms)
    return ppi_ms[m].astype(np.float32)


def ppi_to_time_series(ppi_ms: np.ndarray, fs_ts: float, duration_s: float) -> np.ndarray:
    """
    把不等间隔的 PPI 序列映射成等采样率时间序列（piecewise-constant）。
    返回 1D: [T]
    """
    n = int(np.ceil(duration_s * fs_ts))
    if n <= 0:
        return np.array([], dtype=np.float32)
    if len(ppi_ms) == 0:
        return np.zeros((n,), dtype=np.float32)

    ppi_s = ppi_ms.astype(np.float32) / 1000.0
    cum = np.cumsum(ppi_s)
    cum = np.insert(cum, 0, 0.0)
    cum = np.clip(cum, 0.0, float(duration_s))

    t = np.arange(n, dtype=np.float32) / float(fs_ts)
    out = np.zeros((n,), dtype=np.float32)
    for i, ti in enumerate(t):
        idx = int(np.searchsorted(cum, ti, side="right") - 1)
        idx = max(0, min(idx, len(ppi_ms) - 1))
        out[i] = float(ppi_ms[idx])
    return out


def sliding_window_1d(x: np.ndarray, win_len: int, step_len: int, min_window_len: int) -> List[np.ndarray]:
    x = np.asarray(x, dtype=np.float32).ravel()
    windows: List[np.ndarray] = []
    if len(x) == 0:
        return windows
    start = 0
    while start + win_len <= len(x):
        w = x[start : start + win_len]
        if np.isfinite(w).sum() >= min_window_len:
            windows.append(w.astype(np.float32))
        start += step_len
    # 补最后一个窗口（如果能放下）
    if len(x) > win_len:
        last_start = len(x) - win_len
        if last_start > start - step_len:
            w = x[last_start:]
            if np.isfinite(w).sum() >= min_window_len:
                if len(w) < win_len:
                    pad = np.full((win_len,), np.nan, dtype=np.float32)
                    pad[: len(w)] = w
                    windows.append(pad)
                else:
                    windows.append(w.astype(np.float32))
    return windows


def extract_ppi_features_window(ppi_ts: np.ndarray, fs: float) -> np.ndarray:
    """
    对单个 PPI 时间序列窗口提取特征（轻量版，固定维度）。
    输出维度 feat_dim = 20
    """
    x = np.asarray(ppi_ts, dtype=np.float32).ravel()
    x = x[np.isfinite(x)]
    if len(x) < int(fs * 3):
        return np.zeros((20,), dtype=np.float32)

    feats: List[float] = []
    feats += [float(np.mean(x)), float(np.median(x)), float(np.std(x)), float(np.min(x)), float(np.max(x))]
    feats += [float(np.percentile(x, 25)), float(np.percentile(x, 75)), float(np.mean(np.abs(x)))]

    d1 = np.diff(x)
    if len(d1) > 0:
        feats += [float(np.mean(d1)), float(np.std(d1)), float(np.mean(np.abs(d1))), float(np.max(np.abs(d1)))]
    else:
        feats += [0.0, 0.0, 0.0, 0.0]

    feats += [float(np.sum(x**2)), float(np.mean(x**2))]

    # 频域（如果 welch 可用）
    if welch is not None:
        nperseg = min(len(x), 256)
        f, pxx = welch(x, fs=fs, nperseg=nperseg)
        # 简单 3 个带
        def band_power(lo: float, hi: float) -> float:
            m = (f >= lo) & (f <= hi)
            return float(np.trapz(pxx[m], f[m])) if np.any(m) else 0.0

        feats += [band_power(0.01, 0.04), band_power(0.04, 0.15), band_power(0.15, 0.4)]
    else:
        feats += [0.0, 0.0, 0.0]

    # 补齐到 20
    while len(feats) < 20:
        feats.append(0.0)
    return np.asarray(feats[:20], dtype=np.float32)


def main():
    cfg = PPIConfig()

    if not os.path.exists(IN_PPG_FILT):
        raise FileNotFoundError(f"找不到输入: {IN_PPG_FILT}")
    if not os.path.exists(IN_LABELS):
        raise FileNotFoundError(f"找不到标签: {IN_LABELS}")

    X = np.load(IN_PPG_FILT, allow_pickle=True)
    y = np.load(IN_LABELS, allow_pickle=True)
    n = min(len(X), len(y))
    X = X[:n]
    y = y[:n]

    all_windows: List[np.ndarray] = []
    all_feats: List[np.ndarray] = []

    for i in range(n):
        ppg = np.asarray(X[i], dtype=np.float32).ravel()
        if ppg.size == 0 or np.isfinite(ppg).sum() < int(cfg.fs_ppg * 5):
            all_windows.append(np.zeros((0, cfg.win_len), dtype=np.float32))
            all_feats.append(np.zeros((0, 20), dtype=np.float32))
            continue

        peaks = detect_peaks_ppg(ppg, cfg.fs_ppg, cfg)
        ppi_ms = peaks_to_ppi_ms(peaks, cfg.fs_ppg, cfg)
        dur_s = float(len(ppg) / cfg.fs_ppg)
        ppi_ts = ppi_to_time_series(ppi_ms, cfg.fs_ppi_ts, dur_s)

        windows = sliding_window_1d(ppi_ts, cfg.win_len, cfg.step_len, cfg.min_window_len)
        if len(windows) == 0:
            all_windows.append(np.zeros((0, cfg.win_len), dtype=np.float32))
            all_feats.append(np.zeros((0, 20), dtype=np.float32))
            continue

        win_arr = np.stack(windows, axis=0).astype(np.float32)  # [T, WIN_LEN]
        feat_arr = np.stack([extract_ppi_features_window(w, cfg.fs_ppi_ts) for w in windows], axis=0).astype(np.float32)

        all_windows.append(win_arr)
        all_feats.append(feat_arr)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"[{i+1}/{n}] windows={win_arr.shape} feats={feat_arr.shape} label={int(y[i])}")

    os.makedirs(os.path.dirname(OUT_PPI_WINDOWS), exist_ok=True)
    np.save(OUT_PPI_WINDOWS, np.array(all_windows, dtype=object))
    np.save(OUT_PPI_FEATS, np.array(all_feats, dtype=object))

    print("Saved:")
    print(" -", OUT_PPI_WINDOWS)
    print(" -", OUT_PPI_FEATS)
    print("Labels used (existing):")
    print(" -", IN_LABELS)


if __name__ == "__main__":
    main()

