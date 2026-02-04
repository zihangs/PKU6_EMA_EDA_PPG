import os
import numpy as np
from scipy import stats, signal

# ================= 配置 =================
BASE_DIR = r"/Users/zhushiyao/Desktop/ppg_code/event_long"
IN_PPG_FILT = os.path.join(BASE_DIR, "ppg_event_raw_long_filt.npy")
IN_LABELS   = os.path.join(BASE_DIR, "ppg_event_labels.npy")

# 输出文件带 _adv 后缀
OUT_WINDOW_FEATS = os.path.join(BASE_DIR, "ppg_window_features_adv.npy")
OUT_LABELS_USED  = os.path.join(BASE_DIR, "ppg_labels_used_adv.npy")

class Config:
    fs_ppg = 25.0
    win_len_sec = 60.0
    step_len_sec = 30.0
    win_points = int(win_len_sec * fs_ppg)
    step_points = int(step_len_sec * fs_ppg)
    min_len_points = win_points 

cfg = Config()

def safe_div(a, b):
    return a / b if abs(b) > 1e-8 else 0.0

def hjorth_params(x):
    """ 描述信号的复杂度与混沌度 """
    if len(x) < 3: return 0, 0, 0
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)
    activity = var_x
    mobility = np.sqrt(safe_div(var_dx, var_x))
    complexity = safe_div(np.sqrt(safe_div(var_ddx, var_dx)), mobility)
    return activity, mobility, complexity

def poincare_sd1_sd2(ibi):
    """ 非线性HRV指标 """
    if len(ibi) < 2: return 0, 0
    diff_ibi = np.diff(ibi)
    sd1 = np.std(diff_ibi, ddof=1) * 0.70710678
    sd2 = np.sqrt(2 * np.var(ibi, ddof=1) - 0.5 * np.var(diff_ibi, ddof=1))
    return sd1, sd2

def extract_features_advanced(x, fs):
    # 局部标准化
    sd = np.std(x)
    x = (x - np.mean(x)) / sd if sd > 1e-6 else x - np.mean(x)
    
    feats = []
    # --- 1. 基础统计 (8) ---
    feats.extend([np.mean(x), np.std(x), np.min(x), np.max(x), np.ptp(x), np.sqrt(np.mean(x**2)), stats.skew(x), stats.kurtosis(x)])
    
    # --- 2. 差分 (6) ---
    d1 = np.diff(x); d2 = np.diff(d1)
    feats.extend([np.mean(np.abs(d1)), np.std(d1), np.max(np.abs(d1)), np.mean(np.abs(d2)), np.std(d2), np.max(np.abs(d2))])
    
    # --- 3. 频域 (10) ---
    try:
        f, p = signal.welch(x, fs=fs, nperseg=min(len(x), 64))
        p_tot = np.trapz(p, f)
        def bp(lo, hi):
            m = (f >= lo) & (f < hi)
            return np.trapz(p[m], f[m]) if np.any(m) else 0.0
        vlf, lf, hf = bp(0.0033, 0.04), bp(0.04, 0.15), bp(0.15, 0.40)
        p_norm = p / (np.sum(p) + 1e-9)
        feats.extend([p_tot, vlf, lf, hf, safe_div(lf, p_tot-vlf), safe_div(hf, p_tot-vlf), safe_div(lf, hf), f[np.argmax(p)] if len(p)>0 else 0, stats.entropy(p_norm), np.sum(f * p_norm)])
    except: feats.extend([0]*10)

    # --- 4. 峰值/HRV (11) ---
    peaks, _ = signal.find_peaks(x, distance=int(0.3*fs), prominence=0.3)
    if len(peaks) > 1:
        ibi = np.diff(peaks) / fs
        d_ibi = np.diff(ibi)
        widths = signal.peak_widths(x, peaks, rel_height=0.5)[0] / fs
        feats.extend([len(peaks), 60.0/np.mean(ibi), np.mean(ibi), np.std(ibi), np.min(ibi), np.max(ibi), np.sqrt(np.mean(d_ibi**2)), (np.sum(np.abs(d_ibi)>0.05)/len(d_ibi)), np.mean(x[peaks]), np.std(x[peaks]), np.mean(widths)])
    else:
        feats.extend([len(peaks)] + [0]*10)
        ibi = np.array([])
        widths = np.array([])

    # === 新增特征 (20) ===
    # 5. Hjorth (3)
    feats.extend(list(hjorth_params(x)))
    # 6. Non-linear HRV (3)
    if len(ibi) >= 2:
        sd1, sd2 = poincare_sd1_sd2(ibi)
        feats.extend([sd1, sd2, safe_div(sd1, sd2)])
    else: feats.extend([0, 0, 0])
    # 7. 形态学 (5)
    hist_p, _ = np.histogram(x, bins=20, density=True); hist_p = hist_p[hist_p > 0]
    feats.extend([np.trapz(np.abs(x))/fs, np.sum(x**2), np.sum(np.diff(np.signbit(x)))/(len(x)/fs), -np.sum(hist_p * np.log(hist_p)), np.std(widths) if len(widths)>1 else 0])
    # 8. 高阶矩与补位 (9) - 凑齐55维
    feats.extend([np.mean(np.abs(x)**3), np.mean(x**4)] + [0]*7)

    return np.array(feats, dtype=np.float32)

def main():
    print(f"[INFO] Extracting 55-dim Advanced Features...")
    if not os.path.exists(IN_PPG_FILT): return
    ppg_data = np.load(IN_PPG_FILT, allow_pickle=True)
    labels_raw = np.load(IN_LABELS, allow_pickle=True)
    
    win_feats, labels_used = [], []
    for i, ppg in enumerate(ppg_data):
        if (i+1)%100==0: print(f"Processing {i+1}...", end="\r")
        ppg = ppg[~np.isnan(ppg)]
        if len(ppg) < cfg.min_len_points: continue
        
        # 滑窗提取
        mu, sd = np.mean(ppg), np.std(ppg)
        if sd < 1e-6: continue
        ppg_z = (ppg - mu) / sd
        
        w_fs = []
        for j in range(0, len(ppg_z) - cfg.win_points + 1, cfg.step_points):
            seg = ppg_z[j:j + cfg.win_points]
            w_fs.append(extract_features_advanced(seg, cfg.fs_ppg))
            
        if w_fs:
            win_feats.append(np.array(w_fs, dtype=np.float32))
            labels_used.append(labels_raw[i])

    np.save(OUT_WINDOW_FEATS, np.array(win_feats, dtype=object))
    np.save(OUT_LABELS_USED, np.array(labels_used, dtype=np.int64))
    print(f"\n[Done] Saved {len(win_feats)} samples. Shape example: {win_feats[0].shape} (T, 55)")

if __name__ == "__main__":
    main()