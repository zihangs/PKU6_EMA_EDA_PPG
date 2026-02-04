# coding: utf-8
"""
script1_build_event_long_ppg_eda_style.py

功能：
完全依照 EDA 代码 (script1_build_event_long.py) 的逻辑结构重建 PPG 数据构建脚本。

改进点：
1. 继承 EDA 的稳健性：详细的统计输出、无效分数过滤、元数据对齐。
2. 适配 PPG 数据特性：
   - 解析 CSV 中 "[v1, v2...]" 格式的字符串数据。
   - 执行 0-9 到 0/1 的标签映射（解决 Target out of bounds 问题）。
"""

import os
import glob
import ast
import numpy as np
import pandas as pd

# ======================= Configuration =======================

# 输入目录 (请根据实际情况修改)
PPG_DIR = os.environ.get("PPG_DIR", r"/Users/zhushiyao/Desktop/train_set_103_filtered_20")
# 输出目录
OUT_DIR = os.environ.get("OUT_DIR", r"/Users/zhushiyao/Desktop/ppg_code/event_long")

FS = 25.0                       # 采样率 (Hz)
TIME_COL = None                 # 如果为 None，默认第一列为时间
PPG_COL_GUESS = ["ppg", "PPG", "PPG_Raw", "bvp_raw"]  # 猜测 PPG 列名

# =============================================================

os.makedirs(OUT_DIR, exist_ok=True)

# 输出文件名 (保持与后续 script 兼容)
RAW_LONG_NPY = os.path.join(OUT_DIR, "ppg_event_raw_long.npy")
LABEL_NPY    = os.path.join(OUT_DIR, "ppg_event_labels.npy")
META_CSV     = os.path.join(OUT_DIR, "ppg_event_meta_long.csv")


def pick_ppg_col(df: pd.DataFrame) -> str:
    """根据猜测列表选择 PPG 列 (EDA 逻辑)"""
    for c in PPG_COL_GUESS:
        if c in df.columns:
            return c
    # Fallback: 如果没找到，且列数足够，尝试取特定列 (可选)
    return None

def parse_ppg_content(series):
    """
    解析 PPG 列的内容。
    针对你的数据：CSV 里存的是字符串列表 "[-999.0, 123.0, ...]"
    """
    data_list = []
    for val in series:
        try:
            if isinstance(val, (int, float)):
                data_list.append(float(val))
            elif isinstance(val, str):
                val = val.strip()
                if val.startswith('[') and val.endswith(']'):
                    # 解析列表字符串
                    parsed = ast.literal_eval(val)
                    data_list.extend(parsed)
                else:
                    # 单个数字字符串
                    data_list.append(float(val))
        except:
            continue
    return np.array(data_list, dtype=np.float32)

def process_file(csv_path):
    """
    处理单个文件：分组 -> 清洗 -> 拼接
    """
    try:
        df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
    except Exception as e:
        # 如果 python 引擎也失败，尝试只读取不报错的行（备用方案）
        try:
             print(f"[WARN] Standard read failed for {os.path.basename(csv_path)}, trying aggressive recovery...")
             df = pd.read_csv(csv_path, engine='python', error_bad_lines=False, warn_bad_lines=True)
        except Exception as e2:
            print(f"[WARN] Cannot read {csv_path}: {e2}")
            return [], [], [], {'error': 1}
    except Exception as e:
        print(f"[WARN] Cannot read {csv_path}: {e}")
        return [], [], [], {'error': 1}

    ppg_col = pick_ppg_col(df)
    if ppg_col is None:
        return [], [], [], {'no_col': 1}
    
    # 必须有 event_idx
    if 'event_idx' not in df.columns:
        return [], [], [], {'no_event_idx': 1}

    # 确定标签来源
    label_source = 'label' if 'label' in df.columns else None
    if label_source is None and 'raw_score' in df.columns:
        label_source = 'raw_score'

    pid = os.path.basename(csv_path).split('.')[0]
    grouped = df.groupby('event_idx')

    raw_list = []
    label_list = []
    meta_list = []
    stats = {'valid': 0, 'skipped_score': 0, 'skipped_short': 0}

    for eid, group in grouped:
        # 1. 检查分数/标签有效性 (EDA 逻辑)
        # 你的数据中 -999 代表无效
        raw_score_val = None
        if 'raw_score' in group.columns:
            raw_score_val = group['raw_score'].iloc[0]
            # 过滤无效分数 (根据你的数据，-999 是无效的)
            if raw_score_val == -999 or pd.isna(raw_score_val):
                stats['skipped_score'] += 1
                continue

        # 2. 获取并清洗标签 (PPG 特有逻辑：映射)
        # 获取原始值
        orig_label = -1
        if label_source:
            orig_label = group[label_source].iloc[0]
        
        # 执行映射
        try:
            val_int = int(float(orig_label))
            if val_int in [0, 1]:
                final_label = 0
            elif val_int in [5, 6, 7, 8, 9]:
                final_label = 1
            else:
                # 2,3,4 或其他 -> 跳过
                stats['skipped_score'] += 1
                continue
        except:
            stats['skipped_score'] += 1
            continue

        # 3. 提取并拼接波形
        # 如果有时间列，先排序
        if 'time' in group.columns:
            group = group.sort_values('time')
        
        ppg_seq = parse_ppg_content(group[ppg_col])

        # 4. 长度过滤 (防止极短数据导致后续报错)
        if len(ppg_seq) < FS * 600.0: # 至少10分钟
            stats['skipped_short'] += 1
            continue

        # 5. 存入结果
        raw_list.append(ppg_seq)
        label_list.append(final_label)
        
        meta_list.append({
            "pid": pid,
            "event_idx": int(eid),
            "raw_score": raw_score_val,
            "label": final_label,
            "n_points": len(ppg_seq),
            "duration_sec": len(ppg_seq) / FS
        })
        stats['valid'] += 1

    return raw_list, label_list, meta_list, stats

def main():
    # 1. Setup
    csv_files = glob.glob(os.path.join(PPG_DIR, "*.csv"))
    print(f"[INFO] Found {len(csv_files)} CSV files in {PPG_DIR}")
    
    if len(csv_files) == 0:
        print("[ERROR] No files found. Check PPG_DIR.")
        return

    all_raw = []
    all_labels = []
    all_meta = []
    
    total_stats = {'valid': 0, 'skipped_score': 0, 'skipped_short': 0, 'error': 0}

    # 2. Processing Loop
    for i, fpath in enumerate(csv_files):
        if (i+1) % 10 == 0:
            print(f"Processing {i+1}/{len(csv_files)}...", end='\r')
            
        r, l, m, s = process_file(fpath)
        
        all_raw.extend(r)
        all_labels.extend(l)
        all_meta.extend(m)
        
        for k in s:
            if k in total_stats:
                total_stats[k] += s[k]
            else:
                total_stats[k] = s[k]

    print("\n\n========== Processing Complete ==========")
    if len(all_raw) == 0:
        print("[ERROR] No valid events generated. Check labels (-999?) or data format.")
        return

    # 3. Save Output (Standard Format)
    X_raw_obj = np.array(all_raw, dtype=object)
    y_events  = np.array(all_labels, dtype=np.int64)
    meta_df   = pd.DataFrame(all_meta).reset_index(drop=True)

    print("\n========== Event-level PPG Sequence Statistics (EDA-Style) ==========")
    print(f"Valid events: {len(X_raw_obj)}")
    print(f"Label distribution: {np.bincount(y_events)}")
    
    lens = [len(x) for x in X_raw_obj]
    print(f"Sequence length (points): min={np.min(lens)}, max={np.max(lens)}, mean={np.mean(lens):.1f}")
    print(f"Sequence duration (sec) : min={np.min(lens)/FS:.1f}s, max={np.max(lens)/FS:.1f}s, mean={np.mean(lens)/FS:.1f}s")
    
    print("\n[INFO] Skipped Details:")
    print(f"  - Invalid Score/Label (2-4, -999): {total_stats['skipped_score']}")
    print(f"  - Too Short (<1s): {total_stats['skipped_short']}")
    print(f"  - File Errors: {total_stats.get('error', 0)}")

    # Save
    np.save(RAW_LONG_NPY, X_raw_obj)
    np.save(LABEL_NPY,    y_events)
    meta_df.to_csv(META_CSV, index=False)
    
    print(f"\nFiles saved to {OUT_DIR}:")
    print(f"  - {os.path.basename(RAW_LONG_NPY)}")
    print(f"  - {os.path.basename(LABEL_NPY)}")
    print(f"  - {os.path.basename(META_CSV)}")

if __name__ == "__main__":
    main()