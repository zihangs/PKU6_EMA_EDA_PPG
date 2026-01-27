# script1_build_event_long_ppg.py
#
# 说明：
# 这是对 script1_build_event_long.py 的“EDA->PPG”对应改写版，
# 结构/输出形式保持一致：按每个 pid 的 CSV，按 event_idx 分组生成“事件级长序列”，输出为 npy(object) + label.npy + meta.csv。
# 与 EDA 版的主要差异仅在于：
#   - 输入目录变量：EDA_DIR -> PPG_DIR
#   - 信号列猜测：EDA_COL_GUESS -> PPG_COL_GUESS
#   - 信号解析：支持 ppg 列为数值或“列表字符串”
#   - 输出文件名前缀：ppg_event_*
#
# 其余（raw_score -> label 的映射规则、event_idx 分组逻辑、输出格式）与原脚本保持一致。

import os
import glob
import ast
import numpy as np
import pandas as pd

# ========= 可配置 =========
PPG_DIR = r"/Users/zhushiyao/Desktop/train_set_103_filtered_20_ppg"
OUT_DIR = r"/Users/zhushiyao/Desktop/event_long_ppg"

# 用于 meta 里的 duration_sec 统计，不影响序列内容
# 如果你有真实采样率，请改成真实值
FS = 25.0  # Hz

TIME_COL = None  # 默认优先 event_dt / timestamp，否则取第1列
PPG_COL_GUESS = ["ppg", "PPG",  "PPG_Raw", "bvp_raw"]  # 猜测 PPG 列
INVALID_MIN = -90000.0  # 典型缺失哨兵值（如 -99999）会被置 NaN
INVALID_MAX =  90000.0
# ========================

os.makedirs(OUT_DIR, exist_ok=True)

RAW_LONG_NPY = os.path.join(OUT_DIR, "ppg_event_raw_long.npy")
LABEL_NPY    = os.path.join(OUT_DIR, "ppg_event_labels.npy")
META_CSV     = os.path.join(OUT_DIR, "ppg_event_meta_long.csv")


def pick_time_col(df: pd.DataFrame) -> str:
    if TIME_COL is not None and TIME_COL in df.columns:
        return TIME_COL
    for c in ["event_dt", "timestamp", "time", "datetime"]:
        if c in df.columns:
            return c
    return df.columns[0]


def _normalize_colname(c: str) -> str:
    return str(c).strip().lower()


def ppg_is_list_string(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(10).tolist()
    return any(s.strip().startswith("[") and s.strip().endswith("]") for s in sample)


def pick_ppg_col(df: pd.DataFrame) -> str:
    # 1) 直接命中（忽略大小写/空格）
    norm_map = {_normalize_colname(c): c for c in df.columns}
    for c in PPG_COL_GUESS:
        key = _normalize_colname(c)
        if key in norm_map:
            return norm_map[key]

    # 2) 在“非元信息列”中找最像信号的列（object 且包含列表字符串）
    meta_like = {"q14_label", "event_dt", "event_idx", "raw_score"}
    for c in df.columns:
        if _normalize_colname(c) in meta_like:
            continue
        if df[c].dtype == object and ppg_is_list_string(df[c]):
            return c

    # 3) 兜底：第一列
    return df.columns[0]


def flatten_ppg_for_event(sub: pd.DataFrame,
                          time_col: str,
                          ppg_col: str) -> np.ndarray:
    """
    对一个事件（某个pid的某个event_idx子表）展平：
      - 如果 ppg 列是列表字符串，则按行顺序把每行数组拼接成连续序列
      - 如果是单值，则按行顺序直接返回
    """
    if not ppg_is_list_string(sub[ppg_col]):
        ppg_flat = pd.to_numeric(sub[ppg_col], errors="coerce").to_numpy()
        return ppg_flat.astype(np.float32, copy=False)

    ppg_list = []
    for _, row in sub.iterrows():
        txt = str(row[ppg_col]).strip()
        try:
            arr = ast.literal_eval(txt)
        except Exception:
            # 容错：去括号并按逗号/分号切
            txt2 = txt.strip("[]").replace(";", ",")
            parts = [t for t in txt2.split(",") if t.strip() != ""]
            try:
                arr = [float(t) for t in parts]
            except Exception:
                arr = []
        if isinstance(arr, (list, tuple, np.ndarray)) and len(arr) > 0:
            ppg_list.extend(arr)

    return np.asarray(ppg_list, dtype=np.float32)


def map_score_to_label(score):
    """与 EDA 版保持一致：raw_score 映射二分类标签
      - 0~1 -> 0
      - 5~9 -> 1
      - 其他 -> None（跳过）
    """
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
        f for f in glob.glob(os.path.join(PPG_DIR, "*.csv"))
        if os.path.basename(f).lower().endswith(".csv")
    )
    print(f"[INFO] 找到患者级 PPG 文件: {len(csv_files)} 个")

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
            print(f"  ⚠️ {base} 缺少 event_idx/raw_score，跳过")
            continue

        time_col = pick_time_col(df)
        ppg_col = pick_ppg_col(df)

        # 按事件分组
        for eidx, sub in df.groupby("event_idx"):
            sub = sub.copy()

            raw_score = pd.to_numeric(sub["raw_score"].iloc[0], errors="coerce")
            label = map_score_to_label(raw_score)
            if label is None:
                skipped_score += 1
                continue

            # 拉长 PPG 序列
            ppg_flat = flatten_ppg_for_event(sub, time_col, ppg_col)

            # 清洗：极端值置 NaN（例如 -99999）
            if ppg_flat.size > 0:
                bad = (ppg_flat <= INVALID_MIN) | (ppg_flat >= INVALID_MAX) | ~np.isfinite(ppg_flat)
                if bad.any():
                    ppg_flat = ppg_flat.copy()
                    ppg_flat[bad] = np.nan

            if ppg_flat.size == 0 or np.all(~np.isfinite(ppg_flat)):
                skipped_empty += 1
                continue

            all_raw.append(ppg_flat)
            all_labels.append(int(label))
            total_events += 1

            meta_rows.append({
                "pid": pid,
                "event_idx": int(eidx),
                "raw_score": float(raw_score) if pd.notna(raw_score) else None,
                "label": int(label),
                "n_points": int(len(ppg_flat)),
                "duration_sec": float(len(ppg_flat) / FS) if FS else None,
                "ppg_col": str(ppg_col),
            })

    if total_events == 0:
        print("[ERROR] 没有生成任何事件级长序列，请检查输入数据。")
        return

    X_raw_obj = np.array(all_raw, dtype=object)
    y_events = np.array(all_labels, dtype=np.int64)
    meta_df = pd.DataFrame(meta_rows).reset_index(drop=True)

    print("\n========== 事件级长 PPG 序列 统计 ==========")
    print(f"有效事件数: {len(X_raw_obj)}")
    print(f"标签分布: {np.bincount(y_events) if y_events.size else y_events}")
    lens = [len(x) for x in X_raw_obj]
    print(f"每个事件序列长度: min={int(np.min(lens))}, max={int(np.max(lens))}, mean={np.mean(lens):.2f}")
    print(f"[INFO] 跳过事件: 分数越界/无效={skipped_score}, 空序列/全NaN={skipped_empty}")

    # 保存（与 EDA 版一致）
    np.save(RAW_LONG_NPY, X_raw_obj)
    np.save(LABEL_NPY, y_events)
    meta_df.to_csv(META_CSV, index=False, encoding="utf-8-sig")

    print("\n[INFO] 已保存：")
    print(f"  事件级长序列: {RAW_LONG_NPY}   (object 数组, len={len(X_raw_obj)})")
    print(f"  事件级标签:   {LABEL_NPY}      (shape={y_events.shape})")
    print(f"  事件元信息:   {META_CSV}")


if __name__ == "__main__":
    main()
