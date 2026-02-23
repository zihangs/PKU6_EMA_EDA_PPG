# PPG 数据处理流程说明

本文档描述 PPG 代码从原始数据预处理到信号特征提取的完整流程，以及对应的函数说明。

---

## 一、整体流程概览

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| script1_build_event_long_ppg.py | 原始数据构建 | CSV 文件 | ppg_event_raw_long.npy, labels, meta |
| script2_filter_event_long.py | 滤波与预处理 | ppg_event_raw_long.npy | ppg_event_raw_long_filt.npy |
| script3_prepare_data_raw.py | 滑窗与特征提取 | ppg_event_raw_long_filt.npy | ppg_window_features_adv.npy, labels |

```
CSV 原始 PPG 字符串
    → parse_ppg_content / pick_ppg_col
ppg_event_raw_long.npy (原始长序列)
    → preprocess_and_filter_one_ppg (滤波 + 伪影/异常检测)
ppg_event_raw_long_filt.npy (滤波后)
    → 滑窗 + extract_features_advanced
ppg_window_features_adv.npy (55 维窗口特征)
```

---

## 二、Script 1：原始数据构建

**文件**：`script1_build_event_long_ppg.py`

### 步骤与对应函数

| 步骤 | 说明 | 对应函数 |
|------|------|----------|
| 1.1 | 读取 PPG 目录下的 CSV 文件 | `main()` 中 `glob.glob()` + `process_file()` |
| 1.2 | 选择 PPG 列 | `pick_ppg_col(df)` |
| 1.3 | 解析 PPG 内容（`"[v1, v2...]"` 格式） | `parse_ppg_content(series)` |
| 1.4 | 标签映射（0–1→0，5–9→1，2–4 跳过） | `process_file()` 内逻辑 |
| 1.5 | 按 event_idx 分组并排序 | `process_file()` 中 `groupby('event_idx')` |
| 1.6 | 长度过滤（至少 10 分钟） | `process_file()` 中 `len(ppg_seq) < FS * 600` |

### 输出文件

- `ppg_event_raw_long.npy`：原始 PPG 长序列（object 数组）
- `ppg_event_labels.npy`：事件标签
- `ppg_event_meta_long.csv`：事件元数据

---

## 三、Script 2：滤波与预处理

**文件**：`script2_filter_event_long.py`

### 步骤与对应函数

| 步骤 | 说明 | 对应函数 |
|------|------|----------|
| 2.1 | 移除无效值（≤ -8） | `preprocess_and_filter_one_ppg()` |
| 2.2 | 线性插值填补 NaN | `_fill_linear(x, valid_mask)` |
| 2.3 | 全局伪影检测 | `detect_motion_artifacts()` |
| 2.4 | 局部异常检测（滑动窗口 50 点） | `detect_outliers()` |
| 2.5 | 对异常点二次填补 | `_fill_linear(x, ~mask_bad)` |
| 2.6 | 带通滤波（0.5–5 Hz, Butterworth 4 阶） | `apply_filters()` |

### 参数配置

- 采样率 `FS`：25 Hz
- 带通：0.5–5 Hz
- 滤波阶数：4
- 伪影检测阈值：3.5 × 标准差
- 局部异常窗口：50 点，阈值 3.0

### 输出文件

- `ppg_event_raw_long_filt.npy`：滤波后 PPG 序列

---

## 四、Script 3：滑窗与特征提取

**文件**：`script3_prepare_data_raw.py`

### 步骤与对应函数

| 步骤 | 说明 | 对应函数 |
|------|------|----------|
| 3.1 | 去除 NaN 点 | `main()` 中 `ppg[~np.isnan(ppg)]` |
| 3.2 | 按 event 内 Z-score 标准化 | `main()` 中 `ppg_z = (ppg - mu) / sd` |
| 3.3 | 滑窗切分（窗长 60 s，步长 30 s） | `main()` 中 `range(0, len - win_points + 1, step_points)` |
| 3.4 | 单窗 55 维特征提取 | `extract_features_advanced(seg, fs)` |

### 特征详情（共 55 维）

| 类别 | 维度 | 具体内容 |
|------|------|----------|
| 基础统计 | 8 | mean, std, min, max, ptp, rms, skew, kurtosis |
| 差分 | 6 | 一阶/二阶差分的 mean_abs, std, max_abs |
| 频域 | 10 | total/VLF/LF/HF 功率、LF/HF 比、主频、熵等 |
| 峰值/HRV | 11 | 峰数、HR、IBI 统计、RMSSD、pNN50、峰幅/宽度等 |
| Hjorth 参数 | 3 | activity, mobility, complexity |
| Poincaré HRV | 3 | SD1, SD2, SD1/SD2 |
| 形态学 | 5 | AUC、能量、过零率、直方图熵、峰宽 std |
| 高阶矩 | 9 | 三阶矩、四阶矩及补位 |

### 辅助函数

- `hjorth_params(x)`：Hjorth 参数
- `poincare_sd1_sd2(ibi)`：Poincaré SD1/SD2
- `safe_div(a, b)`：除零保护

### 输出文件

- `ppg_window_features_adv.npy`：窗口特征（形状约 `(N, T, 55)`）
- `ppg_labels_used_adv.npy`：对应标签

---

## 五、运行顺序

```bash
python script1_build_event_long_ppg.py   # 1. 构建原始事件
python script2_filter_event_long.py     # 2. 滤波预处理
python script3_prepare_data_raw.py      # 3. 提取窗口特征
```

---

## 六、环境变量（可选）

- `PPG_DIR`：PPG 原始 CSV 目录
- `OUT_DIR`：输出目录（event_long 等）
