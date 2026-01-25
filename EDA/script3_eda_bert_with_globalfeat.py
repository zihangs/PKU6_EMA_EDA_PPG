# -*- coding: utf-8 -*-
"""
Pipeline (event-level classification from filtered long EDA):
1) Sliding window over each event; each window is a token (raw waveform only)
2) Extract global features over the whole event as a "global token"
3) Feed [global token + window tokens] into a TransformerEncoder (BERT-like)
4) Event-level binary classification with 10-fold StratifiedKFold
   Metrics: ACC / Bal_ACC / F1 / AUC
"""

import os
from typing import List, Tuple
import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from scipy import stats
from scipy.signal import welch, butter, sosfiltfilt, find_peaks
from scipy import integrate
from sklearn.linear_model import LinearRegression

try:
    import antropy as ant
except ImportError:
    ant = None

try:
    import pywt
except ImportError:
    pywt = None


# ===================== Paths & Config =====================

BASE_LONG_DIR = os.environ.get("BASE_LONG_DIR", "./data/event_long")

RAW_LONG_FILT_NPY = os.path.join(BASE_LONG_DIR, "eda_event_raw_long_filt.npy")
LABEL_NPY         = os.path.join(BASE_LONG_DIR, "eda_event_labels.npy")
META_CSV          = os.path.join(BASE_LONG_DIR, "eda_event_meta_long.csv")

FS = 4.0
WINDOW_SEC = 60
STEP_SEC   = 30
MIN_VALID_RATIO = 0.9

WIN_LEN  = int(FS * WINDOW_SEC)  # 60s * 4Hz = 240
STEP_LEN = int(FS * STEP_SEC)    # 30s * 4Hz = 120

MAX_TOKENS  = 60
MAX_SEQ_LEN = MAX_TOKENS + 1  # +1 for the global token

N_SPLITS     = 10
RANDOM_SEED  = 42
BATCH_SIZE   = 64

WEIGHT_DECAY = 1e-4

D_MODEL  = 128
N_HEAD   = 4
N_LAYERS = 3
FFN_DIM  = 256
DROPOUT  = 0.1
LR       = 1e-4
EPOCHS   = 80

EARLY_STOP = True
EARLY_STOP_METRIC = "bal_acc"  # "bal_acc" / "f1" / "acc"
EARLY_STOP_PATIENCE = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ==========================================================
#                   1) Global feature token
# ==========================================================

GLOBAL_FEAT_KEYS = [
    "mean_eda", "median_eda", "sd_eda", "mad_eda",
    "min_eda", "max_eda", "range_eda", "iqr_eda",
    "rms_eda", "kurtosis_eda", "skewness_eda",
    "activity", "mobility", "complexity",
    "mean_diff1", "sd_diff1", "mean_diff2", "sd_diff2",
    "vlf_power", "lf_power", "hf_power",
    "total_power", "lf_hf_ratio",
    "sampen_scale1", "sampen_scale2",
    "higuchi_fd",
    "phasic_mean", "phasic_sd",
    "phasic_peaks", "phasic_auc",
    "scr_count", "scr_mean_amp", "scr_mean_rise",
]


def _create_global_feat_template():
    return {k: np.nan for k in GLOBAL_FEAT_KEYS}


def _higuchi_fd(x, k_max=5):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < k_max + 2:
        return np.nan

    l_values = np.zeros(k_max)
    for k in range(1, k_max + 1):
        Lmk = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            diff = np.abs(np.diff(x[idx]))
            Lm = diff.sum() * (n - 1) / (k * (len(idx) - 1) * k)
            Lmk.append(Lm)
        l_values[k - 1] = np.mean(Lmk) if len(Lmk) > 0 else np.nan

    valid = np.isfinite(l_values) & (l_values > 0)
    if valid.sum() < 2:
        return np.nan

    ks = np.arange(1, k_max + 1)[valid]
    Ls = l_values[valid]
    x_log = np.log(ks).reshape(-1, 1)
    y_log = np.log(Ls)
    model = LinearRegression()
    model.fit(x_log, y_log)
    return model.coef_[0]


def extract_global_eda_features(x_raw: np.ndarray, fs: float = 4.0) -> np.ndarray:
    """
    Extract global features for one event-level EDA sequence.
    Returns:
        feat_vec: [feat_dim] float32, with nan_to_num applied.
    """
    feats = _create_global_feat_template()

    x = np.asarray(x_raw, dtype=np.float32)
    mask = np.isfinite(x)
    x_valid = x[mask]
    if x_valid.size < fs * 10:
        return np.zeros(len(GLOBAL_FEAT_KEYS), dtype=np.float32)

    feats["mean_eda"] = float(np.mean(x_valid))
    feats["median_eda"] = float(np.median(x_valid))
    feats["sd_eda"] = float(np.std(x_valid))
    try:
        feats["mad_eda"] = float(stats.median_abs_deviation(x_valid, scale="normal"))
    except Exception:
        feats["mad_eda"] = float(np.median(np.abs(x_valid - np.median(x_valid))))

    feats["min_eda"] = float(np.min(x_valid))
    feats["max_eda"] = float(np.max(x_valid))
    feats["range_eda"] = feats["max_eda"] - feats["min_eda"]
    feats["iqr_eda"] = float(stats.iqr(x_valid))
    feats["rms_eda"] = float(np.sqrt(np.mean(x_valid ** 2)))
    feats["kurtosis_eda"] = float(stats.kurtosis(x_valid))
    feats["skewness_eda"] = float(stats.skew(x_valid))

    diff1 = np.diff(x_valid)
    diff2 = np.diff(diff1) if diff1.size > 1 else np.array([0.0], dtype=np.float32)

    var0 = float(np.var(x_valid))
    var1 = float(np.var(diff1)) if diff1.size > 0 else 0.0
    var2 = float(np.var(diff2)) if diff2.size > 0 else 0.0

    feats["activity"] = var0
    feats["mobility"] = float(np.sqrt(var1 / var0)) if var0 > 0 else 0.0
    feats["complexity"] = (
        float(np.sqrt(var2 / var1) / feats["mobility"])
        if var1 > 0 and feats["mobility"] > 0 else 0.0
    )

    feats["mean_diff1"] = float(np.mean(diff1)) if diff1.size > 0 else 0.0
    feats["sd_diff1"] = float(np.std(diff1)) if diff1.size > 0 else 0.0
    feats["mean_diff2"] = float(np.mean(diff2)) if diff2.size > 0 else 0.0
    feats["sd_diff2"] = float(np.std(diff2)) if diff2.size > 0 else 0.0

    try:
        nperseg = min(len(x_valid), 1024)
        f, pxx = welch(x_valid, fs=fs, nperseg=nperseg)
        bands = {
            "vlf": (0.01, 0.045),
            "lf":  (0.045, 0.15),
            "hf":  (0.15, 0.25),
        }
        for band, (low, high) in bands.items():
            mask_band = (f >= low) & (f <= high)
            power = np.trapz(pxx[mask_band], f[mask_band]) if mask_band.any() else 0.0
            feats[f"{band}_power"] = float(power)

        feats["total_power"] = float(np.trapz(pxx, f))
        feats["lf_hf_ratio"] = feats["lf_power"] / max(feats["hf_power"], 1e-6)
    except Exception:
        pass

    def _safe_sampen(sig, scale):
        if ant is None:
            return np.nan
        sig = np.asarray(sig, dtype=float)
        if scale > 1:
            sig = pd.Series(sig).rolling(scale, min_periods=1).mean().dropna().values
        if sig.size < 50:
            return np.nan
        try:
            return float(ant.sample_entropy(sig, order=2, metric="chebyshev"))
        except Exception:
            return np.nan

    feats["sampen_scale1"] = _safe_sampen(x_valid, 1)
    feats["sampen_scale2"] = _safe_sampen(x_valid, 2)
    feats["higuchi_fd"] = float(_higuchi_fd(x_valid, k_max=5))

    try:
        nyq = fs / 2.0
        high_cut = 0.05 / nyq
        if 0 < high_cut < 1 and len(x_valid) > 10:
            sos = butter(2, high_cut, btype="high", output="sos")
            phasic = sosfiltfilt(sos, x_valid)

            feats["phasic_mean"] = float(np.mean(phasic))
            feats["phasic_sd"] = float(np.std(phasic))
            peaks, _ = find_peaks(phasic, height=0.02)
            feats["phasic_peaks"] = float(len(peaks))
            feats["phasic_auc"] = float(integrate.trapz(np.abs(phasic), dx=1.0 / fs))

            min_peak_height = 0.02
            min_rise_time = 0.5
            min_peak_distance = 1.0

            peaks2, props = find_peaks(
                phasic,
                height=min_peak_height,
                distance=int(fs * min_peak_distance),
                prominence=0.01,
            )
            if len(peaks2) > 0:
                amps = props.get("peak_heights", np.ones_like(peaks2) * np.nan)
                feats["scr_count"] = float(len(peaks2))
                feats["scr_mean_amp"] = float(np.nanmean(amps))
                feats["scr_mean_rise"] = float(min_rise_time)
    except Exception:
        pass

    vec = np.array([feats[k] for k in GLOBAL_FEAT_KEYS], dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec


# ==========================================================
#                   2) Dataset & collate
# ==========================================================

class EDARawEventDatasetWithFeat(Dataset):
    """
    Event-level dataset:
      - raw_seqs: list of [T_i, WIN_LEN]
      - feat_vecs: [N, F]
      - labels: [N]
    """
    def __init__(self, raw_seqs: List[np.ndarray], feat_vecs: np.ndarray, labels: np.ndarray):
        assert len(raw_seqs) == len(labels) == len(feat_vecs)

        self.raw_seqs = []
        for xr in raw_seqs:
            xr = np.nan_to_num(xr, nan=0.0, posinf=0.0, neginf=0.0)
            self.raw_seqs.append(torch.from_numpy(xr).float())

        self.feat_vecs = torch.from_numpy(
            np.nan_to_num(feat_vecs, nan=0.0, posinf=0.0, neginf=0.0)
        ).float()

        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.raw_seqs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.raw_seqs[idx], self.feat_vecs[idx], self.labels[idx]


def collate_event_batch_with_feat(batch):
    raw_seqs, feat_vecs, labels = zip(*batch)
    lengths = [s.size(0) for s in raw_seqs]
    max_len = max(lengths)

    B = len(raw_seqs)
    L = raw_seqs[0].size(1)

    raw_pad = torch.zeros(B, max_len, L, dtype=torch.float32)
    pad_mask = torch.ones(B, max_len, dtype=torch.bool)  # True = padding

    for i, xr in enumerate(raw_seqs):
        T = xr.size(0)
        raw_pad[i, :T, :] = xr
        pad_mask[i, :T] = False

    feat_batch = torch.stack(feat_vecs, dim=0)  # [B, F]
    labels = torch.stack(labels, dim=0)         # [B]

    return raw_pad, pad_mask, feat_batch, labels


# ==========================================================
#                      3) Model
# ==========================================================

class EDARawEventBERTWithGlobalFeat(nn.Module):
    """
    Transformer encoder with a global feature token.
    Inputs:
        x_raw:   [B, T, L]
        pad_mask:[B, T]
        feat_vec:[B, F]
    Process:
        - raw windows -> token embeddings: [B, T, d]
        - global features -> global token: [B, 1, d]
        - concat -> [B, 1+T, d]
        - encoder -> CLS pooling on global token
        - classify -> logits [B, 2]
    """
    def __init__(
        self,
        raw_len: int,
        feat_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_classes: int = 2,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        super().__init__()
        self.raw_len = raw_len
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.raw_proj = nn.Linear(raw_len, d_model)
        self.feat_proj = nn.Linear(feat_dim, d_model)

        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_raw: torch.Tensor, pad_mask: torch.Tensor, feat_vec: torch.Tensor) -> torch.Tensor:
        B, T, L = x_raw.shape
        if L != self.raw_len:
            raise ValueError(f"raw_len mismatch: got {L}, expected {self.raw_len}")
        if pad_mask.shape != (B, T):
            raise ValueError(f"pad_mask shape mismatch: got {pad_mask.shape}, expected {(B, T)}")

        tok_raw = self.raw_proj(x_raw)         # [B, T, d]
        feat_emb = self.feat_proj(feat_vec)    # [B, d]
        feat_emb = feat_emb.unsqueeze(1)       # [B, 1, d]

        tokens = torch.cat([feat_emb, tok_raw], dim=1)  # [B, 1+T, d]
        T_all = tokens.size(1)
        if T_all > self.max_seq_len:
            raise ValueError(f"Sequence length {T_all} > max_seq_len={self.max_seq_len}")

        feat_pad = torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)
        pad_mask_all = torch.cat([feat_pad, pad_mask], dim=1)  # [B, 1+T]

        pos = self.pos_embed[:, :T_all, :]
        tokens = tokens + pos

        x = tokens.transpose(0, 1)  # [S, B, d]
        h = self.encoder(x, src_key_padding_mask=pad_mask_all)
        h = h.transpose(0, 1)       # [B, S, d]

        cls_token = h[:, 0, :]      # global token as CLS
        logits = self.mlp_head(cls_token)
        return logits


# ==========================================================
#                      4) Evaluation
# ==========================================================

def evaluate(model, data_loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for xr, pad_mask, feat_vec, yb in data_loader:
            xr = xr.to(device)
            pad_mask = pad_mask.to(device)
            feat_vec = feat_vec.to(device)
            yb = yb.to(device)

            logits = model(xr, pad_mask, feat_vec)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = (probs >= 0.5).long()

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

    probs = torch.cat(all_probs).numpy()
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    metrics = dict(acc=acc, bal_acc=bal_acc, f1=f1, auc=auc)
    return metrics, labels, preds, probs


def compute_metrics_at_threshold(y_true, y_prob, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)
    return acc, bal_acc, f1, auc, cm


# ==========================================================
#      5) Long -> window tokens + global features
# ==========================================================

def build_event_tokens_from_filtered_long():
    """
    Build:
      - event_raw_seqs: list of [T_i, WIN_LEN]
      - y_used: np.array [N_events]
      - feat_mat: np.array [N_events, F]
    """
    print("[INFO] Loading filtered event-level long sequences, labels, and metadata...")
    X_raw_long_filt = np.load(RAW_LONG_FILT_NPY, allow_pickle=True)
    y_events = np.load(LABEL_NPY)
    meta_df = pd.read_csv(META_CSV)

    if not (len(X_raw_long_filt) == len(y_events) == len(meta_df)):
        raise ValueError("Length mismatch among raw_long_filt / labels / meta. Please check the inputs.")

    print(f"[INFO] Num events = {len(X_raw_long_filt)}")
    print(f"[INFO] Label distribution: {np.bincount(y_events)}")

    x0 = np.asarray(X_raw_long_filt[0], dtype=np.float32)
    print("[DEBUG] Example event 0 length =", len(x0))
    print("[DEBUG] Example event 0 first 20 points:", x0[:20])

    event_raw_seqs, event_labels, event_feats = [], [], []

    num_used = 0
    num_skip_too_short = 0
    num_skip_no_window = 0

    for i in range(len(X_raw_long_filt)):
        x_event = np.asarray(X_raw_long_filt[i], dtype=np.float32)
        label = int(y_events[i])
        row = meta_df.iloc[i]

        if "pid" in row and "event_idx" in row:
            event_id = f"{row['pid']}_{row['event_idx']}"
        else:
            event_id = f"event_{i}"

        if (i + 1) % 50 == 0 or i == 0:
            print(f"[INFO] Windowing event: {i+1}/{len(X_raw_long_filt)} -> {event_id} (len={len(x_event)})")

        feat_vec = extract_global_eda_features(x_event, fs=FS)

        mask_global = np.isfinite(x_event)
        global_vals = x_event[mask_global]
        if global_vals.size == 0:
            num_skip_no_window += 1
            continue

        g_mean = global_vals.mean()
        g_std = global_vals.std()
        if g_std < 1e-6:
            g_std = 1.0

        x_filled = np.nan_to_num(
            x_event.astype(np.float32),
            nan=g_mean, posinf=g_mean, neginf=g_mean
        )
        x_norm = (x_filled - g_mean) / g_std

        n = len(x_norm)
        if n < WIN_LEN:
            num_skip_too_short += 1
            continue

        windows = []
        for start in range(0, n - WIN_LEN + 1, STEP_LEN):
            window = x_norm[start: start + WIN_LEN]

            valid = np.isfinite(window)
            if valid.mean() < MIN_VALID_RATIO:
                continue

            windows.append(window.astype(np.float32))

        if len(windows) == 0:
            num_skip_no_window += 1
            continue

        if len(windows) > MAX_TOKENS:
            mid = len(windows) // 2
            half = MAX_TOKENS // 2
            start_idx = max(0, mid - half)
            end_idx = start_idx + MAX_TOKENS
            windows = windows[start_idx:end_idx]

        raw_seq = np.stack(windows, axis=0)

        event_raw_seqs.append(raw_seq)
        event_labels.append(label)
        event_feats.append(feat_vec)
        num_used += 1

    if num_used == 0:
        raise RuntimeError("No valid events produced any windows. Please check parameters or data quality.")

    y_used = np.array(event_labels, dtype=np.int64)
    feat_mat = np.stack(event_feats, axis=0)

    print("\n[DEBUG] Global feature matrix stats:")
    print("  shape =", feat_mat.shape)
    print("  first 2 samples, first 10 dims:")
    print(feat_mat[:2, :10])

    has_non_finite = (~np.isfinite(feat_mat)).any()
    print("  contains non-finite (NaN/Inf):", bool(has_non_finite))
    if has_non_finite:
        nan_ratio = np.isnan(feat_mat).mean(axis=0)
        print("  per-dim NaN ratio (first 10):", nan_ratio[:10])
        feat_mat = np.nan_to_num(feat_mat, nan=0.0, posinf=0.0, neginf=0.0)
        print("  applied nan_to_num to feat_mat.")

    print("[DEBUG] feat_mat min/max:", float(feat_mat.min()), float(feat_mat.max()))

    print("\n========== Event token statistics ==========")
    print(f"Used events: {num_used}")
    print(f"Label distribution: {np.bincount(y_used)}")
    seq_lens = [seq.shape[0] for seq in event_raw_seqs]
    print(f"Tokens per event: min={int(np.min(seq_lens))}, max={int(np.max(seq_lens))}, mean={np.mean(seq_lens):.2f}")
    print(f"[INFO] Skipped: too_short={num_skip_too_short}, no_valid_window={num_skip_no_window}")

    print("[DEBUG] Example event 0 num tokens =", event_raw_seqs[0].shape[0])
    print("[DEBUG] Example event 0 first window first 20 points:", event_raw_seqs[0][0, :20])

    return event_raw_seqs, y_used, feat_mat


# ==========================================================
#                           Main
# ==========================================================

def main():
    print(f"[INFO] Device: {DEVICE}")

    event_raw_seqs, y_events, feat_mat = build_event_tokens_from_filtered_long()
    raw_len = event_raw_seqs[0].shape[1]
    feat_dim = feat_mat.shape[1]
    print(f"[INFO] raw_len (window length) = {raw_len}")
    print(f"[INFO] feat_dim (global feature dim) = {feat_dim}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    fold_metrics = []
    all_true_folds, all_pred_folds, all_prob_folds = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(event_raw_seqs, y_events), start=1):
        print(f"\n========== Fold {fold_idx}/{N_SPLITS} ==========")

        Xr_train_list = [event_raw_seqs[i] for i in train_idx]
        Xr_test_list  = [event_raw_seqs[i] for i in test_idx]
        y_train       = y_events[train_idx]
        y_test        = y_events[test_idx]

        feat_train = feat_mat[train_idx]
        feat_test  = feat_mat[test_idx]

        print(f"[DEBUG] Fold {fold_idx} y_train={np.bincount(y_train)}, y_test={np.bincount(y_test)}")

        train_dataset = EDARawEventDatasetWithFeat(Xr_train_list, feat_train, y_train)
        test_dataset  = EDARawEventDatasetWithFeat(Xr_test_list,  feat_test,  y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_event_batch_with_feat,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_event_batch_with_feat,
        )

        class_counts = np.bincount(y_train)
        neg, pos = float(class_counts[0]), float(class_counts[1])
        w0 = 1.0
        w1 = neg / max(pos, 1.0)
        class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(DEVICE)
        print(f"[INFO] Fold {fold_idx} class_weights = {class_weights.cpu().numpy()}")

        model = EDARawEventBERTWithGlobalFeat(
            raw_len=raw_len,
            feat_dim=feat_dim,
            d_model=D_MODEL,
            nhead=N_HEAD,
            num_layers=N_LAYERS,
            dim_feedforward=FFN_DIM,
            dropout=DROPOUT,
            num_classes=2,
            max_seq_len=MAX_SEQ_LEN,
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_metric_val = -1.0
        best_epoch = 0
        best_state_dict = None
        epochs_no_improve = 0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            epoch_loss = 0.0

            for xr_batch, pad_mask, feat_batch, yb in train_loader:
                xr_batch = xr_batch.to(DEVICE)
                pad_mask = pad_mask.to(DEVICE)
                feat_batch = feat_batch.to(DEVICE)
                yb = yb.to(DEVICE)

                optimizer.zero_grad()
                logits = model(xr_batch, pad_mask, feat_batch)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * xr_batch.size(0)

            epoch_loss /= len(train_dataset)

            metrics_val, _, _, _ = evaluate(model, test_loader, DEVICE)
            metric_for_early = metrics_val[EARLY_STOP_METRIC]

            if metric_for_early > best_metric_val:
                best_metric_val = metric_for_early
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
                print(
                    f"Fold {fold_idx} | Epoch [{epoch:02d}/{EPOCHS}] "
                    f"Loss={epoch_loss:.4f} "
                    f"ACC={metrics_val['acc']:.4f} "
                    f"Bal_ACC={metrics_val['bal_acc']:.4f} "
                    f"F1={metrics_val['f1']:.4f} "
                    f"AUC={metrics_val['auc']:.4f} "
                    f"(early={EARLY_STOP_METRIC}:{metric_for_early:.4f}, "
                    f"best={best_metric_val:.4f}@{best_epoch}, "
                    f"no_improve={epochs_no_improve})"
                )

            if EARLY_STOP and epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"[INFO] Fold {fold_idx}: Early stopping triggered at epoch {epoch}")
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            print(
                f"[INFO] Fold {fold_idx}: Using best epoch={best_epoch}, "
                f"{EARLY_STOP_METRIC}={best_metric_val:.4f} for final evaluation."
            )

        metrics_fold, y_true_fold, y_pred_fold, y_prob_fold = evaluate(model, test_loader, DEVICE)
        fold_metrics.append(metrics_fold)
        all_true_folds.append(y_true_fold)
        all_pred_folds.append(y_pred_fold)
        all_prob_folds.append(y_prob_fold)

        print(
            f"[Fold {fold_idx}] Final metrics (threshold=0.5): "
            f"ACC={metrics_fold['acc']:.4f}, "
            f"Bal_ACC={metrics_fold['bal_acc']:.4f}, "
            f"F1={metrics_fold['f1']:.4f}, "
            f"AUC={metrics_fold['auc']:.4f}"
        )

    select_key = "bal_acc"
    best_fold_idx = None
    best_score = -1.0
    best_result = None

    for i, m in enumerate(fold_metrics):
        if m[select_key] > best_score:
            best_score = m[select_key]
            best_fold_idx = i + 1
            best_result = m

    print("\n========== Best Fold (by %s) ==========" % select_key)
    print(f"Best fold = {best_fold_idx}")
    print("Metrics =", best_result)

    fold_cm = confusion_matrix(
        all_true_folds[best_fold_idx - 1],
        all_pred_folds[best_fold_idx - 1],
    )
    print("Confusion matrix for best fold:")
    print(fold_cm)
    tn, fp, fn, tp = fold_cm.ravel()
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    all_true = np.concatenate(all_true_folds, axis=0)
    all_pred = np.concatenate(all_pred_folds, axis=0)
    all_prob = np.concatenate(all_prob_folds, axis=0)

    cm_default = confusion_matrix(all_true, all_pred)
    tn, fp, fn, tp = cm_default.ravel()

    print("\n========== 10-fold summary (threshold=0.5) ==========")
    keys = fold_metrics[0].keys()
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics])
        print(f"{k}: {vals.mean():.4f} ± {vals.std():.4f}")

    print("\n========== Final confusion matrix (all folds, threshold=0.5) ==========")
    print(cm_default)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    print("\n========== Global threshold scan ==========")
    candidate_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]

    best_f1 = -1.0
    best_f1_t = 0.5
    best_bal = -1.0
    best_bal_t = 0.5

    for t in candidate_thresholds:
        acc_t, bal_t, f1_t, auc_t, cm_t = compute_metrics_at_threshold(all_true, all_prob, t)
        print(f"  t={t:.2f}: ACC={acc_t:.4f}, Bal_ACC={bal_t:.4f}, F1={f1_t:.4f}, AUC={auc_t:.4f}")
        print(f"      CM=\n{cm_t}")
        if f1_t > best_f1:
            best_f1 = f1_t
            best_f1_t = t
        if bal_t > best_bal:
            best_bal = bal_t
            best_bal_t = t

    print("\n[Best global threshold by F1]")
    acc_f1, bal_f1, f1_f1, auc_f1, cm_f1 = compute_metrics_at_threshold(all_true, all_prob, best_f1_t)
    print(f"  t={best_f1_t:.2f}: ACC={acc_f1:.4f}, Bal_ACC={bal_f1:.4f}, F1={f1_f1:.4f}, AUC={auc_f1:.4f}")
    print(f"  CM=\n{cm_f1}")

    print("\n[Best global threshold by Bal_ACC]")
    acc_b, bal_b, f1_b, auc_b, cm_b = compute_metrics_at_threshold(all_true, all_prob, best_bal_t)
    print(f"  t={best_bal_t:.2f}: ACC={acc_b:.4f}, Bal_ACC={bal_b:.4f}, F1={f1_b:.4f}, AUC={auc_b:.4f}")
    print(f"  CM=\n{cm_b}")


if __name__ == "__main__":
    main()
