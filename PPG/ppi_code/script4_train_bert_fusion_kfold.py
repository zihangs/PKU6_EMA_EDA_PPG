# coding: utf-8
"""
使用 ppg_pipeline 和 K-fold 交叉验证训练 BERT+MLP 模型
替换原有的 script4_train_bert_fusion.py
"""

from __future__ import annotations
import os
import math
import json
import random
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from model_bert_fusion import PPGEventBERTFusionCLSGate

BASE_DIR = "/Users/zhushiyao/Desktop/ppi/ppg_code"

@dataclass
class Config:
    base_dir: str = BASE_DIR
    windows_npy: str = "event_long/ppi_event_windows.npy"
    window_feats_npy: str = "event_long/ppi_event_window_features.npy"
    labels_npy: str = "event_long/ppg_event_labels_used.npy"
    output_dir: str = "models_ppg"
    model_name_prefix: str = "ppi_bert_mlp_kfold"
    history_name_prefix: str = "train_history_kfold"
    
    max_tokens: int = 120
    raw_event_norm: bool = True
    norm_clip: float = 5.0
    global_feat_mode: str = "mean_std_max"
    
    d_model: int = 128
    dropout: float = 0.3
    
    seed: int = 42
    batch_size: int = 16
    epochs: int = 80
    lr_peak: float = 6e-5
    weight_decay: float = 5e-4
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    early_stop_patience: int = 12
    amp: bool = True
    
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    use_class_weights: bool = True
    
    aug_noise_std: float = 0.01
    aug_gain_jitter: float = 0.06
    aug_time_mask_ratio: float = 0.10
    
    select_metric: str = "bal_acc"
    threshold_grid: int = 21
    fixed_threshold: Optional[float] = None  # 设为 None 使用网格搜索；如需固定阈值可改为 0.5
    max_window_rows_for_scaler: int = 200_000
    
    # K-fold 参数
    n_folds: int = 5
    shuffle: bool = True

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def robust_norm(x: np.ndarray, clip: float = 5.0) -> np.ndarray:
    x = np.nan_to_num(x, nan=0, posinf=0, neginf=0).astype(np.float32)
    med, q75, q25 = np.median(x), np.percentile(x, 75), np.percentile(x, 25)
    scale = max(q75 - q25, np.std(x), 1e-6)
    return np.clip((x - med) / scale, -clip, clip)

def crop_seq(w: np.ndarray, f: np.ndarray, max_len: int, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    if w.shape[0] <= max_len:
        return w, f
    start = (w.shape[0] - max_len) // 2 if mode == "center" else np.random.randint(0, w.shape[0] - max_len + 1)
    return w[start:start+max_len], f[start:start+max_len]

def aggregate_feats(feats: np.ndarray, mode: str) -> np.ndarray:
    x = np.nan_to_num(feats, nan=0).astype(np.float32)
    if x.shape[0] == 0:
        dim = x.shape[1] if len(x.shape) > 1 else 0
        if dim == 0:
            raise ValueError("Empty feature array")
        multipliers = {"mean": 1, "mean_std": 2, "mean_std_max": 3, "mean_std_min_max": 4}
        return np.zeros(dim * multipliers.get(mode, 3), dtype=np.float32)
    
    std_val = np.nan_to_num(x.std(axis=0), nan=0.0)
    result = [x.mean(axis=0), std_val]
    if "max" in mode:
        result.append(x.max(axis=0))
    if "min" in mode:
        result.append(x.min(axis=0))
    return np.concatenate(result)

def cosine_warmup_lr(epoch: int, cfg: Config) -> float:
    if epoch <= cfg.warmup_epochs:
        return cfg.lr_peak * (epoch / cfg.warmup_epochs)
    progress = (epoch - cfg.warmup_epochs) / (cfg.epochs - cfg.warmup_epochs)
    return cfg.lr_peak * 0.5 * (1 + math.cos(math.pi * progress))

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('class_weights', class_weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, reduction="none", weight=self.class_weights)
        p_t = F.softmax(inputs, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha_t = self.alpha.to(inputs.device)
        if alpha_t.numel() == 1:
            alpha_t = alpha_t.expand_as(targets)
        else:
            alpha_t = alpha_t.gather(0, targets)
        return (alpha_t * (1 - p_t) ** self.gamma * ce).mean()

class PPGDataset(Dataset):
    def __init__(self, windows: List[np.ndarray], feats: List[np.ndarray],
                 global_feats: np.ndarray, labels: np.ndarray, cfg: Config, train: bool):
        self.cfg, self.train = cfg, train
        self.windows = [robust_norm(w, cfg.norm_clip) if cfg.raw_event_norm else np.nan_to_num(w, nan=0) for w in windows]
        self.feats = [np.nan_to_num(f, nan=0) for f in feats]
        self.global_feats = np.nan_to_num(global_feats, nan=0)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def _augment(self, w: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.cfg.aug_noise_std > 0:
            w = w + np.random.randn(*w.shape).astype(np.float32) * self.cfg.aug_noise_std
        if self.cfg.aug_gain_jitter > 0:
            w = w * (1 + (np.random.rand() - 0.5) * self.cfg.aug_gain_jitter)
        if self.cfg.aug_time_mask_ratio > 0 and w.shape[0] > 8:
            n = max(1, int(self.cfg.aug_time_mask_ratio * w.shape[0]))
            idx = np.random.permutation(w.shape[0])[:n]
            w[idx], f[idx] = 0, 0
        return w, f

    def __getitem__(self, idx):
        w, f = self.windows[idx].copy(), self.feats[idx].copy()
        if w.shape[0] > self.cfg.max_tokens:
            w, f = crop_seq(w, f, self.cfg.max_tokens, "random" if self.train else "center")
        if self.train:
            w, f = self._augment(w, f)
        return (torch.from_numpy(w).float(), torch.from_numpy(f).float(),
                torch.from_numpy(self.global_feats[idx]).float(), torch.tensor(self.labels[idx], dtype=torch.long))

def collate_fn(batch):
    w_list, f_list, g_list, y_list = zip(*batch)
    max_len = max(w.size(0) for w in w_list)
    B, L, Fw = len(w_list), w_list[0].size(1), f_list[0].size(1)
    w_pad = torch.zeros(B, max_len, L, dtype=torch.float32)
    f_pad = torch.zeros(B, max_len, Fw, dtype=torch.float32)
    mask = torch.ones(B, max_len, dtype=torch.bool)
    for i, (w, f) in enumerate(zip(w_list, f_list)):
        T = w.size(0)
        w_pad[i, :T], f_pad[i, :T] = w, f
        mask[i, :T] = False
    return w_pad, f_pad, mask, torch.stack(g_list), torch.stack(y_list)

@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs, ys = [], []
    for w, f, mask, g, y in loader:
        w, f, mask, g = w.to(device), f.to(device), mask.to(device), g.to(device)
        logits = model(w, f, mask, g)
        probs.append(F.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)

def compute_metrics(probs: np.ndarray, y: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (probs >= thr).astype(int)
    try:
        auc = roc_auc_score(y, probs)
    except:
        auc = float("nan")
    prec = precision_score(y, pred, average=None, zero_division=0)
    rec = recall_score(y, pred, average=None, zero_division=0)
    f1 = f1_score(y, pred, average=None, zero_division=0)
    return {
        "acc": accuracy_score(y, pred),
        "bal_acc": balanced_accuracy_score(y, pred),
        "auc": auc,
        "prec0": float(prec[0]), "prec1": float(prec[1]),
        "rec0": float(rec[0]), "rec1": float(rec[1]),
        "f10": float(f1[0]), "f11": float(f1[1]),
    }

def search_threshold(probs: np.ndarray, y: np.ndarray, cfg: Config) -> Tuple[float, Dict]:
    # 如果固定阈值，直接计算
    if cfg.fixed_threshold is not None:
        m = compute_metrics(probs, y, float(cfg.fixed_threshold))
        return float(cfg.fixed_threshold), m

    best_t, best_s, best_m = 0.5, -1e9, {}
    for t in np.linspace(0, 1, cfg.threshold_grid):
        m = compute_metrics(probs, y, float(t))
        if m["prec1"] < 0.15 or m["rec1"] < 0.15:
            continue
        score_map = {"bal_acc": m["bal_acc"], "auc": m["auc"], "f11": m["f11"]}
        score = score_map.get(cfg.select_metric, m["bal_acc"])
        if np.isfinite(score) and score > best_s:
            best_s, best_t, best_m = score, float(t), m
    return best_t, best_m

def _to_array(x):
    return np.array(x, dtype=np.float32) if hasattr(x, '__len__') else np.array([x], dtype=np.float32)

def load_data(cfg: Config):
    print("加载数据...")
    y = np.load(os.path.join(cfg.base_dir, cfg.labels_npy))
    windows_obj = np.load(os.path.join(cfg.base_dir, cfg.windows_npy), allow_pickle=True)
    feats_obj = np.load(os.path.join(cfg.base_dir, cfg.window_feats_npy), allow_pickle=True)
    
    valid_indices = [i for i in range(len(y)) 
                     if _to_array(windows_obj[i]).size > 0 and _to_array(feats_obj[i]).size > 0]
    
    if len(valid_indices) == 0:
        raise ValueError("No valid samples found!")
    
    y = y[valid_indices]
    windows_obj = windows_obj[valid_indices] if isinstance(windows_obj, np.ndarray) else [windows_obj[i] for i in valid_indices]
    feats_obj = feats_obj[valid_indices] if isinstance(feats_obj, np.ndarray) else [feats_obj[i] for i in valid_indices]
    
    print(f"有效样本数: {len(y)}")
    
    print("计算全局特征...")
    global_feats, final_valid_indices = [], []
    for i, f in enumerate(feats_obj):
        f_arr = _to_array(f)
        if f_arr.ndim == 1:
            f_arr = f_arr.reshape(1, -1)
        if f_arr.shape[0] > 0:
            try:
                global_feats.append(aggregate_feats(f_arr, cfg.global_feat_mode))
                final_valid_indices.append(i)
            except:
                continue
    
    if len(final_valid_indices) != len(y):
        y = y[final_valid_indices]
        windows_obj = windows_obj[final_valid_indices] if isinstance(windows_obj, np.ndarray) else [windows_obj[i] for i in final_valid_indices]
        feats_obj = feats_obj[final_valid_indices] if isinstance(feats_obj, np.ndarray) else [feats_obj[i] for i in final_valid_indices]
    
    global_feats = np.array(global_feats)
    print(f"全局特征维度: {global_feats.shape}")
    
    print("标准化特征...")
    global_scaler = StandardScaler()
    if len(global_feats) > cfg.max_window_rows_for_scaler:
        global_scaler.fit(global_feats[np.random.choice(len(global_feats), cfg.max_window_rows_for_scaler, replace=False)])
    else:
        global_scaler.fit(global_feats)
    global_feats = global_scaler.transform(global_feats)
    
    all_win_feats = []
    for f in feats_obj:
        f_arr = _to_array(f)
        if f_arr.ndim == 1:
            f_arr = f_arr.reshape(1, -1)
        if f_arr.shape[0] > 0:
            all_win_feats.append(f_arr)
    
    if len(all_win_feats) > 0:
        sample_size = min(cfg.max_window_rows_for_scaler, sum(f.shape[0] for f in all_win_feats))
        sample_feats = []
        for f in all_win_feats:
            if len(sample_feats) >= sample_size:
                break
            sample_feats.append(f[:min(f.shape[0], sample_size - len(sample_feats))])
        
        win_scaler = StandardScaler()
        win_scaler.fit(np.vstack(sample_feats))
        
        feats_normalized = []
        for f in feats_obj:
            f_arr = _to_array(f)
            if f_arr.ndim == 1:
                f_arr = f_arr.reshape(1, -1)
            feats_normalized.append(win_scaler.transform(f_arr) if f_arr.shape[0] > 0 else f_arr)
        feats_obj = feats_normalized
    else:
        win_scaler = None
    
    return ([windows_obj[i] for i in range(len(y))], [feats_obj[i] for i in range(len(y))], 
            global_feats, y, global_scaler, win_scaler)

def train_fold(cfg: Config, fold_idx: int, train_idx: np.ndarray, val_idx: np.ndarray,
               windows: List, feats: List, global_feats: np.ndarray, labels: np.ndarray,
               global_scaler: StandardScaler, win_scaler: StandardScaler, device):
    """训练单个 fold"""
    print(f"\n{'='*80}")
    print(f"Fold {fold_idx + 1}/{cfg.n_folds}")
    print(f"{'='*80}")
    
    # 划分数据
    train_w = [windows[i] for i in train_idx]
    train_f = [feats[i] for i in train_idx]
    train_g = global_feats[train_idx]
    train_y = labels[train_idx]
    
    val_w = [windows[i] for i in val_idx]
    val_f = [feats[i] for i in val_idx]
    val_g = global_feats[val_idx]
    val_y = labels[val_idx]
    
    print(f"训练集: {len(train_y)}, 验证集: {len(val_y)}")
    
    # 创建数据集
    train_ds = PPGDataset(train_w, train_f, train_g, train_y, cfg, train=True)
    val_ds = PPGDataset(val_w, val_f, val_g, val_y, cfg, train=False)
    
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=pin_memory)
    
    # 创建模型
    sample_w, sample_f = np.array(train_w[0]), np.array(train_f[0])
    raw_len, win_feat_dim, global_feat_dim = sample_w.shape[1], sample_f.shape[1], train_g.shape[1]
    
    print(f"\n模型输入维度: raw_len={raw_len}, win_feat_dim={win_feat_dim}, global_feat_dim={global_feat_dim}, d_model={cfg.d_model}")
    
    model = PPGEventBERTFusionCLSGate(raw_len, win_feat_dim, global_feat_dim, cfg.d_model, cfg.dropout).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_peak, weight_decay=cfg.weight_decay)
    
    class_weights = None
    if cfg.use_class_weights:
        class_counts = np.bincount(train_y)
        class_weights = torch.tensor([1.0 / (c + 1e-6) for c in class_counts], dtype=torch.float32, device=device)
        class_weights = class_weights / class_weights.sum()
        print(f"类别权重: {class_weights.cpu().numpy()}")
    
    criterion = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, class_weights=class_weights) if cfg.use_focal_loss else nn.CrossEntropyLoss(weight=class_weights)
    criterion = criterion.to(device)
    
    best_score, best_epoch, no_improve = -1, 0, 0
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp) if torch.cuda.is_available() and cfg.amp else None
    history = []
    
    amp_device_type = "cuda" if torch.cuda.is_available() and cfg.amp else "cpu"
    amp_enabled = cfg.amp and torch.cuda.is_available()
    
    print("\n开始训练...")
    print("=" * 80)
    
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, total_samples = 0, 0
        
        lr = cosine_warmup_lr(epoch, cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        for w, f, mask, g, y in train_loader:
            w, f, mask, g, y = w.to(device), f.to(device), mask.to(device), g.to(device), y.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=amp_device_type, enabled=amp_enabled):
                logits = model(w, f, mask, g)
                loss = criterion(logits, y)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            
            total_loss += loss.item() * w.size(0)
            total_samples += w.size(0)
        
        train_loss = total_loss / total_samples
        val_probs, val_y_true = predict_probs(model, val_loader, device)
        best_thr, val_metrics = search_threshold(val_probs, val_y_true, cfg)
        
        train_probs, _ = predict_probs(model, train_loader, device)
        _, train_metrics = search_threshold(train_probs, train_y, cfg)
        
        history.append({
            "epoch": epoch, "lr": lr, "train_loss": train_loss,
            "train_acc": train_metrics["acc"], "train_bal_acc": train_metrics["bal_acc"],
            "val_acc": val_metrics["acc"], "val_bal_acc": val_metrics["bal_acc"],
            "val_auc": val_metrics["auc"], "val_prec1": val_metrics["prec1"],
            "val_rec1": val_metrics["rec1"], "val_f11": val_metrics["f11"], "best_thr": best_thr,
        })
        
        print(f"Epoch {epoch:03d}/{cfg.epochs:03d} | LR: {lr:.6f} | Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_metrics['acc']:.4f} | Val Acc: {val_metrics['acc']:.4f} | "
              f"Val BalAcc: {val_metrics['bal_acc']:.4f} | Val AUC: {val_metrics['auc']:.4f} | "
              f"Val F1: {val_metrics['f11']:.4f} | Thr: {best_thr:.3f}")
        
        score = val_metrics[cfg.select_metric]
        if score > best_score:
            best_score, best_epoch = score, epoch
            no_improve = 0
            
            model_path = os.path.join(cfg.base_dir, cfg.output_dir, f"{cfg.model_name_prefix}_fold{fold_idx+1}_best.pt")
            torch.save({
                "state_dict": model.state_dict(), "config": asdict(cfg),
                "best_select_metric": best_score, "best_thr": best_thr, "best_epoch": best_epoch,
                "fold_idx": fold_idx,
                "global_scaler_mean": global_scaler.mean_, "global_scaler_scale": global_scaler.scale_,
                "win_scaler_mean": win_scaler.mean_ if win_scaler else None,
                "win_scaler_scale": win_scaler.scale_ if win_scaler else None,
                "raw_len": raw_len, "win_feat_dim": win_feat_dim, "global_feat_dim": global_feat_dim,
            }, model_path)
            print(f"  ✓ 保存最佳模型 (epoch {epoch}, {cfg.select_metric}={best_score:.4f})")
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                print(f"\n早停于 epoch {epoch} (patience={cfg.early_stop_patience})")
                break
    
    print("=" * 80)
    print(f"\nFold {fold_idx + 1} 训练完成! 最佳epoch: {best_epoch}, 最佳{cfg.select_metric}: {best_score:.4f}")
    
    history_path = os.path.join(cfg.base_dir, cfg.output_dir, f"{cfg.history_name_prefix}_fold{fold_idx+1}.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"训练历史已保存: {history_path}")
    
    return best_score, best_epoch, history

def train(cfg: Config):
    set_seed(cfg.seed)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"使用设备: {device}")
    
    os.makedirs(os.path.join(cfg.base_dir, cfg.output_dir), exist_ok=True)
    
    # 加载数据
    windows, feats, global_feats, labels, global_scaler, win_scaler = load_data(cfg)
    
    # K-fold 交叉验证
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=cfg.shuffle, random_state=cfg.seed)
    
    fold_results = []
    all_histories = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        best_score, best_epoch, history = train_fold(
            cfg, fold_idx, train_idx, val_idx,
            windows, feats, global_feats, labels,
            global_scaler, win_scaler, device
        )
        
        fold_results.append({
            "fold": fold_idx + 1,
            "best_score": best_score,
            "best_epoch": best_epoch
        })
        all_histories.append(history)
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("K-fold 交叉验证结果汇总")
    print("=" * 80)
    
    scores = [r["best_score"] for r in fold_results]
    print(f"\n各 Fold 最佳 {cfg.select_metric}:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: {r['best_score']:.4f} (epoch {r['best_epoch']})")
    
    print(f"\n平均 {cfg.select_metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"最佳 Fold: {np.argmax(scores) + 1} ({max(scores):.4f})")
    print(f"最差 Fold: {np.argmin(scores) + 1} ({min(scores):.4f})")
    
    # 保存汇总结果
    summary_path = os.path.join(cfg.base_dir, cfg.output_dir, f"{cfg.history_name_prefix}_summary.json")
    summary = {
        "config": asdict(cfg),
        "fold_results": fold_results,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "best_fold": int(np.argmax(scores) + 1),
        "worst_fold": int(np.argmin(scores) + 1)
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总结果已保存: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPG事件BERT融合模型训练 (K-fold交叉验证)")
    parser.add_argument("--base_dir", type=str, default=BASE_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--no_amp", action="store_true")
    
    args = parser.parse_args()
    cfg = Config()
    cfg.base_dir = args.base_dir
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr_peak = args.lr
    cfg.d_model = args.d_model
    cfg.dropout = args.dropout
    cfg.n_folds = args.n_folds
    cfg.amp = not args.no_amp
    train(cfg)
