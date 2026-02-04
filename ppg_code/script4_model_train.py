import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

# ================= 1. 配置 =================
BASE_DIR = r"/Users/zhushiyao/Desktop/ppg_code/event_long"
WIN_FEATS_NPY = os.path.join(BASE_DIR, "ppg_window_features.npy")
GLOB_FEATS_NPY = os.path.join(BASE_DIR, "ppg_global_long_feats.npy") 
LABEL_NPY      = os.path.join(BASE_DIR, "ppg_labels_used.npy")

# 超参数
WIN_DIM = 35          
GLOB_DIM = 15         
MAX_WINS = 80        
BATCH_SIZE = 32
EPOCHS = 50           # [修改] 增加到 50 轮
PATIENCE = 15         # [修改] 早停耐心值，给模型更多机会
FOLDS = 10            # [修改] 10折交叉验证
LR = 1e-3             
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 2. 最佳模型定义 (SE-Block + PreNorm + ResMLP) =================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, t, c = x.size()
        y = x.permute(0, 2, 1)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, 1, c)
        return x * y.expand_as(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class PPGBERT_Full(nn.Module):
    def __init__(self, d_model=64, dropout=0.4):
        super().__init__()
        # 1. Projections with SE-Block
        self.win_proj_fc1 = nn.Linear(WIN_DIM, d_model * 2)
        self.win_se = SEBlock(d_model * 2)
        self.win_proj_fc2 = nn.Linear(d_model * 2, d_model)
        self.win_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.cls_proj = nn.Sequential(
            nn.Linear(GLOB_DIM, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model)
        )
        
        # 2. Transformer (Pre-Norm)
        self.pos_enc = PositionalEncoding(d_model, max_len=128)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=128, dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. Res-MLP Head
        self.head_fc1 = nn.Linear(d_model * 3, d_model * 2)
        self.head_ln1 = nn.LayerNorm(d_model * 2)
        self.res_fc = nn.Linear(d_model * 2, d_model * 2)
        self.res_ln = nn.LayerNorm(d_model * 2)
        self.head_out = nn.Linear(d_model * 2, 2)

    def forward(self, win_feats, pad_mask, glob_feats):
        B = win_feats.size(0)
        # Projection
        x_seq = self.win_proj_fc1(win_feats)
        x_seq = F.gelu(x_seq)
        x_seq = self.dropout(x_seq)
        x_seq = self.win_se(x_seq) # SE-Block
        x_seq = self.win_norm(self.win_proj_fc2(x_seq))
        
        x_cls = self.cls_proj(glob_feats).unsqueeze(1)
        tokens = torch.cat([x_cls, x_seq], dim=1)
        
        # Transformer
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=pad_mask.device)
        pad_mask_all = torch.cat([cls_mask, pad_mask], dim=1)
        tokens = self.pos_enc(tokens)
        context = self.transformer(tokens, src_key_padding_mask=pad_mask_all)
        
        # Pooling
        cls_out = context[:, 0, :]
        seq_out = context[:, 1:, :]
        mask_expanded = pad_mask.unsqueeze(-1).float()
        avg_out = (seq_out * (1 - mask_expanded)).sum(1) / ((1 - mask_expanded).sum(1) + 1e-8)
        max_out = seq_out.masked_fill(pad_mask.unsqueeze(-1), -1e9).max(1)[0]
        concat = torch.cat([cls_out, avg_out, max_out], dim=1)
        
        # Res-MLP
        x = self.dropout(F.gelu(self.head_ln1(self.head_fc1(concat))))
        res = x
        x = self.dropout(F.gelu(self.res_ln(self.res_fc(x))))
        x = x + res # Residual
        return self.head_out(x)

# ================= 3. 数据处理 =================
class PPGDataset(Dataset):
    def __init__(self, win_feats, glob_feats, labels, mode='train'):
        self.samples = []; self.glob = torch.from_numpy(glob_feats).float(); self.labels = torch.from_numpy(labels).long()
        self.mode = mode
        for seq in win_feats:
            if seq.shape[0]>MAX_WINS: seq = seq[(seq.shape[0]-MAX_WINS)//2 : (seq.shape[0]-MAX_WINS)//2 + MAX_WINS]
            self.samples.append(torch.from_numpy(seq).float())
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        seq, glob = self.samples[idx], self.glob[idx]
        if self.mode == 'train': # 训练噪声增强
            seq = seq + torch.randn_like(seq) * 0.02
            glob = glob + torch.randn_like(glob) * 0.02
        return seq, glob, self.labels[idx]

def collate_fn(batch):
    wins, globs, labels = zip(*batch)
    max_len = max([x.size(0) for x in wins])
    padded = torch.zeros(len(wins), max_len, WIN_DIM); mask = torch.ones(len(wins), max_len, dtype=torch.bool)
    for i, seq in enumerate(wins):
        padded[i, :seq.size(0), :] = seq
        mask[i, :seq.size(0)] = False
    return padded, mask, torch.stack(globs), torch.stack(labels)

def scale_data(X_win, X_glob):
    s_glob = StandardScaler()
    X_glob_n = s_glob.fit_transform(np.nan_to_num(X_glob))
    s_win = StandardScaler()
    all_wins = np.vstack(X_win)
    s_win.fit(np.nan_to_num(all_wins))
    X_win_n = []
    for seq in X_win:
        seq = np.nan_to_num(seq)
        if len(seq)>0: X_win_n.append(s_win.transform(seq))
        else: X_win_n.append(seq)
    return np.array(X_win_n, dtype=object), X_glob_n

# ================= 4. 十折交叉验证主程序 =================
def run_10fold_cv():
    print(f"[INFO] Starting 10-Fold Cross-Validation (Epochs={EPOCHS}, Patience={PATIENCE})...")
    
    if not os.path.exists(WIN_FEATS_NPY): print("Data missing."); return
    X_wins = np.load(WIN_FEATS_NPY, allow_pickle=True)
    X_glob = np.load(GLOB_FEATS_NPY)
    y_all  = np.load(LABEL_NPY)
    
    # 截断对齐
    min_len = min(len(X_wins), len(X_glob), len(y_all))
    X_wins, X_glob, y_all = X_wins[:min_len], X_glob[:min_len], y_all[:min_len]

    # 10折
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
    fold_aucs = []
    fold_accs = []
    fold_bal_accs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_glob, y_all), 1):
        print(f"\n========== Fold {fold}/{FOLDS} ==========")
        
        # 数据准备
        X_w_tr, X_g_tr, y_tr = X_wins[train_idx], X_glob[train_idx], y_all[train_idx]
        X_w_val, X_g_val, y_val = X_wins[val_idx], X_glob[val_idx], y_all[val_idx]
        
        # 独立标准化 (严谨做法：每折重新 fit scaler)
        X_w_tr, X_g_tr = scale_data(X_w_tr, X_g_tr)
        X_w_val, X_g_val = scale_data(X_w_val, X_g_val)
        
        # DataLoader
        weights = 1. / (np.bincount(y_tr) + 1e-6)
        sampler = WeightedRandomSampler(weights[y_tr], len(y_tr))
        
        train_loader = DataLoader(PPGDataset(X_w_tr, X_g_tr, y_tr, 'train'), batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)
        val_loader = DataLoader(PPGDataset(X_w_val, X_g_val, y_val, 'eval'), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        # 模型初始化
        model = PPGBERT_Full(d_model=64, dropout=0.4).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)
        
        # 状态跟踪
        best_auc = 0
        best_metrics = {}
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(1, EPOCHS+1):
            model.train()
            train_loss = 0
            for w, m, g, y in train_loader:
                w, m, g, y = w.to(DEVICE), m.to(DEVICE), g.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out = model(w, m, g)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            scheduler.step()
            
            # Validation
            model.eval()
            preds, trues, probs = [], [], []
            with torch.no_grad():
                for w, m, g, y in val_loader:
                    w, m, g = w.to(DEVICE), m.to(DEVICE), g.to(DEVICE)
                    out = model(w, m, g)
                    prob = torch.softmax(out, dim=1)[:, 1]
                    probs.extend(prob.cpu().numpy())
                    trues.extend(y.numpy())
            
            auc = roc_auc_score(trues, probs)
            
            # 计算最佳 Acc/BalAcc
            curr_best_bal = 0
            curr_best_acc = 0
            for thr in np.linspace(0.3, 0.7, 50):
                p_bin = (np.array(probs) > thr).astype(int)
                b_acc = balanced_accuracy_score(trues, p_bin)
                if b_acc > curr_best_bal:
                    curr_best_bal = b_acc
                    curr_best_acc = accuracy_score(trues, p_bin)
            
            # Checkpoint & Early Stop
            if auc > best_auc:
                best_auc = auc
                best_metrics = {'acc': curr_best_acc, 'bal': curr_best_bal}
                best_epoch = epoch
                patience_counter = 0
                # 若需要保存每折的最佳模型，可取消注释下行
                # torch.save(model.state_dict(), os.path.join(BASE_DIR, f"fold{fold}_best.pth"))
            else:
                patience_counter += 1
            
            # 只有在有提升，或者每5轮时打印，保持清爽
            if epoch == best_epoch:
                print(f"  Ep {epoch}: Loss={train_loss/len(train_loader):.4f} | AUC={auc:.4f} [Best]")
            elif epoch % 10 == 0:
                print(f"  Ep {epoch}: Loss={train_loss/len(train_loader):.4f} | AUC={auc:.4f}")
            
            if patience_counter >= PATIENCE:
                print(f"  [Early Stop] @ Ep {epoch}. Best was Ep {best_epoch} (AUC={best_auc:.4f})")
                break
        
        print(f"  >> Fold Result: AUC={best_auc:.4f} | Acc={best_metrics.get('acc',0):.4f} | BalAcc={best_metrics.get('bal',0):.4f}")
        fold_aucs.append(best_auc)
        fold_accs.append(best_metrics.get('acc', 0))
        fold_bal_accs.append(best_metrics.get('bal', 0))

    # 最终汇总
    print(f"\n========== 10-Fold CV Final Report ==========")
    print(f"Mean AUC:    {np.mean(fold_aucs):.4f} (±{np.std(fold_aucs):.4f})")
    print(f"Mean Acc:    {np.mean(fold_accs):.4f}")
    print(f"Mean BalAcc: {np.mean(fold_bal_accs):.4f}")
    print(f"Max Fold AUC: {np.max(fold_aucs):.4f}")

if __name__ == "__main__":
    run_10fold_cv()