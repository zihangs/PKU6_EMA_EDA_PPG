import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1,L,D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PPIBertEncoder(nn.Module):
    """
    BERT-like encoder for window-vectors:
      - project each window to d_model
      - prepend learned [CLS]
      - TransformerEncoder (batch_first)
    """

    def __init__(self, input_dim: int, d_model: int = 256, n_layers: int = 4, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: [B,T,input_dim]
        B = x.size(0)
        x = self.proj(x)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B,T+1,D]
        x = self.pos(x)

        if key_padding_mask is not None:
            cls_pad = torch.zeros((key_padding_mask.size(0), 1), dtype=torch.bool, device=key_padding_mask.device)
            key_padding_mask = torch.cat([cls_pad, key_padding_mask], dim=1)

        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return out[:, 0]


def masked_mean(x: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
    # x: [B,T,D], pad_mask: [B,T] True=pad
    if pad_mask is None:
        return x.mean(dim=1)
    keep = (~pad_mask).unsqueeze(-1)
    denom = keep.sum(dim=1).clamp_min(1)
    return (x * keep).sum(dim=1) / denom


class PPIBertMLPFusion(nn.Module):
    """
    Inputs:
      - raw_windows:  [B,T,raw_len]
      - win_feats:    [B,T,win_feat_dim]
      - mask:         [B,T] True=pad
      - global_feats: [B,global_feat_dim]
    """

    def __init__(self, raw_len: int, win_feat_dim: int, global_feat_dim: int, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.raw_enc = PPIBertEncoder(input_dim=raw_len, d_model=d_model, dropout=dropout)
        self.win_proj = nn.Sequential(
            nn.Linear(win_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, raw_windows: torch.Tensor, win_feats: torch.Tensor, mask: Optional[torch.Tensor], global_feats: torch.Tensor) -> torch.Tensor:
        raw_cls = self.raw_enc(raw_windows, key_padding_mask=mask)  # [B,D]
        win_emb = self.win_proj(win_feats)  # [B,T,D]
        win_pool = masked_mean(win_emb, mask)  # [B,D]
        g = self.global_proj(global_feats)  # [B,D]
        h = self.fuse(torch.cat([raw_cls, win_pool, g], dim=-1))
        return self.classifier(h)

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1)]


class PPIBertEncoder(nn.Module):
    """
    BERT-like encoder for PPI windows:
      - projects each window vector to d_model
      - prepends a learned [CLS] token
      - TransformerEncoder (batch_first)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        x = self.proj(x)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, T+1, D]
        x = self.pos(x)

        if key_padding_mask is not None:
            # mask: True means pad; we added cls token at position 0 => not padded
            cls_pad = torch.zeros((key_padding_mask.size(0), 1), dtype=torch.bool, device=key_padding_mask.device)
            key_padding_mask = torch.cat([cls_pad, key_padding_mask], dim=1)

        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return out[:, 0]  # CLS [B, D]


def masked_mean(x: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
    # x: [B, T, D], pad_mask: [B, T] True=pad
    if pad_mask is None:
        return x.mean(dim=1)
    keep = (~pad_mask).unsqueeze(-1)  # [B,T,1]
    denom = keep.sum(dim=1).clamp_min(1)
    return (x * keep).sum(dim=1) / denom


class PPIBertMLPFusion(nn.Module):
    """
    Inputs:
      - raw_ppi_windows: [B, T, raw_len]
      - win_feats:       [B, T, win_feat_dim]
      - mask:            [B, T] True=pad
      - global_feats:    [B, global_feat_dim]
    """

    def __init__(
        self,
        raw_len: int,
        win_feat_dim: int,
        global_feat_dim: int,
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.raw_enc = PPIBertEncoder(input_dim=raw_len, d_model=d_model, dropout=dropout)
        self.win_proj = nn.Sequential(
            nn.Linear(win_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        self.fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(
        self,
        raw_ppi_windows: torch.Tensor,
        win_feats: torch.Tensor,
        mask: Optional[torch.Tensor],
        global_feats: torch.Tensor,
    ) -> torch.Tensor:
        raw_cls = self.raw_enc(raw_ppi_windows, key_padding_mask=mask)  # [B, D]
        win_emb = self.win_proj(win_feats)  # [B,T,D]
        win_pool = masked_mean(win_emb, mask)  # [B,D]
        g = self.global_proj(global_feats)  # [B,D]
        h = self.fuse(torch.cat([raw_cls, win_pool, g], dim=-1))
        return self.classifier(h)

