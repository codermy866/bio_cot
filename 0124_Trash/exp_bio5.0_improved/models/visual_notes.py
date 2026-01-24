#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
复用 exp_bio5.0 的 Visual Notes（含 SCG + warmup beta）。

为了让 exp_bio5.0_improved 目录独立可运行，这里直接拷贝一份实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class VisualNoteLayer(nn.Module):
    def __init__(self, img_dim: int = 768, text_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        img_feats: torch.Tensor,  # [B, N, D]
        text_feats: torch.Tensor,  # [B, D]
        beta=0.1,  # float or Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.text_proj(text_feats).unsqueeze(1)  # [B,1,H]
        k = self.img_proj(img_feats)  # [B,N,H]
        attn_logits = torch.matmul(k, q.transpose(1, 2)) / (k.shape[-1] ** 0.5)  # [B,N,1]
        attn_map = self.sigmoid(attn_logits)

        # 下界保护：避免注意力塌陷
        attn_map = torch.clamp(attn_map, min=0.05, max=1.0)

        if isinstance(beta, torch.Tensor):
            beta = beta.to(img_feats.device)
            if beta.dim() == 1:
                beta = beta.view(-1, 1, 1)
            elif beta.dim() == 2:
                beta = beta.unsqueeze(-1)
        else:
            beta = torch.tensor(beta, device=img_feats.device, dtype=img_feats.dtype)

        mask_weight = attn_map + (1 - attn_map) * beta  # [B,N,1]
        img_focused = img_feats * mask_weight
        return img_focused, attn_map


class VisualNotesModule(nn.Module):
    def __init__(
        self,
        img_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 256,
        warmup_epochs: int = 10,
    ):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.visual_note_layer = VisualNoteLayer(img_dim=img_dim, text_dim=text_dim, hidden_dim=hidden_dim)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def get_beta(self) -> float:
        if self.current_epoch < 10:
            return 1.0
        if self.current_epoch < 30:
            progress = (self.current_epoch - 10) / (30 - 10)
            return 1.0 - (0.7 * progress)  # 1.0 -> 0.3
        return 0.3

    def forward(
        self,
        img_features: torch.Tensor,  # [B,N,D]
        text_features: torch.Tensor,  # [B,D]
        beta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base_beta = self.get_beta() if beta is None else beta
        img_global = img_features.mean(dim=1)  # [B,D]
        consistency = F.cosine_similarity(img_global, text_features, dim=1)
        gate = torch.sigmoid(consistency * 5)

        if isinstance(base_beta, (int, float)):
            base_beta_tensor = torch.full(
                (img_features.shape[0], 1, 1),
                float(base_beta),
                device=img_features.device,
                dtype=img_features.dtype,
            )
        else:
            base_beta_tensor = base_beta

        dynamic_beta = gate.view(-1, 1, 1) * base_beta_tensor + (1 - gate.view(-1, 1, 1)) * 1.0
        img_focused, attn_map = self.visual_note_layer(img_features, text_features, beta=dynamic_beta)
        return img_focused, attn_map


