#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视觉笔记生成模块 (Visual Notes Generation)
借鉴NoteMR方法，使用知识笔记作为Query，对图像特征进行显式掩码
使用Soft Masking（便于反向传播）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class VisualNoteLayer(nn.Module):
    """
    视觉笔记层（按照用户提供的方案实现）
    使用Soft Masking，便于反向传播
    """
    
    def __init__(self, img_dim: int = 768, text_dim: int = 768, hidden_dim: int = 256):
        """
        Args:
            img_dim: ViT输出的特征维度
            text_dim: Knowledge Note Embedding的维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        # 1. 投影层：将图像和文本映射到同一空间计算Attention
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # 2. 门控机制（Sigmoid，用于Soft Attention）
        self.sigmoid = nn.Sigmoid()
    
    def forward(
        self, 
        img_feats: torch.Tensor,  # [B, N, D] (Batch, Patch数量, 维度)
        text_feats: torch.Tensor,  # [B, D] (Batch, 维度)
        beta: float = 0.1  # 背景抑制系数
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成视觉笔记特征
        
        Args:
            img_feats: [B, N, D] 图像Patch特征
            text_feats: [B, D] Knowledge Note特征
            beta: 背景抑制系数 (0.0 = 完全删除背景, 1.0 = 保留原图)
        
        Returns:
            img_focused: [B, N, D] 过滤后的图像特征
            attn_map: [B, N, 1] 注意力热图
        """
        B, N, D = img_feats.shape
        
        # --- Step 1: 计算相关性热图 (Attention Map) ---
        # Q: Text, K: Image
        # [B, D] -> [B, 1, H]
        q = self.text_proj(text_feats).unsqueeze(1)  # [B, 1, hidden_dim]
        # [B, N, D] -> [B, N, H]
        k = self.img_proj(img_feats)  # [B, N, hidden_dim]
        
        # Dot Product Attention: [B, N, H] * [B, H, 1] -> [B, N, 1]
        attn_logits = torch.matmul(k, q.transpose(1, 2))  # [B, N, 1]
        attn_logits = attn_logits / (k.shape[-1] ** 0.5)  # Scale
        
        # 归一化到0-1之间，作为Mask的概率（Soft Masking）
        attn_map = self.sigmoid(attn_logits)  # Shape: [B, N, 1]
        
        # 🔧 修复：添加下界保护，防止注意力完全坍塌为0
        # 🎯 AUC提升改进：提高注意力下界，减少过度抑制
        # 从0.01提高到0.05，保留更多有用信息
        min_attn = 0.05  # 最小注意力值（5%），确保不会完全为0，同时减少过度抑制
        attn_map = torch.clamp(attn_map, min=min_attn, max=1.0)  # 硬下界
        
        # --- Step 2: 应用Visual Note（Soft Masking）---
        # 核心公式: F_note = F * Mask + F * (1 - Mask) * beta
        # 高响应区域保留 (x 1.0)，低响应区域被抑制 (x beta)
        mask_weight = attn_map + (1 - attn_map) * beta  # [B, N, 1]
        
        img_focused = img_feats * mask_weight  # [B, N, D]
        
        return img_focused, attn_map


class VisualNotesModule(nn.Module):
    """
    视觉笔记模块（整合所有功能）
    支持Warm-up策略和动态Beta
    """
    
    def __init__(
        self,
        img_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 256,
        warmup_epochs: int = 5
    ):
        """
        Args:
            img_dim: 图像特征维度
            text_dim: 文本特征维度
            hidden_dim: 隐藏层维度
            warmup_epochs: Warm-up轮数
        """
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
        # Visual Note Layer
        self.visual_note_layer = VisualNoteLayer(
            img_dim=img_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim
        )
    
    def set_epoch(self, epoch: int):
        """设置当前epoch（用于Warm-up）"""
        self.current_epoch = epoch
    
    def get_beta(self) -> float:
        """
        获取当前epoch的背景抑制系数（Warm-up策略）
        
        🎯 AUC提升改进：更温和的Beta策略，减少过度抑制
        动态Beta策略：
        - Epoch 0-10: beta=1.0 (全图保留，不进行过滤)
        - Epoch 10-30: beta线性递减 1.0 -> 0.3
        - Epoch 30+: beta=0.3 (温和过滤模式，从0.1提高到0.3)
        """
        if self.current_epoch < 10:
            return 1.0
        elif self.current_epoch < 30:
            # 线性衰减: 1.0 -> 0.3（更温和）
            progress = (self.current_epoch - 10) / (30 - 10)
            return 1.0 - (0.7 * progress)  # 1.0 -> 0.3
        else:
            return 0.3  # 温和过滤模式（从0.1提高到0.3，减少过度抑制）
    
    def forward(
        self,
        img_features: torch.Tensor,  # [B, N, D]
        text_features: torch.Tensor,  # [B, D]
        beta: Optional[float] = None  # 如果提供，使用提供的beta；否则使用动态beta
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成视觉笔记特征（支持Warm-up）
        
        Args:
            img_features: [B, N, D] 图像Patch特征
            text_features: [B, D] Knowledge Note特征
            beta: 背景抑制系数（如果None，使用动态beta）
        
        Returns:
            img_focused: [B, N, D] 过滤后的图像特征
            attn_map: [B, N, 1] 注意力热图
        """
        if beta is None:
            beta = self.get_beta()
        
        # 生成视觉笔记
        img_focused, attn_map = self.visual_note_layer(
            img_features, text_features, beta=beta
        )
        
        return img_focused, attn_map

