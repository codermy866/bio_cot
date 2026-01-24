"""
跨模态多粒度对齐模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAligner(nn.Module):
    """
    跨模态对齐器
    使用交叉注意力将不同模态的特征对齐到统一空间
    支持多粒度对齐（如局部-局部、全局-全局、语义-视觉等）
    """

    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        """
        Args:
            feat_a: [B, embed_dim] 第一个模态特征（作为Query）
            feat_b: [B, embed_dim] 第二个模态特征（作为Key/Value）
        Returns:
            aligned_feat: [B, embed_dim] 对齐后的特征
        """
        # 添加长度维度供注意力使用
        feat_a = feat_a.unsqueeze(1)  # [B, 1, embed_dim]
        feat_b = feat_b.unsqueeze(1)  # [B, 1, embed_dim]

        # 层归一化
        q = self.norm_q(feat_a)
        kv = self.norm_kv(feat_b)

        aligned_feat, _ = self.cross_attention(q, kv, kv)  # [B, 1, embed_dim]
        aligned_feat = aligned_feat.squeeze(1)
        aligned_feat = self.projection(aligned_feat)
        return aligned_feat
