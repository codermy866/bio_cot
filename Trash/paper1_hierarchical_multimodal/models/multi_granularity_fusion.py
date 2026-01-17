"""
多粒度融合模块
"""

import torch
import torch.nn as nn


class FineGrainFusion(nn.Module):
    """
    细粒度融合（局部特征）
    强调局部纹理、边缘等细节信息
    """
    def __init__(self, embed_dim=768, dropout=0.1):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, feat_a, feat_b):
        fused = torch.cat([feat_a, feat_b], dim=-1)
        return self.fusion(fused)


class MidGrainFusion(nn.Module):
    """
    中粒度融合（全局特征）
    强调整体语义和全局上下文
    """
    def __init__(self, embed_dim=768, dropout=0.1):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, feat_a, feat_b):
        fused = torch.cat([feat_a, feat_b], dim=-1)
        return self.fusion(fused)


class CoarseGrainFusion(nn.Module):
    """
    粗粒度融合（语义特征）
    融合临床语义与图像语义
    """
    def __init__(self, embed_dim=768, dropout=0.1):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, feat_a, feat_b, feat_c):
        fused = torch.cat([feat_a, feat_b, feat_c], dim=-1)
        return self.fusion(fused)
