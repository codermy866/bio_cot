"""
自适应模态权重学习
"""

import torch
import torch.nn as nn


class QualityAssessor(nn.Module):
    """
    特征质量评估器
    评估模态特征的质量（清晰度、信息量等）
    """
    def __init__(self, embed_dim=768):
        super().__init__()
        self.assessor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, feat):
        return self.assessor(feat)


class AdaptiveModalityWeighting(nn.Module):
    """
    自适应模态权重学习
    根据特征质量动态调整各模态的权重
    """
    def __init__(self, embed_dim=768, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.quality_assessors = nn.ModuleDict({
            'local': QualityAssessor(embed_dim),
            'global': QualityAssessor(embed_dim),
            'semantic': QualityAssessor(embed_dim)
        })
        self.weight_generator = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, feat_local, feat_global, feat_semantic):
        quality_local = self.quality_assessors['local'](feat_local)
        quality_global = self.quality_assessors['global'](feat_global)
        quality_semantic = self.quality_assessors['semantic'](feat_semantic)

        quality_feat = torch.cat([
            feat_local * quality_local,
            feat_global * quality_global,
            feat_semantic * quality_semantic
        ], dim=-1)

        weights = self.weight_generator(quality_feat)
        fused = (
            weights[:, 0:1] * feat_local +
            weights[:, 1:2] * feat_global +
            weights[:, 2:3] * feat_semantic
        )
        return fused, weights
