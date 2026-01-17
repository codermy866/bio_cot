"""
层次化分类器
"""

import torch
import torch.nn as nn


class HierarchicalClassifier(nn.Module):
    def __init__(self, embed_dim=768, num_classes=2, dropout=0.2):
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
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, fine_feat, mid_feat, coarse_feat):
        fused = torch.cat([fine_feat, mid_feat, coarse_feat], dim=-1)
        fused = self.fusion(fused)
        logits = self.classifier(fused)
        return logits
