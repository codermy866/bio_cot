"""
对比学习增强的跨模态对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveAlignmentModule(nn.Module):
    def __init__(self, embed_dim=768, temperature=0.2, dropout=0.1, use_supervised=True):
        super().__init__()
        self.temperature = temperature
        self.use_supervised = use_supervised
        self.projections = nn.ModuleDict({
            'oct': nn.Linear(embed_dim, embed_dim),
            'col': nn.Linear(embed_dim, embed_dim),
            'clinical': nn.Linear(embed_dim, embed_dim)
        })
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, oct_feat, col_feat, clinical_feat, labels=None):
        oct_proj = F.normalize(self.head(self.projections['oct'](oct_feat)), dim=-1)
        col_proj = F.normalize(self.head(self.projections['col'](col_feat)), dim=-1)
        clinical_proj = F.normalize(self.head(self.projections['clinical'](clinical_feat)), dim=-1)
        loss = None
        if labels is not None:
            if self.use_supervised:
                loss = self.supervised_contrastive_loss(oct_proj, col_proj, labels)
                loss += self.supervised_contrastive_loss(oct_proj, clinical_proj, labels)
                loss += self.supervised_contrastive_loss(col_proj, clinical_proj, labels)
            else:
                loss = self.compute_contrastive_loss(oct_proj, col_proj)
                loss += self.compute_contrastive_loss(oct_proj, clinical_proj)
                loss += self.compute_contrastive_loss(col_proj, clinical_proj)
        aligned = torch.stack([oct_proj, col_proj, clinical_proj], dim=1)
        return aligned, loss

    def compute_contrastive_loss(self, feat_a, feat_b):
        logits = torch.matmul(feat_a, feat_b.t()) / self.temperature
        labels = torch.arange(feat_a.size(0), device=feat_a.device)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)
        return (loss_a + loss_b) / 2

    def supervised_contrastive_loss(self, feat_a, feat_b, labels):
        """
        监督对比学习损失
        对于batch size = 1的情况，返回一个小的正则化项
        """
        batch_size = feat_a.size(0)
        if batch_size == 1:
            # 对于batch size = 1，计算特征之间的相似度作为正则化项
            similarity = torch.dot(feat_a[0], feat_b[0])
            # 返回一个小的损失，鼓励特征对齐
            return 0.1 * (1.0 - similarity)
        
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(feat_a.device)
        logits = torch.matmul(feat_a, feat_b.t()) / self.temperature
        
        # 数值稳定性：减去最大值
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()
        
        exp_logits = torch.exp(logits)
        positives = exp_logits * mask
        
        # 排除自己（对角线）
        mask_no_self = mask.clone()
        mask_no_self.fill_diagonal_(0)
        positives = positives * mask_no_self
        
        pos_sum = positives.sum(dim=1)
        all_sum = exp_logits.sum(dim=1) + 1e-8
        
        # 避免log(0)
        ratio = (pos_sum + 1e-8) / all_sum
        loss = -torch.log(ratio + 1e-8)
        
        # 只对至少有一个正样本的样本计算损失
        valid_mask = (pos_sum > 0).float()
        if valid_mask.sum() > 0:
            loss = (loss * valid_mask).sum() / valid_mask.sum()
        else:
            # 如果没有正样本对，返回一个小的正则化项
            loss = 0.1 * torch.mean(1.0 - torch.sum(feat_a * feat_b, dim=1))
        
        return loss
