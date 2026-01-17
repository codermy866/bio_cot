#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分层对比学习模块（Hierarchical Contrastive Learning Module）
多层次跨模态对齐

核心创新：
1. 多层次对比：特征级→语义级→决策级
2. 渐进式学习：从粗到细
3. 更丰富的跨模态对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np


class HierarchicalContrastiveLearning(nn.Module):
    """
    分层对比学习模块
    在多个层次进行跨模态对比学习
    
    层次：
    1. 特征级（Feature Level）：原始特征空间的对比
    2. 语义级（Semantic Level）：高级语义表示的对比
    3. 决策级（Decision Level）：预测结果的对比
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_modalities: int = 3,
        temperature: float = 0.07,
        use_hard_negatives: bool = True
    ):
        """
        Args:
            embed_dim: 特征维度
            num_modalities: 模态数量
            temperature: 对比学习温度参数
            use_hard_negatives: 是否使用困难负样本
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        
        # 特征级投影（保持原始特征）
        self.feature_projectors = nn.ModuleList([
            nn.Identity() for _ in range(num_modalities)
        ])
        
        # 语义级投影（提取高级语义）
        self.semantic_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.LayerNorm(embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, embed_dim)
            ) for _ in range(num_modalities)
        ])
        
        # 决策级投影（预测结果空间）
        self.decision_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.LayerNorm(embed_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim // 2, embed_dim // 4)
            ) for _ in range(num_modalities)
        ])
        
        # 权重（平衡不同层次的损失）
        self.level_weights = nn.Parameter(torch.ones(3))  # [feature, semantic, decision]
    
    def forward(
        self,
        features: List[torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        分层对比学习
        
        Args:
            features: 特征列表 [oct_feat, colpo_feat, clinical_feat]
                每个特征形状: [B, embed_dim]
            labels: 标签 [B]（用于决策级对比）
        
        Returns:
            contrastive_losses: 各层次的对比损失
            total_loss: 总损失
        """
        B = features[0].size(0)
        
        # 1. 特征级对比
        feature_loss = self._feature_level_contrast(features)
        
        # 2. 语义级对比
        semantic_loss = self._semantic_level_contrast(features)
        
        # 3. 决策级对比（如果有标签）
        decision_loss = None
        if labels is not None:
            decision_loss = self._decision_level_contrast(features, labels)
        
        # 加权组合
        weights = F.softmax(self.level_weights, dim=0)
        total_loss = (
            weights[0] * feature_loss +
            weights[1] * semantic_loss +
            (weights[2] * decision_loss if decision_loss is not None else 0)
        )
        
        return {
            'feature_level_loss': feature_loss,
            'semantic_level_loss': semantic_loss,
            'decision_level_loss': decision_loss,
            'total_contrastive_loss': total_loss,
            'level_weights': weights
        }
    
    def _feature_level_contrast(
        self,
        features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        特征级对比学习
        在原始特征空间进行对比
        """
        # 投影到对比空间
        projected_features = [
            self.feature_projectors[i](feat)
            for i, feat in enumerate(features)
        ]
        
        # 计算InfoNCE损失（所有模态对之间）
        total_loss = 0.0
        num_pairs = 0
        
        for i in range(self.num_modalities):
            for j in range(i + 1, self.num_modalities):
                loss = self._info_nce(
                    projected_features[i],
                    projected_features[j]
                )
                total_loss += loss
                num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else total_loss
    
    def _semantic_level_contrast(
        self,
        features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        语义级对比学习
        在高级语义空间进行对比
        """
        # 投影到语义空间
        semantic_features = [
            self.semantic_projectors[i](feat)
            for i, feat in enumerate(features)
        ]
        
        # 计算InfoNCE损失
        total_loss = 0.0
        num_pairs = 0
        
        for i in range(self.num_modalities):
            for j in range(i + 1, self.num_modalities):
                loss = self._info_nce(
                    semantic_features[i],
                    semantic_features[j]
                )
                total_loss += loss
                num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else total_loss
    
    def _decision_level_contrast(
        self,
        features: List[torch.Tensor],
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        决策级对比学习
        在预测结果空间进行对比
        """
        # 投影到决策空间
        decision_features = [
            self.decision_projectors[i](feat)
            for i, feat in enumerate(features)
        ]
        
        # 计算决策级对比损失
        # 相同标签的样本应该相似，不同标签的样本应该不同
        B = features[0].size(0)
        total_loss = 0.0
        
        # 对于每个模态对
        for i in range(self.num_modalities):
            for j in range(i + 1, self.num_modalities):
                feat_i = decision_features[i]  # [B, embed_dim // 4]
                feat_j = decision_features[j]  # [B, embed_dim // 4]
                
                # 计算相似度矩阵
                similarity = torch.matmul(feat_i, feat_j.t()) / self.temperature  # [B, B]
                
                # 标签匹配矩阵
                label_match = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [B, B]
                
                # 正样本：相同标签
                # 负样本：不同标签
                pos_mask = label_match
                neg_mask = 1 - label_match
                
                # 计算对比损失
                exp_sim = torch.exp(similarity)
                pos_exp = (exp_sim * pos_mask).sum(dim=1, keepdim=True)  # [B, 1]
                neg_exp = (exp_sim * neg_mask).sum(dim=1, keepdim=True)  # [B, 1]
                
                loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8))
                total_loss += loss.mean()
        
        num_pairs = self.num_modalities * (self.num_modalities - 1) // 2
        return total_loss / num_pairs if num_pairs > 0 else total_loss
    
    def _info_nce(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE损失（对比学习标准损失）
        
        Args:
            feat1: 特征1 [B, D]
            feat2: 特征2 [B, D]
        
        Returns:
            loss: InfoNCE损失
        """
        B = feat1.size(0)
        
        # 检查输入有效性
        if B == 0:
            return torch.tensor(0.0, device=feat1.device, requires_grad=True)
        
        # L2归一化
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)
        
        # 检查NaN/Inf
        if torch.isnan(feat1).any() or torch.isnan(feat2).any():
            return torch.tensor(0.0, device=feat1.device, requires_grad=True)
        
        # 计算相似度矩阵
        similarity = torch.matmul(feat1, feat2.t()) / self.temperature  # [B, B]
        
        # 检查相似度矩阵
        if torch.isnan(similarity).any() or torch.isinf(similarity).any():
            return torch.tensor(0.0, device=feat1.device, requires_grad=True)
        
        # 对角线是正样本（同一样本的不同模态）
        labels = torch.arange(B, device=feat1.device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(similarity, labels)
        
        # 检查损失
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=feat1.device, requires_grad=True)
        
        return loss
    
    def get_contrastive_features(
        self,
        features: List[torch.Tensor],
        level: str = 'semantic'  # 'feature', 'semantic', 'decision'
    ) -> List[torch.Tensor]:
        """
        获取指定层次的对比特征
        
        Args:
            features: 原始特征列表
            level: 层次（'feature', 'semantic', 'decision'）
        
        Returns:
            contrastive_features: 对比特征列表
        """
        if level == 'feature':
            return [self.feature_projectors[i](feat) for i, feat in enumerate(features)]
        elif level == 'semantic':
            return [self.semantic_projectors[i](feat) for i, feat in enumerate(features)]
        elif level == 'decision':
            return [self.decision_projectors[i](feat) for i, feat in enumerate(features)]
        else:
            raise ValueError(f"Unknown level: {level}")

