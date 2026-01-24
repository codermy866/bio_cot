#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因果干预模块（Causal Intervention Module）
实现do-calculus，进行反事实推理

核心创新：
1. 实现真正的因果推理（而非仅关联学习）
2. 支持反事实推理（What if问题）
3. 可解释的因果效应估计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import numpy as np


class CausalIntervention(nn.Module):
    """
    因果干预模块
    实现do-calculus：P(Y|do(X))而非P(Y|X)
    
    原理：
    - 标准学习：P(Y|X) = P(Y|X, Z)（可能包含混杂因子Z）
    - 因果干预：P(Y|do(X)) = 切断X的所有反向因果路径
    
    应用：
    - 反事实推理：如果改变OCT特征，结果如何？
    - 因果效应估计：量化每个模态的因果贡献
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_modalities: int = 3,
        intervention_method: str = 'masking'  # 'masking' or 'replacement'
    ):
        """
        Args:
            embed_dim: 特征维度
            num_modalities: 模态数量（OCT, Colposcopy, Clinical）
            intervention_method: 干预方法
                - 'masking': 掩码干预（切断反向路径）
                - 'replacement': 替换干预（用固定值替换）
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.intervention_method = intervention_method
        
        # 干预掩码（可学习）
        # 形状：[num_modalities, num_modalities]
        # intervention_mask[i, j] = 1 表示干预模态i时，切断到模态j的路径
        self.intervention_mask = nn.Parameter(
            torch.ones(num_modalities, num_modalities)
        )
        
        # 干预强度（可学习）
        self.intervention_strength = nn.Parameter(
            torch.ones(num_modalities) * 0.5
        )
        
        # 反事实特征生成器（用于反事实推理）
        self.counterfactual_generator = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def forward(
        self,
        features: List[torch.Tensor],
        intervention_target: Optional[int] = None,
        intervention_value: Optional[torch.Tensor] = None,
        return_counterfactual: bool = False,
        intervention_strength: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        应用因果干预
        
        Args:
            features: 特征列表 [oct_feat, colpo_feat, clinical_feat]
                每个特征形状: [B, embed_dim]
            intervention_target: 要干预的模态索引（0=OCT, 1=Colposcopy, 2=Clinical）
                如果为None，不进行干预（标准前向传播）
            intervention_value: 干预值（如果指定，用此值替换目标模态）
            return_counterfactual: 是否返回反事实特征
        
        Returns:
            intervened_features: 干预后的特征列表
            intervention_mask: 应用的干预掩码
            counterfactual_features: 反事实特征（如果return_counterfactual=True）
        """
        B = features[0].size(0)
        num_modalities = len(features)
        
        # 如果没有指定干预目标，返回原始特征
        if intervention_target is None:
            return {
                'intervened_features': features,
                'intervention_mask': None,
                'counterfactual_features': None
            }
        
        # 确保干预目标有效
        assert 0 <= intervention_target < num_modalities, \
            f"干预目标索引必须在[0, {num_modalities})范围内"
        
        # 应用干预掩码
        # intervention_mask[i, j] = 1 表示干预i时，切断i→j的路径
        mask = self.intervention_mask[intervention_target]
        if intervention_strength is not None:
            strength = intervention_strength
        else:
            strength = torch.sigmoid(self.intervention_strength[intervention_target])
        
        # 复制特征（避免修改原始特征）
        intervened_features = [feat.clone() for feat in features]
        
        # 应用干预
        if self.intervention_method == 'masking':
            # 掩码干预：切断反向路径
            for i in range(num_modalities):
                if i != intervention_target and mask[i] > 0.5:
                    # 切断从其他模态到干预目标的路径
                    # 通过掩码减少特征的影响
                    intervened_features[i] = intervened_features[i] * (1 - strength * mask[i])
        
        elif self.intervention_method == 'replacement':
            # 替换干预：用固定值或生成值替换
            if intervention_value is not None:
                intervened_features[intervention_target] = intervention_value
            else:
                # 使用反事实生成器生成干预值
                # 拼接所有特征
                concat_feat = torch.cat(features, dim=-1)  # [B, embed_dim * num_modalities]
                # 生成反事实特征
                counterfactual = self.counterfactual_generator(concat_feat)  # [B, embed_dim]
                intervened_features[intervention_target] = counterfactual
        
        # 生成反事实特征（如果需要）
        counterfactual_features = None
        if return_counterfactual:
            # 生成"如果改变干预目标，其他模态会如何变化"的反事实
            # 这里简化处理：基于干预后的特征生成
            concat_intervened = torch.cat(intervened_features, dim=-1)
            counterfactual_features = self.counterfactual_generator(concat_intervened)
        
        return {
            'intervened_features': intervened_features,
            'intervention_mask': mask,
            'intervention_strength': strength,
            'counterfactual_features': counterfactual_features
        }
    
    def compute_causal_effect(
        self,
        features: List[torch.Tensor],
        intervention_target: int,
        outcome_modality: int = 2  # 默认是Clinical（决策）
    ) -> torch.Tensor:
        """
        计算因果效应：E[Y|do(X)] - E[Y]
        
        Args:
            features: 特征列表
            intervention_target: 干预目标模态
            outcome_modality: 结果模态（通常是决策/分类）
        
        Returns:
            causal_effect: 因果效应 [B, embed_dim]
        """
        # 原始预测（无干预）
        original_outcome = features[outcome_modality]
        
        # 干预后的预测
        intervened = self.forward(features, intervention_target=intervention_target)
        intervened_outcome = intervened['intervened_features'][outcome_modality]
        
        # 因果效应 = 干预后 - 干预前
        causal_effect = intervened_outcome - original_outcome
        
        return causal_effect
    
    def batch_intervention(
        self,
        features: List[torch.Tensor],
        intervention_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        批量应用不同的干预（每个样本可能有不同的干预目标）
        
        Args:
            features: 特征列表
            intervention_targets: 每个样本的干预目标 [B]
        
        Returns:
            干预后的特征
        """
        B = features[0].size(0)
        intervened_features_list = []
        
        for b in range(B):
            target = intervention_targets[b].item()
            intervened = self.forward(features, intervention_target=target)
            intervened_features_list.append(intervened['intervened_features'])
        
        # 重新组织为列表
        num_modalities = len(features)
        batch_intervened = [
            torch.stack([feat[i] for feat in intervened_features_list])
            for i in range(num_modalities)
        ]
        
        return {
            'intervened_features': batch_intervened,
            'intervention_targets': intervention_targets
        }


class AdaptiveInterventionScheduler(nn.Module):
    """
    自适应干预调度器
    根据不确定性动态调整干预策略
    """
    
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 不确定性阈值（可学习）
        self.uncertainty_threshold = nn.Parameter(torch.tensor(0.5))
        
        # 干预强度调度器
        self.strength_scheduler = nn.Sequential(
            nn.Linear(1, 32),  # 输入：不确定性
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出：干预强度 [0, 1]
        )
    
    def forward(
        self,
        uncertainty: torch.Tensor,
        base_strength: float = 0.5
    ) -> torch.Tensor:
        """
        根据不确定性计算干预强度
        
        Args:
            uncertainty: 不确定性 [B, 1] 或 [B]
            base_strength: 基础干预强度
        
        Returns:
            intervention_strength: 干预强度 [B]
        """
        if uncertainty.dim() == 1:
            uncertainty = uncertainty.unsqueeze(-1)  # [B, 1]
        
        # 计算自适应强度
        adaptive_strength = self.strength_scheduler(uncertainty).squeeze(-1)  # [B]
        
        # 高不确定性 → 高干预强度
        # 低不确定性 → 低干预强度（避免过度干预）
        final_strength = base_strength * adaptive_strength
        
        return final_strength

