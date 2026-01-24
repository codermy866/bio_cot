#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应因果图模块（Adaptive Causal Graph Module）
根据输入动态调整因果结构

核心创新：
1. 个性化因果推理（每个患者可能有不同的因果结构）
2. 结合数据驱动和领域知识
3. 动态因果图发现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import numpy as np


class AdaptiveCausalGraph(nn.Module):
    """
    自适应因果图模块
    根据输入特征动态学习因果结构
    
    原理：
    - 标准方法：固定因果图（所有患者相同）
    - 自适应方法：每个患者可能有不同的因果结构
    - 结合医学先验：确保发现的因果图符合医学知识
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_modalities: int = 3,
        prior_knowledge: Optional[torch.Tensor] = None,
        use_prior_constraint: bool = True,
        temperature: float = 1.0
    ):
        """
        Args:
            embed_dim: 特征维度
            num_modalities: 模态数量
            prior_knowledge: 先验知识矩阵 [num_modalities, num_modalities]
                1表示必须存在的因果关系，0表示禁止，-1表示未知
            use_prior_constraint: 是否使用先验约束
            temperature: Gumbel-Softmax温度（用于可微采样）
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.use_prior_constraint = use_prior_constraint
        self.temperature = temperature
        
        # 先验知识（硬约束）
        if prior_knowledge is not None:
            self.register_buffer('prior_knowledge', prior_knowledge)
        else:
            # 默认先验：Clinical影响OCT和Colposcopy
            prior = torch.zeros(num_modalities, num_modalities)
            prior[0, 2] = 1  # OCT <- Clinical
            prior[1, 2] = 1  # Colposcopy <- Clinical
            self.register_buffer('prior_knowledge', prior)
        
        # 因果图生成网络（基于输入特征）
        # 输入：所有模态的特征拼接
        # 输出：因果邻接矩阵
        self.causal_graph_net = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_modalities * num_modalities)
        )
        
        # 可学习的因果权重（全局先验）
        self.global_causal_weights = nn.Parameter(
            torch.randn(num_modalities, num_modalities) * 0.1
        )
    
    def forward(
        self,
        features: List[torch.Tensor],
        return_adjacency: bool = True,
        use_gumbel_softmax: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        根据输入特征生成自适应因果图
        
        Args:
            features: 特征列表 [oct_feat, colpo_feat, clinical_feat]
                每个特征形状: [B, embed_dim]
            return_adjacency: 是否返回邻接矩阵
            use_gumbel_softmax: 是否使用Gumbel-Softmax（可微采样）
        
        Returns:
            causal_graph: 因果图字典
                - adjacency: 因果邻接矩阵 [B, num_modalities, num_modalities]
                - weights: 因果权重 [B, num_modalities, num_modalities]
                - prior_constrained: 应用先验约束后的图
        """
        B = features[0].size(0)
        
        # 拼接所有特征
        concat_feat = torch.cat(features, dim=-1)  # [B, embed_dim * num_modalities]
        
        # 生成因果图（数据驱动）
        learned_graph = self.causal_graph_net(concat_feat)  # [B, num_modalities * num_modalities]
        learned_graph = learned_graph.view(B, self.num_modalities, self.num_modalities)
        
        # 应用全局权重
        learned_graph = learned_graph + self.global_causal_weights.unsqueeze(0)
        
        # 应用先验约束
        if self.use_prior_constraint:
            prior_constrained = self._apply_prior_constraints(learned_graph)
        else:
            prior_constrained = learned_graph
        
        # 二值化（如果使用Gumbel-Softmax，保持连续；否则二值化）
        if use_gumbel_softmax:
            # 使用Gumbel-Softmax进行可微采样
            adjacency = self._gumbel_softmax(prior_constrained)
        else:
            # 直接sigmoid + 阈值
            adjacency = torch.sigmoid(prior_constrained)
            # 应用DAG约束（上三角矩阵）
            adjacency = self._enforce_dag(adjacency)
        
        result = {
            'adjacency': adjacency,
            'weights': prior_constrained,
            'prior_constrained': prior_constrained,
            'learned_graph': learned_graph
        }
        
        return result
    
    def _apply_prior_constraints(
        self,
        learned_graph: torch.Tensor
    ) -> torch.Tensor:
        """
        应用先验知识约束
        
        Args:
            learned_graph: 学习到的因果图 [B, num_modalities, num_modalities]
        
        Returns:
            constrained_graph: 应用约束后的图
        """
        B = learned_graph.size(0)
        prior = self.prior_knowledge.unsqueeze(0).expand(B, -1, -1)
        
        # 先验=1：必须存在（强制设为1）
        # 先验=0：禁止存在（强制设为0）
        # 先验=-1：未知（保持学习值）
        mask_must = (prior == 1).float()
        mask_forbid = (prior == 0).float()
        mask_unknown = (prior == -1).float()
        
        # 应用约束
        constrained = learned_graph.clone()
        constrained = constrained * mask_unknown  # 未知部分保持学习值
        constrained = constrained + mask_must  # 必须存在的设为1
        constrained = constrained * (1 - mask_forbid)  # 禁止存在的设为0
        
        return constrained
    
    def _enforce_dag(
        self,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        确保因果图是DAG（有向无环图）
        
        Args:
            adjacency: 因果邻接矩阵 [B, num_modalities, num_modalities]
        
        Returns:
            dag_adjacency: DAG约束后的邻接矩阵
        """
        # 方法：上三角矩阵（避免循环）
        B, N, _ = adjacency.shape
        mask = torch.triu(torch.ones(N, N, device=adjacency.device), diagonal=1)
        mask = mask.unsqueeze(0).expand(B, -1, -1)
        
        dag_adjacency = adjacency * mask
        return dag_adjacency
    
    def _gumbel_softmax(
        self,
        logits: torch.Tensor,
        hard: bool = True
    ) -> torch.Tensor:
        """
        Gumbel-Softmax采样（可微的离散采样）
        
        Args:
            logits: 未归一化的logits [B, num_modalities, num_modalities]
            hard: 是否使用hard模式（离散化）
        
        Returns:
            samples: 采样结果
        """
        # 对于二值化，我们使用Gumbel-Sigmoid
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        y = logits + gumbel_noise
        y = torch.sigmoid(y / self.temperature)
        
        if hard:
            # Hard模式：离散化
            y_hard = (y > 0.5).float()
            y = y_hard - y.detach() + y
        
        return y
    
    def compute_causal_strength(
        self,
        features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        计算因果强度（每个因果边的强度）
        
        Args:
            features: 特征列表
        
        Returns:
            causal_strength: 因果强度矩阵 [B, num_modalities, num_modalities]
        """
        result = self.forward(features, return_adjacency=True)
        return result['adjacency']


class PersonalizedCausalGraph(nn.Module):
    """
    个性化因果图模块
    为每个患者生成个性化的因果结构
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_modalities: int = 3,
        num_patient_clusters: int = 5
    ):
        """
        Args:
            num_patient_clusters: 患者聚类数量（不同患者类型）
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.num_patient_clusters = num_patient_clusters
        
        # 患者类型分类器
        self.patient_classifier = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_patient_clusters)
        )
        
        # 每个患者类型的因果图模板
        self.causal_templates = nn.Parameter(
            torch.randn(num_patient_clusters, num_modalities, num_modalities) * 0.1
        )
        
        # 自适应因果图生成器
        self.adaptive_graph = AdaptiveCausalGraph(
            embed_dim=embed_dim,
            num_modalities=num_modalities
        )
    
    def forward(
        self,
        features: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        为每个患者生成个性化因果图
        
        Args:
            features: 特征列表
        
        Returns:
            personalized_graph: 个性化因果图
        """
        B = features[0].size(0)
        concat_feat = torch.cat(features, dim=-1)
        
        # 分类患者类型
        patient_types = self.patient_classifier(concat_feat)  # [B, num_patient_clusters]
        patient_types = torch.softmax(patient_types, dim=-1)
        
        # 选择对应的因果图模板
        # patient_types: [B, num_patient_clusters]
        # causal_templates: [num_patient_clusters, num_modalities, num_modalities]
        template_graph = torch.einsum('bc,cmn->bmn', patient_types, self.causal_templates)
        
        # 生成自适应因果图
        adaptive_result = self.adaptive_graph(features)
        
        # 结合模板和自适应图
        personalized_graph = 0.7 * template_graph + 0.3 * adaptive_result['adjacency']
        
        return {
            'personalized_graph': personalized_graph,
            'patient_types': patient_types,
            'template_graph': template_graph,
            'adaptive_graph': adaptive_result['adjacency']
        }

