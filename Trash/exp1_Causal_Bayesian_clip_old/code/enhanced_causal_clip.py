#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的因果约束贝叶斯CLIP
包含可学习因果图发现和不确定性分解
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np
from pathlib import Path
import sys

# 导入基础框架
from src.models.causal_bayesian_clip_framework import (
    BayesianCLIPEncoder,
    CausalAttentionMask,
    CausalBayesianCLIPLoss
)


class LearnableCausalGraph(nn.Module):
    """
    可学习的因果图发现模块（修复版）
    结合领域知识和数据驱动学习
    修复：支持batch维度的梯度传播
    """
    
    def __init__(
        self,
        num_modalities: int = 3,
        embed_dim: int = 768,
        prior_knowledge: Optional[torch.Tensor] = None,
        use_dag_constraint: bool = True,
        dag_penalty_weight: float = 0.03,
        sparsity_weight: float = 3e-4,
        intervention_weight: float = 0.02
    ):
        """
        Args:
            num_modalities: 模态数量（OCT, Colposcopy, Clinical）
            embed_dim: 特征维度
            prior_knowledge: 先验知识矩阵 [num_modalities, num_modalities]，1表示必须存在的因果关系
            use_dag_constraint: 是否使用DAG约束
        """
        super().__init__()
        self.num_modalities = num_modalities
        self.embed_dim = embed_dim
        self.use_dag_constraint = use_dag_constraint
        self.dag_penalty_weight = dag_penalty_weight
        self.sparsity_weight = sparsity_weight
        self.intervention_weight = intervention_weight
        
        # 先验知识（硬约束）
        if prior_knowledge is not None:
            self.register_buffer('prior_knowledge', prior_knowledge)
        else:
            # 默认先验：Clinical影响OCT和Colposcopy
            prior = torch.zeros(num_modalities, num_modalities)
            prior[0, 2] = 1  # OCT <- Clinical
            prior[1, 2] = 1  # Colposcopy <- Clinical
            self.register_buffer('prior_knowledge', prior)
        
        # 可学习的因果权重矩阵
        self.causal_weights = nn.Parameter(
            torch.randn(num_modalities, num_modalities) * 0.1
        )
        
        # 因果发现网络（基于特征学习因果图）
        self.causal_discovery = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_modalities * num_modalities),
            nn.Sigmoid()  # 输出0-1的因果强度
        )
    
    def enforce_dag(self, causal_matrix: torch.Tensor) -> torch.Tensor:
        """
        确保因果图是DAG（有向无环图）
        改进版本：使用上三角矩阵（避免循环）
        
        Args:
            causal_matrix: [B, num_modalities, num_modalities] 或 [num_modalities, num_modalities]
        
        Returns:
            causal_matrix: 应用DAG约束后的矩阵
        """
        if not self.use_dag_constraint:
            return causal_matrix
        
        # 方法：上三角矩阵（简单但有效）
        if causal_matrix.dim() == 3:
            # 有batch维度
            mask = torch.triu(torch.ones_like(causal_matrix[0]), diagonal=1)
            mask = mask.unsqueeze(0).expand_as(causal_matrix)
            causal_matrix = causal_matrix * mask
        else:
            # 无batch维度
            mask = torch.triu(torch.ones_like(causal_matrix), diagonal=1)
            causal_matrix = causal_matrix * mask
        
        return causal_matrix
    
    def compute_dag_penalty(self, causal_matrix: torch.Tensor) -> torch.Tensor:
        """
        计算DAG惩罚项（NOTEARS风格，使用 matrix_exp 保持可微）
        h(A)=tr(exp(A∘A))−d，DAG 时 h(A)=0
        """
        if not self.use_dag_constraint:
            return causal_matrix.new_zeros(())
        
        # 支持 batch matrix_exp
        A = causal_matrix
        expm = torch.matrix_exp(A * A)  # A∘A 保证非负
        h = expm.diagonal(dim1=-2, dim2=-1).sum(-1) - self.num_modalities
        penalty = (h * h).mean()
        return penalty
    
    def _apply_prior_and_mask(self, learned_causal: torch.Tensor, causal_weights_expanded: torch.Tensor) -> torch.Tensor:
        """融合可学习权重与先验，返回已约束的因果矩阵"""
        learned_causal = learned_causal * causal_weights_expanded
        
        if self.prior_knowledge is not None:
            prior_expanded = self.prior_knowledge.unsqueeze(0).expand_as(learned_causal)
            learned_causal = learned_causal * (1 - prior_expanded) + prior_expanded
        
        learned_causal = self.enforce_dag(learned_causal)
        learned_causal = torch.sigmoid(learned_causal)
        return learned_causal
    
    def forward(
        self,
        modality_features: List[torch.Tensor],
        return_penalty: bool = False,
        intervention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        输入：各模态特征
        输出：因果邻接矩阵 [B, num_modalities, num_modalities]
        
        Args:
            modality_features: 各模态的特征列表，每个是 [B, embed_dim]
            return_penalty: 是否返回DAG惩罚项
        
        Args:
            intervention_mask: [B, num_modalities]，1表示对该模态做了（真实或伪）干预
        Returns:
            causal_adj: [B, num_modalities, num_modalities]
            penalty_dict: 包含dag/sparsity/intervention/total的字典（如果return_penalty=True）
        """
        B = modality_features[0].size(0)
        
        # 拼接所有模态特征
        concat_features = torch.cat(modality_features, dim=-1)  # [B, embed_dim * num_modalities]
        
        # 数据驱动的因果发现
        learned_causal = self.causal_discovery(concat_features)  # [B, num_modalities * num_modalities]
        learned_causal = learned_causal.view(B, self.num_modalities, self.num_modalities)
        
        # 修复：保持batch维度，不要平均（这样才能支持梯度传播）
        # 结合可学习权重
        causal_weights_expanded = torch.sigmoid(self.causal_weights).unsqueeze(0).expand(B, -1, -1)
        causal_adj = self._apply_prior_and_mask(learned_causal, causal_weights_expanded)
        
        # 计算DAG惩罚
        penalty_dict = None
        if return_penalty:
            dag_penalty = self.compute_dag_penalty(causal_adj)
            sparsity_penalty = causal_adj.abs().mean()
            intervention_penalty = causal_adj.new_zeros(())
            
            # 伪/真实干预：对被干预模态的出边施加抑制，鼓励图对干预敏感
            if intervention_mask is not None:
                # 构造干预后的特征（将被干预模态置零，模拟do操作）
                intervened_features = []
                for i, feat in enumerate(modality_features):
                    mask_i = intervention_mask[:, i].unsqueeze(-1)  # [B,1]
                    intervened_feat = feat * (1 - mask_i)
                    intervened_features.append(intervened_feat)
                inter_concat = torch.cat(intervened_features, dim=-1)
                
                inter_causal = self.causal_discovery(inter_concat).view(B, self.num_modalities, self.num_modalities)
                inter_causal = self._apply_prior_and_mask(inter_causal, causal_weights_expanded)
                
                # 仅对被干预节点的出边施加稀疏/抑制约束
                mask_out = intervention_mask.unsqueeze(-1).expand_as(inter_causal)  # [B, num_modalities, num_modalities]
                intervention_penalty = (inter_causal * mask_out).mean()
            
            total_penalty = (
                self.dag_penalty_weight * dag_penalty
                + self.sparsity_weight * sparsity_penalty
                + self.intervention_weight * intervention_penalty
            )
            
            penalty_dict = {
                'dag': dag_penalty,
                'sparsity': sparsity_penalty,
                'intervention': intervention_penalty,
                'total': total_penalty
            }
        
        return causal_adj, penalty_dict


class UncertaintyDecomposition(nn.Module):
    """
    不确定性分解
    区分认知不确定性（epistemic）和偶然不确定性（aleatoric）
    """
    
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 认知不确定性估计（模型不确定性）
        self.epistemic_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
            nn.Softplus()
        )
        
        # 偶然不确定性估计（数据不确定性）
        self.aleatoric_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
            nn.Softplus()
        )
    
    def forward(
        self,
        features: torch.Tensor,
        variance: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        分解不确定性
        
        Args:
            features: 融合特征 [B, embed_dim]
            variance: 贝叶斯编码器输出的方差 [B, embed_dim] 或 [B, embed_dim * num_modalities]
        
        Returns:
            {
                'epistemic': 认知不确定性 [B, 1],
                'aleatoric': 偶然不确定性 [B, 1],
                'total': 总不确定性 [B, 1]
            }
        """
        # 认知不确定性：来自模型参数的不确定性（使用特征估计）
        epistemic = self.epistemic_head(features)
        
        # 偶然不确定性：来自数据的不确定性（使用方差）
        # variance可能是 [B, embed_dim] 或 [B, embed_dim * num_modalities]
        # 如果是多模态拼接的方差，先平均或取前embed_dim维
        if variance.size(-1) > features.size(-1):
            # 如果是拼接的方差，取前embed_dim维或平均
            if variance.size(-1) % features.size(-1) == 0:
                # 可能是3个模态拼接，reshape后平均
                num_modalities = variance.size(-1) // features.size(-1)
                variance_reshaped = variance.view(-1, num_modalities, features.size(-1))
                variance_mean = variance_reshaped.mean(dim=1)  # [B, embed_dim]
            else:
                # 直接取前embed_dim维
                variance_mean = variance[:, :features.size(-1)]
        else:
            variance_mean = variance
        
        aleatoric = self.aleatoric_head(variance_mean)
        
        # 总不确定性
        total = epistemic + aleatoric
        
        return {
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total': total
        }


class EnhancedCausalBayesianCLIP(nn.Module):
    """
    增强的因果约束贝叶斯CLIP
    包含可学习因果图和不确定性分解
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        clinical_dim: int = 7,
        num_classes: int = 2,
        temperature: float = 0.07,
        kl_weight: float = 0.01,
        use_learnable_causal: bool = True,
        use_uncertainty_decomposition: bool = True,
        causal_loss_weight: float = 0.01,
        num_centers: int = 0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.clinical_dim = clinical_dim
        self.num_classes = num_classes
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.use_learnable_causal = use_learnable_causal
        self.use_uncertainty_decomposition = use_uncertainty_decomposition
        self.causal_loss_weight = causal_loss_weight
        
        # 临床特征投影层
        self.clinical_proj = nn.Linear(clinical_dim, embed_dim)
        
        # 贝叶斯编码器（每个模态一个）
        self.oct_encoder = BayesianCLIPEncoder(embed_dim)
        self.colposcopy_encoder = BayesianCLIPEncoder(embed_dim)
        self.clinical_encoder = BayesianCLIPEncoder(embed_dim)
        
        # 可学习因果图
        if use_learnable_causal:
            self.learnable_causal_graph = LearnableCausalGraph(
                num_modalities=3,
                embed_dim=embed_dim
            )
            # 将因果图转换为CausalAttentionMask
            self.causal_mask_module = None  # 动态创建
        else:
            # 使用固定的因果图
            causal_graph = {
                'OCT': ['Clinical'],
                'Colposcopy': ['Clinical'],
                'Clinical': []
            }
            self.causal_mask_module = CausalAttentionMask(causal_graph)
            self.learnable_causal_graph = None
        
        # 跨模态对比学习投影
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 融合层（增强dropout以减少过拟合）
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.5),  # 增加dropout从0.3到0.5
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # 分类头（增强dropout以减少过拟合）
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.5),  # 增加dropout从0.3到0.5
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # 不确定性分解
        if use_uncertainty_decomposition:
            self.uncertainty_decomposition = UncertaintyDecomposition(embed_dim)
        else:
            self.uncertainty_decomposition = None
        
        # 域分类器（可选）
        self.domain_classifier = None
        if num_centers > 1:
            self.domain_classifier = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim // 2, num_centers)
            )
    
    def forward(
        self,
        oct_feat: torch.Tensor,
        colpo_feat: torch.Tensor,
        clinical_feat: torch.Tensor,
        return_uncertainty: bool = True,
        return_causal_penalty: bool = False,
        intervention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            oct_feat: OCT特征 [B, embed_dim]
            colpo_feat: Colposcopy特征 [B, embed_dim]
            clinical_feat: 临床特征 [B, clinical_dim]
            intervention_mask: [B,3]，1表示该模态被（真实或伪）干预，用于干预反馈闭环
        """
        B = oct_feat.size(0)
        
        # 临床特征投影
        clinical_proj = self.clinical_proj(clinical_feat)
        
        # 贝叶斯编码（均值和方差）
        oct_mean, oct_var = self.oct_encoder(oct_feat)
        colpo_mean, colpo_var = self.colposcopy_encoder(colpo_feat)
        clinical_mean, clinical_var = self.clinical_encoder(clinical_proj)
        
        # 采样（训练时采样，推理时用均值）
        oct_feat_sampled = self.oct_encoder.sample(oct_mean, oct_var, self.training)
        colpo_feat_sampled = self.colposcopy_encoder.sample(colpo_mean, colpo_var, self.training)
        clinical_feat_sampled = self.clinical_encoder.sample(clinical_mean, clinical_var, self.training)
        
        # 可学习因果图
        causal_adj = None
        causal_penalty_dict = None
        if self.use_learnable_causal and self.learnable_causal_graph is not None:
            causal_adj, causal_penalty_dict = self.learnable_causal_graph([
                oct_feat_sampled,
                colpo_feat_sampled,
                clinical_feat_sampled
            ], return_penalty=return_causal_penalty, intervention_mask=intervention_mask)  # [B, 3, 3]
            
            # 使用因果图约束注意力
            # 根据因果图权重调整特征
            causal_weights = causal_adj  # [B, 3, 3]
            
            # 构建序列
            multimodal_seq = torch.stack([
                oct_feat_sampled,
                colpo_feat_sampled,
                clinical_feat_sampled
            ], dim=1)  # [B, 3, embed_dim]
            
            # 应用因果权重
            multimodal_seq_weighted = torch.bmm(causal_weights, multimodal_seq)  # [B, 3, embed_dim]
        else:
            # 不使用可学习因果图
            multimodal_seq_weighted = torch.stack([
                oct_feat_sampled,
                colpo_feat_sampled,
                clinical_feat_sampled
            ], dim=1)  # [B, 3, embed_dim]
        
        # 多头注意力融合
        attn_output, attn_weights = self.multihead_attn(
            multimodal_seq_weighted,
            multimodal_seq_weighted,
            multimodal_seq_weighted
        )
        
        # 融合
        fused = attn_output.mean(dim=1)  # [B, embed_dim]
        fused = self.fusion(torch.cat([
            fused,
            oct_feat_sampled,
            colpo_feat_sampled
        ], dim=-1))  # [B, embed_dim]
        
        # 分类
        logits = self.classifier(fused)
        
        # 导出用于对比学习的嵌入
        emb_oct = F.normalize(oct_feat_sampled, p=2, dim=-1)
        emb_colpo = F.normalize(colpo_feat_sampled, p=2, dim=-1)
        emb_clin = F.normalize(clinical_feat_sampled, p=2, dim=-1)
        
        # 不确定性分解
        uncertainty_dict = None
        if return_uncertainty and self.uncertainty_decomposition is not None:
            # 使用融合特征的方差（平均各模态的方差）
            total_var = (oct_var + colpo_var + clinical_var) / 3.0  # [B, embed_dim]
            uncertainty_dict = self.uncertainty_decomposition(fused, total_var)
        elif return_uncertainty:
            # 简单的不确定性估计
            total_var = torch.cat([oct_var, colpo_var, clinical_var], dim=-1)
            uncertainty = total_var.mean(dim=-1, keepdim=True)
            uncertainty_dict = {'total': uncertainty}
        
        # 域分类器（可选）
        domain_logits = None
        if self.domain_classifier is not None:
            class GradReverse(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x.view_as(x)
                @staticmethod
                def backward(ctx, grad_output):
                    return grad_output.neg()
            
            grl_fused = GradReverse.apply(fused)
            domain_logits = self.domain_classifier(grl_fused)
        
        output = {
            'logits': logits,
            'uncertainty': uncertainty_dict,
            'mean': torch.cat([oct_mean, colpo_mean, clinical_mean], dim=-1),
            'var': torch.cat([oct_var, colpo_var, clinical_var], dim=-1),
            'emb_oct': emb_oct,
            'emb_colpo': emb_colpo,
            'emb_clin': emb_clin,
            'domain_logits': domain_logits,
            'causal_adj': causal_adj if self.use_learnable_causal and self.learnable_causal_graph is not None else None
        }
        
        # 添加因果惩罚到输出（用于损失计算）
        if causal_penalty_dict is not None:
            output['causal_penalties'] = causal_penalty_dict
            # 兼容旧键
            output['causal_penalty'] = causal_penalty_dict.get('total', None)
        
        return output


if __name__ == '__main__':
    print("=" * 60)
    print("🎯 增强的因果约束贝叶斯CLIP框架")
    print("=" * 60)
    
    # 创建模型
    model = EnhancedCausalBayesianCLIP(
        embed_dim=768,
        clinical_dim=7,
        num_classes=2,
        use_learnable_causal=True,
        use_uncertainty_decomposition=True
    )
    
    # 测试前向传播
    B = 4
    oct_feat = torch.randn(B, 768)
    colpo_feat = torch.randn(B, 768)
    clinical_feat = torch.randn(B, 7)
    
    print(f"\n📊 输入形状:")
    print(f"  OCT: {oct_feat.shape}")
    print(f"  Colposcopy: {colpo_feat.shape}")
    print(f"  Clinical: {clinical_feat.shape}")
    
    with torch.no_grad():
        output = model(oct_feat, colpo_feat, clinical_feat)
        
    print(f"\n✅ 输出:")
    print(f"  Logits: {output['logits'].shape}")
    if output['uncertainty'] is not None:
        for k, v in output['uncertainty'].items():
            print(f"  Uncertainty ({k}): {v.shape}")
    if output['causal_adj'] is not None:
        print(f"  Causal Adjacency: {output['causal_adj'].shape}")
    
    print("\n" + "=" * 60)
    print("✅ 增强模型架构测试通过！")
    print("=" * 60)

