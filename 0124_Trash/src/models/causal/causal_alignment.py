#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因果对齐机制（Causal Alignment Mechanism）
核心创新：用临床模态的因果结构指导图像模态学习域不变表示

创新点：
1. 不是简单的特征对齐，而是因果结构对齐
2. 用临床模态的因果结构（域不变）指导图像模态学习域不变表示
3. 自动清洗图像模态中的设备噪声（域特定）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class CausalStructureExtractor(nn.Module):
    """
    因果结构提取器
    从特征中提取因果结构（因果图）
    """
    def __init__(self, embed_dim: int = 768, num_modalities: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        
        # 因果结构编码器
        self.structure_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, num_modalities * num_modalities)  # 因果邻接矩阵
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        提取因果结构
        
        Args:
            features: [B, embed_dim] 或 [B, num_modalities, embed_dim]
        
        Returns:
            causal_structure: [B, num_modalities, num_modalities] 因果邻接矩阵
        """
        if features.dim() == 3:
            # [B, num_modalities, embed_dim] -> [B, embed_dim] (平均池化)
            features = features.mean(dim=1)
        
        # 提取因果结构
        causal_adj = self.structure_encoder(features)  # [B, num_modalities * num_modalities]
        causal_adj = causal_adj.view(-1, self.num_modalities, self.num_modalities)
        
        # 应用sigmoid确保值在[0,1]
        causal_adj = torch.sigmoid(causal_adj)
        
        # 确保DAG性质（无自环）
        causal_adj = causal_adj * (1 - torch.eye(self.num_modalities, device=causal_adj.device))
        
        return causal_adj


class CausalAlignment(nn.Module):
    """
    因果对齐机制
    核心：用临床模态的因果结构指导图像模态学习域不变表示
    """
    def __init__(self, embed_dim: int = 768, num_modalities: int = 3, temperature: float = 0.07):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.temperature = temperature
        
        # 因果结构提取器
        self.structure_extractor = CausalStructureExtractor(embed_dim, num_modalities)
        
        # 因果结构对齐投影
        self.alignment_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
    def extract_causal_structure(self, features: torch.Tensor) -> torch.Tensor:
        """
        提取因果结构
        
        Args:
            features: [B, embed_dim]
        
        Returns:
            causal_structure: [B, num_modalities, num_modalities]
        """
        return self.structure_extractor(features)
    
    def align_causal_structures(
        self, 
        image_structure: torch.Tensor, 
        clinical_structure: torch.Tensor
    ) -> torch.Tensor:
        """
        对齐因果结构
        
        Args:
            image_structure: [B, num_modalities, num_modalities] 图像模态的因果结构
            clinical_structure: [B, num_modalities, num_modalities] 临床模态的因果结构
        
        Returns:
            alignment_loss: 标量
        """
        # 计算因果结构的相似度（Frobenius范数）
        structure_diff = image_structure - clinical_structure
        alignment_loss = torch.norm(structure_diff, p='fro', dim=(1, 2)).mean()
        
        return alignment_loss
    
    def apply_causal_structure(
        self, 
        features: torch.Tensor, 
        causal_structure: torch.Tensor
    ) -> torch.Tensor:
        """
        应用因果结构到特征
        
        Args:
            features: [B, embed_dim]
            causal_structure: [B, num_modalities, num_modalities]
        
        Returns:
            aligned_features: [B, embed_dim]
        """
        # 将特征投影到对齐空间
        aligned_features = self.alignment_proj(features)
        
        # 使用因果结构加权（这里简化处理，实际可以更复杂）
        # 将特征reshape为[B, num_modalities, embed_dim//num_modalities]
        B = features.size(0)
        chunk_size = self.embed_dim // self.num_modalities
        features_chunked = features[:, :chunk_size * self.num_modalities].view(
            B, self.num_modalities, chunk_size
        )
        
        # 应用因果结构
        aligned_chunked = torch.bmm(
            causal_structure, 
            features_chunked
        )  # [B, num_modalities, chunk_size]
        
        # 重新reshape
        aligned_chunked = aligned_chunked.view(B, -1)
        
        # 如果embed_dim不能被num_modalities整除，补齐
        if aligned_chunked.size(1) < self.embed_dim:
            padding = torch.zeros(
                B, 
                self.embed_dim - aligned_chunked.size(1), 
                device=aligned_chunked.device
            )
            aligned_chunked = torch.cat([aligned_chunked, padding], dim=1)
        
        # 与投影后的特征融合
        aligned_features = aligned_features + aligned_chunked
        
        return aligned_features
    
    def forward(
        self, 
        image_feat: torch.Tensor, 
        clinical_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        因果对齐前向传播
        
        Args:
            image_feat: [B, embed_dim] 图像模态特征
            clinical_feat: [B, embed_dim] 临床模态特征
        
        Returns:
            aligned_image_feat: [B, embed_dim] 对齐后的图像特征
            alignment_loss: 标量 对齐损失
        """
        # 1. 提取因果结构
        clinical_structure = self.extract_causal_structure(clinical_feat)  # [B, num_modalities, num_modalities]
        image_structure = self.extract_causal_structure(image_feat)  # [B, num_modalities, num_modalities]
        
        # 2. 对齐因果结构
        alignment_loss = self.align_causal_structures(image_structure, clinical_structure)
        
        # 3. 用临床模态的因果结构指导图像模态
        aligned_image_feat = self.apply_causal_structure(image_feat, clinical_structure)
        
        return aligned_image_feat, alignment_loss


class CausalUncertaintyDecomposition(nn.Module):
    """
    因果不确定性分解
    核心：区分因果结构不确定性和因果强度不确定性
    
    创新点：
    1. 不是标准的不确定性分解（epistemic/aleatoric）
    2. 而是因果不确定性分解（结构不确定性 + 强度不确定性）
    """
    def __init__(self, embed_dim: int = 768, num_modalities: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        
        # 因果结构不确定性估计
        self.structure_uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
            nn.Softplus()
        )
        
        # 因果强度不确定性估计
        self.strength_uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
            nn.Softplus()
        )
        
    def compute_structure_uncertainty(
        self, 
        causal_structure_posterior: torch.Tensor
    ) -> torch.Tensor:
        """
        计算因果结构不确定性
        
        Args:
            causal_structure_posterior: [B, num_modalities, num_modalities] 因果结构的后验分布
        
        Returns:
            structure_uncertainty: [B, 1] 结构不确定性
        """
        # 使用熵来衡量不确定性
        # 将因果结构归一化为概率分布
        structure_prob = F.softmax(
            causal_structure_posterior.view(-1, self.num_modalities * self.num_modalities), 
            dim=-1
        )
        
        # 计算熵
        entropy = -torch.sum(structure_prob * torch.log(structure_prob + 1e-8), dim=-1)
        
        # 归一化到[0,1]
        max_entropy = np.log(self.num_modalities * self.num_modalities)
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy.unsqueeze(-1)  # [B, 1]
    
    def compute_strength_uncertainty(
        self, 
        features: torch.Tensor, 
        causal_structure: torch.Tensor
    ) -> torch.Tensor:
        """
        计算因果强度不确定性
        
        Args:
            features: [B, embed_dim]
            causal_structure: [B, num_modalities, num_modalities]
        
        Returns:
            strength_uncertainty: [B, 1] 强度不确定性
        """
        # 使用特征和因果结构估计强度不确定性
        # 将因果结构展平
        structure_flat = causal_structure.view(-1, self.num_modalities * self.num_modalities)
        
        # 拼接特征和因果结构
        combined = torch.cat([features, structure_flat], dim=-1)
        
        # 如果维度不匹配，只使用features
        if combined.size(-1) > self.embed_dim * 2:
            combined = features
        
        # 估计强度不确定性
        strength_uncertainty = self.strength_uncertainty_head(combined)
        
        return strength_uncertainty
    
    def forward(
        self, 
        features: torch.Tensor, 
        causal_structure: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        因果不确定性分解
        
        Args:
            features: [B, embed_dim]
            causal_structure: [B, num_modalities, num_modalities]
        
        Returns:
            {
                'structure_uncertainty': [B, 1],
                'strength_uncertainty': [B, 1],
                'total_uncertainty': [B, 1]
            }
        """
        # 1. 因果结构不确定性
        structure_uncertainty = self.compute_structure_uncertainty(causal_structure)
        
        # 2. 因果强度不确定性
        strength_uncertainty = self.compute_strength_uncertainty(features, causal_structure)
        
        # 3. 总不确定性
        total_uncertainty = structure_uncertainty + strength_uncertainty
        
        return {
            'structure_uncertainty': structure_uncertainty,
            'strength_uncertainty': strength_uncertainty,
            'total_uncertainty': total_uncertainty
        }


class DomainInvariantLearning(nn.Module):
    """
    域不变性学习
    核心：利用临床模态的天然域不变性作为"锚点"，强制图像模态学习域不变表示
    
    创新点：
    1. 不是简单的域对抗训练
    2. 而是因果对齐的域不变性学习
    """
    def __init__(self, embed_dim: int = 768, num_domains: int = 5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_domains = num_domains
        
        # 域分类器（用于域对抗训练）
        self.domain_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_domains)
        )
        
    def domain_adversarial_loss(
        self, 
        features: torch.Tensor, 
        domain_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        域对抗损失
        
        Args:
            features: [B, embed_dim]
            domain_labels: [B] 域标签
        
        Returns:
            domain_loss: 标量
        """
        # 梯度反转层（GRL）
        class GradReverse(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.view_as(x)
            @staticmethod
            def backward(ctx, grad_output):
                return grad_output.neg()
        
        # 应用GRL
        grl_features = GradReverse.apply(features)
        
        # 域分类
        domain_logits = self.domain_classifier(grl_features)
        
        # 域分类损失（我们希望模型无法区分域）
        domain_loss = F.cross_entropy(domain_logits, domain_labels)
        
        return domain_loss
    
    def forward(
        self, 
        aligned_image_feat: torch.Tensor, 
        domain_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        域不变性学习
        
        Args:
            aligned_image_feat: [B, embed_dim] 对齐后的图像特征
            domain_labels: [B] 域标签
        
        Returns:
            domain_loss: 标量
        """
        # 域对抗损失：对齐后的图像特征应该无法区分域
        domain_loss = self.domain_adversarial_loss(aligned_image_feat, domain_labels)
        
        return domain_loss

