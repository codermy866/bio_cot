#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 5.0 Core Innovation: mHC (Manifold-Constrained Hyper-Connections)

基于流形约束的超连接融合模块
核心思想：通过 Sinkhorn-Knopp 算法将模态交互投影到 Birkhoff 多胞体流形上
实现双随机矩阵（Doubly Stochastic Matrix）约束，保证结构化的一一对应
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ManifoldHyperConnection(nn.Module):
    """
    Bio-COT 5.0 核心创新：mHC (Manifold-Constrained Hyper-Connections)
    
    不同于普通的 Attention (Softmax)，我们通过 Sinkhorn 迭代求解双随机矩阵。
    这实际上是将模态间的交互关系投影到了 Birkhoff 多胞体流形上。
    
    理论优势：
    1. 数学一致性：与 Stage 3 的 Sinkhorn OT Loss 形成理论闭环
    2. 结构化对齐：双随机矩阵保证模态间的一一对应关系
    3. 防止模态淹没：临床数据不会被高维视觉特征淹没
    """
    
    def __init__(
        self, 
        img_dim: int = 768, 
        clinical_dim: int = 128, 
        hidden_dim: int = 512, 
        sinkhorn_iters: int = 3, 
        epsilon: float = 0.05
    ):
        """
        Args:
            img_dim: 图像特征维度 (ViT输出维度)
            clinical_dim: 临床特征维度
            hidden_dim: 隐藏层维度（投影后的维度）
            sinkhorn_iters: Sinkhorn迭代次数（支持梯度反向传播）
            epsilon: 熵正则化系数（温度参数）
        """
        super().__init__()
        self.sinkhorn_iters = sinkhorn_iters
        self.epsilon = epsilon  # 熵正则化系数

        # 1. 特征投影层 (Projection Heads)
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.clin_proj = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 2. 可学习的温度参数（可选）
        self.learnable_temp = nn.Parameter(torch.tensor(1.0))

        # 3. 输出重构层
        self.out_proj = nn.Linear(hidden_dim, img_dim)
        self.norm = nn.LayerNorm(img_dim)

    def sinkhorn_knopp(self, log_alpha: torch.Tensor) -> torch.Tensor:
        """
        在对数域进行 Sinkhorn 迭代，保证数值稳定性。
        
        Args:
            log_alpha: [B, N, N] 未归一化的亲和矩阵（对数域）
        
        Returns:
            P: [B, N, N] 双随机矩阵（Doubly Stochastic Matrix）
               - 行和为1（行归一化）
               - 列和近似平衡（列归一化）
               - 投影到 Birkhoff 多胞体流形上
        """
        log_P = log_alpha
        
        for _ in range(self.sinkhorn_iters):
            # 行归一化 (Row Normalization)
            log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
            # 列归一化 (Column Normalization)
            log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
            
        # 返回概率矩阵（指数化）
        return torch.exp(log_P)

    def forward(
        self, 
        img_feat: torch.Tensor, 
        clin_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        基于流形约束的特征融合
        
        Args:
            img_feat: [B, N, D] 视觉Patch特征
            clin_feat: [B, C] 临床特征向量
        
        Returns:
            img_feat_fused: [B, N, D] Clinical-Aware 视觉特征
        """
        B, N, D = img_feat.shape
        
        # --- Step 1: Subspace Projection ---
        # 将视觉和临床特征投影到同一隐藏空间
        H_v = self.img_proj(img_feat)  # [B, N, hidden_dim]
        H_c = self.clin_proj(clin_feat).unsqueeze(1)  # [B, 1, hidden_dim] (临床特征视为特殊Token)
        
        # --- Step 2: 构建融合特征序列 ---
        # 将临床特征和视觉特征拼接，形成统一序列
        # H_all: [B, N+1, hidden_dim]
        H_all = torch.cat([H_c, H_v], dim=1) 
        
        # --- Step 3: 计算交互矩阵 ---
        # 计算所有Token之间的相似度矩阵
        # S: [B, N+1, N+1]
        S = torch.matmul(H_all, H_all.transpose(1, 2)) / (H_all.shape[-1] ** 0.5)
        
        # 应用可学习温度
        S = S * self.learnable_temp
        
        # --- Step 4: 流形投影 (Manifold Projection) ---
        # 通过 Sinkhorn 迭代将相似度矩阵投影到双随机矩阵
        # 这一步是关键：将自由形式的相似度矩阵约束到 Birkhoff 多胞体上
        # 对数域计算以提高数值稳定性
        log_S = S / self.epsilon
        P = self.sinkhorn_knopp(log_S)  # [B, N+1, N+1] 双随机矩阵
        
        # --- Step 5: 基于流形的特征传播 ---
        # 使用双随机矩阵进行特征传播（类似图神经网络的消息传递）
        # H_refined: [B, N+1, hidden_dim]
        H_refined = torch.matmul(P, H_all)
        
        # --- Step 6: 分离增强后的视觉特征 ---
        # 取出临床增强后的视觉特征（第一个Token是临床特征，后续是视觉）
        # H_v_new: [B, N, hidden_dim]
        H_c_new, H_v_new = H_refined[:, 0:1, :], H_refined[:, 1:, :]
        
        # --- Step 7: 残差连接与输出投影 ---
        # 将融合后的特征投影回原始维度，并与原始特征做残差连接
        out = self.out_proj(H_v_new)  # [B, N, D]
        img_feat_fused = self.norm(img_feat + out)  # 残差连接
        
        return img_feat_fused


# 向后兼容的别名
ManifoldHyperConnectionFusion = ManifoldHyperConnection

