from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseAwareMHC(nn.Module):
    """
    [Bio-COT 5.5 Core Module]
    Noise-Aware Manifold Hyper-Connection (NA-mHC).
    
    Function:
    1. Modeling center-specific noise using learnable embeddings.
    2. Purifying visual features via a noise gate.
    3. Projecting clean visual-clinical interactions onto the Birkhoff Polytope via Sinkhorn.
    """
    def __init__(
        self, 
        img_dim: int = 768, 
        clinical_dim: int = 256, 
        num_centers: int = 5, 
        latent_dim: int = 256, 
        sinkhorn_iters: int = 3, 
        epsilon: float = 0.05,
        dropout: float = 0.3  # 🔥 新增：从config传递dropout_rate
    ):
        super().__init__()
        self.sinkhorn_iters = sinkhorn_iters
        self.epsilon = epsilon
        self.latent_dim = latent_dim

        # --- A. Projection Heads (with Dropout) ---
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, latent_dim),
            nn.Dropout(dropout)  # 🔥 使用传入的dropout参数
        )
        self.clin_proj = nn.Sequential(
            nn.Linear(clinical_dim, latent_dim),
            nn.Dropout(dropout)
        )
        
        # --- B. Noise Modeling (The "Awareness" Part) ---
        # 学习每个中心的"风格原型" (e.g., 偏绿, 过曝, 模糊)
        self.center_embedding = nn.Embedding(num_centers, latent_dim)
        
        # 噪声门控: 判断当前 patch 是否像"噪声"
        # Input: [Visual_Latent; Noise_Prototype] -> Output: Probability [0, 1]
        self.noise_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim // 2),
            nn.Dropout(dropout * 0.67),  # 🔥 稍低的dropout
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )

        # --- C. Reconstruction ---
        self.out_proj = nn.Sequential(
            nn.Linear(latent_dim, img_dim),
            nn.Dropout(dropout * 0.67)  # 🔥 稍低的dropout
        )
        self.norm = nn.LayerNorm(img_dim)

    def sinkhorn_log_space(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """
        Solves Optimal Transport plan in log-space for numerical stability.
        cost_matrix: [B, N, 1] or [B, N, K]
        Returns: Doubly Stochastic Matrix P (approx)
        """
        # P = exp(-C/epsilon)
        log_P = -cost_matrix / self.epsilon
        
        for _ in range(self.sinkhorn_iters):
            # Row Normalization: sum_k P_{ik} = 1
            log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
            
            # Column Normalization (Soft): sum_i P_{ik} = 1
            # 在 N >> K (Sequence >> Queries) 的情况下，列归一化能防止某个 Query 霸占所有 Attention
            log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
            
        return torch.exp(log_P)

    def forward(
        self, 
        img_feat: torch.Tensor, 
        clin_state: torch.Tensor, 
        center_ids: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img_feat: [B, N, D_v] - 视觉特征序列
            clin_state: [B, D_c] - 当前层的临床意图向量
            center_ids: [B] - 样本所属的中心 ID (LongTensor)
        Returns:
            feat_fused: [B, N, D_v] - 融合后的特征
            noise_prob: [B, N, 1] - 噪声概率图
        """
        B, N, _ = img_feat.shape
        
        # 1. 投影到潜在流形空间（Dropout在训练时自动生效）
        H_v = self.img_proj(img_feat)      # [B, N, L]
        H_c = self.clin_proj(clin_state).unsqueeze(1) # [B, 1, L]
        
        # 2. --- Noise Purification Process ---
        # 获取当前中心的噪声原型
        # 如果 center_ids 为 None (如外部测试未知中心)，可以用平均 embedding 或全0
        if center_ids is None:
            Z_noise = torch.zeros(B, 1, self.latent_dim, device=img_feat.device, dtype=img_feat.dtype)
        else:
            Z_noise = self.center_embedding(center_ids).unsqueeze(1) # [B, 1, L]
        
        # 扩展噪声原型以匹配 Visual Sequence
        Z_noise_expanded = Z_noise.expand(-1, N, -1) # [B, N, L]
        
        # 计算噪声概率 (Noise Probability Map)
        # 这一步模型在问："这个 Patch 是病灶还是这个医院特有的伪影？"
        noise_prob = self.noise_gate(torch.cat([H_v, Z_noise_expanded], dim=-1)) # [B, N, 1]
        
        # 净化特征 (Soft Suppression)
        # 我们希望 H_v_clean 只保留与中心风格无关的内容
        H_v_clean = H_v * (1.0 - noise_prob)
        
        # 3. --- Manifold-Constrained Fusion (Sinkhorn) ---
        # 在"干净"的流形上计算几何距离
        H_v_norm = F.normalize(H_v_clean, dim=-1)
        H_c_norm = F.normalize(H_c, dim=-1)
        
        # Cosine Distance (range 0~2)
        # [B, N, L] @ [B, L, 1] -> [B, N, 1]
        cost = 1.0 - torch.matmul(H_v_norm, H_c_norm.transpose(1, 2))
        
        # 计算最优传输计划 (The "Structure")
        transport_plan = self.sinkhorn_log_space(cost) # [B, N, 1]
        
        # 4. 特征传输 (Feature Transport)
        # 将临床信息注入到视觉特征中，注入量由 Transport Plan 决定
        context = transport_plan * H_c # [B, N, L]
        
        # 5. 残差连接与重构
        out = self.out_proj(context)
        
        # 返回:
        # 1. 融合后的特征 (加回原始特征保持信息流)
        # 2. 噪声概率图 (用于辅助 Loss 监督，强迫模型学会识别中心差异)
        return self.norm(img_feat + out), noise_prob


# 保留旧版本以兼容性（如果需要）
class ManifoldHyperConnection(nn.Module):
    """
    [Legacy] 基础 mHC (无噪声感知)
    保留用于向后兼容或消融实验
    """
    def __init__(
        self,
        img_dim: int = 768,
        clinical_dim: int = 256,
        latent_dim: int = 256,
        sinkhorn_iters: int = 3,
        epsilon: float = 0.05,
    ):
        super().__init__()
        self.sinkhorn_iters = sinkhorn_iters
        self.epsilon = epsilon

        self.img_proj = nn.Linear(img_dim, latent_dim)
        self.clin_proj = nn.Linear(clinical_dim, latent_dim)

        self.out_proj = nn.Linear(latent_dim, img_dim)
        self.norm = nn.LayerNorm(img_dim)

    def sinkhorn_log_space(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        log_p = -cost_matrix / self.epsilon
        for _ in range(self.sinkhorn_iters):
            log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
            log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
        return torch.exp(log_p)

    def forward(self, img_feat: torch.Tensor, clin_state: torch.Tensor) -> torch.Tensor:
        h_v = self.img_proj(img_feat)  # [B, N, L]
        h_c = self.clin_proj(clin_state).unsqueeze(1)  # [B, 1, L]

        h_vn = F.normalize(h_v, dim=-1)
        h_cn = F.normalize(h_c, dim=-1)
        cos_sim = torch.matmul(h_vn, h_cn.transpose(1, 2))  # [B, N, 1]
        cost = 1.0 - cos_sim

        p = self.sinkhorn_log_space(cost)  # [B, N, 1]
        context = p * h_c  # [B, N, L]
        out = self.out_proj(context)  # [B, N, D_v]
        return self.norm(img_feat + out)
