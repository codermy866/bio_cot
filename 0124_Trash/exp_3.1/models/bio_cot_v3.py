#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.1 (Logic Loop Version)
核心升级：
1. Adaptive Modality Gating (自适应模态门控)
2. Semantic-Visual Alignment Loop (语义-视觉对齐闭环)
3. Enhanced Visual Notes (增强型视觉笔记 - Cross-Attention)
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Tuple, Optional, List

# 导入基础组件
from src.models.bida.bio_cot_v2 import DualHeadImageEncoder
from src.models.bida.memory_bank import NoiseMemoryBank, CenterDiscriminator
from src.models.bida.losses import SinkhornDistance, CounterfactualConsistencyLoss, AdversarialLoss

# 导入增强后的 Visual Notes
from .visual_notes import VisualNotesModule


class AdaptiveModalityGating(nn.Module):
    """
    [Innovation 1] 自适应模态互补门控 (AMCG)
    模仿医生决策：根据两个模态的特征质量和置信度，动态分配权重。
    """
    def __init__(self, dim: int = 768):
        super().__init__()
        # 压缩特征以计算重要性分数
        self.score_net = nn.Sequential(
            nn.Linear(dim * 2, dim // 2), # 输入 concat 特征
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 2) # 输出 [weight_oct, weight_colpo]
        )
        self.temperature = 1.0 # 控制 Softmax 的平滑度

    def forward(self, f_oct: torch.Tensor, f_colpo: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            f_oct: [B, D]
            f_colpo: [B, D]
        Returns:
            f_fused: [B, D]
            weights: (w_oct, w_colpo) for visualization
        """
        # 1. 拼接特征，捕捉模态间的互补信息
        concat_feat = torch.cat([f_oct, f_colpo], dim=-1) # [B, 2D]
        
        # 2. 计算动态权重
        logits = self.score_net(concat_feat) # [B, 2]
        weights = F.softmax(logits / self.temperature, dim=-1) # [B, 2]
        
        w_oct = weights[:, 0:1]   # [B, 1]
        w_colpo = weights[:, 1:2] # [B, 1]
        
        # 3. 加权融合
        f_fused = w_oct * f_oct + w_colpo * f_colpo
        
        return f_fused, (w_oct, w_colpo)


class BioCOT_v3(nn.Module):
    """
    Bio-COT 3.1 Ultimate Version
    实现"逻辑闭环"的自适应推理系统
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 2,
        num_centers: int = 5,
        input_dim: int = 768,
        use_visual_notes: bool = True,
        use_ot: bool = True,
        use_dual: bool = True,
        use_cross_attn: bool = True,
        warmup_epochs: int = 10,
        hidden_dim: int = 768
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_centers = num_centers
        self.use_visual_notes = use_visual_notes
        self.use_ot = use_ot
        self.use_dual = use_dual
        self.current_epoch = 0
        
        # 1. 语义投影器 (LLM Interface)
        # [FIX] 增加非线性与额外一层，提升文本特征适配能力（加速冷启动）
        self.note_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # ============================================================
        # [PROFESSIONAL FIX] 深度对齐投影头 + 共享语义空间
        # ============================================================
        # 问题诊断：
        # 1. 单层投影无法学习复杂的跨模态映射
        # 2. 图像和文本特征来自不同流程，语义空间不一致
        # 3. 需要更深的网络来学习共享的语义表示
        # ============================================================
        self.align_dim = 256
        
        # 方案1: 深度投影头（带残差连接，防止梯度消失）
        # 使用 2 层 MLP + 残差连接，学习更复杂的映射
        self.align_proj_img = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),  # 第一层：保持维度
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, self.align_dim, bias=False),  # 第二层：降维
            nn.LayerNorm(self.align_dim)
        )
        self.align_proj_text = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, self.align_dim, bias=False),
            nn.LayerNorm(self.align_dim)
        )
        
        # 方案2: 共享的语义空间投影（可选，用于进一步对齐）
        # 将图像和文本特征都映射到同一个共享空间
        self.shared_align_proj = nn.Sequential(
            nn.Linear(self.align_dim, self.align_dim, bias=False),
            nn.LayerNorm(self.align_dim),
            nn.GELU()
        )
        
        # 方案3: 温度系数优化
        # 使用更小的初始值（ln(1/0.1) ≈ 2.3），让模型更容易学习
        # 同时允许更大的学习范围
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / 0.1))  # 2.3026
        
        # 2. 增强型视觉笔记模块 (Visual Notes - Cross-Attention)
        if use_visual_notes:
            self.visual_notes_module = VisualNotesModule(
                img_dim=input_dim,
                text_dim=embed_dim,
                hidden_dim=hidden_dim,
                warmup_epochs=warmup_epochs
            )
        
        # 3. 自适应模态门控 (Adaptive Fusion) [NEW]
        self.adaptive_fusion = AdaptiveModalityGating(dim=embed_dim)
        
        # 4. 双头因果编码器 (Dual-Head Causal Encoder)
        if use_dual:
            self.dual_head = DualHeadImageEncoder(input_dim=input_dim, embed_dim=embed_dim)
        else:
            # Placeholder for non-dual mode
            self.dual_head = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU()
            )
            
        # 5. 跨模态融合 (Final Decision)
        # 将因果视觉特征(z_causal)与医学语义(z_sem)融合进行最终诊断
        self.final_fusion = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True) if use_cross_attn else None
        
        # 6. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # 7. Loss Modules
        if use_ot:
            self.ot_loss = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')
        
        if use_dual:
            self.memory_bank = NoiseMemoryBank(num_centers, embed_dim)
            self.consistency_loss = CounterfactualConsistencyLoss()
            self.adversarial_loss = AdversarialLoss(max(num_centers, 2))
            self.center_discriminator = CenterDiscriminator(embed_dim, max(num_centers, 2))
            
        # [CRITICAL FIX] 统一初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """统一初始化：Xavier for Linear, Standard for LayerNorm"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def set_epoch(self, epoch: int):
        """设置当前epoch（用于Visual Notes的Warm-up）"""
        self.current_epoch = int(epoch)
        if self.use_visual_notes:
            self.visual_notes_module.set_epoch(epoch)

    def extract_features(self, feat_raw, z_sem, beta):
        """Helper to process features with Visual Notes"""
        # feat_raw: [B, N, D] (Patches)
        if self.use_visual_notes and len(feat_raw.shape) == 3:
            feat_focused, attn_map = self.visual_notes_module(feat_raw, z_sem, beta=beta)
            feat_pooled = feat_focused.mean(dim=1) # GAP
            return feat_pooled, attn_map
        else:
            # Fallback for already pooled features or disabled notes
            feat_pooled = feat_raw if len(feat_raw.shape) == 2 else feat_raw.mean(dim=1)
            return feat_pooled, None

    def forward(
        self,
        f_oct: torch.Tensor,     # [B, N, D]
        f_colpo: torch.Tensor,   # [B, N, D]
        note_embeds: torch.Tensor, # [B, D]
        center_labels: Optional[torch.Tensor] = None,
        return_loss_components: bool = False,
        current_beta: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Logic Loop
        
        Args:
            f_oct: OCT图像特征 [B, N, D]
            f_colpo: Colposcopy图像特征 [B, N, D]
            note_embeds: Knowledge Note嵌入 [B, D]
            center_labels: 中心标签（用于对抗损失）
            return_loss_components: 是否返回损失组件
            current_beta: 当前beta值（用于Visual Notes）
        """
        output = {}
        B = f_oct.shape[0]
        device = f_oct.device
        
        # --- Step 1: 语义锚点生成 ---
        if note_embeds.dim() == 3: 
            note_embeds = note_embeds.squeeze(1)
        z_sem = self.note_projector(note_embeds) # [B, D]
        output['z_sem'] = z_sem
        
        # --- Step 2: 视觉笔记引导的特征提取 ---
        f_oct_pooled, attn_oct = self.extract_features(f_oct, z_sem, current_beta)
        f_colpo_pooled, attn_colpo = self.extract_features(f_colpo, z_sem, current_beta)
        
        if attn_oct is not None:
            output['attn_maps'] = [attn_oct, attn_colpo]
            
        # --- Step 3: 自适应模态融合 [INNOVATION] ---
        # 动态决定 OCT 和 Colpo 的权重，而不是固定的 0.6/0.4
        f_fused, (w_oct, w_colpo) = self.adaptive_fusion(f_oct_pooled, f_colpo_pooled)
        output['fusion_weights'] = {'oct': w_oct, 'colpo': w_colpo} # 保存权重用于论文可视化
        
        # --- Step 4: 双头因果解耦 ---
        if self.use_dual:
            z_causal, z_noise = self.dual_head(f_fused)
        else:
            z_causal = self.dual_head(f_fused)
            z_noise = None
        output['z_causal'] = z_causal
        output['z_noise'] = z_noise
        
        # --- Step 5: 最终诊断 (语义-因果特征融合) ---
        if self.final_fusion:
            # Q=z_causal (Image), K=V=z_sem (Knowledge)
            # 我们想用图像特征去查询匹配的知识
            z_causal_expanded = z_causal.unsqueeze(1) # [B, 1, D]
            z_sem_expanded = z_sem.unsqueeze(1)       # [B, 1, D]
            
            f_final, _ = self.final_fusion(z_causal_expanded, z_sem_expanded, z_sem_expanded)
            f_final = f_final.squeeze(1) + z_causal # Residual
        else:
            f_final = z_causal + z_sem # Simple addition
            
        pred = self.classifier(f_final)
        output['pred'] = pred
        output['logits'] = pred  # 兼容性
        
        # --- Step 6: Loss Calculation (Logic Loop) ---
        if return_loss_components:
            loss_dict = {}
            
            # 6.1 OT Loss (Distribution Matching)
            if self.use_ot:
                loss_dict['L_ot'] = self.ot_loss(z_causal, z_sem)
            
            # ============================================================
            # [PROFESSIONAL FIX] 深度对齐模块 (Deep Alignment with Shared Space)
            # ============================================================
            # 核心改进：
            # 1. 深度投影头（2层MLP + 残差）学习复杂映射
            # 2. 共享语义空间投影，强制图像和文本对齐
            # 3. 双重归一化 + 温度优化
            # 4. 添加辅助L2损失，直接约束特征距离
            # ============================================================
            
            # Step 1: 深度投影到对齐空间
            z_img_embed = self.align_proj_img(z_causal)  # [B, 256]
            z_txt_embed = self.align_proj_text(z_sem)    # [B, 256]
            
            # Step 2: 共享语义空间投影（进一步对齐）
            z_img_shared = self.shared_align_proj(z_img_embed)  # [B, 256]
            z_txt_shared = self.shared_align_proj(z_txt_embed)   # [B, 256]
            
            # Step 3: 双重归一化（投影前 + 投影后）
            # 投影前归一化：稳定输入分布
            z_causal_norm_in = F.normalize(z_causal, p=2, dim=-1)
            z_sem_norm_in = F.normalize(z_sem, p=2, dim=-1)
            
            # 投影后归一化：确保点积 = cosine similarity
            z_img_norm = F.normalize(z_img_shared, p=2, dim=-1)
            z_txt_norm = F.normalize(z_txt_shared, p=2, dim=-1)
            
            # Step 4: 温度系数（使用更小的范围，更容易学习）
            logit_scale = self.logit_scale.exp().clamp(min=0.1, max=50.0)  # 更小的范围
            
            # Step 5: 计算相似度矩阵 [B, B]
            logits_per_image = logit_scale * torch.matmul(z_img_norm, z_txt_norm.t())
            logits_per_text = logits_per_image.t()
            labels_align = torch.arange(B, device=device)
            
            # Step 6: 主损失（InfoNCE Loss）
            loss_align_ce = (F.cross_entropy(logits_per_image, labels_align) + 
                            F.cross_entropy(logits_per_text, labels_align)) / 2.0
            
            # Step 7: 辅助损失（L2距离损失，直接约束特征对齐）
            # 对于配对样本（对角线），我们希望它们的特征尽可能接近
            # 对于非配对样本（非对角线），我们希望它们的特征尽可能远离
            diagonal_distances = torch.norm(z_img_norm - z_txt_norm, p=2, dim=-1)  # [B]
            loss_align_l2 = diagonal_distances.mean()  # 最小化配对样本的距离
            
            # 组合损失（CE为主，L2为辅）
            loss_align = loss_align_ce + 0.1 * loss_align_l2
            loss_dict['L_align'] = loss_align
            loss_dict['L_align_ce'] = loss_align_ce  # 用于监控
            loss_dict['L_align_l2'] = loss_align_l2  # 用于监控
            
            # Step 8: 计算 Recall@1
            with torch.no_grad():
                pred_i2t = logits_per_image.argmax(dim=1)
                correct = (pred_i2t == labels_align).float().sum()
                recall_val = correct / B
                loss_dict['Recall'] = recall_val
                loss_dict['Recall_Align'] = recall_val
                
                # 额外监控：平均余弦相似度（对角线）
                diagonal_sim = (z_img_norm * z_txt_norm).sum(dim=-1).mean()
                loss_dict['align_cosine_sim'] = diagonal_sim  # 应该接近1.0

            # ============================================================
            
            # 6.3 Sparse Loss (Attention Regularization)
            if attn_oct is not None:
                 # 使用 Entropy Loss 防止过度稀疏导致的坍塌
                 eps = 1e-8
                 # Normalize for entropy calculation
                 attn_norm = attn_oct / (attn_oct.sum(dim=1, keepdim=True) + eps)
                 entropy = -torch.sum(attn_norm * torch.log(attn_norm + eps), dim=1).mean()
                 loss_dict['L_sparse'] = entropy * 0.1 # Weight can be adjusted
            
            # 6.4 [FIX] 唤醒 Consistency 模块：永不允许 L_consist = 0
            if self.use_dual and center_labels is not None:
                if z_noise is not None:
                    # (1) Adv Loss：确保 z_noise 包含域信息
                    c_logits = self.center_discriminator(z_noise)
                    loss_dict['L_adv'] = self.adversarial_loss(c_logits, center_labels)

                    # (2) MemoryBank 更新：保证反事实噪声来源稳定（训练期才更新）
                    self.memory_bank.update(z_noise, center_labels)

                    # (3) 反事实噪声：优先采样“不同中心”的噪声；如果bank为空会自动fallback为随机噪声
                    # 目标中心 = (当前中心 + 1 + rand) % num_centers，保证 != 当前中心
                    if self.num_centers >= 2:
                        rand = torch.randint(0, max(self.num_centers - 1, 1), (B,), device=device)
                        target_centers = (center_labels + 1 + rand) % self.num_centers
                    else:
                        target_centers = center_labels

                    z_noise_cf = self.memory_bank.get_counterfactual_noise(target_centers, strategy="random")

                    # (4) 反事实预测：只替换噪声分量（用“z_causal + z_noise_cf”模拟干预）
                    z_causal_cf = z_causal + z_noise_cf

                    if self.final_fusion:
                        z_causal_cf_expanded = z_causal_cf.unsqueeze(1)
                        z_sem_expanded = z_sem.unsqueeze(1)
                        f_final_cf, _ = self.final_fusion(z_causal_cf_expanded, z_sem_expanded, z_sem_expanded)
                        f_final_cf = f_final_cf.squeeze(1) + z_causal_cf
                    else:
                        f_final_cf = z_causal_cf + z_sem

                    pred_cf = self.classifier(f_final_cf)

                    # (5) Consistency Loss：使用MSE，并clamp下界保证非零梯度
                    L_consist = self.consistency_loss(pred, pred_cf)
                    loss_dict["L_consist"] = torch.clamp(L_consist, min=1e-6, max=10.0)

            output['loss_components'] = loss_dict
            
        return output

def create_bio_cot_v3(config):
    """Factory function to create BioCOT_v3 model"""
    return BioCOT_v3(
        embed_dim=config.embed_dim,
        num_classes=config.num_classes,
        num_centers=config.num_centers,
        input_dim=config.input_dim,
        use_visual_notes=config.use_visual_notes,
        use_ot=config.use_ot,
        use_dual=config.use_dual,
        use_cross_attn=config.use_cross_attn,
        warmup_epochs=config.warmup_epochs,
        hidden_dim=getattr(config, 'hidden_dim', 768)
    )

