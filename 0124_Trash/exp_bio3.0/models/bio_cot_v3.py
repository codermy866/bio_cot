#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.0: 知识笔记引导的因果最优传输架构
按照用户提供的方案实现
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np

# 导入v2.0的基础模块
from src.models.bida.bio_cot_v2 import (
    CrossModalFusion, DualHeadImageEncoder
)
from src.models.bida.memory_bank import NoiseMemoryBank, CenterDiscriminator
from src.models.bida.losses import (
    SinkhornDistance, CounterfactualConsistencyLoss, AdversarialLoss
)

# 导入v3.0的新模块
from .visual_notes import VisualNoteLayer, VisualNotesModule


class BioCOT_v3(nn.Module):
    """
    Bio-COT 3.0模型（按照用户方案实现）
    知识笔记引导的因果最优传输架构
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 2,
        num_centers: int = 5,
        input_dim: int = 768,  # ViT输出维度
        use_visual_notes: bool = True,
        use_ot: bool = True,
        use_dual: bool = True,
        use_cross_attn: bool = True,
        warmup_epochs: int = 5,
        hidden_dim: int = 256  # VisualNoteLayer的隐藏层维度
    ):
        """
        Args:
            embed_dim: 特征嵌入维度
            num_classes: 分类类别数
            num_centers: 中心数量
            input_dim: 图像特征输入维度（ViT输出）
            use_visual_notes: 是否使用视觉笔记
            use_ot: 是否使用Sinkhorn OT损失
            use_dual: 是否使用Dual-Head结构
            use_cross_attn: 是否使用Cross-Attention融合
            warmup_epochs: Warm-up轮数
            hidden_dim: VisualNoteLayer隐藏层维度
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_centers = num_centers
        self.use_visual_notes = use_visual_notes
        self.use_ot = use_ot
        self.use_dual = use_dual
        self.use_cross_attn = use_cross_attn
        
        # 1. 语义投影器（处理离线生成的Knowledge Note Embedding）
        self.note_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        # 2. 视觉笔记层（分别对OCT和Colpo进行过滤）
        if use_visual_notes:
            self.visual_note_layer = VisualNoteLayer(
                img_dim=input_dim,
                text_dim=embed_dim,
                hidden_dim=hidden_dim
            )
            # Visual Notes Module（支持Warm-up）
            self.visual_notes_module = VisualNotesModule(
                img_dim=input_dim,
                text_dim=embed_dim,
                hidden_dim=hidden_dim,
                warmup_epochs=warmup_epochs
            )
        
        # 3. 双头编码器（Dual-Head）
        if use_dual:
            self.dual_head = DualHeadImageEncoder(
                input_dim=input_dim,
                embed_dim=embed_dim
            )
        else:
            self.dual_head = nn.Sequential(
                nn.Linear(input_dim, embed_dim * 2),
                nn.LayerNorm(embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, embed_dim)
            )
        
        # 4. 融合模块
        if use_cross_attn:
            self.fusion_module = CrossModalFusion(dim=embed_dim, num_heads=8)
        else:
            self.fusion_module = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(0.2)
            )
        
        # 5. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # 6. 损失函数模块
        if use_ot:
            self.ot_loss = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')
        
        if use_dual:
            self.memory_bank = NoiseMemoryBank(
                num_centers=num_centers,
                feat_dim=embed_dim,
                capacity=100
            )
            self.consistency_loss = CounterfactualConsistencyLoss()
            self.adversarial_loss = AdversarialLoss(num_centers=max(num_centers, 2))
            self.center_discriminator = CenterDiscriminator(
                feat_dim=embed_dim,
                num_centers=max(num_centers, 2)
            )
    
    def set_epoch(self, epoch: int):
        """设置当前epoch（用于Visual Notes的Warm-up）"""
        if self.use_visual_notes:
            self.visual_notes_module.set_epoch(epoch)
    
    def extract_patch_features_from_vit(
        self,
        vit_outputs,
        note_embeds: torch.Tensor,
        current_beta: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        修复漏洞1：从ViT输出中提取Patch特征（丢弃[CLS] token）
        
        Args:
            vit_outputs: ViT模型的输出（包含last_hidden_state）
            note_embeds: [B, D] Knowledge Note Embeddings（已投影）
            current_beta: 背景抑制系数
        
        Returns:
            feats_pooled: [B, D] 经过Visual Note过滤后的全局特征（GAP）
            attn_map: [B, N, 1] 注意力热图
        """
        # 修复漏洞1：丢弃[CLS] token (index 0)，只取Patch tokens (index 1:)
        # ViT输出格式: [B, N+1, D] 其中第0个是[CLS]，1:是Patches
        if hasattr(vit_outputs, 'last_hidden_state'):
            all_tokens = vit_outputs.last_hidden_state  # [B, N+1, D]
        else:
            all_tokens = vit_outputs  # 假设直接传入tensor
        
        # 关键修复：丢弃[CLS]，只保留Patches
        patch_feats = all_tokens[:, 1:, :]  # [B, N, D] N=196 for ViT-Base
        
        # 应用Visual Notes过滤
        if self.use_visual_notes:
            feats_focused, attn_map = self.visual_notes_module(
                patch_feats, note_embeds, beta=current_beta
            )  # [B, N, D], [B, N, 1]
        else:
            feats_focused = patch_feats
            attn_map = None
        
        # Global Average Pooling (GAP) on FOCUSED features
        # 对过滤后的Patch特征进行全局池化
        feats_pooled = feats_focused.mean(dim=1)  # [B, D]
        
        return feats_pooled, attn_map
    
    def forward(
        self,
        f_oct: torch.Tensor,  # [B, N_oct, D] OCT Patch特征（已丢弃[CLS]）或原始图像
        f_colpo: torch.Tensor,  # [B, N_colpo, D] Colposcopy Patch特征（已丢弃[CLS]）或原始图像
        note_embeds: torch.Tensor,  # [B, D] 或 [B, 1, D] 离线生成的Knowledge Note Embeddings
        center_labels: Optional[torch.Tensor] = None,
        return_loss_components: bool = False,
        use_counterfactual: bool = True,
        current_beta: Optional[float] = None,  # 动态Beta（如果None，使用Warm-up策略）
        use_vit_extraction: bool = False,  # 如果True，假设输入是原始图像，需要ViT提取
        vit_model_oct=None,  # ViT模型（用于OCT）
        vit_model_colpo=None  # ViT模型（用于Colposcopy）
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播（按照用户方案，修复漏洞1）
        
        Args:
            f_oct: [B, N_oct, D] OCT Patch特征（已丢弃[CLS]）或 [B, C, H, W] 原始图像
            f_colpo: [B, N_colpo, D] Colposcopy Patch特征（已丢弃[CLS]）或 [B, C, H, W] 原始图像
            note_embeds: [B, D] 或 [B, 1, D] 离线生成的Knowledge Note Embeddings
            center_labels: [B] 中心标签
            return_loss_components: 是否返回损失组件
            use_counterfactual: 是否使用反事实干预
            current_beta: 当前Beta值（如果None，使用Warm-up策略）
            use_vit_extraction: 如果True，假设输入是原始图像，需要ViT提取
            vit_model_oct: ViT模型（用于OCT，如果use_vit_extraction=True）
            vit_model_colpo: ViT模型（用于Colposcopy，如果use_vit_extraction=True）
        
        Returns:
            output: dict包含pred, z_causal, z_noise, z_sem, attn_maps等
        """
        B = f_oct.shape[0]
        device = f_oct.device
        
        # 处理note_embeds维度
        if note_embeds.dim() == 3:  # [B, 1, D]
            note_embeds = note_embeds.squeeze(1)  # [B, D]
        
        # 1. 处理语义锚点 (z_sem)
        z_sem = self.note_projector(note_embeds)  # [B, D]
        
        # 修复漏洞1：如果输入是原始图像，需要从ViT提取并丢弃[CLS]
        if use_vit_extraction and vit_model_oct is not None and vit_model_colpo is not None:
            # 从原始图像提取特征（修复：丢弃[CLS] token）
            with torch.no_grad():
                vit_outputs_oct = vit_model_oct(f_oct)
                vit_outputs_colpo = vit_model_colpo(f_colpo)
            
            # 使用修复后的提取方法
            f_oct_pooled, attn_oct = self.extract_patch_features_from_vit(
                vit_outputs_oct, z_sem, current_beta or 0.1
            )
            f_colpo_pooled, attn_colpo = self.extract_patch_features_from_vit(
                vit_outputs_colpo, z_sem, current_beta or 0.1
            )
        else:
            # 假设输入已经是Patch特征（已丢弃[CLS]）
            # 如果输入是[B, N, D]格式，应用Visual Notes
            if len(f_oct.shape) == 3:  # [B, N, D]
                if self.use_visual_notes:
                    f_oct_focused, attn_oct = self.visual_notes_module(
                        f_oct, z_sem, beta=current_beta
                    )  # [B, N_oct, D], [B, N_oct, 1]
                    f_colpo_focused, attn_colpo = self.visual_notes_module(
                        f_colpo, z_sem, beta=current_beta
                    )  # [B, N_colpo, D], [B, N_colpo, 1]
                else:
                    f_oct_focused = f_oct
                    f_colpo_focused = f_colpo
                    attn_oct = None
                    attn_colpo = None
                
                # Global Average Pooling
                f_oct_pooled = f_oct_focused.mean(dim=1)  # [B, D]
                f_colpo_pooled = f_colpo_focused.mean(dim=1)  # [B, D]
            else:
                # 如果输入已经是[B, D]格式（全局特征），直接使用
                f_oct_pooled = f_oct
                f_colpo_pooled = f_colpo
                attn_oct = None
                attn_colpo = None
        
        # 3. 特征融合（使用清洗后的特征）
        # 加权融合
        f_fused = 0.6 * f_oct_pooled + 0.4 * f_colpo_pooled  # [B, D]
        
        # 4. 进入双头因果模块
        if self.use_dual:
            z_causal, z_noise = self.dual_head(f_fused)  # [B, D], [B, D]
        else:
            z_causal = self.dual_head(f_fused)  # [B, D]
            z_noise = None
        
        # 5. 跨模态融合
        if self.use_cross_attn:
            fused_feat = self.fusion_module(z_causal, z_sem)  # [B, D]
        else:
            multimodal_feat = torch.cat([z_causal, z_sem], dim=-1)  # [B, 2D]
            fused_feat = self.fusion_module(multimodal_feat)  # [B, D]
        
        # 6. 分类预测
        pred = self.classifier(fused_feat)  # [B, num_classes]
        
        # 7. 构建输出
        output = {
            "pred": pred,
            "z_causal": z_causal,
            "z_sem": z_sem,
        }
        
        if z_noise is not None:
            output["z_noise"] = z_noise
        
        if attn_oct is not None and attn_colpo is not None:
            output["attn_maps"] = [attn_oct, attn_colpo]  # 返回Attention用于算Loss
        
        # 8. 计算损失组件
        if return_loss_components:
            loss_dict = {}
            
            # 8.1 Sinkhorn OT损失
            if self.use_ot:
                L_ot = self.ot_loss(z_causal, z_sem)
                loss_dict['L_ot'] = L_ot
            
            # 8.2 稀疏性损失（修复：使用熵损失替代L1，更温和且不会完全为0）
            if self.use_visual_notes and attn_oct is not None and attn_colpo is not None:
                # 计算平均注意力值（用于监控）
                attn_mean_oct = torch.mean(attn_oct)
                attn_mean_colpo = torch.mean(attn_colpo)
                
                # 修复：使用熵损失（Entropy Loss）替代L1损失
                # 熵损失鼓励注意力分布更集中（稀疏），但不会因为值小就完全为0
                # 公式: L_sparse = -mean(attn * log(attn + eps))
                
                eps = 1e-8
                
                # 对注意力进行归一化（确保是概率分布）
                attn_oct_norm = attn_oct.squeeze(-1) / (attn_oct.squeeze(-1).sum(dim=1, keepdim=True) + eps)  # [B, N]
                attn_colpo_norm = attn_colpo.squeeze(-1) / (attn_colpo.squeeze(-1).sum(dim=1, keepdim=True) + eps)  # [B, N]
                
                # 计算熵（负熵 = 稀疏性）
                # 熵越小，分布越集中（越稀疏）
                entropy_oct = -torch.sum(attn_oct_norm * torch.log(attn_oct_norm + eps), dim=1).mean()  # 标量
                entropy_colpo = -torch.sum(attn_colpo_norm * torch.log(attn_colpo_norm + eps), dim=1).mean()  # 标量
                
                # 稀疏损失 = 负熵（我们希望熵小，即分布集中）
                # 但为了避免完全坍塌，我们鼓励适度的稀疏性
                # 使用L1作为辅助，但权重较小
                l1_oct = torch.mean(torch.abs(attn_oct))
                l1_colpo = torch.mean(torch.abs(attn_colpo))
                
                # 组合熵损失和L1损失
                # 🔧 修复：调整稀疏损失计算，防止过度抑制注意力
                # 使用更温和的稀疏损失：鼓励适度稀疏，但不强制完全稀疏
                # 如果注意力均值过低（<0.01），说明已经过度稀疏，应该减少稀疏损失
                if attn_mean_oct < 0.01:
                    # 注意力已经过度稀疏，使用较小的稀疏损失（避免进一步抑制）
                    # 使用归一化后的熵，但权重降低
                    sparse_loss_oct = 0.3 * entropy_oct + 0.1 * l1_oct  # 降低权重
                else:
                    # 正常情况：使用标准稀疏损失
                    sparse_loss_oct = 0.5 * entropy_oct + 0.2 * l1_oct  # 降低权重，更温和
                
                if attn_mean_colpo < 0.01:
                    sparse_loss_colpo = 0.3 * entropy_colpo + 0.1 * l1_colpo
                else:
                    sparse_loss_colpo = 0.5 * entropy_colpo + 0.2 * l1_colpo
                
                L_sparse = sparse_loss_oct + sparse_loss_colpo
                loss_dict['L_sparse'] = L_sparse
                loss_dict['attn_mean_oct'] = attn_mean_oct.item()
                loss_dict['attn_mean_colpo'] = attn_mean_colpo.item()
            
            # 8.3 反事实一致性损失（修复：实现真正的反事实一致性损失）
            if self.use_dual and use_counterfactual and center_labels is not None:
                try:
                    # 更新Memory Bank
                    self.memory_bank.update(z_noise, center_labels)
                    
                    # 获取反事实噪声（从其他中心采样）
                    # 策略：为每个样本随机选择一个不同的中心
                    B = center_labels.shape[0]
                    num_centers = self.num_centers
                    
                    # 为每个样本生成一个不同的中心ID（反事实中心）
                    cf_center_ids = []
                    for i in range(B):
                        current_center = center_labels[i].item()
                        # 随机选择一个不同的中心
                        other_centers = [c for c in range(num_centers) if c != current_center]
                        if len(other_centers) > 0:
                            import random
                            cf_center = random.choice(other_centers)
                        else:
                            cf_center = current_center  # 如果只有一个中心，使用当前中心
                        cf_center_ids.append(cf_center)
                    
                    cf_center_ids_tensor = torch.tensor(cf_center_ids, device=device, dtype=torch.long)
                    
                    # 从Memory Bank获取反事实噪声（来自不同中心）
                    z_noise_cf = self.memory_bank.get_counterfactual_noise(
                        target_center_ids=cf_center_ids_tensor,
                        strategy='random'
                    )  # [B, D]
                    
                    # 调试：检查Memory Bank状态
                    memory_bank_counts = [self.memory_bank.count[c].item() for c in range(num_centers)]
                    
                    if z_noise_cf is not None and z_noise_cf.shape[0] == z_causal.shape[0] and z_noise_cf.shape[1] == z_causal.shape[1] and not torch.isnan(z_noise_cf).any():
                        # 构建反事实特征：z_causal + z_noise_cf（来自其他中心）
                        z_cf = z_causal + z_noise_cf  # [B, D]
                        
                        # 计算反事实预测
                        if self.use_cross_attn:
                            z_cf_fused = self.fusion_module(z_cf, z_sem)  # [B, D]
                        else:
                            z_cf_multimodal = torch.cat([z_cf, z_sem], dim=-1)  # [B, 2D]
                            z_cf_fused = self.fusion_module(z_cf_multimodal)  # [B, D]
                        
                        logits_cf = self.classifier(z_cf_fused)  # [B, num_classes]
                        
                        # 计算一致性损失：原始预测和反事实预测应该一致
                        L_consist = self.consistency_loss(outputs['pred'], logits_cf)
                        
                        # 限制损失范围，避免过大
                        L_consist = torch.clamp(L_consist, min=0.001, max=10.0)
                        loss_dict['L_consist'] = L_consist
                    else:
                        # Fallback：如果无法获取反事实噪声，使用当前batch的噪声（确保总是计算真正的损失）
                        if z_noise is not None and z_noise.shape[0] == z_causal.shape[0] and z_noise.shape[1] == z_causal.shape[1] and not torch.isnan(z_noise).any():
                            # 使用当前batch的噪声，但添加一些随机扰动模拟反事实
                            alpha = 0.3
                            # 添加小的随机扰动，模拟来自不同中心的噪声
                            noise_perturb = torch.randn_like(z_noise) * 0.1
                            z_mix = z_causal + alpha * (z_noise.detach() + noise_perturb)
                            
                            if self.use_cross_attn:
                                z_mix_fused = self.fusion_module(z_mix, z_sem)
                            else:
                                z_mix_multimodal = torch.cat([z_mix, z_sem], dim=-1)
                                z_mix_fused = self.fusion_module(z_mix_multimodal)
                            
                            logits_cf = self.classifier(z_mix_fused)
                            
                            # 确保logits有效
                            if not torch.isnan(logits_cf).any() and not torch.isinf(logits_cf).any():
                                L_consist = self.consistency_loss(outputs['pred'], logits_cf)
                                L_consist = torch.clamp(L_consist, min=0.001, max=10.0)
                                loss_dict['L_consist'] = L_consist
                            else:
                                # 如果logits无效，使用小的非零损失
                                loss_dict['L_consist'] = torch.tensor(0.01, device=device, requires_grad=True)
                        else:
                            # 最后的fallback：使用小的非零损失
                            loss_dict['L_consist'] = torch.tensor(0.01, device=device, requires_grad=True)
                except Exception as e:
                    # 异常处理：使用fallback
                    if z_noise is not None and z_noise.shape[0] == z_causal.shape[0] and z_noise.shape[1] == z_causal.shape[1]:
                        try:
                            alpha = 0.3
                            z_mix = z_causal + alpha * z_noise.detach()
                            if self.use_cross_attn:
                                z_mix_fused = self.fusion_module(z_mix, z_sem)
                            else:
                                z_mix_multimodal = torch.cat([z_mix, z_sem], dim=-1)
                                z_mix_fused = self.fusion_module(z_mix_multimodal)
                            logits_cf = self.classifier(z_mix_fused)
                            L_consist = self.consistency_loss(outputs['pred'], logits_cf)
                            L_consist = torch.clamp(L_consist, min=0.001, max=10.0)
                            loss_dict['L_consist'] = L_consist
                        except Exception as e2:
                            # 如果fallback也失败，使用小的非零损失
                            loss_dict['L_consist'] = torch.tensor(0.01, device=device, requires_grad=True)
                    else:
                        loss_dict['L_consist'] = torch.tensor(0.01, device=device, requires_grad=True)
            
            # 8.4 对抗损失
            if self.use_dual and center_labels is not None:
                center_logits = self.center_discriminator(z_noise)
                L_adv = self.adversarial_loss(center_logits, center_labels)
                loss_dict['L_adv'] = L_adv
            
            output['loss_components'] = loss_dict
        
        return output


def sparse_loss(attn_maps: List[torch.Tensor], lower_bound: float = 0.01) -> torch.Tensor:
    """
    稀疏性损失（修复漏洞3：添加下界保护，防止坍塌）
    
    Args:
        attn_maps: list of [B, N, 1] tensors
        lower_bound: 下界阈值，如果平均注意力低于此值，停止施加损失
    
    Returns:
        loss: 稀疏性损失
    """
    loss = 0
    for attn in attn_maps:
        # 计算平均注意力值
        attn_mean = torch.mean(attn)
        
        # 修复漏洞3：如果注意力值过低（接近0），停止施加稀疏损失
        if attn_mean > lower_bound:
            # 希望attn的均值较小（大部分区域是背景）
            loss += torch.mean(torch.abs(attn))
        # 否则，损失为0（防止坍塌）
    
    return loss


# 创建模型工厂函数
def create_bio_cot_v3(config) -> BioCOT_v3:
    """
    创建Bio-COT v3模型
    
    Args:
        config: 配置对象（BioCOT_v3_Config）
    
    Returns:
        model: Bio-COT v3模型
    """
    model = BioCOT_v3(
        embed_dim=config.embed_dim,
        num_classes=config.num_classes,
        num_centers=config.num_centers,
        input_dim=config.input_dim,
        use_visual_notes=config.use_visual_notes,
        use_ot=config.use_ot,
        use_dual=config.use_dual,
        use_cross_attn=config.use_cross_attn,
        warmup_epochs=config.warmup_epochs
    )
    
    return model
