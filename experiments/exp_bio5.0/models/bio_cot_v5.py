#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 5.0: Manifold-Constrained Hyper-Connections (mHC框架)
基于流形约束的超连接融合机制

核心改进：
1. Stage 1: mHC融合（基于Sinkhorn的Birkhoff流形投影）
2. Stage 2: VLM-Guided Causal Disentanglement（视觉笔记/因果解耦）
3. Stage 3: Sinkhorn Optimal Transport（多中心对齐）

理论优势：通过首尾呼应的Sinkhorn算法，形成完整的"最优传输理论"闭环
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

# 导入v3.0的Visual Notes模块
from .visual_notes import VisualNoteLayer, VisualNotesModule

# 导入v4.0的新组件
# 导入v5.0的mHC融合模块
from .mhc_fusion import ManifoldHyperConnection
try:
    from ..knowledge_base.enhanced_knowledge_retriever import VLMAugmentedRetriever
except ImportError:
    # 如果相对导入失败，使用绝对导入
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from knowledge_base.enhanced_knowledge_retriever import VLMAugmentedRetriever


class BioCOT_V5(nn.Module):
    """
    Bio-COT 5.0模型（mHC框架）
    
    核心改进：
    1. Stage 1: mHC融合（Manifold-Constrained Hyper-Connections）
       通过Sinkhorn算法将视觉-临床特征投影到Birkhoff流形上
    2. Stage 2: VLM-Guided Causal Disentanglement（视觉笔记/因果解耦）
    3. Stage 3: Sinkhorn Optimal Transport（与Stage 1形成理论闭环）
    
    理论优势：
    - 数学一致性：首尾呼应的Sinkhorn算法，形成完整的"最优传输理论"框架
    - 模态对齐：双随机矩阵保证视觉-临床的结构化一一对应
    - 防止模态淹没：临床数据不会被高维视觉特征淹没
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 2,
        num_centers: int = 5,
        input_dim: int = 768,  # ViT输出维度
        vlm_json_path: str = None,  # VLM缓存路径（必需）
        text_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        clinical_input_dim: int = 7,  # 🔥 新增：临床特征维度 (age/100, hpv, tct_onehot[5])
        use_mhc: bool = True,  # 🔥 新增：是否使用mHC融合
        mhc_hidden_dim: int = 512,  # 🔥 新增：mHC隐藏层维度
        sinkhorn_iters: int = 3,  # 🔥 新增：Sinkhorn迭代次数
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
            vlm_json_path: VLM缓存JSON文件路径（必需）
            text_model_name: 冻结的文本编码器模型名称
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
        self.use_mhc = use_mhc
        self.use_visual_notes = use_visual_notes
        self.use_ot = use_ot
        self.use_dual = use_dual
        self.use_cross_attn = use_cross_attn
        
        # 🔥🔥🔥 新增：Stage 1 - mHC融合模块 🔥🔥🔥
        # 这是5.0的核心创新：基于流形约束的超连接融合
        if use_mhc:
            self.mhc_fusion = ManifoldHyperConnection(
                img_dim=input_dim,
                clinical_dim=clinical_input_dim,
                hidden_dim=mhc_hidden_dim,
                sinkhorn_iters=sinkhorn_iters,
                epsilon=0.05
            )
            print(f"✅ mHC融合模块已创建 (clinical_dim={clinical_input_dim}, hidden_dim={mhc_hidden_dim}, sinkhorn_iters={sinkhorn_iters})")
        
        # ⚠️ 关键改动1：替换为 VLMAugmentedRetriever
        if vlm_json_path is None:
            raise ValueError("vlm_json_path 是必需的！请提供VLM缓存JSON文件路径。")
        
        self.knowledge_retriever = VLMAugmentedRetriever(
            vlm_json_path=vlm_json_path,
            visual_dim=embed_dim,
            text_model_name=text_model_name
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
        
        # 3. 双头编码器（Dual-Head）用于因果解耦
        if use_dual:
            self.dual_head = DualHeadImageEncoder(
                input_dim=input_dim,
                embed_dim=embed_dim
            )
        else:
            # 简化的单头编码器
            self.dual_head = nn.Sequential(
                nn.Linear(input_dim, embed_dim * 2),
                nn.LayerNorm(embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, embed_dim)
            )
        
        # 4. 融合模块（可选：跨模态融合）
        if use_cross_attn:
            self.fusion_module = CrossModalFusion(dim=embed_dim, num_heads=8)
        else:
            # 简化的融合（直接使用因果特征）
            self.fusion_module = nn.Identity()  # 不做融合
        
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
    
    def forward(
        self,
        images: torch.Tensor,  # [B, C, H, W] 或 [B, N, D] 图像特征
        clinical_features: Optional[torch.Tensor] = None,  # 🔥 新增：临床特征Tensor [B, C]
        clinical_data: Optional[List[str]] = None,  # 临床信息列表（可选，用于VLM）
        image_names: Optional[List[str]] = None,  # ⚠️ 图像文件名列表（必需，用于VLM检索）
        center_labels: Optional[torch.Tensor] = None,
        return_loss_components: bool = False,
        use_counterfactual: bool = True,
        current_beta: Optional[float] = None,
        use_vit_extraction: bool = False,  # 如果True，假设输入是原始图像，需要ViT提取
        vit_model=None  # ViT模型（用于图像特征提取）
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播（Bio-COT 5.0 - mHC框架）
        
        Args:
            images: [B, C, H, W] 原始图像 或 [B, N, D] Patch特征
            clinical_features: [B, C] 临床特征Tensor（age/100, hpv, tct_onehot[5]）
            clinical_data: list of strings 临床信息（可选，用于VLM）
            image_names: list of strings 图像文件名（必需，用于VLM检索）
            center_labels: [B] 中心标签（可选）
            return_loss_components: 是否返回损失组件
            use_counterfactual: 是否使用反事实干预
            current_beta: 当前Beta值（如果None，使用Warm-up策略）
            use_vit_extraction: 如果True，假设输入是原始图像，需要ViT提取
            vit_model: ViT模型（用于图像特征提取，如果use_vit_extraction=True）
        
        Returns:
            output: dict包含logits, z_causal, z_anchor, z_noise, attn_map等
        """
        B = images.shape[0]
        device = images.device
        
        # ⚠️ 关键改动2：验证必需参数
        if image_names is None:
            raise ValueError("image_names 是必需的！请提供图像文件名列表。")
        
        if len(image_names) != B:
            raise ValueError(f"image_names长度({len(image_names)})与batch大小({B})不匹配！")
        
        # 1. 视觉特征提取（Student, 🔥）
        if use_vit_extraction and vit_model is not None:
            # 从原始图像提取特征
            with torch.no_grad():
                vit_outputs = vit_model(images)
                if hasattr(vit_outputs, 'last_hidden_state'):
                    all_tokens = vit_outputs.last_hidden_state  # [B, N+1, D]
                else:
                    all_tokens = vit_outputs
                # 丢弃[CLS] token
                patch_features = all_tokens[:, 1:, :]  # [B, N, D]
        elif len(images.shape) == 4:
            # 假设是原始图像，但没有提供ViT模型，报错
            raise ValueError("如果输入是原始图像，请设置use_vit_extraction=True并提供vit_model")
        else:
            # 假设输入已经是Patch特征
            patch_features = images  # [B, N, D]
        
        # 🔥🔥🔥 Stage 1: mHC融合（Manifold-Constrained Hyper-Connections）🔥🔥🔥
        # 在视觉笔记之前，先进行视觉-临床特征的流形对齐
        # 这一步是关键：将视觉和临床特征投影到Birkhoff流形上，保证结构化对齐
        if self.use_mhc:
            if clinical_features is None:
                # 如果没有提供临床特征，创建零向量（防御性处理）
                clinical_features = torch.zeros(B, self.mhc_fusion.clin_proj[0].in_features, device=device, dtype=patch_features.dtype)
                print(f"⚠️ 警告: clinical_features为None，使用零向量 (shape: {clinical_features.shape})")
            
            # mHC融合：将patch_features和clinical_features投影到流形上
            # 输出：Clinical-Aware的视觉特征
            patch_features = self.mhc_fusion(patch_features, clinical_features)  # [B, N, D]
        
        # 2. ⚠️ 关键改动3：文本锚点生成（Teacher+Adapter, ❄️+🔥）
        # 这里发生了最关键的变化：不再是查表，而是动态生成
        z_anchor = self.knowledge_retriever(
            image_names=image_names,
            clinical_info=clinical_data,
            device=str(device)
        )  # [B, embed_dim]
        
        # 3. 视觉笔记交互（Visual Note Layer）
        # 用 z_anchor 去过滤 patch_features
        if self.use_visual_notes:
            refined_feats, attn_map = self.visual_notes_module(
                patch_features, z_anchor, beta=current_beta
            )  # [B, N, D], [B, N, 1]
        else:
            refined_feats = patch_features
            attn_map = None
        
        # Global Average Pooling
        feats_pooled = refined_feats.mean(dim=1)  # [B, D]
        
        # 4. 因果解耦
        if self.use_dual:
            z_causal, z_noise = self.dual_head(feats_pooled)  # [B, D], [B, D]
        else:
            z_causal = self.dual_head(feats_pooled)  # [B, D]
            z_noise = None
        
        # 5. 跨模态融合（可选）
        if self.use_cross_attn:
            fused_feat = self.fusion_module(z_causal, z_anchor)  # [B, D]
        else:
            fused_feat = z_causal  # 直接使用因果特征
        
        # 6. 分类
        logits = self.classifier(fused_feat)  # [B, num_classes]
        
        # 7. 构建输出
        output = {
            "logits": logits,
            "z_causal": z_causal,  # 用于OT Loss
            "z_anchor": z_anchor,  # ⚠️ 新增：用于OT Loss (Target)
            "z_noise": z_noise,
        }
        
        if attn_map is not None:
            output["attn_map"] = attn_map
        
        # 8. 计算损失组件
        if return_loss_components:
            loss_dict = {}
            
            # 8.1 Sinkhorn OT损失（Visual-Text对齐）
            if self.use_ot:
                L_ot = self.ot_loss(z_causal, z_anchor.detach())  # ⚠️ detach anchor，teacher不更新
                loss_dict['L_ot'] = L_ot
            
            # 8.2 稀疏性损失（如果使用Visual Notes）
            if self.use_visual_notes and attn_map is not None:
                # 使用熵损失
                eps = 1e-8
                attn_norm = attn_map.squeeze(-1) / (attn_map.squeeze(-1).sum(dim=1, keepdim=True) + eps)
                entropy = -torch.sum(attn_norm * torch.log(attn_norm + eps), dim=1).mean()
                l1 = torch.mean(torch.abs(attn_map))
                
                attn_mean = torch.mean(attn_map)
                if attn_mean < 0.01:
                    sparse_loss = 0.3 * entropy + 0.1 * l1
                else:
                    sparse_loss = 0.5 * entropy + 0.2 * l1
                
                L_sparse = sparse_loss
                loss_dict['L_sparse'] = L_sparse
                loss_dict['attn_mean'] = attn_mean.item()
            
            # 8.3 反事实一致性损失
            if self.use_dual and use_counterfactual and center_labels is not None:
                try:
                    self.memory_bank.update(z_noise, center_labels)
                    B_batch = center_labels.shape[0]
                    
                    # 为每个样本生成反事实中心
                    cf_center_ids = []
                    for i in range(B_batch):
                        current_center = center_labels[i].item()
                        other_centers = [c for c in range(self.num_centers) if c != current_center]
                        if len(other_centers) > 0:
                            import random
                            cf_center = random.choice(other_centers)
                        else:
                            cf_center = current_center
                        cf_center_ids.append(cf_center)
                    
                    cf_center_ids_tensor = torch.tensor(cf_center_ids, device=device, dtype=torch.long)
                    z_noise_cf = self.memory_bank.get_counterfactual_noise(
                        target_center_ids=cf_center_ids_tensor,
                        strategy='random'
                    )
                    
                    if z_noise_cf is not None and z_noise_cf.shape[0] == z_causal.shape[0]:
                        z_cf = z_causal + z_noise_cf
                        
                        if self.use_cross_attn:
                            z_cf_fused = self.fusion_module(z_cf, z_anchor.detach())
                        else:
                            z_cf_fused = z_cf
                        
                        logits_cf = self.classifier(z_cf_fused)
                        L_consist = self.consistency_loss(output['logits'], logits_cf)
                        L_consist = torch.clamp(L_consist, min=0.001, max=10.0)
                        loss_dict['L_consist'] = L_consist
                    else:
                        loss_dict['L_consist'] = torch.tensor(0.01, device=device, requires_grad=True)
                except Exception as e:
                    loss_dict['L_consist'] = torch.tensor(0.01, device=device, requires_grad=True)
            
            # 8.4 对抗损失
            if self.use_dual and center_labels is not None:
                center_logits = self.center_discriminator(z_noise)
                L_adv = self.adversarial_loss(center_logits, center_labels)
                loss_dict['L_adv'] = L_adv
            
            output['loss_components'] = loss_dict
        
        return output


# 创建模型工厂函数
def create_bio_cot_v5(config) -> BioCOT_V5:
    """
    创建Bio-COT v5模型
    
    Args:
        config: 配置对象（BioCOT_v5_Config）
    
    Returns:
        model: Bio-COT v5模型
    """
    model = BioCOT_V5(
        embed_dim=config.embed_dim,
        num_classes=config.num_classes,
        num_centers=config.num_centers,
        input_dim=config.input_dim,
        vlm_json_path=config.vlm_json_path,
        text_model_name=config.text_model_name,
        clinical_input_dim=config.clinical_input_dim,
        use_mhc=config.use_mhc,
        mhc_hidden_dim=config.mhc_hidden_dim,
        sinkhorn_iters=config.sinkhorn_iters,
        use_visual_notes=config.use_visual_notes,
        use_ot=config.use_ot,
        use_dual=config.use_dual,
        use_cross_attn=config.use_cross_attn,
        warmup_epochs=config.warmup_epochs
    )
    
    return model

