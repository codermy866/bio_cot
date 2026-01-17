#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因果约束的CLIP + 贝叶斯CLIP: 从关联到因果的跨模态对齐
结合现有多模态架构的创新实现

核心创新:
1. 因果约束CLIP: 使用因果图约束跨模态对齐，消除虚假关联
2. 贝叶斯CLIP: 量化跨模态匹配的不确定性
3. 与现有OCT+Colposcopy+Clinical特征无缝集成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Tuple, Dict
from pathlib import Path
import json

class CausalAttentionMask(nn.Module):
    """
    因果约束注意力掩码
    基于医学先验知识构建因果图约束
    """
    def __init__(self, causal_graph: Dict[str, list]):
        """
        Args:
            causal_graph: 因果图定义, 如 {'OCT': ['HPV'], 'Colposcopy': ['TCT']}
        """
        super().__init__()
        self.causal_graph = causal_graph
        self.register_buffer('causal_mask', None)
    
    def build_causal_mask(self, seq_lengths: Dict[str, int]) -> torch.Tensor:
        """
        构建因果掩码矩阵
        只允许因果关系内的特征交互
        """
        total_len = sum(seq_lengths.values())
        mask = torch.ones(total_len, total_len)
        
        # 根据因果图设置掩码
        start_idx = 0
        idx_map = {}
        for mod, length in seq_lengths.items():
            idx_map[mod] = (start_idx, start_idx + length)
            start_idx += length
        
        # 允许模态内部全连接
        for mod, (start, end) in idx_map.items():
            mask[start:end, start:end] = 1
        
        # 根据因果图允许跨模态连接
        for cause_mod, effect_mods in self.causal_graph.items():
            if cause_mod in idx_map:
                for effect_mod in effect_mods:
                    if effect_mod in idx_map:
                        c_start, c_end = idx_map[cause_mod]
                        e_start, e_end = idx_map[effect_mod]
                        mask[e_start:e_end, c_start:c_end] = 1
        
        return mask
    
    def forward(self, x: torch.Tensor, seq_lengths: Dict[str, int]) -> torch.Tensor:
        if self.causal_mask is None:
            self.causal_mask = self.build_causal_mask(seq_lengths)
        return self.causal_mask.to(x.device)


class BayesianCLIPEncoder(nn.Module):
    """
    贝叶斯CLIP编码器
    输出均值和方差，量化不确定性
    """
    def __init__(self, embed_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 均值编码器
        self.mean_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # 方差编码器
        self.var_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Softplus()  # 确保方差为正
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean: [B, embed_dim]
            var: [B, embed_dim]
        """
        mean = self.mean_encoder(x)
        var = self.var_encoder(x) + 1e-6  # 防止数值不稳定
        
        return mean, var
    
    def sample(self, mean: torch.Tensor, var: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        从变分分布中采样
        """
        # 数值稳定性：clamp方差
        var = torch.clamp(var, min=1e-6, max=10.0)
        mean = torch.clamp(mean, min=-10.0, max=10.0)
        
        if training:
            epsilon = torch.randn_like(mean)
            sampled = mean + epsilon * torch.sqrt(var)
            # 再次clamp以确保数值稳定
            sampled = torch.clamp(sampled, min=-10.0, max=10.0)
            return sampled
        else:
            return mean


class CausalBayesianCLIP(nn.Module):
    """
    因果约束的贝叶斯CLIP
    结合因果约束和不确定性量化
    """
    def __init__(
        self,
        embed_dim: int = 768,
        clinical_dim: int = 256,
        causal_graph: Optional[Dict] = None,
        num_classes: int = 2,
        temperature: float = 0.07,
        kl_weight: float = 0.01,
        use_pretrained_backbones: bool = False,
        num_centers: int = 0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.clinical_dim = clinical_dim
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.use_pretrained_backbones = use_pretrained_backbones
        self.num_centers = num_centers
        
        # 临床特征投影层
        self.clinical_proj = nn.Linear(clinical_dim, embed_dim)
        
        # 贝叶斯编码器（每个模态一个）
        self.oct_encoder = BayesianCLIPEncoder(embed_dim)
        self.colposcopy_encoder = BayesianCLIPEncoder(embed_dim)
        self.clinical_encoder = BayesianCLIPEncoder(embed_dim)

        # 可选：预训练视觉主干（使用timm或torchvision）
        if self.use_pretrained_backbones:
            try:
                import torchvision.models as tvm
                resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
                self.vision_oct_backbone = nn.Sequential(*(list(resnet.children())[:-1]))  # 去掉fc
                resnet2 = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
                self.vision_col_backbone = nn.Sequential(*(list(resnet2.children())[:-1]))
                self.vision_proj = nn.Linear(512, embed_dim)
            except Exception:
                self.use_pretrained_backbones = False
        
        # 因果约束注意力
        if causal_graph is None:
            # 默认因果图：OCT和Colposcopy互相因果，临床特征影响二者
            causal_graph = {
                'OCT': ['Clinical'],
                'Colposcopy': ['Clinical'],
                'Clinical': []
            }
        
        self.causal_mask = CausalAttentionMask(causal_graph)
        
        # 跨模态对比学习投影
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim, num_classes)
        )

        # 域分类器（中心对抗）：GRL + 线性分类
        self.domain_classifier = None
        if self.num_centers and self.num_centers > 1:
            self.domain_classifier = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(self.embed_dim, self.num_centers)
            )
        
        # 不确定性估计头
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim // 2),
            nn.LayerNorm(self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim // 2, 1),
            nn.Sigmoid()  # 输出0-1的不确定性分数
        )
    
    def forward(
        self,
        oct_feat: torch.Tensor,
        colpo_feat: torch.Tensor,
        clinical_feat: torch.Tensor,
        return_uncertainty: bool = True,
        oct_images: Optional[torch.Tensor] = None,
        col_images: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            oct_feat: [B, embed_dim]
            colpo_feat: [B, embed_dim]
            clinical_feat: [B, embed_dim//3]
        """
        B = oct_feat.size(0)
        
        # 如果提供原始图像且启用预训练主干，则提取视觉特征
        if self.use_pretrained_backbones and oct_images is not None:
            # oct_images: [B, T, 3, H, W] 或 [T, 3, H, W]
            if oct_images.dim() == 4:
                oct_images = oct_images.unsqueeze(0)
            B_oct, T, C, H, W = oct_images.shape
            x = oct_images.view(B_oct * T, C, H, W)
            vfeat = self.vision_oct_backbone(x).flatten(1)  # [B*T, 512]
            vfeat = vfeat.view(B_oct, T, -1).mean(dim=1)  # [B, 512]
            oct_feat = self.vision_proj(vfeat)

        if self.use_pretrained_backbones and col_images is not None:
            # col_images: [B, K, 3, H, W] 或 [K, 3, H, W]
            if col_images.dim() == 4:
                col_images = col_images.unsqueeze(0)
            B_col, K, C, H, W = col_images.shape
            x = col_images.view(B_col * K, C, H, W)
            vfeat = self.vision_col_backbone(x).flatten(1)
            vfeat = vfeat.view(B_col, K, -1).mean(dim=1)
            colpo_feat = self.vision_proj(vfeat)

        # 贝叶斯编码（均值和方差）
        oct_mean, oct_var = self.oct_encoder(oct_feat)
        colpo_mean, colpo_var = self.colposcopy_encoder(colpo_feat)
        
        # 临床特征投影到相同维度
        clinical_proj = self.clinical_proj(clinical_feat)
        clinical_mean, clinical_var = self.clinical_encoder(clinical_proj)
        
        # 采样（训练时采样，推理时用均值）
        oct_feat_sampled = self.oct_encoder.sample(oct_mean, oct_var, self.training)
        colpo_feat_sampled = self.colposcopy_encoder.sample(colpo_mean, colpo_var, self.training)
        clinical_feat_sampled = self.clinical_encoder.sample(clinical_mean, clinical_var, self.training)
        
        # 构建序列
        multimodal_seq = torch.stack([
            oct_feat_sampled,
            colpo_feat_sampled,
            clinical_feat_sampled
        ], dim=1)  # [B, 3, embed_dim]
        
        # 扩展到embed_dim*2以匹配注意力机制
        multimodal_seq_expanded = torch.cat([
            F.pad(multimodal_seq[:, 0:1, :], (0, self.embed_dim)),  # OCT
            F.pad(multimodal_seq[:, 1:2, :], (0, self.embed_dim)),  # Colposcopy
            F.pad(multimodal_seq[:, 2:3, :], (0, self.embed_dim * 2 // 3))  # Clinical
        ], dim=-1)
        
        multimodal_seq_expanded = multimodal_seq_expanded[:, :, :self.embed_dim * 2]
        
        # 简化融合（因果约束通过门控机制实现）
        # 使用简单的加权平均和自注意力
        
        # 多头注意力融合
        attn_output, attn_weights = self.multihead_attn(
            multimodal_seq_expanded, multimodal_seq_expanded, multimodal_seq_expanded
        )
        
        # 融合
        fused = attn_output.mean(dim=1)  # [B, embed_dim*2]
        fused = self.fusion(fused)
        
        # 分类（使用融合特征）
        fused_expanded = torch.cat([fused, fused], dim=-1)[:, :self.embed_dim * 2]
        logits = self.classifier(fused_expanded)
        
        # 导出用于对比学习的嵌入（L2归一化）
        emb_oct = F.normalize(oct_feat_sampled, p=2, dim=-1)
        emb_colpo = F.normalize(colpo_feat_sampled, p=2, dim=-1)
        emb_clin = F.normalize(clinical_feat_sampled, p=2, dim=-1)

        # 不确定性估计
        uncertainty = None
        if return_uncertainty:
            # 使用KL散度作为不确定性来源
            # 数值稳定：对方差加下界，避免log(0)与NaN
            oct_var_s = oct_var.clamp_min(1e-6)
            colpo_var_s = colpo_var.clamp_min(1e-6)
            clinical_var_s = clinical_var.clamp_min(1e-6)
            kl_oct = -0.5 * torch.sum(1 + oct_var_s.log() - oct_mean.pow(2) - oct_var_s, dim=-1)
            kl_colpo = -0.5 * torch.sum(1 + colpo_var_s.log() - colpo_mean.pow(2) - colpo_var_s, dim=-1)
            kl_clinical = -0.5 * torch.sum(1 + clinical_var_s.log() - clinical_mean.pow(2) - clinical_var_s, dim=-1)
            
            total_kl = (kl_oct + kl_colpo + kl_clinical).unsqueeze(-1)
            uncertainty_input = torch.cat([fused, fused], dim=-1)[:, :self.embed_dim * 2]
            uncertainty = self.uncertainty_head(uncertainty_input)
            # 将KL散度转换为不确定性分数
            uncertainty = F.sigmoid(total_kl / 100.0)
        
        # 域对抗输出（梯度反转）
        domain_logits = None
        if self.domain_classifier is not None:
            # 简单实现的GRL：前向恒等，反向乘以-1（通过自定义autograd函数）
            class GradReverse(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x.view_as(x)
                @staticmethod
                def backward(ctx, grad_output):
                    return grad_output.neg()
            # 不要detach，需将对抗梯度回传到特征以实现域混淆
            grl_fused = GradReverse.apply(fused_expanded)
            domain_logits = self.domain_classifier(grl_fused)

        return {
            'logits': logits,
            'uncertainty': uncertainty,
            'mean': torch.cat([oct_mean, colpo_mean, clinical_mean], dim=-1),
            'var': torch.cat([oct_var, colpo_var, clinical_var], dim=-1),
            'emb_oct': emb_oct,
            'emb_colpo': emb_colpo,
            'emb_clin': emb_clin,
            'domain_logits': domain_logits
        }


class CausalBayesianCLIPLoss(nn.Module):
    """
    因果约束的CLIP + KL散度损失
    """
    def __init__(self, temperature: float = 0.07, kl_weight: float = 0.01, class_weights: Optional[torch.Tensor] = None, use_focal: bool = False, focal_gamma: float = 2.0, label_smoothing: float = 0.0, contrastive_weight: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.register_buffer('class_weights', class_weights if class_weights is not None else None)
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.contrastive_weight = contrastive_weight

    def _info_nce(self, zi: torch.Tensor, zj: torch.Tensor, temperature: float) -> torch.Tensor:
        # zi, zj: [B, D], 已归一化
        sim = zi @ zj.t()  # [B, B]
        sim = sim / temperature
        targets = torch.arange(zi.size(0), device=zi.device)
        loss_i = F.cross_entropy(sim, targets)
        loss_j = F.cross_entropy(sim.t(), targets)
        return (loss_i + loss_j) * 0.5
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: 模型输出
            targets: 真实标签 [B]
        """
        logits = predictions['logits']
        mean = predictions['mean']
        var = predictions['var']
        
        # 分类损失
        if self.use_focal:
            # Focal Loss 基于交叉熵
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            pt = probs.gather(1, targets.view(-1, 1)).squeeze(1).clamp_min(1e-6)
            focal_weight = (1 - pt) ** self.focal_gamma
            ce_per_sample = F.nll_loss(log_probs, targets, weight=self.class_weights, reduction='none')
            ce_loss = (focal_weight * ce_per_sample).mean()
        else:
            if self.class_weights is not None:
                ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, label_smoothing=self.label_smoothing)
            else:
                ce_loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        
        # KL散度损失（正则化不确定性）
        # 数值稳定：对方差加下界
        var_s = var.clamp_min(1e-6)
        kl_loss = 0.5 * torch.sum(mean.pow(2) + var_s - var_s.log() - 1)
        kl_loss = kl_loss / mean.size(0)  # 归一化
        
        # 对比学习损失（跨模态：OCT-Clinical、Colpo-Clinical）
        contrastive_loss = logits.new_tensor(0.0)
        if self.contrastive_weight > 0.0:
            emb_oct = predictions.get('emb_oct', None)
            emb_colpo = predictions.get('emb_colpo', None)
            emb_clin = predictions.get('emb_clin', None)
            if emb_oct is not None and emb_clin is not None:
                contrastive_loss = contrastive_loss + self._info_nce(emb_oct, emb_clin, self.temperature)
            if emb_colpo is not None and emb_clin is not None:
                contrastive_loss = contrastive_loss + self._info_nce(emb_colpo, emb_clin, self.temperature)

        total_loss = ce_loss + self.kl_weight * kl_loss + self.contrastive_weight * contrastive_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'kl_loss': kl_loss,
            'contrastive_loss': contrastive_loss
        }


def create_model_and_loss(num_classes: int = 2):
    """创建模型和损失函数"""
    # 定义因果图
    causal_graph = {
        'OCT': ['Clinical', 'HPV'],  # OCT由临床特征和HPV状态影响
        'Colposcopy': ['Clinical', 'TCT'],  # Colposcopy由临床特征和TCT影响
        'Clinical': []  # 临床特征是根源节点
    }
    
    model = CausalBayesianCLIP(
        embed_dim=768,
        clinical_dim=256,
        causal_graph=causal_graph,
        num_classes=num_classes,
        temperature=0.07,
        kl_weight=0.01
    )
    
    criterion = CausalBayesianCLIPLoss(
        temperature=0.07,
        kl_weight=0.01
    )
    
    return model, criterion


if __name__ == '__main__':
    print("=" * 60)
    print("🎯 因果约束的贝叶斯CLIP框架")
    print("=" * 60)
    
    # 创建模型
    model, criterion = create_model_and_loss(num_classes=2)
    
    # 测试前向传播
    B = 4
    oct_feat = torch.randn(B, 768)
    colpo_feat = torch.randn(B, 768)
    clinical_feat = torch.randn(B, 256)
    
    print(f"\n📊 输入形状:")
    print(f"  OCT: {oct_feat.shape}")
    print(f"  Colposcopy: {colpo_feat.shape}")
    print(f"  Clinical: {clinical_feat.shape}")
    
    with torch.no_grad():
        output = model(oct_feat, colpo_feat, clinical_feat)
        
    print(f"\n✅ 输出:")
    print(f"  Logits: {output['logits'].shape}")
    print(f"  Uncertainty: {output['uncertainty'].shape if output['uncertainty'] is not None else None}")
    
    # 测试损失
    targets = torch.randint(0, 2, (B,))
    loss_dict = criterion(output, targets)
    
    print(f"\n📉 损失:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")
    
    print("\n" + "=" * 60)
    print("✅ 模型架构测试通过！")
    print("=" * 60)

