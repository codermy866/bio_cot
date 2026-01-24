#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强模型架构
改进特征表示能力和多模态融合策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力块"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MultiHeadCrossModalAttention(nn.Module):
    """多头跨模态注意力机制"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()
        
        # 线性投影
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output, attn_weights

class EnhancedCausalGNN(nn.Module):
    """增强的因果图神经网络"""
    
    def __init__(
        self,
        clinical_dim: int = 8,
        causal_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.causal_dim = causal_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 更深的编码器
        self.encoder = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, causal_dim * 2)  # μ, logσ
        )
        
        # 多头自注意力层
        self.attention_layers = nn.ModuleList([
            MultiHeadCrossModalAttention(causal_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 前馈网络
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(causal_dim, causal_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(causal_dim * 4, causal_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.norm_layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(causal_dim),
                nn.LayerNorm(causal_dim)
            ])
            for _ in range(num_layers)
        ])
        
        # 因果发现网络
        self.causal_discovery = nn.Sequential(
            nn.Linear(causal_dim, causal_dim),
            nn.LayerNorm(causal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(causal_dim, causal_dim * causal_dim),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(causal_dim, causal_dim),
            nn.LayerNorm(causal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(causal_dim, causal_dim)
        )
        
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, clinical_features):
        batch_size = clinical_features.size(0)
        
        # 编码
        encoded = self.encoder(clinical_features)
        mu, logvar = torch.chunk(encoded, 2, dim=-1)
        
        # 重参数化
        z = self.reparameterize(mu, logvar)
        
        # 自注意力处理
        z_seq = z.unsqueeze(1)  # [B, 1, causal_dim]
        
        for i, (attention, ffn, norms) in enumerate(zip(
            self.attention_layers, self.ffn_layers, self.norm_layers
        )):
            # 自注意力
            attn_out, _ = attention(z_seq, z_seq, z_seq)
            z_seq = norms[0](z_seq + attn_out)
            
            # 前馈网络
            ffn_out = ffn(z_seq)
            z_seq = norms[1](z_seq + ffn_out)
        
        # 提取特征
        causal_features = z_seq.squeeze(1)  # [B, causal_dim]
        
        # 因果发现
        causal_matrix = self.causal_discovery(causal_features)
        causal_matrix = causal_matrix.view(batch_size, self.causal_dim, self.causal_dim)
        
        # 去除自环
        causal_matrix = causal_matrix * (1 - torch.eye(self.causal_dim, device=causal_matrix.device))
        
        # 输出投影
        output = self.output_projection(causal_features)
        
        return {
            'causal_features': output,
            'causal_matrix': causal_matrix,
            'mu': mu,
            'logvar': logvar
        }

class EnhancedImageEncoder(nn.Module):
    """增强的图像编码器"""
    
    def __init__(self, embed_dim: int = 768, pretrained: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 使用预训练的ViT作为骨干网络
        import timm
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0  # 移除分类头
        )
        
        # 特征维度调整
        backbone_dim = self.backbone.num_features
        if backbone_dim != embed_dim:
            self.projection = nn.Sequential(
                nn.Linear(backbone_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        else:
            self.projection = nn.Identity()
        
        # SE注意力
        self.se_block = SEBlock(embed_dim)
        
    def forward(self, x):
        # 提取特征
        features = self.backbone(x)  # [B, backbone_dim]
        
        # 投影到目标维度
        features = self.projection(features)  # [B, embed_dim]
        
        # 添加空间维度用于SE注意力
        features_2d = features.unsqueeze(-1).unsqueeze(-1)  # [B, embed_dim, 1, 1]
        features_enhanced = self.se_block(features_2d).squeeze(-1).squeeze(-1)
        
        return features_enhanced

class EnhancedMultimodalFusion(nn.Module):
    """增强的多模态融合模块"""
    
    def __init__(
        self,
        embed_dim: int = 768,
        causal_dim: int = 128,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.causal_dim = causal_dim
        
        # 模态对齐
        self.oct_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.col_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.clinical_projection = nn.Sequential(
            nn.Linear(causal_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 跨模态注意力
        self.cross_modal_attention = MultiHeadCrossModalAttention(
            embed_dim, num_heads, dropout
        )
        
        # 门控融合
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.Sigmoid()
        )
        
        # 融合后的处理
        self.fusion_processor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, oct_features, col_features, clinical_features):
        # 模态对齐
        oct_aligned = self.oct_projection(oct_features)
        col_aligned = self.col_projection(col_features)
        clinical_aligned = self.clinical_projection(clinical_features)
        
        # 堆叠特征用于跨模态注意力
        stacked_features = torch.stack([
            oct_aligned, col_aligned, clinical_aligned
        ], dim=1)  # [B, 3, embed_dim]
        
        # 跨模态注意力
        fused_features, attn_weights = self.cross_modal_attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # 门控加权
        gate_input = torch.cat([oct_aligned, col_aligned, clinical_aligned], dim=-1)
        gate_weights = self.gate_network(gate_input)
        
        # 应用门控
        gated_features = fused_features * gate_weights.unsqueeze(1)
        
        # 模态平均
        final_features = gated_features.mean(dim=1)  # [B, embed_dim]
        
        # 最终处理
        output = self.fusion_processor(final_features)
        
        return output, attn_weights

class EnhancedCausalMultimodalTransformer(nn.Module):
    """增强的因果多模态Transformer"""
    
    def __init__(
        self,
        clinical_dim: int = 8,
        causal_dim: int = 128,
        embed_dim: int = 768,
        num_classes: int = 2,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # 图像编码器
        self.oct_encoder = EnhancedImageEncoder(embed_dim)
        self.col_encoder = EnhancedImageEncoder(embed_dim)
        
        # 因果GNN
        self.causal_gnn = EnhancedCausalGNN(
            clinical_dim, causal_dim, causal_dim * 2, num_heads, 3, dropout
        )
        
        # 多模态融合
        self.multimodal_fusion = EnhancedMultimodalFusion(
            embed_dim, causal_dim, num_heads, dropout
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, num_classes)
        )
        
        # 辅助输出（用于多任务学习）
        self.aux_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, num_classes)
        )
        
    def forward(self, oct_images, col_images, clinical_features):
        # 图像特征提取
        oct_features = self.oct_encoder(oct_images)
        col_features = self.col_encoder(col_images)
        
        # 因果特征提取
        causal_output = self.causal_gnn(clinical_features)
        clinical_features = causal_output['causal_features']
        
        # 多模态融合
        fused_features, attn_weights = self.multimodal_fusion(
            oct_features, col_features, clinical_features
        )
        
        # 主分类
        main_logits = self.classifier(fused_features)
        
        # 辅助分类
        aux_logits = self.aux_classifier(fused_features)
        
        return {
            'main_logits': main_logits,
            'aux_logits': aux_logits,
            'fused_features': fused_features,
            'causal_output': causal_output,
            'attn_weights': attn_weights
        }

class FocalLoss(nn.Module):
    """焦点损失函数，处理类别不平衡"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    """组合损失函数"""
    
    def __init__(
        self,
        num_classes: int = 2,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        causal_weight: float = 0.1,
        aux_weight: float = 0.3
    ):
        super().__init__()
        self.focal_loss = FocalLoss(focal_alpha, focal_gamma)
        self.ce_loss = nn.CrossEntropyLoss()
        self.causal_weight = causal_weight
        self.aux_weight = aux_weight
        
    def forward(self, outputs, targets, causal_output=None):
        # 主任务损失
        main_loss = self.focal_loss(outputs['main_logits'], targets)
        
        # 辅助任务损失
        aux_loss = self.ce_loss(outputs['aux_logits'], targets)
        
        # 因果损失
        causal_loss = 0.0
        if causal_output is not None:
            # KL散度损失
            mu, logvar = causal_output['mu'], causal_output['logvar']
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / mu.size(0)  # 归一化
            
            # 因果稀疏性损失
            causal_matrix = causal_output['causal_matrix']
            sparsity_loss = torch.mean(torch.abs(causal_matrix))
            
            causal_loss = kl_loss + 0.01 * sparsity_loss
        
        # 总损失
        total_loss = main_loss + self.aux_weight * aux_loss + self.causal_weight * causal_loss
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'aux_loss': aux_loss,
            'causal_loss': causal_loss
        }

def create_enhanced_model(config: Dict) -> EnhancedCausalMultimodalTransformer:
    """创建增强模型"""
    
    model = EnhancedCausalMultimodalTransformer(
        clinical_dim=config.get('clinical_dim', 8),
        causal_dim=config.get('causal_dim', 128),
        embed_dim=config.get('embed_dim', 768),
        num_classes=config.get('num_classes', 2),
        num_heads=config.get('num_heads', 12),
        dropout=config.get('dropout', 0.1)
    )
    
    return model

if __name__ == "__main__":
    # 测试模型
    config = {
        'clinical_dim': 8,
        'causal_dim': 128,
        'embed_dim': 768,
        'num_classes': 2,
        'num_heads': 12,
        'dropout': 0.1
    }
    
    model = create_enhanced_model(config)
    
    # 测试前向传播
    batch_size = 4
    oct_images = torch.randn(batch_size, 3, 224, 224)
    col_images = torch.randn(batch_size, 3, 224, 224)
    clinical_features = torch.randn(batch_size, 8)
    
    outputs = model(oct_images, col_images, clinical_features)
    
    print(f"主分类输出形状: {outputs['main_logits'].shape}")
    print(f"辅助分类输出形状: {outputs['aux_logits'].shape}")
    print(f"融合特征形状: {outputs['fused_features'].shape}")
    print(f"因果矩阵形状: {outputs['causal_output']['causal_matrix'].shape}")
    
    # 测试损失函数
    targets = torch.randint(0, 2, (batch_size,))
    loss_fn = CombinedLoss()
    losses = loss_fn(outputs, targets, outputs['causal_output'])
    
    print(f"总损失: {losses['total_loss']:.4f}")
    print(f"主损失: {losses['main_loss']:.4f}")
    print(f"辅助损失: {losses['aux_loss']:.4f}")
    print(f"因果损失: {losses['causal_loss']:.4f}")