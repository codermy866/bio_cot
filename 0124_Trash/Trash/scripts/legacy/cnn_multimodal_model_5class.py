#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5分类多模态模型
基于CNN的多模态5分类模型，支持OCT、Colposcopy和临床特征
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    """改进版 CNN 编码器（Mobile-SE 残差块 + 深度可分离卷积 + 下采样）
    输出: [B, embed_dim]
    """

    def __init__(self, in_channels: int = 3, base_dim: int = 64, depth: int = 5, embed_dim: int = 768, dropout: float = 0.1):
        super().__init__()

        class SEBlock(nn.Module):
            def __init__(self, channels: int, reduction: int = 16):
                super().__init__()
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Sequential(
                    nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=True),
                    nn.GELU(),
                    nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True),
                    nn.Sigmoid(),
                )
            def forward(self, x):
                w = self.pool(x)
                w = self.fc(w)
                return x * w

        class DSConvBlock(nn.Module):
            def __init__(self, channels: int, drop: float = 0.0):
                super().__init__()
                self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
                self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
                self.bn = nn.BatchNorm2d(channels)
                self.act = nn.GELU()
                self.se = SEBlock(channels)
                self.dropout = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
            def forward(self, x):
                out = self.dw(x)
                out = self.pw(out)
                out = self.bn(out)
                out = self.act(out)
                out = self.se(out)
                return self.dropout(out) + x

        # 构建网络
        layers = []
        current_channels = in_channels
        
        # 初始卷积
        layers.append(nn.Conv2d(current_channels, base_dim, kernel_size=7, stride=2, padding=3, bias=False))
        layers.append(nn.BatchNorm2d(base_dim))
        layers.append(nn.GELU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        # 残差块
        for i in range(depth):
            if i > 0:
                layers.append(nn.Conv2d(base_dim, base_dim * 2, kernel_size=1, stride=2, bias=False))
                layers.append(nn.BatchNorm2d(base_dim * 2))
                base_dim *= 2
            
            layers.append(DSConvBlock(base_dim, dropout))
            layers.append(DSConvBlock(base_dim, dropout))
        
        # 全局平均池化
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        
        # 投影到embed_dim
        layers.append(nn.Linear(base_dim, embed_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ClinicalEncoder(nn.Module):
    """临床特征编码器"""
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.encoder(x)

class EnhancedCrossModalAttention(nn.Module):
    """增强的跨模态注意力机制（支持5分类）"""
    
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1, 
                 use_causal_adjustment: bool = False):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_causal_adjustment = use_causal_adjustment
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # 多头注意力
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        # 因果调整（可选）
        if use_causal_adjustment:
            self.causal_proj = nn.Linear(embed_dim, embed_dim)
            self.causal_gate = nn.Sequential(
                nn.Linear(embed_dim, 1),
                nn.Sigmoid()
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, oct_features, col_features, clinical_features):
        """
        Args:
            oct_features: [B, embed_dim] OCT特征
            col_features: [B, embed_dim] Colposcopy特征  
            clinical_features: [B, embed_dim] 临床特征
        Returns:
            fused_features: [B, embed_dim] 融合后的特征
        """
        B = oct_features.size(0)
        
        # 拼接所有模态特征
        all_features = torch.stack([oct_features, col_features, clinical_features], dim=1)  # [B, 3, embed_dim]
        
        # 自注意力
        residual = all_features
        all_features = self.norm1(all_features)
        
        # 计算Q, K, V
        Q = self.q_proj(all_features).view(B, 3, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, 3, head_dim]
        K = self.k_proj(all_features).view(B, 3, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(all_features).view(B, 3, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, num_heads, 3, 3]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attended = torch.matmul(attn_weights, V)  # [B, num_heads, 3, head_dim]
        attended = attended.transpose(1, 2).contiguous().view(B, 3, self.embed_dim)  # [B, 3, embed_dim]
        attended = self.out_proj(attended)
        
        # 残差连接
        attended = attended + residual
        
        # 因果调整（可选）
        if self.use_causal_adjustment:
            causal_weight = self.causal_gate(attended)  # [B, 3, 1]
            causal_adjustment = self.causal_proj(attended)  # [B, 3, embed_dim]
            attended = attended + causal_weight * causal_adjustment
        
        # 前馈网络
        residual = attended
        attended = self.norm2(attended)
        attended = self.ffn(attended)
        attended = attended + residual
        
        # 平均池化得到最终特征
        fused_features = attended.mean(dim=1)  # [B, embed_dim]
        
        return fused_features

class CNNMultimodalTransformer5Class(nn.Module):
    """5分类CNN多模态Transformer"""
    
    def __init__(self, 
                 num_classes: int = 5,
                 embed_dim: int = 512,
                 clinical_dim: int = 8,
                 dropout: float = 0.1,
                 use_causal_adjustment: bool = False,
                 use_text_contrastive: bool = False):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_text_contrastive = use_text_contrastive
        
        # 编码器
        self.oct_encoder = ConvEncoder(in_channels=3, embed_dim=embed_dim, dropout=dropout)
        self.col_encoder = ConvEncoder(in_channels=3, embed_dim=embed_dim, dropout=dropout)
        self.clinical_encoder = ClinicalEncoder(input_dim=clinical_dim, embed_dim=embed_dim, dropout=dropout)
        
        # 跨模态注意力
        self.cross_modal_attn = EnhancedCrossModalAttention(
            embed_dim=embed_dim,
            dropout=dropout,
            use_causal_adjustment=use_causal_adjustment
        )
        
        # 融合层
        self.fusion_layers = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(embed_dim // 4, num_classes)
        )
        
        # 文本对比学习（可选）
        if use_text_contrastive:
            self.text_projection = nn.Linear(embed_dim, embed_dim)
            self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
        # 注意力池化（用于可视化）
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, oct_images, col_images, clinical_features, return_attention=False):
        """
        Args:
            oct_images: [B, T, C, H, W] OCT图像序列
            col_images: [B, N, C, H, W] Colposcopy图像
            clinical_features: [B, clinical_dim] 临床特征
            return_attention: 是否返回注意力权重
        Returns:
            logits: [B, num_classes] 分类logits
            attention_weights: 注意力权重（可选）
        """
        B = oct_images.size(0)
        
        # 编码OCT图像序列
        oct_features_list = []
        for t in range(oct_images.size(1)):
            oct_feat = self.oct_encoder(oct_images[:, t])  # [B, embed_dim]
            oct_features_list.append(oct_feat)
        
        # 注意力池化OCT特征
        oct_features_stack = torch.stack(oct_features_list, dim=1)  # [B, T, embed_dim]
        attention_weights = self.attention_pool(oct_features_stack)  # [B, T, 1]
        oct_features = (oct_features_stack * attention_weights).sum(dim=1)  # [B, embed_dim]
        
        # 编码Colposcopy图像
        col_features_list = []
        for n in range(col_images.size(1)):
            col_feat = self.col_encoder(col_images[:, n])  # [B, embed_dim]
            col_features_list.append(col_feat)
        
        # 平均池化Colposcopy特征
        col_features_stack = torch.stack(col_features_list, dim=1)  # [B, N, embed_dim]
        col_features = col_features_stack.mean(dim=1)  # [B, embed_dim]
        
        # 编码临床特征
        clinical_features = self.clinical_encoder(clinical_features)  # [B, embed_dim]
        
        # 跨模态注意力融合
        fused_features = self.cross_modal_attn(oct_features, col_features, clinical_features)  # [B, embed_dim]
        
        # 融合层
        fused_features = self.fusion_layers(fused_features)  # [B, embed_dim//2]
        
        # 分类
        logits = self.classifier(fused_features)  # [B, num_classes]
        
        if return_attention:
            return logits, attention_weights
        else:
            return logits
    
    def get_feature_importance(self, oct_images, col_images, clinical_features):
        """获取特征重要性（用于可解释性分析）"""
        with torch.no_grad():
            # 获取各模态特征
            oct_features_list = []
            for t in range(oct_images.size(1)):
                oct_feat = self.oct_encoder(oct_images[:, t])
                oct_features_list.append(oct_feat)
            oct_features_stack = torch.stack(oct_features_list, dim=1)
            attention_weights = self.attention_pool(oct_features_stack)
            oct_features = (oct_features_stack * attention_weights).sum(dim=1)
            
            col_features_list = []
            for n in range(col_images.size(1)):
                col_feat = self.col_encoder(col_images[:, n])
                col_features_list.append(col_feat)
            col_features_stack = torch.stack(col_features_list, dim=1)
            col_features = col_features_stack.mean(dim=1)
            
            clinical_features = self.clinical_encoder(clinical_features)
            
            # 计算特征重要性
            fused_features = self.cross_modal_attn(oct_features, col_features, clinical_features)
            fused_features = self.fusion_layers(fused_features)
            
            # 计算梯度（用于Grad-CAM）
            fused_features.requires_grad_(True)
            logits = self.classifier(fused_features)
            
            return {
                'oct_features': oct_features,
                'col_features': col_features,
                'clinical_features': clinical_features,
                'fused_features': fused_features,
                'logits': logits,
                'attention_weights': attention_weights
            }

def create_5class_model(num_classes=5, embed_dim=512, clinical_dim=8, **kwargs):
    """创建5分类模型的工厂函数"""
    return CNNMultimodalTransformer5Class(
        num_classes=num_classes,
        embed_dim=embed_dim,
        clinical_dim=clinical_dim,
        **kwargs
    )

# 测试模型
if __name__ == "__main__":
    # 创建模型
    model = create_5class_model(num_classes=5, embed_dim=512, clinical_dim=8)
    
    # 测试输入
    batch_size = 2
    oct_images = torch.randn(batch_size, 48, 3, 224, 224)  # OCT序列
    col_images = torch.randn(batch_size, 3, 3, 224, 224)   # Colposcopy图像
    clinical_features = torch.randn(batch_size, 8)         # 临床特征
    
    # 前向传播
    logits = model(oct_images, col_images, clinical_features)
    
    print(f"模型输出形状: {logits.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试特征重要性
    feature_info = model.get_feature_importance(oct_images, col_images, clinical_features)
    print(f"特征重要性分析完成")
    print(f"  OCT特征形状: {feature_info['oct_features'].shape}")
    print(f"  Colposcopy特征形状: {feature_info['col_features'].shape}")
    print(f"  临床特征形状: {feature_info['clinical_features'].shape}")
    print(f"  融合特征形状: {feature_info['fused_features'].shape}")
    print(f"  注意力权重形状: {feature_info['attention_weights'].shape}")
