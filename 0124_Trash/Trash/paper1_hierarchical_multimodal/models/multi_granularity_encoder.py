"""
多粒度特征编码器
提取局部、全局、时序、空间等不同粒度的特征
使用预训练的ResNet作为backbone以提升特征提取能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LocalFeatureEncoder(nn.Module):
    """
    局部特征编码器
    提取细粒度的局部特征（如病变边缘、纹理等）
    使用预训练ResNet34的前几层提取局部特征
    """
    def __init__(self, in_channels=3, embed_dim=768, use_pretrained=True):
        super().__init__()
        # 使用预训练的ResNet34作为backbone
        resnet = models.resnet34(pretrained=use_pretrained)
        # 移除最后的全连接层和平均池化层
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # 动态检测backbone的输出通道数
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            backbone_out_channels = dummy_output.size(1)
        # ResNet34的完整backbone输出是512通道
        self.local_conv = nn.Sequential(
            nn.Conv2d(backbone_out_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((14, 14))  # 保持空间信息
        )
        self.projection = nn.Sequential(
            nn.Linear(512 * 14 * 14, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 或 [B, K, C, H, W] (K是帧数)
        Returns:
            local_feat: [B, embed_dim] 或 [B, K, embed_dim]
        """
        # 处理多帧情况
        if x.dim() == 5:
            B, K, C, H, W = x.shape
            x = x.view(B * K, C, H, W)
            # 通过backbone提取特征
            feat = self.backbone(x)
            # 通过局部卷积和投影
            local_feat = self.local_conv(feat)
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_feat = self.projection(local_feat)
            local_feat = local_feat.view(B, K, -1)
            # 平均池化多帧
            local_feat = local_feat.mean(dim=1)  # [B, embed_dim]
        else:
            feat = self.backbone(x)
            local_feat = self.local_conv(feat)
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_feat = self.projection(local_feat)
        return local_feat


class GlobalFeatureEncoder(nn.Module):
    """
    全局特征编码器
    提取全局的语义特征（如整体病变分布、形状等）
    使用预训练ResNet50提取全局特征
    """
    def __init__(self, in_channels=3, embed_dim=768, use_pretrained=True):
        super().__init__()
        # 使用预训练的ResNet50作为backbone
        resnet = models.resnet50(pretrained=use_pretrained)
        # 移除最后的全连接层和平均池化层
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # ResNet50的layer4输出是2048通道
        self.global_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局池化
            nn.Flatten()
        )
        self.projection = nn.Sequential(
            nn.Linear(2048, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 或 [B, K, C, H, W]
        Returns:
            global_feat: [B, embed_dim]
        """
        # 处理多帧情况
        if x.dim() == 5:
            B, K, C, H, W = x.shape
            x = x.view(B * K, C, H, W)
            # 通过backbone提取特征
            feat = self.backbone(x)
            # 通过全局卷积和投影
            global_feat = self.global_conv(feat)
            global_feat = self.projection(global_feat)
            global_feat = global_feat.view(B, K, -1)
            # 平均池化多帧
            global_feat = global_feat.mean(dim=1)  # [B, embed_dim]
        else:
            feat = self.backbone(x)
            global_feat = self.global_conv(feat)
            global_feat = self.projection(global_feat)
        return global_feat


class TemporalFeatureEncoder(nn.Module):
    """
    时序特征编码器（用于OCT序列）
    提取时序动态特征
    """
    def __init__(self, embed_dim=768, hidden_dim=384, num_layers=2):
        super().__init__()
        # 使用LSTM提取时序特征
        self.temporal_encoder = nn.LSTM(
            input_size=embed_dim,  # 假设每帧已经编码为embed_dim维
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        self.projection = nn.Linear(hidden_dim * 2, embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, embed_dim]  T是时间步数（如48帧）
        Returns:
            temporal_feat: [B, embed_dim]
        """
        # LSTM编码
        temporal_feat, _ = self.temporal_encoder(x)
        # 使用最后时刻的输出
        temporal_feat = temporal_feat[:, -1, :]
        temporal_feat = self.projection(temporal_feat)
        return temporal_feat


class SpatialFeatureEncoder(nn.Module):
    """
    空间特征编码器（用于Colposcopy多视图）
    提取空间关系特征
    """
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        # 使用自注意力提取空间关系
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, N, embed_dim]  N是视图数（如3个视图）
        Returns:
            spatial_feat: [B, embed_dim]
        """
        # 自注意力提取空间关系
        spatial_feat, _ = self.spatial_attention(x, x, x)
        # 平均池化
        spatial_feat = spatial_feat.mean(dim=1)
        spatial_feat = self.projection(spatial_feat)
        return spatial_feat


class SemanticFeatureEncoder(nn.Module):
    """
    语义特征编码器（用于临床特征）
    提取语义特征
    """
    def __init__(self, input_dim=8, embed_dim=768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, input_dim]
        Returns:
            semantic_feat: [B, embed_dim]
        """
        return self.encoder(x)


class StatisticalFeatureEncoder(nn.Module):
    """
    统计特征编码器（用于临床特征）
    提取统计特征（如均值、方差等）
    """
    def __init__(self, input_dim=8, embed_dim=768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, input_dim]
        Returns:
            statistical_feat: [B, embed_dim]
        """
        # 可以添加统计特征（均值、方差等）
        mean_feat = x.mean(dim=-1, keepdim=True).expand(-1, x.size(-1))
        std_feat = x.std(dim=-1, keepdim=True).expand(-1, x.size(-1))
        stat_feat = torch.cat([x, mean_feat, std_feat], dim=-1)
        # 如果维度不匹配，只使用原始特征
        return self.encoder(x)

