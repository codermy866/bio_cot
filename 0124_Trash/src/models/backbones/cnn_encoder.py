import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力模块"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class DSConvBlock(nn.Module):
    """深度可分离卷积块"""
    def __init__(self, channels: int, drop: float = 0.0):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.se = SEBlock(channels)
        self.dropout = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dw(x)
        out = self.pw(out)
        out = self.bn(out)
        out = self.act(out)
        out = self.se(out)
        return self.dropout(out) + x


class ImageEncoder(nn.Module):
    """
    图像编码流程:
    输入 [B, K, C, H, W] → Stem(7x7卷积) → 5层DSConvBlock → 自适应池化 → 线性投影 → 注意力聚合
    """
    def __init__(
        self, 
        in_channels: int = 3, 
        base_dim: int = 64, 
        embed_dim: int = 768,
        dropout: float = 0.1,
        num_frames: int = None  # K: 图像序列长度（OCT为48，COL为3）
    ):
        super().__init__()
        self.num_frames = num_frames
        self.base_dim = base_dim
        
        # Stem层: 7x7卷积
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 增强到7层DSConvBlock，每层包含更多block以提升模型容量
        current_channels = base_dim
        self.stages = nn.ModuleList()
        
        # 每层的block数量配置：前几层少一些，后几层多一些（更深的特征）
        blocks_per_stage = [3, 3, 4, 4, 5, 5, 6]  # 总共30个block
        
        for i in range(7):  # 从5层增加到7层
            stage_layers = nn.ModuleList()
            
            # 每个Stage包含更多DSConvBlock以提升容量
            num_blocks = blocks_per_stage[i]
            for _ in range(num_blocks):
                stage_layers.append(DSConvBlock(current_channels, dropout))
            
            # 除了最后一个Stage，其他Stage都有下采样
            if i < 6:
                # 下采样：通道数增长（前几层翻倍，后几层增加50%）
                if i < 3:
                    next_channels = current_channels * 2
                else:
                    next_channels = int(current_channels * 1.5)
                stage_layers.append(
                    nn.Sequential(
                        nn.Conv2d(current_channels, next_channels, kernel_size=1, stride=2, bias=False),
                        nn.BatchNorm2d(next_channels)
                    )
                )
                current_channels = next_channels
            
            self.stages.append(stage_layers)
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        
        # 线性投影
        self.proj = nn.Linear(current_channels, embed_dim)
        
        # 增强的注意力聚合（用于处理多帧输入）
        if num_frames is not None and num_frames > 1:
            # 使用更多注意力头以提升多帧交互能力
            num_attn_heads = min(embed_dim // 64, 16)  # 自适应头数，最多16
            self.frame_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_attn_heads,
                dropout=dropout,
                batch_first=True
            )
            # 添加额外的层归一化和残差连接
            self.attn_norm = nn.LayerNorm(embed_dim)
        else:
            self.frame_attention = None
            self.attn_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 或 [B, K, C, H, W] (K为图像序列长度)
        Returns:
            features: [B, embed_dim] 或 [B, K, embed_dim]
        """
        # 处理多帧输入
        if x.dim() == 5:  # [B, K, C, H, W]
            B, K, C, H, W = x.shape
            x = x.view(B * K, C, H, W)  # [B*K, C, H, W]
            need_reshape = True
        else:
            need_reshape = False
        
        # Stem层
        x = self.stem(x)  # [B*K, base_dim, H', W']
        
        # 5层DSConvBlock
        for stage in self.stages:
            for layer in stage:
                x = layer(x)
        
        # 自适应池化
        x = self.adaptive_pool(x)  # [B*K, current_channels, 1, 1]
        x = x.flatten(1)  # [B*K, current_channels]
        
        # 线性投影
        x = self.proj(x)  # [B*K, embed_dim]
        
        # 增强的注意力聚合（如果有多个帧）
        if need_reshape:
            x = x.view(B, K, -1)  # [B, K, embed_dim]
            if self.frame_attention is not None:
                # 使用残差连接和层归一化
                residual = x
                x = self.attn_norm(x)
                x, _ = self.frame_attention(x, x, x)  # [B, K, embed_dim]
                x = x + residual  # 残差连接
                # 加权聚合所有帧（使用注意力权重）
                x = x.mean(dim=1)  # [B, embed_dim]
            else:
                # 简单平均聚合
                x = x.mean(dim=1)  # [B, embed_dim]
        
        return x  # [B, embed_dim]


class ClinicalEncoder(nn.Module):
    """临床特征编码器"""
    def __init__(self, input_dim: int = 7, embed_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalFusion(nn.Module):
    """
    跨模态融合流程:
    OCT特征 + Colposcopy特征 + 临床特征 → 堆叠为tokens → 自注意力 → 
    OCT-COL交叉注意力 → 图像-临床交叉注意力 → 前馈网络 → 自适应权重融合
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        ff_dim: int = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        ff_dim = ff_dim or embed_dim * 4
        
        # 自注意力（用于所有tokens）
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # OCT-COL交叉注意力（图像模态间交互）
        self.oct_col_cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 图像-临床交叉注意力（图像与临床信息交互）
        self.image_clinical_cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        
        # 自适应权重融合
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),  # 3个模态特征
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        oct_feat: torch.Tensor,  # [B, embed_dim]
        col_feat: torch.Tensor,  # [B, embed_dim]
        clinical_feat: torch.Tensor  # [B, embed_dim]
    ) -> torch.Tensor:
        """
        跨模态融合流程
        """
        B = oct_feat.size(0)
        
        # 堆叠为tokens [B, 3, embed_dim]
        tokens = torch.stack([oct_feat, col_feat, clinical_feat], dim=1)
        
        # 1. 自注意力（所有tokens相互关注）
        tokens_norm = self.norm1(tokens)
        attn_out, _ = self.self_attention(tokens_norm, tokens_norm, tokens_norm)
        tokens = tokens + self.dropout(attn_out)
        
        # 2. OCT-COL交叉注意力（图像模态间交互）
        # 图像tokens: [B, 2, embed_dim] (OCT + COL)
        image_tokens = tokens[:, :2, :]  # [B, 2, embed_dim]
        image_tokens_norm = self.norm2(image_tokens)
        # OCT作为Query，COL作为Key和Value（反之亦可）
        oct_query = image_tokens_norm[:, 0:1, :]  # [B, 1, embed_dim]
        col_kv = image_tokens_norm[:, 1:2, :]  # [B, 1, embed_dim]
        cross_attn_out, _ = self.oct_col_cross_attention(oct_query, col_kv, col_kv)
        
        # 更新OCT特征
        updated_oct = tokens[:, 0:1, :] + self.dropout(cross_attn_out)
        # 更新tokens
        tokens = torch.cat([updated_oct, tokens[:, 1:2, :], tokens[:, 2:3, :]], dim=1)
        
        # 3. 图像-临床交叉注意力（图像与临床信息交互）
        # 图像特征作为Query，临床特征作为Key和Value
        image_feat = tokens[:, :2, :].mean(dim=1, keepdim=True)  # [B, 1, embed_dim] 平均图像特征
        clinical_token = tokens[:, 2:3, :]  # [B, 1, embed_dim]
        
        image_clinical_norm = self.norm3(torch.cat([image_feat, clinical_token], dim=1))
        image_query = image_clinical_norm[:, 0:1, :]  # [B, 1, embed_dim]
        clinical_kv = image_clinical_norm[:, 1:2, :]  # [B, 1, embed_dim]
        
        img_clin_attn_out, _ = self.image_clinical_cross_attention(
            image_query, clinical_kv, clinical_kv
        )
        
        # 更新图像特征
        updated_image = image_feat + self.dropout(img_clin_attn_out)
        # 更新tokens: 将更新的图像特征分配给OCT和COL
        tokens = torch.cat([
            updated_image.expand(-1, 2, -1),  # [B, 2, embed_dim] 复制给OCT和COL
            tokens[:, 2:3, :]  # 临床特征保持不变
        ], dim=1)
        
        # 4. 前馈网络
        tokens_norm = self.norm4(tokens)
        ffn_out = self.ffn(tokens_norm)  # [B, 3, embed_dim]
        tokens = tokens + self.dropout(ffn_out)
        
        # 5. 自适应权重融合
        # 展平所有tokens特征
        flat_tokens = tokens.view(B, -1)  # [B, 3 * embed_dim]
        fused = self.adaptive_fusion(flat_tokens)  # [B, embed_dim]
        
        return fused


class CNNMultimodalTransformer(nn.Module):
    """
    CNN多模态Transformer模型:
    1. 图像编码流程: [B, K, C, H, W] → Stem → 5层DSConvBlock → 池化 → 投影 → 注意力聚合
    2. 跨模态融合流程: 堆叠tokens → 自注意力 → OCT-COL交叉注意力 → 图像-临床交叉注意力 → FFN → 自适应融合
    """
    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        clinical_dim: int = 7,
        oct_num_frames: int = 48,
        col_num_frames: int = 3
    ):
        super().__init__()
        
        # 图像编码器（大幅增大base_dim以提升模型容量和性能）
        self.oct_encoder = ImageEncoder(
            in_channels=3,
            base_dim=128,  # 从96增加到128，大幅提升特征提取能力
            embed_dim=embed_dim,
            dropout=dropout,
            num_frames=oct_num_frames
        )
        
        self.col_encoder = ImageEncoder(
            in_channels=3,
            base_dim=128,  # 从96增加到128，大幅提升特征提取能力
            embed_dim=embed_dim,
            dropout=dropout,
            num_frames=col_num_frames
        )
        
        # 临床特征编码器
        self.clinical_encoder = ClinicalEncoder(
            input_dim=clinical_dim,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # 跨模态融合模块
        self.cross_modal_fusion = CrossModalFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 增强的分类器（更多dropout防止过拟合）
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 1.5),  # 增加dropout
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 1.5),  # 增加dropout
            nn.Linear(embed_dim // 4, embed_dim // 8),
            nn.LayerNorm(embed_dim // 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 8, num_classes)
        )
    
    def forward(
        self,
        oct_images: torch.Tensor,
        col_images: torch.Tensor,
        clinical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            oct_images: [B, C, H, W] 或 [B, T, C, H, W] (T=oct_num_frames)
            col_images: [B, C, H, W] 或 [B, K, C, H, W] (K=col_num_frames)
            clinical_features: [B, clinical_dim]
        Returns:
            logits: [B, num_classes]
        """
        # 图像编码流程
        oct_feat = self.oct_encoder(oct_images)  # [B, embed_dim]
        col_feat = self.col_encoder(col_images)  # [B, embed_dim]
        
        # 临床特征编码
        clinical_feat = self.clinical_encoder(clinical_features)  # [B, embed_dim]
        
        # 跨模态融合流程
        fused_feat = self.cross_modal_fusion(oct_feat, col_feat, clinical_feat)  # [B, embed_dim]
        
        # 分类
        logits = self.classifier(fused_feat)  # [B, num_classes]
        
        return logits
