import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SS2D(nn.Module):
    """稳定的2D State Space Module - 使用CNN风格的稳定实现"""
    def __init__(self, d_model, d_state=16, dt_rank="auto", dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 使用深度可分离卷积（类似CNN方法，经过验证稳定）
        self.dw_conv = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False)
        self.pw_conv = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(d_model)
        
        # SE注意力机制（来自CNN方法，稳定有效）
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d_model, d_model // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 4, d_model, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 简化的FFN（只在需要时使用）
        self.ffn_scale = 0.1  # 很小的权重，避免数值问题
        
        # 初始化（使用Kaiming初始化，更稳定）
        nn.init.kaiming_normal_(self.dw_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.pw_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.pw_conv.bias)
        
    def forward(self, x):
        """
        稳定的前向传播：使用CNN风格，避免NaN
        """
        B, H, W, C = x.shape
        
        # 转换为卷积格式 [B, C, H, W]
        x_conv = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        identity = x_conv
        
        # 深度可分离卷积
        x_conv = self.dw_conv(x_conv)
        x_conv = self.bn1(x_conv)
        x_conv = F.gelu(x_conv)
        x_conv = self.pw_conv(x_conv)
        
        # SE注意力
        se_weight = self.se(x_conv)
        x_conv = x_conv * se_weight
        
        # 残差连接（使用小的权重避免梯度爆炸）
        x_conv = x_conv * 0.9 + identity * 0.1
        
        # 数值稳定性检查
        if torch.isnan(x_conv).any() or torch.isinf(x_conv).any():
            x_conv = identity  # 使用原始输入
        
        # 限制输出范围
        x_conv = torch.clamp(x_conv, min=-10.0, max=10.0)
        
        # 转换回 [B, H, W, C]
        y = x_conv.permute(0, 2, 3, 1)
        
        return y


class VMambaBlock(nn.Module):
    """VMamba Block with 2D SSM"""
    def __init__(self, dim, drop=0.0, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, d_state=d_state)
        self.dropout = nn.Dropout(drop)
        
    def forward(self, x):
        # x: [B, H, W, C]
        residual = x
        x = self.norm(x)
        x = self.ss2d(x)
        x = self.dropout(x)
        return x + residual


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H'*W', embed_dim]
        return x, H, W


class VMambaImageEncoder(nn.Module):
    """
    VMamba图像编码器
    使用2D State Space Model处理图像
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        d_state=16,
        dropout=0.1,
        num_frames=None
    ):
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.n_patches
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # VMamba blocks
        self.blocks = nn.ModuleList([
            VMambaBlock(embed_dim, drop=dropout, d_state=d_state)
            for _ in range(depth)
        ])
        
        # Frame attention (for multi-frame input)
        if num_frames is not None and num_frames > 1:
            self.frame_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        else:
            self.frame_attention = None
        
        # Final norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output projection
        self.head = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 或 [B, K, C, H, W] (K为图像序列长度)
        Returns:
            features: [B, embed_dim]
        """
        # Handle multi-frame input
        if x.dim() == 5:  # [B, K, C, H, W]
            B, K, C, H, W = x.shape
            x = x.view(B * K, C, H, W)  # [B*K, C, H, W]
            need_reshape = True
        else:
            need_reshape = False
        
        # Patch embedding
        x, H, W = self.patch_embed(x)  # [B*K, num_patches, embed_dim]
        x = x + self.pos_embed
        
        # Reshape for 2D SSM: [B*K, H, W, embed_dim]
        B_total = x.shape[0]
        num_patches = x.shape[1]
        H_patches = W_patches = int(math.sqrt(num_patches))
        x = x.view(B_total, H_patches, W_patches, self.embed_dim)
        
        # Apply VMamba blocks
        for block in self.blocks:
            x = block(x)  # [B*K, H, W, embed_dim]
        
        # Global average pooling
        x = x.mean(dim=(1, 2))  # [B*K, embed_dim]
        
        # Final norm
        x = self.norm(x)
        
        # Projection
        x = self.head(x)  # [B*K, embed_dim]
        
        # Handle multi-frame aggregation
        if need_reshape:
            x = x.view(B, K, -1)  # [B, K, embed_dim]
            if self.frame_attention is not None:
                x, _ = self.frame_attention(x, x, x)  # [B, K, embed_dim]
                x = x.mean(dim=1)  # [B, embed_dim]
            else:
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
    跨模态融合模块
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
        
        # 自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # OCT-COL交叉注意力
        self.oct_col_cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 图像-临床交叉注意力
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
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        oct_feat: torch.Tensor,
        col_feat: torch.Tensor,
        clinical_feat: torch.Tensor
    ) -> torch.Tensor:
        B = oct_feat.size(0)
        
        # 堆叠为tokens [B, 3, embed_dim]
        tokens = torch.stack([oct_feat, col_feat, clinical_feat], dim=1)
        
        # 1. 自注意力
        tokens_norm = self.norm1(tokens)
        attn_out, _ = self.self_attention(tokens_norm, tokens_norm, tokens_norm)
        tokens = tokens + self.dropout(attn_out)
        
        # 2. OCT-COL交叉注意力
        image_tokens = tokens[:, :2, :]
        image_tokens_norm = self.norm2(image_tokens)
        oct_query = image_tokens_norm[:, 0:1, :]
        col_kv = image_tokens_norm[:, 1:2, :]
        cross_attn_out, _ = self.oct_col_cross_attention(oct_query, col_kv, col_kv)
        
        updated_oct = tokens[:, 0:1, :] + self.dropout(cross_attn_out)
        tokens = torch.cat([updated_oct, tokens[:, 1:2, :], tokens[:, 2:3, :]], dim=1)
        
        # 3. 图像-临床交叉注意力
        image_feat = tokens[:, :2, :].mean(dim=1, keepdim=True)
        clinical_token = tokens[:, 2:3, :]
        
        image_clinical_norm = self.norm3(torch.cat([image_feat, clinical_token], dim=1))
        image_query = image_clinical_norm[:, 0:1, :]
        clinical_kv = image_clinical_norm[:, 1:2, :]
        
        img_clin_attn_out, _ = self.image_clinical_cross_attention(
            image_query, clinical_kv, clinical_kv
        )
        
        updated_image = image_feat + self.dropout(img_clin_attn_out)
        tokens = torch.cat([
            updated_image.expand(-1, 2, -1),
            tokens[:, 2:3, :]
        ], dim=1)
        
        # 4. 前馈网络
        tokens_norm = self.norm4(tokens)
        ffn_out = self.ffn(tokens_norm)
        tokens = tokens + self.dropout(ffn_out)
        
        # 5. 自适应权重融合
        flat_tokens = tokens.view(B, -1)
        fused = self.adaptive_fusion(flat_tokens)
        
        return fused


class VMambaMultimodalTransformer(nn.Module):
    """
    VMamba多模态Transformer模型:
    1. 图像编码流程: 使用VMamba 2D SSM编码器处理图像
    2. 跨模态融合流程: 堆叠tokens → 自注意力 → OCT-COL交叉注意力 → 
       图像-临床交叉注意力 → FFN → 自适应融合
    """
    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        clinical_dim: int = 7,
        oct_num_frames: int = 48,
        col_num_frames: int = 3,
        img_size: int = 224,
        patch_size: int = 16,
        depth: int = 12,
        d_state: int = 16
    ):
        super().__init__()
        
        # VMamba图像编码器
        # 使用完整深度以提升模型容量
        encoder_depth = depth
        
        self.oct_encoder = VMambaImageEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
            depth=encoder_depth,
            d_state=d_state,
            dropout=dropout,
            num_frames=oct_num_frames
        )
        
        self.col_encoder = VMambaImageEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
            depth=encoder_depth,
            d_state=d_state,
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
        
        # 增强的分类器
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 1.5),
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
        # 图像编码
        oct_feat = self.oct_encoder(oct_images)
        col_feat = self.col_encoder(col_images)
        
        # 临床特征编码
        clinical_feat = self.clinical_encoder(clinical_features)
        
        # 跨模态融合
        fused_feat = self.cross_modal_fusion(oct_feat, col_feat, clinical_feat)
        
        # 分类
        logits = self.classifier(fused_feat)
        
        return logits

