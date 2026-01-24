import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as e:
    timm = None


class ViTImageEncoder(nn.Module):
    """
    使用 timm 的传统 ViT (Vision Transformer) 作为图像编码器
    支持单帧 [B, C, H, W] 或多帧 [B, T, C, H, W]（如 OCT 120 帧）
    - 输出统一为 [B, embed_dim]
    - 对多帧输入，在编码后用 Multi-Head Attention 做帧间聚合
    """

    def __init__(
        self,
        embed_dim: int = 768,
        dropout: float = 0.1,
        num_frames=None,
        model_name: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        input_size: int = 224,
        use_frame_attention: bool = True,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("timm 未安装，请先安装: pip install timm==0.9.12")

        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # 创建 ViT 主干（移除分类头，输出特征）
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 去掉分类头
            global_pool='',  # 不在内部做池化
            img_size=input_size
        )

        # 获取主干输出通道数
        if hasattr(self.backbone, 'num_features'):
            backbone_out = self.backbone.num_features
        elif hasattr(self.backbone, 'embed_dim'):
            backbone_out = self.backbone.embed_dim
        else:
            # 默认 ViT-Base 的输出维度是 768
            backbone_out = 768

        # ViT 返回序列格式 [B, N, C]，其中 N 是 patch 数量 + 1 (CLS token)
        self.proj = nn.Linear(backbone_out, embed_dim) if backbone_out != embed_dim else nn.Identity()

        # 帧间注意力（用于多帧输入聚合）
        if use_frame_attention and (num_frames is not None and num_frames > 1):
            num_attn_heads = max(1, min(embed_dim // 64, 16))
            self.frame_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(embed_dim)
        else:
            self.frame_attention = None

    def _encode_single(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        feat = self.backbone.forward_features(x)
        
        # ViT 通常返回 [B, N, C] 格式（序列格式，包含CLS token）
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        
        # 获取期望的输入维度
        if isinstance(self.proj, nn.Identity):
            expected_in_dim = self.embed_dim
        else:
            expected_in_dim = self.proj.in_features
        
        if feat.dim() == 3:
            # [B, N, C] - 序列格式，使用CLS token或全局平均池化
            # ViT通常第一个token是CLS token，使用它作为全局特征
            if feat.size(1) > 0:
                feat = feat[:, 0]  # 使用CLS token [B, C]
            else:
                feat = feat.mean(dim=1)  # 如果没有CLS token，使用平均池化
            
            # 确保维度匹配
            if feat.size(1) != expected_in_dim:
                if feat.size(1) < expected_in_dim:
                    pad = expected_in_dim - feat.size(1)
                    feat = F.pad(feat, (0, pad))
                else:
                    feat = feat[:, :expected_in_dim]
        elif feat.dim() == 4:
            # [B, C, h, w] - 空间格式（某些变体可能返回）
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)  # [B, C]
            if feat.size(1) != expected_in_dim:
                if feat.size(1) < expected_in_dim:
                    pad = expected_in_dim - feat.size(1)
                    feat = F.pad(feat, (0, pad))
                else:
                    feat = feat[:, :expected_in_dim]
        elif feat.dim() == 2:
            # 已经是 [B, C] 格式
            if feat.size(1) != expected_in_dim:
                if feat.size(1) < expected_in_dim:
                    pad = expected_in_dim - feat.size(1)
                    feat = F.pad(feat, (0, pad))
                else:
                    feat = feat[:, :expected_in_dim]
        
        # 投影到目标维度
        feat = self.proj(feat)  # [B, embed_dim]
        return feat

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, T, C, H, W] 或 [B, C, H, W]
        
        Returns:
            feat: [B, embed_dim]
        """
        if images.dim() == 4:
            # 单帧输入 [B, C, H, W]
            return self._encode_single(images)
        elif images.dim() == 5:
            # 多帧输入 [B, T, C, H, W]
            B, T, C, H, W = images.shape
            images = images.view(B * T, C, H, W)  # [B*T, C, H, W]
            
            # 编码每一帧
            feats = self._encode_single(images)  # [B*T, embed_dim]
            feats = feats.view(B, T, self.embed_dim)  # [B, T, embed_dim]
            
            # 帧间注意力聚合
            if self.frame_attention is not None:
                feats_attn, _ = self.frame_attention(feats, feats, feats)  # [B, T, embed_dim]
                feats = self.attn_norm(feats + feats_attn)  # 残差连接
            
            # 聚合多帧特征（平均池化）
            feat = feats.mean(dim=1)  # [B, embed_dim]
            return feat
        else:
            raise ValueError(f"不支持的输入维度: {images.dim()}, 期望 4 或 5")

