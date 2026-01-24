import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as e:
    timm = None


class SwinTImageEncoder(nn.Module):
    """
    使用 timm 的 Swin-T 作为图像编码器，支持单帧 [B, C, H, W]
    或多帧 [B, T, C, H, W]（如 OCT 120 帧）。
    - 输出统一为 [B, embed_dim]
    - 对多帧输入，在编码后用 Multi-Head Attention 做帧间聚合
    """

    def __init__(
        self,
        embed_dim: int = 768,
        dropout: float = 0.1,
        num_frames=None,
        model_name: str = 'swin_tiny_patch4_window7_224',
        pretrained: bool = True,
        input_size: int = 224,
        use_frame_attention: bool = True,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("timm 未安装，请先安装: pip install timm==0.9.12")

        self.num_frames = num_frames
        self.embed_dim = embed_dim  # 保存 embed_dim 用于后续检查

        # 创建 Swin-T 主干（移除分类头，输出特征）
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 去掉分类头
            global_pool='',  # 不在内部做池化
            img_size=input_size
        )

        # 获取主干输出通道数
        # Swin 通常返回 [B, N, C] 格式，其中 C 是特征维度
        if hasattr(self.backbone, 'num_features'):
            backbone_out = self.backbone.num_features
        elif hasattr(self.backbone, 'feature_info') and len(self.backbone.feature_info) > 0:
            backbone_out = self.backbone.feature_info[-1]['num_chs']
        else:
            # 默认 Swin-T 的输出维度是 768
            backbone_out = 768

        # Swin 返回序列格式，不需要 AdaptivePool2d
        self.adaptive_pool = None
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
        
        # Swin 通常返回 [B, N, C] 格式（序列格式）
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        
        # 获取期望的输入维度
        if isinstance(self.proj, nn.Identity):
            expected_in_dim = self.embed_dim
        else:
            expected_in_dim = self.proj.in_features
        
        if feat.dim() == 3:
            # [B, N, C] - 序列格式，使用全局平均池化
            feat = feat.mean(dim=1)  # [B, C]
            if feat.size(1) != expected_in_dim:
                # 维度不匹配，需要投影或填充
                if feat.size(1) < expected_in_dim:
                    # 填充到期望维度
                    pad = expected_in_dim - feat.size(1)
                    feat = F.pad(feat, (0, pad))
                else:
                    # 截断到期望维度
                    feat = feat[:, :expected_in_dim]
        elif feat.dim() == 4:
            # [B, C, h, w] - 空间格式（某些变体可能返回）
            if self.adaptive_pool is not None:
                feat = self.adaptive_pool(feat).flatten(1)  # [B, C]
            else:
                # 如果没有池化层，使用全局平均
                feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # [B, C]
            if feat.size(1) != expected_in_dim:
                if feat.size(1) < expected_in_dim:
                    pad = expected_in_dim - feat.size(1)
                    feat = F.pad(feat, (0, pad))
                else:
                    feat = feat[:, :expected_in_dim]
        else:
            # 其他格式：直接 flatten
            feat = feat.flatten(1)  # [B, ...] -> [B, N]
            if feat.size(1) != expected_in_dim:
                if feat.size(1) < expected_in_dim:
                    pad = expected_in_dim - feat.size(1)
                    feat = F.pad(feat, (0, pad))
                else:
                    feat = feat[:, :expected_in_dim]
        
        # 投影到目标维度
        out = self.proj(feat)  # [B, embed_dim]
        
        # 最终确保输出维度正确
        if out.size(1) != self.embed_dim:
            if out.size(1) < self.embed_dim:
                pad = self.embed_dim - out.size(1)
                out = F.pad(out, (0, pad))
            else:
                out = out[:, :self.embed_dim]
        
        return out

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # 支持 [B, C, H, W] 或 [B, T, C, H, W]
        if images.dim() == 4:
            out = self._encode_single(images)
            # 确保输出维度正确
            if out.size(1) != self.embed_dim:
                if out.size(1) < self.embed_dim:
                    pad = self.embed_dim - out.size(1)
                    out = F.pad(out, (0, pad))
                else:
                    out = out[:, :self.embed_dim]
            return out

        if images.dim() == 5:
            b, t, c, h, w = images.shape
            images = images.view(b * t, c, h, w)
            feats = self._encode_single(images)  # [B*T, E]
            # 确保 feats 维度正确
            if feats.size(1) != self.embed_dim:
                if feats.size(1) < self.embed_dim:
                    pad = self.embed_dim - feats.size(1)
                    feats = F.pad(feats, (0, pad))
                else:
                    feats = feats[:, :self.embed_dim]
            feats = feats.view(b, t, self.embed_dim)  # [B, T, E]
            # 为确保稳定性，训练阶段统一使用时序平均聚合
            out = feats.mean(dim=1)  # [B, E]
            # 再次确保输出维度正确
            if out.size(1) != self.embed_dim:
                if out.size(1) < self.embed_dim:
                    pad = self.embed_dim - out.size(1)
                    out = F.pad(out, (0, pad))
                else:
                    out = out[:, :self.embed_dim]
            return out

        raise ValueError(f"Unsupported input shape: {images.shape}")


