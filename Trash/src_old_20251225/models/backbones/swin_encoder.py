import torch
import torch.nn as nn

from models.SwinT.swin_image_encoder import SwinTImageEncoder
from models.cnn_multimodal_model import ClinicalEncoder, CrossModalFusion


class SwinTMultimodalTransformer(nn.Module):
    """
    多模态（OCT + Colposcopy + Clinical）分类模型，图像编码器使用 Swin-T。
    - OCT/Col 分支共享统一结构但各自独立实例
    - 帧间通过注意力聚合
    - 与临床特征融合后分类
    """

    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        clinical_dim: int = 7,
        oct_num_frames: int = 120,
        col_num_frames: int = 3,
        swin_name: str = 'swin_tiny_patch4_window7_224',
        pretrained: bool = True,
        input_size: int = 224,
        use_frame_attention: bool = False,
    ):
        super().__init__()

        self.oct_encoder = SwinTImageEncoder(
            embed_dim=embed_dim,
            dropout=dropout,
            num_frames=oct_num_frames,
            model_name=swin_name,
            pretrained=pretrained,
            input_size=input_size,
            use_frame_attention=use_frame_attention,
        )

        self.col_encoder = SwinTImageEncoder(
            embed_dim=embed_dim,
            dropout=dropout,
            num_frames=col_num_frames,
            model_name=swin_name,
            pretrained=pretrained,
            input_size=input_size,
            use_frame_attention=use_frame_attention,
        )

        self.clinical_encoder = ClinicalEncoder(
            input_dim=clinical_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        self.cross_modal_fusion = CrossModalFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, oct_images, col_images, clinical_features):
        oct_feat = self.oct_encoder(oct_images)
        col_feat = self.col_encoder(col_images)
        clinical_feat = self.clinical_encoder(clinical_features)
        
        # 确保所有特征的维度都是 [B, embed_dim]
        embed_dim = self.cross_modal_fusion.embed_dim
        
        # 修复 oct_feat 维度 - 确保是 [B, embed_dim]
        if oct_feat.dim() != 2:
            oct_feat = oct_feat.flatten(1)
        if oct_feat.size(1) != embed_dim:
            if oct_feat.size(1) < embed_dim:
                # 填充零
                pad = embed_dim - oct_feat.size(1)
                oct_feat = torch.nn.functional.pad(oct_feat, (0, pad), mode='constant', value=0)
            else:
                # 截断
                oct_feat = oct_feat[:, :embed_dim]
        
        # 修复 col_feat 维度 - 确保是 [B, embed_dim]
        if col_feat.dim() != 2:
            col_feat = col_feat.flatten(1)
        if col_feat.size(1) != embed_dim:
            if col_feat.size(1) < embed_dim:
                pad = embed_dim - col_feat.size(1)
                col_feat = torch.nn.functional.pad(col_feat, (0, pad), mode='constant', value=0)
            else:
                col_feat = col_feat[:, :embed_dim]
        
        # 修复 clinical_feat 维度 - 确保是 [B, embed_dim]
        if clinical_feat.dim() != 2:
            clinical_feat = clinical_feat.flatten(1)
        if clinical_feat.size(1) != embed_dim:
            if clinical_feat.size(1) < embed_dim:
                pad = embed_dim - clinical_feat.size(1)
                clinical_feat = torch.nn.functional.pad(clinical_feat, (0, pad), mode='constant', value=0)
            else:
                clinical_feat = clinical_feat[:, :embed_dim]
        
        # 验证维度
        assert oct_feat.size(1) == embed_dim, f"oct_feat dimension mismatch: {oct_feat.shape} != [B, {embed_dim}]"
        assert col_feat.size(1) == embed_dim, f"col_feat dimension mismatch: {col_feat.shape} != [B, {embed_dim}]"
        assert clinical_feat.size(1) == embed_dim, f"clinical_feat dimension mismatch: {clinical_feat.shape} != [B, {embed_dim}]"
        
        fused_feat = self.cross_modal_fusion(oct_feat, col_feat, clinical_feat)
        logits = self.classifier(fused_feat)
        return logits


class SwinTOctClassifier(nn.Module):
    """
    OCT 单模态分类模型（Swin-T 图像编码 + 帧间注意力聚合 + 分类头）
    输入: oct_images [B, T, C, H, W] 或 [B, C, H, W]
    输出: logits [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 768,
        dropout: float = 0.1,
        oct_num_frames: int = 120,
        swin_name: str = 'swin_tiny_patch4_window7_224',
        pretrained: bool = True,
    ):
        super().__init__()

        self.oct_encoder = SwinTImageEncoder(
            embed_dim=embed_dim,
            dropout=dropout,
            num_frames=oct_num_frames,
            model_name=swin_name,
            pretrained=pretrained,
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, oct_images):
        feat = self.oct_encoder(oct_images)
        logits = self.classifier(feat)
        return logits


