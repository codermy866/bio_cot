"""
使用视觉大模型（ViT/Swin）作为backbone的层次化多粒度多模态融合模型
"""

import torch
import torch.nn as nn

from .vision_transformer_encoder import (
    VisionTransformerLocalEncoder,
    VisionTransformerGlobalEncoder,
    SwinTransformerLocalEncoder,
    SwinTransformerGlobalEncoder
)
from .multi_granularity_encoder import (
    SemanticFeatureEncoder,
    StatisticalFeatureEncoder
)
from .cross_modal_aligner import CrossModalAligner
from .multi_granularity_fusion import (
    FineGrainFusion,
    MidGrainFusion,
    CoarseGrainFusion
)
from .adaptive_weighting import AdaptiveModalityWeighting
from .contrastive_alignment import ContrastiveAlignmentModule
from .hierarchical_classifier import HierarchicalClassifier


class HierarchicalMultimodalModelViT(nn.Module):
    """
    使用视觉大模型（ViT/Swin）作为backbone的层次化多粒度多模态融合模型
    """
    
    def __init__(
        self,
        embed_dim=768,
        num_classes=2,
        clinical_dim=7,
        contrastive_weight=0.1,
        backbone_type='vit',  # 'vit' or 'swin'
        model_name='vit_base_patch16_224',  # ViT model name
        use_pretrained=True,
        input_size=224
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.contrastive_weight = contrastive_weight
        
        # 根据backbone类型选择编码器
        if backbone_type.lower() == 'vit':
            # 使用ViT作为backbone
            self.oct_local_encoder = VisionTransformerLocalEncoder(
                embed_dim=embed_dim,
                model_name=model_name,
                use_pretrained=use_pretrained,
                input_size=input_size
            )
            self.oct_global_encoder = VisionTransformerGlobalEncoder(
                embed_dim=embed_dim,
                model_name=model_name,
                use_pretrained=use_pretrained,
                input_size=input_size
            )
            self.col_local_encoder = VisionTransformerLocalEncoder(
                embed_dim=embed_dim,
                model_name=model_name,
                use_pretrained=use_pretrained,
                input_size=input_size
            )
            self.col_global_encoder = VisionTransformerGlobalEncoder(
                embed_dim=embed_dim,
                model_name=model_name,
                use_pretrained=use_pretrained,
                input_size=input_size
            )
        elif backbone_type.lower() == 'swin':
            # 使用Swin Transformer作为backbone
            swin_model_name = model_name if 'swin' in model_name.lower() else 'swin_base_patch4_window7_224'
            self.oct_local_encoder = SwinTransformerLocalEncoder(
                embed_dim=embed_dim,
                model_name=swin_model_name,
                use_pretrained=use_pretrained,
                input_size=input_size
            )
            self.oct_global_encoder = SwinTransformerGlobalEncoder(
                embed_dim=embed_dim,
                model_name=swin_model_name,
                use_pretrained=use_pretrained,
                input_size=input_size
            )
            self.col_local_encoder = SwinTransformerLocalEncoder(
                embed_dim=embed_dim,
                model_name=swin_model_name,
                use_pretrained=use_pretrained,
                input_size=input_size
            )
            self.col_global_encoder = SwinTransformerGlobalEncoder(
                embed_dim=embed_dim,
                model_name=swin_model_name,
                use_pretrained=use_pretrained,
                input_size=input_size
            )
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}. Use 'vit' or 'swin'")
        
        # 临床特征编码器（保持不变）
        self.clinical_semantic_encoder = SemanticFeatureEncoder(clinical_dim, embed_dim)
        self.clinical_stat_encoder = StatisticalFeatureEncoder(clinical_dim, embed_dim)
        
        # 跨模态对齐
        self.local_aligner = CrossModalAligner(embed_dim)
        self.global_aligner = CrossModalAligner(embed_dim)
        
        # 多粒度融合
        self.fine_fusion = FineGrainFusion(embed_dim)
        self.mid_fusion = MidGrainFusion(embed_dim)
        self.coarse_fusion = CoarseGrainFusion(embed_dim)
        
        # 自适应权重
        self.adaptive_weighting = AdaptiveModalityWeighting(embed_dim)
        
        # 对比学习
        self.contrastive_module = ContrastiveAlignmentModule(embed_dim)
        
        # 分类器
        self.classifier = HierarchicalClassifier(embed_dim, num_classes)
    
    def encode_modalities(self, oct_images, col_images, clinical_features):
        """
        将原始模态输入编码为多粒度特征
        Args:
            oct_images: [B, T, 3, H, W]
            col_images: [B, K, 3, H, W]
            clinical_features: [B, clinical_dim]
        """
        oct_local = self.oct_local_encoder(oct_images)
        oct_global = self.oct_global_encoder(oct_images)
        col_local = self.col_local_encoder(col_images)
        col_global = self.col_global_encoder(col_images)
        clinical_semantic = self.clinical_semantic_encoder(clinical_features)
        clinical_stat = self.clinical_stat_encoder(clinical_features)
        return oct_local, oct_global, col_local, col_global, clinical_semantic, clinical_stat
    
    def forward(self, oct_images, col_images, clinical_features, labels=None):
        # 编码多粒度特征
        (
            oct_local,
            oct_global,
            col_local,
            col_global,
            clinical_semantic,
            clinical_stat
        ) = self.encode_modalities(oct_images, col_images, clinical_features)
        
        # 跨模态对齐
        local_aligned = self.local_aligner(oct_local, col_local)
        global_aligned = self.global_aligner(oct_global, col_global)
        
        # 对比学习增强对齐
        aligned_features, contrastive_loss = self.contrastive_module(
            oct_global, col_global, clinical_semantic, labels
        )
        
        # 多粒度融合
        fine_feat = self.fine_fusion(local_aligned, local_aligned)
        mid_feat = self.mid_fusion(global_aligned, global_aligned)
        coarse_feat = self.coarse_fusion(
            clinical_semantic, clinical_stat, aligned_features.mean(dim=1)
        )
        
        # 自适应权重
        adaptive_feat, weights = self.adaptive_weighting(fine_feat, mid_feat, coarse_feat)
        
        # 分类
        logits = self.classifier(fine_feat, mid_feat, coarse_feat)
        
        outputs = {
            'logits': logits,
            'adaptive_feat': adaptive_feat,
            'weights': weights,
            'contrastive_loss': self.contrastive_weight * contrastive_loss if contrastive_loss is not None else 0.0
        }
        return outputs

