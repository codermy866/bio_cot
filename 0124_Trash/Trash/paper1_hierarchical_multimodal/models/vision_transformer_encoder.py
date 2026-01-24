"""
基于视觉大模型的特征编码器
使用ViT或Swin Transformer作为backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available, falling back to ResNet")


class VisionTransformerLocalEncoder(nn.Module):
    """
    使用ViT作为backbone的局部特征编码器
    提取细粒度的局部特征
    """
    def __init__(self, embed_dim=768, model_name='vit_base_patch16_224', 
                 use_pretrained=True, input_size=224):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for Vision Transformer. Install with: pip install timm")
        
        # 使用ViT作为backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=use_pretrained,
            num_classes=0,  # 去掉分类头
            global_pool='',  # 保留patch tokens
            img_size=input_size
        )
        
        # 获取backbone的输出维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_size, input_size)
            dummy_output = self.backbone(dummy_input)
            if isinstance(dummy_output, tuple):
                backbone_out_dim = dummy_output[0].size(-1)
            else:
                backbone_out_dim = dummy_output.size(-1)
        
        # 局部特征提取：使用patch tokens的前半部分（更细粒度）
        self.local_projection = nn.Sequential(
            nn.Linear(backbone_out_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self.embed_dim = embed_dim
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 或 [B, K, C, H, W] (K是帧数)
        Returns:
            local_feat: [B, embed_dim]
        """
        if x.dim() == 5:
            B, K, C, H, W = x.shape
            x = x.view(B * K, C, H, W)
            # 通过ViT backbone
            patch_tokens = self.backbone(x)  # [B*K, num_patches, dim]
            if isinstance(patch_tokens, tuple):
                patch_tokens = patch_tokens[0]
            
            # 使用前一半的patch tokens作为局部特征（更细粒度）
            num_patches = patch_tokens.size(1)
            local_patches = patch_tokens[:, :num_patches // 2, :]  # [B*K, num_patches//2, dim]
            
            # 平均池化局部patches
            local_feat = local_patches.mean(dim=1)  # [B*K, dim]
            local_feat = self.local_projection(local_feat)  # [B*K, embed_dim]
            
            local_feat = local_feat.view(B, K, -1)
            # 平均池化多帧
            local_feat = local_feat.mean(dim=1)  # [B, embed_dim]
        else:
            patch_tokens = self.backbone(x)
            if isinstance(patch_tokens, tuple):
                patch_tokens = patch_tokens[0]
            
            num_patches = patch_tokens.size(1)
            local_patches = patch_tokens[:, :num_patches // 2, :]
            local_feat = local_patches.mean(dim=1)
            local_feat = self.local_projection(local_feat)
        
        return local_feat


class VisionTransformerGlobalEncoder(nn.Module):
    """
    使用ViT作为backbone的全局特征编码器
    提取全局的语义特征
    """
    def __init__(self, embed_dim=768, model_name='vit_base_patch16_224',
                 use_pretrained=True, input_size=224):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for Vision Transformer. Install with: pip install timm")
        
        # 使用ViT作为backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=use_pretrained,
            num_classes=0,
            global_pool='',  # 保留patch tokens
            img_size=input_size
        )
        
        # 获取backbone的输出维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_size, input_size)
            dummy_output = self.backbone(dummy_input)
            if isinstance(dummy_output, tuple):
                backbone_out_dim = dummy_output[0].size(-1)
            else:
                backbone_out_dim = dummy_output.size(-1)
        
        # 全局特征提取：使用所有patch tokens或CLS token
        self.global_projection = nn.Sequential(
            nn.Linear(backbone_out_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self.embed_dim = embed_dim
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 或 [B, K, C, H, W]
        Returns:
            global_feat: [B, embed_dim]
        """
        if x.dim() == 5:
            B, K, C, H, W = x.shape
            x = x.view(B * K, C, H, W)
            # 通过ViT backbone
            patch_tokens = self.backbone(x)  # [B*K, num_patches, dim]
            if isinstance(patch_tokens, tuple):
                patch_tokens = patch_tokens[0]
            
            # 使用所有patch tokens的平均作为全局特征
            global_feat = patch_tokens.mean(dim=1)  # [B*K, dim]
            global_feat = self.global_projection(global_feat)  # [B*K, embed_dim]
            
            global_feat = global_feat.view(B, K, -1)
            # 平均池化多帧
            global_feat = global_feat.mean(dim=1)  # [B, embed_dim]
        else:
            patch_tokens = self.backbone(x)
            if isinstance(patch_tokens, tuple):
                patch_tokens = patch_tokens[0]
            
            global_feat = patch_tokens.mean(dim=1)
            global_feat = self.global_projection(global_feat)
        
        return global_feat


class SwinTransformerLocalEncoder(nn.Module):
    """
    使用Swin Transformer作为backbone的局部特征编码器
    """
    def __init__(self, embed_dim=768, model_name='swin_base_patch4_window7_224',
                 use_pretrained=True, input_size=224):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for Swin Transformer. Install with: pip install timm")
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=use_pretrained,
            num_classes=0,
            global_pool='',
            img_size=input_size
        )
        
        # 获取backbone的输出维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_size, input_size)
            dummy_output = self.backbone(dummy_input)
            if isinstance(dummy_output, tuple):
                backbone_out_dim = dummy_output[0].size(-1)
            else:
                backbone_out_dim = dummy_output.size(-1)
        
        self.local_projection = nn.Sequential(
            nn.Linear(backbone_out_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self.embed_dim = embed_dim
    
    def forward(self, x):
        if x.dim() == 5:
            B, K, C, H, W = x.shape
            x = x.view(B * K, C, H, W)
            features = self.backbone(x)
            if isinstance(features, tuple):
                features = features[0]
            # Swin的输出是 [B*K, H*W, dim]，取前一半作为局部特征
            num_tokens = features.size(1)
            local_tokens = features[:, :num_tokens // 2, :]
            local_feat = local_tokens.mean(dim=1)
            local_feat = self.local_projection(local_feat)
            local_feat = local_feat.view(B, K, -1).mean(dim=1)
        else:
            features = self.backbone(x)
            if isinstance(features, tuple):
                features = features[0]
            num_tokens = features.size(1)
            local_tokens = features[:, :num_tokens // 2, :]
            local_feat = local_tokens.mean(dim=1)
            local_feat = self.local_projection(local_feat)
        return local_feat


class SwinTransformerGlobalEncoder(nn.Module):
    """
    使用Swin Transformer作为backbone的全局特征编码器
    """
    def __init__(self, embed_dim=768, model_name='swin_base_patch4_window7_224',
                 use_pretrained=True, input_size=224):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for Swin Transformer. Install with: pip install timm")
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=use_pretrained,
            num_classes=0,
            global_pool='',
            img_size=input_size
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_size, input_size)
            dummy_output = self.backbone(dummy_input)
            if isinstance(dummy_output, tuple):
                backbone_out_dim = dummy_output[0].size(-1)
            else:
                backbone_out_dim = dummy_output.size(-1)
        
        self.global_projection = nn.Sequential(
            nn.Linear(backbone_out_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self.embed_dim = embed_dim
    
    def forward(self, x):
        if x.dim() == 5:
            B, K, C, H, W = x.shape
            x = x.view(B * K, C, H, W)
            features = self.backbone(x)
            if isinstance(features, tuple):
                features = features[0]
            global_feat = features.mean(dim=1)
            global_feat = self.global_projection(global_feat)
            global_feat = global_feat.view(B, K, -1).mean(dim=1)
        else:
            features = self.backbone(x)
            if isinstance(features, tuple):
                features = features[0]
            global_feat = features.mean(dim=1)
            global_feat = self.global_projection(global_feat)
        return global_feat

