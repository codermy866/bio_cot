# Bio-COT模型架构分析

## 当前架构

### 1. 图像编码器（DualHeadImageEncoder）

**当前实现：不是ResNet，而是简单的MLP投影层**

```python
class DualHeadImageEncoder(nn.Module):
    """
    双头图像编码器
    - 输入：512维预提取特征（不是原始图像）
    - 输出：768维嵌入特征（z_causal, z_noise）
    """
    def __init__(self, input_dim=512, embed_dim=768):
        # 特征投影层（MLP）
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # 双头网络（MLP）
        self.causal_head = nn.Sequential(...)
        self.noise_head = nn.Sequential(...)
```

**关键点：**
- ❌ **不是ResNet**：只是一个简单的MLP投影层
- ✅ **输入是预提取特征**：512维特征向量（来自OCT和Colposcopy图像处理）
- ✅ **轻量级**：参数量小，训练快速

### 2. 数据流程

```
原始图像 → 特征提取（512维）→ DualHeadImageEncoder → 768维嵌入
```

## 如果改成大模型

### 方案1：Vision Transformer (ViT)

**优点：**
- 更强的特征提取能力
- 可以处理原始图像
- 支持多尺度特征

**修改点：**
1. 修改`DualHeadImageEncoder`，添加ViT backbone
2. 修改数据加载，直接使用原始图像
3. 添加图像预处理（resize, normalize等）

### 方案2：CLIP视觉编码器

**优点：**
- 预训练的强大视觉特征
- 多模态对齐能力
- 可以直接使用原始图像

**修改点：**
1. 使用CLIP的视觉编码器作为backbone
2. 修改数据加载，使用原始图像
3. 可能需要fine-tuning

### 方案3：ResNet/EfficientNet

**优点：**
- 经典架构，稳定可靠
- 预训练权重丰富
- 计算效率高

**修改点：**
1. 添加ResNet/EfficientNet作为backbone
2. 修改数据加载，使用原始图像
3. 在backbone后添加双头网络

## 推荐方案

### 短期（快速改进）
- 保持当前架构，但增强MLP层数
- 添加注意力机制
- 使用更好的特征融合策略

### 长期（性能提升）
- 使用CLIP视觉编码器（推荐）
- 或使用Vision Transformer
- 直接处理原始图像，端到端训练

## 代码修改示例

### 使用CLIP作为backbone

```python
import clip

class DualHeadImageEncoderWithCLIP(nn.Module):
    def __init__(self, embed_dim=768, clip_model='ViT-B/32'):
        super().__init__()
        # CLIP视觉编码器
        self.clip_model, _ = clip.load(clip_model, device='cuda')
        self.clip_visual = self.clip_model.visual
        
        # 投影到embed_dim
        clip_dim = self.clip_visual.output_dim  # 通常是512
        self.proj = nn.Linear(clip_dim, embed_dim)
        
        # 双头网络
        self.causal_head = nn.Sequential(...)
        self.noise_head = nn.Sequential(...)
    
    def forward(self, images):
        # images: [B, C, H, W]
        with torch.no_grad():
            clip_features = self.clip_visual(images)
        proj_feat = self.proj(clip_features)
        z_causal = self.causal_head(proj_feat)
        z_noise = self.noise_head(proj_feat)
        return z_causal, z_noise
```

### 使用Vision Transformer

```python
from transformers import ViTModel

class DualHeadImageEncoderWithViT(nn.Module):
    def __init__(self, embed_dim=768, vit_model='google/vit-base-patch16-224'):
        super().__init__()
        # ViT backbone
        self.vit = ViTModel.from_pretrained(vit_model)
        vit_dim = self.vit.config.hidden_size  # 768
        
        # 投影层（如果需要改变维度）
        if vit_dim != embed_dim:
            self.proj = nn.Linear(vit_dim, embed_dim)
        else:
            self.proj = nn.Identity()
        
        # 双头网络
        self.causal_head = nn.Sequential(...)
        self.noise_head = nn.Sequential(...)
    
    def forward(self, images):
        # images: [B, C, H, W]
        outputs = self.vit(pixel_values=images)
        vit_features = outputs.last_hidden_state[:, 0]  # [CLS] token
        proj_feat = self.proj(vit_features)
        z_causal = self.causal_head(proj_feat)
        z_noise = self.noise_head(proj_feat)
        return z_causal, z_noise
```

## 注意事项

1. **数据加载修改**：需要修改`EnhancedMultimodalCervicalDataset`，直接返回原始图像而不是512维特征
2. **内存消耗**：大模型会显著增加显存占用
3. **训练时间**：端到端训练会变慢
4. **预训练权重**：建议使用预训练权重，从零训练成本高

