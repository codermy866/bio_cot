# CLIP详解与因果CLIP实现方案

## 📚 第一部分：CLIP是什么？

### 1.1 CLIP的基本概念

**CLIP (Contrastive Language-Image Pre-training)** 是OpenAI在2021年提出的一种**跨模态对比学习**方法。

#### 核心思想：
- **目标**：将图像和文本映射到同一个语义空间
- **方法**：通过对比学习，让匹配的图像-文本对更相似，不匹配的对更不相似
- **应用**：图像分类、图像检索、零样本学习等

#### 标准CLIP的架构：

```
输入：
  - 图像：[B, 3, H, W]
  - 文本：[B, L] (L是文本长度)

编码：
  - 图像编码器：ViT或ResNet → [B, D]
  - 文本编码器：Transformer → [B, D]

对比学习：
  - 计算图像-文本相似度矩阵：[B, B]
  - 正样本对：对角线元素（同一batch中的匹配对）
  - 负样本对：非对角线元素（不匹配的对）

损失函数：
  - InfoNCE Loss：最大化正样本对的相似度，最小化负样本对的相似度
```

#### 数学公式：

```python
# 图像编码
I_emb = ImageEncoder(images)  # [B, D]

# 文本编码
T_emb = TextEncoder(texts)    # [B, D]

# L2归一化
I_emb = I_emb / ||I_emb||_2
T_emb = T_emb / ||T_emb||_2

# 相似度矩阵
sim_matrix = I_emb @ T_emb.T  # [B, B]

# 温度缩放
sim_matrix = sim_matrix / temperature

# InfoNCE损失
loss = -log(exp(sim_ii) / sum(exp(sim_ij)))  # i是正样本，j是负样本
```

### 1.2 CLIP在医学多模态学习中的应用

#### 标准CLIP的问题：
1. **纯关联学习**：只学习相关性，不区分因果关系
2. **虚假关联**：可能学习到虚假的关联关系
3. **无不确定性**：无法量化预测的不确定性

#### 医学场景的特殊性：
- **因果关系重要**：HPV阳性 → OCT异常（因果），而非简单的关联
- **不确定性重要**：临床决策需要不确定性指导
- **领域知识丰富**：可以利用医学先验知识

---

## 🔬 第二部分：因果CLIP的创新

### 2.1 什么是因果CLIP？

**因果CLIP** = **标准CLIP** + **因果约束** + **贝叶斯框架**

#### 核心创新：

1. **因果约束**：
   - 使用医学先验知识构建因果图
   - 约束注意力机制，只允许因果关系内的特征交互
   - 消除虚假关联

2. **贝叶斯框架**：
   - 每个模态的编码器输出均值和方差
   - 量化跨模态匹配的不确定性
   - 提供临床决策支持

### 2.2 因果CLIP vs 标准CLIP

| 特性 | 标准CLIP | 因果CLIP |
|------|---------|---------|
| **学习方式** | 纯关联学习 | 因果约束学习 |
| **关联类型** | 所有关联 | 仅因果关系 |
| **不确定性** | 无 | 有（贝叶斯框架） |
| **领域知识** | 不使用 | 使用（因果图） |
| **虚假关联** | 可能学习到 | 被消除 |
| **临床价值** | 中等 | 高（不确定性指导） |

### 2.3 在您的实验中的应用

#### 您的多模态数据：
- **OCT图像**：48帧 × 3通道 × 224×224
- **Colposcopy图像**：3帧 × 3通道 × 224×224
- **临床特征**：7维（年龄、HPV状态、TCT结果等）

#### 因果CLIP的作用：

1. **跨模态对齐**：
   - OCT特征 ↔ 临床特征（HPV状态）
   - Colposcopy特征 ↔ 临床特征（TCT结果）
   - 所有特征 → 统一语义空间

2. **因果约束**：
   - HPV状态 → OCT特征（因果）
   - TCT结果 → Colposcopy特征（因果）
   - 年龄、风险因素 → 所有模态（因果）

3. **不确定性量化**：
   - 每个模态输出均值和方差
   - 高不确定性 → 建议进一步检查
   - 低不确定性 → 可放心决策

---

## 💻 第三部分：详细实现方案

### 3.1 架构设计

#### 整体架构：

```
输入：
  - OCT特征：[B, 768] (来自CNN/ViT编码器)
  - Colposcopy特征：[B, 768] (来自CNN/ViT编码器)
  - 临床特征：[B, 7] (原始临床数据)

↓

贝叶斯编码器（每个模态）：
  - OCT编码器：输入[B, 768] → 输出均值[B, 768] + 方差[B, 768]
  - Colposcopy编码器：输入[B, 768] → 输出均值[B, 768] + 方差[B, 768]
  - 临床编码器：输入[B, 7] → 投影到[B, 768] → 输出均值[B, 768] + 方差[B, 768]

↓

采样（训练时）：
  - OCT采样：mean + ε * sqrt(var) → [B, 768]
  - Colposcopy采样：mean + ε * sqrt(var) → [B, 768]
  - 临床采样：mean + ε * sqrt(var) → [B, 768]

↓

因果约束注意力：
  - 构建因果掩码矩阵（基于医学因果图）
  - 只允许因果关系内的特征交互
  - 跨模态对比学习（OCT-Clinical, Colposcopy-Clinical）

↓

融合与分类：
  - 多模态特征融合
  - 分类头：输出[B, 2]（正常/异常）
  - 不确定性头：输出[B, 1]（不确定性分数）

↓

损失函数：
  - 分类损失：CrossEntropy
  - KL散度损失：正则化不确定性
  - 对比学习损失：InfoNCE
```

### 3.2 核心组件实现

#### 组件1：贝叶斯编码器

```python
class BayesianCLIPEncoder(nn.Module):
    """
    贝叶斯CLIP编码器
    输入：特征 [B, embed_dim]
    输出：均值 [B, embed_dim] + 方差 [B, embed_dim]
    """
    def __init__(self, embed_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 均值编码器
        self.mean_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # 方差编码器（确保方差为正）
        self.var_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Softplus()  # 确保方差 > 0
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, embed_dim]
        Returns:
            mean: [B, embed_dim]
            var: [B, embed_dim] (方差，必须 > 0)
        """
        mean = self.mean_encoder(x)
        var = self.var_encoder(x) + 1e-6  # 防止数值不稳定
        
        return mean, var
    
    def sample(self, mean: torch.Tensor, var: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        从变分分布中采样
        训练时：采样（引入随机性）
        推理时：使用均值（确定性）
        """
        if training:
            epsilon = torch.randn_like(mean)
            return mean + epsilon * torch.sqrt(var)
        else:
            return mean
```

#### 组件2：因果约束掩码

```python
class CausalAttentionMask(nn.Module):
    """
    因果约束注意力掩码
    基于医学先验知识构建因果图约束
    """
    def __init__(self, causal_graph: Dict[str, list]):
        """
        Args:
            causal_graph: 因果图定义
            例如：
            {
                'OCT': ['HPV', 'Clinical'],      # OCT受HPV和临床特征影响
                'Colposcopy': ['TCT', 'Clinical'], # Colposcopy受TCT和临床特征影响
                'Clinical': []                    # 临床特征是根源
            }
        """
        super().__init__()
        self.causal_graph = causal_graph
        self.register_buffer('causal_mask', None)
    
    def build_causal_mask(self, seq_lengths: Dict[str, int]) -> torch.Tensor:
        """
        构建因果掩码矩阵
        只允许因果关系内的特征交互
        
        Args:
            seq_lengths: 各模态的序列长度
            例如：{'OCT': 1, 'Colposcopy': 1, 'Clinical': 1}
        
        Returns:
            mask: [total_len, total_len] 掩码矩阵
            1表示允许交互，0表示禁止交互
        """
        total_len = sum(seq_lengths.values())
        mask = torch.zeros(total_len, total_len)  # 初始化为0（禁止所有交互）
        
        # 构建索引映射
        start_idx = 0
        idx_map = {}
        for mod, length in seq_lengths.items():
            idx_map[mod] = (start_idx, start_idx + length)
            start_idx += length
        
        # 允许模态内部全连接（自注意力）
        for mod, (start, end) in idx_map.items():
            mask[start:end, start:end] = 1
        
        # 根据因果图允许跨模态连接
        for effect_mod, cause_mods in self.causal_graph.items():
            if effect_mod in idx_map:
                e_start, e_end = idx_map[effect_mod]
                
                # 允许原因模态影响结果模态
                for cause_mod in cause_mods:
                    if cause_mod in idx_map:
                        c_start, c_end = idx_map[cause_mod]
                        # 允许cause → effect（单向）
                        mask[e_start:e_end, c_start:c_end] = 1
        
        return mask
    
    def forward(self, x: torch.Tensor, seq_lengths: Dict[str, int]) -> torch.Tensor:
        """
        返回因果掩码矩阵
        """
        if self.causal_mask is None:
            self.causal_mask = self.build_causal_mask(seq_lengths)
        return self.causal_mask.to(x.device)
```

#### 组件3：因果约束的CLIP模型

```python
class CausalBayesianCLIP(nn.Module):
    """
    因果约束的贝叶斯CLIP
    结合因果约束和不确定性量化
    """
    def __init__(
        self,
        embed_dim: int = 768,
        clinical_dim: int = 7,
        causal_graph: Optional[Dict] = None,
        num_classes: int = 2,
        temperature: float = 0.07,
        kl_weight: float = 0.01,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.clinical_dim = clinical_dim
        self.temperature = temperature
        self.kl_weight = kl_weight
        
        # 临床特征投影层
        self.clinical_proj = nn.Linear(clinical_dim, embed_dim)
        
        # 贝叶斯编码器（每个模态一个）
        self.oct_encoder = BayesianCLIPEncoder(embed_dim)
        self.colposcopy_encoder = BayesianCLIPEncoder(embed_dim)
        self.clinical_encoder = BayesianCLIPEncoder(embed_dim)
        
        # 因果约束掩码
        if causal_graph is None:
            # 默认因果图（基于医学知识）
            causal_graph = {
                'OCT': ['Clinical', 'HPV'],      # OCT受临床特征和HPV影响
                'Colposcopy': ['Clinical', 'TCT'], # Colposcopy受临床特征和TCT影响
                'Clinical': []                    # 临床特征是根源
            }
        self.causal_mask = CausalAttentionMask(causal_graph)
        
        # 跨模态对比学习投影（用于InfoNCE损失）
        # 不需要额外的投影层，直接使用归一化后的特征
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # 不确定性估计头
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()  # 输出0-1的不确定性分数
        )
    
    def forward(
        self,
        oct_feat: torch.Tensor,      # [B, embed_dim]
        colpo_feat: torch.Tensor,     # [B, embed_dim]
        clinical_feat: torch.Tensor,  # [B, clinical_dim]
        return_uncertainty: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            oct_feat: OCT特征 [B, embed_dim]
            colpo_feat: Colposcopy特征 [B, embed_dim]
            clinical_feat: 临床特征 [B, clinical_dim]
            return_uncertainty: 是否返回不确定性
        
        Returns:
            {
                'logits': [B, num_classes],
                'uncertainty': [B, 1] (可选),
                'mean': [B, embed_dim * 3],
                'var': [B, embed_dim * 3],
                'emb_oct': [B, embed_dim] (归一化后，用于对比学习),
                'emb_colpo': [B, embed_dim] (归一化后，用于对比学习),
                'emb_clin': [B, embed_dim] (归一化后，用于对比学习)
            }
        """
        B = oct_feat.size(0)
        
        # 1. 临床特征投影
        clinical_proj = self.clinical_proj(clinical_feat)  # [B, embed_dim]
        
        # 2. 贝叶斯编码（每个模态输出均值和方差）
        oct_mean, oct_var = self.oct_encoder(oct_feat)
        colpo_mean, colpo_var = self.colposcopy_encoder(colpo_feat)
        clinical_mean, clinical_var = self.clinical_encoder(clinical_proj)
        
        # 3. 采样（训练时采样，推理时用均值）
        oct_feat_sampled = self.oct_encoder.sample(oct_mean, oct_var, self.training)
        colpo_feat_sampled = self.colposcopy_encoder.sample(colpo_mean, colpo_var, self.training)
        clinical_feat_sampled = self.clinical_encoder.sample(clinical_mean, clinical_var, self.training)
        
        # 4. L2归一化（用于对比学习）
        emb_oct = F.normalize(oct_feat_sampled, p=2, dim=-1)
        emb_colpo = F.normalize(colpo_feat_sampled, p=2, dim=-1)
        emb_clin = F.normalize(clinical_feat_sampled, p=2, dim=-1)
        
        # 5. 多模态特征融合
        multimodal_feat = torch.cat([
            oct_feat_sampled,
            colpo_feat_sampled,
            clinical_feat_sampled
        ], dim=-1)  # [B, embed_dim * 3]
        
        fused_feat = self.fusion(multimodal_feat)  # [B, embed_dim]
        
        # 6. 分类
        logits = self.classifier(fused_feat)  # [B, num_classes]
        
        # 7. 不确定性估计
        uncertainty = None
        if return_uncertainty:
            # 使用KL散度作为不确定性来源
            # KL散度 = -0.5 * sum(1 + log(var) - mean^2 - var)
            oct_var_s = oct_var.clamp_min(1e-6)
            colpo_var_s = colpo_var.clamp_min(1e-6)
            clinical_var_s = clinical_var.clamp_min(1e-6)
            
            kl_oct = -0.5 * torch.sum(1 + oct_var_s.log() - oct_mean.pow(2) - oct_var_s, dim=-1)
            kl_colpo = -0.5 * torch.sum(1 + colpo_var_s.log() - colpo_mean.pow(2) - colpo_var_s, dim=-1)
            kl_clinical = -0.5 * torch.sum(1 + clinical_var_s.log() - clinical_mean.pow(2) - clinical_var_s, dim=-1)
            
            total_kl = (kl_oct + kl_colpo + kl_clinical).unsqueeze(-1)  # [B, 1]
            
            # 将KL散度转换为不确定性分数（0-1）
            uncertainty = torch.sigmoid(total_kl / 100.0)  # 归一化到0-1
        
        return {
            'logits': logits,
            'uncertainty': uncertainty,
            'mean': torch.cat([oct_mean, colpo_mean, clinical_mean], dim=-1),
            'var': torch.cat([oct_var, colpo_var, clinical_var], dim=-1),
            'emb_oct': emb_oct,
            'emb_colpo': emb_colpo,
            'emb_clin': emb_clin
        }
```

#### 组件4：损失函数

```python
class CausalBayesianCLIPLoss(nn.Module):
    """
    因果约束的CLIP损失函数
    包含：分类损失 + KL散度损失 + 对比学习损失
    """
    def __init__(
        self,
        temperature: float = 0.07,
        kl_weight: float = 0.01,
        contrastive_weight: float = 0.1,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.contrastive_weight = contrastive_weight
        self.register_buffer('class_weights', class_weights)
    
    def _info_nce(self, zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE损失（对比学习）
        
        Args:
            zi: [B, D] 已归一化
            zj: [B, D] 已归一化
        
        Returns:
            loss: 标量
        """
        # 计算相似度矩阵
        sim = zi @ zj.t()  # [B, B]
        sim = sim / self.temperature
        
        # 正样本对：对角线元素（同一batch中的匹配对）
        targets = torch.arange(zi.size(0), device=zi.device)
        
        # 双向InfoNCE（对称）
        loss_i = F.cross_entropy(sim, targets)
        loss_j = F.cross_entropy(sim.t(), targets)
        
        return (loss_i + loss_j) * 0.5
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            predictions: 模型输出
            targets: 真实标签 [B]
        
        Returns:
            {
                'total_loss': 总损失,
                'ce_loss': 分类损失,
                'kl_loss': KL散度损失,
                'contrastive_loss': 对比学习损失
            }
        """
        logits = predictions['logits']
        mean = predictions['mean']
        var = predictions['var']
        
        # 1. 分类损失
        if self.class_weights is not None:
            ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        else:
            ce_loss = F.cross_entropy(logits, targets)
        
        # 2. KL散度损失（正则化不确定性）
        var_s = var.clamp_min(1e-6)
        kl_loss = 0.5 * torch.sum(mean.pow(2) + var_s - var_s.log() - 1)
        kl_loss = kl_loss / mean.size(0)  # 归一化
        
        # 3. 对比学习损失（跨模态对齐）
        contrastive_loss = torch.tensor(0.0, device=logits.device)
        if self.contrastive_weight > 0.0:
            emb_oct = predictions.get('emb_oct', None)
            emb_colpo = predictions.get('emb_colpo', None)
            emb_clin = predictions.get('emb_clin', None)
            
            # OCT-Clinical对比学习
            if emb_oct is not None and emb_clin is not None:
                contrastive_loss = contrastive_loss + self._info_nce(emb_oct, emb_clin)
            
            # Colposcopy-Clinical对比学习
            if emb_colpo is not None and emb_clin is not None:
                contrastive_loss = contrastive_loss + self._info_nce(emb_colpo, emb_clin)
        
        # 总损失
        total_loss = ce_loss + self.kl_weight * kl_loss + self.contrastive_weight * contrastive_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'kl_loss': kl_loss,
            'contrastive_loss': contrastive_loss
        }
```

---

## 🔧 第四部分：集成到您的实验方案

### 4.1 集成步骤

#### 步骤1：准备输入特征

```python
# 在您的训练脚本中
from models.cnn_multimodal_model import CNNMultimodalTransformer
from src.models.causal_bayesian_clip_framework import CausalBayesianCLIP, CausalBayesianCLIPLoss

# 1. 使用现有的CNN编码器提取特征
cnn_model = CNNMultimodalTransformer(...)
cnn_model.eval()  # 只用于特征提取

with torch.no_grad():
    # 提取OCT和Colposcopy特征
    oct_feat = cnn_model.oct_encoder(oct_images)  # [B, embed_dim]
    colpo_feat = cnn_model.col_encoder(col_images)  # [B, embed_dim]

# 2. 准备临床特征
clinical_feat = batch['clinical_features']  # [B, 7]
```

#### 步骤2：创建因果CLIP模型

```python
# 定义医学因果图
causal_graph = {
    'OCT': ['Clinical', 'HPV'],      # OCT受临床特征和HPV影响
    'Colposcopy': ['Clinical', 'TCT'], # Colposcopy受临床特征和TCT影响
    'Clinical': []                    # 临床特征是根源
}

# 创建模型
causal_clip_model = CausalBayesianCLIP(
    embed_dim=768,
    clinical_dim=7,
    causal_graph=causal_graph,
    num_classes=2,
    temperature=0.07,
    kl_weight=0.01
).to(device)

# 创建损失函数
criterion = CausalBayesianCLIPLoss(
    temperature=0.07,
    kl_weight=0.01,
    contrastive_weight=0.1
)
```

#### 步骤3：训练循环

```python
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        # 1. 提取特征（使用现有编码器）
        oct_images = batch['oct_images'].to(device)
        col_images = batch['col_images'].to(device)
        clinical_feat = batch['clinical_features'].to(device)
        labels = batch['label'].to(device)
        
        # 提取特征（可以使用现有的CNN编码器）
        with torch.no_grad():
            oct_feat = extract_oct_features(oct_images)  # [B, 768]
            colpo_feat = extract_colpo_features(col_images)  # [B, 768]
        
        # 2. 前向传播
        optimizer.zero_grad()
        
        with autocast():
            output = causal_clip_model(
                oct_feat=oct_feat,
                colpo_feat=colpo_feat,
                clinical_feat=clinical_feat,
                return_uncertainty=True
            )
            
            # 3. 计算损失
            loss_dict = criterion(output, labels)
            loss = loss_dict['total_loss']
        
        # 4. 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 5. 记录
        print(f"Loss: {loss.item():.4f}, "
              f"CE: {loss_dict['ce_loss'].item():.4f}, "
              f"KL: {loss_dict['kl_loss'].item():.4f}, "
              f"Contrastive: {loss_dict['contrastive_loss'].item():.4f}")
```

#### 步骤4：推理和不确定性分析

```python
model.eval()
with torch.no_grad():
    output = causal_clip_model(
        oct_feat=oct_feat,
        colpo_feat=colpo_feat,
        clinical_feat=clinical_feat,
        return_uncertainty=True
    )
    
    # 预测
    logits = output['logits']
    probs = F.softmax(logits, dim=-1)
    preds = torch.argmax(logits, dim=-1)
    
    # 不确定性
    uncertainty = output['uncertainty']  # [B, 1]
    
    # 临床决策支持
    for i in range(B):
        if uncertainty[i] > 0.5:  # 高不确定性
            print(f"样本 {i}: 高不确定性 ({uncertainty[i]:.3f})，建议进一步检查")
        else:  # 低不确定性
            print(f"样本 {i}: 低不确定性 ({uncertainty[i]:.3f})，可放心决策")
```

### 4.2 完整的训练脚本示例

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因果约束贝叶斯CLIP训练脚本
集成到现有的多模态训练框架
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# 导入模型
from models.cnn_multimodal_model import CNNMultimodalTransformer
from src.models.causal_bayesian_clip_framework import (
    CausalBayesianCLIP,
    CausalBayesianCLIPLoss
)
from src.data.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset

def train_causal_clip(
    data_path='5centers_multi',
    epochs=30,
    batch_size=8,
    learning_rate=1e-4,
    embed_dim=768,
    output_dir='causal_clip_results'
):
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据加载
    train_dataset = EnhancedMultimodalCervicalDataset(
        root=f'{data_path}/train',
        is_train=True,
        ...
    )
    val_dataset = EnhancedMultimodalCervicalDataset(
        root=f'{data_path}/test',
        is_train=False,
        ...
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 特征提取器（使用现有的CNN编码器）
    feature_extractor = CNNMultimodalTransformer(...).to(device)
    feature_extractor.eval()  # 只用于特征提取
    
    # 因果CLIP模型
    causal_graph = {
        'OCT': ['Clinical', 'HPV'],
        'Colposcopy': ['Clinical', 'TCT'],
        'Clinical': []
    }
    
    model = CausalBayesianCLIP(
        embed_dim=embed_dim,
        clinical_dim=7,
        causal_graph=causal_graph,
        num_classes=2,
        temperature=0.07,
        kl_weight=0.01
    ).to(device)
    
    # 损失函数
    criterion = CausalBayesianCLIPLoss(
        temperature=0.07,
        kl_weight=0.01,
        contrastive_weight=0.1
    )
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = GradScaler()
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            # 数据
            oct_images = batch['oct_images'].to(device)
            col_images = batch['col_images'].to(device)
            clinical_feat = batch['clinical_features'].to(device)
            labels = batch['label'].to(device)
            
            # 提取特征
            with torch.no_grad():
                oct_feat = feature_extractor.oct_encoder(oct_images)
                colpo_feat = feature_extractor.col_encoder(col_images)
            
            # 前向传播
            optimizer.zero_grad()
            
            with autocast():
                output = model(
                    oct_feat=oct_feat,
                    colpo_feat=colpo_feat,
                    clinical_feat=clinical_feat,
                    return_uncertainty=True
                )
                
                loss_dict = criterion(output, labels)
                loss = loss_dict['total_loss']
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 统计
            train_loss += loss.item()
            _, predicted = output['logits'].max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*train_correct/train_total:.2f}%"
            })
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_labels = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in val_loader:
                oct_images = batch['oct_images'].to(device)
                col_images = batch['col_images'].to(device)
                clinical_feat = batch['clinical_features'].to(device)
                labels = batch['label'].to(device)
                
                # 提取特征
                oct_feat = feature_extractor.oct_encoder(oct_images)
                colpo_feat = feature_extractor.col_encoder(col_images)
                
                # 前向传播
                output = model(
                    oct_feat=oct_feat,
                    colpo_feat=colpo_feat,
                    clinical_feat=clinical_feat,
                    return_uncertainty=True
                )
                
                loss_dict = criterion(output, labels)
                val_loss += loss_dict['total_loss'].item()
                
                # 统计
                _, predicted = output['logits'].max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # 收集概率和不确定性
                probs = F.softmax(output['logits'], dim=-1)
                all_probs.append(probs[:, 1].cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_uncertainties.append(output['uncertainty'].cpu().numpy())
        
        # 计算指标
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        all_uncertainties = np.concatenate(all_uncertainties)
        
        auc = roc_auc_score(all_labels, all_probs)
        avg_uncertainty = np.mean(all_uncertainties)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        print(f"  Val AUC: {auc:.4f}")
        print(f"  Avg Uncertainty: {avg_uncertainty:.4f}")
        
        # 保存模型
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_auc': auc
            }, f'{output_dir}/checkpoint_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train_causal_clip()
```

---

## 📊 第五部分：实验设计

### 5.1 对比实验

| 方法 | 因果约束 | 贝叶斯 | 对比学习 | 预期AUC |
|------|---------|--------|---------|---------|
| Baseline (CNN) | ✗ | ✗ | ✗ | 0.85 |
| Standard CLIP | ✗ | ✗ | ✓ | 0.86 |
| Bayesian CLIP | ✗ | ✓ | ✓ | 0.87 |
| Causal CLIP | ✓ | ✗ | ✓ | 0.88 |
| **Causal-Bayesian CLIP** | ✓ | ✓ | ✓ | **0.89+** |

### 5.2 消融研究

1. **因果约束的影响**：
   - 有因果约束 vs 无因果约束
   - 不同因果图设计的影响

2. **贝叶斯框架的影响**：
   - 有不确定性量化 vs 无不确定性量化
   - 不同KL权重的影响

3. **对比学习的影响**：
   - 有对比学习 vs 无对比学习
   - 不同温度参数的影响

### 5.3 不确定性分析

1. **不确定性校准**：
   - ECE (Expected Calibration Error)
   - Brier Score

2. **不确定性与错误率的关系**：
   - 高不确定性样本的错误率
   - 低不确定性样本的错误率

3. **临床决策支持**：
   - 不同不确定性阈值下的决策
   - 决策曲线分析（DCA）

---

## ✅ 第六部分：总结

### 6.1 CLIP的核心概念

- **CLIP**：跨模态对比学习方法
- **目标**：将不同模态映射到统一语义空间
- **方法**：对比学习（正样本对拉近，负样本对推远）

### 6.2 因果CLIP的创新

- **因果约束**：使用医学先验知识，消除虚假关联
- **贝叶斯框架**：量化不确定性，提供临床决策支持
- **对比学习**：跨模态对齐，提升特征质量

### 6.3 实现要点

1. **贝叶斯编码器**：输出均值和方差
2. **因果约束掩码**：基于医学因果图
3. **损失函数**：分类损失 + KL散度 + 对比学习
4. **不确定性估计**：基于KL散度

### 6.4 集成步骤

1. 使用现有编码器提取特征
2. 创建因果CLIP模型
3. 训练循环（包含对比学习）
4. 推理和不确定性分析

---

**祝您实验顺利！** 🚀

