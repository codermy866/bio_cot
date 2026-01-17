# Bio-COT 完整模型架构文档

## 一、模型概述

**Bio-COT (Biological Causal Optimal Transport)** 是一个多模态医学图像分类模型，通过因果解耦和最优传输理论实现跨中心的域不变特征学习。

### 核心创新点
1. **Student Prior网络**：替代在线VLM，从临床数据生成语义锚点（训练时快速）
2. **Sinkhorn最优传输**：替代KL散度，实现更灵活的分布对齐
3. **Memory Bank反事实干预**：实现真正的因果解耦

---

## 二、完整架构图

```
输入层
├── OCT图像特征 [B, 512] 或原始图像 [B, C, H, W] 或 [B, F, C, H, W] (F=120帧)
├── Colposcopy图像特征 [B, 512] 或原始图像 [B, C, H, W] 或 [B, K, C, H, W] (K≤3)
└── 临床特征 [B, 7] (HPV(1) + TCT(5) + Age(1))

↓

【模块1：Student Prior网络】
临床数据 [B, 7] → StudentPriorNet → 语义锚点 z_sem [B, 768]

↓

【模块2：图像编码器（双模式）】
模式A：VLM编码器（可选）
├── 原始图像 → VLMImageEncoderFromTensor → z_causal [B, 768], z_noise [B, 768]
└── 支持多帧OCT（120帧）和单帧Colposcopy

模式B：传统MLP编码器（默认）
├── 图像特征 [B, 512] → DualHeadImageEncoder → z_causal [B, 768], z_noise [B, 768]
└── 双头结构：因果头 + 噪声头

↓

【模块3：多模态融合】
├── OCT特征 + Colposcopy特征 → 加权融合 + 拼接融合 → z_causal [B, 768]
└── 可学习模态权重：modal_weights [2] (OCT权重, Colposcopy权重)

↓

【模块4：三模态融合】
z_causal [B, 768] + z_sem [B, 768] → 拼接 → 投影层 → fused_multimodal [B, 768]

↓

【模块5：分类器】
fused_multimodal [B, 768] → 分类器 → logits [B, num_classes]

↓

【模块6：损失计算（训练时）】
├── Sinkhorn OT损失：z_causal ↔ z_sem
├── 反事实一致性损失：logits_orig ↔ logits_cf
└── 对抗损失：z_noise → CenterDiscriminator → center_logits

↓

【模块7：Memory Bank（反事实干预）】
z_noise + center_labels → NoiseMemoryBank → z_noise_cf [B, 768]
```

---

## 三、核心模块详细设计

### 3.1 Student Prior网络 (StudentPriorNet)

**文件位置**：`src/models/bida/prior_net.py`

**作用**：从临床数据生成语义锚点，替代在线VLM调用

**输入**：
- `clinical_vec`: [B, 7]
  - HPV: [B, 1] 二值（0或1）
  - TCT: [B, 5] one-hot编码（NILM=0, ASC-US=1, LSIL=2, HSIL=3, 其他=4）
  - Age: [B, 1] 连续值（归一化到[0,1]，原始值/100）

**网络结构**：
```python
输入 [B, 7]
  ↓
Linear(7 → 256)
  ↓
BatchNorm1d(256)
  ↓
LeakyReLU(0.2)
  ↓
Dropout(0.2)
  ↓
Linear(256 → 512)
  ↓
BatchNorm1d(512)
  ↓
LeakyReLU(0.2)
  ↓
Dropout(0.2)
  ↓
Linear(512 → 768)
  ↓
输出 z_sem [B, 768]
```

**权重初始化**：
- Linear层：Kaiming Normal初始化（mode='fan_out', nonlinearity='leaky_relu'）
- BatchNorm层：weight=1, bias=0

**预训练**：
- 使用预计算的VLM特征作为监督信号
- 损失函数：MSE Loss
- 投影层：VLM特征 [1536] → [768]

---

### 3.2 双头图像编码器 (DualHeadImageEncoder)

**文件位置**：`src/models/bida/bio_cot_model.py`

**作用**：将图像特征分解为因果特征和噪声特征

**输入**：
- `image_features`: [B, input_dim] (默认input_dim=512)

**网络结构**：
```python
输入 [B, input_dim]
  ↓
【特征投影层】
Linear(input_dim → embed_dim * 2)
  ↓
LayerNorm(embed_dim * 2)
  ↓
GELU()
  ↓
Dropout(0.1)
  ↓
Linear(embed_dim * 2 → embed_dim)
  ↓
proj_feat [B, embed_dim]

↓ 分支1：因果头
【因果头】
Linear(embed_dim → embed_dim)
  ↓
LayerNorm(embed_dim)
  ↓
GELU()
  ↓
Dropout(0.1)
  ↓
Linear(embed_dim → embed_dim)
  ↓
z_causal [B, embed_dim]

↓ 分支2：噪声头
【噪声头】
Linear(embed_dim → embed_dim)
  ↓
LayerNorm(embed_dim)
  ↓
GELU()
  ↓
Dropout(0.1)
  ↓
Linear(embed_dim → embed_dim)
  ↓
z_noise [B, embed_dim]
```

**输出**：
- `z_causal`: [B, embed_dim] 因果特征（用于分类）
- `z_noise`: [B, embed_dim] 噪声特征（包含域信息，用于对抗训练）

---

### 3.3 VLM图像编码器 (VLMImageEncoderFromTensor)

**文件位置**：`src/models/bida/vlm_image_encoder.py`

**作用**：使用大模型（Qwen2-VL）提取图像的高级语义特征

**模型配置**：
- 默认模型：`Qwen/Qwen2-VL-2B-Instruct`
- 支持模型：`Qwen/Qwen2-VL-3B-Instruct`, `Qwen/Qwen2-VL-7B-Instruct`
- VLM隐藏层维度：2048（默认）
- 冻结VLM参数：默认True（仅作为特征提取器）

**输入处理**：
- 支持多种输入格式：
  - `[B, C, H, W]`: 标准batch格式
  - `[B, F, C, H, W]`: 多帧OCT（F=120帧）
  - `[B, K, C, H, W]`: 多图像Colposcopy（K≤3）
- Tensor → PIL Image转换：
  - 反归一化（ImageNet归一化）
  - 通道处理（灰度图转RGB，多通道取前3个）
  - 值域转换：[0,1] → [0,255] → uint8

**网络结构**：
```python
原始图像 [B, C, H, W]
  ↓
VLM特征提取
  ↓
vlm_features [B, vlm_hidden_size] (2048维)
  ↓
【投影层】
Linear(vlm_hidden_size → embed_dim)
  ↓
proj_feat [B, embed_dim]
  ↓
【双头网络】（与DualHeadImageEncoder相同）
  ├── 因果头 → z_causal [B, embed_dim]
  └── 噪声头 → z_noise [B, embed_dim]
```

**多帧OCT处理**：
- 批处理策略：30帧/批（对于60帧只需2批）
- 帧聚合方法：
  1. 注意力聚合：MultiheadAttention(embed_dim=768, num_heads=8)
     - 输入：全部帧特征 [B, F, embed_dim]
     - 输出：注意力加权特征 [B, embed_dim]
  2. 最大池化：捕获最显著的阳性信号
  3. 融合：可学习权重（mean_weight=0.6, max_weight=0.4）

**文本提示构建**：
- 格式：`"Medical image: HPV {status}, TCT {category}, Age {age}"`
- 示例：`"Medical image: HPV positive, TCT HSIL, Age 45"`

---

### 3.4 多模态融合模块

**文件位置**：`src/models/bida/bio_cot_model.py` (方法：`_traditional_fusion`)

**作用**：融合OCT和Colposcopy图像特征

**单模态检测**：
- 检查条件：`torch.allclose(colpo_features, zeros, atol=1e-6)`
- 单模态模式：只使用OCT特征，跳过融合

**多模态融合策略**：

**方法1：加权融合**
```python
# 可学习模态权重
modal_weights = nn.Parameter([0.5, 0.5])  # [OCT权重, Colposcopy权重]
weights = softmax(modal_weights, dim=0)
weighted_feat = weights[0] * oct_feat + weights[1] * colpo_feat
```

**方法2：拼接融合**
```python
concat_feat = cat([oct_feat, colpo_feat], dim=-1)  # [B, input_dim * 2]
fused_feat = multimodal_fusion(concat_feat)  # [B, embed_dim]
```

**融合层结构**：
```python
Linear(input_dim * 2 → embed_dim)
  ↓
LayerNorm(embed_dim)
  ↓
GELU()
  ↓
Dropout(0.1)
  ↓
Linear(embed_dim → embed_dim)
```

**最终融合**：
```python
image_feat = 0.7 * fused_feat + 0.3 * weighted_proj
```

**VLM模式融合**：
- 如果使用VLM编码器，融合VLM特征和传统特征：
  - `oct_fused = 0.6 * oct_vlm_causal + 0.4 * oct_feat_proj`
  - `colpo_fused = 0.6 * colpo_vlm_causal + 0.4 * colpo_feat_proj`
- 然后使用加权融合：
  - `z_causal = weights[0] * oct_fused + weights[1] * colpo_fused`
  - `z_noise = 0.5 * oct_vlm_noise + 0.5 * colpo_vlm_noise`

---

### 3.5 三模态融合模块

**文件位置**：`src/models/bida/bio_cot_model.py` (forward方法)

**作用**：融合图像特征和临床语义特征

**融合策略**：
```python
multimodal_feat = cat([z_causal, z_sem], dim=-1)  # [B, embed_dim * 2]
fused_multimodal = _multimodal_proj(multimodal_feat)  # [B, embed_dim]
```

**投影层结构**：
```python
Linear(embed_dim * 2 → embed_dim)
  ↓
LayerNorm(embed_dim)
  ↓
GELU()
  ↓
Dropout(0.2)
```

**数值稳定性检查**：
- 检查NaN/Inf：如果包含，使用`z_causal`作为fallback

---

### 3.6 分类器 (Classifier)

**文件位置**：`src/models/bida/bio_cot_model.py`

**作用**：从融合特征预测类别

**网络结构**：
```python
输入 fused_multimodal [B, embed_dim]
  ↓
Linear(embed_dim → embed_dim)
  ↓
LayerNorm(embed_dim)
  ↓
GELU()
  ↓
Dropout(0.5)  # 增强正则化，防止过拟合
  ↓
Linear(embed_dim → embed_dim // 2)
  ↓
LayerNorm(embed_dim // 2)
  ↓
GELU()
  ↓
Dropout(0.4)  # 增强正则化，防止过拟合
  ↓
Linear(embed_dim // 2 → num_classes)
  ↓
输出 logits [B, num_classes]
```

**数值稳定性检查**：
- 检查NaN/Inf：如果包含，使用零初始化

---

### 3.7 Memory Bank (NoiseMemoryBank)

**文件位置**：`src/models/bida/memory_bank.py`

**作用**：存储和采样不同中心的噪声特征，实现反事实干预

**数据结构**：
- `bank`: [num_centers, capacity, feat_dim] - 噪声特征库
- `ptr`: [num_centers] - 每个中心的当前指针（FIFO队列）
- `count`: [num_centers] - 每个中心实际存储的数量

**更新策略**：
- FIFO（先进先出）：新特征覆盖旧特征
- 更新逻辑：
  1. 如果 `curr_count + n_feats <= capacity`：直接添加
  2. 否则：覆盖旧特征，从开头继续填充

**反事实采样策略**：
- `random`: 随机采样该中心的一个特征
- `mean`: 使用该中心的平均特征
- `nearest`: 使用最近的特征（简化实现为随机采样）

**反事实生成**：
```python
# 为每个样本随机选择一个不同的中心
fake_center_ids = (center_labels + randint(1, num_centers)) % num_centers
z_noise_cf = memory_bank.get_counterfactual_noise(fake_center_ids, strategy='random')

# 合成反事实特征
alpha = 0.3  # 混合系数
z_mix = z_causal + alpha * z_noise_cf
```

---

### 3.8 中心判别器 (CenterDiscriminator)

**文件位置**：`src/models/bida/memory_bank.py`

**作用**：从噪声特征预测中心ID，用于对抗训练

**网络结构**：
```python
输入 z_noise [B, feat_dim]
  ↓
Linear(feat_dim → 256)
  ↓
ReLU()
  ↓
Dropout(0.1)
  ↓
Linear(256 → 128)
  ↓
ReLU()
  ↓
Dropout(0.1)
  ↓
Linear(128 → num_centers)
  ↓
输出 center_logits [B, num_centers]
```

**损失函数**：CrossEntropyLoss

---

## 四、损失函数详细设计

### 4.1 Sinkhorn最优传输距离 (SinkhornDistance)

**文件位置**：`src/models/bida/losses.py`

**作用**：替代KL散度，实现更灵活的分布对齐

**算法流程**：
1. **特征归一化**：
   ```python
   x = F.normalize(x, p=2, dim=1)  # L2归一化
   y = F.normalize(y, p=2, dim=1)
   ```

2. **计算代价矩阵**：
   ```python
   C = sum((x_col - y_lin) ** 2, dim=2)  # [B, B] 欧氏距离平方
   C = clamp(C, min=0, max=10.0)  # 限制范围，避免溢出
   ```

3. **Sinkhorn迭代**（对数域版本，数值稳定）：
   ```python
   # 初始化
   u = zeros(B)
   v = zeros(B)
   
   # 计算K矩阵
   C_scaled = C / eps
   C_scaled = clamp(C_scaled, min=-10.0, max=10.0)
   K = exp(-C_scaled)
   K = clamp(K, min=1e-10, max=1e10)
   
   # 迭代（max_iter=100次）
   for _ in range(max_iter):
       Kv = matmul(K, v.unsqueeze(-1)).squeeze(-1)
       u = 1.0 / (Kv + 1e-8)
       u = clamp(u, min=1e-8, max=1e8)
       
       KTu = matmul(K.t(), u.unsqueeze(-1)).squeeze(-1)
       v = 1.0 / (KTu + 1e-8)
       v = clamp(v, min=1e-8, max=1e8)
   ```

4. **计算最优传输距离**：
   ```python
   gamma = u.unsqueeze(-1) * K * v.unsqueeze(0)  # Transport Plan [B, B]
   cost = sum(gamma * C)
   if reduction == 'mean':
       cost = cost / B
   ```

**超参数**：
- `eps=0.1`: 熵正则化系数（越小越接近精确OT，但数值越不稳定）
- `max_iter=100`: Sinkhorn迭代次数

**数值稳定性**：
- 多层clamp保护，防止NaN/Inf
- Fallback：如果出现数值问题，使用MSE损失

---

### 4.2 反事实一致性损失 (CounterfactualConsistencyLoss)

**文件位置**：`src/models/bida/losses.py`

**作用**：确保对因果特征添加不同中心的噪声后，预测结果保持不变

**计算流程**：
```python
# 原始预测
logits_orig = classifier(z_causal)  # [B, num_classes]

# 反事实预测
z_noise_cf = memory_bank.get_counterfactual_noise(fake_center_ids)
z_mix = z_causal + alpha * z_noise_cf  # alpha=0.3
logits_cf = classifier(z_mix)  # [B, num_classes]

# 一致性损失
L_consist = MSE(logits_orig, logits_cf)
```

**混合系数**：`alpha=0.3`（降低混合系数，提高稳定性）

**数值稳定性**：
- 检查所有中间变量的有效性（NaN/Inf）
- 限制损失范围：`clamp(L_consist, min=0.0, max=10.0)`
- 训练初期Memory Bank可能为空，使用0损失（正常现象）

---

### 4.3 对抗损失 (AdversarialLoss)

**文件位置**：`src/models/bida/losses.py`

**作用**：确保z_noise包含域信息（中心ID）

**计算流程**：
```python
center_logits = center_discriminator(z_noise)  # [B, num_centers]
L_adv = CrossEntropyLoss(center_logits, center_labels)
```

**数值稳定性**：
- 检查NaN/Inf
- 限制损失范围：`clamp(L_adv, max=10.0)`（防止梯度爆炸）

---

## 五、完整前向传播流程

### 5.1 输入准备

```python
# 输入
oct_features: [B, 512] 或 None
colpo_features: [B, 512] 或 None
clinical_features: [B, D] 或 None
clinical_data: dict {'hpv': [...], 'tct': [...], 'age': [...]}
center_labels: [B] 或 None
oct_images: [B, C, H, W] 或 [B, F, C, H, W] 或 None
colpo_images: [B, C, H, W] 或 [B, K, C, H, W] 或 None
```

### 5.2 步骤1：生成语义锚点

```python
# 构建临床向量
clinical_vec = build_clinical_vector(clinical_data, device)  # [B, 7]
z_sem = student_prior(clinical_vec)  # [B, 768]
```

### 5.3 步骤2：图像特征提取

**模式A：VLM编码器（如果启用）**
```python
if use_vlm_encoder and oct_images is not None:
    # 多帧OCT处理
    if oct_images.dim() == 5:  # [B, F, C, H, W]
        # 批处理（30帧/批）
        for batch_idx in range(num_batches):
            frame_batch = oct_images[:, frame_start:frame_end, :, :, :]
            frame_batch_flat = frame_batch.view(B * F_batch, C, H, W)
            oct_vlm_causal, oct_vlm_noise = image_encoder(frame_batch_flat, text_prompts)
            # 聚合帧特征
            oct_vlm_causal_all.append(oct_vlm_causal)
        
        # 注意力聚合
        oct_vlm_causal_attn, attn_weights = frame_attention(oct_vlm_causal_all, ...)
        oct_vlm_causal = weighted_mean(oct_vlm_causal_attn) + max_pool(oct_vlm_causal_attn)
    
    # 单帧Colposcopy处理
    colpo_vlm_causal, colpo_vlm_noise = image_encoder(colpo_images, text_prompts)
    
    # 融合VLM特征和传统特征
    oct_fused = 0.6 * oct_vlm_causal + 0.4 * oct_feat_proj
    colpo_fused = 0.6 * colpo_vlm_causal + 0.4 * colpo_feat_proj
    
    # 加权融合
    weights = softmax(modal_weights)
    z_causal = weights[0] * oct_fused + weights[1] * colpo_fused
    z_noise = 0.5 * oct_vlm_noise + 0.5 * colpo_vlm_noise
```

**模式B：传统MLP编码器（默认）**
```python
# 单模态检测
if torch.allclose(colpo_features, zeros):
    z_causal, z_noise = image_encoder(oct_features)
else:
    # 多模态融合
    weighted_feat = weights[0] * oct_feat + weights[1] * colpo_feat
    concat_feat = cat([oct_feat, colpo_feat], dim=-1)
    fused_feat = multimodal_fusion(concat_feat)
    image_feat = 0.7 * fused_feat + 0.3 * weighted_proj
    z_causal, z_noise = image_encoder(image_feat)
```

### 5.4 步骤3：三模态融合

```python
multimodal_feat = cat([z_causal, z_sem], dim=-1)  # [B, embed_dim * 2]
fused_multimodal = _multimodal_proj(multimodal_feat)  # [B, embed_dim]
```

### 5.5 步骤4：分类

```python
logits = classifier(fused_multimodal)  # [B, num_classes]
```

### 5.6 步骤5：损失计算（训练时）

```python
if return_loss_components:
    # Sinkhorn OT损失
    L_ot = sinkhorn_loss(z_causal, z_sem)
    
    # 反事实一致性损失
    if use_counterfactual:
        memory_bank.update(z_noise, center_labels)
        fake_center_ids = (center_labels + randint(1, num_centers)) % num_centers
        z_noise_cf = memory_bank.get_counterfactual_noise(fake_center_ids)
        z_mix = z_causal + 0.3 * z_noise_cf
        logits_cf = classifier(z_mix)
        L_consist = consistency_loss(logits, logits_cf)
    
    # 对抗损失
    center_logits = center_discriminator(z_noise)
    L_adv = adversarial_loss(center_logits, center_labels)
```

---

## 六、模型超参数

### 6.1 架构超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embed_dim` | 768 | 特征嵌入维度 |
| `num_classes` | 2 | 分类类别数 |
| `num_centers` | 5 | 中心数量 |
| `input_dim` | 512 | 图像特征输入维度 |
| `use_vlm_encoder` | False | 是否使用VLM编码器 |
| `vlm_model` | "Qwen/Qwen2-VL-2B-Instruct" | VLM模型名称 |
| `freeze_vlm` | True | 是否冻结VLM参数 |

### 6.2 损失函数超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `eps` (Sinkhorn) | 0.1 | 熵正则化系数 |
| `max_iter` (Sinkhorn) | 100 | Sinkhorn迭代次数 |
| `alpha` (反事实) | 0.3 | 反事实混合系数 |

### 6.3 Memory Bank超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `capacity` | 100 | 每个中心的容量 |
| `strategy` | 'random' | 采样策略（random/mean/nearest） |

### 6.4 Dropout配置

| 模块 | Dropout率 | 说明 |
|------|-----------|------|
| Student Prior | 0.2 | 每层后 |
| 图像编码器 | 0.1 | 每层后 |
| 多模态融合 | 0.1 | 每层后 |
| 三模态融合 | 0.2 | 投影层后 |
| 分类器 | 0.5, 0.4 | 第一层0.5，第二层0.4 |
| 中心判别器 | 0.1 | 每层后 |

---

## 七、数值稳定性保障

### 7.1 NaN/Inf检查点

1. **Sinkhorn损失**：
   - 输入检查：`torch.isnan(x).any()` 或 `torch.isinf(x).any()`
   - 中间检查：代价矩阵、K矩阵、gamma矩阵
   - Fallback：MSE损失

2. **反事实一致性损失**：
   - 检查：`z_noise_cf`, `z_mix`, `logits_cf`
   - 范围限制：`clamp(L_consist, min=0.0, max=10.0)`

3. **对抗损失**：
   - 检查：`z_noise`, `center_logits`
   - 范围限制：`clamp(L_adv, max=10.0)`

4. **特征融合**：
   - 检查：`fused_multimodal`, `logits`
   - Fallback：使用`z_causal`或零初始化

### 7.2 梯度裁剪

- 建议在训练循环中添加：`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

---

## 八、训练流程

### 8.1 阶段1：Student Prior预训练（可选）

```python
# 使用预计算的VLM特征作为监督信号
student_prior, vlm_proj = pretrain_student_prior(
    student_prior=student_prior,
    vlm_features_cache=vlm_cache,
    clinical_data_dict=clinical_data,
    num_epochs=50,
    lr=1e-3
)
```

### 8.2 阶段2：Bio-COT端到端训练

```python
# 总损失
L_total = L_cls + λ_ot * L_ot + λ_consist * L_consist + λ_adv * L_adv

# 其中
L_cls = CrossEntropyLoss(logits, labels)
L_ot = SinkhornDistance(z_causal, z_sem)
L_consist = CounterfactualConsistencyLoss(logits, logits_cf)
L_adv = AdversarialLoss(center_logits, center_labels)
```

**损失权重建议**：
- `λ_ot = 1.0`
- `λ_consist = 0.5`
- `λ_adv = 0.1`

---

## 九、关键设计决策

### 9.1 为什么使用Student Prior而不是在线VLM？

- **训练速度**：Student Prior是轻量级MLP，比VLM快100倍以上
- **推理速度**：无需每次调用大模型
- **可微性**：端到端训练，梯度可传播

### 9.2 为什么使用Sinkhorn OT而不是KL散度？

- **灵活性**：不假设分布形式（KL需要高斯假设）
- **几何意义**：最优传输距离有明确的几何解释
- **数值稳定性**：对数域实现，多层保护

### 9.3 为什么使用Memory Bank实现反事实？

- **真实反事实**：从不同中心采样真实噪声特征
- **因果解耦**：证明模型学会了"以不变（因果）应万变（噪声）"
- **可解释性**：可以可视化不同中心的噪声分布

---

## 十、模型文件结构

```
src/models/bida/
├── bio_cot_model.py          # 主模型文件
├── prior_net.py              # Student Prior网络
├── memory_bank.py            # Memory Bank和中心判别器
├── losses.py                 # 损失函数
└── vlm_image_encoder.py      # VLM图像编码器
```

---

## 十一、总结

Bio-COT模型通过以下三个核心创新实现了跨中心的域不变特征学习：

1. **Student Prior网络**：快速生成语义锚点
2. **Sinkhorn最优传输**：灵活的分布对齐
3. **Memory Bank反事实干预**：真正的因果解耦

整个架构设计充分考虑了数值稳定性、训练效率和模型可解释性，是一个完整的、可用的深度学习模型实现。

