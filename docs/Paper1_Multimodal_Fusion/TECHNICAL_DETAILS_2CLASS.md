# 2分类模型技术详解

## 📋 模型架构概览

### 整体架构: CNN Multimodal Transformer

你的模型是一个**多模态融合深度学习模型**，用于宫颈病变的二分类诊断（正常 vs 异常）。

```
输入层 → 编码器层 → 融合层 → 分类器层 → 输出
├─ OCT图像 → OCT编码器 ───┐
├─ Colposcopy图像 → Colposcopy编码器 ──┤
└─ 临床特征 → 临床特征编码器 ──┘  融合层
```

---

## 🏗️ 核心组件详解

### 1. OCT图像编码器 (`ConvEncoder`)

**位置**: `cnn_multimodal_model.py:18-105`

#### 架构设计

```python
ConvEncoder(
    in_channels=3,      # RGB图像
    base_dim=64,       # 基础通道数
    depth=5,           # 网络深度
    embed_dim=768,     # 输出嵌入维度
    dropout=0.1        # Dropout率
)
```

#### 核心技术

**a) SE (Squeeze-and-Excitation) 注意力模块**
```python
# SEBlock 工作原理:
# 1. 全局平均池化 → 获取全局上下文
pool = AdaptiveAvgPool2d(1)  # [B, C, H, W] → [B, C, 1, 1]

# 2. 通道注意力计算
fc = Sequential(
    Conv2d(C, C//16, 1),  # 降维
    GELU(),
    Conv2d(C//16, C, 1),   # 升维
    Sigmoid()              # 生成注意力权重
)

# 3. 应用注意力
output = input * attention_weights
```
**优势**: 自适应地重新校准通道响应，关注重要特征

**b) DSConv (深度可分离卷积) 残差块**
```python
DSConvBlock:
    Depthwise Conv (3×3)    # 空间卷积
    Pointwise Conv (1×1)    # 通道卷积
    BatchNorm + GELU        # 归一化和激活
    SE注意力               # 通道增强
    Dropout                 # 正则化
    Residual Connection     # 残差连接
```
**优势**: 参数少、效率高、适合移动端

**c) 网络结构**
```
Stem层: 7×7卷积，stride=2
  ↓ [224×224] → [112×112]
  
5个Stage:
  Stage 1: 64通道 → [112×112]
  Stage 2: 128通道 → [56×56]
  Stage 3: 256通道 → [28×28]
  Stage 4: 512通道 → [14×14]
  Stage 5: 512通道 → [7×7]
  
全局平均池化: [7×7] → [1×1]
投影层: 512 → 768维度
```

**d) 注意力池化**
```python
# 对于多帧OCT图像（48帧序列）
attn_pool_query = Parameter(randn(1, 768))  # 可学习的查询向量

# 每个时间步的特征: [B, 48, 768]
attention_scores = matmul(queries, query)   # [B, 48, 1]
attention_weights = softmax(attention_scores)
output = weighted_sum(features, weights)    # [B, 768]
```
**功能**: 从48帧中学习提取最相关的时间步

---

### 2. Colposcopy图像编码器

**位置**: `cnn_multimodal_model.py:141-145`

```python
# 与OCT编码器结构相同
self.col_encoder = ConvEncoder(
    in_channels=3,
    base_dim=64,
    depth=5,
    embed_dim=768,
    dropout=0.1
)

# 处理3张Colposcopy图像
col_features = []
for img in [img1, img2, img3]:
    feat = col_encoder(img)
    col_features.append(feat)

# 平均池化融合
col_final = mean(col_features)  # [B, 768]
```

**设计理念**: 
- 与OCT编码器共享架构，便于迁移学习
- 分别处理每张图像，避免信息混合
- 平均池化保持所有视角信息

---

### 3. 临床特征编码器

**位置**: `cnn_multimodal_model.py:174-195`

```python
ClinicalEncoder(
    input_dim=8,       # 8维临床特征
    hidden_dim=64,     # 隐藏层维度
    embed_dim=256,     # 输出维度
    dropout=0.1
)

# 网络结构:
Sequential(
    Linear(8, 64),
    LayerNorm(64),
    GELU(),
    Dropout(0.1),
    
    Linear(64, 128),
    LayerNorm(128),
    GELU(),
    Dropout(0.1),
    
    Linear(128, 256),
    LayerNorm(256),
    GELU(),
    Dropout(0.1)
)
```

**临床特征包括**:
- 年龄 (AGE)
- HPV状态 (HPV清洗)
- TCT结果 (TCT清洗)
- 其他相关临床信息

**设计特点**:
- 使用LayerNorm而非BatchNorm（适合小batch）
- 渐进式维度扩展（8→64→128→256）
- 多层Dropout防止过拟合

---

### 4. 跨模态注意力融合

**位置**: `cnn_multimodal_model.py:224-321`

#### 架构图

```
    OCT特征 [B,768]
        ↓
    Colposcopy特征 [B,768]  ──► 跨模态注意力 [B,3,768]
        ↓
    临床特征 [B,256] ─┐
                      ↓
                 拼接后投影 [B,1792]
                      ↓
                 融合层 [B,512]
                      ↓
                  分类器 [B,2]
```

#### 技术细节

**a) 增强跨模态注意力机制**
```python
EnhancedCrossModalAttention(
    embed_dim=512,
    num_heads=8,           # 8个注意力头
    dropout=0.1,
    use_causal_adjustment=True  # 可选因果调整
)

# 注意力计算:
Q = q_proj([oct, col, clinical])      # [B, 3, 512]
K = k_proj([oct, col, clinical])      # [B, 3, 512]
V = v_proj([oct, col, clinical])      # [B, 3, 512]

scores = Q @ K^T / sqrt(d_k)         # [B, 3, 3]
attn_weights = softmax(scores)       # 模态间的交互
output = attn_weights @ V            # [B, 3, 512]
```

**b) 因果调整（Causal Adjustment）**
```python
# 可选的因果调整模块
if use_causal_adjustment:
    causal_weight = sigmoid(features)  # 学习因果权重
    adjustment = causal_proj(features)
    output = features + causal_weight * adjustment
```
**目的**: 调整因果关系，提高跨中心泛化能力

**c) 前馈网络**
```python
FFN = Sequential(
    Linear(512, 2048),   # 扩展到4倍
    GELU(),
    Dropout(0.1),
    Linear(2048, 512),   # 压缩回原始维度
    Dropout(0.1)
)
```

---

### 5. 融合层与分类器

**位置**: `cnn_multimodal_model.py:323-362`

```python
# 融合层
fusion_layers = Sequential(
    Linear(512, 512),
    LayerNorm(512),
    GELU(),
    Dropout(0.5),              # 强正则化
    
    Linear(512, 256),
    LayerNorm(256),
    GELU(),
    Dropout(0.5)
)

# 分类器
classifier = Sequential(
    Linear(256, 128),
    LayerNorm(128),
    GELU(),
    Dropout(0.5),
    
    Linear(128, 2)             # 2分类输出
)
```

**设计理念**:
- 渐进式维度降低（512→256→128→2）
- 高Dropout率（0.5）防止过拟合
- 每层都有LayerNorm保证训练稳定

---

## 📊 数据流程详解

### 1. 数据加载 (`EnhancedMultimodalCervicalDataset`)

**位置**: `enhanced_multimodal_dataset.py:17-402`

#### 数据预处理流程

```
CSV标签文件
  ↓
加载患者ID、OCT ID、标签等
  ↓
查找OCT文件夹（48帧PNG图像）
  ↓
查找Colposcopy文件夹（3张JPG图像）
  ↓
解析临床特征（年龄、HPV、TCT等）
  ↓
应用数据增强（训练时）
  ↓
返回 (oct_images, col_images, clinical_features, label)
```

#### 关键技术

**a) 增强的OCT处理**
```python
EnhancedOCTProcessor(
    num_points=12,                # 12个采样点
    frames_per_point=10,         # 每点10帧
    use_temporal_consistency=True,  # 时间一致性
    use_multi_scale=True        # 多尺度特征
)

# 处理流程:
1. 提取12个关键点
2. 每点采样10帧 → 120帧总数据
3. 下采样到48帧
4. 应用时序增强
5. 多尺度特征融合
```

**b) 数据增强策略**
```python
训练时增强:
- 随机翻转 (RandomFlip)
- 随机旋转 (±15度)
- 颜色抖动 (ColorJitter 0.2)
- 随机擦除 (RandomErasing)

OCT专用增强:
- 时序一致性增强
- 多尺度采样
- 时间轴上的随机裁剪
```

---

### 2. 训练流程

**位置**: `train_with_cnn.py`

#### 训练配置

```python
# 优化器
optimizer = AdamW(
    params=model.parameters(),
    lr=1e-4,           # 学习率
    weight_decay=1e-5  # L2正则化
)

# 学习率调度
scheduler = CosineAnnealingLR(
    optimizer, 
    T_max=num_epochs   # 余弦退火
)

# 混合精度训练
scaler = GradScaler()  # 加速训练
```

#### 损失函数

```python
# 加权交叉熵损失（处理类别不平衡）
class_weights = [0.481, 0.519]  # 负样本权重，正样本权重

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights)
)
```

**类别不平衡处理**:
- 数据中: 负样本530个，正样本255个（约2:1）
- 使用加权损失给予正样本更高权重
- 使用WeightedRandomSampler平衡采样

#### 训练循环

```python
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    for oct, col, clinical, label in train_loader:
        with autocast():  # 混合精度
            outputs = model(oct, col, clinical)
            loss = criterion(outputs, label)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        for oct, col, clinical, label in val_loader:
            outputs = model(oct, col, clinical)
            # 计算指标...
    
    # 保存最佳模型
    if val_f1 > best_f1:
        save_checkpoint()
```

---

## 📈 性能分析

### 当前模型性能（校准后）

| 指标 | 数值 | 评估 |
|------|------|------|
| **准确率** | 78.0% | ⭐⭐ 良好 |
| **F1分数** | 65.6% | ⭐⭐ 良好 |
| **精确率** | 72.7% | ⭐⭐ 良好 |
| **召回率** | 59.7% | ⭐ 一般 |
| **最佳阈值** | 0.35 | 优化后阈值 |
| **AUC** | 0.75+ | ⭐⭐ 良好 |

### 模型复杂度

| 组件 | 参数量 | 占比 |
|------|--------|------|
| OCT编码器 | 10,009,031 | 36.9% |
| Colposcopy编码器 | 10,009,031 | 36.9% |
| 跨模态注意力 | 5,522,444 | 20.4% |
| 融合层 | 791,040 | 2.9% |
| 分类器 | 428,930 | 1.6% |
| 临床特征编码器 | 268,304 | 1.0% |
| **总计** | **27,094,573** | **100%** |

---

## 🔬 技术创新点

### 1. 多模态融合架构

- **异质模态融合**: OCT（时序）+ Colposcopy（静态）+ 临床特征（结构化）
- **自适应注意力**: 动态学习三种模态的重要性权重
- **时序建模**: 48帧OCT序列的时序信息提取

### 2. 轻量高效设计

- **深度可分离卷积**: 减少参数量70%+
- **SE注意力**: 提升特征表达力
- **渐进式融合**: 逐步融合不同层级特征

### 3. 类别不平衡处理

- **加权损失函数**: 给少数类更高权重
- **加权采样**: 平衡训练数据分布
- **阈值优化**: 校准后最佳阈值 0.35

### 4. 训练稳定性

- **LayerNorm**: 小batch下的稳定训练
- **混合精度**: 1.5x加速
- **Dropout 0.5**: 强正则化防止过拟合
- **余弦退火**: 平滑的学习率衰减

---

## 🎯 改进方向

### 已实现的改进

1. ✅ **混合精度训练** - 加速1.5x
2. ✅ **加权采样** - 处理类别不平衡
3. ✅ **余弦退火** - 更好的收敛
4. ✅ **温度缩放校准** - 校准后准确率提升到78%

### 可进一步改进

1. **Focal Loss** - 处理困难样本
   ```python
   FocalLoss(gamma=2.0, alpha=[0.3, 0.7])
   ```

2. **更多数据增强**
   - RandAugment
   - MixUp
   - CutMix

3. **更长的训练**
   - 当前: 2 epochs
   - 建议: 15-20 epochs

4. **预训练模型**
   - 使用ImageNet预训练权重
   - 迁移学习

5. **集成学习**
   - Deep Ensemble
   - 多个模型的投票

---

## 📊 数据统计

### 训练数据

- **总样本数**: 985
- **训练集**: 785 (79.7%)
- **测试集**: 200 (20.3%)
- **OCT目录**: 985个
- **Colposcopy目录**: 985个

### 类别分布

- **负样本（正常）**: 663个 (67.3%)
- **正样本（异常）**: 322个 (32.7%)
- **类别比例**: 约 2.06:1

### 数据来源

- **5个医疗中心**:
  - 恩施 (324样本)
  - 荆州 (55样本)
  - 十堰 (59样本)
  - 武大 (69样本)
  - 襄阳 (278样本)

---

## 💡 下一步: 5分类探索

基于当前2分类的技术基础，我们可以：

1. **保持编码器不变** - 复用OCT、Colposcopy、临床特征编码器
2. **只修改分类器** - 从 `nn.Linear(128, 2)` 改为 `nn.Linear(128, 5)`
3. **使用迁移学习** - 加载2分类预训练权重
4. **处理类别不平衡** - Focal Loss + 加权采样

**预期挑战**:
- 5个类别的样本分布极不平衡
- 需要大量数据增强和正则化
- 可能需要更多训练时间

