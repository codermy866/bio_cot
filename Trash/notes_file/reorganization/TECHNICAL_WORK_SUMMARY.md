# 项目技术工作内容梳理

## 📋 一、研究问题与目标

### 1.1 核心研究问题
**基于多模态医学影像的宫颈病变智能诊断系统**

- **应用场景**：基层医疗机构的宫颈癌筛查
- **临床需求**：提高诊断准确性，减少漏诊和误诊
- **技术挑战**：多模态数据融合、因果关系建模、不确定性量化

### 1.2 研究目标
1. **性能目标**：验证集准确率 > 70%，AUC > 0.75
2. **方法目标**：提出可学习因果图 + 不确定性分解的融合框架
3. **应用目标**：支持多中心验证，具备临床可解释性

---

## 🎯 二、核心技术架构

### 2.1 整体框架：增强因果约束贝叶斯CLIP

```
┌─────────────────────────────────────────────────────────────┐
│            Enhanced Causal Bayesian CLIP Framework           │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌─────▼─────┐        ┌─────▼─────┐
   │  OCT    │          │ Colposcopy│        │ Clinical  │
   │ 120帧   │          │   3帧     │        │   7维     │
   └────┬────┘          └─────┬─────┘        └─────┬─────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Swin-T Encoder    │
                    │  (部分微调)        │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Bayesian Encoder   │
                    │  μ, σ² (变分推断)  │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Learnable Causal  │
                    │ Graph Discovery   │
                    │ (DAG + 稀疏 + 干预)│
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Uncertainty       │
                    │ Decomposition     │
                    │ (认知 + 偶然)      │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Causal-Constrained│
                    │ Multi-Head Attn   │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Classifier      │
                    │   (2-class)       │
                    └───────────────────┘
```

### 2.2 核心创新点

#### ✅ 创新点1：可学习因果图发现
**问题**：传统方法使用固定因果图，无法适应数据分布

**解决方案**：
- **数据驱动学习**：基于特征学习因果邻接矩阵
- **领域知识融合**：硬约束（Clinical → OCT/Colposcopy）
- **DAG约束**：NOTEARS风格惩罚 + 上三角矩阵
- **稀疏正则化**：L1/L0近似，减少虚假因果关系
- **干预反馈**：伪干预训练，增强因果敏感性

**技术细节**：
```python
# 因果发现网络
causal_discovery = MLP(2304 → 512 → 256 → 9)
causal_adj = Sigmoid(causal_discovery(concat_features))

# DAG惩罚
dag_penalty = ReLU(trace(exp(A ∘ A)) - 3)²

# 稀疏惩罚
sparsity_penalty = mean(|causal_adj|)

# 干预惩罚
intervention_penalty = mean(intervened_edges)
```

#### ✅ 创新点2：不确定性分解
**问题**：总不确定性无法区分来源（模型 vs 数据）

**解决方案**：
- **认知不确定性**：基于融合特征的MLP估计（模型参数不确定性）
- **偶然不确定性**：基于贝叶斯编码器方差（数据固有噪声）
- **总不确定性**：epistemic + aleatoric

**技术细节**：
```python
# 认知不确定性
epistemic = MLP(fusion_feat) → Softplus → [B, 1]

# 偶然不确定性
aleatoric = MLP(mean(variances)) → Softplus → [B, 1]

# 总不确定性
total_uncertainty = epistemic + aleatoric
```

#### ✅ 创新点3：贝叶斯CLIP框架
**问题**：标准CLIP无法量化预测不确定性

**解决方案**：
- **变分推断**：每个模态输出均值和方差
- **重参数化技巧**：训练时采样，推理时使用均值
- **KL正则化**：防止方差过大，保持数值稳定

**技术细节**：
```python
# 贝叶斯编码器
μ = Linear(feat)  # [B, 768]
log_σ² = Softplus(Linear(feat))  # [B, 768]
z = μ + ε·σ, ε ~ N(0,1)  # 重参数化

# KL损失
KL = 0.5 * Σ(σ² + μ² - 1 - log(σ²))
```

---

## 📊 三、数据集与实验设置

### 3.1 数据集：5centers_multi
- **训练集**：785个患者
- **验证集**：200个患者
- **任务**：二分类（正常 vs. 异常）

### 3.2 多模态数据
| 模态 | 数据量 | 维度 | 处理方式 |
|------|--------|------|----------|
| **OCT** | 120帧/患者 | [B, 120, 3, 224, 224] | Swin-T编码 → 平均池化 → [B, 768] |
| **Colposcopy** | 3帧/患者 | [B, 3, 3, 224, 224] | Swin-T编码 → 平均池化 → [B, 768] |
| **Clinical** | 7维/患者 | [B, 7] | Linear投影 → [B, 768] |

### 3.3 训练配置（当前最优）
```yaml
优化器: AdamW
学习率: 3e-4
调度器: Cosine Annealing (T_max=100)
批次大小: 24 (针对120帧OCT优化)
总epoch: 100
混合精度: 关闭 (FP32，提高稳定性)

损失权重:
  - Focal Loss: 1.0 (γ=2.0, α=auto)
  - Label Smoothing: 0.01
  - KL Loss: 0.001
  - Contrastive Loss: 0 (关闭)
  - Causal Loss: 0.001
    - DAG Penalty: 0.02
    - Sparsity Penalty: 0.0002
    - Intervention Penalty: 0.01

特征提取器:
  - Backbone: Swin-T (timm)
  - 微调策略: 解冻最后2层
  - 可训练参数: 52.64M / 55.04M (95.6%)
```

---

## 🔬 四、关键技术实现

### 4.1 特征提取器（Swin-T）
**架构**：
- **预训练模型**：`swin_tiny_patch4_window7_224` (ImageNet)
- **输入**：224×224 RGB图像
- **输出**：768维特征向量
- **微调策略**：冻结大部分层，仅解冻最后2层

**OCT处理**：
```python
# 输入: [B, 120, 3, 224, 224]
# 1. 重塑为 [B*120, 3, 224, 224]
# 2. Swin-T编码 → [B*120, 768]
# 3. 重塑为 [B, 120, 768]
# 4. 平均池化 → [B, 768]
```

**Colposcopy处理**：
```python
# 输入: [B, 3, 3, 224, 224]
# 1. 重塑为 [B*3, 3, 224, 224]
# 2. Swin-T编码 → [B*3, 768]
# 3. 重塑为 [B, 3, 768]
# 4. 平均池化 → [B, 768]
```

### 4.2 可学习因果图模块
**核心组件**：

1. **因果发现网络**：
   ```python
   causal_discovery = Sequential(
       Linear(2304, 512) → LayerNorm → GELU → Dropout(0.1),
       Linear(512, 256) → LayerNorm → GELU → Dropout(0.1),
       Linear(256, 9) → Sigmoid
   )
   ```

2. **可学习权重矩阵**：
   ```python
   causal_weights = Parameter([3, 3])  # 初始化为小随机值
   ```

3. **DAG约束**：
   - **上三角矩阵**：强制因果矩阵为上三角形式
   - **NOTEARS惩罚**：`penalty = ReLU(trace(exp(A ∘ A)) - 3)²`

4. **先验知识融合**：
   ```python
   # 硬约束：Clinical → OCT, Clinical → Colposcopy
   prior = [[0, 0, 0],
            [0, 0, 0],
            [1, 1, 0]]  # 第3行（Clinical）影响前两个模态
   ```

5. **伪干预训练**：
   - 50%概率随机选择一个模态进行干预
   - 对被干预模态的出边进行额外惩罚

### 4.3 不确定性分解模块
**认知不确定性（Epistemic）**：
```python
epistemic_net = Sequential(
    Linear(768, 384) → LayerNorm → GELU → Dropout(0.1),
    Linear(384, 1) → Softplus
)
epistemic = epistemic_net(fusion_feat)  # [B, 1]
```

**偶然不确定性（Aleatoric）**：
```python
# 基于多模态方差
variances = [oct_var, colpo_var, clinical_var]  # 每个 [B, 768]
mean_var = mean(variances)  # [B, 768]
aleatoric_net = Sequential(
    Linear(768, 384) → LayerNorm → GELU → Dropout(0.1),
    Linear(384, 1) → Softplus
)
aleatoric = aleatoric_net(mean_var)  # [B, 1]
```

### 4.4 跨模态融合
**因果约束的多头注意力**：
```python
# 1. 应用因果权重
multimodal_seq = [oct_feat, colpo_feat, clinical_feat]  # [B, 3, 768]
causal_weights = causal_adj  # [B, 3, 3]
weighted_seq = causal_weights @ multimodal_seq  # [B, 3, 768]

# 2. 多头注意力
attn_output = MultiHeadAttention(weighted_seq, weighted_seq, weighted_seq)
# heads=8, dropout=0.1

# 3. 平均池化
fusion_feat = mean(attn_output, dim=1)  # [B, 768]
```

**特征融合层**：
```python
fusion_layers = Sequential(
    Linear(2304, 1536) → LayerNorm → GELU → Dropout(0.5),
    Linear(1536, 768)
)
final_feat = fusion_layers(concat([oct, colpo, clinical]))
```

### 4.5 分类器
```python
classifier = Sequential(
    Linear(768, 384) → LayerNorm → GELU → Dropout(0.5),
    Linear(384, 2)
)
logits = classifier(fusion_feat)  # [B, 2]
```

---

## 📈 五、训练优化历程

### 5.1 关键优化点

#### 优化1：确保完整数据加载
**问题**：怀疑未使用全部120帧OCT图像

**解决方案**：
- 设置 `oct_num_frames=120`
- 关闭 `cache_oct_features=False`
- 添加调试日志：`print(f"OCT shape: {oct_images.shape}")`

**验证**：日志显示 `[24, 120, 3, 224, 224]`，确认全部加载

#### 优化2：降低损失值
**问题**：初始损失 > 0.5，难以收敛

**解决方案**：
- 降低 `causal_loss_weight`: 0.002 → 0.001
- 降低 `kl_weight`: 0.003 → 0.001
- 降低 `label_smoothing`: 0.05 → 0.01
- 关闭 `contrastive_weight`: 0.05 → 0

**结果**：损失降至 0.25-0.30 范围

#### 优化3：提高GPU利用率
**问题**：GPU显存占用 < 10%，利用率低

**解决方案**：
- 增加 `batch_size`: 6 → 24 (针对120帧OCT)
- 关闭 `AMP` (使用FP32)
- 确认所有120帧参与训练

**结果**：显存占用约4-5GB，训练稳定

#### 优化4：处理NaN/Inf损失
**问题**：部分batch出现NaN/Inf

**解决方案**：
- 添加损失检查：`if torch.isnan(loss) or torch.isinf(loss): skip`
- 添加梯度裁剪：`max_norm=1.0`
- 降低正则化权重

**结果**：训练稳定，无NaN/Inf

### 5.2 当前训练状态
**运行配置**：`run_bs24_e100_noamp_fullOCT_noclc_causal001_ls001_kl0001`

**关键参数**：
- Batch size: 24
- Learning rate: 3e-4
- Causal loss weight: 0.001
- Label smoothing: 0.01
- KL weight: 0.001
- Contrastive weight: 0 (关闭)

**训练进度**：
- Epoch 2: Loss ≈ 0.25-0.30, Acc ≈ 60%
- 训练稳定，无异常

---

## 🎯 六、评估体系

### 6.1 分类性能指标
- **准确率（Accuracy）**
- **AUC-ROC**
- **F1-Score**
- **精确率（Precision）**
- **召回率（Recall / Sensitivity）**
- **特异性（Specificity）**

### 6.2 最优阈值选择
使用**Youden指数**：
```python
Youden Index = Sensitivity + Specificity - 1
optimal_threshold = argmax(Youden Index)
```

### 6.3 临床指标
- **阳性预测值（PPV）**
- **阴性预测值（NPV）**
- **阳性似然比（LR+）**
- **阴性似然比（LR-）**
- **MCC（Matthews Correlation Coefficient）**

### 6.4 不确定性评估
- **总不确定性**：epistemic + aleatoric
- **不确定性校准**：评估预测置信度与实际准确率的一致性

---

## 🔍 七、技术亮点与创新性分析

### 7.1 方法创新性评估

#### ✅ **高创新性**：可学习因果图 + 不确定性分解
- **独特性**：结合数据驱动学习和领域知识
- **技术深度**：DAG约束、稀疏正则、干预反馈
- **应用价值**：医学场景的特殊需求

#### ⚠️ **中等创新性**：贝叶斯CLIP框架
- **方法来源**：变分推断（标准方法）
- **应用创新**：在多模态医学场景中的应用
- **技术组合**：CLIP + 贝叶斯不确定性

#### ⚠️ **低创新性**：特征提取器（Swin-T）
- **方法来源**：预训练模型（ImageNet）
- **应用方式**：部分微调（标准策略）

### 7.2 与现有方法对比

| 方法 | 因果图 | 不确定性 | 可学习性 | 创新性 |
|------|--------|----------|----------|--------|
| **标准CLIP** | ❌ | ❌ | ❌ | 低 |
| **固定因果CLIP** | ✅ | ❌ | ❌ | 中 |
| **贝叶斯CLIP** | ❌ | ✅ | ❌ | 中 |
| **本方法** | ✅ | ✅ | ✅ | **高** |

### 7.3 核心贡献总结

1. **方法贡献**：
   - 提出可学习因果图发现机制（数据驱动 + 领域知识）
   - 实现不确定性分解（认知 vs 偶然）
   - 设计伪干预训练策略

2. **应用贡献**：
   - 多模态医学影像融合（OCT + Colposcopy + Clinical）
   - 多中心验证（5个医疗中心）
   - 临床可解释性（因果图可视化）

3. **技术贡献**：
   - 完整的训练和评估流程
   - 详细的实验方法文档
   - 可复现的代码实现

---

## 📝 八、当前工作状态

### 8.1 已完成工作
- ✅ 模型架构设计与实现
- ✅ 可学习因果图模块
- ✅ 不确定性分解模块
- ✅ 训练流程优化
- ✅ 实验方法文档编写
- ✅ 因果图可视化工具

### 8.2 进行中工作
- 🔄 模型训练（100 epochs）
- 🔄 超参数调优
- 🔄 损失函数优化

### 8.3 待完成工作
- ⏳ 验证集性能评估
- ⏳ 消融实验
- ⏳ 多中心验证
- ⏳ 可解释性分析
- ⏳ 论文撰写

---

## 🎓 九、技术路线图

### 阶段1：基础框架（已完成）
- [x] 多模态数据加载
- [x] 特征提取器（Swin-T）
- [x] 基础融合框架

### 阶段2：因果约束（已完成）
- [x] 可学习因果图发现
- [x] DAG约束机制
- [x] 先验知识融合

### 阶段3：不确定性量化（已完成）
- [x] 贝叶斯编码器
- [x] 不确定性分解
- [x] KL正则化

### 阶段4：训练优化（进行中）
- [x] 损失函数设计
- [x] 超参数调优
- [ ] 性能验证（目标：AUC > 0.75）

### 阶段5：实验分析（待完成）
- [ ] 消融实验
- [ ] 多中心验证
- [ ] 可解释性分析
- [ ] 临床决策曲线分析

### 阶段6：论文撰写（待完成）
- [ ] 方法部分
- [ ] 实验结果
- [ ] 讨论与结论

---

## 💡 十、关键技术决策分析

### 10.1 为什么选择Swin-T？
- **效率**：相比ViT，计算复杂度更低
- **性能**：在ImageNet上表现优异
- **适用性**：适合医学图像的多尺度特征

### 10.2 为什么使用部分微调？
- **稳定性**：避免过拟合（数据量有限）
- **效率**：减少可训练参数，加快训练
- **迁移性**：保留预训练知识

### 10.3 为什么关闭AMP？
- **稳定性**：FP32避免数值精度问题
- **调试**：更容易发现NaN/Inf问题
- **性能**：当前batch size下，速度差异不大

### 10.4 为什么使用Focal Loss？
- **类别不平衡**：处理正常/异常样本不平衡
- **困难样本**：聚焦难以分类的样本
- **性能**：在医学分类任务中表现优异

---

## 📊 十一、模型复杂度分析

### 11.1 参数量
- **特征提取器**：55.04M（可训练52.64M）
- **主模型**：23.49M
- **总参数量**：约78.53M

### 11.2 显存占用
- **输入显存**：OCT [24, 120, 3, 224, 224] ≈ 1.7GB
- **模型显存**：约2.5GB
- **总显存**：约4-5GB（batch=24，120帧OCT）

### 11.3 训练时间
- **单epoch时间**：约2-3分钟（batch=24，785训练样本）
- **总训练时间**：约3-5小时（100 epochs）

---

## 🎯 十二、下一步工作计划

### 短期目标（1-2周）
1. **完成训练**：运行100 epochs，达到收敛
2. **性能评估**：验证集AUC > 0.75
3. **超参数调优**：进一步优化损失权重

### 中期目标（2-4周）
1. **消融实验**：
   - 移除可学习因果图
   - 移除不确定性分解
   - 移除干预反馈机制
2. **多中心验证**：5个中心独立评估
3. **可解释性分析**：因果图可视化，注意力热力图

### 长期目标（1-2月）
1. **论文撰写**：方法、实验、讨论
2. **临床验证**：与医生合作，实际应用测试
3. **方法扩展**：扩展到5分类任务

---

## 📚 十三、相关文档

### 技术文档
- `EXPERIMENTAL_METHODS.md` - 详细实验方法
- `CAUSAL_CONSTRAINED_CLIP_DEEP_ANALYSIS.md` - 因果CLIP深度分析
- `DEEP_TECHNICAL_ANALYSIS.md` - 技术模块深度分析
- `TECHNICAL_MODULES_ANALYSIS.md` - 技术模块分析

### 代码实现
- `src/models/enhanced_causal_clip.py` - 主模型实现
- `training/train_enhanced_causal_clip.py` - 训练脚本
- `utils/visualize_causal_graph.py` - 可视化工具

### 实验结果
- `enhanced_causal_clip_results/` - 训练结果目录
- `causal_analysis_detailed/` - 因果分析可视化

---

## 🎓 总结

本项目是一个**多模态医学影像智能诊断系统**，核心创新在于：

1. **可学习因果图发现**：结合数据驱动学习和领域知识，动态学习模态间因果关系
2. **不确定性分解**：区分认知不确定性和偶然不确定性，提供更可靠的预测
3. **贝叶斯CLIP框架**：在CLIP基础上引入不确定性量化，提升模型可靠性

当前工作重点：
- ✅ 模型架构已实现
- 🔄 训练优化进行中
- ⏳ 性能验证待完成

**预期贡献**：为医学AI领域提供一种新的多模态融合方法，具有更好的可解释性和可靠性。


