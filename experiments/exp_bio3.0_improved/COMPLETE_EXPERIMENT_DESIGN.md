# Bio-COT 3.0 Improved 完整实验设计（MICCAI发表标准）

> **作为深度学习实验设计专家，设计严谨、完整的实验方案，确保工作符合MICCAI发表标准**

---

## 📋 目录

1. [实验设计总览](#1-实验设计总览)
2. [Baseline对比实验](#2-baseline对比实验)
3. [消融实验](#3-消融实验)
4. [统计显著性检验](#4-统计显著性检验)
5. [性能提升策略](#5-性能提升策略)
6. [实验执行计划](#6-实验执行计划)
7. [结果分析模板](#7-结果分析模板)
8. [代码框架](#8-代码框架)

---

## 1. 实验设计总览

### 1.1 实验设计原则

1. **严谨性**: 所有实验必须可复现，使用固定随机种子
2. **完整性**: 覆盖所有必需的对比和消融实验
3. **统计性**: 所有结果必须包含统计显著性检验
4. **公平性**: 所有方法使用相同的数据划分和评估指标
5. **可解释性**: 所有结果必须有清晰的解释和分析

### 1.2 实验层次结构

```
实验层次:
├── Level 1: Baseline对比实验 (必需) ⭐⭐⭐⭐⭐
│   ├── Simple Fusion Baseline
│   ├── Standard CLIP Baseline
│   ├── ViT + Clinical Fusion
│   ├── CNN Baseline (已有)
│   ├── Swin-T Baseline (已有)
│   └── VMamba Baseline (已有)
│
├── Level 2: 消融实验 (必需) ⭐⭐⭐⭐⭐
│   ├── Knowledge Notes消融
│   ├── Visual Notes消融
│   ├── Sinkhorn OT消融
│   ├── Dual-Head消融
│   └── 组合消融
│
├── Level 3: 统计显著性检验 (必需) ⭐⭐⭐⭐⭐
│   ├── 多次运行统计
│   ├── t-test / Wilcoxon检验
│   └── 置信区间计算
│
└── Level 4: 我们的完整方法 (当前)
    └── Bio-COT 3.0 Improved (Full Model)
```

### 1.3 实验配置标准

**统一配置**:
- **随机种子**: 固定为42（所有实验）
- **数据划分**: 使用相同的数据划分（train/val/test）
- **评估指标**: AUC, Accuracy, Precision, Recall, Specificity, F1-Score
- **运行次数**: 每个实验至少运行5次（用于统计检验）
- **硬件环境**: 相同GPU型号和CUDA版本

---

## 2. Baseline对比实验

### 2.1 实验设计总览

| Baseline方法 | 优先级 | 预期AUC | 目的 | 实现状态 |
|-------------|--------|---------|------|---------|
| **Simple Fusion** | ⭐⭐⭐⭐⭐ | 70-75% | 证明注意力机制的必要性 | ❌ 待实现 |
| **Standard CLIP** | ⭐⭐⭐⭐⭐ | 75-80% | 证明Knowledge/Visual Notes的有效性 | ⚠️ 部分实现 |
| **ViT + Clinical Fusion** | ⭐⭐⭐⭐ | 72-78% | 证明最优传输的必要性 | ❌ 待实现 |
| **CNN Baseline** | ⭐⭐⭐ | 70-75% | 传统深度学习方法 | ✅ 已实现 |
| **Swin-T Baseline** | ⭐⭐⭐ | 80-85% | 现代Transformer方法 | ✅ 已实现 |
| **VMamba Baseline** | ⭐⭐⭐ | 80-85% | 不同backbone对比 | ✅ 已实现 |

### 2.2 Baseline 1: Simple Fusion Baseline ⭐⭐⭐⭐⭐

**方法描述**:
- 最简单的多模态融合方法
- 特征拼接 + MLP分类器
- 无注意力机制，无Knowledge Notes，无Visual Notes，无最优传输

**架构**:
```python
# 伪代码
F_oct = ViT(OCT_images)  # [B, 768]
F_colpo = ViT(Colposcopy_images)  # [B, 768]
F_clinical = MLP(Clinical_data)  # [B, 256]

# 简单拼接
F_concat = Concat([F_oct, F_colpo, F_clinical])  # [B, 1792]

# MLP分类器
y = MLP(F_concat)  # [B, 2]
```

**实验配置**:
```python
config = {
    'method': 'simple_fusion',
    'fusion_type': 'concatenation',
    'use_attention': False,
    'use_knowledge_notes': False,
    'use_visual_notes': False,
    'use_ot': False,
    'use_dual': False,
    'batch_size': 48,
    'learning_rate': 0.0002,
    'num_epochs': 100,
    'random_seed': 42,
}
```

**预期结果**:
- AUC: 70-75%
- Accuracy: 65-70%
- **目的**: 作为最基础的baseline，证明我们方法的各个模块都有价值

**实现文件**: `experiments/baseline/train_simple_fusion_baseline.py` (需要完善)

---

### 2.3 Baseline 2: Standard CLIP Baseline ⭐⭐⭐⭐⭐

**方法描述**:
- 标准CLIP方法（无Knowledge Notes，无Visual Notes）
- 有跨模态对比学习
- 无因果解耦，无最优传输

**架构**:
```python
# 伪代码
F_oct = ViT(OCT_images)  # [B, 768]
F_colpo = ViT(Colposcopy_images)  # [B, 768]
F_clinical = MLP(Clinical_data)  # [B, 768]

# 跨模态对比学习
L_contrastive = InfoNCE(F_oct, F_clinical) + InfoNCE(F_colpo, F_clinical)

# 简单融合
F_fused = 0.6 * F_oct + 0.4 * F_colpo  # [B, 768]
F_final = CrossAttn(F_fused, F_clinical)  # [B, 768]

# 分类
y = Classifier(F_final)  # [B, 2]
```

**实验配置**:
```python
config = {
    'method': 'standard_clip',
    'use_knowledge_notes': False,
    'use_visual_notes': False,
    'use_ot': False,
    'use_dual': False,
    'use_contrastive': True,
    'batch_size': 48,
    'learning_rate': 0.0002,
    'num_epochs': 100,
    'random_seed': 42,
}
```

**预期结果**:
- AUC: 75-80%
- Accuracy: 70-75%
- **目的**: 证明Knowledge Notes和Visual Notes的有效性

**实现文件**: `experiments/baseline/train_standard_clip_baseline.py` (需要完善)

---

### 2.4 Baseline 3: ViT + Clinical Fusion ⭐⭐⭐⭐

**方法描述**:
- ViT特征 + 临床特征简单融合
- 无Knowledge Notes，无Visual Notes，无最优传输
- 使用MSE/KL散度对齐（而非Sinkhorn OT）

**架构**:
```python
# 伪代码
F_oct = ViT(OCT_images)  # [B, 768]
F_colpo = ViT(Colposcopy_images)  # [B, 768]
F_clinical = MLP(Clinical_data)  # [B, 768]

# 简单融合
F_fused = 0.6 * F_oct + 0.4 * F_colpo  # [B, 768]

# MSE对齐（而非OT）
L_align = MSE(F_fused, F_clinical)

# 融合
F_final = Concat([F_fused, F_clinical])  # [B, 1536]
F_final = MLP(F_final)  # [B, 768]

# 分类
y = Classifier(F_final)  # [B, 2]
```

**实验配置**:
```python
config = {
    'method': 'vit_clinical_fusion',
    'use_knowledge_notes': False,
    'use_visual_notes': False,
    'use_ot': False,
    'use_mse_align': True,
    'use_dual': False,
    'batch_size': 48,
    'learning_rate': 0.0002,
    'num_epochs': 100,
    'random_seed': 42,
}
```

**预期结果**:
- AUC: 72-78%
- Accuracy: 68-73%
- **目的**: 证明Sinkhorn OT的有效性（优于MSE/KL散度）

**实现文件**: `experiments/baseline/train_vit_clinical_fusion.py` (需要创建)

---

### 2.5 Baseline 4-6: 已有Baseline

**CNN Baseline** ✅
- 文件: `experiments/baseline/train_cnn_baseline.py`
- 预期AUC: 70-75%
- **用途**: 传统深度学习方法对比

**Swin-T Baseline** ✅
- 文件: `experiments/baseline/train_swin_baseline.py`
- 预期AUC: 80-85%
- **用途**: 现代Transformer方法对比

**VMamba Baseline** ✅
- 文件: `experiments/baseline/train_vmamba_baseline.py`
- 预期AUC: 80-85%
- **用途**: 不同backbone对比

---

## 3. 消融实验

### 3.1 消融实验设计总览

| 消融实验 | 移除模块 | 预期AUC下降 | 目的 | 优先级 |
|---------|---------|------------|------|--------|
| **Ablation 1** | Knowledge Notes | -3~-5% | 证明Knowledge Notes的有效性 | ⭐⭐⭐⭐⭐ |
| **Ablation 2** | Visual Notes | -2~-4% | 证明Visual Notes的有效性 | ⭐⭐⭐⭐⭐ |
| **Ablation 3** | Sinkhorn OT | -2~-3% | 证明最优传输的有效性 | ⭐⭐⭐⭐⭐ |
| **Ablation 4** | Dual-Head | -1~-2% | 证明因果解耦的有效性 | ⭐⭐⭐⭐⭐ |
| **Ablation 5** | Knowledge + Visual | -5~-7% | 证明组合效果 | ⭐⭐⭐⭐ |
| **Ablation 6** | OT + Dual-Head | -3~-5% | 证明组合效果 | ⭐⭐⭐⭐ |

### 3.2 Ablation 1: Knowledge Notes消融 ⭐⭐⭐⭐⭐

**方法描述**:
- 移除Knowledge Notes模块
- 使用原始临床数据（而非Knowledge Note Embeddings）
- 保留Visual Notes、OT、Dual-Head

**架构变化**:
```python
# 原方法
z_sem = TextProjector(LLM_Embedding(Knowledge_Note))  # [B, 768]

# 消融后
z_sem = MLP(Clinical_data)  # [B, 768] (直接使用原始临床数据)
```

**实验配置**:
```python
config = {
    'method': 'bio_cot_v3_ablation',
    'ablation_type': 'remove_knowledge_notes',
    'use_knowledge_notes': False,  # 移除
    'use_visual_notes': True,
    'use_ot': True,
    'use_dual': True,
    'use_raw_clinical': True,  # 使用原始临床数据
    'batch_size': 48,
    'learning_rate': 0.0002,
    'num_epochs': 100,
    'random_seed': 42,
}
```

**预期结果**:
- AUC: 76-82% (vs Full Model 85%+)
- **目的**: 证明Knowledge Notes的有效性（预期下降3-5%）

**实现文件**: `experiments/exp_bio3.0_improved/training/train_ablation_knowledge_notes.py` (需要创建)

---

### 3.3 Ablation 2: Visual Notes消融 ⭐⭐⭐⭐⭐

**方法描述**:
- 移除Visual Notes模块
- 使用全局特征（而非过滤后的特征）
- 保留Knowledge Notes、OT、Dual-Head

**架构变化**:
```python
# 原方法
F_note = VisualNotes(F_patch, z_sem)  # [B, 196, 768] → [B, 768]
F_pooled = GAP(F_note)  # [B, 768]

# 消融后
F_pooled = GAP(F_patch)  # [B, 768] (直接全局池化，无过滤)
```

**实验配置**:
```python
config = {
    'method': 'bio_cot_v3_ablation',
    'ablation_type': 'remove_visual_notes',
    'use_knowledge_notes': True,
    'use_visual_notes': False,  # 移除
    'use_ot': True,
    'use_dual': True,
    'batch_size': 48,
    'learning_rate': 0.0002,
    'num_epochs': 100,
    'random_seed': 42,
}
```

**预期结果**:
- AUC: 81-85% (vs Full Model 85%+)
- **目的**: 证明Visual Notes的有效性（预期下降2-4%）

**实现文件**: `experiments/exp_bio3.0_improved/training/train_ablation_visual_notes.py` (需要创建)

---

### 3.4 Ablation 3: Sinkhorn OT消融 ⭐⭐⭐⭐⭐

**方法描述**:
- 移除Sinkhorn OT损失
- 使用MSE/KL散度替代
- 保留Knowledge Notes、Visual Notes、Dual-Head

**架构变化**:
```python
# 原方法
L_ot = SinkhornOT(z_causal, z_sem)  # Sinkhorn最优传输

# 消融后
L_align = MSE(z_causal, z_sem)  # 或 KL散度
```

**实验配置**:
```python
config = {
    'method': 'bio_cot_v3_ablation',
    'ablation_type': 'remove_ot',
    'use_knowledge_notes': True,
    'use_visual_notes': True,
    'use_ot': False,  # 移除
    'use_mse_align': True,  # 使用MSE替代
    'use_dual': True,
    'batch_size': 48,
    'learning_rate': 0.0002,
    'num_epochs': 100,
    'random_seed': 42,
}
```

**预期结果**:
- AUC: 82-86% (vs Full Model 85%+)
- **目的**: 证明Sinkhorn OT的有效性（预期下降2-3%）

**实现文件**: `experiments/exp_bio3.0_improved/training/train_ablation_ot.py` (需要创建)

---

### 3.5 Ablation 4: Dual-Head消融 ⭐⭐⭐⭐⭐

**方法描述**:
- 移除Dual-Head结构
- 使用单头编码器
- 保留Knowledge Notes、Visual Notes、OT

**架构变化**:
```python
# 原方法
z_causal, z_noise = DualHead(F_fused)  # [B, 768], [B, 768]

# 消融后
z_causal = SingleHead(F_fused)  # [B, 768]
z_noise = None
```

**实验配置**:
```python
config = {
    'method': 'bio_cot_v3_ablation',
    'ablation_type': 'remove_dual_head',
    'use_knowledge_notes': True,
    'use_visual_notes': True,
    'use_ot': True,
    'use_dual': False,  # 移除
    'batch_size': 48,
    'learning_rate': 0.0002,
    'num_epochs': 100,
    'random_seed': 42,
}
```

**预期结果**:
- AUC: 83-87% (vs Full Model 85%+)
- **目的**: 证明因果解耦的有效性（预期下降1-2%）

**实现文件**: `experiments/exp_bio3.0_improved/training/train_ablation_dual_head.py` (需要创建)

---

### 3.6 Ablation 5-6: 组合消融 ⭐⭐⭐⭐

**Ablation 5: Knowledge + Visual Notes消融**
- 同时移除Knowledge Notes和Visual Notes
- 预期AUC下降: -5~-7%

**Ablation 6: OT + Dual-Head消融**
- 同时移除OT和Dual-Head
- 预期AUC下降: -3~-5%

**目的**: 证明模块组合的协同效应

---

## 4. 统计显著性检验

### 4.1 实验设计

**运行次数**: 每个实验至少运行**5次**（使用不同随机种子）

**随机种子**: [42, 123, 456, 789, 2024]

**统计方法**:
1. **描述性统计**: 均值、标准差、95%置信区间
2. **假设检验**: 
   - t-test（如果数据正态分布）
   - Wilcoxon符号秩检验（如果数据非正态分布）
3. **效应量**: Cohen's d

### 4.2 统计检验流程

```python
# 伪代码
results = {
    'baseline_1': [auc_1, auc_2, ..., auc_5],  # 5次运行
    'baseline_2': [auc_1, auc_2, ..., auc_5],
    ...
    'full_model': [auc_1, auc_2, ..., auc_5],
}

# 1. 描述性统计
for method, aucs in results.items():
    mean = np.mean(aucs)
    std = np.std(aucs)
    ci_95 = scipy.stats.t.interval(0.95, len(aucs)-1, loc=mean, scale=std/np.sqrt(len(aucs)))
    print(f"{method}: {mean:.4f} ± {std:.4f} (95% CI: {ci_95})")

# 2. 假设检验（与Full Model对比）
for baseline in ['baseline_1', 'baseline_2', ...]:
    statistic, p_value = scipy.stats.ttest_rel(results['full_model'], results[baseline])
    # 或使用Wilcoxon检验
    statistic, p_value = scipy.stats.wilcoxon(results['full_model'], results[baseline])
    print(f"Full Model vs {baseline}: p-value = {p_value:.4f}")
    
    # 效应量
    cohens_d = (np.mean(results['full_model']) - np.mean(results[baseline])) / np.std(results[baseline])
    print(f"Cohen's d = {cohens_d:.4f}")
```

### 4.3 结果报告格式

**表格格式**:
| 方法 | AUC (Mean ± Std) | 95% CI | vs Full Model (p-value) | Cohen's d |
|------|-----------------|--------|------------------------|-----------|
| Simple Fusion | 0.7234 ± 0.0123 | [0.7123, 0.7345] | < 0.001 | 1.23 |
| Standard CLIP | 0.7821 ± 0.0098 | [0.7745, 0.7897] | < 0.001 | 0.89 |
| Full Model | **0.8523 ± 0.0076** | **[0.8476, 0.8570]** | - | - |

**显著性标准**:
- p < 0.001: *** (极显著)
- p < 0.01: ** (非常显著)
- p < 0.05: * (显著)
- p ≥ 0.05: ns (不显著)

---

## 5. 性能提升策略

### 5.1 目标性能

**当前性能**:
- AUC: 79.80%
- Accuracy: 73.81%

**目标性能** (MICCAI标准):
- AUC: **≥85%** (+5.2%)
- Accuracy: **≥80%** (+6.19%)
- F1-Score: **≥70%** (+6.67%)

### 5.2 提升策略

#### 策略1: 损失函数优化

**当前配置**:
```python
lambda_cls = 2.0
lambda_ot = 0.5
lambda_sparse = 0.01
lambda_consist = 0.2
lambda_adv = 0.5
```

**优化方案**:
```python
# 方案A: 进一步增加分类损失权重
lambda_cls = 3.0  # 从2.0增加到3.0
lambda_ot = 0.3   # 从0.5降低到0.3

# 方案B: 动态权重调整
lambda_cls = schedule(epoch)  # 早期大，后期小
lambda_ot = schedule(epoch)  # 早期小，后期大

# 方案C: 自适应权重
lambda_cls = adaptive_weight(L_cls, L_ot, ...)  # 根据损失值自适应调整
```

**预期提升**: +2-3%

#### 策略2: 数据增强

**当前**: 基础数据增强（Resize, Normalize）

**优化方案**:
```python
# 更强的数据增强
transforms = [
    RandomRotation(degrees=15),
    RandomHorizontalFlip(),
    ColorJitter(brightness=0.2, contrast=0.2),
    RandomAffine(degrees=0, translate=(0.1, 0.1)),
    RandomErasing(p=0.1),
]

# 处理类别不平衡
- SMOTE (Synthetic Minority Oversampling)
- ADASYN (Adaptive Synthetic Sampling)
- 加权采样 (Weighted Sampling)
```

**预期提升**: +2-3%

#### 策略3: 模型架构优化

**当前**: 标准架构

**优化方案**:
```python
# 1. 更深的分类器
classifier = nn.Sequential(
    nn.Linear(768, 512),
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),  # 新增
    nn.LayerNorm(256),   # 新增
    nn.GELU(),           # 新增
    nn.Dropout(0.2),
    nn.Linear(256, 2)
)

# 2. 注意力机制优化
- 使用Multi-Head Attention (8 heads → 16 heads)
- 添加残差连接
- 使用Layer Normalization

# 3. 特征融合优化
- 尝试不同的融合策略（加权、注意力、门控等）
```

**预期提升**: +1-2%

#### 策略4: 训练策略优化

**当前**: 固定学习率，100 epochs

**优化方案**:
```python
# 1. 学习率调度
scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)
# 或
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

# 2. 增加训练轮数
num_epochs = 150  # 从100增加到150

# 3. 早停策略
early_stopping = EarlyStopping(patience=20, mode='max', monitor='val_auc')

# 4. 模型集成
ensemble_models = [model_1, model_2, model_3, model_4, model_5]
final_pred = average([m(x) for m in ensemble_models])
```

**预期提升**: +1-2%

---

## 6. 实验执行计划

### 6.1 执行时间表

**总预计时间**: 8-12周 (2-3个月)

#### 阶段1: Baseline对比实验 (3-4周)

**Week 1-2**: 实现Baseline方法
- [ ] Simple Fusion Baseline (3天)
- [ ] Standard CLIP Baseline (3天)
- [ ] ViT + Clinical Fusion (2天)
- [ ] 代码测试和调试 (2天)

**Week 3-4**: 运行Baseline实验
- [ ] 每个Baseline运行5次（使用不同随机种子）
- [ ] 结果收集和分析
- [ ] 初步对比分析

#### 阶段2: 消融实验 (2-3周)

**Week 5-6**: 实现消融实验
- [ ] Knowledge Notes消融 (2天)
- [ ] Visual Notes消融 (2天)
- [ ] Sinkhorn OT消融 (2天)
- [ ] Dual-Head消融 (2天)
- [ ] 组合消融 (2天)

**Week 7**: 运行消融实验
- [ ] 每个消融实验运行5次
- [ ] 结果收集和分析

#### 阶段3: 性能提升 (2-3周)

**Week 8-9**: 性能优化
- [ ] 损失函数优化 (3天)
- [ ] 数据增强 (3天)
- [ ] 模型架构优化 (3天)
- [ ] 训练策略优化 (3天)

**Week 10**: 最终模型训练
- [ ] 完整方法训练（5次运行）
- [ ] 结果验证

#### 阶段4: 统计分析和论文撰写 (1-2周)

**Week 11**: 统计分析
- [ ] 统计显著性检验
- [ ] 结果表格生成
- [ ] 可视化图表生成

**Week 12**: 论文撰写
- [ ] 实验部分撰写
- [ ] 结果分析
- [ ] 图表整理

### 6.2 并行执行策略

**可以并行执行的任务**:
1. Baseline实验可以并行运行（如果有多个GPU）
2. 消融实验可以并行运行
3. 性能优化实验可以并行测试不同配置

**建议**:
- 使用多个GPU并行训练
- 使用实验管理工具（如Weights & Biases, TensorBoard）
- 自动化实验脚本

---

## 7. 结果分析模板

### 7.1 结果表格模板

**Table 1: Baseline对比结果**

| Method | AUC | Acc | Prec | Rec | Spec | F1 | Params (M) |
|--------|-----|-----|------|-----|------|----|-----------| 
| Simple Fusion | 0.7234±0.0123 | 0.6845±0.0156 | 0.6234±0.0189 | 0.7123±0.0145 | 0.6567±0.0167 | 0.6645±0.0145 | 45.2 |
| Standard CLIP | 0.7821±0.0098 | 0.7234±0.0123 | 0.6789±0.0156 | 0.7456±0.0134 | 0.7012±0.0145 | 0.7101±0.0123 | 125.6 |
| ViT+Clinical | 0.7567±0.0112 | 0.7012±0.0134 | 0.6456±0.0167 | 0.7234±0.0145 | 0.6789±0.0156 | 0.6823±0.0134 | 98.3 |
| CNN Baseline | 0.7234±0.0123 | 0.6845±0.0156 | 0.6234±0.0189 | 0.7123±0.0145 | 0.6567±0.0167 | 0.6645±0.0145 | 352.1 |
| Swin-T Baseline | 0.8377±0.0089 | 0.7821±0.0112 | 0.7234±0.0145 | 0.8012±0.0123 | 0.7634±0.0134 | 0.7601±0.0112 | 28.3 |
| VMamba Baseline | 0.8401±0.0087 | 0.7898±0.0109 | 0.7345±0.0142 | 0.8123±0.0121 | 0.7678±0.0132 | 0.7712±0.0110 | 37.2 |
| **Bio-COT 3.0** | **0.8523±0.0076*** | **0.8123±0.0098*** | **0.7654±0.0123*** | **0.8234±0.0109*** | **0.8012±0.0112*** | **0.7934±0.0098*** | **156.8** |

*表示与所有baseline相比p < 0.001

**Table 2: 消融实验结果**

| Ablation | Removed Module | AUC | vs Full Model | p-value |
|----------|---------------|-----|---------------|---------|
| Full Model | - | 0.8523±0.0076 | - | - |
| Ablation 1 | Knowledge Notes | 0.8012±0.0098 | -5.11% | < 0.001 |
| Ablation 2 | Visual Notes | 0.8234±0.0089 | -2.89% | < 0.001 |
| Ablation 3 | Sinkhorn OT | 0.8298±0.0087 | -2.25% | < 0.001 |
| Ablation 4 | Dual-Head | 0.8401±0.0085 | -1.22% | < 0.05 |
| Ablation 5 | Knowledge+Visual | 0.7789±0.0101 | -7.34% | < 0.001 |
| Ablation 6 | OT+Dual-Head | 0.8123±0.0092 | -4.00% | < 0.001 |

### 7.2 可视化图表

**Figure 2: Baseline对比ROC曲线**
- 所有baseline方法的ROC曲线
- 标注AUC值
- 使用不同颜色和线型

**Figure 3: 消融实验对比**
- 条形图显示各消融实验的AUC
- 标注与Full Model的差距
- 使用误差棒显示95%置信区间

**Figure 4: 统计显著性检验**
- 箱线图显示各方法的AUC分布
- 标注统计显著性（*表示p < 0.05, **表示p < 0.01, ***表示p < 0.001）

---

## 8. 代码框架

### 8.1 实验管理框架

**文件结构**:
```
experiments/exp_bio3.0_improved/
├── experiments/
│   ├── baseline/
│   │   ├── train_simple_fusion.py
│   │   ├── train_standard_clip.py
│   │   └── train_vit_clinical_fusion.py
│   ├── ablation/
│   │   ├── train_ablation_knowledge_notes.py
│   │   ├── train_ablation_visual_notes.py
│   │   ├── train_ablation_ot.py
│   │   └── train_ablation_dual_head.py
│   └── full_model/
│       └── train_full_model.py
├── utils/
│   ├── experiment_manager.py  # 实验管理
│   ├── statistics.py          # 统计检验
│   └── result_analyzer.py     # 结果分析
└── scripts/
    ├── run_all_baselines.sh   # 运行所有baseline
    ├── run_all_ablations.sh   # 运行所有消融实验
    └── analyze_results.py     # 结果分析脚本
```

### 8.2 实验管理代码框架

**experiment_manager.py**:
```python
class ExperimentManager:
    """实验管理器"""
    
    def __init__(self, experiment_name, config):
        self.experiment_name = experiment_name
        self.config = config
        self.results = []
        self.random_seeds = [42, 123, 456, 789, 2024]
    
    def run_experiment(self, num_runs=5):
        """运行实验（多次）"""
        for seed in self.random_seeds[:num_runs]:
            result = self._run_single(seed)
            self.results.append(result)
        return self.results
    
    def _run_single(self, seed):
        """运行单次实验"""
        # 设置随机种子
        set_seed(seed)
        
        # 训练模型
        model = create_model(self.config)
        trainer = Trainer(model, self.config)
        result = trainer.train()
        
        return result
    
    def analyze_results(self):
        """分析结果"""
        # 描述性统计
        stats = self._compute_statistics()
        
        # 统计检验
        if self.baseline_results:
            significance = self._test_significance(self.baseline_results)
        
        return stats, significance
```

### 8.3 统计检验代码框架

**statistics.py**:
```python
def compute_statistics(results):
    """计算描述性统计"""
    mean = np.mean(results)
    std = np.std(results)
    ci_95 = scipy.stats.t.interval(0.95, len(results)-1, loc=mean, scale=std/np.sqrt(len(results)))
    return {
        'mean': mean,
        'std': std,
        'ci_95': ci_95,
        'min': np.min(results),
        'max': np.max(results)
    }

def test_significance(results_a, results_b, test_type='ttest'):
    """统计显著性检验"""
    if test_type == 'ttest':
        statistic, p_value = scipy.stats.ttest_rel(results_a, results_b)
    elif test_type == 'wilcoxon':
        statistic, p_value = scipy.stats.wilcoxon(results_a, results_b)
    
    # 效应量
    cohens_d = (np.mean(results_a) - np.mean(results_b)) / np.std(results_b)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05
    }
```

---

## 9. 检查清单

### 9.1 实验完整性检查

**Baseline对比**:
- [ ] Simple Fusion Baseline (5次运行)
- [ ] Standard CLIP Baseline (5次运行)
- [ ] ViT + Clinical Fusion (5次运行)
- [ ] CNN Baseline (已有结果)
- [ ] Swin-T Baseline (已有结果)
- [ ] VMamba Baseline (已有结果)

**消融实验**:
- [ ] Knowledge Notes消融 (5次运行)
- [ ] Visual Notes消融 (5次运行)
- [ ] Sinkhorn OT消融 (5次运行)
- [ ] Dual-Head消融 (5次运行)
- [ ] 组合消融 (5次运行)

**统计检验**:
- [ ] 描述性统计（均值、标准差、95% CI）
- [ ] 统计显著性检验（t-test/Wilcoxon）
- [ ] 效应量计算（Cohen's d）

**性能提升**:
- [ ] AUC ≥ 85%
- [ ] Accuracy ≥ 80%
- [ ] F1-Score ≥ 70%

### 9.2 论文撰写检查

**实验部分**:
- [ ] 数据集描述完整
- [ ] 实验设置清晰
- [ ] Baseline方法描述完整
- [ ] 消融实验设计清晰
- [ ] 评估指标明确

**结果部分**:
- [ ] 结果表格完整
- [ ] 统计显著性标注
- [ ] 可视化图表清晰
- [ ] 结果分析深入

---

## 10. 总结

### 10.1 实验设计要点

1. **严谨性**: 所有实验使用固定随机种子，可复现
2. **完整性**: 覆盖所有必需的Baseline和消融实验
3. **统计性**: 所有结果包含统计显著性检验
4. **公平性**: 所有方法使用相同的数据划分和评估指标

### 10.2 预期成果

**完成所有实验后**:
- ✅ 完整的Baseline对比（6个方法）
- ✅ 完整的消融实验（6个实验）
- ✅ 统计显著性检验结果
- ✅ 性能达到MICCAI标准（AUC ≥ 85%）
- ✅ 符合MICCAI发表要求

**预计时间**: 8-12周 (2-3个月)

---

**文档版本**: v1.0  
**创建日期**: 2025-01-15  
**作者**: 深度学习实验设计专家

