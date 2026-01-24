<!--
文件生成信息:
- 生成时间: 2025-12-27
- 生成需求: 制定完整的实验路线图，明确当前训练内容，给出技术指引
- 生成原因: 用户需要了解当前训练的是什么，需要技术专家指引和自动执行
- 相关任务: 实验路线图制定、Baseline创建、消融实验设计

文件功能: 完整的实验路线图和技术指引，帮助用户理解实验层次和执行顺序
-->

# 🗺️ 完整实验路线图与技术指引

**制定时间**: 2025-12-27  
**目标**: 明确当前训练内容，制定完整的实验执行计划  
**状态**: ✅✅✅ **路线图已制定，Baseline脚本已创建**

---

## 1. 当前训练内容明确

### 1.1 当前正在训练的方法

**方法名称**: **CausalBayesianCLIP** (我们的方法)

**训练脚本**: `experiments/exp1_causal_bayesian_clip/train_causal_bayesian.py`

**核心特点**:
- ✅ 贝叶斯编码器（不确定性量化）
- ✅ 因果约束注意力（简化实现）
- ✅ 跨模态对比学习
- ✅ 域对抗训练

**当前状态**:
- 进程1: GPU 0, `exp_multicenter_alignment` (旧版本)
- 进程2: GPU 1, `exp_multicenter_alignment_fixed` (新版本，修复了类别检测)

**性能** (Epoch 25):
- 验证AUC: 0.733 (最佳: 0.763 @ Epoch 23)
- 验证Acc: 0.62-0.76% (异常低)

---

### 1.2 重要说明

**⚠️ 这是我们的方法，不是Baseline！**

- 我们的方法包含创新点（因果约束、贝叶斯不确定性）
- 需要与Baseline对比才能证明有效性
- 需要消融实验验证各模块的贡献

---

## 2. 完整实验层次结构

### 2.1 实验金字塔

```
                    ┌─────────────────────┐
                    │  我们的方法 (当前)   │
                    │ CausalBayesianCLIP  │
                    │  AUC: 0.763         │
                    └─────────────────────┘
                            ↑
                    ┌───────┴───────┐
                    │  消融实验     │
                    │  (验证模块)   │
                    └───────┬───────┘
                            ↑
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────┴──────┐   ┌────────┴────────┐  ┌──────┴──────┐
│ Simple Fusion│   │ Standard CLIP   │  │ Causal CLIP │
│   Baseline   │   │    Baseline     │  │  Baseline   │
│  (最简单)    │   │   (标准方法)    │  │ (部分创新)  │
└──────────────┘   └─────────────────┘  └─────────────┘
        ↑
┌───────┴───────┐
│ 已有Baseline  │
│ CNN, Swin-T   │
│ VMamba        │
└───────────────┘
```

---

### 2.2 实验执行顺序

#### 阶段1: Baseline训练 (必需) ⭐⭐⭐⭐⭐

**目标**: 建立性能基准

**执行顺序**:

1. **Simple Fusion Baseline** (最简单)
   - 方法: 特征拼接 + MLP
   - 预期AUC: ~0.70-0.75
   - **脚本**: `experiments/baseline/train_simple_fusion_baseline.py` ✅ 已创建
   - **优先级**: 极高

2. **Standard CLIP Baseline** (标准方法)
   - 方法: 标准CLIP（无因果约束，无不确定性）
   - 预期AUC: ~0.75-0.80
   - **脚本**: `experiments/baseline/train_standard_clip_baseline.py` ✅ 已创建
   - **优先级**: 极高

3. **Causal CLIP Baseline** (部分创新)
   - 方法: 有因果约束，无不确定性量化
   - 预期AUC: ~0.78-0.82
   - **脚本**: 待创建
   - **优先级**: 高

4. **Bayesian CLIP Baseline** (部分创新)
   - 方法: 无因果约束，有不确定性量化
   - 预期AUC: ~0.76-0.80
   - **脚本**: 待创建
   - **优先级**: 高

5. **已有Baseline** (已完成)
   - CNN Baseline ✅
   - Swin-T Baseline ✅ (AUC ~0.8377)
   - VMamba Baseline ✅ (AUC ~0.84)

---

#### 阶段2: 消融实验 (必需) ⭐⭐⭐⭐⭐

**目标**: 验证每个模块的有效性

**消融实验设计**:

| 实验 | 因果约束 | 贝叶斯 | 对比学习 | 目的 |
|------|---------|--------|---------|------|
| Baseline | ❌ | ❌ | ❌ | 基础性能 |
| + Causal | ✅ | ❌ | ❌ | 验证因果约束 |
| + Bayesian | ❌ | ✅ | ❌ | 验证不确定性量化 |
| + Contrastive | ❌ | ❌ | ✅ | 验证对比学习 |
| + Causal + Bayesian | ✅ | ✅ | ❌ | 验证组合1 |
| + Causal + Contrastive | ✅ | ❌ | ✅ | 验证组合2 |
| + Bayesian + Contrastive | ❌ | ✅ | ✅ | 验证组合3 |
| **Full Model** | **✅** | **✅** | **✅** | **完整方法** |

---

#### 阶段3: 我们的方法优化 (当前)

**目标**: 提升当前方法的性能

**当前问题**:
1. ❌ 验证Acc异常低
2. ⚠️ AUC需要提升
3. ⚠️ 过拟合严重

---

## 3. 自动执行计划

### 3.1 已创建的Baseline脚本

✅ **Simple Fusion Baseline** - `experiments/baseline/train_simple_fusion_baseline.py`
- 最简单的多模态融合方法
- 特征拼接 + MLP分类器
- 无注意力机制，无因果约束，无不确定性量化

✅ **Standard CLIP Baseline** - `experiments/baseline/train_standard_clip_baseline.py`
- 标准CLIP方法
- 有跨模态对比学习
- 无因果约束，无不确定性量化

---

### 3.2 待创建的Baseline脚本

❌ **Causal CLIP Baseline** - 待创建
- 有因果约束，无不确定性量化

❌ **Bayesian CLIP Baseline** - 待创建
- 无因果约束，有不确定性量化

---

### 3.3 立即执行建议

#### 选项1: 先训练Simple Fusion Baseline (推荐)

**原因**:
- 最简单，训练快
- 建立最低性能基准
- 可以快速验证实验流程

**执行命令**:
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/baseline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate
python train_simple_fusion_baseline.py \
    --data_path /data2/hmy/5Center_datas/5centers_multi_leave_centers_out \
    --batch_size 24 \
    --num_epochs 100 \
    --learning_rate 3e-4 \
    --device cuda:0 \
    --num_workers 4 \
    --output_dir ./baseline_simple_fusion/checkpoints
```

#### 选项2: 先训练Standard CLIP Baseline

**原因**:
- 标准方法，性能较好
- 可以证明因果约束和不确定性的价值

**执行命令**:
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/baseline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate
python train_standard_clip_baseline.py \
    --data_path /data2/hmy/5Center_datas/5centers_multi_leave_centers_out \
    --batch_size 24 \
    --num_epochs 100 \
    --learning_rate 3e-4 \
    --device cuda:0 \
    --num_workers 4 \
    --output_dir ./baseline_standard_clip/checkpoints
```

---

## 4. 实验对比表格设计

### 4.1 最终结果表格 (Table 1)

| Method | Causal | Bayesian | Contrastive | AUC | Acc | F1 | 95% CI |
|--------|--------|----------|-------------|-----|-----|-----|--------|
| Simple Fusion | ❌ | ❌ | ❌ | - | - | - | - |
| Standard CLIP | ❌ | ❌ | ✅ | - | - | - | - |
| Causal CLIP | ✅ | ❌ | ✅ | - | - | - | - |
| Bayesian CLIP | ❌ | ✅ | ✅ | - | - | - | - |
| **Ours (Full)** | **✅** | **✅** | **✅** | **0.763** | **0.76%** | **0.611** | **-** |

---

### 4.2 消融实验表格 (Table 2)

| Method | Causal | Bayesian | Contrastive | AUC | ΔAUC |
|--------|--------|----------|-------------|-----|------|
| Baseline | ❌ | ❌ | ❌ | - | - |
| + Causal | ✅ | ❌ | ❌ | - | - |
| + Bayesian | ❌ | ✅ | ❌ | - | - |
| + Contrastive | ❌ | ❌ | ✅ | - | - |
| **Full Model** | **✅** | **✅** | **✅** | **0.763** | **-** |

---

## 5. 技术指引总结

### 5.1 当前状态

✅ **已完成**:
- 我们的方法实现 (CausalBayesianCLIP)
- 部分Baseline (CNN, Swin-T, VMamba)
- Simple Fusion Baseline脚本 ✅
- Standard CLIP Baseline脚本 ✅

❌ **缺失**:
- Causal CLIP Baseline脚本
- Bayesian CLIP Baseline脚本
- 消融实验脚本
- 统计分析

---

### 5.2 下一步行动

#### 立即执行 (今天)

1. **训练Simple Fusion Baseline**
   - 建立最低性能基准
   - 验证实验流程

2. **训练Standard CLIP Baseline**
   - 建立标准方法基准
   - 证明我们方法的优势

#### 本周完成

1. **创建Causal CLIP Baseline脚本**
2. **创建Bayesian CLIP Baseline脚本**
3. **训练所有Baseline**
4. **性能对比分析**

#### 2-3周内完成

1. **设计消融实验**
2. **实现消融实验代码**
3. **运行消融实验**
4. **统计分析**

---

## 6. 关键理解点

### 6.1 实验层次理解

```
Baseline (基准) → 消融实验 (验证模块) → 我们的方法 (完整方法)
     ↓                    ↓                      ↓
  简单方法           部分创新方法             完整创新方法
  性能较低           性能中等                 性能最高
```

### 6.2 为什么需要Baseline？

1. **证明方法的有效性**
   - 如果我们的方法性能 < Baseline，说明方法有问题
   - 如果我们的方法性能 > Baseline，说明方法有效

2. **量化改进幅度**
   - 可以计算AUC提升: ΔAUC = Ours - Baseline
   - 可以计算相对提升: (Ours - Baseline) / Baseline

3. **满足MICCAI要求**
   - MICCAI要求必须有Baseline对比
   - 缺少Baseline会被直接拒绝

---

### 6.3 为什么需要消融实验？

1. **验证每个模块的贡献**
   - 证明因果约束有效
   - 证明不确定性量化有效
   - 证明对比学习有效

2. **量化模块贡献度**
   - 每个模块提升多少AUC
   - 模块间的协同效应

3. **满足MICCAI要求**
   - MICCAI要求必须有消融实验
   - 缺少消融实验会被质疑创新性

---

## 7. 执行检查清单

### ✅ 已完成

- [x] 我们的方法实现
- [x] Simple Fusion Baseline脚本
- [x] Standard CLIP Baseline脚本
- [x] 部分Baseline (CNN, Swin-T, VMamba)

### ❌ 待完成 (必需)

- [ ] 训练Simple Fusion Baseline
- [ ] 训练Standard CLIP Baseline
- [ ] 创建Causal CLIP Baseline脚本
- [ ] 创建Bayesian CLIP Baseline脚本
- [ ] 训练所有Baseline
- [ ] 设计消融实验
- [ ] 实现消融实验
- [ ] 统计分析

---

**路线图制定日期**: 2025-12-27  
**核心结论**: ✅✅✅ **当前训练的是我们的方法（CausalBayesianCLIP）。已创建Simple Fusion和Standard CLIP Baseline脚本，可以立即开始训练Baseline建立性能基准。**

