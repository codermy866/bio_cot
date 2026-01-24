# 类别不平衡问题解决方案

## 📊 问题分析

### 当前数据分布
- **训练集**: 类别0 (530) vs 类别1 (255) = **2.08:1**
- **测试集**: 类别0 (133) vs 类别1 (67) = **1.99:1**
- **不平衡比例**: 约2:1，属于中等程度不平衡

### 影响
- ❌ 模型严重偏向多数类（类别0）
- ❌ 少数类（类别1）召回率低
- ❌ 当前ACC和AUC不高的主要原因之一
- ❌ AUC可能虚高但实际性能差

---

## 🎯 解决方案（已实现）

### 方案1: 类别加权交叉熵（最简单有效）⭐推荐

**原理**: 直接给少数类更高的权重

**实现**:
```python
# 自动计算类别权重（逆频率）
class_weights = [0.7406, 1.5392]  # [class_0, class_1]
criterion = ClassWeightedCrossEntropy(class_weights=class_weights)
```

**优点**:
- ✅ 实现简单
- ✅ 计算效率高
- ✅ 效果稳定
- ✅ 不需要调参

**使用方法**:
```bash
python train_hierarchical_multimodal.py \
  --use_weighted_ce \
  --use_class_weights
```

---

### 方案2: Focal Loss（更关注难样本）⭐推荐

**原理**: 自动关注难分类样本，对类别不平衡更鲁棒

**优化参数**:
- **gamma**: 2.0 → **4.0** (更关注难样本)
- **alpha**: [1.0, 0.32] (少数类权重更高)
- **class_weights**: 同时使用类别权重

**实现**:
```python
criterion = CombinedLoss(
    focal_alpha=[1.0, 0.32],  # [majority, minority]
    focal_gamma=4.0,          # 更关注难样本
    class_weights=class_weights,
    focal_weight=0.7,
    smoothing_weight=0.3
)
```

**优点**:
- ✅ 自动关注难样本
- ✅ 对类别不平衡更鲁棒
- ✅ 结合Label Smoothing提升泛化

**使用方法**:
```bash
python train_hierarchical_multimodal.py \
  --use_focal_loss \
  --focal_gamma 4.0 \
  --use_class_weights
```

---

### 方案3: 组合方案（最佳效果）⭐⭐⭐

**同时使用**:
1. 类别权重
2. Focal Loss (gamma=4.0)
3. Label Smoothing

**预期效果**: 提升5-8% ACC, 3-5% AUC

---

## 🚀 快速开始

### 方法1: 使用类别加权交叉熵（推荐新手）

```bash
cd /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713
source my_retfound/bin/activate

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python -u \
  paper1_hierarchical_multimodal/training/train_hierarchical_multimodal.py \
  --data_path 5centers_multi \
  --output_dir paper1_hierarchical_multimodal/results/cuda0_class_weighted \
  --batch_size 8 \
  --num_epochs 30 \
  --num_workers 4 \
  --device cuda \
  --log_interval 10 \
  --contrastive_weight 0.3 \
  --learning_rate 5e-5 \
  --weight_decay 1e-4 \
  --warmup_epochs 3 \
  --use_amp \
  --use_weighted_ce \
  --use_class_weights \
  --label_smoothing 0.1 \
  --input_size 224 \
  --oct_num_frames 32 \
  --oct_frames_per_point 5 \
  --max_grad_norm 0.5 \
  > paper1_hierarchical_multimodal/logs/train_cuda0_class_weighted.log 2>&1 &
```

### 方法2: 使用优化的Focal Loss（推荐）

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 nohup python -u \
  paper1_hierarchical_multimodal/training/train_hierarchical_multimodal.py \
  --data_path 5centers_multi \
  --output_dir paper1_hierarchical_multimodal/results/cuda1_focal_optimized \
  --batch_size 4 \
  --num_epochs 30 \
  --num_workers 4 \
  --device cuda \
  --log_interval 10 \
  --contrastive_weight 0.3 \
  --learning_rate 5e-5 \
  --weight_decay 1e-4 \
  --warmup_epochs 3 \
  --use_amp \
  --use_focal_loss \
  --focal_gamma 4.0 \
  --use_class_weights \
  --label_smoothing 0.1 \
  --input_size 224 \
  --oct_num_frames 32 \
  --oct_frames_per_point 5 \
  --max_grad_norm 0.5 \
  --use_vit_backbone \
  --backbone_type vit \
  --vit_model_name vit_base_patch16_224 \
  > paper1_hierarchical_multimodal/logs/train_cuda1_focal_optimized.log 2>&1 &
```

### 方法3: 手动指定类别权重

```bash
# 如果自动计算的权重不满意，可以手动指定
python train_hierarchical_multimodal.py \
  --use_class_weights \
  --class_weights "1.0,2.08" \
  --use_focal_loss \
  --focal_gamma 4.0
```

---

## 📈 预期效果

### 改进前
- Val ACC: 0.565-0.665
- Val AUC: 0.58-0.62
- 少数类召回率: 低

### 改进后（预期）
- Val ACC: **0.70-0.75** (+5-10%)
- Val AUC: **0.75-0.85** (+15-25%)
- 少数类召回率: **显著提升**
- F1-score: **提升10-15%**

---

## 🔍 监控指标

### 关键指标

1. **平衡准确率 (Balanced Accuracy)**
   ```python
   balanced_acc = (sensitivity + specificity) / 2
   ```

2. **F1-Score**
   ```python
   f1 = 2 * (precision * recall) / (precision + recall)
   ```

3. **Precision-Recall AUC**
   - 对类别不平衡更敏感
   - 比ROC AUC更能反映真实性能

4. **混淆矩阵**
   - 查看各类别的分类情况
   - 重点关注少数类的召回率

### 训练过程监控

```bash
# 查看训练日志
tail -f paper1_hierarchical_multimodal/logs/train_cuda0_class_weighted.log

# 关注这些指标：
# - Train/Val ACC
# - Val AUC
# - 各类别的Precision和Recall
```

---

## 🎛️ 参数调优建议

### Focal Loss参数

| 参数 | 当前值 | 推荐值 | 说明 |
|------|--------|--------|------|
| **gamma** | 2.0 | **4.0** | 更关注难样本 |
| **alpha** | 1.0 | **[1.0, 0.32]** | 少数类权重 |
| **class_weights** | None | **[0.74, 1.54]** | 类别权重 |

### 学习率调整

- **Backbone**: 1e-5 (微调)
- **新层**: 1e-4 (从头训练)
- 使用分层学习率效果更好

### Batch Size

- **ResNet**: 8-16
- **ViT**: 4-8 (显存限制)
- 使用Gradient Accumulation模拟大batch

---

## 📊 对比实验建议

### 实验1: 基线对比
- 无类别权重 vs 有类别权重
- 标准CE vs Focal Loss vs 加权CE

### 实验2: Focal Loss参数
- gamma: 2.0, 3.0, 4.0, 5.0
- alpha: [1.0, 1.0], [1.0, 0.5], [1.0, 0.32]

### 实验3: 组合策略
- 仅类别权重
- 仅Focal Loss
- 类别权重 + Focal Loss

---

## ⚠️ 注意事项

1. **不要过度调整**
   - 类别权重过大可能导致过拟合
   - gamma过大可能导致训练不稳定

2. **验证集评估**
   - 重点关注验证集上的平衡指标
   - 不要只看ACC，要看各类别的性能

3. **数据质量**
   - 确保标注质量
   - 检查是否有标注错误

4. **与其他改进结合**
   - 类别不平衡 + Contrastive Loss修复
   - 类别不平衡 + 更好的数据增强

---

## 🔄 后续优化方向

1. **数据层面**
   - SMOTE过采样
   - 对少数类进行更强的数据增强

2. **模型层面**
   - 集成多个模型
   - 使用Cost-Sensitive Learning

3. **评估层面**
   - 使用Cost Matrix
   - 关注临床相关指标

---

## 📝 实施检查清单

- [x] 实现类别加权交叉熵
- [x] 优化Focal Loss参数
- [x] 添加自动类别权重计算
- [x] 更新训练脚本
- [x] 创建详细文档
- [ ] 运行对比实验
- [ ] 评估效果
- [ ] 根据结果调整参数

---

## 🎯 总结

**核心策略**:
1. ✅ 使用类别权重（最简单有效）
2. ✅ 优化Focal Loss参数（gamma=4.0, alpha=[1.0, 0.32]）
3. ✅ 结合Label Smoothing提升泛化

**预期提升**: 
- ACC: +5-10%
- AUC: +15-25%
- 少数类召回率: 显著提升

**下一步**: 运行训练，监控效果，根据结果进一步优化！

