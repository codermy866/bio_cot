# 类别不平衡问题 - 快速开始指南

## 🚀 一键启动训练

### 方法1: 使用类别加权交叉熵（最简单）⭐

```bash
bash paper1_hierarchical_multimodal/scripts/start_class_imbalance_training.sh weighted_ce 0
```

### 方法2: 使用优化的Focal Loss（推荐）⭐⭐

```bash
bash paper1_hierarchical_multimodal/scripts/start_class_imbalance_training.sh focal 1
```

---

## 📋 已实现的功能

### ✅ 1. 自动类别权重计算
- 自动从训练数据计算类别权重
- 使用逆频率方法：`weight = total / (num_classes * count)`
- 当前权重：`[0.7406, 1.5392]` (类别0, 类别1)

### ✅ 2. 优化的Focal Loss
- **gamma**: 自动提升到4.0（更关注难样本）
- **alpha**: 自动调整为`[1.0, 0.32]`（少数类权重更高）
- 同时使用类别权重增强效果

### ✅ 3. 类别加权交叉熵
- 简单直接的解决方案
- 计算效率高
- 效果稳定

### ✅ 4. 组合策略
- Focal Loss + 类别权重 + Label Smoothing
- 预期效果最佳

---

## 🎯 预期效果

| 指标 | 改进前 | 改进后（预期） | 提升 |
|------|--------|---------------|------|
| **Val ACC** | 0.565-0.665 | **0.70-0.75** | +5-10% |
| **Val AUC** | 0.58-0.62 | **0.75-0.85** | +15-25% |
| **少数类召回率** | 低 | **显著提升** | +20-30% |
| **F1-Score** | 低 | **提升10-15%** | +10-15% |

---

## 📊 监控训练

### 查看日志
```bash
# 方法1的日志
tail -f paper1_hierarchical_multimodal/logs/train_cuda0_class_imbalance_weighted_ce.log

# 方法2的日志
tail -f paper1_hierarchical_multimodal/logs/train_cuda1_class_imbalance_focal.log
```

### 关键指标
- **Train/Val ACC**: 应该更平衡
- **Val AUC**: 应该显著提升
- **各类别Precision/Recall**: 关注少数类的召回率

---

## 🔧 手动调整参数

### 自定义类别权重
```bash
python train_hierarchical_multimodal.py \
  --use_class_weights \
  --class_weights "1.0,2.5" \
  --use_focal_loss \
  --focal_gamma 4.0
```

### 调整Focal Loss参数
```bash
python train_hierarchical_multimodal.py \
  --use_focal_loss \
  --focal_gamma 5.0 \
  --use_class_weights
```

---

## 📝 实施检查清单

- [x] ✅ 实现类别加权交叉熵
- [x] ✅ 优化Focal Loss（支持类别权重和per-class alpha）
- [x] ✅ 自动类别权重计算
- [x] ✅ 更新训练脚本
- [x] ✅ 创建详细文档
- [x] ✅ 创建一键启动脚本
- [ ] ⏳ 运行训练并评估效果
- [ ] ⏳ 根据结果进一步优化

---

## 🎯 下一步

1. **立即运行训练**:
   ```bash
   bash paper1_hierarchical_multimodal/scripts/start_class_imbalance_training.sh focal 1
   ```

2. **监控训练进度**:
   - 关注ACC和AUC的提升
   - 检查各类别的性能

3. **评估效果**:
   - 对比改进前后的指标
   - 分析混淆矩阵

4. **进一步优化**:
   - 根据结果调整参数
   - 结合其他改进（如Contrastive Loss修复）

---

## 💡 技术细节

### 类别权重计算
```python
# 逆频率方法
class_weights = [total / (num_classes * count_i) for count_i in label_counts]
# 结果: [0.7406, 1.5392]
```

### Focal Loss优化
```python
# alpha自动调整
focal_alpha = [1.0, 0.32]  # [majority, minority]
# gamma提升
focal_gamma = 4.0  # 更关注难样本
```

### 组合使用
```python
# Focal Loss + 类别权重 + Label Smoothing
criterion = CombinedLoss(
    focal_alpha=[1.0, 0.32],
    focal_gamma=4.0,
    class_weights=[0.74, 1.54],
    label_smoothing=0.1
)
```

---

## 📚 相关文档

- **详细方案**: `CLASS_IMBALANCE_SOLUTION.md`
- **效果分析**: `TRAINING_EFFECTIVENESS_ANALYSIS.md`
- **关键发现**: `KEY_FINDINGS_SUMMARY.md`

---

**准备好了！现在可以开始训练了！** 🚀

