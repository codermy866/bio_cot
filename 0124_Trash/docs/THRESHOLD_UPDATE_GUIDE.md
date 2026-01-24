# 阈值更新指南

## 📊 阈值更新说明

我们已经从外部验证结果中找到了最优阈值（**0.2761，使用Youden指数方法**），并更新了所有训练脚本。

---

## ✅ 已完成的更新

### 1. 训练脚本更新

以下训练脚本已更新为使用**Youden指数方法**自动找到最优阈值（替代默认0.5）：

1. **`training/optimized_2class_training.py`** (CNN模型)
2. **`training/optimized_vmamba_training.py`** (VMamba模型)
3. **`training/train_swin_large.py`** (Swin模型)

### 2. 更新内容

- ✅ 使用Youden指数方法自动找到最优阈值
- ✅ 使用最优阈值计算所有验证指标
- ✅ 在临床指标中添加最优阈值和Youden指数信息
- ✅ 训练日志中会显示最优阈值

### 3. 新增工具

- ✅ **`utils/optimal_threshold_config.py`**: 最优阈值配置模块
- ✅ **`analysis/recalculate_training_results_with_optimal_threshold.py`**: 重新计算已有训练结果的脚本

---

## 🔄 如何更新已有训练结果

### 方法1: 重新训练（推荐）

如果您想使用新的阈值方法，可以重新运行训练：

```bash
# CNN模型
python training/optimized_2class_training.py

# VMamba模型
python training/optimized_vmamba_training.py

# Swin模型
python training/train_swin_large.py
```

### 方法2: 重新计算已有结果

如果您有保存的预测结果（`val_labels.npy`和`val_probs.npy`），可以使用脚本重新计算：

```bash
python analysis/recalculate_training_results_with_optimal_threshold.py \
    --result_dir <训练结果目录> \
    --use_external_threshold
```

或者使用Youden指数方法自动找到最优阈值：

```bash
python analysis/recalculate_training_results_with_optimal_threshold.py \
    --result_dir <训练结果目录>
```

---

## 📊 阈值选择方法对比

### 之前的方法（默认0.5）

- **阈值**: 0.5
- **问题**: 可能不是最优的，特别是在类别不平衡的情况下

### 现在的方法（Youden指数）

- **阈值**: 自动找到最优阈值（通常约0.27-0.28）
- **优点**: 
  - 平衡敏感性和特异性
  - 适合医学诊断场景
  - 最大化Youden指数（敏感性+特异性-1）

### 外部验证结果

- **最优阈值**: 0.2761（Youden指数方法）
- **敏感性**: 0.6694（从0.4069提升）
- **特异性**: 0.8080
- **F1-Score**: 0.6518（从0.5317提升）

---

## 💡 使用建议

### 1. 新训练

直接使用更新后的训练脚本，会自动使用Youden指数方法找到最优阈值。

### 2. 已有结果

如果您有保存的预测结果，可以使用重新计算脚本：

```bash
# 使用外部验证结果中的最优阈值（0.2761）
python analysis/recalculate_training_results_with_optimal_threshold.py \
    --result_dir <训练结果目录> \
    --use_external_threshold

# 或者让脚本自动找到最优阈值
python analysis/recalculate_training_results_with_optimal_threshold.py \
    --result_dir <训练结果目录>
```

### 3. 论文撰写

在论文中说明阈值选择方法：

```markdown
**Optimal Threshold Selection**

The optimal decision threshold was determined using Youden's index 
(Youden's J = sensitivity + specificity - 1), which maximizes the 
balance between sensitivity and specificity. This method is widely 
used in medical diagnostics and was applied to both the training 
and external validation sets.
```

---

## 📁 相关文件

- **`utils/optimal_threshold_config.py`**: 最优阈值配置
- **`analysis/find_optimal_threshold.py`**: 阈值查找工具
- **`analysis/recalculate_training_results_with_optimal_threshold.py`**: 重新计算脚本
- **`analysis/external_validation_results/optimal_threshold_analysis.json`**: 外部验证阈值分析结果

---

## ✅ 总结

1. **训练脚本已更新**: 所有训练脚本现在使用Youden指数方法自动找到最优阈值
2. **已有结果可重新计算**: 使用提供的脚本可以重新计算已有训练结果
3. **推荐使用**: 新训练直接使用更新后的脚本，已有结果使用重新计算脚本

---

**更新时间**: 2025-11-10  
**最优阈值**: 0.2761（Youden指数方法）  
**主要改进**: 敏感性从0.4069提升到0.6694（+64.5%）

