# 数据集划分总结

## 📋 划分策略

根据您的要求，数据集已按照医疗中心划分为：

### 内部开发集（Internal Development Set）
**用于训练和验证**

- **中心**: 恩施、襄阳、十堰
- **训练集**: 660个样本
- **验证集**: 166个样本
- **总计**: 826个样本

### 外部独立测试集（External Test Set）
**严禁参与训练，仅用于最终评估**

- **中心**: 荆州、武大
- **样本数**: 159个样本

---

## 📊 详细统计

### 各中心样本分布

| 中心 | 总样本数 | 阳性数 | 阴性数 | 阳性率 | 用途 |
|------|----------|--------|--------|--------|------|
| **十堰** | 78 | 26 | 52 | 33.33% | 内部开发集 |
| **恩施** | 404 | 131 | 273 | 32.43% | 内部开发集 |
| **襄阳** | 344 | 53 | 291 | 15.41% | 内部开发集 |
| **内部合计** | **826** | **210** | **616** | **25.42%** | **训练/验证** |
| **荆州** | 70 | 23 | 47 | 32.86% | 外部测试集 |
| **武大** | 89 | 89 | 0 | 100.00% | 外部测试集 |
| **外部合计** | **159** | **112** | **47** | **70.44%** | **最终测试** |

### 划分结果详情

#### 训练集 (`train_labels.csv`)
- **样本数**: 660
- **阳性**: 168 (25.45%)
- **阴性**: 492 (74.55%)
- **中心分布**:
  - 恩施: 322
  - 襄阳: 277
  - 十堰: 61

#### 验证集 (`val_labels.csv`)
- **样本数**: 166
- **阳性**: 42 (25.30%)
- **阴性**: 124 (74.70%)
- **中心分布**:
  - 恩施: 82
  - 襄阳: 67
  - 十堰: 17

#### 外部测试集 (`external_test_labels.csv`)
- **样本数**: 159
- **阳性**: 112 (70.44%)
- **阴性**: 47 (29.56%)
- **中心分布**:
  - 武大: 89
  - 荆州: 70

---

## ✅ 数据检查结果

所有数据划分已通过检查：

- ✅ **训练集**: 仅包含内部开发集样本（恩施、襄阳、十堰）
- ✅ **验证集**: 仅包含内部开发集样本（恩施、襄阳、十堰）
- ✅ **外部测试集**: 仅包含外部测试集样本（荆州、武大）

**无数据泄露风险！**

---

## 📁 文件位置

所有划分后的文件保存在：

```
5centers_multi_internal_external/
├── train_labels.csv              # 内部训练集（660样本）
├── val_labels.csv                # 内部验证集（166样本）
├── external_test_labels.csv      # 外部测试集（159样本）
├── split_statistics.json         # 统计信息
├── split_report.md               # 详细报告
└── README.md                     # 使用说明
```

---

## 🔧 使用方法

### 1. 在训练代码中使用

```python
import pandas as pd
from pathlib import Path

# 数据路径
data_root = Path('5centers_multi_internal_external')

# 加载训练集
train_df = pd.read_csv(data_root / 'train_labels.csv')

# 加载验证集
val_df = pd.read_csv(data_root / 'val_labels.csv')

# ⚠️ 外部测试集仅在最终评估时使用
# external_test_df = pd.read_csv(data_root / 'external_test_labels.csv')
```

### 2. 修改训练脚本

在 `train_enhanced_causal_clip.py` 中修改数据路径：

```python
# 修改前
train_loader, val_loader = prepare_data_loaders(
    data_path='5centers_multi',
    ...
)

# 修改后
train_loader, val_loader = prepare_data_loaders(
    data_path='5centers_multi_internal_external',
    train_labels_file='train_labels.csv',
    val_labels_file='val_labels.csv',
    ...
)
```

### 3. 添加数据检查

在训练前添加检查：

```python
# 在训练脚本开头添加
from exp1_Causal_Bayesian_clip.code.check_data_split import DataSplitChecker

checker = DataSplitChecker()
results = checker.check_all_splits(Path('5centers_multi_internal_external'))
checker.print_check_results(results)
```

---

## ⚠️ 重要注意事项

### 1. 外部测试集隔离

- **严禁**在训练过程中使用外部测试集
- **严禁**用于超参数调优
- **严禁**用于模型选择
- **严禁**用于早停策略
- **仅用于**最终模型性能评估

### 2. 数据特点

- **武大中心**: 阳性率100%，这可能影响外部测试集的代表性
- **建议**: 在论文中分别报告荆州和武大的结果
- **注意**: 外部测试集阳性率较高（70.44%），与训练集（25.45%）存在分布差异

### 3. 可复现性

- 划分脚本使用固定随机种子（random_state=42）
- 验证集比例：20%（从内部开发集中划分）
- 使用分层采样保持类别平衡

---

## 🔄 重新划分

如果需要重新划分数据集：

```bash
python exp1_Causal_Bayesian_clip/code/split_dataset_by_centers.py \
    --data_path 5centers_multi \
    --output_dir 5centers_multi_internal_external \
    --val_ratio 0.2 \
    --random_state 42
```

---

## 📝 论文中如何描述

建议在论文的Methods部分这样描述：

> "数据集按照医疗中心划分为内部开发集和外部测试集。内部开发集包含来自恩施、襄阳、十堰三个中心的826个样本（训练集660个，验证集166个），用于模型训练、超参数调优和模型选择。外部测试集包含来自荆州、武大两个中心的159个样本，完全独立于训练过程，仅用于最终性能评估，以验证模型的泛化能力。"

---

## 🎯 下一步工作

1. **修改训练脚本**: 更新数据加载路径
2. **添加数据检查**: 在训练前验证数据划分
3. **开始训练**: 使用新的数据划分进行训练
4. **最终评估**: 训练完成后，在外部测试集上评估模型

---

**创建时间**: 2025-12-17

**版本**: v1.0

