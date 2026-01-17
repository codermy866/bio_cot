# 数据集划分对比分析

## 📊 两个数据集划分的对比

### 划分1: `5centers_multi_internal_external` ✅ **推荐使用**

#### 划分策略
- **内部开发集（训练+验证）**: 恩施、襄阳、十堰
- **外部测试集**: 荆州、武大

#### 统计信息
```json
{
  "train_samples": 660,
  "val_samples": 166,
  "external_test_samples": 159,
  "train_positive": 168 (25.45%),
  "train_negative": 492 (74.55%),
  "val_positive": 42 (25.30%),
  "val_negative": 124 (74.70%),
  "external_test_positive": 112 (70.44%),
  "external_test_negative": 47 (29.56%)
}
```

#### 特点
- ✅ **符合您的要求**: 内部=恩施+襄阳+十堰，外部=荆州+武大
- ✅ **有完整文档**: README.md, split_report.md, split_statistics.json
- ✅ **有可视化**: 包含可视化图表（PNG和PDF）
- ✅ **数据检查**: 有数据检查脚本
- ✅ **结构清晰**: train_labels.csv, val_labels.csv, external_test_labels.csv

#### 文件结构
```
5centers_multi_internal_external/
├── train_labels.csv              # 660样本（内部训练集）
├── val_labels.csv                # 166样本（内部验证集）
├── external_test_labels.csv      # 159样本（外部测试集）
├── README.md                     # 详细说明
├── split_report.md               # 划分报告
├── split_statistics.json         # 统计信息
├── visualizations/               # 可视化图表
└── internal_train/               # 内部训练数据
└── external_validation/          # 外部测试数据
```

---

### 划分2: `5centers_multi_internal_external_recommended` ❌ **不推荐**

#### 划分策略
- **内部训练集**: 恩施、武大、襄阳
- **外部验证集**: 十堰、荆州

#### 统计信息
```json
{
  "internal_train": {
    "total_samples": 837,
    "positive": 273,
    "negative": 564,
    "centers": ["恩施", "武大", "襄阳"]
  },
  "external_validation": {
    "total_samples": 148,
    "positive": 49,
    "negative": 99,
    "centers": ["十堰", "荆州"]
  }
}
```

#### 特点
- ❌ **不符合您的要求**: 武大在内部，十堰在外部（与您的要求相反）
- ❌ **缺少文档**: 没有README说明
- ❌ **结构不完整**: 只有train_labels.csv和test_labels.csv
- ❌ **缺少验证集**: 没有单独的验证集划分

#### 文件结构
```
5centers_multi_internal_external_recommended/
├── train_labels.csv              # 837样本
├── test_labels.csv                # 148样本（外部）
├── split_statistics.json         # 统计信息（简单）
└── internal_train/                # 内部训练数据
└── external_validation/          # 外部测试数据
```

---

## 🔍 关键差异对比

| 特征 | 划分1 (internal_external) | 划分2 (recommended) |
|------|---------------------------|---------------------|
| **内部中心** | 恩施、襄阳、十堰 ✅ | 恩施、武大、襄阳 ❌ |
| **外部中心** | 荆州、武大 ✅ | 十堰、荆州 ❌ |
| **是否符合要求** | ✅ 完全符合 | ❌ 不符合 |
| **文档完整性** | ✅ 完整 | ❌ 缺少 |
| **验证集划分** | ✅ 有（166样本） | ❌ 无 |
| **可视化** | ✅ 有 | ❌ 无 |
| **数据检查** | ✅ 有 | ❌ 无 |
| **样本数** | 826内部 + 159外部 | 837内部 + 148外部 |

---

## ✅ 推荐结论

### **强烈推荐使用: `5centers_multi_internal_external`**

#### 理由：

1. **✅ 完全符合您的要求**
   - 内部开发集：恩施、襄阳、十堰（混合训练）
   - 外部测试集：荆州、武大（严格隔离，作为最终考场）

2. **✅ 文档完整**
   - 有详细的README说明
   - 有划分报告和统计信息
   - 有可视化图表

3. **✅ 结构规范**
   - 清晰的train/val/external_test划分
   - 有数据检查机制
   - 符合学术规范

4. **✅ 数据质量**
   - 内部开发集：826样本（660训练 + 166验证）
   - 外部测试集：159样本
   - 类别分布合理

---

## 📋 使用建议

### 在新项目中使用

```python
# 推荐使用这个路径
DATA_PATH = "/data2/hmy/5Center_datas/5centers_multi_internal_external"

# 或者在新项目中（已创建软链接）
DATA_PATH = "/data2/hmy/VLM_Caus_Rm/data/5centers_multi_internal_external"

# 加载数据
train_df = pd.read_csv(f"{DATA_PATH}/train_labels.csv")      # 660样本
val_df = pd.read_csv(f"{DATA_PATH}/val_labels.csv")          # 166样本
external_test_df = pd.read_csv(f"{DATA_PATH}/external_test_labels.csv")  # 159样本（仅最终评估）
```

### 在训练脚本中使用

```python
# train_vlm_causal_clip.py
parser.add_argument(
    '--data_path', 
    type=str, 
    default='../../data/5centers_multi_internal_external',  # 使用推荐的划分
    help='数据路径'
)
```

---

## ⚠️ 重要提示

### 数据使用规范

1. **外部测试集严禁参与训练**
   - 外部测试集（荆州、武大）必须完全隔离
   - 不能用于：
     - 模型训练
     - 超参数调优
     - 模型选择
     - 早停策略
   - 仅用于：
     - 最终模型性能评估
     - 泛化能力验证
     - 论文结果报告

2. **内部开发集使用**
   - 训练集（660样本）：用于模型训练
   - 验证集（166样本）：用于超参数调优、模型选择、早停

3. **数据泄露防范**
   - 确保训练代码中不会意外加载外部测试集
   - 建议在代码中添加检查机制
   - 定期验证数据划分的正确性

---

## 📝 论文中的描述

如果使用此数据集划分，请在论文中说明：

> "数据集按照医疗中心划分为内部开发集和外部测试集。内部开发集（恩施、襄阳、十堰，n=826）用于模型训练和验证，其中训练集660样本，验证集166样本。外部测试集（荆州、武大，n=159）完全独立于训练过程，仅用于最终性能评估，验证模型的泛化能力。"

---

## 🔄 如果使用recommended划分

**不推荐**，但如果必须使用，需要注意：

1. **划分不符合要求**: 武大在内部，十堰在外部（与您的要求相反）
2. **缺少验证集**: 需要从内部训练集中手动划分验证集
3. **文档不完整**: 需要自己补充文档说明

---

## ✅ 最终建议

**使用 `5centers_multi_internal_external`**，因为：
- ✅ 完全符合您的要求
- ✅ 文档完整
- ✅ 结构规范
- ✅ 有数据检查机制
- ✅ 有可视化支持

**不要使用 `5centers_multi_internal_external_recommended`**，因为：
- ❌ 划分不符合您的要求
- ❌ 缺少文档
- ❌ 结构不完整

---

**最后更新**: 2025-12-17

