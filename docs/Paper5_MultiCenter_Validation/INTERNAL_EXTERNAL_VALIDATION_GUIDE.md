# 内部外部验证快速指南
## 使用5个多中心数据集创建外部验证集

### 📋 概述

本指南说明如何使用现有的5个多中心数据集创建内部外部验证集，这是Lancet期刊可接受的折中方案。

---

## 🎯 方案选择

### 方案A：中心分割验证（推荐）⭐

**方法**: 将5个中心中的1-2个中心作为外部验证集

**优点**:
- ✅ 不同中心、不同设备、不同操作人员
- ✅ 符合Lancet期刊要求（不同中心）
- ✅ 数据收集成本低
- ✅ 实施简单

**Lancet接受度**: ✅ **可接受**

**推荐配置**:
- 外部验证集: 1-2个中心（如恩施、荆州）
- 内部训练集: 其余3-4个中心
- 样本量: 外部验证集≥100例

### 方案B：时间分割验证

**方法**: 按时间顺序分割，后期数据作为外部验证集

**优点**:
- ✅ 时间独立性
- ✅ 模拟真实临床应用场景

**Lancet接受度**: ⚠️ **部分接受**（不如中心分割）

**推荐配置**:
- 外部验证集: 20%的后期数据
- 内部训练集: 80%的早期数据

---

## 🚀 快速开始

### 步骤1: 分析中心分布

```bash
cd /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713
source my_retfound/bin/activate

# 分析中心分布（查看各中心的样本量）
python utils/create_internal_external_validation.py \
    --method center \
    --data_root 5centers_multi \
    --external_centers 恩施 荆州
```

### 步骤2: 创建中心分割验证数据集

```bash
# 将恩施和荆州作为外部验证集
python utils/create_internal_external_validation.py \
    --method center \
    --external_centers 恩施 荆州 \
    --data_root 5centers_multi \
    --output_dir 5centers_multi_internal_external
```

### 步骤3: 创建时间分割验证数据集（可选）

```bash
# 按时间分割，20%作为外部验证集
python utils/create_internal_external_validation.py \
    --method time \
    --split_ratio 0.2 \
    --data_root 5centers_multi \
    --output_dir 5centers_multi_internal_external_time
```

---

## 📊 结果结构

分割完成后，数据将按以下结构组织：

```
5centers_multi_internal_external/
├── internal_train/          # 内部训练集
│   ├── train/
│   │   ├── oct/             # OCT图像
│   │   └── col/             # 阴道镜图像
│   ├── test/
│   │   ├── oct/
│   │   └── col/
│   └── train_labels.csv     # 训练标签
├── external_validation/     # 外部验证集
│   ├── oct/                 # OCT图像
│   ├── col/                 # 阴道镜图像
│   └── external_labels.csv  # 外部验证标签
└── split_statistics.json    # 分割统计报告
```

---

## 🔍 验证分割结果

### 检查统计报告

```bash
cat 5centers_multi_internal_external/split_statistics.json
```

报告包含：
- 分割方法（center_split 或 time_split）
- 内部训练集样本量和分布
- 外部验证集样本量和分布
- 各中心分布情况

### 检查样本量

确保：
- ✅ 外部验证集≥100例
- ✅ 阳性样本≥50例
- ✅ 阴性样本≥50例
- ✅ 类别平衡（阳性/阴性比例在0.3-3.0之间）

---

## 📈 运行外部验证评估

### 使用内部外部验证集评估模型

```bash
# 1. 在内部训练集上重新训练模型（或使用已有模型）
# 2. 在外部验证集上评估

python analysis/external_validation_evaluation.py \
    --model_path models/SwinT/_results/multimodal/best_model.pth \
    --external_data_path 5centers_multi_internal_external/external_validation \
    --model_type swint \
    --output_dir analysis/external_validation_results \
    --device cuda
```

---

## 📝 论文撰写建议

### 在Methods部分说明

```markdown
**External Validation**

We performed center-split validation by holding out data from 2 centers 
(Enshi and Jingzhou) as an external validation set. The remaining 3 centers 
were used for model training. This approach ensures that the external 
validation set comes from different centers with potentially different 
equipment and operators, which is consistent with TRIPOD guidelines for 
external validation.

- Internal training set: 3 centers, N=XXX samples
- External validation set: 2 centers, N=XXX samples
  - Center 1 (Enshi): N=XX samples
  - Center 2 (Jingzhou): N=XX samples
```

### 在Results部分报告

```markdown
**External Validation Performance**

On the external validation set (2 centers, N=XXX), the model achieved:
- AUC: 0.XX (95% CI: 0.XX-0.XX)
- Sensitivity: 0.XX (95% CI: 0.XX-0.XX)
- Specificity: 0.XX (95% CI: 0.XX-0.XX)

The performance was comparable to the internal validation set, with an 
AUC decrease of 0.XX (acceptable threshold: <0.05).
```

---

## ⚠️ 注意事项

### 1. 样本量要求

- **最小要求**: 外部验证集≥100例
- **推荐**: 外部验证集≥200例
- **类别平衡**: 阳性/阴性比例在0.3-3.0之间

### 2. 中心选择

- **推荐**: 选择样本量较小的1-2个中心作为外部验证集
- **避免**: 选择样本量最大的中心（会导致训练集样本量不足）

### 3. 论文说明

- **必须说明**: 使用中心分割验证方法
- **必须报告**: 各中心的样本量和分布
- **必须比较**: 内部验证集和外部验证集的性能差异

### 4. Lancet接受度

- ✅ **中心分割**: 可接受（符合TRIPOD声明）
- ⚠️ **时间分割**: 部分接受（不如中心分割）
- ❌ **随机分割**: 不可接受（不是真正的外部验证）

---

## 🔗 相关文档

- [完整外部验证方案](EXTERNAL_VALIDATION_PLAN.md)
- [外部验证评估脚本](../analysis/external_validation_evaluation.py)
- [数据质量检查脚本](../utils/external_data_quality_check.py)

---

## 📞 常见问题

### Q1: 中心分割验证是否符合Lancet要求？

**A**: ✅ 是的。Lancet期刊接受中心分割验证，因为它确保了外部验证集来自不同的中心、设备和操作人员，符合TRIPOD声明要求。

### Q2: 应该选择哪些中心作为外部验证集？

**A**: 推荐选择样本量较小的1-2个中心，确保：
- 外部验证集≥100例
- 内部训练集仍有足够的样本量
- 各中心样本量相对平衡

### Q3: 时间分割验证是否可接受？

**A**: ⚠️ 部分接受。时间分割验证不如中心分割验证，但如果数据收集时间跨度大（如1年以上），可以作为补充方案。

### Q4: 如何报告内部外部验证结果？

**A**: 在论文中明确说明：
1. 使用中心分割验证方法
2. 各中心的样本量和分布
3. 内部验证集和外部验证集的性能比较
4. 性能下降是否在可接受范围内（AUC下降<0.05）

---

**最后更新**: 2025-01-XX
**版本**: 1.0

