# Lancet期刊发表所需实验 - 中文说明

## 📋 你需要完成的实验清单

根据你当前的实验结果（SwinT AUC=0.8069），我已经为你创建了完整的实验计划和脚本。以下是**必须完成**的实验：

---

## 🔴 **优先级1：关键实验**（必须完成）

### 1. Bootstrap置信区间计算 ⏰ **1-2天**

**目标**: 为所有指标提供95%置信区间（Lancet要求）

**已完成**:
- ✅ 脚本已创建: `analysis/bootstrap_confidence_intervals.py`
- ✅ 可以计算所有指标的Bootstrap CI

**你需要做的**:
1. **修改训练脚本保存预测结果**（最重要！）
   - 文件: `training/optimized_2class_training.py`
   - 在验证循环中保存所有预测概率和标签
   - 保存为 `.npy` 文件：`val_labels.npy`, `val_probs.npy`, `val_preds.npy`
   - 同时保存元数据：`val_metadata.csv`（包含年龄、HPV、TCT、中心等信息）

2. **运行Bootstrap CI分析**
   ```bash
   python analysis/bootstrap_confidence_intervals.py \
       --result_dirs models/SwinT/_results/multimodal cnn_result_unified vmamba_result_unified \
       --output_dir analysis/bootstrap_ci_results
   ```

**输出**: 所有指标都带有95%置信区间，格式如：`AUC: 0.8069 (0.7823-0.8315)`

---

### 2. 亚组分析 ⏰ **3-5天**

**目标**: 评估模型在不同患者亚组中的性能（Lancet要求）

**已完成**:
- ✅ 脚本已创建: `analysis/subgroup_analysis.py`
- ✅ 可以分析年龄组、HPV类型、TCT结果、中心等亚组

**你需要做的**:
1. **确保元数据可用**
   - 从 `train_labels.csv` 和 `test_labels.csv` 提取：
     - 年龄 (AGE)
     - HPV类型 (HPV清洗)
     - TCT结果 (TCT清洗)
     - 中心ID（如果可用）

2. **运行亚组分析**
   ```bash
   python analysis/subgroup_analysis.py \
       --result_dir models/SwinT/_results/multimodal \
       --data_path 5centers_multi \
       --output_dir analysis/subgroup_results \
       --model_name SwinT
   ```

**输出**:
- 亚组分析表格（每个亚组的AUC、敏感性、特异性等）
- 森林图（Forest Plot）显示各亚组性能

---

### 3. 决策曲线分析 (DCA) ⏰ **2-3天**

**目标**: 评估临床实用性和最优决策阈值（Lancet非常重视）

**已完成**:
- ✅ 脚本已创建: `analysis/decision_curve_analysis.py`
- ✅ 可以计算净收益和最优阈值

**你需要做的**:
1. **运行DCA分析**
   ```bash
   python analysis/decision_curve_analysis.py \
       --y_true models/SwinT/_results/multimodal/val_labels.npy \
       --y_probs models/SwinT/_results/multimodal/val_probs.npy \
       --output_dir analysis/dca_results \
       --model_name SwinT
   ```

**输出**:
- DCA曲线图（显示不同阈值下的净收益）
- 最优阈值推荐
- 与"全部治疗"和"全部不治疗"策略的比较

---

### 4. 外部验证 ⏰ **2-4周**（数据收集）

**目标**: 在独立中心验证模型性能（Lancet最重要的要求）

**你需要做的**:
1. **收集新的独立中心数据**
   - 至少100个样本（理想≥200）
   - 确保数据分布与训练数据匹配
   - 不能是训练时使用的5个中心

2. **评估模型性能**
   - 使用最佳模型（SwinT）
   - 计算所有指标和CI
   - 比较内部vs外部验证性能

**为什么重要**: 外部验证是Lancet发表的最重要要求，没有外部验证很难被接受

---

## 🟡 **优先级2：重要实验**（强烈推荐）

### 5. 跨中心验证 (Leave-One-Center-Out) ⏰ **3-5天**

**目标**: 评估模型在不同中心的泛化能力

**你需要做的**:
- 实现LOCO交叉验证（每个中心作为测试集）
- 分析中心特异性性能
- 识别影响性能的中心因素

---

### 6. 校准分析 ⏰ **2-3天**

**目标**: 评估预测概率的可靠性

**你需要做的**:
- 计算校准曲线
- ECE (Expected Calibration Error)
- Brier Score
- Hosmer-Lemeshow test

---

### 7. 与临床基线比较 ⏰ **1-2周**（数据依赖）

**目标**: 比较AI模型与标准临床实践（HPV+TCT）

**你需要做的**:
- 收集基线模型结果（HPV+TCT组合）
- 统计比较（DeLong test for AUC）
- 计算NRI和IDI

---

## 🛠️ **立即行动：修改训练脚本**

**最重要的一步**：修改训练脚本保存预测结果，这样后续所有分析才能运行。

**文件**: `training/optimized_2class_training.py`

**需要添加的代码**（在验证循环中）:

```python
# 在验证循环开始前初始化
all_val_labels = []
all_val_probs = []
all_val_preds = []
all_val_patient_ids = []
all_val_ages = []
all_val_hpv = []
all_val_tct = []
all_val_centers = []

# 在验证循环中收集数据
for batch in val_loader:
    # ... 现有代码 ...
    
    # 收集预测结果
    all_val_labels.extend(labels.cpu().numpy())
    all_val_probs.extend(probs.cpu().numpy())
    all_val_preds.extend(preds.cpu().numpy())
    
    # 收集元数据（如果可用）
    # all_val_patient_ids.extend(patient_ids)
    # all_val_ages.extend(ages)
    # ...

# 训练结束后保存
output_dir = args.output_dir  # 或你的输出目录
np.save(os.path.join(output_dir, 'val_labels.npy'), np.array(all_val_labels))
np.save(os.path.join(output_dir, 'val_probs.npy'), np.array(all_val_probs))
np.save(os.path.join(output_dir, 'val_preds.npy'), np.array(all_val_preds))

# 保存元数据
metadata = pd.DataFrame({
    'patient_id': all_val_patient_ids,
    'age': all_val_ages,
    'hpv': all_val_hpv,
    'tct': all_val_tct,
    'center_id': all_val_centers
})
metadata.to_csv(os.path.join(output_dir, 'val_metadata.csv'), index=False)
```

---

## 📊 **使用综合分析脚本（推荐）**

我已经创建了一个综合分析脚本，可以一次性运行所有分析：

```bash
# 为SwinT模型运行所有分析
python analysis/run_lancet_analysis.py \
    --result_dir models/SwinT/_results/multimodal \
    --data_path 5centers_multi \
    --model_name SwinT

# 为CNN模型运行所有分析
python analysis/run_lancet_analysis.py \
    --result_dir cnn_result_unified \
    --data_path 5centers_multi \
    --model_name CNN

# 为Vmamba模型运行所有分析
python analysis/run_lancet_analysis.py \
    --result_dir vmamba_result_unified \
    --data_path 5centers_multi \
    --model_name Vmamba
```

---

## 📈 **预期输出**

完成所有分析后，你将得到：

### 表格:
1. **Table 1**: 患者特征（包含所有亚组信息）
2. **Table 2**: 模型性能（所有指标+95% CI）
3. **Table 3**: 亚组分析结果
4. **Table 4**: DCA分析结果（最优阈值）

### 图表:
1. **Figure 1**: ROC曲线（所有模型）
2. **Figure 2**: 决策曲线分析 (DCA)
3. **Figure 3**: 亚组分析森林图（AUC）
4. **Figure 4**: 亚组分析森林图（Sensitivity/Specificity）

---

## ✅ **检查清单**

### 立即完成（本周）:
- [ ] 修改训练脚本保存预测结果
- [ ] 重新运行SwinT模型（保存预测结果）
- [ ] 运行Bootstrap CI分析
- [ ] 运行DCA分析
- [ ] 运行亚组分析

### 短期完成（2-4周）:
- [ ] 跨中心验证
- [ ] 校准分析
- [ ] 基线比较（如果数据可用）

### 中期完成（4-8周）:
- [ ] 外部验证（数据收集）
- [ ] 失败案例分析
- [ ] 成本效益分析

---

## 🎯 **成功标准**

### 最低要求（可发表）:
- ✅ 外部验证AUC ≥ 0.75
- ✅ 所有指标有95% CI
- ✅ 亚组分析显示一致性能
- ✅ DCA显示临床实用性

### 理想目标（高影响）:
- ✅ 外部验证AUC ≥ 0.80
- ✅ 非劣效或优于基线
- ✅ 所有亚组性能稳定

---

## 📚 **详细文档**

我已经创建了以下文档，你可以查看详细信息：

1. **实验计划**: `docs/LANCET_PUBLICATION_EXPERIMENT_PLAN.md`
   - 完整的实验计划（英文）
   - 详细的实施步骤
   - 时间线规划

2. **实验清单**: `docs/EXPERIMENT_CHECKLIST.md`
   - 检查清单
   - 成功标准
   - 预期输出

---

## 🚀 **下一步**

1. **立即修改训练脚本**保存预测结果（最重要！）
2. **重新运行最佳模型**（SwinT）并保存预测结果
3. **运行综合分析脚本**生成所有结果
4. **开始收集外部验证数据**

---

**有问题随时问我！** 我可以帮你：
- 修改训练脚本
- 调试分析脚本
- 解释分析结果
- 准备论文材料

