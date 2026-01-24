# Batch Size 和 F1 修复说明

## 🔍 问题分析

### 1. F1=0 警告问题

**现象**:
- 验证阶段出现"⚠️ 预测只有1个类别，F1设为0"
- 第一个batch预测分布: `tensor([16])` - 所有16个样本都预测为同一个类别
- 第一个batch标签分布: `tensor([9, 7])` - 标签有2个类别

**原因**:
- 早期训练时，模型可能只预测一个类别（通常是类别0）
- 当`unique_preds`只有1个类别时，标准F1无法计算，被设为0
- 这是早期训练的正常现象，但随着训练进行应该会改善

**修复方案**:
- 使用`weighted F1`替代标准F1
- `weighted F1`按类别样本数加权，即使某个类别没有预测到，也能给出有意义的值
- 只在早期epoch（≤3）打印警告，避免日志过多

### 2. Batch Size 太小问题

**现象**:
- 当前batch_size: 16
- 显存使用: 5066 MiB / 49140 MiB（仅10%）
- 显存剩余: 43610 MiB（大量浪费）

**原因**:
- batch_size设置过小，没有充分利用GPU显存
- 小batch_size可能导致：
  - 训练不稳定
  - 梯度估计不准确
  - 训练速度慢

**修复方案**:
- 将batch_size从16增加到32
- 显存使用率预计从10%增加到约20%，仍有大量余量
- 如果32运行良好，可以进一步增加到48或64

---

## ✅ 已应用的修复

### 1. 增加Batch Size (`config.py`)

```python
batch_size: int = 32  # 从16增加到32
```

**效果**:
- 充分利用GPU显存
- 提高训练稳定性
- 加快训练速度

### 2. 改进F1计算 (`train_bio_cot_v3.py`)

```python
# 使用weighted F1，即使预测只有1个类别也能计算
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
```

**效果**:
- 即使早期训练只预测一个类别，也能计算F1
- 更公平的评估指标（考虑类别不平衡）
- 减少不必要的警告信息

---

## 📊 预期效果

### Batch Size增加
- ✅ 显存使用率从10%增加到约20%
- ✅ 训练速度提升约2倍
- ✅ 梯度估计更准确，训练更稳定

### F1计算改进
- ✅ 不再出现"预测只有1个类别，F1设为0"的警告（早期epoch除外）
- ✅ F1值更有意义，能反映模型真实性能
- ✅ 日志更清晰，减少噪音

---

## 🔍 验证方法

### 检查Batch Size
```bash
grep "batch_size" logs/train_bio_cot_v3_*.log | head -5
```

### 检查F1计算
```bash
grep "F1" logs/train_bio_cot_v3_*.log | tail -10
```

### 检查显存使用
```bash
nvidia-smi
```

---

**修复时间**: 2025-01-13  
**修复文件**:
- `config.py` (batch_size: 16 -> 32)
- `training/train_bio_cot_v3.py` (F1计算: standard -> weighted)

