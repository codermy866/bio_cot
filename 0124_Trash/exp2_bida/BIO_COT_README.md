# Bio-COT 实验实施指南

## 📋 概述

Bio-COT (Bio-Invariant Counterfactual Optimal Transport) 是在BIDA基础上的升级方案，主要改进：

1. **Phase 1**: VLM知识蒸馏 - 离线提取VLM特征，训练轻量级Student Prior网络
2. **Phase 2**: Bio-COT核心模块 - Sinkhorn OT Loss和Memory Bank
3. **Phase 3**: TTPA推理流程 - 测试时适配提升性能
4. **Phase 4**: 反事实轨迹可视化 - 展示模型学习效果

---

## 🚀 快速开始

### Step 1: 离线提取VLM特征（必须）

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics

# 提取训练集VLM特征
python experiments/exp2_bida/extract_vlm_features.py \
    --data_root /data2/hmy/5Center_datas/5centers_multi_leave_centers_out \
    --split train \
    --output_file /data2/hmy/5Center_datas/5centers_multi_leave_centers_out/vlm_features_cache/train_vlm_features.npy \
    --device cuda:1 \
    --batch_size 8

# 提取验证集VLM特征
python experiments/exp2_bida/extract_vlm_features.py \
    --data_root /data2/hmy/5Center_datas/5centers_multi_leave_centers_out \
    --split val \
    --output_file /data2/hmy/5Center_datas/5centers_multi_leave_centers_out/vlm_features_cache/val_vlm_features.npy \
    --device cuda:1 \
    --batch_size 8
```

**预期时间**：2-3小时（取决于数据量）

### Step 2: 训练Bio-COT模型

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics

python experiments/exp2_bida/train_bio_cot.py
```

训练脚本会自动：
1. 预训练Student Prior网络（如果启用）
2. 训练完整的Bio-COT模型
3. 保存最佳模型

**配置参数**（在`train_bio_cot.py`的`BioCOTArgs`类中）：
- `batch_size = 32`
- `learning_rate = 2e-4`
- `lambda_ot = 1.0` (Sinkhorn OT损失权重)
- `lambda_consist = 2.0` (反事实一致性损失权重)
- `lambda_adv = 0.5` (对抗损失权重)

### Step 3: 测试（可选TTPA）

```bash
# 标准测试
python experiments/exp2_bida/test_adaptation.py \
    --model_path experiments/exp2_bida/exp_bio_cot/best_model.pth \
    --split val \
    --device cuda:1

# TTPA测试（提升AUC）
python experiments/exp2_bida/test_adaptation.py \
    --model_path experiments/exp2_bida/exp_bio_cot/best_model.pth \
    --split val \
    --device cuda:1 \
    --use_ttpa \
    --adaptation_steps 1 \
    --ttpa_lr 1e-3
```

### Step 4: 可视化反事实轨迹

```bash
python experiments/exp2_bida/visualize_counterfactual.py \
    --model_path experiments/exp2_bida/exp_bio_cot/best_model.pth \
    --split val \
    --device cuda:1 \
    --num_samples 10 \
    --output_file counterfactual_trajectory.pdf
```

---

## 📁 文件结构

```
experiments/exp2_bida/
├── extract_vlm_features.py          # Phase 1: VLM特征提取
├── train_bio_cot.py                 # Phase 3: 训练脚本
├── test_adaptation.py               # Phase 3: TTPA测试
├── visualize_counterfactual.py      # Phase 4: 可视化
└── BIO_COT_README.md                # 本文档

src/models/bida/
├── prior_net.py                     # Phase 1: Student Prior网络
├── losses.py                        # Phase 2: Sinkhorn OT等损失
├── memory_bank.py                   # Phase 2: Memory Bank
└── bio_cot_model.py                # Phase 2: Bio-COT模型
```

---

## 🔧 核心模块说明

### 1. Student Prior网络 (`prior_net.py`)

**功能**：替代在线VLM，从临床数据生成语义锚点

**输入**：临床数据向量 [HPV(1) + TCT(5) + Age(1)] = 7维
**输出**：语义锚点特征 [768维]

**预训练**：使用VLM特征缓存预训练50个epoch，使其输出拟合VLM特征

### 2. Sinkhorn OT Loss (`losses.py`)

**功能**：替代KL散度，实现更灵活的分布对齐

**优势**：
- 不需要假设分布形式（KL需要高斯分布）
- 更灵活的传输计划
- 数值稳定性更好

### 3. Memory Bank (`memory_bank.py`)

**功能**：存储和采样不同中心的噪声特征，实现反事实干预

**使用**：
- 训练时：实时更新，存储每个中心的噪声特征
- 推理时：从Memory Bank采样反事实噪声，生成合成特征

### 4. Bio-COT模型 (`bio_cot_model.py`)

**架构**：
- Student Prior：生成语义锚点
- Dual Head Encoder：提取因果和噪声特征
- Memory Bank：存储噪声特征
- Center Discriminator：预测中心ID

**损失函数**：
- L_cls：分类损失
- L_ot：Sinkhorn OT损失（因果特征 <-> 语义锚点）
- L_consist：反事实一致性损失
- L_adv：对抗损失（噪声特征预测中心ID）

---

## 📊 预期效果

### 训练速度

- **BIDA（原版）**：每个epoch ~2小时（VLM瓶颈）
- **Bio-COT**：每个epoch ~5-10分钟（Student Prior快速）
- **提升**：**10-20倍**

### 性能指标

- **Val Acc**：从67%提升到**75-80%**
- **Val AUC**：从0.61提升到**0.75-0.85**
- **TTPA后**：AUC进一步提升**2-5个百分点**

---

## ⚠️ 注意事项

1. **VLM特征提取**：必须在训练前完成，否则Student Prior无法预训练
2. **Memory Bank容量**：默认100，可根据数据量调整
3. **Sinkhorn迭代次数**：默认100，可调整`eps`和`max_iter`平衡精度和速度
4. **TTPA学习率**：建议1e-3，过大可能导致过拟合

---

## 🐛 故障排除

### 问题1：VLM特征提取失败

**症状**：`extract_vlm_features.py`报错

**解决**：
- 检查Qwen2-VL模型是否正确下载
- 检查显存是否充足（需要~10GB）
- 减小batch_size（从8降到4）

### 问题2：Student Prior预训练失败

**症状**：找不到VLM特征缓存

**解决**：
- 确保先运行`extract_vlm_features.py`
- 检查`vlm_features_cache`目录是否存在
- 检查文件路径是否正确

### 问题3：训练时Memory Bank为空

**症状**：反事实一致性损失为0

**解决**：
- 确保`center_labels`正确传递
- 检查Memory Bank的`update`方法是否被调用
- 增加训练步数，让Memory Bank积累特征

---

## 📝 实验排期建议

**Day 1**：
- ✅ 完成`extract_vlm_features.py`并运行（可能需要跑一晚上）
- ✅ 编写并测试`StudentPriorNet`

**Day 2**：
- ✅ 实现`SinkhornDistance`和`NoiseMemoryBank`类
- ✅ 测试各个模块

**Day 3**：
- ✅ 整合训练Loop
- ✅ 先用小Batch跑通流程
- ✅ 观察Loss下降曲线

**Day 4**：
- ✅ 开启全量LCO训练
- ✅ 监控训练指标

**Day 5**：
- ✅ 编写TTPA脚本
- ✅ 对比开启前后的AUC变化
- ✅ 生成可视化结果

---

## 📚 相关文档

- `BIDA_TECHNICAL_ANALYSIS_FOR_MICCAI.md`：BIDA方法技术分析
- `BIDA_EXPERIMENT_DETAILS_ANALYSIS.md`：实验细节分析
- `BIDA_COMPLETE_TECHNICAL_REPORT.md`：完整技术报告

---

**创建日期**：2025-12-29  
**状态**：✅ 所有模块已实现，可以开始实验

