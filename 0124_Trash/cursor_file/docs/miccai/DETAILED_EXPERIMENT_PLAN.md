<!--
文件生成信息:
- 生成时间: 2025-12-25 22:18:01 (使用nvidia-smi获取CUDA时间)
- 生成需求: 用户要求制定详细的实验方案，今天晚上就要开始训练
- 生成原因: 需要基于故事线制定可直接执行的详细实验方案，包括数据准备、模型训练、实验设置、评估指标等
- 相关任务: 详细实验方案制定和执行计划

文件功能: 制定详细的实验方案，确保可以直接执行，包括所有必要的步骤、参数设置和评估指标
-->

# MICCAI实验详细方案：多模态多中心验证

**制定日期**: 2025-12-25  
**执行时间**: 今晚开始  
**实验主题**: 多模态多中心验证 + 解决设备差异（域偏移）  
**数据集**: Leave-Centers-Out (5个医疗中心)

---

## 📊 实验概览

### 实验目标

**主要目标**:
1. 验证多模态因果解耦方法在跨中心泛化中的有效性
2. 证明对齐机制能解决设备差异问题
3. 实现Zero-Shot跨中心稳定泛化

**预期结果**:
- Source AUC: ~0.94-0.96
- Unseen AUC: ~0.84-0.88
- 性能下降: <10% (vs 传统方法33%)

---

## 1. 数据集准备

### 1.1 数据集路径

**数据集根目录**:
```bash
/data2/hmy/5Center_datas/5centers_multi_leave_centers_out
```

**数据集结构**:
```
5centers_multi_leave_centers_out/
├── internal_train/
│   ├── train/
│   │   ├── oct/
│   │   ├── col/
│   │   └── train_labels.csv
│   └── val/
│       ├── oct/
│       ├── col/
│       └── val_labels.csv
└── external_validation/
    ├── oct/
    ├── col/
    └── external_test_labels.csv
```

**数据集统计**:
- **训练集**: 669样本 (恩施、襄阳、武大)
- **验证集**: 168样本 (恩施、襄阳新数据)
- **外部测试集**: 148样本 (十堰、荆州，完全独立)

---

### 1.2 数据检查清单

**执行前检查**:
```bash
# 1. 检查数据集路径
ls -lh /data2/hmy/5Center_datas/5centers_multi_leave_centers_out

# 2. 检查标签文件
head -5 /data2/hmy/5Center_datas/5centers_multi_leave_centers_out/internal_train/train/train_labels.csv
head -5 /data2/hmy/5Center_datas/5centers_multi_leave_centers_out/internal_train/val/val_labels.csv
head -5 /data2/hmy/5Center_datas/5centers_multi_leave_centers_out/external_validation/external_test_labels.csv

# 3. 检查图像文件（随机抽样）
find /data2/hmy/5Center_datas/5centers_multi_leave_centers_out/internal_train/train/oct -name "*.png" | head -5
find /data2/hmy/5Center_datas/5centers_multi_leave_centers_out/internal_train/train/col -name "*.jpg" | head -5
```

---

## 2. 实验设置

### 2.1 实验配置

**实验名称**: `exp_multicenter_alignment`

**实验目录**:
```bash
/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp1_causal_bayesian_clip/exp_multicenter_alignment
```

**创建实验目录**:
```bash
mkdir -p /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp1_causal_bayesian_clip/exp_multicenter_alignment/{checkpoints,logs,results}
```

---

### 2.2 模型配置

**模型架构**: VLM-Enhanced Causal Bayesian CLIP

**核心组件**:
1. **图像编码器**: ResNet50 (ImageNet预训练)
2. **临床特征编码器**: VLM (Qwen-VL) + MLP
3. **对齐机制**: Cosine Similarity Alignment
4. **分类器**: MLP (512 → 2)

**关键参数**:
```python
config = {
    # 数据路径
    'data_path': '/data2/hmy/5Center_datas/5centers_multi_leave_centers_out',
    
    # 模型参数
    'image_encoder': 'resnet50',
    'image_embed_dim': 512,
    'clinical_embed_dim': 512,
    'hidden_dim': 256,
    'num_classes': 2,
    
    # 对齐机制
    'use_alignment': True,
    'alignment_loss_weight': 0.1,  # λ_align
    'alignment_temperature': 0.07,
    
    # 训练参数
    'batch_size': 8,
    'num_epochs': 100,
    'learning_rate': 3e-4,
    'weight_decay': 1e-5,
    'warmup_epochs': 5,
    
    # 数据增强
    'use_augmentation': True,
    'augmentation_prob': 0.5,
    
    # 设备
    'device': 'cuda:0',
    'num_workers': 4,
    'pin_memory': True,
    
    # 混合精度
    'use_amp': True,
    'grad_scaler': True,
    
    # 保存和日志
    'save_dir': './exp_multicenter_alignment/checkpoints',
    'log_dir': './exp_multicenter_alignment/logs',
    'save_freq': 10,  # 每10个epoch保存一次
    'eval_freq': 5,   # 每5个epoch评估一次
}
```

---

### 2.3 损失函数设计

**总损失函数**:
```python
L_total = L_CE + λ_align * L_align + λ_kl * L_kl

其中:
- L_CE: 分类损失 (Cross-Entropy Loss)
- L_align: 对齐损失 (Alignment Loss)
- L_kl: KL散度损失 (Bayesian Regularization)
- λ_align = 0.1 (对齐损失权重)
- λ_kl = 0.001 (KL损失权重)
```

**对齐损失**:
```python
L_align = 1 - CosineSimilarity(Z_img, Z_clinical)

其中:
- Z_img: 图像特征 [B, 512]
- Z_clinical: 临床特征 [B, 512]
- CosineSimilarity: 余弦相似度
```

**分类损失**:
```python
L_CE = CrossEntropyLoss(logits, labels, label_smoothing=0.1)
```

---

## 3. 训练流程

### 3.1 训练脚本

**主训练脚本**: `train_multicenter_alignment.py`

**脚本位置**:
```bash
/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp1_causal_bayesian_clip/train_multicenter_alignment.py
```

**执行命令**:
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp1_causal_bayesian_clip

# 激活虚拟环境
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

# 运行训练
python train_multicenter_alignment.py \
    --data_path /data2/hmy/5Center_datas/5centers_multi_leave_centers_out \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 3e-4 \
    --alignment_loss_weight 0.1 \
    --use_amp \
    --save_dir ./exp_multicenter_alignment/checkpoints \
    --log_dir ./exp_multicenter_alignment/logs
```

---

### 3.2 训练阶段

**阶段1: 基础训练 (Epochs 1-30)**
- 学习率: 3e-4
- 对齐损失权重: 0.05 (较小，先学习基础特征)
- 目标: 模型学习基本的多模态特征

**阶段2: 对齐强化 (Epochs 31-70)**
- 学习率: 1e-4 (降低学习率)
- 对齐损失权重: 0.1 (增加对齐强度)
- 目标: 强化对齐机制，去除设备噪声

**阶段3: 精细调优 (Epochs 71-100)**
- 学习率: 3e-5 (进一步降低)
- 对齐损失权重: 0.1 (保持)
- 目标: 精细调优，提升泛化性能

---

### 3.3 验证策略

**内部验证**:
- 每5个epoch在验证集上评估
- 保存最佳模型（基于验证集AUC）

**外部测试**:
- 训练完成后在外部测试集上评估
- 不参与模型选择，只用于最终报告

---

## 4. 对比实验

### 4.1 基线方法

**实验1: ResNet50 (纯图像基线)**
```bash
python train_baseline.py \
    --model resnet50 \
    --modality image_only \
    --data_path /data2/hmy/5Center_datas/5centers_multi_leave_centers_out
```

**实验2: Late Fusion (传统多模态融合)**
```bash
python train_baseline.py \
    --model late_fusion \
    --modality image_clinical \
    --data_path /data2/hmy/5Center_datas/5centers_multi_leave_centers_out
```

**实验3: Cross-Attention (跨模态注意力)**
```bash
python train_baseline.py \
    --model cross_attention \
    --modality image_clinical \
    --data_path /data2/hmy/5Center_datas/5centers_multi_leave_centers_out
```

**实验4: Clinical-IV (Ours, Image Only)**
```bash
python train_multicenter_alignment.py \
    --data_path /data2/hmy/5Center_datas/5centers_multi_leave_centers_out \
    --test_modality image_only \
    --alignment_loss_weight 0.1
```

**实验5: Clinical-IV (Ours, Full)**
```bash
python train_multicenter_alignment.py \
    --data_path /data2/hmy/5Center_datas/5centers_multi_leave_centers_out \
    --test_modality image_clinical \
    --alignment_loss_weight 0.1
```

---

### 4.2 消融实验

**消融1: 对齐机制的影响**
```bash
# 无对齐机制
python train_multicenter_alignment.py --alignment_loss_weight 0.0

# 弱对齐 (λ=0.05)
python train_multicenter_alignment.py --alignment_loss_weight 0.05

# 中等对齐 (λ=0.1)
python train_multicenter_alignment.py --alignment_loss_weight 0.1

# 强对齐 (λ=0.2)
python train_multicenter_alignment.py --alignment_loss_weight 0.2
```

**消融2: 临床先验的形式**
```bash
# One-hot Encoding
python train_multicenter_alignment.py --clinical_encoder onehot

# VLM Embeddings (Ours)
python train_multicenter_alignment.py --clinical_encoder vlm
```

**消融3: 对齐策略**
```bash
# MSE Loss
python train_multicenter_alignment.py --alignment_loss mse

# Cosine Similarity (Ours)
python train_multicenter_alignment.py --alignment_loss cosine
```

---

## 5. 评估指标

### 5.1 主要指标

**分类性能**:
- AUC (Area Under ROC Curve)
- Accuracy
- Sensitivity (Recall)
- Specificity
- F1-Score
- Precision

**跨中心泛化**:
- Source AUC (源域性能)
- Unseen AUC (目标域性能)
- Performance Drop (性能下降百分比)

**对齐质量**:
- Cosine Similarity (Z_img, Z_clinical)
- Alignment Loss

---

### 5.2 评估脚本

**评估脚本**: `evaluate_multicenter.py`

**执行命令**:
```bash
python evaluate_multicenter.py \
    --checkpoint ./exp_multicenter_alignment/checkpoints/best_model.pth \
    --data_path /data2/hmy/5Center_datas/5centers_multi_leave_centers_out \
    --split internal_val \
    --output_dir ./exp_multicenter_alignment/results
```

---

## 6. 实验执行计划

### 6.1 今晚执行计划

**步骤1: 环境准备 (15分钟)**
```bash
# 1. 激活虚拟环境
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

# 2. 检查GPU
nvidia-smi

# 3. 检查数据集
ls -lh /data2/hmy/5Center_datas/5centers_multi_leave_centers_out
```

**步骤2: 数据检查 (10分钟)**
```bash
# 检查标签文件
python check_dataset.py \
    --data_path /data2/hmy/5Center_datas/5centers_multi_leave_centers_out
```

**步骤3: 开始训练 (主实验)**
```bash
# 训练主实验 (Clinical-IV Full)
python train_multicenter_alignment.py \
    --data_path /data2/hmy/5Center_datas/5centers_multi_leave_centers_out \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 3e-4 \
    --alignment_loss_weight 0.1 \
    --use_amp \
    --save_dir ./exp_multicenter_alignment/checkpoints \
    --log_dir ./exp_multicenter_alignment/logs \
    --experiment_name clinical_iv_full
```

**预计时间**: 
- 每个epoch: ~5-10分钟
- 100个epoch: ~8-16小时
- 建议: 今晚开始训练，明天早上检查结果

---

### 6.2 明天执行计划

**步骤1: 检查训练结果**
```bash
# 查看训练日志
tail -f ./exp_multicenter_alignment/logs/train.log

# 检查最佳模型
ls -lh ./exp_multicenter_alignment/checkpoints/
```

**步骤2: 评估主实验**
```bash
# 在验证集上评估
python evaluate_multicenter.py \
    --checkpoint ./exp_multicenter_alignment/checkpoints/best_model.pth \
    --data_path /data2/hmy/5Center_datas/5centers_multi_leave_centers_out \
    --split internal_val

# 在外部测试集上评估
python evaluate_multicenter.py \
    --checkpoint ./exp_multicenter_alignment/checkpoints/best_model.pth \
    --data_path /data2/hmy/5Center_datas/5centers_multi_leave_centers_out \
    --split external_test
```

**步骤3: 开始对比实验**
```bash
# 实验1: ResNet50基线
python train_baseline.py --model resnet50 --modality image_only

# 实验2: Late Fusion
python train_baseline.py --model late_fusion --modality image_clinical

# 实验3: Cross-Attention
python train_baseline.py --model cross_attention --modality image_clinical
```

---

### 6.3 后续执行计划

**第2-3天: 完成所有对比实验**

**第4-5天: 完成消融实验**

**第6-7天: 结果分析和可视化**

---

## 7. 监控和日志

### 7.1 训练监控

**TensorBoard日志**:
```bash
# 启动TensorBoard
tensorboard --logdir ./exp_multicenter_alignment/logs --port 6006

# 访问: http://localhost:6006
```

**关键监控指标**:
- Train Loss
- Val Loss
- Train AUC
- Val AUC
- Alignment Loss
- Cosine Similarity

---

### 7.2 日志文件

**训练日志**: `./exp_multicenter_alignment/logs/train.log`
- 每个epoch的训练和验证指标
- 最佳模型保存信息

**评估日志**: `./exp_multicenter_alignment/logs/eval.log`
- 各数据集的详细评估结果

**错误日志**: `./exp_multicenter_alignment/logs/error.log`
- 训练过程中的错误信息

---

## 8. 预期结果

### 8.1 主实验预期结果

**Table 1: 跨中心泛化性能**

| Method | Modality (Train) | Modality (Test) | Source AUC | Unseen AUC | Drop |
|--------|-----------------|----------------|-----------|-----------|------|
| ResNet50 | Image | Image | 0.95 | 0.62 | -33% |
| Late Fusion | Image+Clinical | Image+Clinical | 0.98 | 0.72 | -26% |
| Cross-Attn | Image+Clinical | Image+Clinical | 0.98 | 0.75 | -23% |
| Clinical-IV (Ours) | Image+Clinical | Image Only | 0.94 | 0.84 | -10% |
| Clinical-IV (Ours) | Image+Clinical | Image+Clinical | 0.96 | 0.88 | -8% |

**关键发现**:
- ✅ 我们的方法性能下降从33%降低到8-10%
- ✅ 解决了设备差异问题
- ✅ 实现了跨中心稳定泛化

---

### 8.2 消融实验预期结果

**消融1: 对齐机制的影响**
- 无对齐: Unseen AUC ~0.72
- 弱对齐 (λ=0.05): Unseen AUC ~0.80
- 中等对齐 (λ=0.1): Unseen AUC ~0.84 ✅
- 强对齐 (λ=0.2): Unseen AUC ~0.82

**消融2: 临床先验的形式**
- One-hot: Unseen AUC ~0.78
- VLM Embeddings: Unseen AUC ~0.84 ✅

**消融3: 对齐策略**
- MSE Loss: Unseen AUC ~0.80
- Cosine Similarity: Unseen AUC ~0.84 ✅

---

## 9. 故障排除

### 9.1 常见问题

**问题1: CUDA Out of Memory**
```bash
# 解决方案: 减小batch_size
--batch_size 4  # 从8减小到4

# 或使用梯度累积
--gradient_accumulation_steps 2
```

**问题2: 数据加载错误**
```bash
# 检查数据路径
python check_dataset.py --data_path /data2/hmy/5Center_datas/5centers_multi_leave_centers_out

# 检查软链接
ls -l /data2/hmy/5Center_datas/5centers_multi_leave_centers_out/internal_train/train/oct
```

**问题3: 模型收敛慢**
```bash
# 调整学习率
--learning_rate 5e-4  # 从3e-4增加到5e-4

# 或使用学习率调度器
--scheduler cosine
```

---

### 9.2 调试技巧

**启用详细日志**:
```bash
--log_level DEBUG
```

**保存中间结果**:
```bash
--save_intermediate --save_freq 5
```

**可视化特征**:
```bash
--visualize_features --vis_freq 10
```

---

## 10. 检查清单

### 10.1 训练前检查

- [ ] 数据集路径正确
- [ ] 标签文件完整
- [ ] 图像文件可访问
- [ ] GPU可用
- [ ] 虚拟环境激活
- [ ] 依赖包安装
- [ ] 实验目录创建

---

### 10.2 训练中检查

- [ ] 训练损失下降
- [ ] 验证损失下降
- [ ] AUC提升
- [ ] 对齐损失收敛
- [ ] 无CUDA错误
- [ ] 模型保存正常

---

### 10.3 训练后检查

- [ ] 最佳模型保存
- [ ] 评估结果保存
- [ ] 日志文件完整
- [ ] 结果可视化
- [ ] 统计显著性检验

---

## 11. 快速开始命令

### 11.1 一键启动训练

```bash
#!/bin/bash
# 快速启动脚本: quick_start.sh

# 激活环境
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

# 创建实验目录
mkdir -p ./exp_multicenter_alignment/{checkpoints,logs,results}

# 开始训练
python train_multicenter_alignment.py \
    --data_path /data2/hmy/5Center_datas/5centers_multi_leave_centers_out \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 3e-4 \
    --alignment_loss_weight 0.1 \
    --use_amp \
    --save_dir ./exp_multicenter_alignment/checkpoints \
    --log_dir ./exp_multicenter_alignment/logs \
    --experiment_name clinical_iv_full \
    2>&1 | tee ./exp_multicenter_alignment/logs/train.log
```

**执行**:
```bash
chmod +x quick_start.sh
./quick_start.sh
```

---

## 12. 总结

### 12.1 实验要点

**核心实验**:
- ✅ 主实验: Clinical-IV (Full)
- ✅ 对比实验: ResNet50, Late Fusion, Cross-Attn
- ✅ 消融实验: 对齐机制、临床先验、对齐策略

**关键指标**:
- ✅ Source AUC
- ✅ Unseen AUC
- ✅ Performance Drop

**预期结果**:
- ✅ 性能下降从33%降低到8-10%
- ✅ 解决设备差异问题
- ✅ 实现跨中心稳定泛化

---

### 12.2 执行时间表

**今晚**: 开始主实验训练
**明天**: 评估主实验，开始对比实验
**第2-3天**: 完成所有对比实验
**第4-5天**: 完成消融实验
**第6-7天**: 结果分析和可视化

---

**方案制定日期**: 2025-12-25 22:18:01  
**核心结论**: ✅ **实验方案完整，可直接执行。今晚开始训练主实验，预计明天早上完成。**

