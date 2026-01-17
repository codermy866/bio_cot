# 视觉大模型Backbone训练说明

## 📋 概述

已成功实现并启动使用视觉大模型（ViT/Swin Transformer）作为backbone的训练。

## 🚀 已启动的训练

### 1. ResNet Backbone (cuda=0)
- **状态**: 正在运行
- **日志**: `paper1_hierarchical_multimodal/logs/train_cuda0_optimized_v2.log`
- **输出目录**: `paper1_hierarchical_multimodal/results/cuda0_optimized_v2/`
- **Backbone**: ResNet34 (局部) + ResNet50 (全局)
- **Batch Size**: 8
- **Epochs**: 30

### 2. ViT Backbone (cuda=1) ⭐ 新增
- **状态**: 正在运行
- **日志**: `paper1_hierarchical_multimodal/logs/train_cuda1_vit_backbone.log`
- **输出目录**: `paper1_hierarchical_multimodal/results/cuda1_vit_backbone/`
- **Backbone**: ViT-Base (vit_base_patch16_224)
- **Batch Size**: 4 (ViT需要更多显存)
- **Epochs**: 30

## 🔧 实现细节

### 新增文件

1. **`models/vision_transformer_encoder.py`**
   - `VisionTransformerLocalEncoder`: 使用ViT提取局部特征
   - `VisionTransformerGlobalEncoder`: 使用ViT提取全局特征
   - `SwinTransformerLocalEncoder`: 使用Swin Transformer提取局部特征
   - `SwinTransformerGlobalEncoder`: 使用Swin Transformer提取全局特征

2. **`models/hierarchical_multimodal_model_vit.py`**
   - `HierarchicalMultimodalModelViT`: 使用视觉大模型作为backbone的完整模型

### 训练参数

```bash
CUDA_VISIBLE_DEVICES=1 python train_hierarchical_multimodal.py \
  --use_vit_backbone \
  --backbone_type vit \
  --vit_model_name vit_base_patch16_224 \
  --batch_size 4 \
  --num_epochs 30 \
  --learning_rate 5e-5 \
  --use_amp \
  --use_focal_loss
```

## 📊 预期优势

### ViT Backbone的优势

1. **更强的特征提取能力**
   - ViT使用全局自注意力机制，能够捕获长距离依赖
   - 对医学图像的细微病变特征有更好的识别能力

2. **更好的多尺度特征**
   - 通过patch tokens可以提取不同粒度的特征
   - 局部和全局特征的区分更明显

3. **预训练权重优势**
   - 使用ImageNet预训练的ViT权重
   - 可以更好地迁移到医学图像任务

## 🔍 监控训练

### 查看ResNet训练进度
```bash
tail -f paper1_hierarchical_multimodal/logs/train_cuda0_optimized_v2.log
```

### 查看ViT训练进度
```bash
tail -f paper1_hierarchical_multimodal/logs/train_cuda1_vit_backbone.log
```

### 检查GPU使用情况
```bash
nvidia-smi
```

## 📈 对比实验

训练完成后，可以对比：
- **ResNet Backbone**: 参数量较小，训练速度快
- **ViT Backbone**: 参数量较大，但特征提取能力更强

预期ViT backbone在ACC和AUC指标上会有更好的表现。

## 🎯 下一步

1. 监控两个训练进程的进度
2. 等待训练完成（30个epochs）
3. 对比两个模型的性能指标
4. 选择表现更好的模型作为最终方案

## ⚠️ 注意事项

1. **显存占用**: ViT模型需要更多显存，batch size设置为4
2. **训练时间**: ViT模型训练时间可能更长
3. **学习率**: 使用相同的学习率（5e-5）和优化策略

