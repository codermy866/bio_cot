# Bio-COT 多模态实验总结

## 实验目标

比较单模态（OCT）和多模态（OCT + Colposcopy + Clinical）在Bio-COT方法下的效果差异。

## 数据集

### 单模态数据集
- **路径**: `/data2/hmy/5Center_datas/襄阳按点图片分类/襄阳按点图片分类`
- **模态**: 仅OCT图像
- **训练脚本**: `train_bio_cot_xiangyang.py`

### 多模态数据集
- **路径**: `/data2/hmy/5Center_datas/襄阳按点图片分类_binary_multimodal`
- **模态**: OCT + Colposcopy + Clinical (HPV, TCT, Age)
- **训练脚本**: `train_bio_cot_multimodal_xiangyang.py`
- **数据格式**: CSV文件（train_labels.csv, val_labels.csv, test_labels.csv）

## 数据集统计

### 多模态数据集
- **训练集**: 230个样本（阴性=203, 阳性=27）
- **验证集**: 35个样本（阴性=32, 阳性=3）
- **数据格式**:
  - OCT路径：分号分隔的多个路径
  - Colposcopy路径：分号分隔的多个路径（最多3张）
  - 临床数据：age, hpv, tct

## 模型配置

### 共同配置
- **模型**: BioCOTModel
- **embed_dim**: 768
- **num_classes**: 2
- **input_dim**: 512
- **use_vlm_encoder**: False（使用传统MLP编码器）

### 多模态特定配置
- **batch_size**: 8（多模态数据batch size稍小）
- **oct_num_frames**: 60
- **max_col_images**: 3
- **损失权重**:
  - λ_cls = 1.0
  - λ_ot = 1.0
  - λ_consist = 0.5
  - λ_adv = 0.1

## 关键实现细节

### 1. 数据集加载器
- **类**: `XiangyangMultimodalDatasetFromCSV`
- **功能**:
  - 从CSV文件加载数据
  - 解析OCT路径（分号分隔）
  - 解析Colposcopy路径（分号分隔）
  - 解析临床数据（HPV, TCT, Age）
  - 转换为Bio-COT需要的格式

### 2. 特征提取
- **OCT特征**: 使用ResNet50提取，多帧平均池化
- **Colposcopy特征**: 使用ResNet50提取，多张图像平均池化
- **临床特征**: 直接使用（[B, 7]）

### 3. 模型前向传播
- **输入**:
  - `oct_features`: [B, 512]
  - `colpo_features`: [B, 512]
  - `clinical_features`: [B, 7]
  - `clinical_data`: dict（用于Student Prior）
- **输出**:
  - `logits`: [B, 2]
  - `loss_components`: dict（包含各个损失组件）

## 训练流程

1. **数据加载**: 从CSV文件加载多模态数据
2. **特征提取**: 使用ResNet50提取图像特征
3. **模型前向**: Bio-COT模型处理多模态特征
4. **损失计算**: 
   - 分类损失（Focal Loss）
   - Sinkhorn OT损失
   - 反事实一致性损失（单中心数据集不使用）
   - 对抗损失（单中心数据集不使用）
5. **反向传播**: 更新模型参数

## 输出文件

### 训练结果
- **结果JSON**: `results_multimodal/results_bio_cot_multimodal_*.json`
- **训练曲线**: `results_multimodal/training_curves_bio_cot_multimodal_*.png`
- **混淆矩阵**: `results_multimodal/confusion_matrix_bio_cot_multimodal_*.png`
- **模型检查点**: `checkpoints_multimodal/best_bio_cot_multimodal_*.pth`

### 对比结果
- **对比图**: `results_multimodal/comparison_oct_vs_multimodal.png`
- **对比JSON**: `results_multimodal/comparison_results.json`

## 运行命令

### 训练多模态模型
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics
python experiments/exp_xiangyang/train_bio_cot_multimodal_xiangyang.py
```

### 比较单模态和多模态结果
```bash
python experiments/exp_xiangyang/compare_oct_vs_multimodal.py
```

## 预期结果

多模态方法应该比单模态方法有更好的性能，因为：
1. **更多信息**: Colposcopy提供形态学信息，Clinical提供病因学信息
2. **互补性**: 不同模态提供互补的诊断信息
3. **Bio-COT优势**: Bio-COT的多模态融合机制能够有效利用这些信息

## 注意事项

1. **单中心数据集**: 不使用反事实干预（`use_counterfactual=False`）
2. **类别不平衡**: 训练集阳性样本较少（27/230），使用Focal Loss
3. **路径问题**: CSV中的路径指向`_multimodal`目录，脚本会自动处理

## 下一步

1. 等待训练完成
2. 运行对比脚本
3. 分析结果差异
4. 可视化多模态融合的效果

