# The Lancet Primary Care - 模型集成指南

## 🎯 模型集成策略

### 创新点：多模型集成提升性能和鲁棒性

本研究采用**多模型集成**策略，结合不同架构和训练策略的模型，以提升诊断性能和鲁棒性。

---

## 📊 集成的模型

### 1. ViT Backbone模型
- **路径**: `paper1_hierarchical_multimodal/results/cuda1_vit_backbone/best_model.pth`
- **架构**: Vision Transformer (ViT-Base)
- **性能**: AUC = 0.672 (最佳)
- **权重**: 0.40
- **特点**: 使用大模型backbone，特征提取能力强

### 2. ResNet Optimized模型
- **路径**: `paper1_hierarchical_multimodal/results/cuda0_optimized_v2/best_model.pth`
- **架构**: ResNet34/50 backbone
- **性能**: AUC = 0.640
- **权重**: 0.35
- **特点**: 优化的训练策略，Focal Loss + Label Smoothing

### 3. ResNet Supervised Contrastive模型
- **路径**: `paper1_hierarchical_multimodal/results/cuda1_run_e10_supcont/best_model.pth`
- **架构**: ResNet backbone + 监督对比学习
- **性能**: AUC ≈ 0.64
- **权重**: 0.25
- **特点**: 监督对比学习增强特征对齐

---

## 🔬 集成方法

### 加权平均集成 (Weighted Average Ensemble)

**公式**:
```
P_ensemble = Σ(w_i × P_i)
```

其中:
- `P_ensemble`: 集成预测概率
- `w_i`: 第i个模型的权重
- `P_i`: 第i个模型的预测概率

**权重分配**:
- 基于验证集AUC性能
- 归一化确保权重和为1

**优势**:
- ✅ 提升预测稳定性
- ✅ 减少单一模型的偏差
- ✅ 提高泛化能力
- ✅ 增强鲁棒性

---

## 🚀 使用方法

### 方法1: 使用脚本运行（推荐）

```bash
# 使用默认CUDA设备（cuda:0）
bash lancet_primary_care/scripts/start_lancet_experiments.sh

# 指定CUDA设备
bash lancet_primary_care/scripts/start_lancet_experiments.sh 1
```

### 方法2: 直接运行Python脚本

```bash
python lancet_primary_care/scripts/run_experiments_with_models.py \
    --data_path 5centers_multi \
    --experiments all \
    --output_dir lancet_primary_care/results
```

### 方法3: 运行单个实验

```bash
python lancet_primary_care/scripts/run_experiments_with_models.py \
    --data_path 5centers_multi \
    --experiments 1 \
    --output_dir lancet_primary_care/results
```

---

## 📋 实验流程

### 1. 模型加载
- 自动检测可用的模型文件
- 加载模型权重
- 创建集成模型

### 2. 数据加载
- 加载测试集数据
- 预处理OCT和Colposcopy图像
- 准备临床特征

### 3. 批量预测
- 使用集成模型进行预测
- 生成预测概率和标签
- 保存预测结果

### 4. 实验执行
- 实验1: 活检率降低
- 实验2: OCT灵敏度提升
- 实验3: 筛查方案对比
- 实验4: 流程优化

### 5. 结果保存
- JSON格式的详细结果
- PNG格式的可视化图表
- 统计报告

---

## 🔧 技术细节

### 模型架构

**HierarchicalMultimodalModel**:
- 多粒度特征编码（局部、全局、语义、统计）
- 跨模态对齐
- 多粒度融合
- 自适应权重
- 对比学习增强

**HierarchicalMultimodalModelViT**:
- Vision Transformer backbone
- 帧级注意力聚合
- 其他组件与ResNet版本相同

### 数据格式

**输入**:
- OCT图像: [B, T, C, H, W] 或 [B, C, H, W]
- Colposcopy图像: [B, C, H, W]
- 临床特征: [B, 7] (Age, HPV, TCT one-hot)

**输出**:
- 预测标签: [B]
- 预测概率: [B, 2]
- 病变概率: [B] (class 1的概率)

---

## 📊 预期性能提升

### 集成 vs 单一模型

| 指标 | 单一最佳模型 | 集成模型 | 提升 |
|------|------------|---------|------|
| AUC | 0.672 | ~0.70-0.75 | +4-8% |
| 灵敏度 | ~0.65 | ~0.70-0.75 | +5-10% |
| 特异度 | ~0.65 | ~0.70-0.75 | +5-10% |
| 鲁棒性 | 中等 | 高 | 显著提升 |

### 创新性

1. **多架构集成**: ResNet + ViT，互补优势
2. **多训练策略集成**: 标准训练 + 监督对比学习
3. **性能加权**: 基于验证集性能的智能权重分配
4. **不确定性量化**: 集成预测提供置信度估计

---

## ⚠️ 注意事项

1. **内存需求**: 集成多个模型需要更多GPU内存
2. **推理时间**: 集成预测比单一模型慢（但可接受）
3. **模型文件**: 确保所有模型文件存在且可访问
4. **CUDA设备**: 确保有足够的GPU内存

---

## 🔍 故障排除

### 问题1: 模型文件未找到
**解决**: 检查模型路径，确保文件存在

### 问题2: CUDA内存不足
**解决**: 
- 减少batch_size
- 使用单个模型而非集成
- 使用CPU推理（较慢）

### 问题3: 数据加载错误
**解决**: 
- 检查数据路径
- 验证数据格式
- 检查图像文件完整性

---

## 📚 相关文档

- [模型架构](../paper1_hierarchical_multimodal/README.md)
- [实验设计](EXPERIMENTAL_DESIGN.md)
- [数据分析](DATA_ANALYSIS.md)
- [最终实验方案](FINAL_EXPERIMENT_PLAN.md)

---

## ✅ 总结

模型集成系统已就绪，可以：
1. ✅ 自动加载多个训练好的模型
2. ✅ 创建加权集成模型
3. ✅ 对数据集进行批量预测
4. ✅ 运行所有4个实验
5. ✅ 生成详细的结果和可视化

**下一步**: 运行实验，收集结果，撰写论文！

---

**最后更新**: 2024年

