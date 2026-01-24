# 添加更多Baseline实验

## 📊 当前状态

### ✅ 已实现的Baseline（3个）
1. **Simple Fusion** - 特征拼接 + MLP
2. **Standard CLIP** - 标准CLIP方法
3. **ViT + Clinical Fusion** - ViT特征 + 临床特征融合

### ⚠️ 需要添加的Baseline（根据实验设计文档）

根据`COMPLETE_EXPERIMENT_DESIGN.md`，应该有以下Baseline：

1. **CNN Baseline** ⭐⭐⭐
   - 传统深度学习方法
   - 文件位置：可能在`experiments/baseline/train_cnn_baseline.py`
   - 需要：检查是否存在，如果存在则集成到对比实验中

2. **Swin-T Baseline** ⭐⭐⭐
   - 现代Transformer方法
   - 文件位置：可能在`experiments/baseline/train_swin_baseline.py`
   - 预期AUC: 80-85%
   - 需要：检查是否存在，如果存在则集成

3. **VMamba Baseline** ⭐⭐⭐
   - 不同backbone对比
   - 文件位置：可能在`experiments/baseline/train_vmamba_baseline.py`
   - 预期AUC: 80-85%
   - 需要：检查是否存在，如果存在则集成

## 🎯 建议的完整Baseline列表

为了符合MICCAI发表标准，建议包含以下6个Baseline：

1. ✅ **Simple Fusion** - 最简单方法
2. ✅ **Standard CLIP** - 标准方法
3. ✅ **ViT + Clinical Fusion** - 无OT方法
4. ⚠️ **CNN Baseline** - 传统方法（需要添加）
5. ⚠️ **Swin-T Baseline** - 现代Transformer（需要添加）
6. ⚠️ **VMamba Baseline** - 不同backbone（需要添加）

## 📝 下一步行动

1. 检查现有baseline脚本是否存在
2. 如果存在，创建适配脚本集成到对比实验框架
3. 如果不存在，需要创建这些baseline脚本
4. 运行所有baseline实验
5. 生成包含所有baseline的可视化结果

