# 所有Baseline优化完成总结

## ✅ 已完成的优化

### 1. 设备统一为cuda:1

已将所有baseline脚本的设备配置改为cuda:1：

| 脚本 | 设备配置 | 状态 |
|------|---------|------|
| simple_fusion/train_simple_fusion.py | ✅ cuda:1 | 已更新 |
| standard_clip/train_standard_clip.py | ✅ cuda:1 | 已更新 |
| vit_clinical_fusion/train_vit_clinical.py | ✅ cuda:1 | 已更新 |
| cnn_baseline/train_cnn.py | ✅ cuda:1 | 已更新 |
| swin_t_baseline/train_swin.py | ✅ cuda:1 | 已更新 |
| vmamba_baseline/train_vmamba.py | ✅ cuda:1 | 已更新 |

### 2. 显存优化

已将所有baseline的OCT帧数从20减少到10：

| 脚本 | OCT帧数 | 状态 |
|------|---------|------|
| simple_fusion | ✅ 10 (从20减少) | 已更新 |
| standard_clip | ✅ 10 (从20减少) | 已更新 |
| vit_clinical_fusion | ✅ 10 (从20减少) | 已更新 |
| cnn_baseline | ✅ 10 (从20减少) | 已更新 |
| swin_t_baseline | ✅ 10 (从20减少) | 已更新 |
| vmamba_baseline | ✅ 10 (从20减少) | 已更新 |

### 3. Batch Size优化

- 所有实验使用batch_size=8（通过命令行参数传递）
- 显存占用大幅降低

## 📊 完整实验列表（9个）

### SOTA方法（3个）
1. ✅ MedCLIP - 已运行/运行中
2. ✅ ConVIRT - 已运行/运行中
3. ✅ mmFormer - 已运行/运行中

### 简单Baseline（3个）- 正在执行
4. ⏳ **Simple Fusion** - 当前执行中
5. ⏳ Standard CLIP - 等待中
6. ⏳ ViT + Clinical Fusion - 等待中

### Backbone对比（3个）- 等待执行
7. ⏳ CNN Baseline - 等待中
8. ⏳ Swin-T Baseline - 等待中
9. ⏳ VMamba Baseline - 等待中

## 🎯 执行顺序

```
1. MedCLIP (SOTA) ✅
2. ConVIRT (SOTA) ✅
3. mmFormer (SOTA) ✅
4. Simple Fusion (当前执行中) ⏳
5. Standard CLIP (等待中) ⏸️
6. ViT + Clinical Fusion (等待中) ⏸️
7. CNN Baseline (等待中) ⏸️
8. Swin-T Baseline (等待中) ⏸️
9. VMamba Baseline (等待中) ⏸️
```

## 📝 实验配置

- **设备**: cuda:1 (统一)
- **Batch Size**: 8 (优化后)
- **OCT帧数**: 10 (优化后，从20减少)
- **运行次数**: 每个方法3次
- **训练轮数**: 50 epochs

## 🔍 监控命令

```bash
# 查看主进程
ps aux | grep "run_all_baselines_sequential" | grep -v grep

# 查看当前运行的实验
ps aux | grep "train_" | grep -v grep

# 查看主日志
tail -f comparison_experiments/logs/all_baselines_sequential_*.log

# 查看GPU 1使用情况
nvidia-smi --id=1
```

## ✅ 确认

**所有baseline脚本已优化完成，顺序执行正在进行中！** 🎉

- ✅ 设备统一为cuda:1
- ✅ 显存优化（帧数减少，batch size减少）
- ✅ 按顺序执行，避免显存冲突

