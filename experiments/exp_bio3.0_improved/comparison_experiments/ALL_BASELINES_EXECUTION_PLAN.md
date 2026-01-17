# 所有对比实验顺序执行计划

## 📋 实验列表（共9个）

### SOTA方法（3个）- 已运行或正在运行
1. ✅ **MedCLIP** - 医学领域专用CLIP模型
2. ✅ **ConVIRT** - 对比学习的医学Vision-Representation Transformer
3. ✅ **mmFormer** - 多模态医学Transformer

### 简单Baseline（3个）- 待执行
4. ⏳ **Simple Fusion** - 特征拼接 + MLP
5. ⏳ **Standard CLIP** - 标准CLIP方法
6. ⏳ **ViT + Clinical Fusion** - ViT特征 + 临床特征融合

### Backbone对比（3个）- 待执行
7. ⏳ **CNN Baseline** - CNN编码器 + 多模态融合
8. ⏳ **Swin-T Baseline** - Swin-T编码器 + 多模态融合
9. ⏳ **VMamba Baseline** - VMamba编码器 + 多模态融合

## 🎯 执行策略

### 顺序执行
- ✅ 一次只运行一个实验，避免显存冲突
- ✅ 每个实验完成后等待15秒，确保GPU显存释放
- ✅ 支持跳过已完成的实验（`--skip_completed`）
- ✅ 支持跳过SOTA方法（`--skip_sota`）

### 实验配置
- **运行次数**: 每个方法3次
- **训练轮数**: 50 epochs
- **Batch Size**: 8（优化后的值）
- **设备**: cuda:1
- **显存优化**: 已启用

## 📊 当前执行状态

### 已启动
- ✅ 所有对比实验顺序执行脚本已启动
- ✅ 使用`--skip_sota`跳过SOTA方法（如果已运行）
- ✅ 将按顺序执行：Simple Fusion → Standard CLIP → ViT+Clinical → CNN → Swin-T → VMamba

## ⏱️ 预计完成时间

- 每个方法：3次运行 × 50 epochs ≈ 数小时
- 6个待执行方法：**预计12-18小时**（取决于每个方法的训练速度）

## 🔍 监控命令

```bash
# 查看主进程
ps aux | grep "run_all_baselines_sequential" | grep -v grep

# 查看当前运行的实验
ps aux | grep "train_" | grep -v grep

# 查看主日志
tail -f comparison_experiments/logs/all_baselines_sequential_*.log

# 查看各实验的详细日志
tail -f comparison_experiments/logs/*_sequential_*.log

# 查看GPU使用情况
nvidia-smi --id=1

# 查看已完成的实验结果
ls -lh comparison_experiments/results/baseline_*/results/all_results.json
```

## 📝 结果保存位置

所有结果将保存在：
```
comparison_experiments/results/
├── baseline_simple_fusion/results/
├── baseline_standard_clip/results/
├── baseline_vit_clinical_fusion/results/
├── baseline_cnn/results/
├── baseline_swin_t/results/
└── baseline_vmamba/results/
```

## ✅ 优势

1. **避免显存不足** - 顺序执行，一次一个
2. **充分利用显存** - 显存充足时可以正常运行
3. **清晰的进度** - 可以清楚看到每个实验的进度
4. **错误隔离** - 一个实验失败不影响其他实验
5. **完整对比** - 包含所有9个baseline方法

---

**所有对比实验现在按顺序执行中！** 🎉

