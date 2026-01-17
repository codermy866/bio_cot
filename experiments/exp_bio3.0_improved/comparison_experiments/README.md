# 对比实验文件夹 (Comparison Experiments)

> **存放所有Baseline对比实验的代码和结果**

---

## 📁 文件夹结构

```
comparison_experiments/
├── README.md                    # 本文件
├── __init__.py                  # Python包初始化
├── baselines/                   # Baseline方法实现
│   ├── simple_fusion/          # Simple Fusion Baseline
│   ├── standard_clip/          # Standard CLIP Baseline
│   ├── vit_clinical_fusion/    # ViT + Clinical Fusion
│   ├── cnn_baseline/           # CNN Baseline
│   ├── swin_t_baseline/        # Swin-T Baseline
│   └── vmamba_baseline/        # VMamba Baseline
├── results/                     # 实验结果
│   ├── baseline_simple_fusion/
│   ├── baseline_standard_clip/
│   └── ...
└── scripts/                     # 运行脚本
    ├── run_all_baselines.sh
    └── analyze_baseline_results.py
```

---

## 🎯 实验列表

### Baseline 1: Simple Fusion ⭐⭐⭐⭐⭐

**方法**: 最简单的多模态融合 - 特征拼接 + MLP分类器

**文件位置**: `baselines/simple_fusion/train_simple_fusion.py`

**预期结果**: AUC 70-75%

**目的**: 证明注意力机制的必要性

---

### Baseline 2: Standard CLIP ⭐⭐⭐⭐⭐

**方法**: 标准CLIP方法 - 无Knowledge Notes，无Visual Notes

**文件位置**: `baselines/standard_clip/train_standard_clip.py`

**预期结果**: AUC 75-80%

**目的**: 证明Knowledge Notes和Visual Notes的有效性

---

### Baseline 3: ViT + Clinical Fusion ⭐⭐⭐⭐

**方法**: ViT特征 + 临床特征简单融合 - 无最优传输

**文件位置**: `baselines/vit_clinical_fusion/train_vit_clinical.py`

**预期结果**: AUC 72-78%

**目的**: 证明Sinkhorn OT的有效性

---

### Baseline 4: CNN Baseline ⭐⭐⭐

**方法**: CNN编码器 + 多模态融合

**文件位置**: `baselines/cnn_baseline/` (已有实现)

**预期结果**: AUC 70-75%

**目的**: 传统深度学习方法对比

---

### Baseline 5: Swin-T Baseline ⭐⭐⭐

**方法**: Swin-T编码器 + 多模态融合

**文件位置**: `baselines/swin_t_baseline/` (已有实现)

**预期结果**: AUC 80-85%

**目的**: 现代Transformer方法对比

---

### Baseline 6: VMamba Baseline ⭐⭐⭐

**方法**: VMamba编码器 + 多模态融合

**文件位置**: `baselines/vmamba_baseline/` (已有实现)

**预期结果**: AUC 80-85%

**目的**: 不同backbone对比

---

## 🚀 快速开始

### 运行所有Baseline实验

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0_improved
bash comparison_experiments/scripts/run_all_baselines.sh
```

### 运行单个Baseline

```bash
# Simple Fusion
python comparison_experiments/baselines/simple_fusion/train_simple_fusion.py \
    --experiment_name baseline_simple_fusion \
    --num_runs 5

# Standard CLIP
python comparison_experiments/baselines/standard_clip/train_standard_clip.py \
    --experiment_name baseline_standard_clip \
    --num_runs 5
```

---

## 📊 结果分析

### 查看结果

```bash
# 查看所有Baseline结果
ls comparison_experiments/results/

# 查看统计信息
cat comparison_experiments/results/baseline_simple_fusion/results/statistics.json
```

### 生成对比表格

```bash
python comparison_experiments/scripts/analyze_baseline_results.py \
    --results_dir comparison_experiments/results \
    --output_dir comparison_experiments/results/analysis
```

---

## ✅ 实验完成检查

- [ ] Simple Fusion Baseline (5次运行)
- [ ] Standard CLIP Baseline (5次运行)
- [ ] ViT + Clinical Fusion (5次运行)
- [ ] CNN Baseline (已有或重新运行)
- [ ] Swin-T Baseline (已有或重新运行)
- [ ] VMamba Baseline (已有或重新运行)

**完成标准**: 所有Baseline运行完成，结果优于所有Baseline (p < 0.05)

---

**最后更新**: 2025-01-15

