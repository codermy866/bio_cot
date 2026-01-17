# 实验文件夹结构说明

> **对比实验和消融实验的文件夹组织结构**

---

## 📁 文件夹结构总览

```
exp_bio3.0_improved/
├── comparison_experiments/          # 对比实验文件夹
│   ├── README.md                    # 对比实验说明
│   ├── __init__.py                  # Python包初始化
│   ├── baselines/                   # Baseline方法实现
│   │   ├── simple_fusion/          # Simple Fusion Baseline
│   │   ├── standard_clip/           # Standard CLIP Baseline
│   │   ├── vit_clinical_fusion/     # ViT + Clinical Fusion
│   │   ├── cnn_baseline/            # CNN Baseline
│   │   ├── swin_t_baseline/         # Swin-T Baseline
│   │   └── vmamba_baseline/         # VMamba Baseline
│   ├── results/                     # 实验结果
│   │   ├── baseline_simple_fusion/
│   │   ├── baseline_standard_clip/
│   │   └── ...
│   └── scripts/                     # 运行脚本
│       ├── run_all_baselines.sh
│       └── analyze_baseline_results.py
│
└── ablation_experiments/            # 消融实验文件夹
    ├── README.md                    # 消融实验说明
    ├── __init__.py                  # Python包初始化
    ├── ablations/                   # 消融实验实现
    │   ├── no_knowledge_notes/     # Knowledge Notes消融
    │   ├── no_visual_notes/        # Visual Notes消融
    │   ├── no_ot/                  # Sinkhorn OT消融
    │   ├── no_dual_head/           # Dual-Head消融
    │   └── no_knowledge_visual/    # Knowledge + Visual Notes消融
    ├── results/                     # 实验结果
    │   ├── ablation_no_knowledge_notes/
    │   ├── ablation_no_visual_notes/
    │   └── ...
    └── scripts/                     # 运行脚本
        ├── run_all_ablations.sh
        └── analyze_ablation_results.py
```

---

## 🎯 对比实验文件夹 (comparison_experiments)

### 用途
存放所有Baseline对比实验的代码和结果

### 子文件夹说明

#### `baselines/` - Baseline方法实现
每个Baseline方法有独立的子文件夹：
- `simple_fusion/` - Simple Fusion Baseline
- `standard_clip/` - Standard CLIP Baseline
- `vit_clinical_fusion/` - ViT + Clinical Fusion
- `cnn_baseline/` - CNN Baseline
- `swin_t_baseline/` - Swin-T Baseline
- `vmamba_baseline/` - VMamba Baseline

#### `results/` - 实验结果
每个Baseline实验的结果保存在独立的子文件夹中：
- `baseline_simple_fusion/` - Simple Fusion结果
- `baseline_standard_clip/` - Standard CLIP结果
- `baseline_vit_clinical_fusion/` - ViT + Clinical结果
- ...

每个结果文件夹包含：
- `results/` - 每次运行的结果（JSON格式）
- `results/statistics.json` - 统计信息
- `results/all_results.csv` - 所有结果的CSV汇总
- `checkpoints/` - 模型检查点
- `logs/` - 训练日志

#### `scripts/` - 运行脚本
- `run_all_baselines.sh` - 运行所有Baseline实验
- `analyze_baseline_results.py` - 分析Baseline结果

---

## 🔬 消融实验文件夹 (ablation_experiments)

### 用途
存放所有消融实验的代码和结果

### 子文件夹说明

#### `ablations/` - 消融实验实现
每个消融实验有独立的子文件夹：
- `no_knowledge_notes/` - Knowledge Notes消融
- `no_visual_notes/` - Visual Notes消融
- `no_ot/` - Sinkhorn OT消融
- `no_dual_head/` - Dual-Head消融
- `no_knowledge_visual/` - Knowledge + Visual Notes消融

#### `results/` - 实验结果
每个消融实验的结果保存在独立的子文件夹中：
- `ablation_no_knowledge_notes/` - Knowledge Notes消融结果
- `ablation_no_visual_notes/` - Visual Notes消融结果
- `ablation_no_ot/` - Sinkhorn OT消融结果
- ...

每个结果文件夹包含：
- `results/` - 每次运行的结果（JSON格式）
- `results/statistics.json` - 统计信息
- `results/all_results.csv` - 所有结果的CSV汇总
- `checkpoints/` - 模型检查点
- `logs/` - 训练日志

#### `scripts/` - 运行脚本
- `run_all_ablations.sh` - 运行所有消融实验
- `analyze_ablation_results.py` - 分析消融实验结果

---

## 🚀 使用方法

### 运行所有对比实验

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0_improved
bash comparison_experiments/scripts/run_all_baselines.sh
```

### 运行所有消融实验

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0_improved
bash ablation_experiments/scripts/run_all_ablations.sh
```

### 运行单个实验

```bash
# 运行单个Baseline
python comparison_experiments/baselines/simple_fusion/train_simple_fusion.py \
    --experiment_name baseline_simple_fusion \
    --num_runs 5

# 运行单个消融实验
python ablation_experiments/ablations/no_knowledge_notes/train_ablation_knowledge_notes.py \
    --experiment_name ablation_no_knowledge_notes \
    --num_runs 5
```

---

## 📊 结果查看

### 查看对比实验结果

```bash
# 查看所有Baseline结果
ls comparison_experiments/results/

# 查看特定Baseline的统计信息
cat comparison_experiments/results/baseline_simple_fusion/results/statistics.json

# 查看所有结果的CSV汇总
cat comparison_experiments/results/*/results/all_results.csv
```

### 查看消融实验结果

```bash
# 查看所有消融实验结果
ls ablation_experiments/results/

# 查看特定消融实验的统计信息
cat ablation_experiments/results/ablation_no_knowledge_notes/results/statistics.json

# 查看所有结果的CSV汇总
cat ablation_experiments/results/*/results/all_results.csv
```

---

## ✅ 文件夹完整性检查

### 对比实验文件夹
- [x] `comparison_experiments/` 主文件夹已创建
- [x] `baselines/` 子文件夹已创建（6个Baseline）
- [x] `results/` 子文件夹已创建
- [x] `scripts/` 子文件夹已创建
- [x] `README.md` 说明文档已创建
- [x] `__init__.py` 已创建

### 消融实验文件夹
- [x] `ablation_experiments/` 主文件夹已创建
- [x] `ablations/` 子文件夹已创建（5个消融实验）
- [x] `results/` 子文件夹已创建
- [x] `scripts/` 子文件夹已创建
- [x] `README.md` 说明文档已创建
- [x] `__init__.py` 已创建

---

## 📝 下一步工作

### 对比实验
1. [ ] 在 `baselines/simple_fusion/` 中实现Simple Fusion Baseline
2. [ ] 在 `baselines/standard_clip/` 中实现Standard CLIP Baseline
3. [ ] 在 `baselines/vit_clinical_fusion/` 中实现ViT + Clinical Fusion
4. [ ] 完善 `scripts/analyze_baseline_results.py` 分析脚本

### 消融实验
1. [ ] 在 `ablations/no_knowledge_notes/` 中实现Knowledge Notes消融
2. [ ] 在 `ablations/no_visual_notes/` 中实现Visual Notes消融
3. [ ] 在 `ablations/no_ot/` 中实现Sinkhorn OT消融
4. [ ] 在 `ablations/no_dual_head/` 中实现Dual-Head消融
5. [ ] 在 `ablations/no_knowledge_visual/` 中实现组合消融
6. [ ] 完善 `scripts/analyze_ablation_results.py` 分析脚本

---

## 🔗 相关文档

- `COMPLETE_EXPERIMENT_DESIGN.md` - 完整实验设计
- `EXPERIMENT_EXECUTION_GUIDE.md` - 实验执行指南
- `EXPERIMENT_CHECKLIST.md` - 实验检查清单
- `comparison_experiments/README.md` - 对比实验说明
- `ablation_experiments/README.md` - 消融实验说明

---

**文档版本**: v1.0  
**创建日期**: 2025-01-15

