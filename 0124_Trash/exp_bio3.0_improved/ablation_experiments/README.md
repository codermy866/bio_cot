# 消融实验文件夹 (Ablation Experiments)

> **存放所有消融实验的代码和结果**

---

## 📁 文件夹结构

```
ablation_experiments/
├── README.md                    # 本文件
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

## 🎯 实验列表

### Ablation 1: Knowledge Notes消融 ⭐⭐⭐⭐⭐

**移除模块**: Knowledge Notes

**方法**: 使用原始临床数据（而非Knowledge Note Embeddings）

**文件位置**: `ablations/no_knowledge_notes/train_ablation_knowledge_notes.py`

**预期结果**: AUC 76-82% (vs Full Model 85%+)

**预期下降**: -3~-5%

**目的**: 证明Knowledge Notes的有效性

---

### Ablation 2: Visual Notes消融 ⭐⭐⭐⭐⭐

**移除模块**: Visual Notes

**方法**: 使用全局特征（而非过滤后的特征）

**文件位置**: `ablations/no_visual_notes/train_ablation_visual_notes.py`

**预期结果**: AUC 81-85% (vs Full Model 85%+)

**预期下降**: -2~-4%

**目的**: 证明Visual Notes的有效性

---

### Ablation 3: Sinkhorn OT消融 ⭐⭐⭐⭐⭐

**移除模块**: Sinkhorn OT

**方法**: 使用MSE/KL散度替代最优传输

**文件位置**: `ablations/no_ot/train_ablation_ot.py`

**预期结果**: AUC 82-86% (vs Full Model 85%+)

**预期下降**: -2~-3%

**目的**: 证明Sinkhorn OT的有效性

---

### Ablation 4: Dual-Head消融 ⭐⭐⭐⭐⭐

**移除模块**: Dual-Head结构

**方法**: 使用单头编码器

**文件位置**: `ablations/no_dual_head/train_ablation_dual_head.py`

**预期结果**: AUC 83-87% (vs Full Model 85%+)

**预期下降**: -1~-2%

**目的**: 证明因果解耦的有效性

---

### Ablation 5: Knowledge + Visual Notes消融 ⭐⭐⭐⭐

**移除模块**: Knowledge Notes + Visual Notes

**方法**: 同时移除两个模块

**文件位置**: `ablations/no_knowledge_visual/train_ablation_knowledge_visual.py`

**预期结果**: AUC 78-82% (vs Full Model 85%+)

**预期下降**: -5~-7%

**目的**: 证明模块组合的协同效应

---

## 🚀 快速开始

### 运行所有消融实验

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0_improved
bash ablation_experiments/scripts/run_all_ablations.sh
```

### 运行单个消融实验

```bash
# Knowledge Notes消融
python ablation_experiments/ablations/no_knowledge_notes/train_ablation_knowledge_notes.py \
    --experiment_name ablation_no_knowledge_notes \
    --num_runs 5

# Visual Notes消融
python ablation_experiments/ablations/no_visual_notes/train_ablation_visual_notes.py \
    --experiment_name ablation_no_visual_notes \
    --num_runs 5
```

---

## 📊 结果分析

### 查看结果

```bash
# 查看所有消融实验结果
ls ablation_experiments/results/

# 查看统计信息
cat ablation_experiments/results/ablation_no_knowledge_notes/results/statistics.json
```

### 生成消融实验表格

```bash
python ablation_experiments/scripts/analyze_ablation_results.py \
    --results_dir ablation_experiments/results \
    --output_dir ablation_experiments/results/analysis \
    --full_model_path ../comparison_experiments/results/full_model
```

---

## ✅ 实验完成检查

- [ ] Knowledge Notes消融 (5次运行)
- [ ] Visual Notes消融 (5次运行)
- [ ] Sinkhorn OT消融 (5次运行)
- [ ] Dual-Head消融 (5次运行)
- [ ] Knowledge + Visual Notes消融 (5次运行)

**完成标准**: 所有消融实验运行完成，每个模块都有显著贡献 (p < 0.05)

---

## 📈 预期结果

### 与Full Model对比

| 消融实验 | 移除模块 | 预期AUC | vs Full Model | 预期下降 |
|---------|---------|---------|---------------|---------|
| Full Model | - | 85%+ | - | - |
| Ablation 1 | Knowledge Notes | 76-82% | -3~-5% | ⬇️ |
| Ablation 2 | Visual Notes | 81-85% | -2~-4% | ⬇️ |
| Ablation 3 | Sinkhorn OT | 82-86% | -2~-3% | ⬇️ |
| Ablation 4 | Dual-Head | 83-87% | -1~-2% | ⬇️ |
| Ablation 5 | Knowledge+Visual | 78-82% | -5~-7% | ⬇️ |

---

**最后更新**: 2025-01-15

