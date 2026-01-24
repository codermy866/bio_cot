# 消融实验快速开始指南

## 📁 文件夹结构

```
ablation_studies/
├── README.md                    # 详细说明文档
├── QUICKSTART.md               # 本文件（快速开始）
├── run_ablation.py             # 统一运行脚本
├── compare_results.py          # 结果对比脚本
├── baseline/                   # 基础模型（移除所有高级模块）
│   └── config.py
├── w/o_visual_notes/           # 移除视觉笔记
│   └── config.py
├── w/o_adaptive_gating/        # 移除自适应门控
│   └── config.py
├── w/o_alignment_loss/         # 移除对齐损失
│   └── config.py
├── w/o_ot_loss/                # 移除OT损失
│   └── config.py
├── w/o_dual_head/              # 移除双头解耦
│   └── config.py
└── w/o_cross_attn/             # 移除Cross-Attention
    └── config.py
```

## 🚀 快速开始

### 1. 查看所有可用实验

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_3.1
python ablation_studies/run_ablation.py --list
```

### 2. 运行单个实验

```bash
# 运行 Baseline 实验
python ablation_studies/run_ablation.py --experiment baseline

# 运行移除 Visual Notes 的实验
python ablation_studies/run_ablation.py --experiment w/o_visual_notes

# 指定GPU
python ablation_studies/run_ablation.py --experiment baseline --gpu 0
```

### 3. 运行所有实验（按顺序）

```bash
python ablation_studies/run_ablation.py --all
```

### 4. 对比所有实验结果

```bash
python ablation_studies/compare_results.py
```

这将生成：
- 控制台对比表格
- `comparison_results.csv` 文件

## 📊 实验说明

| 实验ID | 实验名称 | 描述 |
|--------|---------|------|
| `baseline` | Baseline | 移除所有高级模块，仅保留基础分类 |
| `w/o_visual_notes` | w/o Visual Notes | 移除增强型视觉笔记模块 |
| `w/o_adaptive_gating` | w/o Adaptive Gating | 移除自适应模态门控 |
| `w/o_alignment_loss` | w/o Alignment Loss | 移除语义-视觉对齐损失 |
| `w/o_ot_loss` | w/o OT Loss | 移除 Optimal Transport 损失 |
| `w/o_dual_head` | w/o Dual Head | 移除双头因果解耦模块 |
| `w/o_cross_attn` | w/o Cross-Attention | 移除 Cross-Attention 机制 |

## ⚙️ 手动运行（不使用统一脚本）

如果你想手动运行某个实验：

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_3.1
python training/train_bio_cot_v3.py --config ablation_studies/baseline/config.py
```

## 📈 结果查看

每个实验的结果保存在各自的目录中：

```
ablation_studies/
├── baseline/
│   ├── logs/          # 训练日志
│   ├── checkpoints/   # 模型检查点
│   └── results/       # 其他结果
├── w/o_visual_notes/
│   └── ...
```

## 🔍 注意事项

1. **实验顺序**：建议先运行 `baseline` 建立基准，然后按重要性顺序运行其他实验
2. **训练时间**：每个实验大约需要 2-4 小时（取决于GPU）
3. **存储空间**：确保有足够的磁盘空间存储检查点和日志
4. **随机种子**：所有实验使用相同的随机种子（42）确保可复现性

## 🐛 故障排除

### 问题1: 找不到配置文件
```bash
# 确保在正确的目录
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_3.1
```

### 问题2: 导入错误
```bash
# 确保 Python 路径正确
export PYTHONPATH=/data2/hmy/VLM_Caus_Rm_Mics:$PYTHONPATH
```

### 问题3: GPU 内存不足
```bash
# 减小 batch_size（在各自的 config.py 中修改）
# 或使用更小的模型
```

