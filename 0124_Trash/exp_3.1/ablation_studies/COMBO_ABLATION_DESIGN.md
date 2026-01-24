# 组合消融实验设计方案

## 📊 实验总览

本消融实验方案包含 **17 个实验**，分为两类：

### 1️⃣ 单模块消融实验（7个）
移除单个模块，测试该模块的独立贡献度。

| 实验名称 | 描述 | 保留的模块 |
|---------|------|-----------|
| `baseline` | 移除所有高级模块 | 仅基础分类器 |
| `w/o_visual_notes` | 移除 Visual Notes | 其他全部保留 |
| `w/o_alignment_loss` | 移除 Alignment Loss | 其他全部保留 |
| `w/o_ot_loss` | 移除 OT Loss | 其他全部保留 |
| `w/o_dual_head` | 移除 Dual Head | 其他全部保留 |
| `w/o_adaptive_gating` | 移除 Adaptive Gating | 其他全部保留 |
| `w/o_cross_attn` | 移除 Cross-Attention | 其他全部保留 |

### 2️⃣ 组合消融实验（10个）
**只保留两个模块，其他全部关闭**，测试两个模块的协同效应。

| 实验名称 | 保留的模块 | 测试的协同效应 |
|---------|-----------|---------------|
| `only_visual_notes_align` | Visual Notes + Alignment Loss | 视觉增强与语义对齐的协同 |
| `only_visual_notes_ot` | Visual Notes + OT Loss | 视觉增强与分布匹配的协同 |
| `only_visual_notes_dual` | Visual Notes + Dual Head | 视觉增强与因果解耦的协同 |
| `only_visual_notes_crossattn` | Visual Notes + Cross-Attention | 视觉增强与跨模态融合的协同 |
| `only_align_ot` | Alignment Loss + OT Loss | 对齐损失与分布匹配的协同 |
| `only_align_dual` | Alignment Loss + Dual Head | 对齐损失与因果解耦的协同 |
| `only_align_crossattn` | Alignment Loss + Cross-Attention | 对齐损失与跨模态融合的协同 |
| `only_ot_dual` | OT Loss + Dual Head | 分布匹配与因果解耦的协同 |
| `only_ot_crossattn` | OT Loss + Cross-Attention | 分布匹配与跨模态融合的协同 |
| `only_dual_crossattn` | Dual Head + Cross-Attention | 因果解耦与跨模态融合的协同 |

## 🎯 设计理念

### 为什么需要组合消融实验？

1. **协同效应检测**：两个模块组合可能产生 `1+1>2` 的效果
2. **冗余性分析**：如果两个模块单独效果差，但组合效果好，说明它们互补
3. **模块依赖关系**：识别哪些模块必须配合使用才能发挥效果

### 实验执行顺序

```
单模块消融（7个） → 组合消融（10个）
```

**建议执行策略**：
- 先完成所有单模块消融，了解每个模块的独立贡献
- 再执行组合消融，找出最优的模块组合
- 最后对比分析，确定最终模型配置

## 📁 文件结构

```
ablation_studies/
├── baseline/
├── w/o_visual_notes/
├── w/o_alignment_loss/
├── w/o_ot_loss/
├── w/o_dual_head/
├── w/o_adaptive_gating/
├── w/o_cross_attn/
├── only_visual_notes_align/
├── only_visual_notes_ot/
├── only_visual_notes_dual/
├── only_visual_notes_crossattn/
├── only_align_ot/
├── only_align_dual/
├── only_align_crossattn/
├── only_ot_dual/
├── only_ot_crossattn/
├── only_dual_crossattn/
└── auto_sequential_ablation.py  # 自动执行脚本
```

## 🚀 使用方法

### 自动执行所有实验

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_3.1
python ablation_studies/auto_sequential_ablation.py \
    --auto_gpu \
    --gpus "0,1" \
    --gpu_strategy random \
    --min_free_mem_mb 12000 \
    --max_util 60 \
    --interval 120 \
    --start_from baseline
```

### 从组合实验开始执行

```bash
python ablation_studies/auto_sequential_ablation.py \
    --auto_gpu \
    --gpus "0,1" \
    --start_from only_visual_notes_align
```

## 📈 结果分析建议

### 1. 单模块贡献度排序
根据单模块消融结果，按性能下降幅度排序：
- 下降最大 → 该模块最重要
- 下降最小 → 该模块可考虑移除

### 2. 组合协同效应分析
对比组合实验与单模块实验：
- 如果 `only_A_B` 的性能 > `w/o_A` 且 > `w/o_B` → A和B有协同效应
- 如果 `only_A_B` 的性能 ≈ `w/o_A` 或 ≈ `w/o_B` → A和B存在冗余

### 3. 最优配置推荐
综合所有实验结果，推荐：
- **核心模块**：贡献度最高的模块
- **协同模块对**：协同效应最强的组合
- **可移除模块**：贡献度低且无协同效应的模块

## ⚙️ 实验配置

所有实验统一使用：
- **Epochs**: 20（消融实验快速验证）
- **Batch Size**: 48（与主实验保持一致）
- **其他超参数**: 继承自 `BioCOT_v3_Config`

## 📝 注意事项

1. **实验时间**：17个实验 × 20 epochs ≈ 需要较长时间，建议后台运行
2. **GPU资源**：脚本会自动选择空闲GPU，不会打断正在运行的训练
3. **日志监控**：每个实验都有独立的日志文件，便于追踪进度
4. **结果保存**：所有实验结果保存在各自的 `results/` 目录下

