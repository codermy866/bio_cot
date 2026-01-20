# Bio-COT 3.1 消融实验 (Ablation Studies)

本目录包含 Bio-COT 3.1 的完整消融实验，用于评估各个组件的贡献。

## 实验列表

### 1. **Baseline** (`baseline/`)
- **描述**: 基础模型，移除所有高级模块
- **配置**: 
  - `use_visual_notes = False`
  - `use_ot = False`
  - `use_dual = False`
  - `use_cross_attn = False`
  - `lambda_align = 0.0`

### 2. **w/o Visual Notes** (`w/o_visual_notes/`)
- **描述**: 移除增强型视觉笔记模块（Cross-Attention）
- **配置**: `use_visual_notes = False`

### 3. **w/o Adaptive Gating** (`w/o_adaptive_gating/`)
- **描述**: 移除自适应模态门控（AMG），使用固定权重融合
- **配置**: 在模型中禁用 `AdaptiveModalityGating`

### 4. **w/o Alignment Loss** (`w/o_alignment_loss/`)
- **描述**: 移除语义-视觉对齐损失（L_align）
- **配置**: `lambda_align = 0.0`

### 5. **w/o OT Loss** (`w/o_ot_loss/`)
- **描述**: 移除 Optimal Transport 损失
- **配置**: `use_ot = False`, `lambda_ot = 0.0`

### 6. **w/o Dual Head** (`w/o_dual_head/`)
- **描述**: 移除双头因果解耦模块
- **配置**: `use_dual = False`, `lambda_consist = 0.0`, `lambda_adv = 0.0`

### 7. **w/o Cross-Attention** (`w/o_cross_attn/`)
- **描述**: 移除 Cross-Attention 机制（使用简单点积注意力）
- **配置**: `use_cross_attn = False`

## 运行方式

### 方法1: 使用统一运行脚本
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_3.1
python ablation_studies/run_ablation.py --experiment baseline
```

### 方法2: 单独运行每个实验
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_3.1/ablation_studies/baseline
python ../../training/train_bio_cot_v3.py --config config.py
```

## 结果对比

训练完成后，使用 `compare_results.py` 脚本对比所有实验的结果：

```bash
python ablation_studies/compare_results.py
```

这将生成一个对比表格，包含：
- 验证集 AUC
- 验证集 F1-Score
- 验证集 Recall@1 (对齐召回率)
- 训练时间
- 模型参数量

## 实验顺序建议

1. **Baseline** - 建立基准性能
2. **w/o Visual Notes** - 评估视觉笔记的重要性
3. **w/o Alignment Loss** - 评估对齐损失的作用
4. **w/o OT Loss** - 评估 OT 损失的贡献
5. **w/o Dual Head** - 评估因果解耦的必要性
6. **w/o Adaptive Gating** - 评估自适应融合的优势
7. **w/o Cross-Attention** - 评估 Cross-Attention 的改进

## 注意事项

- 每个实验使用相同的随机种子（42）以确保可复现性
- 所有实验使用相同的训练/验证集划分
- 训练轮数统一为 100 epochs
- Batch Size 统一为 48

