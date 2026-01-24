# Bio-COT 3.0 Improved 实验执行指南

> **完整的实验执行指南，确保所有实验按照MICCAI标准执行**

---

## 📋 快速开始

### 1. 环境准备

```bash
# 激活虚拟环境
cd /data2/hmy/VLM_Caus_Rm_Mics
source my_retfound/bin/activate

# 进入实验目录
cd experiments/exp_bio3.0_improved
```

### 2. 检查实验环境

```bash
# 检查GPU
nvidia-smi

# 检查Python环境
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 3. 创建实验目录结构

```bash
# 创建必要的目录
mkdir -p experiments/{baseline,ablation,full_model}
mkdir -p experiments/results
mkdir -p experiments/logs
mkdir -p utils
mkdir -p scripts
```

---

## 🚀 实验执行流程

### 阶段1: Baseline对比实验 (3-4周)

#### Step 1.1: Simple Fusion Baseline

**目标**: 最简单的多模态融合方法，作为最基础的baseline

**执行**:
```bash
# 运行Simple Fusion Baseline（5次运行）
python experiments/baseline/train_simple_fusion_baseline.py \
    --experiment_name baseline_simple_fusion \
    --num_runs 5 \
    --output_dir experiments/results
```

**预期结果**:
- AUC: 70-75%
- 运行时间: 每个run约2-3小时
- 总时间: 10-15小时（5次运行）

**验证**:
```bash
# 检查结果
ls experiments/results/baseline_simple_fusion/results/
cat experiments/results/baseline_simple_fusion/results/statistics.json
```

#### Step 1.2: Standard CLIP Baseline

**目标**: 标准CLIP方法，无Knowledge Notes和Visual Notes

**执行**:
```bash
python experiments/baseline/train_standard_clip_baseline.py \
    --experiment_name baseline_standard_clip \
    --num_runs 5 \
    --output_dir experiments/results
```

**预期结果**:
- AUC: 75-80%
- 运行时间: 每个run约2-3小时

#### Step 1.3: ViT + Clinical Fusion

**目标**: ViT特征与临床特征简单融合，无最优传输

**执行**:
```bash
python experiments/baseline/train_vit_clinical_fusion.py \
    --experiment_name baseline_vit_clinical_fusion \
    --num_runs 5 \
    --output_dir experiments/results
```

**预期结果**:
- AUC: 72-78%
- 运行时间: 每个run约2-3小时

#### Step 1.4: 批量运行所有Baseline

**使用自动化脚本**:
```bash
# 运行所有Baseline实验
bash scripts/run_all_baselines.sh
```

**或手动运行**:
```bash
# 并行运行（如果有多个GPU）
CUDA_VISIBLE_DEVICES=0 python experiments/baseline/train_simple_fusion_baseline.py --num_runs 5 &
CUDA_VISIBLE_DEVICES=1 python experiments/baseline/train_standard_clip_baseline.py --num_runs 5 &
CUDA_VISIBLE_DEVICES=2 python experiments/baseline/train_vit_clinical_fusion.py --num_runs 5 &
wait
```

---

### 阶段2: 消融实验 (2-3周)

#### Step 2.1: Knowledge Notes消融

**目标**: 验证Knowledge Notes模块的有效性

**执行**:
```bash
python experiments/ablation/train_ablation_knowledge_notes.py \
    --experiment_name ablation_no_knowledge_notes \
    --num_runs 5 \
    --output_dir experiments/results
```

**预期结果**:
- AUC: 76-82% (vs Full Model 85%+)
- 预期下降: -3~-5%

#### Step 2.2: Visual Notes消融

**目标**: 验证Visual Notes模块的有效性

**执行**:
```bash
python experiments/ablation/train_ablation_visual_notes.py \
    --experiment_name ablation_no_visual_notes \
    --num_runs 5 \
    --output_dir experiments/results
```

**预期结果**:
- AUC: 81-85% (vs Full Model 85%+)
- 预期下降: -2~-4%

#### Step 2.3: Sinkhorn OT消融

**目标**: 验证最优传输的有效性

**执行**:
```bash
python experiments/ablation/train_ablation_ot.py \
    --experiment_name ablation_no_ot \
    --num_runs 5 \
    --output_dir experiments/results
```

**预期结果**:
- AUC: 82-86% (vs Full Model 85%+)
- 预期下降: -2~-3%

#### Step 2.4: Dual-Head消融

**目标**: 验证因果解耦的有效性

**执行**:
```bash
python experiments/ablation/train_ablation_dual_head.py \
    --experiment_name ablation_no_dual_head \
    --num_runs 5 \
    --output_dir experiments/results
```

**预期结果**:
- AUC: 83-87% (vs Full Model 85%+)
- 预期下降: -1~-2%

#### Step 2.5: 批量运行所有消融实验

```bash
bash scripts/run_all_ablations.sh
```

---

### 阶段3: 完整方法训练 (1-2周)

#### Step 3.1: 性能优化

**在运行完整方法前，先进行性能优化**:

1. **损失函数优化** (尝试不同权重组合)
2. **数据增强** (添加更强的数据增强)
3. **模型架构优化** (尝试更深的网络)
4. **训练策略优化** (学习率调度、早停等)

#### Step 3.2: 完整方法训练

**执行**:
```bash
python training/train_bio_cot_v3.py \
    --experiment_name full_model \
    --num_runs 5 \
    --output_dir experiments/results \
    --optimize_performance  # 使用优化后的配置
```

**预期结果**:
- AUC: ≥85%
- Accuracy: ≥80%
- F1-Score: ≥70%

---

### 阶段4: 结果分析和统计检验 (1周)

#### Step 4.1: 汇总所有结果

```bash
python scripts/analyze_all_results.py \
    --results_dir experiments/results \
    --output_dir experiments/results/analysis \
    --reference_method full_model \
    --metric auc
```

**输出**:
- Baseline对比表格 (CSV + LaTeX)
- 消融实验表格 (CSV + LaTeX)
- 统计显著性检验结果
- 可视化图表

#### Step 4.2: 生成论文表格

**自动生成**:
- `baseline_comparison_auc.csv` / `.tex`
- `ablation_comparison_auc.csv` / `.tex`

**手动检查**:
```bash
# 查看结果
cat experiments/results/analysis/baseline_comparison_auc.csv
cat experiments/results/analysis/ablation_comparison_auc.csv
```

---

## 📊 实验监控

### 实时监控训练进度

```bash
# 方法1: 使用tail查看日志
tail -f experiments/logs/*.log

# 方法2: 使用TensorBoard
tensorboard --logdir experiments/results --port 6006

# 方法3: 使用Weights & Biases (如果配置)
# wandb会自动记录训练过程
```

### 检查实验状态

```bash
# 检查正在运行的实验
ps aux | grep python | grep train

# 检查GPU使用情况
nvidia-smi

# 检查实验结果
ls -lh experiments/results/*/results/
```

---

## 🔍 结果验证

### 验证实验完整性

**检查清单**:
```bash
# 1. 检查所有Baseline是否完成
ls experiments/results/baseline_*/

# 2. 检查所有消融实验是否完成
ls experiments/results/ablation_*/

# 3. 检查完整方法是否完成
ls experiments/results/full_model/

# 4. 检查每个实验是否有5次运行结果
for exp_dir in experiments/results/*/; do
    echo "检查: $exp_dir"
    ls $exp_dir/results/result_run_*.json | wc -l
done
```

### 验证统计检验

**检查统计显著性**:
```python
# 运行验证脚本
python scripts/verify_statistics.py \
    --results_dir experiments/results \
    --min_runs 5  # 每个实验至少5次运行
```

---

## ⚠️ 常见问题处理

### 问题1: 实验运行失败

**检查**:
1. GPU内存是否足够
2. 数据路径是否正确
3. 依赖库是否安装完整

**解决**:
```bash
# 减少batch_size
# 修改config.py中的batch_size

# 检查数据路径
ls /data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal

# 重新安装依赖
pip install -r requirements.txt
```

### 问题2: 结果不一致

**检查**:
1. 随机种子是否固定
2. 数据划分是否一致
3. 模型配置是否相同

**解决**:
```python
# 确保使用固定随机种子
set_seed(42)  # 在所有实验中使用相同的种子

# 使用相同的数据划分
train_csv = 'internal_train/labels.csv'
val_csv = 'internal_val/labels.csv'
```

### 问题3: 性能不达标

**检查**:
1. 训练是否充分（检查训练曲线）
2. 损失函数权重是否合理
3. 数据增强是否有效

**解决**:
```python
# 增加训练轮数
num_epochs = 150  # 从100增加到150

# 调整损失权重
lambda_cls = 3.0  # 增加分类损失权重

# 使用更强的数据增强
transforms = [更强的数据增强策略]
```

---

## 📈 性能目标检查

### 目标性能指标

| 指标 | 当前值 | 目标值 | 状态 |
|------|--------|--------|------|
| AUC | 79.80% | ≥85% | ⚠️ 需提升 |
| Accuracy | 73.81% | ≥80% | ⚠️ 需提升 |
| F1-Score | 63.33% | ≥70% | ⚠️ 需提升 |

### 性能提升检查点

**检查点1**: Baseline对比完成
- [ ] 所有Baseline运行完成
- [ ] 结果优于所有Baseline (p < 0.05)

**检查点2**: 消融实验完成
- [ ] 所有消融实验运行完成
- [ ] 每个模块都有显著贡献 (p < 0.05)

**检查点3**: 性能达标
- [ ] AUC ≥ 85%
- [ ] Accuracy ≥ 80%
- [ ] F1-Score ≥ 70%

**检查点4**: 统计检验完成
- [ ] 所有实验运行5次以上
- [ ] 统计显著性检验完成
- [ ] 结果表格生成完成

---

## 🎯 实验执行时间表

### 详细时间表

| 阶段 | 任务 | 预计时间 | 状态 |
|------|------|---------|------|
| **Week 1-2** | 实现Baseline方法 | 2周 | ⏳ 待开始 |
| **Week 3-4** | 运行Baseline实验 | 2周 | ⏳ 待开始 |
| **Week 5-6** | 实现消融实验 | 2周 | ⏳ 待开始 |
| **Week 7** | 运行消融实验 | 1周 | ⏳ 待开始 |
| **Week 8-9** | 性能优化 | 2周 | ⏳ 待开始 |
| **Week 10** | 完整方法训练 | 1周 | ⏳ 待开始 |
| **Week 11** | 统计分析 | 1周 | ⏳ 待开始 |
| **Week 12** | 论文撰写 | 1周 | ⏳ 待开始 |

**总预计时间**: 12周 (3个月)

---

## 📝 实验记录模板

### 实验记录表

**实验名称**: `_________________`

**执行日期**: `_________________`

**配置**:
- Random Seed: `_________________`
- Batch Size: `_________________`
- Learning Rate: `_________________`
- Num Epochs: `_________________`

**结果**:
- AUC: `_________________`
- Accuracy: `_________________`
- Precision: `_________________`
- Recall: `_________________`
- F1-Score: `_________________`

**备注**: `_________________`

---

## ✅ 实验完成检查清单

### 实验完整性检查

**Baseline对比**:
- [ ] Simple Fusion Baseline (5次运行)
- [ ] Standard CLIP Baseline (5次运行)
- [ ] ViT + Clinical Fusion (5次运行)
- [ ] CNN Baseline (已有或重新运行)
- [ ] Swin-T Baseline (已有或重新运行)
- [ ] VMamba Baseline (已有或重新运行)

**消融实验**:
- [ ] Knowledge Notes消融 (5次运行)
- [ ] Visual Notes消融 (5次运行)
- [ ] Sinkhorn OT消融 (5次运行)
- [ ] Dual-Head消融 (5次运行)
- [ ] 组合消融 (5次运行)

**完整方法**:
- [ ] Full Model训练 (5次运行)
- [ ] 性能达到目标 (AUC ≥ 85%)

**统计分析**:
- [ ] 所有结果汇总完成
- [ ] 统计显著性检验完成
- [ ] 结果表格生成完成
- [ ] LaTeX表格生成完成

---

## 🎉 完成标准

### MICCAI发表标准检查

**实验完整性**: ✅
- [ ] 6个Baseline对比完成
- [ ] 6个消融实验完成
- [ ] 所有实验运行5次以上

**性能要求**: ✅
- [ ] AUC ≥ 85%
- [ ] Accuracy ≥ 80%
- [ ] F1-Score ≥ 70%

**统计严谨性**: ✅
- [ ] 所有结果包含统计显著性检验
- [ ] p-value < 0.05
- [ ] 95%置信区间计算完成

**结果呈现**: ✅
- [ ] 结果表格完整
- [ ] 可视化图表清晰
- [ ] LaTeX表格生成完成

---

**文档版本**: v1.0  
**创建日期**: 2025-01-15  
**最后更新**: 2025-01-15

