# MICCAI论文实验执行计划

## 📋 实验优先级和执行顺序

### 阶段1：消融实验（已完成大部分，需补充5次运行）
**优先级**：⭐⭐⭐⭐⭐ **最高**

#### 当前状态
- ✅ 所有消融实验配置已就绪
- ✅ 部分实验已完成1次运行
- ⚠️ **需要**：每个实验运行**5次**（不同随机种子）用于统计检验

#### 执行步骤
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/ablation_studies

# 方案1：使用现有的自动运行脚本（需要修改为5次运行）
python run_all_ablations_with_monitor.py

# 方案2：手动运行每个实验5次（推荐，更可控）
for exp in baseline w/o_visual_notes w/o_adaptive_gating w/o_alignment_loss w/o_ot_loss w/o_dual_head w/o_cross_attn w/o_hierarchical w/o_noise_aware w/o_clinical_evolver w/o_text_adapter w/o_vlm_retriever; do
    for seed in 42 123 456 789 2024; do
        python training/train_bio_cot_v3.2.py \
            --config ablation_studies/${exp}/config.py \
            --seed ${seed} \
            --output_dir ablation_studies/${exp}/results/seed_${seed}
    done
done
```

---

### 阶段2：对比实验（部分已完成，需补充SOTA方法）
**优先级**：⭐⭐⭐⭐⭐ **最高**

#### 当前状态
- ✅ MedCLIP - 已实现，需运行5次
- ✅ ConVIRT - 已实现，需运行5次
- ✅ mmFormer - 已实现，需运行5次
- ✅ Swin-T - 已实现，需运行5次
- ⚠️ BioMedCLIP - 待实现
- ⚠️ MATR - 待实现
- ⚠️ Simple Fusion - 待实现

#### 执行步骤
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2

# 1. 运行已实现的SOTA方法（5次）
for method in medclip convirt mmformer swin_t; do
    for seed in 42 123 456 789 2024; do
        python comparison_experiments/baselines/sota_baselines/${method}/train_${method}.py \
            --seed ${seed} \
            --output_dir comparison_experiments/results/${method}/seed_${seed}
    done
done

# 2. 实现并运行BioMedCLIP（如果可用）
# 3. 实现并运行MATR（如果可用）
# 4. 实现并运行Simple Fusion（简单baseline）
```

---

### 阶段3：统计分析工具
**优先级**：⭐⭐⭐⭐ **高**

#### 需要实现的工具
1. **McNemar检验**：比较两个方法的分类结果
2. **Wilcoxon符号秩检验**：比较AUC分布
3. **置信区间计算**：95% CI for AUC
4. **结果汇总表格生成**：自动生成LaTeX表格

#### 执行步骤
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2

# 运行统计分析脚本（待创建）
python analysis/statistical_analysis.py \
    --comparison_results comparison_experiments/results/ \
    --ablation_results ablation_studies/ \
    --output_dir analysis/statistical_results/
```

---

### 阶段4：跨中心泛化分析
**优先级**：⭐⭐⭐ **中**

#### 执行步骤
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2

# 按中心分组评估（使用已有的评估脚本）
python analysis/cross_center_analysis.py \
    --checkpoint_dir results/best_model/ \
    --output_dir analysis/cross_center_results/
```

---

### 阶段5：额外分析（可选但推荐）
**优先级**：⭐⭐ **低**

#### 模态重要性分析
```bash
python analysis/modality_importance.py \
    --config config.py \
    --output_dir analysis/modality_analysis/
```

#### 计算效率分析
```bash
python analysis/computational_efficiency.py \
    --model_checkpoint results/best_model/ \
    --output_dir analysis/efficiency_analysis/
```

---

## 📊 实验执行时间估算

| 实验类型 | 实验数量 | 每次运行时间 | 总时间（单次） | 总时间（5次） |
|:---------|---------:|-------------:|--------------:|-------------:|
| **消融实验** | 12 | ~4小时 | 48小时 | **240小时（10天）** |
| **对比实验（已实现）** | 4 | ~4小时 | 16小时 | **80小时（3.3天）** |
| **对比实验（待实现）** | 3 | ~4小时 | 12小时 | **60小时（2.5天）** |
| **统计分析** | 1 | ~1小时 | 1小时 | **1小时** |
| **跨中心分析** | 1 | ~2小时 | 2小时 | **2小时** |
| **额外分析** | 2 | ~2小时 | 4小时 | **4小时** |
| **总计** | - | - | **83小时** | **387小时（16天）** |

**建议**：
- 使用**多GPU并行**运行不同实验，可大幅缩短时间
- 优先完成**消融实验**和**已实现的对比实验**（核心内容）
- 待实现的对比实验可以后续补充

---

## ✅ 检查清单

### 消融实验
- [ ] Baseline - 运行5次（不同随机种子）
- [ ] w/o_hierarchical - 运行5次
- [ ] w/o_noise_aware - 运行5次
- [ ] w/o_clinical_evolver - 运行5次
- [ ] w/o_text_adapter - 运行5次
- [ ] w/o_visual_notes - 运行5次
- [ ] w/o_adaptive_gating - 运行5次
- [ ] w/o_cross_attn - 运行5次
- [ ] w/o_alignment_loss - 运行5次
- [ ] w/o_ot_loss - 运行5次
- [ ] w/o_dual_head - 运行5次
- [ ] w/o_vlm_retriever - 运行5次

### 对比实验
- [ ] MedCLIP - 运行5次
- [ ] ConVIRT - 运行5次
- [ ] mmFormer - 运行5次
- [ ] Swin-T - 运行5次
- [ ] BioMedCLIP - 实现并运行5次
- [ ] MATR - 实现并运行5次
- [ ] Simple Fusion - 实现并运行5次

### 统计分析
- [ ] 实现McNemar检验工具
- [ ] 实现Wilcoxon检验工具
- [ ] 实现置信区间计算
- [ ] 生成结果汇总表格

### 跨中心分析
- [ ] 按中心分组评估
- [ ] 生成跨中心性能表格

---

## 🚀 快速开始

### 1. 立即开始消融实验（5次运行）
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/ablation_studies
python run_ablations_5_runs.py  # 需要创建此脚本
```

### 2. 立即开始对比实验（5次运行）
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2
python run_comparisons_5_runs.py  # 需要创建此脚本
```

### 3. 实时监控实验进度
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/ablation_studies
python realtime_monitor.py
```

---

**生成时间**：2026-01-23  
**预计完成时间**：16天（使用多GPU可缩短至5-7天）

