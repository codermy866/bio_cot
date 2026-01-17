# Bio-COT 3.0 Improved 快速开始指南

> **快速开始执行实验，确保符合MICCAI发表标准**

---

## 🚀 5分钟快速开始

### Step 1: 环境检查 (1分钟)

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics
source my_retfound/bin/activate
cd experiments/exp_bio3.0_improved

# 检查GPU
nvidia-smi

# 检查Python环境
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 2: 创建实验目录 (1分钟)

```bash
# 创建必要的目录
mkdir -p experiments/{baseline,ablation,full_model,results,logs}
mkdir -p utils scripts
```

### Step 3: 运行第一个实验 (3分钟)

```bash
# 运行Simple Fusion Baseline（测试流程）
python experiments/baseline/train_simple_fusion_baseline.py \
    --experiment_name baseline_simple_fusion \
    --num_runs 1 \
    --batch_size 32 \
    --num_epochs 10  # 先用少量epoch测试
```

---

## 📋 完整实验执行流程

### 方案A: 使用自动化脚本（推荐）

```bash
# 运行所有实验（按顺序）
bash scripts/run_all_experiments.sh
```

**预计时间**: 8-12周

### 方案B: 手动执行（逐步进行）

#### 1. Baseline对比实验

```bash
# 1. Simple Fusion
python experiments/baseline/train_simple_fusion_baseline.py --num_runs 5

# 2. Standard CLIP
python experiments/baseline/train_standard_clip_baseline.py --num_runs 5

# 3. ViT + Clinical Fusion
python experiments/baseline/train_vit_clinical_fusion.py --num_runs 5
```

#### 2. 消融实验

```bash
# 1. Knowledge Notes消融
python experiments/ablation/train_ablation_knowledge_notes.py --num_runs 5

# 2. Visual Notes消融
python experiments/ablation/train_ablation_visual_notes.py --num_runs 5

# 3. Sinkhorn OT消融
python experiments/ablation/train_ablation_ot.py --num_runs 5

# 4. Dual-Head消融
python experiments/ablation/train_ablation_dual_head.py --num_runs 5
```

#### 3. 完整方法训练

```bash
# Full Model（使用优化后的配置）
python training/train_bio_cot_v3.py \
    --experiment_name full_model \
    --num_runs 5 \
    --optimize_performance
```

#### 4. 结果分析

```bash
# 汇总所有结果并生成表格
python scripts/analyze_all_results.py \
    --results_dir experiments/results \
    --output_dir experiments/results/analysis \
    --reference_method full_model
```

---

## 📊 实验监控

### 实时监控

```bash
# 查看训练日志
tail -f experiments/logs/*.log

# 查看GPU使用情况
watch -n 1 nvidia-smi

# 查看实验结果
ls -lh experiments/results/*/results/
```

### 检查实验状态

```bash
# 检查正在运行的实验
ps aux | grep python | grep train

# 检查已完成实验
for exp_dir in experiments/results/*/; do
    echo "$(basename $exp_dir): $(ls $exp_dir/results/result_run_*.json 2>/dev/null | wc -l) runs"
done
```

---

## ✅ 实验完成验证

### 快速检查

```bash
# 运行验证脚本
python scripts/verify_experiments.py \
    --results_dir experiments/results \
    --min_runs 5
```

### 手动检查

```bash
# 1. 检查所有实验是否完成
ls experiments/results/

# 2. 检查每个实验的运行次数
for exp_dir in experiments/results/*/; do
    count=$(ls $exp_dir/results/result_run_*.json 2>/dev/null | wc -l)
    echo "$(basename $exp_dir): $count/5 runs"
done

# 3. 检查统计信息
cat experiments/results/*/results/statistics.json
```

---

## 🎯 关键文件说明

### 实验设计文档
- `COMPLETE_EXPERIMENT_DESIGN.md` - 完整实验设计
- `EXPERIMENT_EXECUTION_GUIDE.md` - 实验执行指南
- `EXPERIMENT_CHECKLIST.md` - 实验检查清单

### 代码文件
- `utils/experiment_manager.py` - 实验管理器
- `utils/statistics.py` - 统计检验工具
- `scripts/run_all_experiments.sh` - 运行所有实验
- `scripts/analyze_all_results.py` - 结果分析脚本

### 实验脚本
- `experiments/baseline/` - Baseline实验脚本
- `experiments/ablation/` - 消融实验脚本

---

## ⚠️ 常见问题

### Q1: 实验运行失败

**检查**:
1. GPU内存是否足够（减少batch_size）
2. 数据路径是否正确
3. 依赖库是否安装完整

**解决**:
```bash
# 减少batch_size
# 修改config.py或命令行参数

# 检查数据路径
ls /data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal

# 重新安装依赖
pip install -r requirements.txt
```

### Q2: 结果不一致

**检查**:
1. 随机种子是否固定
2. 数据划分是否一致
3. 模型配置是否相同

**解决**:
```python
# 确保使用固定随机种子
set_seed(42)  # 在所有实验中使用相同的种子
```

### Q3: 性能不达标

**检查**:
1. 训练是否充分
2. 损失函数权重是否合理
3. 数据增强是否有效

**解决**:
```python
# 增加训练轮数
num_epochs = 150

# 调整损失权重
lambda_cls = 3.0
```

---

## 📈 进度跟踪

### 使用检查清单

```bash
# 查看实验进度
cat EXPERIMENT_CHECKLIST.md | grep -E "\[ \]|\[x\]"
```

### 使用进度脚本

```bash
# 生成进度报告
python scripts/generate_progress_report.py \
    --results_dir experiments/results
```

---

## 🎉 完成标准

### 所有实验完成标志

✅ **Baseline对比**: 6个方法，每个运行5次  
✅ **消融实验**: 5个实验，每个运行5次  
✅ **完整方法**: 1个实验，运行5次  
✅ **性能达标**: AUC ≥ 85%, Acc ≥ 80%  
✅ **统计检验**: 所有结果包含p-value  
✅ **结果表格**: CSV和LaTeX格式生成完成  

---

**快速开始指南版本**: v1.0  
**创建日期**: 2025-01-15

