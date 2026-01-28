# 对比实验（Comparison Experiments）
## MICCAI论文 Table 1

---

## 📋 实验列表

### 1. Bio-COT 3.2 (Full) - Our Method
- **脚本**: `training/train_bio_cot_v3.2.py`
- **描述**: 我们提出的完整方法
- **运行次数**: 5次（不同随机种子）

### 2. MedCLIP
- **脚本**: `exp_bio3.0_improved/comparison_experiments/baselines/sota_baselines/medclip/train_medclip.py`
- **描述**: 医学领域专用CLIP模型
- **运行次数**: 5次

### 3. ConVIRT
- **脚本**: `exp_bio3.0_improved/comparison_experiments/baselines/sota_baselines/convirt/train_convirt.py`
- **描述**: 对比学习的医学Vision-Representation Transformer
- **运行次数**: 5次

### 4. mmFormer
- **脚本**: `exp_bio3.0_improved/comparison_experiments/baselines/sota_baselines/mmformer/train_mmformer.py`
- **描述**: 多模态医学Transformer
- **运行次数**: 5次

### 5. Swin-T + Fusion
- **脚本**: `exp_bio3.0_improved/comparison_experiments/baselines/swin_t_baseline/train_swin.py`
- **描述**: Swin-T编码器 + 简单融合
- **运行次数**: 5次

---

## 🚀 运行方式

### 自动运行所有对比实验

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/comparison_experiments
python run_all_comparisons.py
```

### 后台运行（推荐）

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/comparison_experiments
nohup python run_all_comparisons.py > run_all_comparisons.log 2>&1 &
```

### 查看运行状态

```bash
# 查看日志
tail -f comparison_experiments/logs/*.log

# 查看运行进程
ps aux | grep run_all_comparisons

# 查看结果汇总
cat comparison_experiments/results/results_summary.json
```

---

## 📁 目录结构

```
comparison_experiments/
├── README.md                    # 本文件
├── run_all_comparisons.py       # 自动执行脚本
├── results/                     # 实验结果
│   ├── Bio-COT_3.2_Full/
│   │   ├── run_1_seed_42/
│   │   ├── run_2_seed_123/
│   │   └── ...
│   ├── MedCLIP/
│   ├── ConVIRT/
│   ├── mmFormer/
│   └── Swin-T_Fusion/
├── logs/                        # 运行日志
│   ├── Bio-COT_3.2_Full_run1_seed42_*.log
│   ├── MedCLIP_run1_seed42_*.log
│   └── ...
└── experiment_config.json       # 实验配置
```

---

## ⏱️ 预计运行时间

- **每个实验**: ~4小时（单次运行）
- **总运行次数**: 5个方法 × 5次 = 25次运行
- **总预计时间**: ~100小时（约4-5天）

**建议**: 使用多GPU并行运行不同实验，可大幅缩短时间。

---

## ✅ 检查清单

- [ ] 所有实验脚本存在且可执行
- [ ] 数据路径正确
- [ ] 输出目录有写权限
- [ ] GPU资源充足
- [ ] 日志目录可写

---

**生成时间**: 2026-01-23

