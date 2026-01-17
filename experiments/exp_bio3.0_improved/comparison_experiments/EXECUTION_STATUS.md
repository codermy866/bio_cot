# 对比实验执行状态

## 📋 实验概述

本目录包含所有对比实验（Baselines）的代码和结果，用于与Bio-COT 3.0 Improved方法进行对比。

## 🎯 已实现的Baseline方法

### 1. Simple Fusion Baseline
- **路径**: `baselines/simple_fusion/train_simple_fusion.py`
- **方法**: 特征拼接 + MLP分类器
- **特点**: 无注意力机制，无因果约束，无不确定性量化

### 2. Standard CLIP Baseline
- **路径**: `baselines/standard_clip/train_standard_clip.py`
- **方法**: 标准CLIP方法
- **特点**: 有对比学习，无因果约束，无不确定性量化

### 3. ViT + Clinical Fusion Baseline
- **路径**: `baselines/vit_clinical_fusion/train_vit_clinical.py`
- **方法**: ViT特征 + 临床特征简单融合
- **特点**: 无最优传输，使用MSE对齐

## 📁 文件结构

```
comparison_experiments/
├── baselines/
│   ├── simple_fusion/
│   │   └── train_simple_fusion.py
│   ├── standard_clip/
│   │   └── train_standard_clip.py
│   └── vit_clinical_fusion/
│       └── train_vit_clinical.py
├── results/
│   ├── baseline_simple_fusion/
│   │   ├── results/
│   │   │   ├── all_results.json      # 所有运行结果（JSON格式）
│   │   │   ├── all_results.csv       # 所有运行结果（CSV格式）
│   │   │   └── statistics.json        # 统计信息
│   │   └── logs/                      # 训练日志
│   ├── baseline_standard_clip/
│   │   └── ...
│   └── baseline_vit_clinical_fusion/
│       └── ...
├── visualizations/
│   ├── performance_comparison.png      # 性能对比图
│   ├── box_plots.png                   # 箱线图
│   ├── statistical_significance.png   # 统计显著性分析
│   ├── summary_table.png               # 汇总表格
│   ├── summary_table.csv               # 汇总表格（CSV）
│   └── p_values.json                   # P值统计
├── logs/
│   └── run_YYYYMMDD_HHMMSS.log         # 执行日志
├── run_all_baselines.py                # 自动执行所有baseline实验
├── visualize_results.py                # 生成可视化结果
├── run_and_visualize.py                 # 完整执行流程（实验+可视化）
└── README.md                            # 说明文档
```

## 🚀 使用方法

### 方法1: 自动执行所有实验并生成可视化

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0_improved
python comparison_experiments/run_and_visualize.py \
    --num_runs 5 \
    --num_epochs 100 \
    --batch_size 32 \
    --data_root /data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal
```

### 方法2: 分别执行

#### 2.1 运行所有baseline实验

```bash
python comparison_experiments/run_all_baselines.py \
    --num_runs 5 \
    --num_epochs 100 \
    --batch_size 32
```

#### 2.2 生成可视化结果

```bash
python comparison_experiments/visualize_results.py \
    --results_dir comparison_experiments/results \
    --output_dir comparison_experiments/visualizations
```

### 方法3: 单独运行某个baseline

```bash
python comparison_experiments/baselines/simple_fusion/train_simple_fusion.py \
    --experiment_name baseline_simple_fusion \
    --num_runs 5 \
    --num_epochs 100 \
    --batch_size 32
```

## 📊 结果文件说明

### JSON结果文件 (`all_results.json`)
包含每次运行的详细结果：
```json
{
  "experiment_name": "baseline_simple_fusion",
  "run_id": 1,
  "random_seed": 42,
  "auc": 0.8234,
  "accuracy": 0.7619,
  "precision": 0.7456,
  "recall": 0.7812,
  "specificity": 0.7523,
  "f1_score": 0.7634,
  "best_epoch": 85,
  "train_loss": 0.3421,
  "val_loss": 0.3892,
  "training_time": 1234.56
}
```

### CSV结果文件 (`all_results.csv`)
包含所有运行的表格数据，便于在Excel或其他工具中分析。

### 统计信息文件 (`statistics.json`)
包含每个指标的统计信息：
- 均值 (mean)
- 标准差 (std)
- 中位数 (median)
- 95%置信区间 (ci_95)
- 最小值/最大值 (min/max)
- 四分位数 (q25/q75)

### 可视化文件
- **performance_comparison.png**: 所有baseline的性能对比条形图
- **box_plots.png**: 各指标的箱线图分布
- **statistical_significance.png**: 统计显著性热力图
- **summary_table.png**: 汇总表格（适合论文使用）

## 🔍 实验配置

### 默认配置
- **运行次数**: 5次（使用不同随机种子：42, 123, 456, 789, 2024）
- **训练轮数**: 100 epochs
- **Batch Size**: 32
- **学习率**: 0.0002
- **优化器**: AdamW
- **损失函数**: CrossEntropyLoss

### 数据配置
- **数据路径**: `/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal`
- **训练集**: `internal_train/labels.csv`
- **验证集**: `internal_val/labels.csv`
- **OCT帧数**: 20
- **Colposcopy图像数**: 3

## 📈 评估指标

所有实验都会计算以下指标：
1. **AUC** (Area Under ROC Curve)
2. **Accuracy** (准确率)
3. **Precision** (精确率)
4. **Recall** (召回率)
5. **Specificity** (特异性)
6. **F1-Score** (F1分数)

## ✅ 当前执行状态

实验正在后台运行中。可以通过以下命令查看进度：

```bash
# 查看运行中的进程
ps aux | grep "train_simple_fusion\|train_standard_clip\|train_vit_clinical" | grep -v grep

# 查看最新日志
tail -f comparison_experiments/logs/run_*.log

# 查看结果目录
ls -lh comparison_experiments/results/*/results/
```

## 📝 注意事项

1. **数据路径**: 确保数据路径正确，数据集应包含`internal_train`和`internal_val`目录
2. **GPU内存**: 根据GPU内存调整batch_size
3. **训练时间**: 每个baseline实验大约需要1-2小时（取决于epochs和batch_size）
4. **结果保存**: 所有结果会自动保存到`comparison_experiments/results/`目录
5. **日志保存**: 执行日志保存在`comparison_experiments/logs/`目录

## 🔄 下一步

实验完成后：
1. 检查所有结果文件是否生成
2. 查看可视化图表
3. 分析统计显著性
4. 准备论文中的对比表格和图表

