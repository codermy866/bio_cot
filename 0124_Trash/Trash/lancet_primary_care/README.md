# The Lancet Primary Care - 宫颈癌筛查优化研究

## 📋 研究概述

本研究旨在通过多模态AI技术优化宫颈癌初筛流程，重点关注临床实际应用价值和患者获益。

### 🎯 核心研究目标

1. **降低HPV+患者的过度活检率**
   - 通过精准的风险分层，减少不必要的活检
   - 提高活检的阳性预测值（PPV）

2. **提高OCT的灵敏度**
   - 通过AI增强OCT图像分析能力
   - 提高早期病变的检出率

3. **优于传统筛查方案**
   - 证明多模态AI优于HPV+TCT联合筛查
   - 证明多模态AI优于单独TCT筛查

4. **优化宫颈癌初筛流程**
   - 建立新的筛查决策流程
   - 提供临床可操作的决策支持

---

## 📁 文件夹结构

```
lancet_primary_care/
├── README.md                          # 本文件
├── docs/                              # 研究文档
│   ├── RESEARCH_PROPOSAL.md          # 研究提案
│   ├── EXPERIMENTAL_DESIGN.md        # 实验设计
│   ├── CLINICAL_SIGNIFICANCE.md      # 临床意义分析
│   └── STATISTICAL_ANALYSIS_PLAN.md  # 统计分析计划
├── experiments/                       # 实验代码
│   ├── experiment1_biopsy_reduction.py      # 实验1: 活检率降低
│   ├── experiment2_oct_sensitivity.py       # 实验2: OCT灵敏度提升
│   ├── experiment3_screening_comparison.py  # 实验3: 筛查方案对比
│   └── experiment4_workflow_optimization.py # 实验4: 流程优化
├── evaluation/                        # 评估工具
│   ├── clinical_metrics_calculator.py      # 临床指标计算
│   ├── biopsy_rate_analyzer.py             # 活检率分析
│   ├── screening_workflow_evaluator.py     # 筛查流程评估
│   └── cost_effectiveness_analysis.py      # 成本效益分析
├── clinical_analysis/                 # 临床分析
│   ├── subgroup_analysis.py           # 亚组分析
│   ├── risk_stratification.py        # 风险分层
│   └── decision_curve_analysis.py    # 决策曲线分析
├── data_preparation/                  # 数据准备
│   ├── prepare_clinical_data.py       # 准备临床数据
│   └── create_screening_cohorts.py   # 创建筛查队列
├── visualization/                     # 可视化
│   ├── plot_biopsy_reduction.py      # 活检率降低可视化
│   ├── plot_screening_comparison.py  # 筛查对比可视化
│   └── plot_workflow_optimization.py # 流程优化可视化
├── models/                            # 模型文件
│   └── (模型定义和权重)
├── results/                           # 实验结果
│   ├── experiment1_results/          # 实验1结果
│   ├── experiment2_results/          # 实验2结果
│   ├── experiment3_results/          # 实验3结果
│   └── experiment4_results/          # 实验4结果
├── reports/                           # 报告
│   ├── main_results_report.md        # 主要结果报告
│   └── clinical_impact_report.md     # 临床影响报告
└── scripts/                           # 执行脚本
    ├── run_all_experiments.sh        # 运行所有实验
    └── generate_figures.sh           # 生成图表
```

---

## 🔬 实验方案概览

### 实验1: HPV+患者活检率降低实验
**目标**: 证明多模态AI可以降低HPV+患者的过度活检率，同时保持高灵敏度

### 实验2: OCT灵敏度提升实验
**目标**: 证明AI增强的OCT分析显著提高病变检出灵敏度

### 实验3: 筛查方案对比实验
**目标**: 系统对比多模态AI vs HPV+TCT vs 单独TCT

### 实验4: 筛查流程优化实验
**目标**: 设计并验证优化的筛查决策流程

---

## 📊 关键评估指标

### 临床指标
- **活检率 (Biopsy Rate)**: 需要活检的患者比例
- **活检阳性预测值 (PPV)**: 活检阳性的比例
- **灵敏度 (Sensitivity)**: 病变检出率
- **特异度 (Specificity)**: 正常样本正确识别率
- **阴性预测值 (NPV)**: 阴性预测的准确性

### 流程优化指标
- **筛查效率**: 减少的筛查步骤
- **成本效益**: 每检出1例病变的成本
- **患者体验**: 减少的不必要检查

---

## 🚀 快速开始

详见各实验的详细文档和代码。

---

## 📚 相关文档

- [实验设计方案](docs/EXPERIMENTAL_DESIGN.md)
- [临床意义分析](docs/CLINICAL_SIGNIFICANCE.md)
- [统计分析计划](docs/STATISTICAL_ANALYSIS_PLAN.md)

