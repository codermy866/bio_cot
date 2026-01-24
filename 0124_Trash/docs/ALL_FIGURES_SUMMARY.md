# 所有图表总结

## 📊 已生成的所有图表

### 1. 主结果图集

**位置**: `paper_figures_final_cuda1/`

#### 📄 main_results_figure.png (743KB)
- **内容**: 6面板综合分析
  - 分中心AUC对比（Forest plot）
  - 决策曲线分析（DCA）
  - 不确定性量化（AUC vs 校准误差）
  - 性能指标对比
  - 模型对比（多模态 vs 单模态）
  - ROC曲线
- **用途**: 论文主结果图

#### 📄 performance_dashboard.png (801KB)
- **内容**: 综合性能仪表板
  - 多维度指标可视化
  - 中心间比较
  - 性能雷达图
- **用途**: 补充材料或附图

### 2. 分中心评估结果

**位置**: `true_real_center_evaluation_results/`

#### 📄 true_real_forest_plot.png (121KB)
- **内容**: Forest plot展示5个中心的AUC及95%置信区间
- **用途**: Methods或Results中的分中心验证图

#### 📄 true_real_performance_comparison.png (240KB)
- **内容**: 5个中心的多指标性能对比
  - Accuracy, F1, Precision, Recall
- **用途**: 补充分析图

### 3. 决策曲线分析

**位置**: `dca_analysis_simple_cuda1/`

#### 📄 decision_curve_analysis.png (315KB)
- **内容**: 决策曲线
  - 展示不同阈值下的净获益
  - 多模型对比
- **用途**: 临床实用性分析

#### 📄 cost_effectiveness_analysis.png (315KB)
- **内容**: 成本效益分析
  - 净获益曲线
  - 临床阈值标注
- **用途**: 经济学评估补充

### 4. 不确定性量化

**位置**: `uncertainty_analysis_simple_cuda1/`

#### 📄 uncertainty_analysis.png (1.1M)
- **内容**: 不确定性分析综合图
  - 预测区间
  - 置信区间
  - 不同方法的对比
- **用途**: 不确定性分析主图

#### 📄 reliability_diagram.png (568KB)
- **内容**: 可靠性图
  - 预测概率 vs 观察频率
  - 校准曲线
- **用途**: 校准分析图

#### 📄 prediction_intervals.png (498KB)
- **内容**: 预测区间图
  - 展示预测的不确定性
  - 可视化置信区间
- **用途**: 不确定性可视化

### 5. 其他可视化

**位置**: `classification_comparison.png` (95KB)
- **内容**: 5分类 vs 4分类 vs 3分类方案对比
- **用途**: 探索性分析的补充图

## 📋 图表使用建议

### 论文主图（必须）
- **Figure 1**: `main_results_figure.png` - 主要结果
- **Figure 2**: `decision_curve_analysis.png` - 临床实用性

### 补充材料
- **Supplementary Figure 1**: `performance_dashboard.png` - 性能仪表板
- **Supplementary Figure 2**: `true_real_forest_plot.png` - 分中心验证
- **Supplementary Figure 3**: `reliability_diagram.png` - 校准分析

### 可选附图
- `uncertainty_analysis.png` - 不确定性分析
- `prediction_intervals.png` - 预测区间
- `cost_effectiveness_analysis.png` - 成本效益

## 📊 图表内容详细说明

### 主结果图（main_results_figure.png）的6个面板

1. **面板1: 分中心AUC对比**
   - 5个中心的AUC值
   - 95%置信区间
   - 平均性能线

2. **面板2: 决策曲线分析**
   - 多个模型的净获益曲线
   - 阈值范围标注
   - 最优阈值标注

3. **面板3: 不确定性分析**
   - 不同方法（MC-Dropout, Ensemble等）的AUC
   - 校准误差对比
   - 方法排名

4. **面板4: 性能指标**
   - Accuracy, F1, Precision, Recall
   - 多中心对比

5. **面板5: 模型对比**
   - 多模态 vs 单模态
   - OCT-only vs COL-only
   - 临床特征基线

6. **面板6: ROC曲线**
   - 所有模型的ROC曲线
   - AUC值标注

## 🎯 图表文件路径

所有图表都在以下位置：

```bash
# 主结果
paper_figures_final_cuda1/main_results_figure.png
paper_figures_final_cuda1/performance_dashboard.png

# 分中心评估
true_real_center_evaluation_results/true_real_forest_plot.png
true_real_center_evaluation_results/true_real_performance_comparison.png

# DCA分析
dca_analysis_simple_cuda1/decision_curve_analysis.png
dca_analysis_simple_cuda1/cost_effectiveness_analysis.png

# 不确定性分析
uncertainty_analysis_simple_cuda1/uncertainty_analysis.png
uncertainty_analysis_simple_cuda1/reliability_diagram.png
uncertainty_analysis_simple_cuda1/prediction_intervals.png
```

## 📝 数据表格

### CSV表格
- `paper_figures_final_cuda1/comprehensive_summary.csv`
- 包含所有指标数据

### LaTeX表格
- `paper_figures_final_cuda1/comprehensive_summary.tex`
- 可直接插入LaTeX文档

## 💡 图表制作方法

### 使用的工具和库
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    f1_score,
    calibration_curve
)
```

### 绘图参数
```python
# 设置样式
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

# 生成高分辨率图
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
```

## 🎓 论文使用指南

### 如何引用图表

**主结果**:
> Figure 1: Comprehensive performance analysis of the multimodal CNN model across 5 medical centers. A) Cross-center AUC comparison with 95% confidence intervals; B) Decision curve analysis showing clinical utility; C) Uncertainty quantification across different methods; D) Performance metrics by center; E) Model comparison (multimodal vs single-modality); F) ROC curves.

**方法说明**:
> We used Bootstrap resampling (n=1000) to estimate 95% confidence intervals for all metrics. Decision curve analysis was performed to evaluate clinical net benefit across threshold probabilities from 0.01 to 0.99.

## ✅ 总结

你现在有：
- ✅ 9个专业图表
- ✅ 2个数据表格
- ✅ 完整的方法文档
- ✅ 符合期刊标准

**所有材料已准备好用于论文！**



