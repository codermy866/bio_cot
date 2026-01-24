# Bio-COT 3.0 Improved Experiment

## 📁 文件夹结构

```
exp_bio3.0_improved/
├── logs/              # 日志文件、数据文件
├── checkpoints/       # 模型检查点
├── results/           # 实验结果
├── visualizations/    # 可视化图片（PDF和PNG）
└── README.md         # 本文件
```

## 📊 实验信息

### 实验日期
- 创建时间: 2025-01-13
- 数据来源: exp_bio3.0

### 关键发现
1. **分类性能问题**: 初始准确率只有53.57%
2. **最优阈值**: 0.580（而非默认0.5）
3. **性能提升**: 使用最优阈值后，准确率提升到73.81%

### 主要改进
- ✅ 找到最优决策阈值（0.580）
- ✅ 性能提升：准确率 +20.24%，特异性 +41.59%
- ✅ 生成了完整的可视化分析

## 📈 性能指标

### 当前性能（阈值=0.580）
- **准确率 (Accuracy)**: 73.81%
- **精确率 (Precision)**: 58.46%
- **召回率 (Recall)**: 69.09%
- **特异性 (Specificity)**: 76.11%
- **F1分数**: 63.33%
- **ROC AUC**: 79.80%

### 改进前性能（阈值=0.5）
- **准确率**: 53.57%
- **精确率**: 40.80%
- **召回率**: 92.73%
- **特异性**: 34.51%
- **F1分数**: 56.67%

## 📁 文件说明

### 日志文件
- `training_history_*.json`: 训练历史记录
- `train_bio_cot_v3_*.log`: 训练日志
- `optimal_threshold.txt`: 最优阈值

### 数据文件
- `visualization_data_*.pkl`: 完整特征数据
- `visualization_data_*.csv`: CSV格式数据
- `feature_importance_data_*.csv`: 特征重要性数据
- `center_comparison_data_*.csv`: 中心对比数据

### 可视化文件
- `roc_pr_curves_*.pdf`: ROC和PR曲线
- `confusion_matrix_*.pdf`: 混淆矩阵
- `calibration_curve_*.pdf`: 校准曲线
- `feature_importance_*.pdf`: 特征重要性分析
- `center_comparison_*.pdf`: 中心性能对比
- `error_analysis_*.pdf`: 错误分析
- `causal_vs_noise_*.pdf`: 因果vs噪声特征对比
- `training_curves_detailed_*.pdf`: 详细训练曲线
- `attention_evolution_*.pdf`: 注意力演化分析
- `feature_correlation_*.pdf`: 特征相关性热图
- `decision_boundary_*.pdf`: 决策边界可视化
- `tsne_umap_3d_*.pdf`: 3D t-SNE/UMAP可视化
- `distribution_3d_*.pdf`: 3D分布可视化
- `classification_improvement_comparison.pdf`: 性能改进对比图

## 🎯 下一步改进方向

1. **调整损失函数权重**
   - 增加分类损失权重（lambda_cls从1.0增加到2.0）
   - 减少其他损失权重

2. **调整类别权重**
   - 在Focal Loss中调整alpha和gamma参数
   - 处理类别不平衡问题

3. **重新训练模型**
   - 使用调整后的参数重新训练
   - 预期准确率可提升到75-80%

## 📝 相关文档

- `../exp_bio3.0/CLASSIFICATION_PERFORMANCE_ANALYSIS.md`: 详细性能分析
- `../exp_bio3.0/ALL_VISUALIZATIONS_SUMMARY.md`: 所有可视化类型总结

---
**最后更新**: 2025-01-13
