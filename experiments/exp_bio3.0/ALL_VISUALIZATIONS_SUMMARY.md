# Bio-COT 3.0 所有可视化类型总结

## 📊 完整可视化清单

### ✅ 已生成的所有可视化（共17种）

---

## 一、基础可视化（已生成）

### 1. Knowledge Notes分布图
- **文件**: `knowledge_notes_distribution_*.png`
- **内容**: PCA降维、小提琴图、箱线图、统计检验
- **意义**: 验证Knowledge Notes的有效性

### 2. Visual Notes注意力分布图
- **文件**: `visual_notes_attention_*.png`
- **内容**: 注意力分布、热图、统计信息
- **意义**: 验证Visual Notes的病灶定位能力

### 3. 火山图
- **文件**: `volcano_plot_*.png`
- **内容**: Fold Change vs p-value，显著特征标记
- **意义**: 识别显著差异的因果特征

### 4. t-SNE/UMAP可视化（2D）
- **文件**: `tsne_umap_*.png`
- **内容**: 2D特征空间分布（按标签、中心、概率着色）
- **意义**: 展示特征空间结构和域不变性

### 5. t-SNE/UMAP可视化（3D）
- **文件**: `tsne_umap_3d_*.pdf`
- **内容**: 3D特征空间分布（PDF矢量图）
- **意义**: 提供更丰富的空间信息

### 6. 综合小提琴图
- **文件**: `violin_plots_comprehensive_*.png`
- **内容**: 预测概率分布（按标签、中心分组）
- **意义**: 评估分类性能和跨中心一致性

### 7. CAM图
- **文件**: `cam_samples_*.png`
- **内容**: 12个样本的Grad-CAM可视化
- **意义**: 提供模型决策的可解释性

### 8. 3D分布可视化
- **文件**: `distribution_3d_*.pdf`
- **内容**: 3D散点图、柱状图、表面图、直方图
- **意义**: 多维度展示预测概率分布

---

## 二、性能评估可视化（新增）

### 9. ROC曲线和PR曲线 ✅
- **文件**: `roc_pr_curves_20260113_152941.pdf`
- **内容**: 
  - ROC曲线（AUC值、最优阈值）
  - PR曲线（AP值、基线对比）
- **意义**: 
  - 评估模型整体分类性能
  - 确定最佳决策阈值
  - 对比不同方法的性能

### 10. 混淆矩阵热图 ✅
- **文件**: `confusion_matrix_20260113_152941.pdf`
- **内容**: 
  - 原始混淆矩阵（计数）
  - 归一化混淆矩阵（百分比）
- **意义**: 
  - 直观展示分类性能
  - 识别错误模式（假阳性vs假阴性）
  - 评估类别不平衡影响

### 11. 校准曲线 ✅
- **文件**: `calibration_curve_20260113_152941.pdf`
- **内容**: 
  - 校准曲线（预测概率vs实际概率）
  - Brier Score
  - 预测概率分布直方图
- **意义**: 
  - 评估预测概率的可靠性
  - 识别过度自信或不足自信的区域
  - 为临床决策提供置信度参考

---

## 三、特征分析可视化（新增）

### 12. 特征重要性分析 ✅
- **文件**: `feature_importance_20260113_152941.pdf`
- **数据文件**: `feature_importance_data_20260113_152941.csv`
- **内容**: 
  - Top 50特征重要性（t-statistic）
  - 特征重要性分布直方图
  - p值分布
  - 特征重要性vs均值差异散点图
- **意义**: 
  - 识别对分类最重要的特征
  - 理解模型的决策依据
  - 为特征工程提供指导

### 13. 特征相关性热图 ✅
- **文件**: `feature_correlation_20260113_152941.pdf`
- **内容**: 
  - Top 50特征的相关性矩阵
  - 相关性分布直方图
- **意义**: 
  - 识别冗余特征
  - 理解特征之间的关系
  - 为特征选择提供指导

---

## 四、域不变性分析可视化（新增）

### 14. 中心性能对比（森林图风格） ✅
- **文件**: `center_comparison_20260113_152941.pdf`
- **数据文件**: `center_comparison_data_20260113_152941.csv`
- **内容**: 
  - 每个中心的Accuracy、Precision、Recall、F1-Score、AUC
  - 样本数量分布
- **意义**: 
  - 验证模型的域不变性
  - 识别性能差异较大的中心
  - 评估多中心研究的有效性

### 15. 因果特征vs噪声特征对比 ✅
- **文件**: `causal_vs_noise_20260113_152941.pdf`
- **内容**: 
  - 因果特征和噪声特征的PCA分布（按标签、中心着色）
  - 分离度对比（类间距离/类内距离）
  - 中心重叠度对比（余弦相似度）
- **意义**: 
  - **验证因果解耦的有效性**: 因果特征应该实现类别分离，噪声特征不应该
  - **验证域不变性**: 因果特征应该跨中心重叠，噪声特征应该跨中心分离
  - **理解模型的表示学习**: 展示模型如何区分因果和噪声特征

---

## 五、错误分析可视化（新增）

### 16. 错误分析可视化 ✅
- **文件**: `error_analysis_20260113_152941.pdf`
- **内容**: 
  - 错误类型分布（正确、假阳性、假阴性）
  - 错误样本的预测概率分布
  - 错误样本在特征空间中的分布
  - 错误率统计
- **意义**: 
  - 理解模型的失败模式
  - 识别困难样本
  - 为模型改进提供方向

### 17. 决策边界可视化 ✅
- **文件**: `decision_boundary_20260113_152941.pdf`
- **内容**: 
  - KNN近似的决策边界
  - 预测概率等高线
- **意义**: 
  - 直观展示模型的决策过程
  - 理解特征空间中的分类区域
  - 识别决策边界的复杂性

---

## 六、训练过程可视化（新增）

### 18. 详细训练曲线 ✅
- **文件**: `training_curves_detailed_20260113_152941.pdf`
- **内容**: 
  - Loss曲线（训练和验证）
  - Accuracy曲线（训练和验证）
  - AUC曲线（验证集）
  - 损失组件详细分析（分类、OT、稀疏、一致性、对抗损失）
- **意义**: 
  - 监控训练过程
  - 识别过拟合
  - 理解不同损失组件的作用

### 19. 注意力演化分析 ✅
- **文件**: `attention_evolution_20260113_152941.pdf`
- **内容**: 
  - OCT和Colposcopy的注意力分布（按标签）
  - 注意力相关性（OCT vs Colposcopy）
  - 注意力稀疏性分析（熵）
- **意义**: 
  - 理解Visual Notes的注意力模式
  - 验证注意力的稀疏性
  - 分析不同模态的注意力差异

---

## 📊 可视化分类总结

### 按用途分类

1. **性能评估** (3个):
   - ROC/PR曲线
   - 混淆矩阵
   - 校准曲线

2. **特征分析** (4个):
   - 特征重要性分析
   - 特征相关性热图
   - 火山图
   - Knowledge Notes分布

3. **域不变性验证** (3个):
   - 中心性能对比
   - 因果vs噪声特征对比
   - t-SNE/UMAP可视化

4. **可解释性** (3个):
   - CAM图
   - Visual Notes注意力分布
   - 决策边界可视化

5. **错误分析** (1个):
   - 错误分析可视化

6. **训练过程** (2个):
   - 详细训练曲线
   - 注意力演化分析

7. **分布分析** (1个):
   - 综合小提琴图
   - 3D分布可视化

---

## 🎯 论文配图建议

### 主图（Figure 1-3）
1. **Figure 1**: 方法架构图（已有）
2. **Figure 2**: 
   - ROC/PR曲线（性能评估）
   - 混淆矩阵（分类性能）
   - 中心性能对比（域不变性）
3. **Figure 3**: 
   - t-SNE/UMAP可视化（特征空间）
   - 因果vs噪声特征对比（因果解耦验证）

### 方法验证图（Figure 4-6）
4. **Figure 4**: 
   - Knowledge Notes分布（Knowledge Notes有效性）
   - Visual Notes注意力热图（Visual Notes有效性）
5. **Figure 5**: 
   - CAM图（可解释性）
   - 注意力演化分析（注意力机制）
6. **Figure 6**: 
   - 特征重要性分析（特征贡献）
   - 火山图（显著特征）

### 补充材料
- 详细训练曲线
- 错误分析
- 校准曲线
- 特征相关性热图
- 决策边界可视化
- 3D可视化（t-SNE/UMAP、分布）

---

## 📁 文件位置

所有可视化文件保存在：
```
/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/logs/
```

### 2D可视化（PNG格式）
- `knowledge_notes_distribution_*.png`
- `visual_notes_attention_*.png`
- `volcano_plot_*.png`
- `tsne_umap_*.png`
- `violin_plots_comprehensive_*.png`
- `cam_samples_*.png`

### 3D可视化（PDF格式）
- `tsne_umap_3d_*.pdf`
- `distribution_3d_*.pdf`

### 补充可视化（PDF格式）
- `roc_pr_curves_*.pdf`
- `confusion_matrix_*.pdf`
- `calibration_curve_*.pdf`
- `feature_importance_*.pdf`
- `center_comparison_*.pdf`
- `error_analysis_*.pdf`
- `causal_vs_noise_*.pdf`
- `training_curves_detailed_*.pdf`
- `attention_evolution_*.pdf`
- `feature_correlation_*.pdf`
- `decision_boundary_*.pdf`

### 数据文件
- `visualization_data_*.pkl` - 完整特征数据
- `visualization_data_*.csv` - CSV格式数据
- `feature_importance_data_*.csv` - 特征重要性数据
- `center_comparison_data_*.csv` - 中心对比数据

---

## 🚀 生成脚本

### 基础可视化
```bash
python generate_sci_visualizations.py
```

### 3D可视化
```bash
python generate_3d_visualizations.py
```

### 补充可视化
```bash
python generate_additional_visualizations.py
```

---

## 📝 总结

✅ **总共19种可视化类型**，全面覆盖：
- 性能评估
- 特征分析
- 域不变性验证
- 可解释性
- 错误分析
- 训练过程

✅ **所有可视化均以PDF格式保存**，适合SCI顶刊发表

✅ **包含详细的数据文件**，便于后续分析和复现

---

**最后更新**: 2025-01-13  
**可视化总数**: 19种  
**文件格式**: PDF（矢量图）+ PNG（位图）  
**分辨率**: 300 DPI

