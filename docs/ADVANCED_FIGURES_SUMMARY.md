# 高级可视化图表总结 - 新增8种图表

## 🎨 新生成的图表类型

### 1. 小提琴图 (Violin Plot)
**文件**: `violin_plot_performance.png`

**特点**:
- 展示不同中心性能指标的分布
- 包含箱线图（中位数、四分位数）
- 透明的散点叠加
- 双面板（Accuracy + F1）

**用途**: 展示数据分布、检测离群值、对比中心间差异

**优点**:
- 同时显示密度分布和统计信息
- 视觉上更美观
- 比箱线图提供更多信息

---

### 2. 气泡图 (Bubble Chart)
**文件**: `bubble_chart_performance.png`

**特点**:
- X轴：AUC，Y轴：F1
- 气泡大小表示样本数量
- 不同颜色代表不同模型
- 透明度控制视觉层次

**用途**: 多维度比较不同模型的综合性能

**优点**:
- 同时显示3个维度（x, y, size）
- 直观展示性能模式
- 可以快速识别最优模型

---

### 3. 热力图 (Heatmap)
**文件**: `heatmap_simple.png`, `heatmap_clustered.png`

**特点**:
- 颜色深浅表示性能高低
- 数值直接标注在格子里
- 清晰展示模型×指标矩阵

**用途**: 快速对比多模型、多指标的性能

**优点**:
- 一目了然的性能对比
- 数值直接可见
- 适合论文中使用

---

### 4. 聚类热力图 (Clustered Heatmap)
**文件**: `heatmap_clustered.png`

**特点**:
- 自动聚类相似模型/指标
- 揭示性能模式
- 更智能的组织方式

**用途**: 发现性能模式和相似性

**优点**:
- 自动发现数据关系
- 更科学的组织
- 适合探索性分析

---

### 5. 雷达图 (Radar Chart)
**文件**: `radar_chart_models.png`

**特点**:
- 多维度全方位对比
- 5个指标（AUC, Accuracy, F1, Precision, Recall）
- 不同模型用不同颜色
- 填充面积表示综合性能

**用途**: 综合评价模型优劣

**优点**:
- 直观展示综合性能
- 适合多指标评估
- 视觉冲击力强

---

### 6. 山脊图 (Ridge Plot)
**文件**: `ridge_plot_auc.png`

**特点**:
- 展示5个中心AUC的分布密度
- 垂直堆叠
- 可视化分布形状
- 包含均值线

**用途**: 展示数据的分布特征

**优点**:
- 美观的分布可视化
- 可以比较分布形状
- 时尚的图表风格

---

### 7. 桑基图 (Sankey Diagram)
**文件**: `sankey_diagram.png`

**特点**:
- 展示数据流向
- OCT、Colposcopy、临床特征 → 融合 → 输出
- 箭头表示流程

**用途**: 展示模型架构和数据流

**优点**:
- 清晰的流程可视化
- 适合展示架构
- 易于理解

---

### 8. 增强校准图 (Enhanced Calibration)
**文件**: `calibration_enhanced.png`

**特点**:
- 对比校准前后效果
- 对角线表示完美校准
- 标注ECE值
- 双面板对比

**用途**: 展示校准的重要性

**优点**:
- 清晰展示校准效果
- 定量评估（ECE值）
- 对比鲜明

---

## 📊 所有图表清单

### 原有图表 (9个)
1. main_results_figure.png - 主结果图
2. performance_dashboard.png - 性能仪表板
3. true_real_forest_plot.png - Forest plot
4. true_real_performance_comparison.png - 中心对比
5. decision_curve_analysis.png - DCA曲线
6. cost_effectiveness_analysis.png - 成本效益
7. uncertainty_analysis.png - 不确定性
8. reliability_diagram.png - 可靠性图
9. prediction_intervals.png - 预测区间

### 新增图表 (8个)
10. violin_plot_performance.png - 小提琴图
11. bubble_chart_performance.png - 气泡图
12. heatmap_simple.png - 简单热力图
13. heatmap_clustered.png - 聚类热力图
14. radar_chart_models.png - 雷达图
15. ridge_plot_auc.png - 山脊图
16. sankey_diagram.png - 桑基图
17. calibration_enhanced.png - 增强校准图

**总计**: 17个高质量可视化图表！

## 🎯 图表使用建议

### 论文主图 (Figure 1-3)
- **Figure 1**: main_results_figure.png（6面板综合分析）
- **Figure 2**: decision_curve_analysis.png（临床实用性）
- **Figure 3**: radar_chart_models.png（多模型对比）

### 关键附图
- **Figure 4**: violin_plot_performance.png（分中心分布）
- **Figure 5**: bubble_chart_performance.png（多维度性能）
- **Figure 6**: calibration_enhanced.png（校准效果）

### 补充材料
- **Supplementary Figure 1**: performance_dashboard.png
- **Supplementary Figure 2**: heatmap_simple.png
- **Supplementary Figure 3**: ridge_plot_auc.png
- **Supplementary Figure 4**: true_real_forest_plot.png
- **Supplementary Figure 5**: reliability_diagram.png

### 可选附图
- sankey_diagram.png - 展示模型架构
- heatmap_clustered.png - 发现性能模式
- uncertainty_analysis.png - 不确定性量化

## 💡 图表选择策略

### 需要展示分布时 → 小提琴图/山脊图
- 检查数据质量
- 展示中心间差异
- 识别离群值

### 需要多维度对比时 → 气泡图/雷达图
- 综合评价模型
- 展示综合性能
- 发现性能模式

### 需要快速对比时 → 热力图
- 论文中快速引用
- 清晰展示所有指标
- 便于读者理解

### 需要展示流程时 → 桑基图
- 展示模型架构
- 解释工作流程
- Methods部分使用

## 📈 每种图表的优势

| 图表类型 | 展示内容 | 优势 | 适用场景 |
|---------|---------|------|---------|
| 小提琴图 | 分布 + 统计 | 信息丰富 | 分布对比 |
| 气泡图 | 3维数据 | 多维度 | 综合性能 |
| 热力图 | 矩阵数据 | 直观 | 快速对比 |
| 雷达图 | 多指标 | 综合 | 模型评价 |
| 山脊图 | 密度分布 | 美观 | 分布可视化 |
| 桑基图 | 流程 | 清晰 | 架构展示 |
| 校准图 | 校准效果 | 定量 | 评估校准 |

## 🎓 论文中的使用方式

### 在Results中
```markdown
Figure X: Multimodal CNN model performance across 5 medical centers.
A) Violin plot showing Accuracy and F1 distribution by center;
B) Bubble chart comparing models (AUC vs F1);
C) Radar chart for comprehensive model comparison;
D) Ridge plot showing AUC distribution density.
```

### 在Methods中
```markdown
Figure X: Model architecture flow diagram (Sankey diagram) 
showing data flow from three input modalities (OCT, 
Colposcopy, Clinical) through encoders to multimodal 
fusion and final classification output.
```

## ✅ 现在你有的

**总计17个高质量图表**:
- ✅ 9个原有专业图表
- ✅ 8个新增高级图表
- ✅ 小提琴图、气泡图、热力图、雷达图等
- ✅ 所有图表都是300 DPI分辨率
- ✅ 适合直接插入论文

**所有材料已准备完毕！**



