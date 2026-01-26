# Bio-COT 3.2 训练结果和可视化

## 📊 训练结果

本次训练的最佳结果已保存在 `best_metrics.csv` 文件中。

## 🎨 生成的可视化图表

### 已生成的图表（在 `figures/` 目录下）：

1. **Confusion_Matrix_Best_Results.png/pdf**
   - 混淆矩阵可视化
   - 红色背景，高对比度
   - 显示数值和百分比

2. **ROC_Curve_Best_Results.png/pdf**
   - ROC曲线
   - 显示最佳AUC值
   - 使用统一配色方案（DDAB9F到C7CCD6色阶）

3. **Performance_Metrics_Best_Results.png/pdf**
   - 性能指标柱状图
   - 包含：Accuracy, Precision, Recall, Specificity, F1-Score, AUC
   - 使用统一配色方案

### 关于 t-SNE 和 UMAP 图

**注意**：t-SNE 和 UMAP 可视化图需要：
1. 从训练好的模型中提取特征向量
2. 对大量样本进行降维计算（通常需要几分钟到几十分钟）

由于这些计算需要：
- 加载训练好的模型
- 通过验证集提取特征
- 执行降维算法（t-SNE/UMAP）

这些步骤计算时间较长，且需要GPU资源。如果需要生成这些图，请运行：

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate
python visualization/code/generate_best_results_visualization.py
```

## 📁 文件结构

```
newlog_0126/
├── best_metrics.csv              # 最佳性能指标（CSV格式）
├── training_report.txt           # 训练总结报告
├── figures/                      # 可视化图表目录
│   ├── Confusion_Matrix_Best_Results.png
│   ├── Confusion_Matrix_Best_Results.pdf
│   ├── ROC_Curve_Best_Results.png
│   ├── ROC_Curve_Best_Results.pdf
│   ├── Performance_Metrics_Best_Results.png
│   └── Performance_Metrics_Best_Results.pdf
└── README.md                     # 本文件
```

## 🎨 配色方案

所有图表使用统一的配色方案：
- **主色调**：DDAB9F（粉棕色）到 C7CCD6（蓝灰色）的渐变色阶
- **字体**：Calibri（如果系统不支持，自动回退到DejaVu Sans）

## 📝 训练日志

完整的训练日志保存在：
`logs/train_bio_cot_v3.2_final_20260126_111511.log`

