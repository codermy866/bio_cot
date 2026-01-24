# Bio-COT 3.0 可视化快速查看指南

## 📊 已生成的可视化文件

### 2D可视化（PNG格式）
- `knowledge_notes_distribution_20260113_144429.png` - Knowledge Notes分布
- `visual_notes_attention_20260113_144429.png` - Visual Notes注意力分布
- `volcano_plot_20260113_144429.png` - 火山图
- `tsne_umap_20260113_144429.png` - t-SNE/UMAP可视化
- `violin_plots_comprehensive_20260113_144429.png` - 综合小提琴图
- `cam_samples_20260113_144429.png` - CAM图

### 3D可视化（PDF格式，矢量图）
- `tsne_umap_3d_20260113_145816.pdf` - 3D t-SNE/UMAP可视化
- `distribution_3d_20260113_145816.pdf` - 3D分布可视化

### 数据文件
- `visualization_data_20260113_145816.pkl` - 完整特征数据
- `visualization_data_20260113_145816.csv` - CSV格式数据
- `visualization_code_20260113_145816.py` - 可视化代码
- `visualization_config_20260113_145816.json` - 配置信息

---

## 🚀 快速查看

### 查看PNG图片
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/logs
# 使用系统图片查看器打开
xdg-open *.png  # Linux
open *.png      # macOS
```

### 查看PDF文件
```bash
# 使用PDF阅读器打开（支持旋转观察3D图表）
evince tsne_umap_3d_20260113_145816.pdf  # Linux
open distribution_3d_20260113_145816.pdf  # macOS
```

### 加载数据进行分析
```python
import pickle
import pandas as pd
import numpy as np

# 加载完整特征数据
with open('logs/visualization_data_20260113_145816.pkl', 'rb') as f:
    features_dict = pickle.load(f)

# 加载CSV数据
df = pd.read_csv('logs/visualization_data_20260113_145816.csv')

# 查看数据
print(f"样本数量: {len(df)}")
print(f"标签分布:\n{df['label'].value_counts()}")
print(f"中心分布:\n{df['center'].value_counts()}")
```

---

## 📝 详细分析报告

请查看 `VISUALIZATION_ANALYSIS_REPORT.md` 和 `DETAILED_VISUALIZATION_ANALYSIS.md` 获取：
- 每个图表的详细含义
- 实验结果解读
- 绘制质量评估
- 改进建议

---

**最后更新**: 2025-01-13

