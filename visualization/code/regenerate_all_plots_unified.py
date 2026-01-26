#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一重新生成所有可视化图片
- 使用更新的鲜明莫兰迪色系配色
- 确保所有特征名称都使用具体名称而非 feature_0, feature_1 等
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# 设置路径
CODE_DIR = Path(__file__).parent
VIS_DIR = CODE_DIR.parent
FIGURES_DIR = VIS_DIR / 'figures'

# 确保输出目录存在
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🎨 统一重新生成所有可视化图片")
print("=" * 80)
print(f"📁 输出目录: {FIGURES_DIR}")
print()

# 导入所有可视化函数
print("📦 导入可视化模块...")
from generate_all_plots import (
    generate_sample_data,
    plot_pair_plot,
    plot_violin_plot,
    plot_box_plot,
    plot_kde,
    plot_correlation_heatmap,
    plot_pca_3d,
    plot_density_scatter,
    plot_hierarchical_clustering,
    plot_raincloud,
    plot_3d_multipeak
)

from generate_additional_plots import (
    plot_venn_diagram,
    plot_volcano,
    plot_survival_curves,
    plot_feature_heatmap,
    plot_bean_plot,
    plot_3d_surface
)

from visualize_results_v2 import (
    plot_roc_curves_comparison,
    plot_tsne_3d_visualization,
    plot_umap_3d_visualization,
    plot_confusion_matrix
)

print("✅ 模块导入完成")
print()

# 生成模拟数据
print("📊 生成模拟数据...")
df = generate_sample_data()
print(f"✅ 数据生成完成: {len(df)} 个样本")
print()

# 生成所有图表
print("=" * 80)
print("🎨 开始生成可视化图片...")
print("=" * 80)
print()

# 1. 基础图表（generate_all_plots.py）
print("\n1️⃣ 生成基础图表...")
plot_pair_plot(df, FIGURES_DIR)
plot_violin_plot(df, FIGURES_DIR)
plot_box_plot(df, FIGURES_DIR)
plot_kde(df, FIGURES_DIR)
plot_correlation_heatmap(df, FIGURES_DIR)
plot_pca_3d(df, FIGURES_DIR)
plot_density_scatter(df, FIGURES_DIR)
plot_hierarchical_clustering(df, FIGURES_DIR)
plot_raincloud(df, FIGURES_DIR)
plot_3d_multipeak(df, FIGURES_DIR)
print("✅ 基础图表生成完成")

# 2. 扩展图表（generate_additional_plots.py）
print("\n2️⃣ 生成扩展图表...")
plot_venn_diagram(FIGURES_DIR)  # 只需要save_dir参数
plot_volcano(df, FIGURES_DIR)
plot_survival_curves(df, FIGURES_DIR)
plot_feature_heatmap(df, FIGURES_DIR)
plot_bean_plot(df, FIGURES_DIR)
plot_3d_surface(df, FIGURES_DIR)
print("✅ 扩展图表生成完成")

# 3. 降维可视化（visualize_results_v2.py）
print("\n3️⃣ 生成降维可视化...")
# 准备数据
features = df[[f'feature_{i}' for i in range(10)]].values
labels = df['label'].values
center_ids = df['center_id'].values

# 生成ROC曲线（需要模拟数据）
import numpy as np
from sklearn.metrics import roc_curve, auc
results_dict = {
    'Bio-COT 3.2': {
        'fpr': np.linspace(0, 1, 100),
        'tpr': np.linspace(0, 1, 100) ** 0.8,
        'auc': 0.92
    },
    'Baseline': {
        'fpr': np.linspace(0, 1, 100),
        'tpr': np.linspace(0, 1, 100) ** 0.6,
        'auc': 0.85
    }
}
results_dict['Bio-COT 3.2']['auc'] = auc(results_dict['Bio-COT 3.2']['fpr'], results_dict['Bio-COT 3.2']['tpr'])
results_dict['Baseline']['auc'] = auc(results_dict['Baseline']['fpr'], results_dict['Baseline']['tpr'])

plot_roc_curves_comparison(results_dict, FIGURES_DIR)

# 生成3D t-SNE和UMAP（可能需要较长时间）
print("   ⏳ 生成3D t-SNE（可能需要几分钟）...")
plot_tsne_3d_visualization(features, labels, center_ids, FIGURES_DIR)
print("   ⏳ 生成3D UMAP（可能需要几分钟）...")
plot_umap_3d_visualization(features, labels, center_ids, FIGURES_DIR)

# 生成混淆矩阵
from sklearn.metrics import confusion_matrix
y_true = df['label'].values
y_pred = (df['auc'] > 0.8).astype(int)  # 模拟预测
cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(y_true, y_pred, FIGURES_DIR)  # 修正参数顺序

print("✅ 降维可视化生成完成")

print()
print("=" * 80)
print("✅ 所有可视化图片生成完成！")
print("=" * 80)
print(f"📁 输出目录: {FIGURES_DIR}")
print(f"📊 共生成 {len(list(FIGURES_DIR.glob('*.png')))} 张PNG图片")
print(f"📄 共生成 {len(list(FIGURES_DIR.glob('*.pdf')))} 张PDF图片")
print()
print("🎨 配色方案: 鲜明莫兰迪色系（高区分度）")
print("📝 特征名称: 已全部使用具体名称（OCT Texture, OCT Intensity等）")
print("=" * 80)

