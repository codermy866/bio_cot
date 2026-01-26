#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高级可视化图表生成 - 基于最佳模型权重
1. Different cubehelix palettes
2. Regression fit over a strip plot
3. Plotting large distributions
4. Scatterplot Matrix
5. Conditional means with observations
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# 配置路径
ROOT = Path(__file__).parent
CHECKPOINT_PATH = ROOT / 'checkpoints' / 'best_model_v3_20260126_111515.pth'
OUTPUT_DIR = ROOT / 'newlog_0126'
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# 设置字体和样式
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['figure.dpi'] = 300

# 统一配色：D69584到C7CCD6
COLORS = {
    'positive': '#D69584',
    'negative': '#C7CCD6',
    'center_0': '#D69584',
    'center_1': '#D2A392',
    'center_2': '#CEB1A0',
    'center_3': '#CABFAE',
    'center_4': '#C7CCD6',
    'background': '#FAFAFA',
}

# 标签映射函数
def get_label_name(label_val):
    """将Label值转换为英文显示名称"""
    if label_val == 0:
        return 'Negative'
    elif label_val == 1:
        return 'Positive'
    else:
        return f'Label {label_val}'

# 特征名称映射
FEATURE_NAMES = {
    'feature_0': 'OCT Texture',
    'feature_1': 'OCT Intensity',
    'feature_2': 'OCT Morphology',
    'feature_3': 'Colposcopy Texture',
    'feature_4': 'Colposcopy Color',
    'feature_5': 'Colposcopy Vascular',
    'feature_6': 'Clinical Age',
    'feature_7': 'Clinical HPV',
    'feature_8': 'Clinical TCT',
    'feature_9': 'Fused Feature',
}

print("=" * 80)
print("🎨 高级可视化图表生成")
print("=" * 80)

# ========================================
# 1. 加载数据
# ========================================
print("\n📊 步骤 1: 加载数据...")

data_file = ROOT / 'visualization' / 'data' / 'Complete_Dataset.csv'
if data_file.exists():
    print(f"📖 从文件加载数据: {data_file}")
    df = pd.read_csv(data_file)
    
    feature_cols = [f'feature_{i}' for i in range(10) if f'feature_{i}' in df.columns]
    if feature_cols:
        features = df[feature_cols].values
        labels = df['label'].values if 'label' in df.columns else None
        center_ids = df['center_id'].values if 'center_id' in df.columns else None
        
        # 创建特征DataFrame
        feature_df = pd.DataFrame(features, columns=[FEATURE_NAMES.get(f'feature_{i}', f'Feature {i}') 
                                                     for i in range(len(feature_cols))])
        
        if labels is not None:
            feature_df['Label'] = labels
            feature_df['Label_Name'] = feature_df['Label'].map({0: 'Negative', 1: 'Positive'})
        if center_ids is not None:
            feature_df['Center'] = center_ids
        
        print(f"✅ 加载了 {len(features)} 个样本，{len(feature_cols)} 个特征")
    else:
        print("⚠️  未找到特征列，使用模拟数据")
        feature_df = None
else:
    print("⚠️  数据文件不存在，使用模拟数据")
    feature_df = None

if feature_df is None:
    print("💡 生成模拟特征数据...")
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    features = np.random.randn(n_samples, n_features)
    labels = (features[:, 0] + features[:, 1] > 0).astype(int)
    center_ids = (features[:, 2] * 2 + 2).astype(int) % 5
    
    feature_df = pd.DataFrame(features, columns=[FEATURE_NAMES.get(f'feature_{i}', f'Feature {i}') 
                                                 for i in range(n_features)])
    feature_df['Label'] = labels
    feature_df['Label_Name'] = feature_df['Label'].map({0: 'Negative', 1: 'Positive'})
    feature_df['Center'] = center_ids
    print(f"✅ 生成了 {n_samples} 个模拟样本")

print()

# ========================================
# 2. 生成可视化图表
# ========================================

# 2.1 Different cubehelix palettes
print("=" * 80)
print("🎨 图表 1: Different cubehelix palettes")
print("=" * 80)

try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor(COLORS['background'])
    
    # 准备数据（使用前两个特征）
    x = feature_df.iloc[:, 0].values
    y = feature_df.iloc[:, 1].values
    z = feature_df.iloc[:, 2].values if feature_df.shape[1] > 2 else x + y
    
    # 创建网格
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # 插值Z值
    from scipy.interpolate import griddata
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')
    
    # 不同的cubehelix调色板
    palettes = [
        ('Default', 'cubehelix'),
        ('Start=0.5', 'cubehelix', {'start': 0.5}),
        ('Rot=-0.5', 'cubehelix', {'rot': -0.5}),
        ('Gamma=2', 'cubehelix', {'gamma': 2.0}),
    ]
    
    for idx, (title, palette, *kwargs) in enumerate(palettes):
        ax = axes[idx // 2, idx % 2]
        
        if kwargs:
            cmap = sns.cubehelix_palette(**kwargs[0], as_cmap=True)
        else:
            cmap = sns.cubehelix_palette(as_cmap=True)
        
        im = ax.contourf(Xi, Yi, Zi, levels=20, cmap=cmap, alpha=0.8)
        ax.scatter(x, y, c=z, cmap=cmap, s=30, edgecolors='white', linewidths=0.5)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(feature_df.columns[0], fontsize=10)
        ax.set_ylabel(feature_df.columns[1], fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_facecolor(COLORS['background'])
    
    plt.suptitle('Different Cubehelix Palettes', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Cubehelix_Palettes.png', dpi=300, bbox_inches='tight',
               facecolor=COLORS['background'])
    plt.savefig(FIGURES_DIR / 'Cubehelix_Palettes.pdf', dpi=300, bbox_inches='tight',
               facecolor=COLORS['background'])
    plt.close()
    print("✅ Cubehelix palettes 已生成")
except Exception as e:
    print(f"⚠️  Cubehelix palettes 生成失败: {e}")
    import traceback
    traceback.print_exc()

print()

# 2.2 Regression fit over a strip plot
print("=" * 80)
print("🎨 图表 2: Regression fit over a strip plot")
print("=" * 80)

try:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(COLORS['background'])
    
    # 使用Label作为分组
    if 'Label' in feature_df.columns:
        for idx, feat_col in enumerate(feature_df.columns[:2]):
            if feat_col in ['Label', 'Center']:
                continue
            
            ax = axes[idx]
            
            # Strip plot with regression
            sns.stripplot(data=feature_df, x='Label_Name', y=feat_col, 
                         palette=[COLORS['negative'], COLORS['positive']],
                         alpha=0.6, ax=ax, size=4, order=['Negative', 'Positive'])
            
            # 添加回归线
            for label_val in feature_df['Label'].unique():
                mask = feature_df['Label'] == label_val
                label_name = 'Positive' if label_val == 1 else 'Negative'
                x_vals = np.arange(len(feature_df[mask]))
                y_vals = feature_df.loc[mask, feat_col].values
                
                if len(y_vals) > 1:
                    z = np.polyfit(x_vals, y_vals, 1)
                    p = np.poly1d(z)
                    ax.plot(x_vals, p(x_vals), color=COLORS['positive'] if label_val == 1 else COLORS['negative'],
                           linewidth=2.5, linestyle='--', alpha=0.8, label=f'Fit ({label_name})')
            
            ax.set_title(f'Regression Fit: {feat_col}', fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel('Label', fontsize=10)
            ax.set_ylabel(feat_col, fontsize=10)
            ax.legend(fontsize=9)
            ax.set_facecolor(COLORS['background'])
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Regression Fit over Strip Plot', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'Regression_Strip_Plot.png', dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.savefig(FIGURES_DIR / 'Regression_Strip_Plot.pdf', dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.close()
        print("✅ Regression fit over strip plot 已生成")
    else:
        print("⚠️  未找到Label列，跳过此图表")
except Exception as e:
    print(f"⚠️  Regression fit over strip plot 生成失败: {e}")
    import traceback
    traceback.print_exc()

print()

# 2.3 Plotting large distributions
print("=" * 80)
print("🎨 图表 3: Plotting large distributions")
print("=" * 80)

try:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(COLORS['background'])
    axes = axes.flatten()
    
    # 选择前4个特征
    for idx, feat_col in enumerate(feature_df.columns[:4]):
        if feat_col in ['Label', 'Center']:
            continue
        
        ax = axes[idx]
        data = feature_df[feat_col].values
        
        # 多种分布可视化方法
        # 1. Histogram + KDE
        ax.hist(data, bins=50, density=True, alpha=0.6, color=COLORS['center_2'],
               edgecolor='white', linewidth=0.5)
        
        # KDE曲线
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        ax.plot(x_range, kde(x_range), color=COLORS['positive'], linewidth=2.5, label='KDE')
        
        # 2. Violin plot (如果有分组)
        if 'Label' in feature_df.columns:
            # 在右侧添加violin plot
            ax2 = ax.twinx()
            parts = ax2.violinplot([feature_df[feature_df['Label']==0][feat_col].values,
                                    feature_df[feature_df['Label']==1][feat_col].values],
                                   positions=[data.min() + (data.max()-data.min())*0.8,
                                            data.min() + (data.max()-data.min())*0.9],
                                   widths=(data.max()-data.min())*0.05, showmeans=True)
            for pc in parts['bodies']:
                pc.set_facecolor(COLORS['center_2'])
                pc.set_alpha(0.6)
        
        ax.set_title(f'Distribution: {feat_col}', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(feat_col, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=9)
        ax.set_facecolor(COLORS['background'])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Large Distributions Visualization', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Large_Distributions.png', dpi=300, bbox_inches='tight',
               facecolor=COLORS['background'])
    plt.savefig(FIGURES_DIR / 'Large_Distributions.pdf', dpi=300, bbox_inches='tight',
               facecolor=COLORS['background'])
    plt.close()
    print("✅ Large distributions 已生成")
except Exception as e:
    print(f"⚠️  Large distributions 生成失败: {e}")
    import traceback
    traceback.print_exc()

print()

# 2.4 Scatterplot Matrix
print("=" * 80)
print("🎨 图表 4: Scatterplot Matrix")
print("=" * 80)

try:
    # 选择前6个特征（如果可用）
    feature_cols_for_matrix = [col for col in feature_df.columns if col not in ['Label', 'Center']][:6]
    
    if len(feature_cols_for_matrix) >= 3:
        # 创建pairplot
        if 'Label_Name' in feature_df.columns:
            plot_df = feature_df[feature_cols_for_matrix + ['Label_Name']].copy()
            hue_col = 'Label_Name'
        else:
            plot_df = feature_df[feature_cols_for_matrix + (['Label'] if 'Label' in feature_df.columns else [])].copy()
            hue_col = 'Label' if 'Label' in plot_df.columns else None
        
        # 使用seaborn的pairplot
        g = sns.pairplot(plot_df, hue=hue_col,
                        palette=[COLORS['negative'], COLORS['positive']] if hue_col else None,
                        hue_order=['Negative', 'Positive'] if hue_col == 'Label_Name' else None,
                        diag_kind='kde', plot_kws={'alpha': 0.6, 's': 20},
                        diag_kws={'alpha': 0.7, 'fill': True})
        
        g.fig.suptitle('Scatterplot Matrix', fontsize=14, fontweight='bold', y=1.02)
        g.fig.patch.set_facecolor(COLORS['background'])
        
        # 设置所有子图的背景色
        for ax in g.axes.flatten():
            ax.set_facecolor(COLORS['background'])
        
        plt.savefig(FIGURES_DIR / 'Scatterplot_Matrix.png', dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.savefig(FIGURES_DIR / 'Scatterplot_Matrix.pdf', dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.close()
        print("✅ Scatterplot matrix 已生成")
    else:
        print("⚠️  特征数量不足，跳过scatterplot matrix")
except Exception as e:
    print(f"⚠️  Scatterplot matrix 生成失败: {e}")
    import traceback
    traceback.print_exc()

print()

# 2.5 Conditional means with observations
print("=" * 80)
print("🎨 图表 5: Conditional means with observations")
print("=" * 80)

try:
    if 'Label' in feature_df.columns and 'Center' in feature_df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor(COLORS['background'])
        axes = axes.flatten()
        
        # 选择前4个特征
        for idx, feat_col in enumerate(feature_df.columns[:4]):
            if feat_col in ['Label', 'Center']:
                continue
            
            ax = axes[idx]
            
            # 按Center分组计算条件均值
            centers = sorted(feature_df['Center'].unique())
            colors_list = [COLORS['center_0'], COLORS['center_1'], COLORS['center_2'], 
                          COLORS['center_3'], COLORS['center_4']]
            
            # 绘制原始观测值（按Label分组）
            if 'Label_Name' in feature_df.columns:
                for label_name in ['Negative', 'Positive']:
                    mask = (feature_df['Center'] == centers[0]) & (feature_df['Label_Name'] == label_name)
                    if mask.sum() > 0:
                        data = feature_df.loc[mask, feat_col]
                        x_pos = np.random.normal(0, 0.1, len(data))
                        color = COLORS['positive'] if label_name == 'Positive' else COLORS['negative']
                        ax.scatter(x_pos, data, alpha=0.4, s=30, 
                                  color=color, 
                                  edgecolors='white', linewidths=0.5, label=label_name)
            
            # 绘制原始观测值（按Center分组）
            for i, center in enumerate(centers):
                mask = feature_df['Center'] == center
                data = feature_df.loc[mask, feat_col]
                x_pos = np.random.normal(i, 0.1, len(data))
                ax.scatter(x_pos, data, alpha=0.4, s=30, 
                          color=colors_list[i % len(colors_list)], 
                          edgecolors='white', linewidths=0.5, label=f'Center {center}')
            
            # 绘制条件均值
            means = []
            stds = []
            for center in centers:
                mask = feature_df['Center'] == center
                data = feature_df.loc[mask, feat_col]
                means.append(data.mean())
                stds.append(data.std())
            
            x_positions = np.arange(len(centers))
            ax.errorbar(x_positions, means, yerr=stds, fmt='o', 
                       markersize=10, capsize=5, capthick=2,
                       color=COLORS['positive'], linewidth=2.5,
                       label='Mean ± Std', zorder=10)
            
            # 连接均值点
            ax.plot(x_positions, means, color=COLORS['positive'], 
                   linewidth=2, linestyle='--', alpha=0.7, zorder=9)
            
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f'Center {c}' for c in centers], rotation=45, ha='right')
            ax.set_title(f'Conditional Means: {feat_col}', fontsize=12, fontweight='bold', pad=10)
            ax.set_ylabel(feat_col, fontsize=10)
            ax.legend(fontsize=9, loc='best')
            ax.set_facecolor(COLORS['background'])
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Conditional Means with Observations', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'Conditional_Means.png', dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.savefig(FIGURES_DIR / 'Conditional_Means.pdf', dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.close()
        print("✅ Conditional means with observations 已生成")
    else:
        print("⚠️  缺少Label或Center列，跳过此图表")
except Exception as e:
    print(f"⚠️  Conditional means 生成失败: {e}")
    import traceback
    traceback.print_exc()

print()

# ========================================
# 完成
# ========================================
print("=" * 80)
print("✅ 高级可视化图表生成完成！")
print("=" * 80)
print(f"📁 输出目录: {FIGURES_DIR}")
print()
print("生成的文件:")
for fig_file in sorted(FIGURES_DIR.glob('*.png')):
    if any(keyword in fig_file.name for keyword in ['Cubehelix', 'Regression', 'Large_Distributions', 
                                                     'Scatterplot', 'Conditional']):
        size = fig_file.stat().st_size / 1024
        print(f"  ✅ {fig_file.name} ({size:.1f} KB)")
print("=" * 80)

