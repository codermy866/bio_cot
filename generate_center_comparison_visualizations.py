#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成多中心对比的可视化图表（替代Conditional_Means的紧凑版本）
1. 箱线图（Boxplot）- 简洁展示各中心分布
2. 热力图（Heatmap）- 展示特征×中心的矩阵
3. 分组条形图（Grouped Bar Chart）- 展示各中心均值对比
4. 点图（Dot Plot）- 展示均值和置信区间
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
from scipy.stats import f_oneway, kruskal

# 配置路径
ROOT = Path(__file__).parent
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
    'text': '#000000',
}

# 中心名称映射（使用Center A/B/C/D/E格式）
CENTER_NAMES = {
    0: 'Center A',
    1: 'Center B',
    2: 'Center C',
    3: 'Center D',
    4: 'Center E'
}

def get_center_name(center_id):
    """获取中心显示名称"""
    return CENTER_NAMES.get(int(center_id), f'Center {chr(65 + int(center_id))}')

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

# 加载数据
print("=" * 80)
print("🎨 多中心对比可视化生成（紧凑版本）")
print("=" * 80)

data_file = ROOT / 'visualization' / 'data' / 'Complete_Dataset.csv'
if data_file.exists():
    print(f"📂 加载数据: {data_file}")
    feature_df = pd.read_csv(data_file)
    
    # 检查列名
    if 'label' in feature_df.columns:
        feature_df['Label'] = feature_df['label']
    if 'center_id' in feature_df.columns:
        feature_df['Center'] = feature_df['center_id']
    
    if 'Label' not in feature_df.columns or 'Center' not in feature_df.columns:
        print("⚠️  未找到Label或Center列，使用模拟数据")
        feature_df = None
else:
    print("⚠️  数据文件不存在，使用模拟数据")
    feature_df = None

# 如果没有真实数据，使用模拟数据
if feature_df is None:
    print("💡 使用模拟数据...")
    np.random.seed(42)
    n_samples = 600
    n_features = 10
    
    all_features = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, 2, n_samples)
    centers = np.random.randint(0, 5, n_samples)
    
    feature_df = pd.DataFrame(all_features, columns=[f'feature_{i}' for i in range(n_features)])
    feature_df['Label'] = labels
    feature_df['Center'] = centers

print(f"✅ 数据加载完成: {len(feature_df)} 个样本")

# ============================================================================
# 1. 箱线图（Boxplot）- 最简洁的版本
# ============================================================================
print("\n" + "=" * 80)
print("🎨 图表 1: 箱线图（Boxplot）- 各中心特征分布对比")
print("=" * 80)

try:
    if 'Label' in feature_df.columns and 'Center' in feature_df.columns:
        # 选择前4个最重要的特征
        feature_cols = [col for col in feature_df.columns if col.startswith('feature_')][:4]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor('#FFFFFF')  # 纯白背景，更专业
        axes = axes.flatten()
        
        # D69584到C7CCD6渐变配色系列
        center_colors = [
            '#D69584',  # Center A
            '#D2A392',  # Center B
            '#CEB1A0',  # Center C
            '#CABFAE',  # Center D
            '#C7CCD6',  # Center E
        ]
        
        for idx, feat_col in enumerate(feature_cols):
            ax = axes[idx]
            ax.set_facecolor('#FFFFFF')
            
            # 准备数据
            centers = sorted(feature_df['Center'].unique())
            data_list = []
            labels_list = []
            
            for center in centers:
                mask = feature_df['Center'] == center
                data = feature_df.loc[mask, feat_col].values
                data_list.append(data)
                labels_list.append(get_center_name(center))
            
            # 绘制箱线图 - 顶刊级别样式
            bp = ax.boxplot(data_list, labels=labels_list, patch_artist=True,
                           boxprops=dict(facecolor='white', alpha=0.85, linewidth=2.0, edgecolor='#2C3E50'),
                           medianprops=dict(color='#2C3E50', linewidth=2.5),
                           whiskerprops=dict(color='#2C3E50', linewidth=2.0),
                           capprops=dict(color='#2C3E50', linewidth=2.0),
                           flierprops=dict(marker='o', markersize=5, alpha=0.6, markeredgecolor='#2C3E50', 
                                         markeredgewidth=1.0, markerfacecolor='none'))
            
            # 设置颜色 - 使用顶刊级别配色
            for i, patch in enumerate(bp['boxes']):
                color = center_colors[i % len(center_colors)]
                patch.set_facecolor(color)
                patch.set_alpha(0.75)  # 适度的透明度，更优雅
                patch.set_edgecolor('#2C3E50')  # 深色边框，更专业
                patch.set_linewidth(2.0)
            
            # 设置标签 - 顶刊级别样式
            feature_name = FEATURE_NAMES.get(feat_col, feat_col)
            ax.set_ylabel(feature_name, fontsize=12, fontweight='bold', color='#2C3E50')
            ax.set_xlabel('Medical Center', fontsize=12, fontweight='bold', color='#2C3E50')
            ax.set_title(f'{feature_name} Distribution by Center', 
                        fontsize=13, fontweight='bold', pad=12, color='#2C3E50')
            
            # 网格和刻度 - 更专业的样式
            ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, axis='y', color='#95A5A6')
            ax.tick_params(axis='x', rotation=45, colors='#2C3E50', labelsize=10)
            ax.tick_params(axis='y', colors='#2C3E50', labelsize=10)
            
            # 设置边框样式
            for spine in ax.spines.values():
                spine.set_color('#BDC3C7')
                spine.set_linewidth(1.5)
        
        plt.suptitle('Feature Distribution Comparison Across Medical Centers (Boxplot)', 
                    fontsize=15, fontweight='bold', y=0.995, color='#2C3E50')
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(FIGURES_DIR / 'Center_Comparison_Boxplot.png', dpi=300, bbox_inches='tight',
                   facecolor='#FFFFFF', edgecolor='none')
        plt.savefig(FIGURES_DIR / 'Center_Comparison_Boxplot.pdf', dpi=300, bbox_inches='tight',
                   facecolor='#FFFFFF', edgecolor='none')
        plt.close()
        print("✅ 箱线图 已生成")
except Exception as e:
    print(f"⚠️  箱线图 生成失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 2. 热力图（Heatmap）- 展示特征×中心的均值矩阵
# ============================================================================
print("\n" + "=" * 80)
print("🎨 图表 2: 热力图（Heatmap）- 特征×中心均值矩阵")
print("=" * 80)

try:
    if 'Label' in feature_df.columns and 'Center' in feature_df.columns:
        # 智能选择特征：基于各中心间差异最大的特征
        all_feature_cols = [col for col in feature_df.columns if col.startswith('feature_')]
        centers = sorted(feature_df['Center'].unique())
        
        # 计算每个特征在各中心间的差异（使用方差或标准差）
        feature_variance_scores = []
        for feat_col in all_feature_cols:
            center_means = []
            for center in centers:
                mask = feature_df['Center'] == center
                mean_val = feature_df.loc[mask, feat_col].mean()
                center_means.append(mean_val)
            # 计算各中心均值的方差（方差越大，说明各中心差异越大）
            variance = np.var(center_means)
            feature_variance_scores.append((feat_col, variance))
        
        # 按方差排序，选择差异最大的前6个特征
        feature_variance_scores.sort(key=lambda x: x[1], reverse=True)
        selected_features = [col for col, _ in feature_variance_scores[:6]]
        
        print(f"  📊 选择特征（基于中心间差异）: {len(selected_features)}个特征")
        for col, var in feature_variance_scores[:6]:
            print(f"     - {FEATURE_NAMES.get(col, col)}: variance={var:.4f}")
        
        # 计算每个中心每个特征的均值
        heatmap_data = []
        feature_names_list = []
        
        for feat_col in selected_features:
            row = []
            for center in centers:
                mask = feature_df['Center'] == center
                mean_val = feature_df.loc[mask, feat_col].mean()
                row.append(mean_val)
            heatmap_data.append(row)
            feature_names_list.append(FEATURE_NAMES.get(feat_col, feat_col))
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                  index=feature_names_list,
                                  columns=[get_center_name(c) for c in centers])
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(7, 8))
        fig.patch.set_facecolor(COLORS['background'])
        ax.set_facecolor(COLORS['background'])
        
        # 使用经典热图配色：RdBu（红-白-蓝，最经典的科学热图配色）
        # 或者使用 RdYlBu（红-黄-蓝），viridis等
        # 这里使用RdBu_r（反转，使红色表示高值，蓝色表示低值）
        cmap = 'RdBu_r'  # 经典红-白-蓝配色（反转）
        
        # 计算颜色范围（以0为中心，对称分布）
        vmax = max(abs(heatmap_df.values.min()), abs(heatmap_df.values.max()))
        vmin = -vmax
        
        # 绘制热力图，调整colorbar大小和位置
        heatmap_plot = sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap=cmap, 
                   cbar_kws={'label': 'Mean Feature Value', 
                            'shrink': 0.4,  # 进一步缩小colorbar高度（从0.5到0.4）
                            'aspect': 25,   # 调整colorbar宽高比（更细）
                            'pad': 0.05},   # 调整colorbar与图的间距（稍微增加）
                   linewidths=1.0, linecolor='white',
                   square=True, ax=ax, 
                   vmin=vmin, vmax=vmax,
                   center=0,  # 以0为中心，使配色对称
                   annot_kws={'fontsize': 9, 'fontweight': 'bold'})
        
        # 获取colorbar并进一步调整
        cbar = heatmap_plot.collections[0].colorbar
        if cbar is not None:
            cbar.ax.tick_params(labelsize=9)  # 减小colorbar标签字体
            cbar.set_label('Mean Feature Value', fontsize=10, fontweight='bold')  # 减小colorbar标题字体
        
        # 调整标题位置，确保不被colorbar遮挡（y值更高）
        ax.set_title('Mean Feature Values Across Medical Centers (Heatmap)', 
                    fontsize=14, fontweight='bold', pad=20, y=1.05)
        ax.set_xlabel('Medical Center', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'Center_Comparison_Heatmap.png', dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.savefig(FIGURES_DIR / 'Center_Comparison_Heatmap.pdf', dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.close()
        print("✅ 热力图 已生成")
except Exception as e:
    print(f"⚠️  热力图 生成失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 3. 分组条形图（Grouped Bar Chart）- 展示各中心均值对比
# ============================================================================
print("\n" + "=" * 80)
print("🎨 图表 3: 分组条形图（Grouped Bar Chart）- 各中心均值对比")
print("=" * 80)

try:
    if 'Label' in feature_df.columns and 'Center' in feature_df.columns:
        # 选择前6个特征
        feature_cols = [col for col in feature_df.columns if col.startswith('feature_')][:6]
        centers = sorted(feature_df['Center'].unique())
        
        # 计算均值和标准差
        means_data = []
        stds_data = []
        feature_names_list = []
        
        for feat_col in feature_cols:
            row_means = []
            row_stds = []
            for center in centers:
                mask = feature_df['Center'] == center
                mean_val = feature_df.loc[mask, feat_col].mean()
                std_val = feature_df.loc[mask, feat_col].std()
                row_means.append(mean_val)
                row_stds.append(std_val)
            means_data.append(row_means)
            stds_data.append(row_stds)
            feature_names_list.append(FEATURE_NAMES.get(feat_col, feat_col))
        
        means_df = pd.DataFrame(means_data, 
                               index=feature_names_list,
                               columns=[get_center_name(c) for c in centers])
        stds_df = pd.DataFrame(stds_data,
                              index=feature_names_list,
                              columns=[get_center_name(c) for c in centers])
        
        # 绘制分组条形图
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(COLORS['background'])
        ax.set_facecolor(COLORS['background'])
        
        x = np.arange(len(feature_names_list))
        width = 0.15  # 每个bar的宽度
        n_centers = len(centers)
        
        for i, center_name in enumerate(means_df.columns):
            offset = (i - n_centers/2 + 0.5) * width
            ax.bar(x + offset, means_df[center_name].values, width,
                  label=center_name, color=deeper_center_colors[i % len(deeper_center_colors)],
                  alpha=0.8, edgecolor='white', linewidth=1.0,
                  yerr=stds_df[center_name].values, capsize=3, error_kw={'linewidth': 1.5})
        
        ax.set_xlabel('Feature', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Feature Value', fontsize=12, fontweight='bold')
        ax.set_title('Mean Feature Values Across Medical Centers (Grouped Bar Chart)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names_list, rotation=45, ha='right', fontsize=10)
        ax.legend(loc='best', fontsize=9, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'Center_Comparison_GroupedBar.png', dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.savefig(FIGURES_DIR / 'Center_Comparison_GroupedBar.pdf', dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.close()
        print("✅ 分组条形图 已生成")
except Exception as e:
    print(f"⚠️  分组条形图 生成失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 4. 点图（Dot Plot）- 展示均值和置信区间
# ============================================================================
print("\n" + "=" * 80)
print("🎨 图表 4: 点图（Dot Plot）- 均值和置信区间")
print("=" * 80)

try:
    if 'Label' in feature_df.columns and 'Center' in feature_df.columns:
        # 选择前4个特征
        feature_cols = [col for col in feature_df.columns if col.startswith('feature_')][:4]
        centers = sorted(feature_df['Center'].unique())
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor(COLORS['background'])
        axes = axes.flatten()
        
        for idx, feat_col in enumerate(feature_cols):
            ax = axes[idx]
            ax.set_facecolor(COLORS['background'])
            
            # 计算均值和标准差
            means = []
            stds = []
            center_names = []
            
            for center in centers:
                mask = feature_df['Center'] == center
                data = feature_df.loc[mask, feat_col].values
                means.append(data.mean())
                stds.append(data.std())
                center_names.append(get_center_name(center))
            
            x_pos = np.arange(len(centers))
            
            # 绘制点图和误差棒
            for i, (mean_val, std_val, center_name) in enumerate(zip(means, stds, center_names)):
                ax.errorbar(i, mean_val, yerr=std_val, fmt='o', 
                           markersize=12, capsize=6, capthick=2,
                           color=deeper_center_colors[i % len(deeper_center_colors)],
                           linewidth=2.5, alpha=0.8,
                           label=center_name if idx == 0 else '')
            
            # 连接均值点
            ax.plot(x_pos, means, '--', color='gray', alpha=0.5, linewidth=1.5, zorder=0)
            
            feature_name = FEATURE_NAMES.get(feat_col, feat_col)
            ax.set_ylabel(feature_name, fontsize=11, fontweight='bold')
            ax.set_xlabel('Medical Center', fontsize=11, fontweight='bold')
            ax.set_title(f'{feature_name} Mean ± Std by Center', fontsize=12, fontweight='bold', pad=10)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(center_names, rotation=45, ha='right', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
            
            if idx == 0:
                ax.legend(loc='best', fontsize=8, framealpha=0.95)
        
        plt.suptitle('Mean Feature Values with Standard Deviation Across Centers (Dot Plot)', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'Center_Comparison_DotPlot.png', dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.savefig(FIGURES_DIR / 'Center_Comparison_DotPlot.pdf', dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.close()
        print("✅ 点图 已生成")
except Exception as e:
    print(f"⚠️  点图 生成失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 5. 小提琴图（Violin Plot）- 展示分布形状
# ============================================================================
print("\n" + "=" * 80)
print("🎨 图表 5: 小提琴图（Violin Plot）- 分布形状对比")
print("=" * 80)

try:
    if 'Label' in feature_df.columns and 'Center' in feature_df.columns:
        # 选择前4个特征
        feature_cols = [col for col in feature_df.columns if col.startswith('feature_')][:4]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor(COLORS['background'])
        axes = axes.flatten()
        
        for idx, feat_col in enumerate(feature_cols):
            ax = axes[idx]
            ax.set_facecolor(COLORS['background'])
            
            # 准备数据
            plot_data = []
            for center in sorted(feature_df['Center'].unique()):
                mask = feature_df['Center'] == center
                data = feature_df.loc[mask, feat_col].values
                plot_data.append(data)
            
            # 绘制小提琴图
            parts = ax.violinplot(plot_data, positions=range(len(plot_data)), 
                                 showmeans=True, showmedians=True,
                                 widths=0.6)
            
            # 设置颜色
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(deeper_center_colors[i % len(deeper_center_colors)])
                pc.set_alpha(0.7)
            
            # 设置标签
            center_names = [get_center_name(c) for c in sorted(feature_df['Center'].unique())]
            ax.set_xticks(range(len(center_names)))
            ax.set_xticklabels(center_names, rotation=45, ha='right', fontsize=10)
            
            feature_name = FEATURE_NAMES.get(feat_col, feat_col)
            ax.set_ylabel(feature_name, fontsize=11, fontweight='bold')
            ax.set_xlabel('Medical Center', fontsize=11, fontweight='bold')
            ax.set_title(f'{feature_name} Distribution Shape by Center', fontsize=12, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        
        plt.suptitle('Feature Distribution Shape Comparison Across Medical Centers (Violin Plot)', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'Center_Comparison_Violin.png', dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.savefig(FIGURES_DIR / 'Center_Comparison_Violin.pdf', dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.close()
        print("✅ 小提琴图 已生成")
except Exception as e:
    print(f"⚠️  小提琴图 生成失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("✅ 多中心对比可视化图表生成完成！")
print("=" * 80)
print(f"📁 输出目录: {FIGURES_DIR}")
print("\n生成的文件:")
for fig_file in sorted(FIGURES_DIR.glob('Center_Comparison*.png')):
    size_kb = fig_file.stat().st_size / 1024
    print(f"  ✅ {fig_file.name} ({size_kb:.1f} KB)")
print("\n推荐使用（按紧凑程度排序）：")
print("  1. 热力图（Heatmap）- 最紧凑，信息量最大")
print("  2. 箱线图（Boxplot）- 简洁，展示分布")
print("  3. 分组条形图（Grouped Bar）- 清晰展示均值对比")
print("  4. 点图（Dot Plot）- 展示均值和误差")
print("  5. 小提琴图（Violin Plot）- 展示分布形状")
print("=" * 80)

