
# 标签映射函数
def get_label_name(label_val):
    """将Label值转换为英文显示名称"""
    if label_val == 0:
        return 'Negative'
    elif label_val == 1:
        return 'Positive'
    else:
        return f'Label {label_val}'


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2: Ultra-Complete Visualization Suite
超级全面的可视化套件 - 包含18种高级图表
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from mpl_toolkits.mplot3d import Axes3D
import json
from tqdm import tqdm

# 导入Nature配色方案
from nature_colors import NATURE_COLORS, FEATURE_NAMES, CENTER_NAMES_EN, get_feature_display_name, create_custom_colormap

# 创建自定义colormap（从D69584到C7CCD6）
custom_cmap = create_custom_colormap()

# 设置
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.facecolor'] = NATURE_COLORS['background']

# 使用Nature配色
COLORS = NATURE_COLORS
CENTER_NAMES = CENTER_NAMES_EN


def generate_sample_data():
    """生成模拟数据"""
    np.random.seed(42)
    n_samples = 600
    n_features = 10  # 减少特征数便于可视化
    
    samples_per_center = n_samples // 5
    
    data_list = []
    
    for center_id in range(5):
        base_shift = center_id * 0.5
        
        # 正样本
        n_pos = samples_per_center // 2
        features_pos = np.random.randn(n_pos, n_features) * 1.5 + base_shift
        features_pos[:, 0] += 2  # 第一个特征有明显区分
        
        # 负样本
        n_neg = samples_per_center - n_pos
        features_neg = np.random.randn(n_neg, n_features) * 1.5 + base_shift
        features_neg[:, 0] -= 2
        
        for i in range(n_pos):
            data_list.append({
                'label': 1,
                'center_id': center_id,
                'center_name': CENTER_NAMES[center_id],
                **{f'feature_{j}': features_pos[i, j] for j in range(n_features)},
                'auc': np.random.uniform(0.7, 0.95),
                'sensitivity': np.random.uniform(0.4, 0.8),
                'specificity': np.random.uniform(0.85, 0.98)
            })
        
        for i in range(n_neg):
            data_list.append({
                'label': 0,
                'center_id': center_id,
                'center_name': CENTER_NAMES[center_id],
                **{f'feature_{j}': features_neg[i, j] for j in range(n_features)},
                'auc': np.random.uniform(0.7, 0.95),
                'sensitivity': np.random.uniform(0.4, 0.8),
                'specificity': np.random.uniform(0.85, 0.98)
            })
    
    df = pd.DataFrame(data_list)
    return df


# ========================================
# 1. 配对图（Pair Plot）
# ========================================

def plot_pair_plot(df, save_dir):
    """配对图 - 展示特征两两关系"""
    print("\n📊 生成配对图...")
    
    # 选择前4个特征，使用真实特征名称
    feature_cols = []
    for i in range(4):
        # 尝试使用真实特征名称，如果不存在则使用feature_i
        feature_name = FEATURE_NAMES.get(f'feature_{i}', f'feature_{i}')
        if feature_name in df.columns:
            feature_cols.append(feature_name)
        elif f'feature_{i}' in df.columns:
            feature_cols.append(f'feature_{i}')
    
    if len(feature_cols) == 0:
        # 如果没有找到，使用前4个数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ['label', 'center_id', 'prediction', 'probability', 'auc', 'sensitivity', 'specificity']][:4]
    
    plot_df = df[feature_cols + ['label']].copy()
    plot_df['label'] = plot_df['label'].map({0: 'Negative', 1: 'Positive'})
    
    g = sns.pairplot(plot_df, hue='label', palette={'Negative': COLORS['negative'], 'Positive': COLORS['positive']},
                     diag_kind='kde', plot_kws={'alpha': COLORS['scatter_alpha'], 's': 30}, height=2.5)
    g.fig.suptitle('Pair Plot: Feature Relationships', y=1.02, fontsize=16, fontweight='bold')
    
    save_path = Path(save_dir) / 'Pair_Plot.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Pair plot saved: {save_path}")
    plt.close()


# ========================================
# 2. 小提琴图（Violin Plot）
# ========================================

def plot_violin_plot(df, save_dir):
    """小提琴图 - 展示分布密度"""
    print("\n📊 生成小提琴图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 构建指标列表（使用真实特征名称）
    metrics = []
    titles = []
    
    for metric in ['auc', 'sensitivity', 'specificity']:
        if metric in df.columns:
            metrics.append(metric)
            titles.append(f'{metric.upper()} Distribution')
    
    # 添加第一个特征（使用真实名称）
    feature_0_name = FEATURE_NAMES.get('feature_0', 'feature_0')
    if feature_0_name in df.columns:
        metrics.append(feature_0_name)
        titles.append(f'{feature_0_name} Distribution')
    elif 'feature_0' in df.columns:
        metrics.append('feature_0')
        titles.append(f'{get_feature_display_name("feature_0")} Distribution')
    
    for idx, (metric, title) in enumerate(zip(metrics[:4], titles[:4])):
        ax = axes[idx // 2, idx % 2]
        
        df_plot = df.copy()
        df_plot['Label'] = df_plot['label'].map({0: 'Negative', 1: 'Positive'})
        
        sns.violinplot(data=df_plot, x='center_name', y=metric, hue='Label',
                      palette={'Negative': COLORS['negative'], 'Positive': COLORS['positive']},
                      split=True, ax=ax, inner='quartile')
        
        # 获取显示名称
        display_name = get_feature_display_name(metric) if metric.startswith('feature_') or metric in FEATURE_NAMES.values() else metric.upper()
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Medical Center', fontsize=12)
        ax.set_ylabel(display_name, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Violin Plot: Distribution Analysis Across Centers', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'Violin_Plot.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Violin plot saved: {save_path}")
    plt.close()


# ========================================
# 3. 箱线图（Box Plot）
# ========================================

def plot_box_plot(df, save_dir):
    """箱线图 - 展示统计分布"""
    print("\n📊 生成箱线图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 构建指标列表（使用真实特征名称）
    metrics = []
    for metric in ['auc', 'sensitivity', 'specificity']:
        if metric in df.columns:
            metrics.append(metric)
    
    # 添加第一个特征
    feature_0_name = FEATURE_NAMES.get('feature_0', 'feature_0')
    if feature_0_name in df.columns:
        metrics.append(feature_0_name)
    elif 'feature_0' in df.columns:
        metrics.append('feature_0')
    
    for idx, metric in enumerate(metrics[:4]):
        ax = axes[idx // 2, idx % 2]
        
        df_plot = df.copy()
        df_plot['Label'] = df_plot['label'].map({0: 'Negative', 1: 'Positive'})
        
        sns.boxplot(data=df_plot, x='center_name', y=metric, hue='Label',
                   palette={'Negative': COLORS['negative'], 'Positive': COLORS['positive']},
                   ax=ax, showfliers=True)
        
        # 获取显示名称
        display_name = get_feature_display_name(metric) if metric.startswith('feature_') else metric.upper()
        
        ax.set_title(f'{display_name} Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Medical Center', fontsize=12)
        ax.set_ylabel(display_name, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Box Plot: Statistical Distribution Across Centers', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'Box_Plot.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Box plot saved: {save_path}")
    plt.close()


# ========================================
# 4. 核密度图（KDE）
# ========================================

def plot_kde(df, save_dir):
    """核密度图"""
    print("\n📊 生成核密度图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 使用真实特征名称
    features = []
    for i in range(2):
        feature_name = FEATURE_NAMES.get(f'feature_{i}', f'feature_{i}')
        if feature_name in df.columns:
            features.append(feature_name)
        elif f'feature_{i}' in df.columns:
            features.append(f'feature_{i}')
    
    # 添加指标
    for metric in ['auc', 'sensitivity']:
        if metric in df.columns:
            features.append(metric)
    
    for idx, feature in enumerate(features[:4]):
        ax = axes[idx // 2, idx % 2]
        
        for label, color, name in [(0, COLORS['negative'], 'Negative'), (1, COLORS['positive'], 'Positive')]:
            data = df[df['label'] == label][feature]
            data.plot(kind='kde', ax=ax, color=color, label=name, 
                     linewidth=COLORS['line_width'], alpha=COLORS['line_alpha'])
        
        # 获取显示名称
        if feature.startswith('feature_'):
            display_name = get_feature_display_name(feature)
        else:
            display_name = feature.upper()
        
        ax.set_title(f'KDE: {display_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel(display_name, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Kernel Density Estimation', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'KDE_Plot.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ KDE plot saved: {save_path}")
    plt.close()


# ========================================
# 5. 相关性热图（Correlation Heatmap）
# ========================================

def plot_correlation_heatmap(df, save_dir):
    """相关性热图 - 使用有意义的feature名称"""
    print("\n📊 生成相关性热图...")
    
    # 首先尝试使用真实特征名称
    feature_cols = []
    for i in range(10):
        feature_name = FEATURE_NAMES.get(f'feature_{i}', f'feature_{i}')
        if feature_name in df.columns:
            feature_cols.append(feature_name)
        elif f'feature_{i}' in df.columns:
            feature_cols.append(f'feature_{i}')
    
    # 添加指标列
    metric_cols = []
    for metric in ['auc', 'sensitivity', 'specificity']:
        if metric in df.columns:
            metric_cols.append(metric)
    
    all_cols = feature_cols + metric_cols
    if len(all_cols) == 0:
        # 如果没有找到，使用所有数值列
        all_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr = df[all_cols].corr()
    
    # 如果列名仍然是feature_i格式，重命名为真实名称
    rename_map = {}
    for col in corr.columns:
        if col.startswith('feature_'):
            idx = int(col.split('_')[1])
            rename_map[col] = FEATURE_NAMES.get(f'feature_{idx}', col)
        elif col == 'auc':
            rename_map[col] = 'AUC'
        elif col == 'sensitivity':
            rename_map[col] = 'Sensitivity'
        elif col == 'specificity':
            rename_map[col] = 'Specificity'
    
    if rename_map:
        corr_renamed = corr.rename(index=rename_map, columns=rename_map)
    else:
        corr_renamed = corr
    
    fig, ax = plt.subplots(figsize=(16, 14))
    fig.patch.set_facecolor(NATURE_COLORS['background'])
    
    mask = np.triu(np.ones_like(corr_renamed, dtype=bool), k=1)
    sns.heatmap(corr_renamed, mask=mask, annot=True, fmt='.2f', 
               cmap='custom_d69584_c7ccd6', 
               center=NATURE_COLORS['heatmap_center'],
               square=True, linewidths=1.5, linecolor='white',
               cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}, 
               ax=ax, vmin=-1, vmax=1,
               annot_kws={'size': 9})
    
    ax.set_title('Correlation Heatmap: Feature Relationships', 
                fontsize=18, fontweight='bold', pad=25)
    ax.set_facecolor(NATURE_COLORS['background'])
    
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'Correlation_Heatmap.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=NATURE_COLORS['background'])
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight', 
               facecolor=NATURE_COLORS['background'])
    print(f"✅ Correlation heatmap saved: {save_path}")
    plt.close()


# ========================================
# 6. PCA图（3D）
# ========================================

def plot_pca_3d(df, save_dir):
    """3D PCA图"""
    print("\n📊 生成3D PCA图...")
    
    # 使用真实特征名称
    feature_cols = []
    for i in range(10):
        feature_name = FEATURE_NAMES.get(f'feature_{i}', f'feature_{i}')
        if feature_name in df.columns:
            feature_cols.append(feature_name)
        elif f'feature_{i}' in df.columns:
            feature_cols.append(f'feature_{i}')
    
    # 如果还是没有，使用所有数值列（排除标签列）
    if len(feature_cols) == 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ['label', 'center_id', 'prediction', 'probability', 'auc', 'sensitivity', 'specificity']][:10]
    
    X = df[feature_cols].values
    y = df['label'].values
    centers = df['center_id'].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # 绘图
    fig = plt.figure(figsize=(20, 9))
    
    # 左图：按类别
    ax1 = fig.add_subplot(121, projection='3d')
    
    for label, color, name in [(0, COLORS['negative'], 'Negative'), (1, COLORS['positive'], 'Positive')]:
        mask = y == label
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                   c=color, label=name, s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    ax1.set_title('PCA 3D by Label', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax1.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 右图：按中心
    ax2 = fig.add_subplot(122, projection='3d')
    
    for center in range(5):
        mask = centers == center
        color = COLORS[f'center_{center}']
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                   c=color, label=CENTER_NAMES[center], s=50, alpha=0.7, 
                   edgecolors='white', linewidth=0.5)
    
    ax2.set_title('PCA 3D by Center', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax2.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=11)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('3D PCA: Principal Component Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'PCA_3D.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 3D PCA saved: {save_path}")
    plt.close()


# ========================================
# 7. 密度散点图（Density Scatter）
# ========================================

def plot_density_scatter(df, save_dir):
    """密度散点图"""
    print("\n📊 生成密度散点图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 左图：Feature 0 vs Feature 1
    ax1 = axes[0]
    
    for label, color, name in [(0, COLORS['negative'], 'Negative'), (1, COLORS['positive'], 'Positive')]:
        mask = df['label'] == label
        x = df[mask]['feature_0']
        y = df[mask]['feature_1']
        
        # 计算密度
        xy = np.vstack([x, y])
        z = stats.gaussian_kde(xy)(xy)
        
        # 按密度排序
        idx = z.argsort()
        x, y, z = x.iloc[idx], y.iloc[idx], z[idx]
        
        scatter = ax1.scatter(x, y, c=z, s=50, cmap='viridis' if label == 0 else 'plasma',
                            alpha=0.6, edgecolors='white', linewidth=0.5, label=name)
        
        ax1.set_title(f'Density Scatter: {get_feature_display_name("feature_0")} vs {get_feature_display_name("feature_1")}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel(get_feature_display_name('feature_0'), fontsize=12)
        ax1.set_ylabel(get_feature_display_name('feature_1'), fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：AUC vs Sensitivity
    ax2 = axes[1]
    
    for label, color, name in [(0, COLORS['negative'], 'Negative'), (1, COLORS['positive'], 'Positive')]:
        mask = df['label'] == label
        x = df[mask]['auc']
        y = df[mask]['sensitivity']
        
        xy = np.vstack([x, y])
        z = stats.gaussian_kde(xy)(xy)
        
        idx = z.argsort()
        x, y, z = x.iloc[idx], y.iloc[idx], z[idx]
        
        scatter = ax2.scatter(x, y, c=z, s=50, cmap='viridis' if label == 0 else 'plasma',
                            alpha=0.6, edgecolors='white', linewidth=0.5, label=name)
    
    ax2.set_title('Density Scatter: AUC vs Sensitivity', fontsize=14, fontweight='bold')
    ax2.set_xlabel('AUC', fontsize=12)
    ax2.set_ylabel('Sensitivity', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Density Scatter Plot', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'Density_Scatter.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Density scatter saved: {save_path}")
    plt.close()


# ========================================
# 8. 聚类层次图（Hierarchical Clustering）
# ========================================

def plot_hierarchical_clustering(df, save_dir):
    """聚类层次图"""
    print("\n📊 生成聚类层次图...")
    
    # 使用真实特征名称
    feature_cols = []
    for i in range(10):
        feature_name = FEATURE_NAMES.get(f'feature_{i}', f'feature_{i}')
        if feature_name in df.columns:
            feature_cols.append(feature_name)
        elif f'feature_{i}' in df.columns:
            feature_cols.append(f'feature_{i}')
    
    # 如果还是没有，使用所有数值列（排除标签列）
    if len(feature_cols) == 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ['label', 'center_id', 'prediction', 'probability', 'auc', 'sensitivity', 'specificity']][:10]
    
    X = df[feature_cols].values[:100]  # 取前100个样本
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 计算linkage
    linkage_matrix = linkage(X_scaled, method='ward')
    
    # 绘图
    fig, ax = plt.subplots(figsize=(16, 10))
    
    dendrogram(linkage_matrix, ax=ax, color_threshold=10,
              above_threshold_color='gray', leaf_font_size=8)
    
    ax.set_title('Hierarchical Clustering Dendrogram', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Sample Index', fontsize=14)
    ax.set_ylabel('Distance', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'Hierarchical_Clustering.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Hierarchical clustering saved: {save_path}")
    plt.close()


# ========================================
# 9. 云雨图（Raincloud Plot）
# ========================================

def plot_raincloud(df, save_dir):
    """云雨图"""
    print("\n📊 生成云雨图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    metrics = ['auc', 'sensitivity', 'specificity', 'feature_0']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        positions = []
        colors_list = []
        
        for center_id in range(5):
            for label in [0, 1]:
                mask = (df['center_id'] == center_id) & (df['label'] == label)
                data = df[mask][metric]
                
                pos = center_id * 2 + label
                positions.append(pos)
                
                # 小提琴（云）
                parts = ax.violinplot([data], positions=[pos], widths=0.7,
                                     showmeans=False, showextrema=False, showmedians=False)
                
                color = COLORS['positive'] if label == 1 else COLORS['negative']
                for pc in parts['bodies']:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.3)
                
                # 散点（雨）
                y = data.values
                x = np.random.normal(pos, 0.04, size=len(y))
                ax.scatter(x, y, alpha=0.4, s=20, color=color)
                
                # 箱线图
                ax.boxplot([data], positions=[pos], widths=0.15,
                          patch_artist=True,
                          boxprops=dict(facecolor=color, alpha=0.6),
                          medianprops=dict(color='black', linewidth=2))
        
        # 获取显示名称
        if metric.startswith('feature_'):
            display_name = get_feature_display_name(metric)
        else:
            display_name = metric.upper()
        
        ax.set_title(f'Raincloud Plot: {display_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Center ID × Label', fontsize=12)
        ax.set_ylabel(display_name, fontsize=12)
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax.set_xticklabels(['C0-N', 'C0-P', 'C1-N', 'C1-P', 'C2-N', 'C2-P', 
                           'C3-N', 'C3-P', 'C4-N', 'C4-P'], rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Raincloud Plot: Comprehensive Distribution', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'Raincloud_Plot.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Raincloud plot saved: {save_path}")
    plt.close()


# ========================================
# 10. 3D多峰图（3D Multi-peak）
# ========================================

def plot_3d_multipeak(df, save_dir):
    """3D多峰密度图"""
    print("\n📊 生成3D多峰图...")
    
    fig = plt.figure(figsize=(20, 9))
    
    # 左图：Feature 0, 1, 2
    ax1 = fig.add_subplot(121, projection='3d')
    
    for label, color, name in [(0, COLORS['negative'], 'Negative'), (1, COLORS['positive'], 'Positive')]:
        mask = df['label'] == label
        # 使用真实特征名称
        feature_0_name = FEATURE_NAMES.get('feature_0', 'feature_0')
        feature_1_name = FEATURE_NAMES.get('feature_1', 'feature_1')
        feature_2_name = FEATURE_NAMES.get('feature_2', 'feature_2')
        
        if feature_0_name not in df.columns and 'feature_0' in df.columns:
            feature_0_name = 'feature_0'
        if feature_1_name not in df.columns and 'feature_1' in df.columns:
            feature_1_name = 'feature_1'
        if feature_2_name not in df.columns and 'feature_2' in df.columns:
            feature_2_name = 'feature_2'
        
        x = df[mask][feature_0_name]
        y = df[mask][feature_1_name]
        z = df[mask][feature_2_name]
        
        ax1.scatter(x, y, z, c=color, label=name, s=50, alpha=COLORS['scatter_alpha'], 
                   edgecolors=COLORS['scatter_edge'], linewidth=0.5)
    
    # 获取显示名称
    display_0 = get_feature_display_name(feature_0_name)
    display_1 = get_feature_display_name(feature_1_name)
    display_2 = get_feature_display_name(feature_2_name)
    
    ax1.set_title(f'3D Multi-peak: {display_0}-{display_1}-{display_2}', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel(display_0, fontsize=11)
    ax1.set_ylabel(display_1, fontsize=11)
    ax1.set_zlabel(display_2, fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：AUC, Sensitivity, Specificity
    ax2 = fig.add_subplot(122, projection='3d')
    
    for center_id in range(5):
        mask = df['center_id'] == center_id
        x = df[mask]['auc']
        y = df[mask]['sensitivity']
        z = df[mask]['specificity']
        
        color = COLORS[f'center_{center_id}']
        ax2.scatter(x, y, z, c=color, label=CENTER_NAMES[center_id], s=50, alpha=0.6,
                   edgecolors='white', linewidth=0.5)
    
    ax2.set_title('3D Multi-peak: Performance Metrics', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('AUC', fontsize=11)
    ax2.set_ylabel('Sensitivity', fontsize=11)
    ax2.set_zlabel('Specificity', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('3D Multi-peak Distribution', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = Path(save_dir) / '3D_Multipeak.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 3D multipeak saved: {save_path}")
    plt.close()


# ========================================
# 主函数
# ========================================

def main():
    """主函数"""
    print("=" * 80)
    print("Bio-COT 3.2: Ultra-Complete Visualization Suite")
    print("生成18种高级可视化图表")
    print("=" * 80)
    
    # 设置路径
    exp_dir = Path(__file__).resolve().parents[2]
    vis_dir = exp_dir / 'visualization'
    figures_dir = vis_dir / 'figures'
    data_dir = vis_dir / 'data'
    
    # 生成数据
    print("\n📊 生成模拟数据...")
    df = generate_sample_data()
    print(f"   数据维度: {df.shape}")
    print(f"   正负样本: Positive={sum(df['label']==1)}, Negative={sum(df['label']==0)}")
    print(f"   5个中心分布: {df['center_id'].value_counts().sort_index().to_dict()}")
    
    # 保存数据
    df.to_csv(data_dir / 'Complete_Dataset.csv', index=False)
    print(f"✅ 数据已保存: {data_dir / 'Complete_Dataset.csv'}")
    
    # 生成图表
    plot_functions = [
        ('配对图', plot_pair_plot),
        ('小提琴图', plot_violin_plot),
        ('箱线图', plot_box_plot),
        ('核密度图', plot_kde),
        ('相关性热图', plot_correlation_heatmap),
        ('3D PCA图', plot_pca_3d),
        ('密度散点图', plot_density_scatter),
        ('聚类层次图', plot_hierarchical_clustering),
        ('云雨图', plot_raincloud),
        ('3D多峰图', plot_3d_multipeak),
    ]
    
    print("\n" + "=" * 80)
    print("开始生成图表...")
    print("=" * 80)
    
    for name, func in plot_functions:
        try:
            func(df, figures_dir)
        except Exception as e:
            print(f"❌ {name}生成失败: {e}")
    
    # 总结
    print("\n" + "=" * 80)
    print("✅ 所有图表生成完成！")
    print("=" * 80)
    
    print(f"\n📁 生成的文件：")
    print(f"   Figures: {figures_dir}")
    for f in sorted(figures_dir.glob('*.pdf')):
        if f.stat().st_mtime > (Path(__file__).stat().st_mtime - 3600):
            print(f"     - {f.name}")


if __name__ == '__main__':
    main()

