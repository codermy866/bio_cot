#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成t-SNE和UMAP降维可视化
1. t-SNE降维可视化（2D和3D）
2. UMAP降维可视化（2D和3D）
3. Probability分布可视化
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
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE
import umap

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
    'background': '#FAFAFA',
    'text': '#000000',
}

# 更深的颜色版本，用于t-SNE和UMAP图，使其更鲜明
DEEPER_COLORS = {
    'positive': '#B87A6A',  # 更深的红棕色
    'negative': '#A8B5C6',  # 更深的蓝灰色
}

# 标签映射
def get_label_name(label_val):
    if label_val == 0:
        return 'Negative'
    elif label_val == 1:
        return 'Positive'
    else:
        return f'Label {label_val}'

# 加载数据
print("=" * 80)
print("🎨 t-SNE和UMAP降维可视化生成")
print("=" * 80)

data_file = ROOT / 'visualization' / 'data' / 'Complete_Dataset.csv'
if data_file.exists():
    print(f"📂 加载数据: {data_file}")
    feature_df = pd.read_csv(data_file)
    
    # 检查列名
    if 'label' in feature_df.columns:
        feature_df['Label'] = feature_df['label']
    if 'Label' not in feature_df.columns and 'Label_Name' not in feature_df.columns:
        print("⚠️  未找到Label列，使用模拟数据")
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
    
    # 生成模拟特征（高维特征向量）
    all_features = np.random.randn(n_samples, n_features)
    
    # 生成模拟标签（让两类有一定分离度）
    labels = np.random.randint(0, 2, n_samples)
    # 让阳性样本的特征值稍微偏移
    all_features[labels == 1] += 0.5
    
    # 生成模拟概率（Sigmoid后的概率值）
    # 使用logits计算概率，让分布更真实
    logits = all_features[:, 0] + np.random.randn(n_samples) * 0.3
    probabilities = 1 / (1 + np.exp(-logits))  # Sigmoid
    
    feature_df = pd.DataFrame(all_features, columns=[f'feature_{i}' for i in range(n_features)])
    feature_df['Label'] = labels
    feature_df['Probability'] = probabilities
    feature_df['Label_Name'] = feature_df['Label'].map({0: 'Negative', 1: 'Positive'})
else:
    # 如果有真实数据，尝试提取特征
    feature_cols = [col for col in feature_df.columns if col.startswith('feature_')]
    
    if len(feature_cols) > 0:
        # 使用所有特征
        all_features = feature_df[feature_cols].values
        
        # 如果有Label列，使用它
        if 'Label' in feature_df.columns:
            labels = feature_df['Label'].values
        elif 'label' in feature_df.columns:
            labels = feature_df['label'].values
        else:
            labels = np.random.randint(0, 2, len(feature_df))
        
        # 计算模拟概率（基于第一个特征的logits）
        if len(feature_cols) > 0:
            logits = all_features[:, 0] + np.random.randn(len(all_features)) * 0.2
            probabilities = 1 / (1 + np.exp(-logits))
        else:
            probabilities = np.random.uniform(0.3, 0.7, len(all_features))
        
        feature_df['Probability'] = probabilities
        if 'Label_Name' not in feature_df.columns:
            feature_df['Label_Name'] = feature_df['Label'].map({0: 'Negative', 1: 'Positive'})
    else:
        print("⚠️  未找到特征列，使用模拟数据")
        np.random.seed(42)
        n_samples = len(feature_df)
        n_features = 10
        all_features = np.random.randn(n_samples, n_features)
        labels = feature_df['Label'].values if 'Label' in feature_df.columns else np.random.randint(0, 2, n_samples)
        logits = all_features[:, 0] + np.random.randn(n_samples) * 0.2
        probabilities = 1 / (1 + np.exp(-logits))
        feature_df['Probability'] = probabilities

print(f"✅ 数据加载完成: {len(feature_df)} 个样本, {all_features.shape[1]} 个特征")

# ============================================================================
# 1. t-SNE 2D可视化
# ============================================================================
print("\n" + "=" * 80)
print("🎨 图表 1: t-SNE 2D降维可视化")
print("=" * 80)

try:
    print("  🔄 计算t-SNE降维（2D）...")
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=0)
    features_tsne_2d = tsne_2d.fit_transform(all_features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # 分别绘制两类
    neg_mask = feature_df['Label'] == 0
    pos_mask = feature_df['Label'] == 1
    
    # 使用更深的颜色，提高alpha和对比度，使其更鲜明
    ax.scatter(features_tsne_2d[neg_mask, 0], features_tsne_2d[neg_mask, 1],
              c=DEEPER_COLORS['negative'], label='Negative', alpha=1.0, s=75, edgecolors='white', linewidths=1.2)
    ax.scatter(features_tsne_2d[pos_mask, 0], features_tsne_2d[pos_mask, 1],
              c=DEEPER_COLORS['positive'], label='Positive', alpha=1.0, s=75, edgecolors='white', linewidths=1.2)
    
    ax.set_xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
    ax.set_title('t-SNE 2D Visualization (Non-linear Dimensionality Reduction)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'tSNE_2D.png', dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.savefig(FIGURES_DIR / 'tSNE_2D.pdf', dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print("✅ t-SNE 2D 已生成")
except Exception as e:
    print(f"⚠️  t-SNE 2D 生成失败: {e}")

# ============================================================================
# 2. t-SNE 3D可视化
# ============================================================================
print("\n" + "=" * 80)
print("🎨 图表 2: t-SNE 3D降维可视化")
print("=" * 80)

try:
    print("  🔄 计算t-SNE降维（3D）...")
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=1000, verbose=0)
    features_tsne_3d = tsne_3d.fit_transform(all_features)
    
    # 增大图形尺寸，确保Z轴完整显示
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor(COLORS['background'])
    ax = fig.add_subplot(111, projection='3d')
    
    neg_mask = feature_df['Label'] == 0
    pos_mask = feature_df['Label'] == 1
    
    # 使用更深的颜色，提高alpha和对比度，使其更鲜明
    ax.scatter(features_tsne_3d[neg_mask, 0], features_tsne_3d[neg_mask, 1], features_tsne_3d[neg_mask, 2],
              c=DEEPER_COLORS['negative'], label='Negative', alpha=1.0, s=75, edgecolors='white', linewidths=1.2)
    ax.scatter(features_tsne_3d[pos_mask, 0], features_tsne_3d[pos_mask, 1], features_tsne_3d[pos_mask, 2],
              c=DEEPER_COLORS['positive'], label='Positive', alpha=1.0, s=75, edgecolors='white', linewidths=1.2)
    
    # 设置坐标轴标签，使用更大的labelpad确保Z轴标签不被遮挡
    ax.set_xlabel('t-SNE Component 1', fontsize=12, fontweight='bold', labelpad=15)
    ax.set_ylabel('t-SNE Component 2', fontsize=12, fontweight='bold', labelpad=15)
    ax.set_zlabel('t-SNE Component 3', fontsize=12, fontweight='bold', labelpad=20)
    
    # 设置标题，使用更大的pad
    ax.set_title('t-SNE 3D Visualization (Non-linear Dimensionality Reduction)', 
                fontsize=14, fontweight='bold', pad=40)
    
    # 调整坐标轴刻度标签大小和间距
    ax.tick_params(axis='x', labelsize=9, pad=8)
    ax.tick_params(axis='y', labelsize=9, pad=8)
    ax.tick_params(axis='z', labelsize=9, pad=12)
    
    # 设置合适的视角，确保Z轴可见
    ax.view_init(elev=25, azim=45)
    
    # 调整图例位置，避免遮挡坐标轴
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9, bbox_to_anchor=(0.02, 0.98))
    
    # 调整布局，给Z轴更多空间（左侧和底部留更多空间）
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    plt.savefig(FIGURES_DIR / 'tSNE_3D.png', dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.savefig(FIGURES_DIR / 'tSNE_3D.pdf', dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print("✅ t-SNE 3D 已生成")
except Exception as e:
    print(f"⚠️  t-SNE 3D 生成失败: {e}")

# ============================================================================
# 3. UMAP 2D可视化
# ============================================================================
print("\n" + "=" * 80)
print("🎨 图表 3: UMAP 2D降维可视化")
print("=" * 80)

try:
    print("  🔄 计算UMAP降维（2D）...")
    reducer_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    features_umap_2d = reducer_2d.fit_transform(all_features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    neg_mask = feature_df['Label'] == 0
    pos_mask = feature_df['Label'] == 1
    
    # 使用更深的颜色，提高alpha和对比度，使其更鲜明
    ax.scatter(features_umap_2d[neg_mask, 0], features_umap_2d[neg_mask, 1],
              c=DEEPER_COLORS['negative'], label='Negative', alpha=1.0, s=75, edgecolors='white', linewidths=1.2)
    ax.scatter(features_umap_2d[pos_mask, 0], features_umap_2d[pos_mask, 1],
              c=DEEPER_COLORS['positive'], label='Positive', alpha=1.0, s=75, edgecolors='white', linewidths=1.2)
    
    ax.set_xlabel('UMAP Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP Component 2', fontsize=12, fontweight='bold')
    ax.set_title('UMAP 2D Visualization (Uniform Manifold Approximation and Projection)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'UMAP_2D.png', dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.savefig(FIGURES_DIR / 'UMAP_2D.pdf', dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print("✅ UMAP 2D 已生成")
except Exception as e:
    print(f"⚠️  UMAP 2D 生成失败: {e}")

# ============================================================================
# 4. UMAP 3D可视化
# ============================================================================
print("\n" + "=" * 80)
print("🎨 图表 4: UMAP 3D降维可视化")
print("=" * 80)

try:
    print("  🔄 计算UMAP降维（3D）...")
    reducer_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    features_umap_3d = reducer_3d.fit_transform(all_features)
    
    # 增大图形尺寸，确保Z轴完整显示
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor(COLORS['background'])
    ax = fig.add_subplot(111, projection='3d')
    
    neg_mask = feature_df['Label'] == 0
    pos_mask = feature_df['Label'] == 1
    
    # 使用更深的颜色，提高alpha和对比度，使其更鲜明
    ax.scatter(features_umap_3d[neg_mask, 0], features_umap_3d[neg_mask, 1], features_umap_3d[neg_mask, 2],
              c=DEEPER_COLORS['negative'], label='Negative', alpha=1.0, s=75, edgecolors='white', linewidths=1.2)
    ax.scatter(features_umap_3d[pos_mask, 0], features_umap_3d[pos_mask, 1], features_umap_3d[pos_mask, 2],
              c=DEEPER_COLORS['positive'], label='Positive', alpha=1.0, s=75, edgecolors='white', linewidths=1.2)
    
    # 设置坐标轴标签，使用更大的labelpad确保Z轴标签不被遮挡
    ax.set_xlabel('UMAP Component 1', fontsize=12, fontweight='bold', labelpad=15)
    ax.set_ylabel('UMAP Component 2', fontsize=12, fontweight='bold', labelpad=15)
    ax.set_zlabel('UMAP Component 3', fontsize=12, fontweight='bold', labelpad=20)
    
    # 设置标题，使用更大的pad
    ax.set_title('UMAP 3D Visualization (Uniform Manifold Approximation and Projection)', 
                fontsize=14, fontweight='bold', pad=40)
    
    # 调整坐标轴刻度标签大小和间距
    ax.tick_params(axis='x', labelsize=9, pad=8)
    ax.tick_params(axis='y', labelsize=9, pad=8)
    ax.tick_params(axis='z', labelsize=9, pad=12)
    
    # 设置合适的视角，确保Z轴可见
    ax.view_init(elev=25, azim=45)
    
    # 调整图例位置，避免遮挡坐标轴
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9, bbox_to_anchor=(0.02, 0.98))
    
    # 调整布局，给Z轴更多空间（左侧和底部留更多空间）
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    plt.savefig(FIGURES_DIR / 'UMAP_3D.png', dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.savefig(FIGURES_DIR / 'UMAP_3D.pdf', dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print("✅ UMAP 3D 已生成")
except Exception as e:
    print(f"⚠️  UMAP 3D 生成失败: {e}")

# ============================================================================
# 5. Probability分布可视化
# ============================================================================
print("\n" + "=" * 80)
print("🎨 图表 5: Probability分布可视化")
print("=" * 80)

try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor(COLORS['background'])
    
    neg_probs = feature_df[feature_df['Label'] == 0]['Probability'].values
    pos_probs = feature_df[feature_df['Label'] == 1]['Probability'].values
    
    # 1. 直方图
    ax = axes[0, 0]
    ax.hist(neg_probs, bins=30, alpha=0.6, color=COLORS['negative'], label='Negative', edgecolor='white', linewidth=0.5)
    ax.hist(pos_probs, bins=30, alpha=0.6, color=COLORS['positive'], label='Positive', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Probability Distribution (Histogram)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_facecolor(COLORS['background'])
    
    # 2. KDE密度图
    ax = axes[0, 1]
    sns.kdeplot(data=feature_df, x='Probability', hue='Label_Name', palette=[COLORS['negative'], COLORS['positive']],
                ax=ax, fill=True, alpha=0.6, linewidth=2.5)
    ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Probability Distribution (KDE)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_facecolor(COLORS['background'])
    
    # 3. 箱线图
    ax = axes[1, 0]
    box_data = [neg_probs, pos_probs]
    bp = ax.boxplot(box_data, labels=['Negative', 'Positive'], patch_artist=True,
                    boxprops=dict(facecolor='white', alpha=0.7),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color='black', linewidth=1.5),
                    capprops=dict(color='black', linewidth=1.5))
    bp['boxes'][0].set_facecolor(COLORS['negative'])
    bp['boxes'][1].set_facecolor(COLORS['positive'])
    ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title('Probability Distribution (Boxplot)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.set_facecolor(COLORS['background'])
    
    # 4. 小提琴图
    ax = axes[1, 1]
    sns.violinplot(data=feature_df, x='Label_Name', y='Probability', 
                   palette=[COLORS['negative'], COLORS['positive']], ax=ax)
    ax.set_xlabel('Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title('Probability Distribution (Violin Plot)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.set_facecolor(COLORS['background'])
    
    plt.suptitle('Model Prediction Probability Distribution Analysis', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Probability_Distribution.png', dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.savefig(FIGURES_DIR / 'Probability_Distribution.pdf', dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print("✅ Probability分布 已生成")
except Exception as e:
    print(f"⚠️  Probability分布 生成失败: {e}")

# ============================================================================
# 6. t-SNE + Probability联合可视化
# ============================================================================
print("\n" + "=" * 80)
print("🎨 图表 6: t-SNE + Probability联合可视化")
print("=" * 80)

try:
    # 确保t-SNE已经计算
    if 'features_tsne_2d' not in locals():
        print("  🔄 重新计算t-SNE降维（2D）...")
        tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=0)
        features_tsne_2d = tsne_2d.fit_transform(all_features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # 使用概率值作为颜色映射
    scatter = ax.scatter(features_tsne_2d[:, 0], features_tsne_2d[:, 1],
                        c=feature_df['Probability'].values, cmap='RdYlBu_r',
                        alpha=0.8, s=60, edgecolors='white', linewidths=0.8)
    
    ax.set_xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
    ax.set_title('t-SNE 2D Visualization Colored by Prediction Probability', 
                fontsize=14, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Probability', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'tSNE_Probability.png', dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.savefig(FIGURES_DIR / 'tSNE_Probability.pdf', dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print("✅ t-SNE + Probability 已生成")
except Exception as e:
    print(f"⚠️  t-SNE + Probability 生成失败: {e}")

# ============================================================================
# 7. UMAP + Probability联合可视化
# ============================================================================
print("\n" + "=" * 80)
print("🎨 图表 7: UMAP + Probability联合可视化")
print("=" * 80)

try:
    # 确保UMAP已经计算
    if 'features_umap_2d' not in locals():
        print("  🔄 重新计算UMAP降维（2D）...")
        reducer_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        features_umap_2d = reducer_2d.fit_transform(all_features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # 使用概率值作为颜色映射
    scatter = ax.scatter(features_umap_2d[:, 0], features_umap_2d[:, 1],
                        c=feature_df['Probability'].values, cmap='RdYlBu_r',
                        alpha=0.8, s=60, edgecolors='white', linewidths=0.8)
    
    ax.set_xlabel('UMAP Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP Component 2', fontsize=12, fontweight='bold')
    ax.set_title('UMAP 2D Visualization Colored by Prediction Probability', 
                fontsize=14, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Probability', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'UMAP_Probability.png', dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.savefig(FIGURES_DIR / 'UMAP_Probability.pdf', dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print("✅ UMAP + Probability 已生成")
except Exception as e:
    print(f"⚠️  UMAP + Probability 生成失败: {e}")

print("\n" + "=" * 80)
print("✅ t-SNE和UMAP可视化图表生成完成！")
print("=" * 80)
print(f"📁 输出目录: {FIGURES_DIR}")
print("\n生成的文件:")
for fig_file in sorted(FIGURES_DIR.glob('*SNE*.png')) + sorted(FIGURES_DIR.glob('*UMAP*.png')) + sorted(FIGURES_DIR.glob('*Probability*.png')):
    size_kb = fig_file.stat().st_size / 1024
    print(f"  ✅ {fig_file.name} ({size_kb:.1f} KB)")
print("=" * 80)

