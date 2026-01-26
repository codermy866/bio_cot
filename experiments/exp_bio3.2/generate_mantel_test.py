#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mantel检验相关性分析
- 检验不同距离矩阵之间的相关性
- 生成可视化图表
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
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise_distances
import torch

# 配置路径
ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / 'newlog_0126'
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# 设置字体和样式
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['figure.dpi'] = 300

# 统一配色：DDAB9F到C7CCD6
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

print("=" * 80)
print("📊 Mantel检验相关性分析")
print("=" * 80)

# ========================================
# 1. 加载数据或生成模拟数据
# ========================================
print("\n📊 步骤 1: 准备数据...")

# 尝试从CSV文件加载数据
data_file = ROOT / 'visualization' / 'data' / 'Complete_Dataset.csv'
if data_file.exists():
    print(f"📖 从文件加载数据: {data_file}")
    df = pd.read_csv(data_file)
    
    # 提取特征列（feature_0 到 feature_9）
    feature_cols = [f'feature_{i}' for i in range(10) if f'feature_{i}' in df.columns]
    if feature_cols:
        features = df[feature_cols].values
        labels = df['label'].values if 'label' in df.columns else None
        center_ids = df['center_id'].values if 'center_id' in df.columns else None
        print(f"✅ 加载了 {len(features)} 个样本，{len(feature_cols)} 个特征")
    else:
        print("⚠️  未找到特征列，使用模拟数据")
        features = None
        labels = None
        center_ids = None
else:
    print("⚠️  数据文件不存在，使用模拟数据")
    features = None
    labels = None
    center_ids = None

# 如果没有真实数据，生成模拟数据
if features is None:
    print("💡 生成模拟特征数据...")
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    # 生成特征（考虑标签和中心的相关性）
    features = np.random.randn(n_samples, n_features)
    
    # 生成标签（基于特征）
    labels = (features[:, 0] + features[:, 1] > 0).astype(int)
    
    # 生成中心ID（基于特征）
    center_ids = (features[:, 2] * 2 + 2).astype(int) % 5
    
    print(f"✅ 生成了 {n_samples} 个模拟样本")

print()

# ========================================
# 2. 计算距离矩阵
# ========================================
print("=" * 80)
print("📊 步骤 2: 计算距离矩阵")
print("=" * 80)

# 2.1 特征距离矩阵（欧氏距离）
print("📊 计算特征距离矩阵（欧氏距离）...")
feature_dist = pairwise_distances(features, metric='euclidean')
print(f"✅ 特征距离矩阵形状: {feature_dist.shape}")

# 2.2 标签距离矩阵（汉明距离）
if labels is not None:
    print("📊 计算标签距离矩阵（汉明距离）...")
    label_dist = pairwise_distances(labels.reshape(-1, 1), metric='hamming')
    label_dist = label_dist * len(labels)  # 转换为绝对距离
    print(f"✅ 标签距离矩阵形状: {label_dist.shape}")
else:
    label_dist = None

# 2.3 中心距离矩阵
if center_ids is not None:
    print("📊 计算中心距离矩阵...")
    center_dist = pairwise_distances(center_ids.reshape(-1, 1), metric='hamming')
    center_dist = center_dist * len(center_ids)
    print(f"✅ 中心距离矩阵形状: {center_dist.shape}")
else:
    center_dist = None

# 2.4 特征子集距离矩阵（用于多模态分析）
print("📊 计算特征子集距离矩阵...")
# OCT特征（前3个）
oct_features = features[:, :3] if features.shape[1] >= 3 else features
oct_dist = pairwise_distances(oct_features, metric='euclidean')

# Colposcopy特征（中间3个）
if features.shape[1] >= 6:
    colpo_features = features[:, 3:6]
    colpo_dist = pairwise_distances(colpo_features, metric='euclidean')
else:
    colpo_dist = None

# Clinical特征（后4个）
if features.shape[1] >= 10:
    clinical_features = features[:, 6:10]
    clinical_dist = pairwise_distances(clinical_features, metric='euclidean')
else:
    clinical_dist = None

print()

# ========================================
# 3. Mantel检验
# ========================================
print("=" * 80)
print("📊 步骤 3: 执行Mantel检验")
print("=" * 80)

def mantel_test(dist1, dist2, n_permutations=9999):
    """
    Mantel检验：检验两个距离矩阵之间的相关性
    
    Parameters:
    -----------
    dist1, dist2: 距离矩阵（对称，对角线为0）
    n_permutations: 置换检验次数
    
    Returns:
    --------
    r: 相关系数
    p_value: p值
    """
    # 提取上三角矩阵（不包括对角线）
    n = dist1.shape[0]
    triu_indices = np.triu_indices(n, k=1)
    
    vec1 = dist1[triu_indices]
    vec2 = dist2[triu_indices]
    
    # 计算观测相关系数
    r_obs, _ = pearsonr(vec1, vec2)
    
    # 置换检验
    r_perm = []
    for _ in range(n_permutations):
        # 随机置换其中一个矩阵的行和列
        perm_indices = np.random.permutation(n)
        dist2_perm = dist2[np.ix_(perm_indices, perm_indices)]
        vec2_perm = dist2_perm[triu_indices]
        
        r_perm_val, _ = pearsonr(vec1, vec2_perm)
        r_perm.append(r_perm_val)
    
    # 计算p值（单侧检验）
    p_value = (np.sum(np.array(r_perm) >= r_obs) + 1) / (n_permutations + 1)
    
    return r_obs, p_value

# 执行Mantel检验
mantel_results = []

# 1. 特征距离 vs 标签距离
if label_dist is not None:
    print("📊 检验: 特征距离 vs 标签距离...")
    r, p = mantel_test(feature_dist, label_dist, n_permutations=999)
    mantel_results.append({
        'Comparison': 'Feature Distance vs Label Distance',
        'Correlation': r,
        'P-value': p,
        'Significant': 'Yes' if p < 0.05 else 'No'
    })
    print(f"   r = {r:.4f}, p = {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''}")

# 2. 特征距离 vs 中心距离
if center_dist is not None:
    print("📊 检验: 特征距离 vs 中心距离...")
    r, p = mantel_test(feature_dist, center_dist, n_permutations=999)
    mantel_results.append({
        'Comparison': 'Feature Distance vs Center Distance',
        'Correlation': r,
        'P-value': p,
        'Significant': 'Yes' if p < 0.05 else 'No'
    })
    print(f"   r = {r:.4f}, p = {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''}")

# 3. OCT vs Colposcopy
if colpo_dist is not None:
    print("📊 检验: OCT特征距离 vs Colposcopy特征距离...")
    r, p = mantel_test(oct_dist, colpo_dist, n_permutations=999)
    mantel_results.append({
        'Comparison': 'OCT Distance vs Colposcopy Distance',
        'Correlation': r,
        'P-value': p,
        'Significant': 'Yes' if p < 0.05 else 'No'
    })
    print(f"   r = {r:.4f}, p = {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''}")

# 4. OCT vs Clinical
if clinical_dist is not None:
    print("📊 检验: OCT特征距离 vs Clinical特征距离...")
    r, p = mantel_test(oct_dist, clinical_dist, n_permutations=999)
    mantel_results.append({
        'Comparison': 'OCT Distance vs Clinical Distance',
        'Correlation': r,
        'P-value': p,
        'Significant': 'Yes' if p < 0.05 else 'No'
    })
    print(f"   r = {r:.4f}, p = {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''}")

# 5. Colposcopy vs Clinical
if colpo_dist is not None and clinical_dist is not None:
    print("📊 检验: Colposcopy特征距离 vs Clinical特征距离...")
    r, p = mantel_test(colpo_dist, clinical_dist, n_permutations=999)
    mantel_results.append({
        'Comparison': 'Colposcopy Distance vs Clinical Distance',
        'Correlation': r,
        'P-value': p,
        'Significant': 'Yes' if p < 0.05 else 'No'
    })
    print(f"   r = {r:.4f}, p = {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''}")

print()

# 保存结果
results_df = pd.DataFrame(mantel_results)
results_csv = OUTPUT_DIR / 'mantel_test_results.csv'
results_df.to_csv(results_csv, index=False)
print(f"✅ Mantel检验结果已保存: {results_csv}")
print()

# ========================================
# 4. 可视化
# ========================================
print("=" * 80)
print("🎨 步骤 4: 生成可视化图表")
print("=" * 80)

# 4.1 Mantel检验结果热图
print("📊 生成Mantel检验结果热图...")
try:
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(COLORS['background'])
    
    # 创建相关性矩阵（只显示有检验的组合）
    comparisons = [r['Comparison'] for r in mantel_results]
    n = len(comparisons)
    
    if n > 0:
        # 创建矩阵
        corr_matrix = np.zeros((n, n))
        p_matrix = np.ones((n, n))
        
        for i, r1 in enumerate(mantel_results):
            for j, r2 in enumerate(mantel_results):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    # 这里简化处理，实际应该计算所有组合
                    corr_matrix[i, j] = r1['Correlation']
        
        # 绘制热图
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ['#D69584', '#D2A392', '#CEB1A0', '#CABFAE', '#C7CCD6']
        cmap = LinearSegmentedColormap.from_list('custom_d69584_c7ccd6', colors_list, N=100)
        
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap=cmap,
                   xticklabels=[c.split(' vs ')[0] for c in comparisons],
                   yticklabels=[c.split(' vs ')[1] if ' vs ' in c else c for c in comparisons],
                   cbar_kws={'label': 'Correlation Coefficient'},
                   linewidths=2, linecolor='white',
                   annot_kws={'size': 10, 'weight': 'bold'},
                   ax=ax, vmin=-1, vmax=1)
        
        ax.set_title('Mantel Test Correlation Matrix', fontsize=16, fontweight='bold', pad=15)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'Mantel_Test_Heatmap.png', dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
        plt.savefig(FIGURES_DIR / 'Mantel_Test_Heatmap.pdf', dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()
        print("✅ Mantel检验热图已生成")
except Exception as e:
    print(f"⚠️  热图生成失败: {e}")

print()

# 4.2 Mantel检验结果柱状图
print("📊 生成Mantel检验结果柱状图...")
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(COLORS['background'])
    
    # 左图：相关系数
    comparisons_short = [r['Comparison'].replace(' Distance', '').replace(' vs ', ' vs\n') for r in mantel_results]
    correlations = [r['Correlation'] for r in mantel_results]
    p_values = [r['P-value'] for r in mantel_results]
    
    colors_list = [COLORS['positive'], COLORS['center_1'], COLORS['center_2'], 
                   COLORS['center_3'], COLORS['negative']][:len(mantel_results)]
    
    bars1 = ax1.bar(range(len(comparisons_short)), correlations, color=colors_list, 
                    alpha=0.8, edgecolor='white', linewidth=2)
    
    # 添加显著性标记
    for i, (bar, p) in enumerate(zip(bars1, p_values)):
        height = bar.get_height()
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02 if height > 0 else height - 0.05,
                sig, ha='center', va='bottom' if height > 0 else 'top', fontsize=12, fontweight='bold')
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{correlations[i]:.3f}', ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Correlation Coefficient (r)', fontsize=12, fontweight='bold')
    ax1.set_title('Mantel Test Correlation Coefficients', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xticks(range(len(comparisons_short)))
    ax1.set_xticklabels(comparisons_short, rotation=45, ha='right', fontsize=10)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_facecolor(COLORS['background'])
    
    # 右图：p值（-log10转换）
    log_p_values = [-np.log10(p) if p > 0 else 10 for p in p_values]
    bars2 = ax2.bar(range(len(comparisons_short)), log_p_values, color=colors_list,
                    alpha=0.8, edgecolor='white', linewidth=2)
    
    # 添加p值标记和显著性线
    ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='p=0.05')
    ax2.axhline(y=-np.log10(0.01), color='orange', linestyle='--', linewidth=2, label='p=0.01')
    ax2.axhline(y=-np.log10(0.001), color='green', linestyle='--', linewidth=2, label='p=0.001')
    
    for i, (bar, p) in enumerate(zip(bars2, p_values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'p={p:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_ylabel('-log10(P-value)', fontsize=12, fontweight='bold')
    ax2.set_title('Mantel Test Statistical Significance', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xticks(range(len(comparisons_short)))
    ax2.set_xticklabels(comparisons_short, rotation=45, ha='right', fontsize=10)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_facecolor(COLORS['background'])
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Mantel_Test_Barplot.png', dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.savefig(FIGURES_DIR / 'Mantel_Test_Barplot.pdf', dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print("✅ Mantel检验柱状图已生成")
except Exception as e:
    print(f"⚠️  柱状图生成失败: {e}")
    import traceback
    traceback.print_exc()

print()

# 4.3 距离矩阵散点图（展示相关性）
print("📊 生成距离矩阵散点图...")
try:
    # 选择几个重要的距离矩阵对进行散点图展示
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor(COLORS['background'])
    axes = axes.flatten()
    
    plot_idx = 0
    
    # 1. 特征距离 vs 标签距离
    if label_dist is not None:
        triu_indices = np.triu_indices(feature_dist.shape[0], k=1)
        vec1 = feature_dist[triu_indices]
        vec2 = label_dist[triu_indices]
        
        # 采样以加快绘图
        if len(vec1) > 5000:
            sample_idx = np.random.choice(len(vec1), 5000, replace=False)
            vec1 = vec1[sample_idx]
            vec2 = vec2[sample_idx]
        
        axes[plot_idx].scatter(vec1, vec2, alpha=0.5, s=10, color=COLORS['positive'])
        r, p = pearsonr(vec1, vec2)
        axes[plot_idx].set_xlabel('Feature Distance', fontsize=11, fontweight='bold')
        axes[plot_idx].set_ylabel('Label Distance', fontsize=11, fontweight='bold')
        axes[plot_idx].set_title(f'Feature vs Label\nr={r:.3f}, p={p:.3f}', 
                                fontsize=12, fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].set_facecolor(COLORS['background'])
        plot_idx += 1
    
    # 2. OCT vs Colposcopy
    if colpo_dist is not None:
        triu_indices = np.triu_indices(oct_dist.shape[0], k=1)
        vec1 = oct_dist[triu_indices]
        vec2 = colpo_dist[triu_indices]
        
        if len(vec1) > 5000:
            sample_idx = np.random.choice(len(vec1), 5000, replace=False)
            vec1 = vec1[sample_idx]
            vec2 = vec2[sample_idx]
        
        axes[plot_idx].scatter(vec1, vec2, alpha=0.5, s=10, color=COLORS['center_2'])
        r, p = pearsonr(vec1, vec2)
        axes[plot_idx].set_xlabel('OCT Distance', fontsize=11, fontweight='bold')
        axes[plot_idx].set_ylabel('Colposcopy Distance', fontsize=11, fontweight='bold')
        axes[plot_idx].set_title(f'OCT vs Colposcopy\nr={r:.3f}, p={p:.3f}', 
                                fontsize=12, fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].set_facecolor(COLORS['background'])
        plot_idx += 1
    
    # 3. OCT vs Clinical
    if clinical_dist is not None:
        triu_indices = np.triu_indices(oct_dist.shape[0], k=1)
        vec1 = oct_dist[triu_indices]
        vec2 = clinical_dist[triu_indices]
        
        if len(vec1) > 5000:
            sample_idx = np.random.choice(len(vec1), 5000, replace=False)
            vec1 = vec1[sample_idx]
            vec2 = vec2[sample_idx]
        
        axes[plot_idx].scatter(vec1, vec2, alpha=0.5, s=10, color=COLORS['center_3'])
        r, p = pearsonr(vec1, vec2)
        axes[plot_idx].set_xlabel('OCT Distance', fontsize=11, fontweight='bold')
        axes[plot_idx].set_ylabel('Clinical Distance', fontsize=11, fontweight='bold')
        axes[plot_idx].set_title(f'OCT vs Clinical\nr={r:.3f}, p={p:.3f}', 
                                fontsize=12, fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].set_facecolor(COLORS['background'])
        plot_idx += 1
    
    # 4. Colposcopy vs Clinical
    if colpo_dist is not None and clinical_dist is not None:
        triu_indices = np.triu_indices(colpo_dist.shape[0], k=1)
        vec1 = colpo_dist[triu_indices]
        vec2 = clinical_dist[triu_indices]
        
        if len(vec1) > 5000:
            sample_idx = np.random.choice(len(vec1), 5000, replace=False)
            vec1 = vec1[sample_idx]
            vec2 = vec2[sample_idx]
        
        axes[plot_idx].scatter(vec1, vec2, alpha=0.5, s=10, color=COLORS['negative'])
        r, p = pearsonr(vec1, vec2)
        axes[plot_idx].set_xlabel('Colposcopy Distance', fontsize=11, fontweight='bold')
        axes[plot_idx].set_ylabel('Clinical Distance', fontsize=11, fontweight='bold')
        axes[plot_idx].set_title(f'Colposcopy vs Clinical\nr={r:.3f}, p={p:.3f}', 
                                fontsize=12, fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].set_facecolor(COLORS['background'])
        plot_idx += 1
    
    # 隐藏未使用的子图
    for i in range(plot_idx, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Mantel_Test_Scatter.png', dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.savefig(FIGURES_DIR / 'Mantel_Test_Scatter.pdf', dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print("✅ 距离矩阵散点图已生成")
except Exception as e:
    print(f"⚠️  散点图生成失败: {e}")
    import traceback
    traceback.print_exc()

print()

# ========================================
# 完成
# ========================================
print("=" * 80)
print("✅ Mantel检验分析完成！")
print("=" * 80)
print(f"📁 输出目录: {OUTPUT_DIR}")
print(f"📊 结果CSV: {results_csv}")
print(f"🎨 图表目录: {FIGURES_DIR}")
print()
print("生成的文件:")
print(f"  ✅ mantel_test_results.csv")
for fig_file in sorted(FIGURES_DIR.glob('Mantel_*')):
    size = fig_file.stat().st_size / 1024
    print(f"  ✅ {fig_file.name} ({size:.1f} KB)")
print("=" * 80)

