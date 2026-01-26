#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mantel检验相关性分析 - 增强版（参考标准样式）
- 左侧：下三角相关性矩阵（Pearson相关性 + Mantel检验结果）
- 右侧：网络图（特征与标签/中心之间的Mantel检验关系）
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
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise_distances
import networkx as nx

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
print("📊 Mantel检验相关性分析 - 增强版")
print("=" * 80)

# ========================================
# 1. 加载数据
# ========================================
print("\n📊 步骤 1: 准备数据...")

data_file = ROOT / 'visualization' / 'data' / 'Complete_Dataset.csv'
if data_file.exists():
    print(f"📖 从文件加载数据: {data_file}")
    df = pd.read_csv(data_file)
    
    feature_cols = [f'feature_{i}' for i in range(10) if f'feature_{i}' in df.columns]
    if feature_cols:
        features = df[feature_cols].values
        labels = df['label'].values if 'label' in df.columns else None
        center_ids = df['center_id'].values if 'center_id' in df.columns else None
        
        # 特征名称映射
        feature_names = {
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
        
        feature_display_names = [feature_names.get(col, col) for col in feature_cols]
        print(f"✅ 加载了 {len(features)} 个样本，{len(feature_cols)} 个特征")
    else:
        print("⚠️  未找到特征列，使用模拟数据")
        features = None
        labels = None
        center_ids = None
        feature_display_names = None
else:
    print("⚠️  数据文件不存在，使用模拟数据")
    features = None
    labels = None
    center_ids = None
    feature_display_names = None

if features is None:
    print("💡 生成模拟特征数据...")
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    features = np.random.randn(n_samples, n_features)
    labels = (features[:, 0] + features[:, 1] > 0).astype(int)
    center_ids = (features[:, 2] * 2 + 2).astype(int) % 5
    feature_display_names = [f'Feature {i}' for i in range(n_features)]
    print(f"✅ 生成了 {n_samples} 个模拟样本")

print()

# ========================================
# 2. 计算距离矩阵和Mantel检验
# ========================================
print("=" * 80)
print("📊 步骤 2: 计算距离矩阵和执行Mantel检验")
print("=" * 80)

# 计算特征距离矩阵
feature_dist = pairwise_distances(features, metric='euclidean')
print(f"✅ 特征距离矩阵: {feature_dist.shape}")

# 计算标签距离矩阵
if labels is not None:
    label_dist = pairwise_distances(labels.reshape(-1, 1), metric='hamming') * len(labels)
    print(f"✅ 标签距离矩阵: {label_dist.shape}")
else:
    label_dist = None

# 计算中心距离矩阵
if center_ids is not None:
    center_dist = pairwise_distances(center_ids.reshape(-1, 1), metric='hamming') * len(center_ids)
    print(f"✅ 中心距离矩阵: {center_dist.shape}")
else:
    center_dist = None

# Mantel检验函数
def mantel_test(dist1, dist2, n_permutations=999):
    """Mantel检验"""
    n = dist1.shape[0]
    triu_indices = np.triu_indices(n, k=1)
    vec1 = dist1[triu_indices]
    vec2 = dist2[triu_indices]
    
    r_obs, _ = pearsonr(vec1, vec2)
    
    r_perm = []
    for _ in range(n_permutations):
        perm_indices = np.random.permutation(n)
        dist2_perm = dist2[np.ix_(perm_indices, perm_indices)]
        vec2_perm = dist2_perm[triu_indices]
        r_perm_val, _ = pearsonr(vec1, vec2_perm)
        r_perm.append(r_perm_val)
    
    p_value = (np.sum(np.array(r_perm) >= r_obs) + 1) / (n_permutations + 1)
    return r_obs, p_value

# 计算特征之间的Pearson相关性矩阵
print("📊 计算特征之间的Pearson相关性...")
feature_corr = np.corrcoef(features.T)
print(f"✅ Pearson相关性矩阵: {feature_corr.shape}")

# 计算每个特征与标签/中心的Mantel检验
print("📊 计算特征与标签/中心的Mantel检验...")
mantel_results = []

# 为每个特征计算与标签的距离矩阵
if label_dist is not None:
    for i, feat_name in enumerate(feature_display_names):
        feat_dist = pairwise_distances(features[:, i:i+1], metric='euclidean')
        r, p = mantel_test(feat_dist, label_dist, n_permutations=999)
        mantel_results.append({
            'Feature': feat_name,
            'Target': 'Label',
            'Mantel_r': r,
            'Mantel_p': p
        })

# 为每个特征计算与中心的距离矩阵
if center_dist is not None:
    for i, feat_name in enumerate(feature_display_names):
        feat_dist = pairwise_distances(features[:, i:i+1], metric='euclidean')
        r, p = mantel_test(feat_dist, center_dist, n_permutations=999)
        mantel_results.append({
            'Feature': feat_name,
            'Target': 'Center',
            'Mantel_r': r,
            'Mantel_p': p
        })

mantel_df = pd.DataFrame(mantel_results)
print(f"✅ Mantel检验完成: {len(mantel_results)} 个检验")

# 保存结果
results_csv = OUTPUT_DIR / 'mantel_test_results_enhanced.csv'
mantel_df.to_csv(results_csv, index=False)
print(f"✅ 结果已保存: {results_csv}")
print()

# ========================================
# 3. 生成组合可视化（参考标准样式）
# ========================================
print("=" * 80)
print("🎨 步骤 3: 生成组合可视化图表")
print("=" * 80)

# 创建大图（左侧相关性矩阵 + 右侧网络图）
fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor(COLORS['background'])
gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.3)

# ========================================
# 左侧：下三角相关性矩阵
# ========================================
print("📊 生成下三角相关性矩阵...")
ax1 = fig.add_subplot(gs[0, 0])

# 创建下三角掩码
n_features = len(feature_display_names)
mask = np.triu(np.ones_like(feature_corr, dtype=bool), k=1)

# 创建显示矩阵（下三角 + 对角线）
display_corr = feature_corr.copy()
display_corr[mask] = np.nan  # 上三角设为NaN

# 计算每个特征对的Mantel检验（用于方块大小）
mantel_r_matrix = np.zeros_like(feature_corr)
mantel_p_matrix = np.ones_like(feature_corr)

# 为特征对计算Mantel检验（简化：使用特征距离）
for i in range(n_features):
    for j in range(i+1):
        if i == j:
            mantel_r_matrix[i, j] = 1.0
            mantel_p_matrix[i, j] = 0.0
        else:
            feat_i_dist = pairwise_distances(features[:, i:i+1], metric='euclidean')
            feat_j_dist = pairwise_distances(features[:, j:j+1], metric='euclidean')
            r, p = mantel_test(feat_i_dist, feat_j_dist, n_permutations=99)
            mantel_r_matrix[i, j] = r
            mantel_p_matrix[i, j] = p
            mantel_r_matrix[j, i] = r  # 对称
            mantel_p_matrix[j, i] = p

# 创建自定义colormap（从红色到白色到蓝色）
from matplotlib.colors import LinearSegmentedColormap
colors_list = ['#8B0000', '#DC143C', '#FF6B6B', '#FFFFFF', '#C7CCD6', '#5B8FA3', '#2874A6']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('correlation_cmap', colors_list, N=n_bins)

# 绘制热图
sns.heatmap(display_corr, mask=mask, annot=True, fmt='.2f', cmap=cmap,
           center=0, vmin=-1, vmax=1,
           square=True, linewidths=0.5, linecolor='white',
           cbar_kws={'label': "Pearson's correlation", 'shrink': 0.8, 'pad': 0.02},
           xticklabels=feature_display_names, yticklabels=feature_display_names,
           annot_kws={'size': 8, 'weight': 'bold'},
           ax=ax1)

# 在下三角中添加Mantel检验信息（方块大小和星号）
for i in range(n_features):
    for j in range(i+1):
        if i != j:
            r_mantel = mantel_r_matrix[i, j]
            p_mantel = mantel_p_matrix[i, j]
            
            # 根据Mantel's r值设置方块大小（通过添加额外的方块）
            # 这里我们通过调整annot的字体大小来模拟
            size_factor = max(0.3, min(1.0, abs(r_mantel) / 0.4))
            
            # 添加显著性星号
            if p_mantel < 0.001:
                sig = '***'
            elif p_mantel < 0.01:
                sig = '**'
            elif p_mantel < 0.05:
                sig = '*'
            else:
                sig = ''
            
            # 在相关性数值下方添加星号
            if sig:
                x_pos = j + 0.5
                y_pos = i + 0.7
                ax1.text(x_pos, y_pos, sig, ha='center', va='bottom', 
                        fontsize=10, fontweight='bold', color='black')

ax1.set_title("Pearson's Correlation Matrix\n(with Mantel Test Significance)", 
             fontsize=14, fontweight='bold', pad=15)
ax1.set_facecolor(COLORS['background'])

print("✅ 相关性矩阵已生成")

# ========================================
# 右侧：网络图
# ========================================
print("📊 生成网络图...")
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(COLORS['background'])
ax2.axis('off')

# 创建网络图
G = nx.Graph()

# 添加特征节点
feature_nodes = feature_display_names
for node in feature_nodes:
    G.add_node(node, node_type='feature')

# 添加目标节点（Label和Center）
target_nodes = []
if label_dist is not None:
    G.add_node('Label', node_type='target')
    target_nodes.append('Label')
if center_dist is not None:
    G.add_node('Center', node_type='target')
    target_nodes.append('Center')

# 添加边（基于Mantel检验结果）
for _, row in mantel_df.iterrows():
    feature = row['Feature']
    target = row['Target']
    r = row['Mantel_r']
    p = row['Mantel_p']
    
    if abs(r) > 0.1:  # 只显示有意义的连接
        G.add_edge(feature, target, weight=abs(r), p_value=p, r_value=r)

# 使用spring布局
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# 调整节点位置：特征节点在左侧，目标节点在右侧
feature_x = -1.5
target_x = 1.5
feature_y_positions = np.linspace(1, -1, len(feature_nodes))
target_y_positions = np.linspace(0.5, -0.5, len(target_nodes))

for i, node in enumerate(feature_nodes):
    pos[node] = (feature_x, feature_y_positions[i])

for i, node in enumerate(target_nodes):
    pos[node] = (target_x, target_y_positions[i])

# 绘制边（根据Mantel's r值和p值设置颜色和粗细）
edges = G.edges(data=True)
for u, v, data in edges:
    r = data['r_value']
    p = data['p_value']
    weight = data['weight']
    
    # 线条粗细：根据Mantel's r值
    if weight >= 0.4:
        width = 3.0
    elif weight >= 0.2:
        width = 2.0
    else:
        width = 1.0
    
    # 线条颜色：根据p值
    if p < 0.001:
        color = '#8B0000'  # 深红色
    elif p < 0.01:
        color = COLORS['positive']  # 粉棕色
    elif p < 0.05:
        color = COLORS['center_2']  # 中粉棕灰
    else:
        color = '#CCCCCC'  # 灰色
    
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, 
                           edge_color=color, alpha=0.7, ax=ax2)

# 绘制节点
feature_nodes_list = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'feature']
target_nodes_list = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'target']

nx.draw_networkx_nodes(G, pos, nodelist=feature_nodes_list, 
                      node_color=COLORS['center_2'], node_size=1500,
                      alpha=0.8, ax=ax2)
nx.draw_networkx_nodes(G, pos, nodelist=target_nodes_list,
                      node_color=COLORS['positive'], node_size=2000,
                      alpha=0.8, ax=ax2)

# 绘制标签
nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax2)

ax2.set_title("Mantel Test Network\n(Features vs Targets)", 
             fontsize=14, fontweight='bold', pad=15)

# 添加图例
print("📊 添加图例...")

# 图例1：Mantel's p value（颜色）
from matplotlib.patches import Patch
legend_elements_p = [
    Patch(facecolor='#8B0000', label='p < 0.001'),
    Patch(facecolor=COLORS['positive'], label='0.001 ≤ p < 0.01'),
    Patch(facecolor=COLORS['center_2'], label='0.01 ≤ p < 0.05'),
    Patch(facecolor='#CCCCCC', label='p ≥ 0.05'),
]

legend1 = ax2.legend(handles=legend_elements_p, loc='upper left', 
                    title="Mantel's p value", fontsize=9, title_fontsize=10)
legend1.get_frame().set_facecolor('white')
legend1.get_frame().set_alpha(0.9)

# 图例2：Mantel's r value（线条粗细）
from matplotlib.lines import Line2D
legend_elements_r = [
    Line2D([0], [0], color='black', lw=3, label='r ≥ 0.4'),
    Line2D([0], [0], color='black', lw=2, label='0.2 ≤ r < 0.4'),
    Line2D([0], [0], color='black', lw=1, label='r < 0.2'),
]

legend2 = ax2.legend(handles=legend_elements_r, loc='lower left',
                    title="Mantel's r value", fontsize=9, title_fontsize=10)
legend2.get_frame().set_facecolor('white')
legend2.get_frame().set_alpha(0.9)

# 恢复第一个图例
ax2.add_artist(legend1)

print("✅ 网络图已生成")

# 保存
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'Mantel_Test_Combined.png', dpi=300, bbox_inches='tight', 
           facecolor=COLORS['background'])
plt.savefig(FIGURES_DIR / 'Mantel_Test_Combined.pdf', dpi=300, bbox_inches='tight',
           facecolor=COLORS['background'])
plt.close()
print("✅ 组合图已保存")

print()

# ========================================
# 4. 生成详细的相关性矩阵（带Mantel信息）
# ========================================
print("📊 生成详细的相关性矩阵...")
try:
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor(COLORS['background'])
    
    # 创建下三角矩阵，包含Pearson相关性和Mantel检验信息
    display_matrix = np.full_like(feature_corr, np.nan)
    annot_matrix = np.full((n_features, n_features), '', dtype=object)
    
    for i in range(n_features):
        for j in range(i+1):
            if i == j:
                display_matrix[i, j] = 1.0
                annot_matrix[i, j] = '1.00'
            else:
                pearson_r = feature_corr[i, j]
                mantel_r = mantel_r_matrix[i, j]
                mantel_p = mantel_p_matrix[i, j]
                
                display_matrix[i, j] = pearson_r
                
                # 标注：Pearson相关性 + Mantel显著性
                if mantel_p < 0.001:
                    sig = '***'
                elif mantel_p < 0.01:
                    sig = '**'
                elif mantel_p < 0.05:
                    sig = '*'
                else:
                    sig = ''
                
                annot_matrix[i, j] = f'{pearson_r:.2f}\n{sig}'
    
    # 绘制热图
    mask = np.triu(np.ones_like(display_matrix, dtype=bool), k=1)
    sns.heatmap(display_matrix, mask=mask, annot=annot_matrix, fmt='',
               cmap=cmap, center=0, vmin=-1, vmax=1,
               square=True, linewidths=1, linecolor='white',
               cbar_kws={'label': "Pearson's correlation", 'shrink': 0.8},
               xticklabels=feature_display_names, yticklabels=feature_display_names,
               annot_kws={'size': 9, 'weight': 'bold', 'va': 'center'},
               ax=ax)
    
    ax.set_title("Feature Correlation Matrix with Mantel Test\n" +
                "(Lower triangle: Pearson's r; Significance: Mantel's p)", 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_facecolor(COLORS['background'])
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Mantel_Test_Correlation_Matrix.png', dpi=300, 
               bbox_inches='tight', facecolor=COLORS['background'])
    plt.savefig(FIGURES_DIR / 'Mantel_Test_Correlation_Matrix.pdf', dpi=300,
               bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print("✅ 详细相关性矩阵已生成")
except Exception as e:
    print(f"⚠️  详细矩阵生成失败: {e}")
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
print(f"  ✅ mantel_test_results_enhanced.csv")
for fig_file in sorted(FIGURES_DIR.glob('Mantel_Test_*')):
    size = fig_file.stat().st_size / 1024
    print(f"  ✅ {fig_file.name} ({size:.1f} KB)")
print("=" * 80)

