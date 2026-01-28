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
    return CENTER_NAMES.get(int(center_id), f'Center {chr(65 + int(center_id))}')  # 65是'A'的ASCII码

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
        
        # 统计检验信息
        stats_text = []
        
        # 正态性检验
        from scipy.stats import shapiro, normaltest
        try:
            # 如果样本量>5000，使用normaltest；否则使用shapiro
            if len(data) > 5000:
                stat, p_norm = normaltest(data)
                test_name = 'D\'Agostino'
            else:
                stat, p_norm = shapiro(data)
                test_name = 'Shapiro-Wilk'
            
            is_normal = p_norm > 0.05
            norm_label = 'Normal' if is_normal else f'Non-normal (p={p_norm:.3f})'
            stats_text.append(f'{test_name}: {norm_label}')
        except:
            pass
        
        # 如果按标签分组，添加组间比较
        if 'Label' in feature_df.columns:
            feat_neg = feature_df[feature_df['Label'] == 0][feat_col].values
            feat_pos = feature_df[feature_df['Label'] == 1][feat_col].values
            
            if len(feat_neg) > 0 and len(feat_pos) > 0:
                from scipy.stats import ttest_ind, mannwhitneyu
                
                # 先检查正态性，决定使用参数还是非参数检验
                try:
                    _, p_neg = shapiro(feat_neg) if len(feat_neg) <= 5000 else normaltest(feat_neg)
                    _, p_pos = shapiro(feat_pos) if len(feat_pos) <= 5000 else normaltest(feat_pos)
                    both_normal = p_neg > 0.05 and p_pos > 0.05
                except:
                    both_normal = False
                
                if both_normal:
                    # 使用t检验
                    from scipy.stats import ttest_ind
                    t_stat, p_val = ttest_ind(feat_neg, feat_pos)
                    test_name = 't-test'
                    stat_val = f't={t_stat:.2f}'
                else:
                    # 使用Mann-Whitney U检验
                    from scipy.stats import mannwhitneyu
                    u_stat, p_val = mannwhitneyu(feat_neg, feat_pos, alternative='two-sided')
                    test_name = 'Mann-Whitney U'
                    stat_val = f'U={u_stat:.0f}'
                
                # 显著性标注
                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                else:
                    sig = 'ns'
                
                # 统计信息不显示在图表中（根据用户要求）
                # 图例中不包含任何统计信息或显著性符号
                pass
        
        # 统计信息框已移除，图例只显示数据分组信息
        
        # 2. Violin plot (如果有分组) - 优化位置
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
                pc.set_alpha=0.6
        
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
    # 智能选择特征：优先选择对标签分离有显著贡献的特征
    feature_cols_available = [col for col in feature_df.columns if col not in ['Label', 'Center', 'Label_Name']]
    
    if len(feature_cols_available) >= 3:
        # 如果有标签，计算每个特征对标签分离的贡献
        if 'Label' in feature_df.columns:
            from scipy.stats import ttest_ind
            feature_scores = []
            for col in feature_cols_available:
                feat_neg = feature_df[feature_df['Label'] == 0][col].values
                feat_pos = feature_df[feature_df['Label'] == 1][col].values
                if len(feat_neg) > 0 and len(feat_pos) > 0:
                    t_stat, p_val = ttest_ind(feat_neg, feat_pos)
                    # 使用t统计量的绝对值作为重要性评分
                    feature_scores.append((col, abs(t_stat), p_val))
            
            # 按重要性排序，选择前6个
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            feature_cols_for_matrix = [col for col, _, _ in feature_scores[:6]]
            print(f"  📊 选择特征（基于标签分离度）: {len(feature_cols_for_matrix)}个特征")
        else:
            # 如果没有标签，选择前6个特征
            feature_cols_for_matrix = feature_cols_available[:6]
        
        if len(feature_cols_for_matrix) >= 3:
            # 优化：直接选择类别分离度最高的前4个特征（而不是基于相关性筛选）
            # 重新计算每个特征的类别分离度（Cohen's d），选择分离度最高的
            from scipy.stats import ttest_ind
            
            feature_separation_scores = []
            for col in feature_cols_for_matrix:
                if 'Label' in feature_df.columns:
                    feat_neg = feature_df[feature_df['Label'] == 0][col].values
                    feat_pos = feature_df[feature_df['Label'] == 1][col].values
                elif 'Label_Name' in feature_df.columns:
                    feat_neg = feature_df[feature_df['Label_Name'] == 'Negative'][col].values
                    feat_pos = feature_df[feature_df['Label_Name'] == 'Positive'][col].values
                else:
                    continue
                    
                if len(feat_neg) > 0 and len(feat_pos) > 0:
                    # 计算Cohen's d（效应量，更好的分离度指标）
                    mean_diff = abs(feat_pos.mean() - feat_neg.mean())
                    pooled_std = np.sqrt((feat_neg.std()**2 + feat_pos.std()**2) / 2)
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                    t_stat, p_val = ttest_ind(feat_neg, feat_pos)
                    feature_separation_scores.append((col, cohens_d, abs(t_stat), p_val))
            
            # 按Cohen's d排序，选择分离度最高的特征
            feature_separation_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 只选择分离度足够高的特征（Cohen's d > 0.3 且 p < 0.05）
            # 这样可以确保选择的特征真正能区分阳性和阴性
            high_separation_features = [(col, d, t, p) for col, d, t, p in feature_separation_scores 
                                       if d > 0.3 and p < 0.05]
            
            if len(high_separation_features) >= 2:
                # 如果分离度高的特征足够，选择前4个（或全部，如果少于4个）
                selected_features = [col for col, _, _, _ in high_separation_features[:4]]
                print(f"  📊 选择类别分离度足够高的 {len(selected_features)} 个特征 (Cohen's d > 0.3, p < 0.05):")
                for i, (col, d, t_stat, p_val) in enumerate([(col, d, t, p) for col, d, t, p in feature_separation_scores if col in selected_features]):
                    sep_status = "✅ 高分离度" if d > 0.3 and p_val < 0.05 else ("⚠️ 中等分离度" if d > 0.2 or p_val < 0.05 else "❌ 低分离度")
                    print(f"     {i+1}. {col}: Cohen's d={d:.3f}, |t|={t_stat:.2f}, p={p_val:.4f} {sep_status}")
                
                # 创建筛选后的pairplot（使用原始特征）
                if 'Label_Name' in feature_df.columns:
                    plot_df = feature_df[selected_features + ['Label_Name']].copy()
                    hue_col = 'Label_Name'
                else:
                    plot_df = feature_df[selected_features + (['Label'] if 'Label' in feature_df.columns else [])].copy()
                    hue_col = 'Label' if 'Label' in plot_df.columns else None
            else:
                # 如果分离度高的特征太少，使用PCA降维创建更好的特征空间
                print(f"  ⚠️  高分离度特征较少（只有{len(high_separation_features)}个），使用PCA降维创建特征空间...")
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                
                # 使用所有特征进行PCA
                all_feature_data = feature_df[feature_cols_for_matrix].values
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(all_feature_data)
                
                # PCA降维到4个主成分
                pca = PCA(n_components=min(4, len(feature_cols_for_matrix)))
                pca_features = pca.fit_transform(scaled_data)
                
                # 创建PCA特征DataFrame
                pca_df = pd.DataFrame(pca_features, columns=[f'PC{i+1}' for i in range(pca_features.shape[1])])
                
                # 计算PCA主成分的类别分离度
                pca_separation = []
                for pc_col in pca_df.columns:
                    if 'Label' in feature_df.columns:
                        pc_neg = pca_df.loc[feature_df['Label'] == 0, pc_col].values
                        pc_pos = pca_df.loc[feature_df['Label'] == 1, pc_col].values
                    elif 'Label_Name' in feature_df.columns:
                        pc_neg = pca_df.loc[feature_df['Label_Name'] == 'Negative', pc_col].values
                        pc_pos = pca_df.loc[feature_df['Label_Name'] == 'Positive', pc_col].values
                    else:
                        continue
                    
                    if len(pc_neg) > 0 and len(pc_pos) > 0:
                        mean_diff = abs(pc_pos.mean() - pc_neg.mean())
                        pooled_std = np.sqrt((pc_neg.std()**2 + pc_pos.std()**2) / 2)
                        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                        t_stat, p_val = ttest_ind(pc_neg, pc_pos)
                        pca_separation.append((pc_col, cohens_d, abs(t_stat), p_val))
                
                pca_separation.sort(key=lambda x: x[1], reverse=True)
                
                # 计算并显示每个主成分的主要载荷（loadings）
                pca_loadings = pca.components_  # [n_components, n_features]
                feature_names_list = [FEATURE_NAMES.get(col, col) for col in feature_cols_for_matrix]
                
                # 选择分离度最高的PCA主成分
                selected_pc = [pc for pc, _, _, _ in pca_separation[:4]]
                selected_features = selected_pc
                
                # 更新plot_df使用PCA特征
                plot_df = pca_df[selected_pc].copy()
                if 'Label_Name' in feature_df.columns:
                    plot_df['Label_Name'] = feature_df['Label_Name'].values
                    hue_col = 'Label_Name'
                elif 'Label' in feature_df.columns:
                    plot_df['Label'] = feature_df['Label'].values
                    hue_col = 'Label'
                else:
                    hue_col = None
                
                print(f"  📊 PCA主成分分离度:")
                print(f"     (PC1-PC4是主成分分析的结果，将原始特征线性组合得到的新特征空间)")
                
                # 存储PCA信息用于标题
                pca_info_dict = {}
                for i, (pc, d, t_stat, p_val) in enumerate(pca_separation[:len(selected_pc)]):
                    pc_idx = int(pc.replace('PC', '')) - 1
                    var_explained = pca.explained_variance_ratio_[pc_idx] * 100
                    
                    # 获取该主成分的载荷
                    loadings = pca_loadings[pc_idx]
                    # 找到载荷最大的前2个特征
                    top_indices = np.argsort(np.abs(loadings))[-2:][::-1]
                    top_features = [f"{feature_names_list[idx]}" for idx in top_indices]
                    
                    print(f"     {i+1}. {pc}: Cohen's d={d:.3f}, |t|={t_stat:.2f}, p={p_val:.4f}, 方差解释={var_explained:.1f}%")
                    print(f"        主要特征: {', '.join(top_features)}")
                    
                    # 存储信息用于标题
                    pca_info_dict[pc] = {
                        'top_features': top_features,
                        'var_explained': var_explained
                    }
            
            # 使用seaborn的pairplot，配色与Conditional_Means保持一致
            # 使用更深的颜色，与Conditional_Means保持一致
            deeper_palette = ['#A8B5C6', '#B87A6A']  # 更深的蓝灰色(Negative)和红棕色(Positive)，与Conditional_Means一致
            g = sns.pairplot(plot_df, hue=hue_col,
                            palette=deeper_palette if hue_col else None,
                            hue_order=['Negative', 'Positive'] if hue_col == 'Label_Name' else None,
                            diag_kind='kde', 
                            plot_kws={'alpha': 0.8, 's': 60, 'edgecolors': 'white', 'linewidths': 1.0},  # 提高alpha和size，颜色更深
                            diag_kws={'alpha': 0.9, 'fill': True, 'linewidth': 3.0})  # 增强KDE曲线
            
            # 根据是否使用PCA设置标题和说明
            if selected_features[0].startswith('PC'):
                title = 'Scatterplot Matrix (PCA Components)'
                # 生成详细的PCA说明
                if 'pca_info_dict' in locals():
                    pca_info = []
                    for pc in selected_features:
                        if pc in pca_info_dict:
                            info = pca_info_dict[pc]
                            top_feat = info['top_features'][0] if info['top_features'] else 'Mixed'
                            var_exp = info['var_explained']
                            pca_info.append(f"{pc}: {top_feat} ({var_exp:.1f}%)")
                    subtitle = f"PCA Components: {'; '.join(pca_info)}"
                else:
                    subtitle = 'PC1-PC4: Principal Components from PCA dimensionality reduction'
            else:
                title = 'Scatterplot Matrix (Selected Features)'
                subtitle = None
            
            g.fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
            if subtitle:
                g.fig.text(0.5, 0.98, subtitle, ha='center', fontsize=8, style='italic', color='gray', wrap=True)
            g.fig.patch.set_facecolor(COLORS['background'])
            
            # 获取实际的特征列名（可能是原始特征或PCA主成分）
            actual_feature_cols = [col for col in plot_df.columns if col not in ['Label', 'Label_Name']]
            n_features = len(actual_feature_cols)
            
            # 设置所有子图的背景色、坐标轴标签，并优化可视化
            for i in range(n_features):
                for j in range(n_features):
                    ax = g.axes[i, j]
                    ax.set_facecolor(COLORS['background'])
                    # 增强网格线以提高可读性
                    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                    
                    # 设置坐标轴标签为特征名称（英文）
                    if i == n_features - 1:  # 最后一行，设置x轴标签
                        x_col = actual_feature_cols[j]
                        if x_col.startswith('PC'):
                            # PCA主成分，使用主要特征名称
                            pc_idx = int(x_col.replace('PC', '')) - 1
                            if 'pca_info_dict' in locals() and x_col in pca_info_dict:
                                top_feat = pca_info_dict[x_col]['top_features'][0] if pca_info_dict[x_col]['top_features'] else x_col
                                ax.set_xlabel(top_feat, fontsize=10, fontweight='bold')
                            else:
                                ax.set_xlabel(x_col, fontsize=10, fontweight='bold')
                        else:
                            # 原始特征，使用特征名称
                            feature_name = FEATURE_NAMES.get(x_col, x_col)
                            ax.set_xlabel(feature_name, fontsize=10, fontweight='bold')
                    
                    if j == 0:  # 第一列，设置y轴标签
                        y_col = actual_feature_cols[i]
                        if y_col.startswith('PC'):
                            # PCA主成分，使用主要特征名称
                            pc_idx = int(y_col.replace('PC', '')) - 1
                            if 'pca_info_dict' in locals() and y_col in pca_info_dict:
                                top_feat = pca_info_dict[y_col]['top_features'][0] if pca_info_dict[y_col]['top_features'] else y_col
                                ax.set_ylabel(top_feat, fontsize=10, fontweight='bold')
                            else:
                                ax.set_ylabel(y_col, fontsize=10, fontweight='bold')
                        else:
                            # 原始特征，使用特征名称
                            feature_name = FEATURE_NAMES.get(y_col, y_col)
                            ax.set_ylabel(feature_name, fontsize=10, fontweight='bold')
            
            # 添加相关系数标注到每个散点图（只标注显著的相关性）
            from scipy.stats import pearsonr
            # actual_feature_cols 和 n_features 已在上面定义
            for i in range(n_features):
                for j in range(n_features):
                    if i != j:  # 非对角线
                        ax = g.axes[i, j]
                        x_col = actual_feature_cols[j]
                        y_col = actual_feature_cols[i]
                        x_data = plot_df[x_col].values
                        y_data = plot_df[y_col].values
                        # 检查数据是否有效
                        if len(x_data) > 0 and len(y_data) > 0 and len(x_data) == len(y_data):
                            # 检查是否有变化（避免除以0）
                            if np.std(x_data) > 1e-10 and np.std(y_data) > 1e-10:
                                r, p = pearsonr(x_data, y_data)
                                
                                # 检查r值是否有效
                                if not np.isnan(r) and not np.isinf(r):
                                    # 使用实际的r值，显示真实计算结果
                                    r_display = r
                                    
                                    # 只有在r值确实不为0时才在图例和图上显示（如果接近0，不显示）
                                    if abs(r_display) > 1e-4:  # 如果r值不是接近0（阈值设为1e-4，避免显示数值误差）
                                        # 在图例中显示r值
                                        legend = ax.get_legend()
                                        if legend is not None:
                                            # 获取图例的handles和labels
                                            try:
                                                handles = legend.legendHandles
                                            except AttributeError:
                                                handles = legend.get_lines() + legend.get_patches()
                                            labels = [t.get_text() for t in legend.get_texts()]
                                            # 更新图例标签，添加r值
                                            new_labels = []
                                            for label in labels:
                                                if label in ['Negative', 'Positive']:
                                                    # 添加该子图的r值，保留1位小数
                                                    new_labels.append(f"{label} (r={r_display:.1f})")
                                                else:
                                                    new_labels.append(label)
                                            # 重新创建图例
                                            ax.legend(handles, new_labels, fontsize=8, loc='best', framealpha=0.9)
                                        
                                        # 在图上显示r值（保留1位小数）
                                        if abs(r_display) > 0.1 or p < 0.05:
                                            # 显著相关性用粗体显示
                                            ax.text(0.95, 0.95, f'r={r_display:.1f}', 
                                                   transform=ax.transAxes,
                                                   fontsize=10, fontweight='bold',
                                                   ha='right', va='top',
                                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray', linewidth=1.5))
                                        else:
                                            # 弱相关性也显示，但用浅色
                                            ax.text(0.95, 0.95, f'r={r_display:.1f}', 
                                                   transform=ax.transAxes,
                                                   fontsize=9, alpha=0.7,
                                                   ha='right', va='top',
                                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray'))
                                    # 如果r值接近0（<1e-4），不显示（不写）
                                else:
                                    # r值无效，显示N/A
                                    ax.text(0.95, 0.95, 'r=N/A', 
                                           transform=ax.transAxes,
                                           fontsize=7, alpha=0.4,
                                           ha='right', va='top')
                            else:
                                # 数据无变化，r值确实为0，不显示（不写）
                                pass
                        else:
                            # 数据长度不匹配
                            ax.text(0.95, 0.95, 'r=N/A', 
                                   transform=ax.transAxes,
                                   fontsize=7, alpha=0.4,
                                   ha='right', va='top')
        
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
            # 使用更深的颜色，与UMAP_2D和Scatterplot_Matrix保持一致
            # 使用DEEPER_COLORS风格的更深颜色
            deeper_center_colors = ['#B87A6A', '#A88B7F', '#aba09f', '#adb5bf', '#d6dadf']  # 更深的渐变色
            
            # 绘制原始观测值（按Center分组，使用更深的颜色）
            for i, center in enumerate(centers):
                mask = feature_df['Center'] == center
                data = feature_df.loc[mask, feat_col]
                x_pos = np.random.normal(i, 0.1, len(data))
                center_name = get_center_name(center)
                # 使用更深的颜色，提高alpha和size，使其更清晰
                ax.scatter(x_pos, data, alpha=0.8, s=50, 
                          color=deeper_center_colors[i % len(deeper_center_colors)], 
                          edgecolors='white', linewidths=1.0, label=center_name, zorder=5)
            
            # 绘制条件均值
            means = []
            stds = []
            n_samples = []
            center_data_list = []
            
            for center in centers:
                mask = feature_df['Center'] == center
                data = feature_df.loc[mask, feat_col]
                means.append(data.mean())
                stds.append(data.std())
                n_samples.append(len(data))
                center_data_list.append(data.values)
            
            x_positions = np.arange(len(centers))
            # 使用更深的颜色绘制均值线和误差棒，与UMAP_2D保持一致
            # 使用圆形标记（'o'），确保图例中显示为圆
            ax.errorbar(x_positions, means, yerr=stds, fmt='o', 
                       markersize=14, capsize=8, capthick=3.5,
                       color='#B87A6A', linewidth=3.5,  # 使用更深的红棕色
                       label='Mean ± Std', zorder=10, elinewidth=3.0,
                       markerfacecolor='#B87A6A', markeredgecolor='white', markeredgewidth=2)
            
            # 连接均值点（使用更深的颜色）
            ax.plot(x_positions, means, color='#B87A6A', 
                   linewidth=3.5, linestyle='--', alpha=0.95, zorder=9)
            
            # 统计检验：ANOVA或Kruskal-Wallis
            from scipy.stats import f_oneway, kruskal, shapiro
            
            stats_text = []
            
            # 统计检验：ANOVA或Kruskal-Wallis
            from scipy.stats import f_oneway, kruskal, shapiro, normaltest
            
            stats_text = []
            
            # 检查数据是否满足ANOVA假设（正态性和方差齐性）
            try:
                # 检查每个组的正态性
                all_normal = True
                for data in center_data_list:
                    if len(data) > 5000:
                        _, p_norm = normaltest(data)
                    elif len(data) >= 3:
                        _, p_norm = shapiro(data)
                    else:
                        all_normal = False
                        break
                    if p_norm <= 0.05:
                        all_normal = False
                        break
                
                # 选择检验方法
                if all_normal and len(centers) >= 2:
                    # 使用ANOVA
                    f_stat, p_val = f_oneway(*center_data_list)
                    test_name = 'ANOVA'
                    stat_val = f'F={f_stat:.2f}'
                else:
                    # 使用Kruskal-Wallis（非参数）
                    h_stat, p_val = kruskal(*center_data_list)
                    test_name = 'Kruskal-Wallis'
                    stat_val = f'H={h_stat:.2f}'
                
                # 显著性标注
                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                else:
                    sig = 'ns'
                
                stats_text.append(f'{test_name}: {stat_val}, p={p_val:.3f} {sig}')
                
                # 添加样本量信息
                n_str = ', '.join([f'C{c}:n={n}' for c, n in zip(centers, n_samples)])
                stats_text.append(f'({n_str})')
                
            except Exception as e:
                stats_text.append(f'Statistical test failed: {str(e)[:30]}')
            
            # 设置x轴标签（使用中心名称）
            ax.set_xticks(x_positions)
            ax.set_xticklabels([get_center_name(c) for c in centers], 
                             rotation=45, ha='right', fontsize=10)
            
            ax.set_title(f'Conditional Means: {feat_col}', fontsize=12, fontweight='bold', pad=10)
            ax.set_ylabel(feat_col, fontsize=10)
            
            # 图例位置优化 - 放在图中右上角，避免与x轴标签重叠
            # 先获取y轴范围，确保图例在数据区域上方
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            # 调整y轴范围，为图例留出空间
            ax.set_ylim(y_min, y_max + y_range * 0.12)
            
            # 创建自定义图例，确保Mean ± Std显示为圆形标记
            handles, labels = ax.get_legend_handles_labels()
            # 找到Mean ± Std的handle并确保它是圆形
            for i, label in enumerate(labels):
                if 'Mean ± Std' in label:
                    # 创建一个新的圆形标记用于图例
                    from matplotlib.lines import Line2D
                    handles[i] = Line2D([0], [0], marker='o', color='w', 
                                       markerfacecolor='#B87A6A', markersize=12,
                                       markeredgecolor='white', markeredgewidth=2,
                                       linestyle='--', linewidth=3.5, alpha=0.95)
            
            ax.legend(handles, labels, fontsize=9, loc='upper right', framealpha=0.95, 
                     bbox_to_anchor=(0.98, 0.98), ncol=1, 
                     edgecolor='gray', fancybox=True, shadow=False)
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

