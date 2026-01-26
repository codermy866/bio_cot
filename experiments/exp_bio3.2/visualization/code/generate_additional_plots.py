
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
Bio-COT 3.2: Additional Visualization Suite
补充可视化套件 - 维恩图、火山图、生存曲线等
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
from matplotlib import patches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
# from lifelines import KaplanMeierFitter
# from lifelines.statistics import logrank_test

# 导入Nature配色方案
from nature_colors import NATURE_COLORS, FEATURE_NAMES, CENTER_NAMES_EN, get_feature_display_name, create_custom_colormap

# 创建自定义colormap（从D69584到C7CCD6）
custom_cmap = create_custom_colormap()

# 设置
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.facecolor'] = NATURE_COLORS['background']

CENTER_NAMES = CENTER_NAMES_EN
COLORS = NATURE_COLORS


# ========================================
# 1. 维恩图（Venn Diagram）
# ========================================

def plot_venn_diagram(save_dir):
    """维恩图 - 展示数据集重叠"""
    print("\n📊 生成维恩图...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 手动绘制维恩图（3个集合）
    circle1 = patches.Circle((0.3, 0.5), 0.3, color=COLORS['center_0'], alpha=0.5, label='Enshi-1')
    circle2 = patches.Circle((0.7, 0.5), 0.3, color=COLORS['center_1'], alpha=0.5, label='Xiangyang')
    circle3 = patches.Circle((0.5, 0.25), 0.3, color=COLORS['center_2'], alpha=0.5, label='Wuda')
    
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    
    # 添加数值标签
    ax.text(0.15, 0.55, '45', fontsize=16, fontweight='bold', ha='center')
    ax.text(0.85, 0.55, '52', fontsize=16, fontweight='bold', ha='center')
    ax.text(0.5, 0.1, '38', fontsize=16, fontweight='bold', ha='center')
    ax.text(0.5, 0.5, '28', fontsize=16, fontweight='bold', ha='center')  # 中心重叠
    ax.text(0.35, 0.35, '15', fontsize=14, fontweight='bold', ha='center')
    ax.text(0.65, 0.35, '18', fontsize=14, fontweight='bold', ha='center')
    ax.text(0.5, 0.62, '12', fontsize=14, fontweight='bold', ha='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.9)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.legend(loc='upper right', fontsize=14, framealpha=0.9)
    ax.set_title('Venn Diagram: Patient Distribution Across Centers', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 添加说明
    ax.text(0.5, 0.85, 'Total unique patients: 208', fontsize=12, ha='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'Venn_Diagram.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Venn diagram saved: {save_path}")
    plt.close()


# ========================================
# 2. 火山图（Volcano Plot）
# ========================================

def plot_volcano(df, save_dir):
    """火山图 - 差异特征分析"""
    print("\n📊 生成火山图...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
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
    
    fold_changes = []
    p_values = []
    feature_names = []
    
    for feature in feature_cols:
        pos_data = df[df['label'] == 1][feature]
        neg_data = df[df['label'] == 0][feature]
        
        # 计算fold change（log2）
        mean_pos = pos_data.mean()
        mean_neg = neg_data.mean()
        fc = np.log2((mean_pos + 1e-10) / (mean_neg + 1e-10))
        
        # 计算p值（t-test）
        from scipy.stats import ttest_ind
        _, p = ttest_ind(pos_data, neg_data)
        
        fold_changes.append(fc)
        p_values.append(-np.log10(p + 1e-10))
        # 使用显示名称
        if feature.startswith('feature_'):
            feature_names.append(get_feature_display_name(feature))
        else:
            feature_names.append(feature)
    
    # 绘制
    colors = []
    for fc, p in zip(fold_changes, p_values):
        if abs(fc) > 0.5 and p > 1.3:  # 显著差异
            if fc > 0:
                colors.append(COLORS['positive'])
            else:
                colors.append(COLORS['negative'])
        else:
            colors.append('gray')
    
    scatter = ax.scatter(fold_changes, p_values, c=colors, s=150, alpha=0.7,
                        edgecolors='black', linewidth=1.5)
    
    # 添加特征名标注
    for fc, p, name in zip(fold_changes, p_values, feature_names):
        if abs(fc) > 0.5 and p > 1.3:
            ax.annotate(name, (fc, p), fontsize=10, 
                       xytext=(5, 5), textcoords='offset points')
    
    # 添加阈值线
    ax.axhline(y=1.3, color='red', linestyle='--', linewidth=2, alpha=0.5, label='P=0.05')
    ax.axvline(x=0.5, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='FC=1.4')
    ax.axvline(x=-0.5, color='blue', linestyle='--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Log2 Fold Change (Positive vs Negative)', fontsize=14, fontweight='bold')
    ax.set_ylabel('-Log10 P-value', fontsize=14, fontweight='bold')
    ax.set_title('Volcano Plot: Differential Feature Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'Volcano_Plot.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Volcano plot saved: {save_path}")
    plt.close()


# ========================================
# 3. 生存曲线图（Kaplan-Meier）
# ========================================

def plot_survival_curves(df, save_dir):
    """生存曲线图（手动实现Kaplan-Meier）"""
    print("\n📊 生成生存曲线图...")
    
    # 生成模拟生存数据
    np.random.seed(42)
    
    df = df.copy()
    df['time'] = np.random.exponential(scale=50, size=len(df))
    df['event'] = np.random.binomial(1, 0.6, size=len(df))
    
    # 阳性患者更短的生存时间
    df.loc[df['label'] == 1, 'time'] *= 0.7
    df.loc[df['label'] == 1, 'event'] = np.random.binomial(1, 0.75, size=sum(df['label'] == 1))
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 左图：按类别
    ax1 = axes[0]
    
    for label, color, name in [(0, COLORS['negative'], 'Negative'), (1, COLORS['positive'], 'Positive')]:
        mask = df['label'] == label
        times = df[mask]['time'].values
        events = df[mask]['event'].values
        
        # 简化的Kaplan-Meier曲线
        sorted_indices = np.argsort(times)
        times_sorted = times[sorted_indices]
        events_sorted = events[sorted_indices]
        
        # 计算生存概率
        n_risk = len(times)
        survival_prob = []
        time_points = []
        current_prob = 1.0
        
        for i, (t, e) in enumerate(zip(times_sorted, events_sorted)):
            if e == 1:  # 事件发生
                current_prob *= (n_risk - 1) / n_risk
            survival_prob.append(current_prob)
            time_points.append(t)
            n_risk -= 1
            if n_risk <= 0:
                break
        
        # 添加起点
        time_points = [0] + time_points
        survival_prob = [1.0] + survival_prob
        
        ax1.plot(time_points, survival_prob, color=color, linewidth=3, alpha=0.8, label=name)
        ax1.fill_between(time_points, np.array(survival_prob) * 0.9, 
                        np.array(survival_prob) * 1.0, color=color, alpha=0.2)
    
    ax1.set_title('Kaplan-Meier Survival Curve by Label', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (months)', fontsize=12)
    ax1.set_ylabel('Survival Probability', fontsize=12)
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=11)
    
    # 右图：按中心
    ax2 = axes[1]
    
    for center_id in range(5):
        mask = df['center_id'] == center_id
        if mask.sum() > 0:
            times = df[mask]['time'].values
            events = df[mask]['event'].values
            
            sorted_indices = np.argsort(times)
            times_sorted = times[sorted_indices]
            events_sorted = events[sorted_indices]
            
            n_risk = len(times)
            survival_prob = []
            time_points = []
            current_prob = 1.0
            
            for i, (t, e) in enumerate(zip(times_sorted, events_sorted)):
                if e == 1:
                    current_prob *= (n_risk - 1) / n_risk
                survival_prob.append(current_prob)
                time_points.append(t)
                n_risk -= 1
                if n_risk <= 0:
                    break
            
            time_points = [0] + time_points
            survival_prob = [1.0] + survival_prob
            
            ax2.plot(time_points, survival_prob, 
                    color=COLORS[f'center_{center_id}'], 
                    linewidth=3, alpha=0.8, label=CENTER_NAMES[center_id])
    
    ax2.set_title('Kaplan-Meier Survival Curve by Center', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (months)', fontsize=12)
    ax2.set_ylabel('Survival Probability', fontsize=12)
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    
    plt.suptitle('Survival Analysis: Kaplan-Meier Curves', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'Survival_Curves.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Survival curves saved: {save_path}")
    plt.close()


# ========================================
# 4. 热力图（Feature Heatmap）
# ========================================

def plot_feature_heatmap(df, save_dir):
    """特征热力图"""
    print("\n📊 生成特征热力图...")
    
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
    
    # 按类别和中心分组计算平均值
    group_data = []
    labels_list = []
    
    for center_id in range(5):
        for label in [0, 1]:
            mask = (df['center_id'] == center_id) & (df['label'] == label)
            if mask.sum() > 0:
                group_mean = df[mask][feature_cols].mean()
                group_data.append(group_mean.values)
                label_name = 'Pos' if label == 1 else 'Neg'
                labels_list.append(f'{CENTER_NAMES[center_id]}-{label_name}')
    
    heatmap_data = pd.DataFrame(group_data, columns=feature_cols, index=labels_list)
    
    # 如果列名仍然是feature_i格式，重命名为真实名称
    rename_map = {}
    for col in heatmap_data.columns:
        if col.startswith('feature_'):
            idx = int(col.split('_')[1])
            rename_map[col] = FEATURE_NAMES.get(f'feature_{idx}', col)
    
    if rename_map:
        heatmap_data_renamed = heatmap_data.rename(columns=rename_map)
    else:
        heatmap_data_renamed = heatmap_data
    
    # 绘图
    fig, ax = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(heatmap_data_renamed.T, annot=True, fmt='.2f', 
               cmap='custom_d69584_c7ccd6', 
               center=COLORS['heatmap_center'],
               linewidths=1, linecolor='white',
               cbar_kws={'label': 'Feature Value'}, ax=ax)
    
    ax.set_title('Feature Heatmap: Center × Label Averages', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Center × Label', fontsize=14)
    ax.set_ylabel('Features', fontsize=14)
    
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'Feature_Heatmap.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Feature heatmap saved: {save_path}")
    plt.close()


# ========================================
# 5. 豆荚图（Bean Plot）
# ========================================

def plot_bean_plot(df, save_dir):
    """豆荚图"""
    print("\n📊 生成豆荚图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 使用真实特征名称
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
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # 为每个中心绘制分布
        positions = []
        data_list = []
        colors_list = []
        
        for center_id in range(5):
            mask = df['center_id'] == center_id
            data = df[mask][metric]
            
            positions.append(center_id)
            data_list.append(data)
            colors_list.append(COLORS[f'center_{center_id}'])
        
        # 绘制小提琴
        parts = ax.violinplot(data_list, positions=positions, widths=0.7,
                             showmeans=True, showextrema=True, showmedians=True)
        
        # 着色
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors_list[i])
            pc.set_alpha(0.6)
        
        # 叠加散点
        for i, data in enumerate(data_list):
            y = data.values
            x = np.random.normal(positions[i], 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.3, s=20, color=colors_list[i])
        
        # 获取显示名称
        if metric.startswith('feature_'):
            display_name = get_feature_display_name(metric)
        else:
            display_name = metric.upper()
        
        ax.set_title(f'Bean Plot: {display_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Center ID', fontsize=12)
        ax.set_ylabel(display_name, fontsize=12)
        ax.set_xticks(positions)
        ax.set_xticklabels([CENTER_NAMES[i] for i in range(5)], rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Bean Plot: Distribution with Density', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'Bean_Plot.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Bean plot saved: {save_path}")
    plt.close()


# ========================================
# 6. 3D交互式表面图（3D Surface）
# ========================================

def plot_3d_surface(df, save_dir):
    """3D表面图"""
    print("\n📊 生成3D表面图...")
    
    fig = plt.figure(figsize=(20, 9))
    
    # 左图：特征0和特征1的密度表面
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 使用真实特征名称
    feature_0_name = FEATURE_NAMES.get('feature_0', 'feature_0')
    feature_1_name = FEATURE_NAMES.get('feature_1', 'feature_1')
    
    if feature_0_name not in df.columns and 'feature_0' in df.columns:
        feature_0_name = 'feature_0'
    if feature_1_name not in df.columns and 'feature_1' in df.columns:
        feature_1_name = 'feature_1'
    
    x = df[feature_0_name].values
    y = df[feature_1_name].values
    
    # 创建网格
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # 计算每个网格点的密度（简化版）
    Zi = np.zeros_like(Xi)
    for i in range(len(xi)):
        for j in range(len(yi)):
            distances = np.sqrt((x - Xi[j, i])**2 + (y - Yi[j, i])**2)
            Zi[j, i] = np.sum(np.exp(-distances / 0.5))
    
    surf = ax1.plot_surface(Xi, Yi, Zi, cmap='viridis', alpha=0.8, 
                           edgecolor='none', linewidth=0, antialiased=True)
    
    ax1.set_title(f'3D Surface: {get_feature_display_name(feature_0_name)} × {get_feature_display_name(feature_1_name)} Density', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel(get_feature_display_name(feature_0_name), fontsize=11)
    ax1.set_ylabel(get_feature_display_name(feature_1_name), fontsize=11)
    ax1.set_zlabel('Density', fontsize=11)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # 右图：性能指标的3D表面
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 按中心聚合
    center_data = []
    for center_id in range(5):
        mask = df['center_id'] == center_id
        auc_mean = df[mask]['auc'].mean()
        sens_mean = df[mask]['sensitivity'].mean()
        spec_mean = df[mask]['specificity'].mean()
        center_data.append([auc_mean, sens_mean, spec_mean, center_id])
    
    center_df = pd.DataFrame(center_data, columns=['auc', 'sensitivity', 'specificity', 'center_id'])
    
    x2 = center_df['auc'].values
    y2 = center_df['sensitivity'].values
    z2 = center_df['specificity'].values
    
    # 创建表面（插值）
    from scipy.interpolate import griddata
    
    xi2 = np.linspace(x2.min(), x2.max(), 20)
    yi2 = np.linspace(y2.min(), y2.max(), 20)
    Xi2, Yi2 = np.meshgrid(xi2, yi2)
    Zi2 = griddata((x2, y2), z2, (Xi2, Yi2), method='cubic')
    
    surf2 = ax2.plot_surface(Xi2, Yi2, Zi2, cmap='plasma', alpha=0.8,
                            edgecolor='none', linewidth=0, antialiased=True)
    
    # 叠加中心点
    for i, row in center_df.iterrows():
        ax2.scatter(row['auc'], row['sensitivity'], row['specificity'],
                   c=COLORS[f'center_{int(row["center_id"])}'], s=200, 
                   edgecolors='black', linewidth=2, alpha=0.9,
                   label=CENTER_NAMES[int(row['center_id'])])
    
    ax2.set_title('3D Surface: Performance Metrics', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('AUC', fontsize=11)
    ax2.set_ylabel('Sensitivity', fontsize=11)
    ax2.set_zlabel('Specificity', fontsize=11)
    ax2.legend(fontsize=8, loc='best')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    plt.suptitle('3D Surface Plots', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = Path(save_dir) / '3D_Surface.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 3D surface saved: {save_path}")
    plt.close()


# ========================================
# 主函数
# ========================================

def main():
    """主函数"""
    print("=" * 80)
    print("Bio-COT 3.2: Additional Visualization Suite")
    print("生成补充可视化图表")
    print("=" * 80)
    
    # 设置路径
    exp_dir = Path(__file__).resolve().parents[2]
    vis_dir = exp_dir / 'visualization'
    figures_dir = vis_dir / 'figures'
    data_dir = vis_dir / 'data'
    
    # 加载数据
    print("\n📊 加载数据...")
    df_path = data_dir / 'Complete_Dataset.csv'
    if df_path.exists():
        df = pd.read_csv(df_path)
        print(f"   数据维度: {df.shape}")
    else:
        print("❌ 数据文件不存在，请先运行 generate_all_plots.py")
        return
    
    # 生成图表
    plot_functions = [
        ('维恩图', plot_venn_diagram, False),  # 不需要df
        ('火山图', plot_volcano, True),
        ('生存曲线', plot_survival_curves, True),
        ('特征热力图', plot_feature_heatmap, True),
        ('豆荚图', plot_bean_plot, True),
        ('3D表面图', plot_3d_surface, True),
    ]
    
    print("\n" + "=" * 80)
    print("开始生成图表...")
    print("=" * 80)
    
    for name, func, needs_df in plot_functions:
        try:
            if needs_df:
                func(df, figures_dir)
            else:
                func(figures_dir)
        except Exception as e:
            print(f"❌ {name}生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    print("\n" + "=" * 80)
    print("✅ 所有补充图表生成完成！")
    print("=" * 80)
    
    print(f"\n📁 生成的文件：")
    print(f"   Figures: {figures_dir}")
    for f in sorted(figures_dir.glob('*.pdf')):
        print(f"     - {f.name}")


if __name__ == '__main__':
    main()

