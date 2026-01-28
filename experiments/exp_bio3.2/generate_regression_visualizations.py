#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回归分析可视化图表生成
1. Linear regression with marginal distributions
2. Multiple linear regression
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
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde

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
print("📊 回归分析可视化图表生成")
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

# 2.1 Linear regression with marginal distributions
print("=" * 80)
print("🎨 图表 1: Linear regression with marginal distributions")
print("=" * 80)

try:
    # 智能选择特征对：选择相关性最高的特征对
    feature_cols_available = [col for col in feature_df.columns if col not in ['Label', 'Center', 'Label_Name']]
    
    if len(feature_cols_available) >= 2:
        # 计算所有特征对的相关性
        from scipy.stats import pearsonr
        max_corr = -1
        best_pair = None
        
        for i, col1 in enumerate(feature_cols_available):
            for col2 in feature_cols_available[i+1:]:
                data1 = feature_df[col1].values
                data2 = feature_df[col2].values
                r, _ = pearsonr(data1, data2)
                if abs(r) > abs(max_corr):
                    max_corr = r
                    best_pair = (col1, col2)
        
        if best_pair:
            x_col, y_col = best_pair
            print(f"  📊 选择特征对: {x_col} vs {y_col} (r={max_corr:.3f})")
        else:
            # 如果找不到，使用前两个特征
            x_col = feature_cols_available[0]
            y_col = feature_cols_available[1]
        
        x_data = feature_df[x_col].values
        y_data = feature_df[y_col].values
        
        # 创建图形，使用GridSpec布局
        fig = plt.figure(figsize=(12, 10))
        fig.patch.set_facecolor(COLORS['background'])
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 主散点图区域（右下角，3x3）
        ax_main = fig.add_subplot(gs[1:4, 0:3])
        
        # 上边缘分布（顶部，1x3）
        ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
        
        # 右边缘分布（右侧，3x1）
        ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
        
        # 主散点图
        if 'Label_Name' in feature_df.columns:
            for label_name in ['Negative', 'Positive']:
                mask = feature_df['Label_Name'] == label_name
                color = COLORS['positive'] if label_name == 'Positive' else COLORS['negative']
                ax_main.scatter(feature_df.loc[mask, x_col], feature_df.loc[mask, y_col],
                              alpha=0.6, s=50, color=color, edgecolors='white', linewidths=0.5,
                              label=label_name)
        else:
            ax_main.scatter(x_data, y_data, alpha=0.6, s=50, 
                          color=COLORS['center_2'], edgecolors='white', linewidths=0.5)
        
        # 线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
        x_line = np.linspace(x_data.min(), x_data.max(), 100)
        y_line = slope * x_line + intercept
        
        ax_main.plot(x_line, y_line, color=COLORS['positive'], linewidth=3, 
                    linestyle='--', label=f'Linear Fit (R²={r_value**2:.3f})', zorder=10)
        
        # 置信区间
        n = len(x_data)
        t_critical = stats.t.ppf(0.975, n-2)
        se_fit = std_err * np.sqrt(1/n + (x_line - x_data.mean())**2 / np.sum((x_data - x_data.mean())**2))
        y_upper = y_line + t_critical * se_fit
        y_lower = y_line - t_critical * se_fit
        ax_main.fill_between(x_line, y_lower, y_upper, alpha=0.2, color=COLORS['positive'],
                            label='95% Confidence Interval')
        
        ax_main.set_xlabel(x_col, fontsize=11, fontweight='bold')
        ax_main.set_ylabel(y_col, fontsize=11, fontweight='bold')
        ax_main.legend(fontsize=9, loc='best')
        ax_main.grid(True, alpha=0.3)
        ax_main.set_facecolor(COLORS['background'])
        
        # 上边缘分布（X轴）
        ax_top.hist(x_data, bins=40, density=True, alpha=0.7, color=COLORS['center_2'],
                   edgecolor='white', linewidth=0.5)
        kde_x = gaussian_kde(x_data)
        x_kde_range = np.linspace(x_data.min(), x_data.max(), 200)
        ax_top.plot(x_kde_range, kde_x(x_kde_range), color=COLORS['positive'], linewidth=2)
        ax_top.set_ylabel('Density', fontsize=9)
        ax_top.set_title(f'Linear Regression: {x_col} vs {y_col}\n' +
                        f'R² = {r_value**2:.3f}, p = {p_value:.2e}', 
                        fontsize=12, fontweight='bold', pad=10)
        ax_top.set_facecolor(COLORS['background'])
        ax_top.grid(True, alpha=0.3)
        plt.setp(ax_top.get_xticklabels(), visible=False)
        
        # 右边缘分布（Y轴）
        ax_right.hist(y_data, bins=40, density=True, alpha=0.7, color=COLORS['center_2'],
                     edgecolor='white', linewidth=0.5, orientation='horizontal')
        kde_y = gaussian_kde(y_data)
        y_kde_range = np.linspace(y_data.min(), y_data.max(), 200)
        ax_right.plot(kde_y(y_kde_range), y_kde_range, color=COLORS['positive'], linewidth=2)
        ax_right.set_xlabel('Density', fontsize=9)
        ax_right.set_facecolor(COLORS['background'])
        ax_right.grid(True, alpha=0.3)
        plt.setp(ax_right.get_yticklabels(), visible=False)
        
        plt.savefig(FIGURES_DIR / 'Linear_Regression_Marginal.png', dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.savefig(FIGURES_DIR / 'Linear_Regression_Marginal.pdf', dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.close()
        print("✅ Linear regression with marginal distributions 已生成")
    else:
        print("⚠️  特征数量不足，跳过此图表")
except Exception as e:
    print(f"⚠️  Linear regression with marginal distributions 生成失败: {e}")
    import traceback
    traceback.print_exc()

print()

# 2.2 Multiple linear regression
print("=" * 80)
print("🎨 图表 2: Multiple linear regression")
print("=" * 80)

try:
    # 选择多个特征进行多元线性回归
    feature_cols_for_mlr = [col for col in feature_df.columns if col not in ['Label', 'Center']][:6]
    
    if len(feature_cols_for_mlr) >= 3:
        # 准备数据
        X = feature_df[feature_cols_for_mlr].values
        y = feature_df[feature_cols_for_mlr[-1]].values  # 使用最后一个特征作为因变量
        
        # 多元线性回归
        reg = LinearRegression()
        reg.fit(X[:, :-1], y)  # 使用前n-1个特征预测最后一个特征
        
        y_pred = reg.predict(X[:, :-1])
        r2 = r2_score(y, y_pred)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor(COLORS['background'])
        axes = axes.flatten()
        
        # 1. 预测值 vs 真实值
        ax = axes[0]
        ax.scatter(y, y_pred, alpha=0.6, s=50, color=COLORS['center_2'],
                  edgecolors='white', linewidths=0.5)
        
        # 理想线（y=x）
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               color=COLORS['positive'], linewidth=2.5, linestyle='--',
               label='Perfect Prediction')
        
        ax.set_xlabel('True Values', fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
        ax.set_title(f'Predicted vs True Values\nR² = {r2:.3f}', 
                    fontsize=12, fontweight='bold', pad=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor(COLORS['background'])
        
        # 2. 残差图
        ax = axes[1]
        residuals = y - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6, s=50, color=COLORS['center_2'],
                  edgecolors='white', linewidths=0.5)
        ax.axhline(y=0, color=COLORS['positive'], linewidth=2.5, linestyle='--')
        ax.set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
        ax.set_ylabel('Residuals', fontsize=11, fontweight='bold')
        ax.set_title('Residual Plot', fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor(COLORS['background'])
        
        # 3. 系数条形图
        ax = axes[2]
        coef_names = feature_cols_for_mlr[:-1]
        coef_values = reg.coef_
        colors_coef = [COLORS['positive'] if c > 0 else COLORS['negative'] for c in coef_values]
        
        bars = ax.barh(range(len(coef_names)), coef_values, color=colors_coef, alpha=0.7,
                      edgecolor='white', linewidth=1)
        ax.set_yticks(range(len(coef_names)))
        ax.set_yticklabels(coef_names, fontsize=9)
        ax.set_xlabel('Coefficient Value', fontsize=11, fontweight='bold')
        ax.set_title('Regression Coefficients', fontsize=12, fontweight='bold', pad=10)
        ax.axvline(x=0, color='black', linewidth=1, linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_facecolor(COLORS['background'])
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, coef_values)):
            ax.text(val + (0.01 if val > 0 else -0.01), i, f'{val:.3f}',
                   va='center', ha='left' if val > 0 else 'right', fontsize=8, fontweight='bold')
        
        # 4. 特征重要性（基于系数的绝对值）
        ax = axes[3]
        importance = np.abs(coef_values)
        sorted_idx = np.argsort(importance)[::-1]
        
        bars = ax.barh(range(len(coef_names)), importance[sorted_idx], 
                      color=COLORS['center_2'], alpha=0.7,
                      edgecolor='white', linewidth=1)
        ax.set_yticks(range(len(coef_names)))
        ax.set_yticklabels([coef_names[i] for i in sorted_idx], fontsize=9)
        ax.set_xlabel('Absolute Coefficient Value', fontsize=11, fontweight='bold')
        ax.set_title('Feature Importance', fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_facecolor(COLORS['background'])
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, importance[sorted_idx])):
            ax.text(val + 0.01, i, f'{val:.3f}',
                   va='center', ha='left', fontsize=8, fontweight='bold')
        
        plt.suptitle(f'Multiple Linear Regression Analysis\n' +
                    f'Target: {feature_cols_for_mlr[-1]}, R² = {r2:.3f}',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'Multiple_Linear_Regression.png', dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.savefig(FIGURES_DIR / 'Multiple_Linear_Regression.pdf', dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.close()
        print("✅ Multiple linear regression 已生成")
    else:
        print("⚠️  特征数量不足，跳过此图表")
except Exception as e:
    print(f"⚠️  Multiple linear regression 生成失败: {e}")
    import traceback
    traceback.print_exc()

print()

# ========================================
# 完成
# ========================================
print("=" * 80)
print("✅ 回归分析可视化图表生成完成！")
print("=" * 80)
print(f"📁 输出目录: {FIGURES_DIR}")
print()
print("生成的文件:")
for fig_file in sorted(FIGURES_DIR.glob('*.png')):
    if any(keyword in fig_file.name for keyword in ['Linear_Regression_Marginal', 'Multiple_Linear_Regression']):
        size = fig_file.stat().st_size / 1024
        print(f"  ✅ {fig_file.name} ({size:.1f} KB)")
print("=" * 80)

