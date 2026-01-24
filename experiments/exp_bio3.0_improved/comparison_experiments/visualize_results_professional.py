#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专业级可视化结果生成 - 高质量图表，适合论文发表
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy import stats
import argparse
from datetime import datetime

# 添加项目路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 设置专业级样式
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("Set2")
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'patch.linewidth': 1.2,
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


def load_experiment_results(results_dir):
    """加载所有实验结果"""
    results_dir = Path(results_dir)
    experiments = {}
    
    if not results_dir.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return experiments
    
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith('.'):
            continue
        
        # 尝试多个可能的位置
        possible_paths = [
            exp_dir / 'results' / 'all_results.json',
            exp_dir / 'all_results.json',
        ]
        
        results_file = None
        for path in possible_paths:
            if path.exists():
                results_file = path
                break
        
        if results_file is None:
            continue
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            # 清理实验名称（移除_test后缀）
            clean_name = exp_dir.name.replace('_test', '').replace('baseline_', '')
            experiments[clean_name] = df
        except Exception as e:
            print(f"⚠️  加载 {results_file} 失败: {e}")
    
    return experiments


def plot_performance_comparison_professional(experiments, output_dir):
    """绘制专业级性能对比图"""
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score']
    metric_labels = {
        'auc': 'AUC',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'specificity': 'Specificity',
        'f1_score': 'F1-Score'
    }
    
    # 准备数据
    comparison_data = []
    for exp_name, df in experiments.items():
        for metric in metrics:
            if metric in df.columns:
                values = df[metric].values
                comparison_data.append({
                    'Experiment': exp_name,
                    'Metric': metric_labels[metric],
                    'Mean': np.mean(values),
                    'Std': np.std(values),
                    'Min': np.min(values),
                    'Max': np.max(values)
                })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # 创建图表 - 使用更专业的布局
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 定义颜色方案
    colors = sns.color_palette("Set2", len(experiments))
    exp_names = list(experiments.keys())
    color_map = {exp: colors[i] for i, exp in enumerate(exp_names)}
    
    for idx, metric in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        metric_label = metric_labels[metric]
        metric_data = comp_df[comp_df['Metric'] == metric_label]
        
        # 按均值排序
        metric_data = metric_data.sort_values('Mean', ascending=True)
        
        # 绘制条形图（水平）
        y_pos = np.arange(len(metric_data))
        bars = ax.barh(y_pos, metric_data['Mean'], 
                      xerr=metric_data['Std'],
                      color=[color_map.get(exp, 'gray') for exp in metric_data['Experiment']],
                      alpha=0.8,
                      edgecolor='black',
                      linewidth=1.2,
                      capsize=3,
                      error_kw={'elinewidth': 1.5, 'capthick': 1.5})
        
        # 添加数值标签
        for i, (bar, mean, std) in enumerate(zip(bars, metric_data['Mean'], metric_data['Std'])):
            width = bar.get_width()
            ax.text(width + std + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{mean:.3f}',
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metric_data['Experiment'], fontsize=11)
        ax.set_xlabel(metric_label, fontsize=13, fontweight='bold')
        ax.set_title(metric_label, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlim([0, 1.0])
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle('Performance Comparison Across All Metrics', 
                fontsize=18, fontweight='bold', y=0.995)
    
    output_path = Path(output_dir) / 'performance_comparison_professional.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 专业级性能对比图已保存: {output_path}")


def plot_radar_chart(experiments, output_dir):
    """绘制雷达图 - 多维度性能对比"""
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score']
    metric_labels = ['AUC', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
    
    # 计算每个实验的平均值
    exp_data = {}
    for exp_name, df in experiments.items():
        values = []
        for metric in metrics:
            if metric in df.columns:
                values.append(np.mean(df[metric].values))
            else:
                values.append(0.0)
        exp_data[exp_name] = values
    
    # 创建雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = sns.color_palette("Set2", len(exp_data))
    for idx, (exp_name, values) in enumerate(exp_data.items()):
        values += values[:1]  # 闭合
        ax.plot(angles, values, 'o-', linewidth=2, label=exp_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.5)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.set_title('Multi-Dimensional Performance Comparison', 
                fontsize=16, fontweight='bold', pad=20)
    
    output_path = Path(output_dir) / 'radar_chart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 雷达图已保存: {output_path}")


def plot_comprehensive_comparison(experiments, output_dir):
    """绘制综合对比图 - 主图"""
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score']
    metric_labels = {
        'auc': 'AUC',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'specificity': 'Specificity',
        'f1_score': 'F1-Score'
    }
    
    # 准备数据
    data_list = []
    for exp_name, df in experiments.items():
        for metric in metrics:
            if metric in df.columns:
                values = df[metric].values
                for val in values:
                    data_list.append({
                        'Experiment': exp_name,
                        'Metric': metric_labels[metric],
                        'Value': val
                    })
    
    plot_df = pd.DataFrame(data_list)
    
    # 创建综合对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = sns.color_palette("Set2", len(experiments))
    exp_names = sorted(experiments.keys())
    
    for idx, metric_label in enumerate([metric_labels[m] for m in metrics]):
        ax = axes[idx]
        metric_data = plot_df[plot_df['Metric'] == metric_label]
        
        # 绘制小提琴图 + 箱线图
        parts = ax.violinplot([metric_data[metric_data['Experiment'] == exp]['Value'].values 
                              for exp in exp_names],
                             positions=range(len(exp_names)),
                             showmeans=True, showmedians=True)
        
        # 美化小提琴图
        for pc in parts['bodies']:
            pc.set_facecolor(colors[0])
            pc.set_alpha(0.6)
        
        # 添加箱线图
        bp = ax.boxplot([metric_data[metric_data['Experiment'] == exp]['Value'].values 
                        for exp in exp_names],
                       positions=range(len(exp_names)),
                       widths=0.3,
                       patch_artist=True,
                       showfliers=False)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
        
        # 添加均值点
        for i, exp in enumerate(exp_names):
            exp_data = metric_data[metric_data['Experiment'] == exp]['Value'].values
            mean_val = np.mean(exp_data)
            ax.scatter(i, mean_val, color='red', s=100, zorder=3, 
                      marker='D', edgecolor='black', linewidth=1)
            ax.text(i, mean_val + 0.02, f'{mean_val:.3f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xticks(range(len(exp_names)))
        ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
        ax.set_title(metric_label, fontsize=13, fontweight='bold', pad=8)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle('Comprehensive Performance Comparison', 
                fontsize=18, fontweight='bold', y=0.995)
    
    output_path = Path(output_dir) / 'comprehensive_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 综合对比图已保存: {output_path}")


def plot_publication_table(experiments, output_dir):
    """生成发表级表格"""
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score']
    metric_labels = {
        'auc': 'AUC',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'specificity': 'Specificity',
        'f1_score': 'F1-Score'
    }
    
    # 准备数据
    table_data = []
    for exp_name, df in experiments.items():
        row = {'Method': exp_name}
        for metric in metrics:
            if metric in df.columns:
                values = df[metric].values
                mean = np.mean(values)
                std = np.std(values)
                row[metric_labels[metric]] = f"{mean:.4f} ± {std:.4f}"
            else:
                row[metric_labels[metric]] = "N/A"
        table_data.append(row)
    
    # 按AUC排序
    def get_auc(exp_name):
        if exp_name in experiments:
            df = experiments[exp_name]
            if 'auc' in df.columns:
                return np.mean(df['auc'].values)
        return 0.0
    
    table_data.sort(key=lambda x: get_auc(x['Method']), reverse=True)
    
    # 创建表格图
    fig, ax = plt.subplots(figsize=(14, max(6, len(table_data) * 0.6 + 2)))
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    columns = ['Method'] + [metric_labels[m] for m in metrics]
    cell_text = []
    for row in table_data:
        cell_text.append([row[col] for col in columns])
    
    table = ax.table(cellText=cell_text,
                    colLabels=columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    
    # 设置表头样式
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.08)
    
    # 设置行样式
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F2F2F2')
            else:
                table[(i, j)].set_facecolor('white')
            table[(i, j)].set_height(0.06)
    
    # 高亮最佳结果
    for metric in metrics:
        metric_label = metric_labels[metric]
        col_idx = columns.index(metric_label)
        best_val = 0.0
        best_row = 0
        for i, row in enumerate(table_data):
            val_str = row[metric_label]
            if val_str != "N/A":
                val = float(val_str.split(' ± ')[0])
                if val > best_val:
                    best_val = val
                    best_row = i + 1
        
        if best_row > 0:
            table[(best_row, col_idx)].set_facecolor('#FFD700')
            table[(best_row, col_idx)].set_text_props(weight='bold')
    
    plt.title('Performance Comparison Table', fontsize=16, fontweight='bold', pad=20)
    
    output_path = Path(output_dir) / 'publication_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 保存CSV
    df_table = pd.DataFrame(table_data)
    csv_path = Path(output_dir) / 'publication_table.csv'
    df_table.to_csv(csv_path, index=False)
    
    print(f"✅ 发表级表格已保存: {output_path}")
    print(f"✅ CSV已保存: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='生成专业级可视化结果')
    parser.add_argument('--results_dir', type=str,
                       default='comparison_experiments/results',
                       help='结果目录')
    parser.add_argument('--output_dir', type=str,
                       default='comparison_experiments/visualizations',
                       help='可视化输出目录')
    
    args = parser.parse_args()
    
    results_dir = ROOT / args.results_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"专业级可视化结果生成")
    print(f"{'='*80}")
    print(f"结果目录: {results_dir}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}\n")
    
    # 加载实验结果
    experiments = load_experiment_results(results_dir)
    
    if not experiments:
        print("❌ 未找到实验结果!")
        return
    
    print(f"找到 {len(experiments)} 个实验结果:")
    for exp_name in experiments.keys():
        print(f"  - {exp_name}")
    print()
    
    # 生成各种可视化
    print("📊 生成专业级性能对比图...")
    plot_performance_comparison_professional(experiments, output_dir)
    
    print("📊 生成雷达图...")
    plot_radar_chart(experiments, output_dir)
    
    print("📊 生成综合对比图...")
    plot_comprehensive_comparison(experiments, output_dir)
    
    print("📊 生成发表级表格...")
    plot_publication_table(experiments, output_dir)
    
    print(f"\n{'='*80}")
    print(f"✅ 所有专业级可视化结果已生成!")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

