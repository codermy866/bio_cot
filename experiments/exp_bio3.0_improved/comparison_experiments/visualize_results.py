#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化对比实验结果
生成各种图表：性能对比、统计显著性、ROC曲线等
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import seaborn as sns
from scipy import stats
import argparse
from datetime import datetime

# 添加项目路径
ROOT = Path(__file__).resolve().parents[1]  # exp_bio3.0_improved目录
sys.path.insert(0, str(ROOT))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_experiment_results(results_dir):
    """加载所有实验结果"""
    results_dir = Path(results_dir)
    experiments = {}
    
    print(f"正在查找结果目录: {results_dir}")
    print(f"目录是否存在: {results_dir.exists()}")
    
    if not results_dir.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return experiments
    
    # 查找所有包含all_results.json的目录
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
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
            print(f"⚠️  跳过 {exp_dir.name}: 未找到all_results.json")
            continue
        
        print(f"✅ 找到结果文件: {results_file}")
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换为DataFrame
            df = pd.DataFrame(data)
            experiments[exp_dir.name] = df
            print(f"  加载了 {len(df)} 条记录")
        except Exception as e:
            print(f"❌ 加载 {results_file} 失败: {e}")
    
    print(f"\n总共加载了 {len(experiments)} 个实验结果")
    return experiments


def plot_performance_comparison(experiments, output_dir):
    """绘制性能对比图"""
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score']
    
    # 准备数据
    comparison_data = []
    for exp_name, df in experiments.items():
        for metric in metrics:
            if metric in df.columns:
                values = df[metric].values
                comparison_data.append({
                    'Experiment': exp_name,
                    'Metric': metric.upper(),
                    'Mean': np.mean(values),
                    'Std': np.std(values),
                    'Values': values
                })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        metric_data = comp_df[comp_df['Metric'] == metric.upper()]
        
        # 绘制条形图
        x_pos = np.arange(len(metric_data))
        bars = ax.bar(x_pos, metric_data['Mean'], yerr=metric_data['Std'],
                     capsize=5, alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for i, (bar, mean, std) in enumerate(zip(bars, metric_data['Mean'], metric_data['Std'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_data['Experiment'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'performance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 性能对比图已保存: {output_path}")


def plot_box_plots(experiments, output_dir):
    """绘制箱线图"""
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # 准备数据
        data_for_plot = []
        labels = []
        for exp_name, df in experiments.items():
            if metric in df.columns:
                data_for_plot.append(df[metric].values)
                labels.append(exp_name)
        
        if data_for_plot:
            bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
            
            # 美化箱线图
            colors = sns.color_palette("husl", len(data_for_plot))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_title(f'{metric.upper()} Distribution', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'box_plots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 箱线图已保存: {output_path}")


def plot_statistical_significance(experiments, output_dir):
    """绘制统计显著性分析"""
    if len(experiments) < 2:
        print("⚠️  实验数量不足，跳过统计显著性分析")
        return
    
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score']
    exp_names = list(experiments.keys())
    
    # 创建显著性矩阵
    significance_results = {}
    
    for metric in metrics:
        matrix = np.zeros((len(exp_names), len(exp_names)))
        p_values = {}
        
        for i, exp1 in enumerate(exp_names):
            for j, exp2 in enumerate(exp_names):
                if i >= j:
                    continue
                
                if metric in experiments[exp1].columns and metric in experiments[exp2].columns:
                    values1 = experiments[exp1][metric].values
                    values2 = experiments[exp2][metric].values
                    
                    # Wilcoxon符号秩检验
                    try:
                        stat, p_value = stats.wilcoxon(values1, values2, alternative='two-sided')
                        p_values[f"{exp1}_vs_{exp2}"] = p_value
                        
                        # 显著性标记
                        if p_value < 0.001:
                            matrix[i, j] = 3  # ***
                        elif p_value < 0.01:
                            matrix[i, j] = 2  # **
                        elif p_value < 0.05:
                            matrix[i, j] = 1  # *
                        else:
                            matrix[i, j] = 0  # ns
                    except:
                        p_values[f"{exp1}_vs_{exp2}"] = 1.0
                        matrix[i, j] = 0
        
        significance_results[metric] = {
            'matrix': matrix,
            'p_values': p_values
        }
    
    # 绘制热力图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        if metric in significance_results:
            matrix = significance_results[metric]['matrix']
            
            # 创建对称矩阵
            sym_matrix = matrix + matrix.T
            
            # 绘制热力图
            sns.heatmap(sym_matrix, annot=True, fmt='.0f', cmap='RdYlGn_r',
                       xticklabels=exp_names, yticklabels=exp_names,
                       ax=ax, cbar_kws={'label': 'Significance Level'})
            
            ax.set_title(f'{metric.upper()} Statistical Significance', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Experiment', fontsize=12)
            ax.set_ylabel('Experiment', fontsize=12)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'statistical_significance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存p值
    p_values_file = Path(output_dir) / 'p_values.json'
    p_values_clean = {}
    for metric, result in significance_results.items():
        p_values_clean[metric] = result['p_values']
    
    with open(p_values_file, 'w', encoding='utf-8') as f:
        json.dump(p_values_clean, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 统计显著性分析已保存: {output_path}")
    print(f"✅ P值已保存: {p_values_file}")


def plot_summary_table(experiments, output_dir):
    """生成汇总表格图"""
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score']
    
    # 计算统计信息
    summary_data = []
    for exp_name, df in experiments.items():
        row = {'Experiment': exp_name}
        for metric in metrics:
            if metric in df.columns:
                values = df[metric].values
                mean = np.mean(values)
                std = np.std(values)
                row[metric] = f"{mean:.4f}±{std:.4f}"
            else:
                row[metric] = "N/A"
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # 创建表格图
    fig, ax = plt.subplots(figsize=(14, len(summary_df) * 0.8 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置行样式
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Experimental Results Summary', fontsize=16, fontweight='bold', pad=20)
    
    output_path = Path(output_dir) / 'summary_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存CSV
    csv_path = Path(output_dir) / 'summary_table.csv'
    summary_df.to_csv(csv_path, index=False)
    
    print(f"✅ 汇总表格已保存: {output_path}")
    print(f"✅ CSV已保存: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='可视化对比实验结果')
    parser.add_argument('--results_dir', type=str,
                       default='comparison_experiments/results',
                       help='结果目录')
    parser.add_argument('--output_dir', type=str,
                       default='comparison_experiments/visualizations',
                       help='可视化输出目录')
    
    args = parser.parse_args()
    
    # 处理相对路径和绝对路径
    if Path(args.results_dir).is_absolute():
        results_dir = Path(args.results_dir)
    else:
        results_dir = ROOT / args.results_dir
    
    if Path(args.output_dir).is_absolute():
        output_dir = Path(args.output_dir)
    else:
        output_dir = ROOT / args.output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"开始生成可视化结果")
    print(f"{'='*80}")
    print(f"结果目录: {results_dir}")
    print(f"输出目录: {output_dir}")
    print(f"结果目录是否存在: {results_dir.exists()}")
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
    print("📊 生成性能对比图...")
    plot_performance_comparison(experiments, output_dir)
    
    print("📊 生成箱线图...")
    plot_box_plots(experiments, output_dir)
    
    print("📊 生成统计显著性分析...")
    plot_statistical_significance(experiments, output_dir)
    
    print("📊 生成汇总表格...")
    plot_summary_table(experiments, output_dir)
    
    print(f"\n{'='*80}")
    print(f"✅ 所有可视化结果已生成!")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

