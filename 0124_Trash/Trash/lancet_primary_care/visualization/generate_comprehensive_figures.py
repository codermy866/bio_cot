#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成The Lancet Primary Care研究的综合可视化图表
使用2025年最新技术，英文Arial字体，高质量输出
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# 2025年最新可视化设置
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 字体配置（优先使用Arial，不存在则回退到DejaVu Sans）
AVAILABLE_FONTS = {font.name for font in fm.fontManager.ttflist}
FONT_FAMILY = 'Arial' if 'Arial' in AVAILABLE_FONTS else 'DejaVu Sans'

# 设置字体
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['font.sans-serif'] = [FONT_FAMILY, 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# 2025年最新seaborn样式
sns.set_style("whitegrid", {
    'font.family': FONT_FAMILY,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.color': '#E5E5E5',
    'grid.linewidth': 0.5
})

# 2025年最新配色方案
COLORS = {
    'primary': '#2C3E50',
    'secondary': '#3498DB',
    'success': '#27AE60',
    'warning': '#F39C12',
    'danger': '#E74C3C',
    'info': '#1ABC9C',
    'light': '#ECF0F1',
    'dark': '#34495E'
}

PALETTE = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def extract_method_metrics(results: Dict) -> List[Dict]:
    """
    将各实验的关键指标汇总为统一的长表结构
    返回字段:
        method, experiment, biopsy_rate, sensitivity, specificity, ppv, npv, auc, cost
    """
    records = []

    if 'experiment1' in results:
        exp1 = results['experiment1']
        for label, metrics in [('Traditional HPV+TCT', exp1['traditional']),
                               ('Multimodal AI (Exp1)', exp1['multimodal_ai'])]:
            records.append({
                'method': label,
                'experiment': 'Experiment1',
                'biopsy_rate': metrics.get('biopsy_rate'),
                'sensitivity': metrics.get('sensitivity'),
                'specificity': metrics.get('specificity'),
                'ppv': metrics.get('ppv'),
                'npv': metrics.get('npv'),
                'auc': metrics.get('auc')
            })

    if 'experiment2' in results:
        exp2 = results['experiment2']
        for label, metrics in [('Traditional OCT', exp2['traditional_oct']),
                               ('AI-Enhanced OCT', exp2['ai_enhanced_oct'])]:
            records.append({
                'method': label,
                'experiment': 'Experiment2',
                'biopsy_rate': None,
                'sensitivity': metrics.get('sensitivity'),
                'specificity': metrics.get('specificity'),
                'ppv': metrics.get('ppv'),
                'npv': metrics.get('npv'),
                'auc': metrics.get('auc')
            })

    if 'experiment3' in results:
        exp3 = results['experiment3']
        for label, metrics in [('TCT Only', exp3['tct_only']),
                               ('HPV+TCT', exp3['hpv_tct_combined']),
                               ('Multimodal AI (Exp3)', exp3['multimodal_ai'])]:
            records.append({
                'method': label,
                'experiment': 'Experiment3',
                'biopsy_rate': metrics.get('biopsy_rate'),
                'sensitivity': metrics.get('sensitivity'),
                'specificity': metrics.get('specificity'),
                'ppv': metrics.get('ppv'),
                'npv': metrics.get('npv'),
                'auc': metrics.get('auc')
            })

    if 'experiment4' in results:
        exp4 = results['experiment4']
        for label, metrics in [('Standard Workflow', exp4['standard_workflow']['metrics']),
                               ('Optimized Workflow', exp4['optimized_workflow']['metrics'])]:
            records.append({
                'method': label,
                'experiment': 'Experiment4',
                'biopsy_rate': metrics.get('biopsy_rate'),
                'sensitivity': metrics.get('sensitivity'),
                'specificity': metrics.get('specificity'),
                'ppv': metrics.get('ppv'),
                'npv': metrics.get('npv'),
                'auc': metrics.get('auc'),
                'cost_per_case': metrics.get('cost_per_detected_case')
            })
    return records


def normalize_metric(metric: str, value: float) -> float:
    if value is None:
        return None
    if metric == 'biopsy_rate':
        return value / 100.0
    return value


def prepare_long_dataframe(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        for metric in ['sensitivity', 'specificity', 'ppv', 'npv', 'auc', 'biopsy_rate']:
            value = rec.get(metric)
            if value is not None:
                rows.append({
                    'method': rec['method'],
                    'experiment': rec['experiment'],
                    'metric': metric,
                    'value': normalize_metric(metric, value)
                })
    return pd.DataFrame(rows)


def load_experiment_results(results_dir: str) -> Dict:
    """加载所有实验的结果"""
    results = {}
    
    experiments = ['experiment1', 'experiment2', 'experiment3', 'experiment4']
    for exp in experiments:
        result_file = os.path.join(results_dir, f'{exp}_results', f'{exp}_results.json')
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                results[exp] = json.load(f)
        else:
            print(f"⚠️  {exp} results not found: {result_file}")
    
    return results


def create_summary_figure(results: Dict, output_path: str):
    """
    创建综合总结图表
    展示所有4个实验的关键结果
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    fig.suptitle('The Lancet Primary Care: Comprehensive Experimental Results', 
                 fontsize=18, fontweight='bold', fontfamily=FONT_FAMILY, y=0.98)
    
    # 实验1: 活检率降低
    if 'experiment1' in results:
        ax1 = fig.add_subplot(gs[0, 0])
        exp1 = results['experiment1']
        methods = ['Traditional\nHPV+TCT', 'Multimodal AI']
        biopsy_rates = [
            exp1['traditional']['biopsy_rate'],
            exp1['multimodal_ai']['biopsy_rate']
        ]
        bars = ax1.bar(methods, biopsy_rates, color=['#FF6B6B', '#4ECDC4'], 
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Biopsy Rate (%)', fontsize=11, fontfamily=FONT_FAMILY, fontweight='bold')
        ax1.set_title('Exp 1: Biopsy Rate Reduction', fontsize=12, fontweight='bold', fontfamily=FONT_FAMILY)
        reduction = exp1['improvements']['biopsy_rate_reduction_percent']
        ax1.text(0.5, max(biopsy_rates) * 1.15, f'Reduction: {reduction:.1f}%',
                ha='center', fontsize=10, fontweight='bold', color='green', fontfamily=FONT_FAMILY,
                transform=ax1.transAxes)
        for bar, rate in zip(bars, biopsy_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{rate:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', fontfamily=FONT_FAMILY)
    
    # 实验2: OCT灵敏度
    if 'experiment2' in results:
        ax2 = fig.add_subplot(gs[0, 1])
        exp2 = results['experiment2']
        methods = ['Traditional\nOCT', 'AI-Enhanced\nOCT']
        sensitivities = [
            exp2['traditional_oct']['sensitivity'],
            exp2['ai_enhanced_oct']['sensitivity']
        ]
        bars = ax2.bar(methods, sensitivities, color=['#FF6B6B', '#4ECDC4'],
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Sensitivity', fontsize=11, fontfamily=FONT_FAMILY, fontweight='bold')
        ax2.set_title('Exp 2: OCT Sensitivity', fontsize=12, fontweight='bold', fontfamily=FONT_FAMILY)
        ax2.set_ylim(0, 1.1)
        improvement = exp2['improvements']['sensitivity_improvement_percent']
        ax2.text(0.5, 1.15, f'Improvement: {improvement:.1f}%',
                ha='center', fontsize=10, fontweight='bold', color='green', fontfamily=FONT_FAMILY,
                transform=ax2.transAxes)
        for bar, sens in zip(bars, sensitivities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{sens:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', fontfamily=FONT_FAMILY)
    
    # 实验3: AUC对比
    if 'experiment3' in results:
        ax3 = fig.add_subplot(gs[0, 2])
        exp3 = results['experiment3']
        methods = ['TCT Only', 'HPV+TCT', 'Multimodal AI']
        aucs = [
            exp3['tct_only']['auc'],
            exp3['hpv_tct_combined']['auc'],
            exp3['multimodal_ai']['auc']
        ]
        bars = ax3.bar(methods, aucs, color=['#FF6B6B', '#FFA07A', '#4ECDC4'],
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('AUC', fontsize=11, fontfamily=FONT_FAMILY, fontweight='bold')
        ax3.set_title('Exp 3: AUC Comparison', fontsize=12, fontweight='bold', fontfamily=FONT_FAMILY)
        ax3.set_ylim(0, 1.1)
        for bar, auc_val in zip(bars, aucs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height, f'{auc_val:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', fontfamily=FONT_FAMILY)
    
    # 实验4: 成本降低
    if 'experiment4' in results:
        ax4 = fig.add_subplot(gs[0, 3])
        exp4 = results['experiment4']
        methods = ['Standard', 'Optimized']
        cost_reduction = exp4['cost_effectiveness_analysis']['cost_reduction_percent']
        costs = [100, 100 - cost_reduction]  # 相对成本
        bars = ax4.bar(methods, costs, color=['#FF6B6B', '#4ECDC4'],
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Relative Cost (%)', fontsize=11, fontfamily=FONT_FAMILY, fontweight='bold')
        ax4.set_title('Exp 4: Cost Reduction', fontsize=12, fontweight='bold', fontfamily=FONT_FAMILY)
        ax4.text(0.5, max(costs) * 1.15, f'Reduction: {cost_reduction:.1f}%',
                ha='center', fontsize=10, fontweight='bold', color='green', fontfamily=FONT_FAMILY,
                transform=ax4.transAxes)
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height, f'{cost:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', fontfamily=FONT_FAMILY)
    
    # 综合性能雷达图
    ax5 = fig.add_subplot(gs[1, :2], projection='polar')
    if all(exp in results for exp in ['experiment1', 'experiment2', 'experiment3', 'experiment4']):
        categories = ['Biopsy\nReduction', 'OCT\nSensitivity', 'AUC\nPerformance', 'Cost\nEfficiency']
        
        # 归一化指标到0-1范围
        biopsy_red = min(results['experiment1']['improvements']['biopsy_rate_reduction_percent'] / 50, 1.0)
        oct_sens = results['experiment2']['ai_enhanced_oct']['sensitivity']
        auc_perf = results['experiment3']['multimodal_ai']['auc']
        cost_eff = min(results['experiment4']['cost_effectiveness_analysis']['cost_reduction_percent'] / 30, 1.0)
        
        values = [biopsy_red, oct_sens, auc_perf, cost_eff]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 闭合
        angles += angles[:1]
        
        ax5.plot(angles, values, 'o-', linewidth=2.5, color='#4ECDC4', markersize=8)
        ax5.fill(angles, values, alpha=0.25, color='#4ECDC4')
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(categories, fontsize=10, fontfamily=FONT_FAMILY, fontweight='bold')
        ax5.set_ylim(0, 1)
        ax5.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax5.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9, fontfamily=FONT_FAMILY)
        ax5.grid(True, linestyle='--', alpha=0.3)
        ax5.set_title('Comprehensive Performance Profile', fontsize=12, fontweight='bold', 
                     fontfamily=FONT_FAMILY, pad=20)
    
    # 关键指标对比表
    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.axis('off')
    
    if all(exp in results for exp in ['experiment1', 'experiment2', 'experiment3', 'experiment4']):
        table_data = []
        table_data.append(['Metric', 'Traditional', 'Multimodal AI', 'Improvement'])
        table_data.append(['Biopsy Rate Reduction', '0%', 
                          f"{results['experiment1']['improvements']['biopsy_rate_reduction_percent']:.1f}%",
                          f"{results['experiment1']['improvements']['biopsy_rate_reduction_percent']:.1f}%"])
        table_data.append(['OCT Sensitivity', 
                          f"{results['experiment2']['traditional_oct']['sensitivity']:.3f}",
                          f"{results['experiment2']['ai_enhanced_oct']['sensitivity']:.3f}",
                          f"+{results['experiment2']['improvements']['sensitivity_improvement_percent']:.1f}%"])
        table_data.append(['AUC', 
                          f"{results['experiment3']['hpv_tct_combined']['auc']:.3f}",
                          f"{results['experiment3']['multimodal_ai']['auc']:.3f}",
                          f"+{(results['experiment3']['multimodal_ai']['auc'] - results['experiment3']['hpv_tct_combined']['auc'])*100:.1f}%"])
        table_data.append(['Cost Reduction', '0%',
                          f"{results['experiment4']['cost_effectiveness_analysis']['cost_reduction_percent']:.1f}%",
                          f"{results['experiment4']['cost_effectiveness_analysis']['cost_reduction_percent']:.1f}%"])
        
        table = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                         cellLoc='center', loc='center',
                         colWidths=[0.3, 0.2, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表格样式
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#34495E')
            table[(0, i)].set_text_props(weight='bold', color='white', fontfamily=FONT_FAMILY)
        
        for i in range(1, len(table_data)):
            for j in range(len(table_data[0])):
                table[(i, j)].set_text_props(fontfamily=FONT_FAMILY)
                if j == len(table_data[0]) - 1:  # 最后一列（Improvement）
                    table[(i, j)].set_facecolor('#D5F4E6')
        
        ax6.set_title('Key Metrics Summary', fontsize=12, fontweight='bold', 
                     fontfamily=FONT_FAMILY, pad=20)
    
    # 临床影响评估
    ax7 = fig.add_subplot(gs[2, :])
    if all(exp in results for exp in ['experiment1', 'experiment2', 'experiment3', 'experiment4']):
        impact_categories = ['Reduced\nUnnecessary Biopsies', 'Improved\nEarly Detection', 
                            'Enhanced\nDiagnostic Accuracy', 'Optimized\nHealthcare Costs']
        impact_values = [
            results['experiment1']['improvements']['biopsy_rate_reduction_percent'] / 10,  # 归一化
            results['experiment2']['improvements']['sensitivity_improvement_percent'] / 5,
            (results['experiment3']['multimodal_ai']['auc'] - 0.5) * 2,  # 归一化
            results['experiment4']['cost_effectiveness_analysis']['cost_reduction_percent'] / 10
        ]
        
        bars = ax7.barh(impact_categories, impact_values, color=PALETTE[:4],
                        alpha=0.8, edgecolor='black', linewidth=1.5)
        ax7.set_xlabel('Impact Score (Normalized)', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax7.set_title('Clinical Impact Assessment', fontsize=13, fontweight='bold', fontfamily=FONT_FAMILY)
        ax7.set_xlim(0, 10)
        
        for i, (bar, val) in enumerate(zip(bars, impact_values)):
            width = bar.get_width()
            ax7.text(width, bar.get_y() + bar.get_height()/2, f'{width:.1f}',
                    ha='left', va='center', fontsize=10, fontweight='bold', fontfamily=FONT_FAMILY)
    
    # 设置所有文本为Arial
    for ax in fig.get_axes():
        for label in ax.get_xticklabels():
            label.set_fontfamily(FONT_FAMILY)
        for label in ax.get_yticklabels():
            label.set_fontfamily(FONT_FAMILY)
    
    plt.savefig(output_path, dpi=200, facecolor='white', edgecolor='none')
    print(f"✅ Comprehensive figure saved to: {output_path}")
    plt.close()


def create_bubble_chart(records: List[Dict], output_path: str):
    df = pd.DataFrame(records)
    df = df.dropna(subset=['biopsy_rate', 'sensitivity', 'auc'])
    if df.empty:
        print("⚠️  Bubble chart skipped: insufficient data.")
        return

    plt.figure(figsize=(10, 8))
    sizes = (df['auc'].fillna(0.5) * 1000).clip(50, 1200)
    scatter = plt.scatter(
        df['biopsy_rate'],
        df['sensitivity'],
        s=sizes,
        c=np.linspace(0, 1, len(df)),
        cmap='viridis',
        alpha=0.7,
        edgecolors='black'
    )
    for _, row in df.iterrows():
        plt.text(row['biopsy_rate'], row['sensitivity'], row['method'],
                 fontsize=9, fontfamily=FONT_FAMILY, ha='center', va='center')

    plt.xlabel('Biopsy Rate (%)', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
    plt.ylabel('Sensitivity', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
    plt.title('Biopsy vs Sensitivity Bubble Plot (Bubble size = AUC)', fontsize=14,
              fontfamily=FONT_FAMILY, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.colorbar(scatter, label='Method Index')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Bubble chart saved to: {output_path}")
    plt.close()


def create_violin_plot(long_df: pd.DataFrame, output_path: str):
    if long_df.empty:
        print("⚠️  Violin plot skipped: insufficient data.")
        return

    order = ['sensitivity', 'specificity', 'ppv', 'npv', 'auc', 'biopsy_rate']
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(
        data=long_df,
        x='metric',
        y='value',
        hue='experiment',
        order=[m for m in order if m in long_df['metric'].unique()],
        palette='Set2',
        cut=0,
        linewidth=1.2,
        inner='quartile'
    )
    sns.stripplot(
        data=long_df,
        x='metric',
        y='value',
        hue='experiment',
        dodge=True,
        order=[m for m in order if m in long_df['metric'].unique()],
        palette='dark:grey',
        alpha=0.4,
        legend=False,
        size=3
    )
    ax.set_ylabel('Normalized Metric Value', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
    ax.set_title('Distribution of Key Metrics per Experiment', fontsize=14,
                 fontfamily=FONT_FAMILY, fontweight='bold', pad=12)
    ax.set_ylim(0, 1.1)
    ax.legend(title='Experiment', fontsize=9, title_fontsize=10, frameon=True)
    plt.xticks(rotation=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Violin plot saved to: {output_path}")
    plt.close()


def create_metric_heatmap(records: List[Dict], output_path: str):
    df = pd.DataFrame(records)
    if df.empty:
        print("⚠️  Heatmap skipped: insufficient data.")
        return
    metrics = ['sensitivity', 'specificity', 'ppv', 'npv', 'auc']
    heatmap_df = df[['method'] + metrics].set_index('method')
    heatmap_df = heatmap_df.applymap(lambda x: x if x is not None else np.nan)
    plt.figure(figsize=(10, max(4, len(heatmap_df) * 0.4)))
    sns.heatmap(heatmap_df, annot=True, cmap='YlGnBu', fmt='.3f',
                cbar_kws={'label': 'Score'})
    plt.title('Method-wise Performance Heatmap', fontsize=14,
              fontfamily=FONT_FAMILY, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
    plt.ylabel('Methods', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
    print(f"✅ Heatmap saved to: {output_path}")
    plt.close()


def _style_axes(ax):
    for label in ax.get_xticklabels():
        label.set_fontfamily(FONT_FAMILY)
    for label in ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)
    ax.tick_params(labelsize=10)


def create_experiment_specific_figures(results: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    def save(fig, name):
        path = os.path.join(output_dir, name)
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"✅ Per-experiment figure saved to: {path}")

    if 'experiment1' in results:
        exp1 = results['experiment1']
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle('Experiment 1 · HPV+ Biopsy Reduction', fontfamily=FONT_FAMILY,
                     fontsize=15, fontweight='bold')

        ax = axes[0]
        methods = ['Traditional', 'Multimodal AI']
        rates = [exp1['traditional']['biopsy_rate'], exp1['multimodal_ai']['biopsy_rate']]
        ax.bar(methods, rates, color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
        ax.set_ylabel('Biopsy Rate (%)', fontfamily=FONT_FAMILY, fontweight='bold')
        ax.set_ylim(0, max(rates) * 1.2)
        ax.set_title('Biopsy Rate Comparison', fontfamily=FONT_FAMILY, fontweight='bold')
        for i, rate in enumerate(rates):
            ax.text(i, rate + 1, f'{rate:.1f}%', ha='center', fontfamily=FONT_FAMILY)
        _style_axes(ax)

        ax = axes[1]
        metrics = ['sensitivity', 'specificity', 'ppv', 'npv']
        x = np.arange(len(metrics))
        width = 0.35
        trad = [exp1['traditional'][m] for m in metrics]
        ai = [exp1['multimodal_ai'][m] for m in metrics]
        ax.bar(x - width/2, trad, width, label='Traditional', color='#FF6B6B')
        ax.bar(x + width/2, ai, width, label='Multimodal AI', color='#4ECDC4')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics], rotation=20)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score', fontfamily=FONT_FAMILY, fontweight='bold')
        ax.set_title('Diagnostic Metrics', fontfamily=FONT_FAMILY, fontweight='bold')
        ax.legend(frameon=True)
        _style_axes(ax)

        save(fig, 'experiment1_summary.png')

    if 'experiment2' in results:
        exp2 = results['experiment2']
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle('Experiment 2 · OCT Sensitivity Improvement', fontfamily=FONT_FAMILY,
                     fontsize=15, fontweight='bold')

        metrics = ['sensitivity', 'specificity', 'auc']
        df_plot = pd.DataFrame({
            'Metric': np.tile([m.upper() for m in metrics], 2),
            'Value': [exp2['traditional_oct'][m] for m in metrics] +
                     [exp2['ai_enhanced_oct'][m] for m in metrics],
            'Method': ['Traditional OCT'] * len(metrics) + ['AI-Enhanced OCT'] * len(metrics)
        })
        sns.barplot(data=df_plot, x='Metric', y='Value', hue='Method',
                    palette=['#FFA07A', '#2ECC71'], ax=axes[0])
        axes[0].set_ylim(0, 1.1)
        axes[0].set_ylabel('Score', fontfamily=FONT_FAMILY, fontweight='bold')
        axes[0].set_title('Key Metrics', fontfamily=FONT_FAMILY, fontweight='bold')
        _style_axes(axes[0])

        axes[1].bar(['ΔSensitivity', 'ΔSpecificity', 'ΔAUC'],
                    [exp2['improvements']['sensitivity_improvement_percent'] / 100,
                     exp2['ai_enhanced_oct']['specificity'] - exp2['traditional_oct']['specificity'],
                     exp2['ai_enhanced_oct']['auc'] - exp2['traditional_oct']['auc']],
                    color='#45B7D1')
        axes[1].set_ylim(-0.1, 0.4)
        axes[1].set_ylabel('Absolute Change', fontfamily=FONT_FAMILY, fontweight='bold')
        axes[1].set_title('Improvements', fontfamily=FONT_FAMILY, fontweight='bold')
        _style_axes(axes[1])

        save(fig, 'experiment2_summary.png')

    if 'experiment3' in results:
        exp3 = results['experiment3']
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle('Experiment 3 · Screening Strategy Comparison', fontfamily=FONT_FAMILY,
                     fontsize=15, fontweight='bold')

        methods = ['TCT Only', 'HPV+TCT', 'Multimodal AI']
        aucs = [exp3['tct_only']['auc'], exp3['hpv_tct_combined']['auc'], exp3['multimodal_ai']['auc']]
        axes[0].bar(methods, aucs, color=['#F4B183', '#E67E22', '#4ECDC4'], edgecolor='black')
        axes[0].set_ylim(0, 1.1)
        axes[0].set_ylabel('AUC', fontfamily=FONT_FAMILY, fontweight='bold')
        axes[0].set_title('AUC Comparison', fontfamily=FONT_FAMILY, fontweight='bold')
        for i, value in enumerate(aucs):
            axes[0].text(i, value + 0.03, f'{value:.2f}', ha='center', fontfamily=FONT_FAMILY)
        _style_axes(axes[0])

        metrics = ['biopsy_rate', 'sensitivity', 'specificity']
        df_plot = []
        for method_key, label in [('tct_only', 'TCT Only'),
                                  ('hpv_tct_combined', 'HPV+TCT'),
                                  ('multimodal_ai', 'Multimodal AI')]:
            for m in metrics:
                df_plot.append({
                    'Method': label,
                    'Metric': m.replace('_', ' ').title(),
                    'Value': exp3[method_key][m]
                })
        df_plot = pd.DataFrame(df_plot)
        sns.barplot(data=df_plot, x='Metric', y='Value', hue='Method', palette='Set3', ax=axes[1])
        axes[1].set_ylim(0, 1.1)
        axes[1].set_ylabel('Score', fontfamily=FONT_FAMILY, fontweight='bold')
        axes[1].set_title('Biopsy & Sensitivity Metrics', fontfamily=FONT_FAMILY, fontweight='bold')
        axes[1].legend(frameon=True)
        _style_axes(axes[1])

        save(fig, 'experiment3_summary.png')

    if 'experiment4' in results:
        exp4 = results['experiment4']
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle('Experiment 4 · Workflow Optimization', fontfamily=FONT_FAMILY,
                     fontsize=15, fontweight='bold')

        metrics = exp4['standard_workflow']['metrics']
        metrics_opt = exp4['optimized_workflow']['metrics']

        axes[0].bar(['Standard', 'Optimized'],
                    [metrics['total_cost'], metrics_opt['total_cost']],
                    color=['#95A5A6', '#2ECC71'], edgecolor='black')
        axes[0].set_ylabel('Total Cost (CNY)', fontfamily=FONT_FAMILY, fontweight='bold')
        axes[0].set_title('Total Cost', fontfamily=FONT_FAMILY, fontweight='bold')
        _style_axes(axes[0])

        axes[1].bar(['Standard', 'Optimized'],
                    [metrics['cost_per_detected_case'], metrics_opt['cost_per_detected_case']],
                    color=['#95A5A6', '#2ECC71'], edgecolor='black')
        axes[1].set_ylabel('Cost per Detected Case (CNY)', fontfamily=FONT_FAMILY, fontweight='bold')
        axes[1].set_title('Cost Efficiency', fontfamily=FONT_FAMILY, fontweight='bold')
        _style_axes(axes[1])

        save(fig, 'experiment4_summary.png')
def main():
    results_dir = 'lancet_primary_care/results_final'
    output_dir = 'lancet_primary_care/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("Generating Comprehensive Visualization Figures")
    print("Using 2025 Latest Technologies with Arial Font")
    print("="*70)
    
    # 加载结果
    results = load_experiment_results(results_dir)
    
    if len(results) == 0:
        print("⚠️  No experiment results found. Please run experiments first.")
        return
    
    print(f"\nLoaded results from {len(results)} experiments")
    
    # 生成综合图表
    summary_fig_path = os.path.join(output_dir, 'comprehensive_results_summary.png')
    if os.path.exists(summary_fig_path):
        print(f"ℹ️  Summary figure already exists, reusing: {summary_fig_path}")
    else:
        try:
            create_summary_figure(results, summary_fig_path)
        except ValueError as exc:
            print(f"⚠️  Summary figure skipped due to rendering limit: {exc}")

    records = extract_method_metrics(results)
    long_df = prepare_long_dataframe(records)

    bubble_path = os.path.join(output_dir, 'bubble_performance.png')
    violin_path = os.path.join(output_dir, 'violin_metric_distribution.png')
    heatmap_path = os.path.join(output_dir, 'metric_heatmap.png')

    create_bubble_chart(records, bubble_path)
    create_violin_plot(long_df, violin_path)
    create_metric_heatmap(records, heatmap_path)
    per_exp_dir = os.path.join(output_dir, 'per_experiment')
    create_experiment_specific_figures(results, per_exp_dir)
    
    print("\n" + "="*70)
    print("✅ All figures generated successfully!")
    print(f"Output directory: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()

