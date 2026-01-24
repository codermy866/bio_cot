#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
结果分析脚本 - 汇总所有实验结果，进行统计检验，生成论文表格
"""

import sys
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

# 添加项目路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.statistics import compute_statistics, test_significance, compare_multiple_groups, format_statistics_string


def load_experiment_results(results_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    """
    加载所有实验结果
    
    Args:
        results_dir: 结果目录
    
    Returns:
        results_dict: 方法名到指标字典的映射
    """
    results_dict = {}
    
    # 遍历所有实验目录
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        results_file = exp_dir / "results" / "all_results.json"
        
        if not results_file.exists():
            print(f"⚠️  警告: {exp_name} 的结果文件不存在: {results_file}")
            continue
        
        # 加载结果
        with open(results_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        
        # 提取指标
        metrics = ['auc', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score']
        results_dict[exp_name] = {}
        
        for metric in metrics:
            values = [r[metric] for r in all_results if metric in r]
            if values:
                results_dict[exp_name][metric] = values
    
    return results_dict


def create_baseline_comparison_table(
    results_dict: Dict[str, Dict[str, List[float]]],
    metric: str = 'auc',
    reference_method: str = 'full_model'
) -> pd.DataFrame:
    """
    创建Baseline对比表格
    
    Args:
        results_dict: 结果字典
        metric: 要对比的指标
        reference_method: 参考方法
    
    Returns:
        comparison_table: 对比表格
    """
    # 提取所有方法的该指标
    groups = {}
    for method_name, metrics_dict in results_dict.items():
        if metric in metrics_dict:
            groups[method_name] = metrics_dict[metric]
    
    if not groups:
        return pd.DataFrame()
    
    # 如果没有参考方法，使用均值最高的方法
    if reference_method not in groups:
        reference_method = max(groups.keys(), key=lambda k: np.mean(groups[k]))
        print(f"⚠️  参考方法 '{reference_method}' 不在结果中，使用性能最好的方法: {reference_method}")
    
    # 对比
    comparison_df = compare_multiple_groups(groups, reference_method, test_type='ttest')
    
    # 格式化
    comparison_df['mean_std'] = comparison_df.apply(
        lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}",
        axis=1
    )
    comparison_df['ci_95'] = comparison_df.apply(
        lambda row: f"[{row['ci_95_lower']:.4f}, {row['ci_95_upper']:.4f}]",
        axis=1
    )
    
    return comparison_df


def create_ablation_table(
    results_dict: Dict[str, Dict[str, List[float]]],
    metric: str = 'auc',
    full_model_name: str = 'full_model'
) -> pd.DataFrame:
    """
    创建消融实验表格
    
    Args:
        results_dict: 结果字典
        metric: 要对比的指标
        full_model_name: 完整模型名称
    
    Returns:
        ablation_table: 消融实验表格
    """
    # 筛选消融实验
    ablation_experiments = {k: v for k, v in results_dict.items() if 'ablation' in k.lower()}
    
    if full_model_name in results_dict:
        ablation_experiments[full_model_name] = results_dict[full_model_name]
    
    if not ablation_experiments:
        return pd.DataFrame()
    
    # 提取指标
    groups = {}
    for method_name, metrics_dict in ablation_experiments.items():
        if metric in metrics_dict:
            groups[method_name] = metrics_dict[metric]
    
    if not groups:
        return pd.DataFrame()
    
    # 对比（以完整模型为参考）
    if full_model_name not in groups:
        full_model_name = max(groups.keys(), key=lambda k: np.mean(groups[k]))
    
    comparison_df = compare_multiple_groups(groups, full_model_name, test_type='ttest')
    
    # 添加移除模块信息
    comparison_df['removed_module'] = comparison_df['group'].apply(
        lambda x: extract_removed_module(x)
    )
    
    # 格式化
    comparison_df['mean_std'] = comparison_df.apply(
        lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}",
        axis=1
    )
    comparison_df['vs_full_model'] = comparison_df.apply(
        lambda row: f"{row['vs_reference_improvement']:.2%}" if row['vs_reference_improvement'] < 0 else f"+{row['vs_reference_improvement']:.2%}",
        axis=1
    )
    
    return comparison_df


def extract_removed_module(experiment_name: str) -> str:
    """从实验名称提取移除的模块"""
    name_lower = experiment_name.lower()
    
    if 'knowledge' in name_lower and 'visual' in name_lower:
        return 'Knowledge Notes + Visual Notes'
    elif 'knowledge' in name_lower:
        return 'Knowledge Notes'
    elif 'visual' in name_lower:
        return 'Visual Notes'
    elif 'ot' in name_lower:
        return 'Sinkhorn OT'
    elif 'dual' in name_lower:
        return 'Dual-Head'
    else:
        return 'Full Model'


def generate_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    output_path: Path
):
    """生成LaTeX表格"""
    latex_str = "\\begin{table}[htbp]\n"
    latex_str += "\\centering\n"
    latex_str += f"\\caption{{{caption}}}\n"
    latex_str += f"\\label{{{label}}}\n"
    
    # 表格内容
    latex_str += "\\begin{tabular}{" + "l" * len(df.columns) + "}\n"
    latex_str += "\\toprule\n"
    
    # 表头
    latex_str += " & ".join(df.columns) + " \\\\\n"
    latex_str += "\\midrule\n"
    
    # 数据行
    for _, row in df.iterrows():
        latex_str += " & ".join([str(val) for val in row.values]) + " \\\\\n"
    
    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "\\end{table}\n"
    
    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_str)
    
    print(f"✅ LaTeX表格已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='分析所有实验结果')
    parser.add_argument('--results_dir', type=str, default='experiments/results',
                       help='结果目录')
    parser.add_argument('--output_dir', type=str, default='experiments/results/analysis',
                       help='输出目录')
    parser.add_argument('--reference_method', type=str, default='full_model',
                       help='参考方法名称')
    parser.add_argument('--metric', type=str, default='auc',
                       choices=['auc', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score'],
                       help='主要对比指标')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("结果分析脚本")
    print("=" * 80)
    print(f"结果目录: {results_dir}")
    print(f"输出目录: {output_dir}")
    print(f"参考方法: {args.reference_method}")
    print(f"主要指标: {args.metric}")
    print("=" * 80)
    
    # 加载结果
    print("\n📊 加载实验结果...")
    results_dict = load_experiment_results(results_dir)
    
    if not results_dict:
        print("❌ 没有找到任何实验结果！")
        return
    
    print(f"✅ 加载了 {len(results_dict)} 个实验的结果")
    for exp_name in results_dict.keys():
        print(f"   - {exp_name}")
    
    # 创建Baseline对比表格
    print(f"\n📊 创建Baseline对比表格 ({args.metric})...")
    baseline_table = create_baseline_comparison_table(
        results_dict,
        metric=args.metric,
        reference_method=args.reference_method
    )
    
    if not baseline_table.empty:
        # 保存CSV
        csv_path = output_dir / f"baseline_comparison_{args.metric}.csv"
        baseline_table.to_csv(csv_path, index=False)
        print(f"✅ Baseline对比表格已保存: {csv_path}")
        
        # 保存LaTeX
        latex_path = output_dir / f"baseline_comparison_{args.metric}.tex"
        generate_latex_table(
            baseline_table[['group', 'mean_std', 'ci_95', 'vs_reference_improvement_pct', 'p_value', 'significance']],
            f"Baseline Comparison Results ({args.metric.upper()})",
            f"tab:baseline_comparison_{args.metric}",
            latex_path
        )
        
        # 打印表格
        print("\n" + "=" * 80)
        print("Baseline对比结果:")
        print("=" * 80)
        print(baseline_table[['group', 'mean_std', 'ci_95', 'vs_reference_improvement_pct', 'p_value', 'significance']].to_string(index=False))
    
    # 创建消融实验表格
    print(f"\n📊 创建消融实验表格 ({args.metric})...")
    ablation_table = create_ablation_table(
        results_dict,
        metric=args.metric,
        full_model_name=args.reference_method
    )
    
    if not ablation_table.empty:
        # 保存CSV
        csv_path = output_dir / f"ablation_comparison_{args.metric}.csv"
        ablation_table.to_csv(csv_path, index=False)
        print(f"✅ 消融实验表格已保存: {csv_path}")
        
        # 保存LaTeX
        latex_path = output_dir / f"ablation_comparison_{args.metric}.tex"
        generate_latex_table(
            ablation_table[['removed_module', 'mean_std', 'vs_full_model', 'p_value', 'significance']],
            f"Ablation Study Results ({args.metric.upper()})",
            f"tab:ablation_{args.metric}",
            latex_path
        )
        
        # 打印表格
        print("\n" + "=" * 80)
        print("消融实验结果:")
        print("=" * 80)
        print(ablation_table[['removed_module', 'mean_std', 'vs_full_model', 'p_value', 'significance']].to_string(index=False))
    
    # 生成汇总报告
    print("\n📊 生成汇总报告...")
    summary = {
        'total_experiments': len(results_dict),
        'baseline_experiments': len([k for k in results_dict.keys() if 'baseline' in k.lower()]),
        'ablation_experiments': len([k for k in results_dict.keys() if 'ablation' in k.lower()]),
        'full_model': args.reference_method if args.reference_method in results_dict else None,
        'main_metric': args.metric
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 汇总报告已保存: {summary_path}")
    
    print("\n" + "=" * 80)
    print("✅ 结果分析完成！")
    print("=" * 80)
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    main()

