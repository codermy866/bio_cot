#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统计检验工具 - 用于实验结果的统计显著性检验
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import pandas as pd


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    计算描述性统计
    
    Args:
        values: 数值列表
    
    Returns:
        stats: 统计信息字典
    """
    values = np.array(values)
    n = len(values)
    
    if n == 0:
        return {}
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # 样本标准差
    
    # 95%置信区间
    if n > 1:
        ci_95 = stats.t.interval(
            0.95, n-1,
            loc=mean,
            scale=stats.sem(values)
        )
    else:
        ci_95 = (mean, mean)
    
    # 中位数
    median = np.median(values)
    
    # 四分位数
    q25 = np.percentile(values, 25)
    q75 = np.percentile(values, 75)
    
    return {
        'n': n,
        'mean': mean,
        'std': std,
        'median': median,
        'min': np.min(values),
        'max': np.max(values),
        'q25': q25,
        'q75': q75,
        'ci_95_lower': ci_95[0],
        'ci_95_upper': ci_95[1],
        'ci_95': ci_95
    }


def test_significance(
    group_a: List[float],
    group_b: List[float],
    test_type: str = 'ttest',
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    统计显著性检验
    
    Args:
        group_a: 第一组数据
        group_b: 第二组数据
        test_type: 检验类型 ('ttest', 'wilcoxon', 'mannwhitney')
        alternative: 备择假设 ('two-sided', 'greater', 'less')
    
    Returns:
        result: 检验结果字典
    """
    group_a = np.array(group_a)
    group_b = np.array(group_b)
    
    if len(group_a) != len(group_b):
        # 如果样本数不同，使用独立样本检验
        if test_type == 'ttest':
            statistic, p_value = stats.ttest_ind(group_a, group_b, alternative=alternative)
        elif test_type == 'mannwhitney':
            statistic, p_value = stats.mannwhitneyu(group_a, group_b, alternative=alternative)
        else:
            raise ValueError(f"Unsupported test_type for independent samples: {test_type}")
    else:
        # 如果样本数相同，使用配对样本检验
        if test_type == 'ttest':
            statistic, p_value = stats.ttest_rel(group_a, group_b, alternative=alternative)
        elif test_type == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(group_a, group_b, alternative=alternative)
        else:
            raise ValueError(f"Unsupported test_type for paired samples: {test_type}")
    
    # 效应量 (Cohen's d)
    pooled_std = np.sqrt((np.var(group_a, ddof=1) + np.var(group_b, ddof=1)) / 2)
    cohens_d = (np.mean(group_a) - np.mean(group_b)) / pooled_std if pooled_std > 0 else 0
    
    # 显著性标记
    if p_value < 0.001:
        significance = '***'
    elif p_value < 0.01:
        significance = '**'
    elif p_value < 0.05:
        significance = '*'
    else:
        significance = 'ns'
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05,
        'significance': significance,
        'mean_a': np.mean(group_a),
        'mean_b': np.mean(group_b),
        'std_a': np.std(group_a, ddof=1),
        'std_b': np.std(group_b, ddof=1)
    }


def compare_multiple_groups(
    groups: Dict[str, List[float]],
    reference_group: str,
    test_type: str = 'ttest'
) -> pd.DataFrame:
    """
    多个组与参考组对比
    
    Args:
        groups: 组名到数据列表的字典
        reference_group: 参考组名称
        test_type: 检验类型
    
    Returns:
        comparison_df: 对比结果DataFrame
    """
    if reference_group not in groups:
        raise ValueError(f"Reference group '{reference_group}' not found in groups")
    
    reference_values = groups[reference_group]
    results = []
    
    for group_name, group_values in groups.items():
        if group_name == reference_group:
            continue
        
        # 统计检验
        test_result = test_significance(group_values, reference_values, test_type=test_type)
        
        # 计算改进幅度
        improvement = np.mean(group_values) - np.mean(reference_values)
        improvement_pct = (improvement / np.mean(reference_values)) * 100 if np.mean(reference_values) > 0 else 0
        
        # 统计信息
        stats_a = compute_statistics(group_values)
        stats_b = compute_statistics(reference_values)
        
        results.append({
            'group': group_name,
            'mean': stats_a['mean'],
            'std': stats_a['std'],
            'ci_95_lower': stats_a['ci_95_lower'],
            'ci_95_upper': stats_a['ci_95_upper'],
            'vs_reference_improvement': improvement,
            'vs_reference_improvement_pct': improvement_pct,
            'p_value': test_result['p_value'],
            'cohens_d': test_result['cohens_d'],
            'significant': test_result['significant'],
            'significance': test_result['significance']
        })
    
    # 添加参考组
    stats_ref = compute_statistics(reference_values)
    results.append({
        'group': reference_group,
        'mean': stats_ref['mean'],
        'std': stats_ref['std'],
        'ci_95_lower': stats_ref['ci_95_lower'],
        'ci_95_upper': stats_ref['ci_95_upper'],
        'vs_reference_improvement': 0.0,
        'vs_reference_improvement_pct': 0.0,
        'p_value': 1.0,
        'cohens_d': 0.0,
        'significant': False,
        'significance': '-'
    })
    
    df = pd.DataFrame(results)
    df = df.sort_values('mean', ascending=False)
    
    return df


def format_statistics_string(
    mean: float,
    std: float,
    ci_95: Optional[Tuple[float, float]] = None,
    precision: int = 4
) -> str:
    """
    格式化统计信息字符串
    
    Args:
        mean: 均值
        std: 标准差
        ci_95: 95%置信区间
        precision: 小数位数
    
    Returns:
        formatted_string: 格式化字符串
    """
    if ci_95 is not None:
        return f"{mean:.{precision}f} ± {std:.{precision}f} (95% CI: [{ci_95[0]:.{precision}f}, {ci_95[1]:.{precision}f}])"
    else:
        return f"{mean:.{precision}f} ± {std:.{precision}f}"


def create_comparison_table(
    results_dict: Dict[str, Dict[str, List[float]]],
    metric: str = 'auc',
    reference_method: Optional[str] = None
) -> pd.DataFrame:
    """
    创建对比表格
    
    Args:
        results_dict: 方法名到指标字典的映射
        metric: 要对比的指标
        reference_method: 参考方法（如果None，使用性能最好的方法）
    
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
    
    # 如果没有指定参考方法，使用均值最高的方法
    if reference_method is None:
        reference_method = max(groups.keys(), key=lambda k: np.mean(groups[k]))
    
    # 对比
    comparison_df = compare_multiple_groups(groups, reference_method)
    
    return comparison_df


if __name__ == '__main__':
    # 示例用法
    # 模拟数据
    group_a = [0.85, 0.86, 0.84, 0.87, 0.85]
    group_b = [0.80, 0.81, 0.79, 0.82, 0.80]
    
    # 描述性统计
    stats_a = compute_statistics(group_a)
    stats_b = compute_statistics(group_b)
    
    print("Group A Statistics:")
    print(f"  Mean: {stats_a['mean']:.4f} ± {stats_a['std']:.4f}")
    print(f"  95% CI: [{stats_a['ci_95_lower']:.4f}, {stats_a['ci_95_upper']:.4f}]")
    
    print("\nGroup B Statistics:")
    print(f"  Mean: {stats_b['mean']:.4f} ± {stats_b['std']:.4f}")
    print(f"  95% CI: [{stats_b['ci_95_lower']:.4f}, {stats_b['ci_95_upper']:.4f}]")
    
    # 统计检验
    test_result = test_significance(group_a, group_b, test_type='ttest')
    
    print(f"\nStatistical Test:")
    print(f"  p-value: {test_result['p_value']:.4f}")
    print(f"  Cohen's d: {test_result['cohens_d']:.4f}")
    print(f"  Significant: {test_result['significant']}")
    print(f"  Significance: {test_result['significance']}")

