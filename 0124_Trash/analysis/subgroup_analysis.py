#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
亚组分析 (Subgroup Analysis)
评估模型在不同患者亚组中的性能
Lancet期刊要求：必须报告亚组分析结果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from scipy import stats
from typing import Dict, List, Tuple
import json
import os
from analysis.bootstrap_confidence_intervals import BootstrapCI

# 设置字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


class SubgroupAnalyzer:
    """亚组分析器"""
    
    def __init__(self, n_bootstrap=1000):
        self.n_bootstrap = n_bootstrap
        self.bootstrap_ci = BootstrapCI(n_bootstrap=n_bootstrap)
    
    def define_subgroups(self, metadata: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        定义亚组
        
        Args:
            metadata: 包含患者元数据的DataFrame
        
        Returns:
            亚组字典 {subgroup_name: boolean_mask}
        """
        subgroups = {}
        
        # 1. 年龄组
        if 'AGE' in metadata.columns:
            age = metadata['AGE'].values
            subgroups['Age <30'] = age < 30
            subgroups['Age 30-40'] = (age >= 30) & (age < 40)
            subgroups['Age 40-50'] = (age >= 40) & (age < 50)
            subgroups['Age ≥50'] = age >= 50
        
        # 2. HPV类型（如果可用）
        if 'HPV清洗' in metadata.columns:
            hpv = metadata['HPV清洗'].values
            # 根据实际数据调整
            subgroups['HPV16/18'] = hpv.isin(['16', '18']) if hasattr(hpv, 'isin') else False
            subgroups['Other HR-HPV'] = ~subgroups.get('HPV16/18', np.zeros(len(metadata), dtype=bool))
        
        # 3. TCT结果（如果可用）
        if 'TCT清洗' in metadata.columns:
            tct = metadata['TCT清洗'].values
            subgroups['TCT Positive'] = tct == 1  # 根据实际编码调整
            subgroups['TCT Negative'] = tct == 0
        
        # 4. 中心（如果可用）
        if 'center_id' in metadata.columns:
            centers = metadata['center_id'].unique()
            for center in centers:
                subgroups[f'Center_{center}'] = metadata['center_id'] == center
        
        return subgroups
    
    def evaluate_subgroup(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_probs: np.ndarray, mask: np.ndarray) -> Dict:
        """
        评估单个亚组的性能
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_probs: 预测概率
            mask: 亚组掩码
        
        Returns:
            包含所有指标的字典
        """
        if np.sum(mask) == 0:
            return None
        
        y_true_sub = y_true[mask]
        y_pred_sub = y_pred[mask]
        y_probs_sub = y_probs[mask]
        
        n = len(y_true_sub)
        
        # 基础指标
        acc = accuracy_score(y_true_sub, y_pred_sub)
        f1 = f1_score(y_true_sub, y_pred_sub, zero_division=0)
        precision = precision_score(y_true_sub, y_pred_sub, zero_division=0)
        recall = recall_score(y_true_sub, y_pred_sub, zero_division=0)
        
        # AUC
        try:
            auc = roc_auc_score(y_true_sub, y_probs_sub)
        except:
            auc = np.nan
        
        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true_sub, y_pred_sub).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # Bootstrap CI for AUC
        if not np.isnan(auc):
            auc_mean, auc_low, auc_high = self.bootstrap_ci.calculate_ci(
                lambda y_true, y_probs: roc_auc_score(y_true, y_probs),
                y_true_sub, y_probs=y_probs_sub
            )
        else:
            auc_mean, auc_low, auc_high = np.nan, np.nan, np.nan
        
        return {
            'n_samples': int(n),
            'n_positive': int(np.sum(y_true_sub == 1)),
            'n_negative': int(np.sum(y_true_sub == 0)),
            'accuracy': float(acc),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'ppv': float(ppv),
            'npv': float(npv),
            'auc': float(auc),
            'auc_ci_lower': float(auc_low),
            'auc_ci_upper': float(auc_high),
            'auc_formatted': f"{auc:.4f} ({auc_low:.4f}-{auc_high:.4f})" if not np.isnan(auc) else "N/A"
        }
    
    def analyze_all_subgroups(self, y_true: np.ndarray, y_pred: np.ndarray,
                             y_probs: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        分析所有亚组
        
        Returns:
            包含所有亚组结果的DataFrame
        """
        subgroups = self.define_subgroups(metadata)
        
        results = []
        for subgroup_name, mask in subgroups.items():
            result = self.evaluate_subgroup(y_true, y_pred, y_probs, mask)
            if result is not None:
                result['subgroup'] = subgroup_name
                results.append(result)
        
        return pd.DataFrame(results)
    
    def plot_forest_plot(self, subgroup_results: pd.DataFrame, 
                        metric: str = 'auc', output_path: str = None):
        """
        绘制森林图 (Forest Plot)
        
        Args:
            subgroup_results: 亚组分析结果DataFrame
            metric: 要绘制的指标 ('auc', 'sensitivity', 'specificity')
            output_path: 输出路径
        """
        if metric not in subgroup_results.columns:
            print(f"⚠️  指标 {metric} 不存在")
            return
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(subgroup_results) * 0.5)))
        
        subgroups = subgroup_results['subgroup'].values
        values = subgroup_results[metric].values
        
        # 如果有CI，使用CI；否则只显示点估计
        if f'{metric}_ci_lower' in subgroup_results.columns:
            ci_lower = subgroup_results[f'{metric}_ci_lower'].values
            ci_upper = subgroup_results[f'{metric}_ci_upper'].values
            
            y_pos = np.arange(len(subgroups))
            
            # 绘制置信区间
            ax.errorbar(values, y_pos, 
                       xerr=[values - ci_lower, ci_upper - values],
                       fmt='o', capsize=5, capthick=2, markersize=8,
                       color='#2E86AB', linewidth=2)
        else:
            y_pos = np.arange(len(subgroups))
            ax.scatter(values, y_pos, s=100, color='#2E86AB', zorder=3)
        
        # 设置标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels(subgroups, fontsize=11)
        ax.set_xlabel(f'{metric.upper()} (95% CI)', fontsize=14, fontweight='bold')
        ax.set_title(f'Subgroup Analysis: {metric.upper()}', fontsize=16, fontweight='bold', pad=20)
        
        # 添加参考线
        if metric == 'auc':
            ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Random (0.5)')
            ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='Good Performance (0.8)')
        
        # 添加数值标签
        for i, (subgroup, value) in enumerate(zip(subgroups, values)):
            if f'{metric}_formatted' in subgroup_results.columns:
                label = subgroup_results[f'{metric}_formatted'].iloc[i]
            else:
                label = f'{value:.3f}'
            ax.text(value + 0.02, i, label, va='center', fontsize=10)
        
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ 森林图已保存: {output_path}")
        plt.close()
    
    def statistical_comparison(self, subgroup_results: pd.DataFrame) -> pd.DataFrame:
        """
        统计比较不同亚组（使用DeLong test for AUC）
        
        Returns:
            包含p值的比较结果
        """
        # TODO: 实现DeLong test for AUC comparison
        # 这里先返回占位符
        print("⚠️  Statistical comparison (DeLong test) 需要实现")
        return pd.DataFrame()


def load_data_and_predictions(result_dir: str, data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    加载数据和预测结果
    
    Returns:
        (y_true, y_pred, y_probs, metadata)
    """
    # TODO: 实现实际的数据加载
    # 需要从模型重新预测或从保存的结果加载
    print("⚠️  需要实现数据加载逻辑")
    return None, None, None, None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='亚组分析')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='模型结果目录')
    parser.add_argument('--data_path', type=str, default='5centers_multi',
                       help='数据路径')
    parser.add_argument('--output_dir', default='analysis/subgroup_results',
                       help='输出目录')
    parser.add_argument('--model_name', default='SwinT',
                       help='模型名称')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据和预测
    y_true, y_pred, y_probs, metadata = load_data_and_predictions(args.result_dir, args.data_path)
    
    if y_true is None:
        print("⚠️  需要从模型重新预测或加载保存的预测结果")
        print("   建议：在训练脚本中保存所有预测概率和元数据")
        return
    
    # 执行亚组分析
    analyzer = SubgroupAnalyzer()
    subgroup_results = analyzer.analyze_all_subgroups(y_true, y_pred, y_probs, metadata)
    
    # 保存结果
    subgroup_results.to_csv(os.path.join(args.output_dir, 'subgroup_analysis.csv'), index=False)
    
    # 绘制森林图
    for metric in ['auc', 'sensitivity', 'specificity']:
        if metric in subgroup_results.columns:
            output_path = os.path.join(args.output_dir, f'forest_plot_{metric}.png')
            analyzer.plot_forest_plot(subgroup_results, metric, output_path)
    
    print(f"\n✅ 亚组分析完成！结果保存在: {args.output_dir}")
    print(f"\n📊 亚组分析汇总:")
    print(subgroup_results[['subgroup', 'n_samples', 'auc_formatted', 
                           'sensitivity', 'specificity']].to_string(index=False))


if __name__ == '__main__':
    main()

