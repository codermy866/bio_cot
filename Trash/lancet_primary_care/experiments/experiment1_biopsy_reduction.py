#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验1: HPV+患者活检率降低实验

目标: 证明多模态AI可以降低HPV+患者的过度活检率，同时保持高灵敏度

主要评估指标:
- 活检率降低百分比
- 灵敏度保持（非劣效性）
- PPV提升
- NPV保持
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score
)
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

AVAILABLE_FONTS = {font.name for font in fm.fontManager.ttflist}
FONT_FAMILY = 'Arial' if 'Arial' in AVAILABLE_FONTS else 'DejaVu Sans'

# 2025年最新设置：使用Arial字体（若不可用则使用DejaVu Sans）
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['font.sans-serif'] = [FONT_FAMILY, 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

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
sns.set_palette("husl")

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from utils.advanced_clinical_metrics import calculate_clinical_metrics


def ensure_serializable(obj):
    if isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [ensure_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


class BiopsyReductionExperiment:
    """HPV+患者活检率降低实验"""
    
    def __init__(
        self,
        data_path: str,
        model_path: str = None,
        output_dir: str = None
    ):
        self.data_path = data_path
        self.model_path = model_path
        self.output_dir = output_dir or "lancet_primary_care/results/experiment1_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载数据
        self.train_df = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))
        self.test_df = pd.read_csv(os.path.join(data_path, 'test_labels.csv'))
        
        # 筛选HPV+患者
        self.hpv_positive_train = self._filter_hpv_positive(self.train_df)
        self.hpv_positive_test = self._filter_hpv_positive(self.test_df)
        
        print(f"HPV+训练样本: {len(self.hpv_positive_train)}")
        print(f"HPV+测试样本: {len(self.hpv_positive_test)}")
    
    def _filter_hpv_positive(self, df: pd.DataFrame) -> pd.DataFrame:
        """筛选HPV+患者"""
        # 根据实际数据：HPV阳性为"1"，其他为阴性或缺失
        hpv_col = df['HPV清洗'].astype(str)
        # 方法1: 直接检查是否为"1"
        hpv_positive = (hpv_col == '1')
        # 方法2: 也支持包含"1"的情况（如果数据格式不一致）
        # hpv_positive = hpv_col.str.contains('^1$', regex=True, na=False)
        return df[hpv_positive].copy()
    
    def _traditional_screening_rule(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        传统HPV+TCT筛查规则
        规则: HPV+ 或 TCT异常（ASC-US+或"1"）→ 建议活检
        """
        # HPV状态：值为"1"表示阳性
        hpv_col = df['HPV清洗'].astype(str)
        hpv_positive = (hpv_col == '1').values
        
        # TCT状态（异常：ASC-US, LSIL, HSIL, 或"1"表示异常但类型未知）
        tct_col = df['TCT清洗'].astype(str).str.upper()
        tct_abnormal = (
            tct_col.str.contains('ASC-US|LSIL|HSIL', case=False, na=False) |
            (df['TCT清洗'].astype(str) == '1')
        ).values
        
        # 任一阳性即建议活检
        biopsy_recommendation = (hpv_positive | tct_abnormal).astype(int)
        
        # 真实标签
        true_labels = df['label'].values
        
        return biopsy_recommendation, true_labels
    
    def _multimodal_ai_rule(
        self, 
        df: pd.DataFrame, 
        predictions: np.ndarray,
        risk_threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        多模态AI筛查规则
        规则: 基于AI预测概率进行风险分层
        - 高风险 (>threshold): 建议活检
        - 中低风险 (≤threshold): 建议随访
        """
        # 高风险患者建议活检
        biopsy_recommendation = (predictions > risk_threshold).astype(int)
        
        # 真实标签
        true_labels = df['label'].values
        
        return biopsy_recommendation, true_labels
    
    def _calculate_biopsy_rate(self, recommendations: np.ndarray) -> float:
        """计算活检率"""
        return np.mean(recommendations) * 100
    
    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_probs: np.ndarray = None
    ) -> Dict:
        """计算评估指标"""
        metrics = calculate_clinical_metrics(y_true, y_pred, y_probs)
        
        # 添加活检率
        metrics['biopsy_rate'] = self._calculate_biopsy_rate(y_pred)
        
        return metrics
    
    def _non_inferiority_test(
        self,
        sensitivity_new: float,
        sensitivity_control: float,
        n_new: int,
        n_control: int,
        margin: float = 0.05
    ) -> Dict:
        """
        非劣效性检验
        检验新方法的灵敏度是否非劣于对照方法
        """
        # 计算差异和标准误
        diff = sensitivity_new - sensitivity_control
        se_new = np.sqrt(sensitivity_new * (1 - sensitivity_new) / n_new)
        se_control = np.sqrt(sensitivity_control * (1 - sensitivity_control) / n_control)
        se_diff = np.sqrt(se_new**2 + se_control**2)
        
        # 单侧检验：H0: diff <= -margin, H1: diff > -margin
        z_score = (diff + margin) / se_diff
        p_value = 1 - stats.norm.cdf(z_score)
        
        # 95%置信区间
        ci_lower = diff - 1.96 * se_diff
        ci_upper = diff + 1.96 * se_diff
        
        return {
            'difference': diff,
            'margin': margin,
            'z_score': z_score,
            'p_value': p_value,
            'non_inferior': p_value < 0.05,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }

    def _mcnemar_test(self, baseline_pred: np.ndarray, ai_pred: np.ndarray) -> Dict:
        baseline_pred = np.asarray(baseline_pred)
        ai_pred = np.asarray(ai_pred)
        b01 = np.sum((baseline_pred == 0) & (ai_pred == 1))
        b10 = np.sum((baseline_pred == 1) & (ai_pred == 0))
        denom = b01 + b10
        if denom == 0:
            statistic = 0.0
            p_value = 1.0
        else:
            statistic = (abs(b01 - b10) - 1) ** 2 / denom
            p_value = stats.chi2.sf(statistic, 1)
        return {
            'statistic': float(statistic),
            'p_value': float(p_value)
        }
    
    def run_experiment(
        self,
        ai_predictions: np.ndarray = None,
        ai_probabilities: np.ndarray = None,
        risk_threshold: float = 0.5
    ) -> Dict:
        """
        运行实验
        
        Args:
            ai_predictions: AI模型的预测结果（0/1）
            ai_probabilities: AI模型的预测概率
            risk_threshold: 风险阈值
        """
        print("\n" + "="*60)
        print("实验1: HPV+患者活检率降低实验")
        print("="*60)
        
        # 使用测试集
        test_df = self.hpv_positive_test
        
        # 1. 传统HPV+TCT方案
        print("\n[1] 评估传统HPV+TCT方案...")
        traditional_pred, true_labels = self._traditional_screening_rule(test_df)
        traditional_metrics = self._calculate_metrics(
            true_labels, 
            traditional_pred,
            y_probs=None
        )
        
        print(f"  活检率: {traditional_metrics['biopsy_rate']:.2f}%")
        print(f"  灵敏度: {traditional_metrics['sensitivity']:.4f}")
        print(f"  特异度: {traditional_metrics['specificity']:.4f}")
        print(f"  PPV: {traditional_metrics['ppv']:.4f}")
        print(f"  NPV: {traditional_metrics['npv']:.4f}")
        
        # 2. 多模态AI方案
        print("\n[2] 评估多模态AI方案...")
        if ai_predictions is None:
            # 如果没有提供预测，使用概率和阈值生成
            if ai_probabilities is None:
                raise ValueError("需要提供ai_predictions或ai_probabilities")
            ai_predictions = (ai_probabilities > risk_threshold).astype(int)
        
        ai_metrics = self._calculate_metrics(
            true_labels,
            ai_predictions,
            y_probs=ai_probabilities
        )
        
        print(f"  活检率: {ai_metrics['biopsy_rate']:.2f}%")
        print(f"  灵敏度: {ai_metrics['sensitivity']:.4f}")
        print(f"  特异度: {ai_metrics['specificity']:.4f}")
        print(f"  PPV: {ai_metrics['ppv']:.4f}")
        print(f"  NPV: {ai_metrics['npv']:.4f}")
        if 'auc' in ai_metrics:
            print(f"  AUC: {ai_metrics['auc']:.4f}")
        
        # 3. 计算改善指标
        print("\n[3] 计算改善指标...")
        biopsy_reduction = (
            (traditional_metrics['biopsy_rate'] - ai_metrics['biopsy_rate']) 
            / traditional_metrics['biopsy_rate'] * 100
        )
        ppv_improvement = ai_metrics['ppv'] - traditional_metrics['ppv']
        npv_change = ai_metrics['npv'] - traditional_metrics['npv']
        sensitivity_change = ai_metrics['sensitivity'] - traditional_metrics['sensitivity']
        
        print(f"  活检率降低: {biopsy_reduction:.2f}%")
        print(f"  PPV提升: {ppv_improvement:.4f}")
        print(f"  NPV变化: {npv_change:.4f}")
        print(f"  灵敏度变化: {sensitivity_change:.4f}")
        
        # 4. 非劣效性检验
        print("\n[4] 非劣效性检验（灵敏度）...")
        n_traditional = len(true_labels)
        n_ai = len(true_labels)
        non_inferiority = self._non_inferiority_test(
            ai_metrics['sensitivity'],
            traditional_metrics['sensitivity'],
            n_ai,
            n_traditional,
            margin=0.05
        )
        
        print(f"  差异: {non_inferiority['difference']:.4f}")
        print(f"  95% CI: [{non_inferiority['ci_lower']:.4f}, {non_inferiority['ci_upper']:.4f}]")
        print(f"  P值: {non_inferiority['p_value']:.4f}")
        print(f"  非劣效: {'是' if non_inferiority['non_inferior'] else '否'}")
        
        # 5. 统计检验
        print("\n[5] 统计检验...")
        # McNemar检验（配对分类结果）
        contingency_table = pd.crosstab(traditional_pred, ai_predictions)
        mcnemar_result = self._mcnemar_test(traditional_pred, ai_predictions)
        print(f"  McNemar检验 P值: {mcnemar_result['p_value']:.4f}")
        
        # 6. 整理结果
        results = {
            'traditional': traditional_metrics,
            'multimodal_ai': ai_metrics,
            'improvements': {
                'biopsy_rate_reduction_percent': float(biopsy_reduction),
                'ppv_improvement': float(ppv_improvement),
                'npv_change': float(npv_change),
                'sensitivity_change': float(sensitivity_change)
            },
            'non_inferiority_test': non_inferiority,
            'mcnemar_test': mcnemar_result,
            'sample_size': {
                'total': len(true_labels),
                'positive': int(np.sum(true_labels)),
                'negative': int(np.sum(1 - true_labels))
            }
        }
        
        # 7. 保存结果
        results_file = os.path.join(self.output_dir, 'experiment1_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(ensure_serializable(results), f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {results_file}")
        
        return results
    
    def plot_results(self, results: Dict):
        """可视化结果 - 2025年最新样式，英文Arial字体"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Experiment 1: Biopsy Rate Reduction in HPV+ Patients', 
                     fontsize=16, fontweight='bold', fontfamily=FONT_FAMILY)
        
        # 1. 活检率对比
        ax1 = axes[0, 0]
        methods = ['Traditional\nHPV+TCT', 'Multimodal AI']
        biopsy_rates = [
            results['traditional']['biopsy_rate'],
            results['multimodal_ai']['biopsy_rate']
        ]
        colors = ['#FF6B6B', '#4ECDC4']
        bars = ax1.bar(methods, biopsy_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Biopsy Rate (%)', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax1.set_title('Biopsy Rate Comparison', fontsize=13, fontweight='bold', fontfamily=FONT_FAMILY)
        ax1.set_ylim(0, max(biopsy_rates) * 1.2)
        
        # 添加数值标签
        for bar, rate in zip(bars, biopsy_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 添加降低百分比
        reduction = results['improvements']['biopsy_rate_reduction_percent']
        ax1.text(0.5, max(biopsy_rates) * 1.1, 
                f'Reduction: {reduction:.1f}%',
                ha='center', fontsize=11, fontweight='bold', color='green', fontfamily=FONT_FAMILY)
        
        # 2. 灵敏度对比
        ax2 = axes[0, 1]
        sensitivities = [
            results['traditional']['sensitivity'],
            results['multimodal_ai']['sensitivity']
        ]
        bars = ax2.bar(methods, sensitivities, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Sensitivity', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax2.set_title('Sensitivity Comparison\n(Non-inferiority Test)', fontsize=13, fontweight='bold', fontfamily=FONT_FAMILY)
        ax2.set_ylim(0, 1.1)
        
        for bar, sens in zip(bars, sensitivities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{sens:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 3. PPV对比
        ax3 = axes[1, 0]
        ppvs = [
            results['traditional']['ppv'],
            results['multimodal_ai']['ppv']
        ]
        bars = ax3.bar(methods, ppvs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('PPV (Positive Predictive Value)', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax3.set_title('PPV Comparison', fontsize=13, fontweight='bold', fontfamily=FONT_FAMILY)
        ax3.set_ylim(0, 1.1)
        
        for bar, ppv in zip(bars, ppvs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ppv:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 4. 综合指标雷达图
        ax4 = axes[1, 1]
        categories = ['Sensitivity', 'Specificity', 'PPV', 'NPV']
        traditional_values = [
            results['traditional']['sensitivity'],
            results['traditional']['specificity'],
            results['traditional']['ppv'],
            results['traditional']['npv']
        ]
        ai_values = [
            results['multimodal_ai']['sensitivity'],
            results['multimodal_ai']['specificity'],
            results['multimodal_ai']['ppv'],
            results['multimodal_ai']['npv']
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        ax4.bar(x - width/2, traditional_values, width, label='Traditional HPV+TCT', 
               color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax4.bar(x + width/2, ai_values, width, label='Multimodal AI', 
               color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Metric Value', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax4.set_title('Comprehensive Metrics Comparison', fontsize=13, fontweight='bold', fontfamily=FONT_FAMILY)
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories, fontfamily=FONT_FAMILY)
        ax4.set_ylim(0, 1.1)
        ax4.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 设置所有文本字体
        for label in ax4.get_xticklabels():
            label.set_fontfamily(FONT_FAMILY)
        for label in ax4.get_yticklabels():
            label.set_fontfamily(FONT_FAMILY)
        
        plt.tight_layout()
        
        # 设置所有文本字体
        for ax in axes.flat:
            for label in ax.get_xticklabels():
                label.set_fontfamily(FONT_FAMILY)
            for label in ax.get_yticklabels():
                label.set_fontfamily(FONT_FAMILY)
            ax.tick_params(labelsize=10)
        
        # 保存图片 - 2025年高质量输出
        fig_path = os.path.join(self.output_dir, 'experiment1_results.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✅ Figure saved to: {fig_path}")
        plt.close()


def main():
    """主函数 - 示例用法"""
    # 初始化实验
    experiment = BiopsyReductionExperiment(
        data_path='5centers_multi',
        output_dir='lancet_primary_care/results/experiment1_results'
    )
    
    # 注意: 这里需要实际的AI模型预测结果
    # 在实际使用时，需要加载模型并生成预测
    # 这里使用模拟数据进行演示
    
    # 模拟AI预测概率（实际应该从模型获取）
    test_df = experiment.hpv_positive_test
    n_samples = len(test_df)
    
    # 生成模拟预测（实际应该使用真实模型）
    print("\n⚠️  注意: 当前使用模拟数据，实际使用时需要加载真实模型")
    np.random.seed(42)
    ai_probabilities = np.random.beta(2, 3, n_samples)  # 模拟概率
    # 根据真实标签调整，使模拟更合理
    true_labels = test_df['label'].values
    ai_probabilities[true_labels == 1] = np.random.beta(5, 2, np.sum(true_labels == 1))
    ai_probabilities[true_labels == 0] = np.random.beta(2, 5, np.sum(true_labels == 0))
    
    # 运行实验
    results = experiment.run_experiment(
        ai_probabilities=ai_probabilities,
        risk_threshold=0.5
    )
    
    # 可视化结果
    experiment.plot_results(results)
    
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)


if __name__ == '__main__':
    main()

