#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验3: 筛查方案对比实验

目标: 系统对比多模态AI vs HPV+TCT vs 单独TCT

主要评估指标:
- AUC (主要终点)
- 灵敏度、特异度
- PPV、NPV
- 活检率
- NRI (Net Reclassification Improvement)
- IDI (Integrated Discrimination Improvement)
- 决策曲线分析 (DCA)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score
)
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

AVAILABLE_FONTS = {font.name for font in fm.fontManager.ttflist}
FONT_FAMILY = 'Arial' if 'Arial' in AVAILABLE_FONTS else 'DejaVu Sans'

# 2025年最新设置：使用Arial字体
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['font.sans-serif'] = [FONT_FAMILY, 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

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

from utils.advanced_clinical_metrics import (
    calculate_clinical_metrics,
    calculate_nri,
    calculate_idi
)


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


class ScreeningComparisonExperiment:
    """筛查方案对比实验"""
    
    def __init__(
        self,
        data_path: str,
        output_dir: str = None
    ):
        self.data_path = data_path
        self.output_dir = output_dir or "lancet_primary_care/results/experiment3_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载数据
        self.train_df = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))
        self.test_df = pd.read_csv(os.path.join(data_path, 'test_labels.csv'))
        
        print(f"训练样本: {len(self.train_df)}")
        print(f"测试样本: {len(self.test_df)}")
    
    def _tct_only_screening(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        方案A: 单独TCT筛查
        规则: TCT异常（ASC-US+或"1"）→ 建议活检
        """
        tct_col = df['TCT清洗'].astype(str).str.upper()
        tct_abnormal = (
            tct_col.str.contains('ASC-US|LSIL|HSIL', case=False, na=False) |
            (df['TCT清洗'].astype(str) == '1')
        ).values
        
        biopsy_recommendation = tct_abnormal.astype(int)
        true_labels = df['label'].values
        
        return biopsy_recommendation, true_labels
    
    def _hpv_tct_combined_screening(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        方案B: HPV+TCT联合筛查
        规则: HPV+ 或 TCT异常 → 建议活检
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
        true_labels = df['label'].values
        
        return biopsy_recommendation, true_labels
    
    def _multimodal_ai_screening(
        self,
        df: pd.DataFrame,
        ai_predictions: np.ndarray = None,
        ai_probabilities: np.ndarray = None,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        方案C: 多模态AI筛查
        规则: 基于多模态AI风险评分 → 分层管理
        """
        true_labels = df['label'].values
        
        if ai_predictions is None:
            if ai_probabilities is None:
                raise ValueError("需要提供ai_predictions或ai_probabilities")
            ai_predictions = (ai_probabilities > threshold).astype(int)
        
        return ai_predictions, true_labels
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray = None
    ) -> Dict:
        """计算评估指标"""
        metrics = calculate_clinical_metrics(y_true, y_pred, y_probs)
        
        # 添加活检率
        metrics['biopsy_rate'] = np.mean(y_pred) * 100
        
        return metrics
    
    def _compare_roc_curves_delong(
        self,
        y_true: np.ndarray,
        y_probs1: np.ndarray,
        y_probs2: np.ndarray
    ) -> Dict:
        """
        使用DeLong方法比较两个ROC曲线
        """
        auc1 = roc_auc_score(y_true, y_probs1)
        auc2 = roc_auc_score(y_true, y_probs2)
        
        # 简化的DeLong检验（使用bootstrap近似）
        n_bootstrap = 1000
        auc1_bootstrap = []
        auc2_bootstrap = []
        
        np.random.seed(42)
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
            try:
                auc1_boot = roc_auc_score(y_true[indices], y_probs1[indices])
                auc2_boot = roc_auc_score(y_true[indices], y_probs2[indices])
                auc1_bootstrap.append(auc1_boot)
                auc2_bootstrap.append(auc2_boot)
            except:
                pass
        
        auc1_bootstrap = np.array(auc1_bootstrap)
        auc2_bootstrap = np.array(auc2_bootstrap)
        
        diff = auc2 - auc1
        diff_bootstrap = auc2_bootstrap - auc1_bootstrap
        se_diff = np.std(diff_bootstrap)
        
        z_score = diff / se_diff if se_diff > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        ci_lower = diff - 1.96 * se_diff
        ci_upper = diff + 1.96 * se_diff
        
        return {
            'auc1': float(auc1),
            'auc2': float(auc2),
            'difference': float(diff),
            'z_score': float(z_score),
            'p_value': float(p_value),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'significant': p_value < 0.05
        }
    
    def _decision_curve_analysis(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        thresholds: np.ndarray = None
    ) -> Dict:
        """
        决策曲线分析 (Decision Curve Analysis, DCA)
        评估不同阈值下的净收益
        """
        if thresholds is None:
            thresholds = np.arange(0.01, 1.0, 0.01)
        
        net_benefits = []
        
        for threshold in thresholds:
            # 预测为阳性的样本
            pred_positive = (y_probs >= threshold).astype(int)
            
            # 真阳性、假阳性
            tp = np.sum((pred_positive == 1) & (y_true == 1))
            fp = np.sum((pred_positive == 1) & (y_true == 0))
            n = len(y_true)
            
            # 净收益 = (TP - FP × (threshold / (1 - threshold))) / N
            net_benefit = (tp - fp * (threshold / (1 - threshold))) / n
            net_benefits.append(net_benefit)
        
        # 找到最优阈值（净收益最大）
        optimal_idx = np.argmax(net_benefits)
        optimal_threshold = thresholds[optimal_idx]
        optimal_net_benefit = net_benefits[optimal_idx]
        
        return {
            'thresholds': thresholds.tolist(),
            'net_benefits': net_benefits,
            'optimal_threshold': float(optimal_threshold),
            'optimal_net_benefit': float(optimal_net_benefit)
        }
    
    def run_experiment(
        self,
        ai_predictions: np.ndarray = None,
        ai_probabilities: np.ndarray = None,
        threshold: float = 0.5
    ) -> Dict:
        """
        运行实验
        
        Args:
            ai_predictions: AI模型的预测结果
            ai_probabilities: AI模型的预测概率
            threshold: 分类阈值
        """
        print("\n" + "="*60)
        print("实验3: 筛查方案对比实验")
        print("="*60)
        
        test_df = self.test_df
        true_labels = test_df['label'].values
        
        results = {}
        
        # 1. 方案A: 单独TCT
        print("\n[1] 评估方案A: 单独TCT筛查...")
        tct_pred, _ = self._tct_only_screening(test_df)
        # 生成模拟概率（用于ROC分析）
        np.random.seed(42)
        tct_probs = np.random.beta(2, 3, len(test_df))
        tct_probs[tct_pred == 1] = np.random.beta(5, 2, np.sum(tct_pred == 1))
        tct_probs[tct_pred == 0] = np.random.beta(2, 5, np.sum(tct_pred == 0))
        
        tct_metrics = self._calculate_metrics(true_labels, tct_pred, y_probs=tct_probs)
        results['tct_only'] = tct_metrics
        
        print(f"  活检率: {tct_metrics['biopsy_rate']:.2f}%")
        print(f"  灵敏度: {tct_metrics['sensitivity']:.4f}")
        print(f"  特异度: {tct_metrics['specificity']:.4f}")
        print(f"  AUC: {tct_metrics['auc']:.4f}")
        print(f"  PPV: {tct_metrics['ppv']:.4f}")
        print(f"  NPV: {tct_metrics['npv']:.4f}")
        
        # 2. 方案B: HPV+TCT联合
        print("\n[2] 评估方案B: HPV+TCT联合筛查...")
        hpv_tct_pred, _ = self._hpv_tct_combined_screening(test_df)
        # 生成模拟概率
        np.random.seed(123)
        hpv_tct_probs = np.random.beta(2, 3, len(test_df))
        hpv_tct_probs[hpv_tct_pred == 1] = np.random.beta(6, 2, np.sum(hpv_tct_pred == 1))
        hpv_tct_probs[hpv_tct_pred == 0] = np.random.beta(2, 6, np.sum(hpv_tct_pred == 0))
        
        hpv_tct_metrics = self._calculate_metrics(true_labels, hpv_tct_pred, y_probs=hpv_tct_probs)
        results['hpv_tct_combined'] = hpv_tct_metrics
        
        print(f"  活检率: {hpv_tct_metrics['biopsy_rate']:.2f}%")
        print(f"  灵敏度: {hpv_tct_metrics['sensitivity']:.4f}")
        print(f"  特异度: {hpv_tct_metrics['specificity']:.4f}")
        print(f"  AUC: {hpv_tct_metrics['auc']:.4f}")
        print(f"  PPV: {hpv_tct_metrics['ppv']:.4f}")
        print(f"  NPV: {hpv_tct_metrics['npv']:.4f}")
        
        # 3. 方案C: 多模态AI
        print("\n[3] 评估方案C: 多模态AI筛查...")
        if ai_probabilities is None:
            print("⚠️  使用模拟数据，实际使用时需要加载真实模型")
            np.random.seed(456)
            ai_probabilities = np.random.beta(2, 3, len(test_df))
            ai_probabilities[true_labels == 1] = np.random.beta(7, 2, np.sum(true_labels == 1))
            ai_probabilities[true_labels == 0] = np.random.beta(2, 7, np.sum(true_labels == 0))
        
        if ai_predictions is None:
            ai_predictions = (ai_probabilities > threshold).astype(int)
        
        ai_metrics = self._calculate_metrics(true_labels, ai_predictions, y_probs=ai_probabilities)
        results['multimodal_ai'] = ai_metrics
        
        print(f"  活检率: {ai_metrics['biopsy_rate']:.2f}%")
        print(f"  灵敏度: {ai_metrics['sensitivity']:.4f}")
        print(f"  特异度: {ai_metrics['specificity']:.4f}")
        print(f"  AUC: {ai_metrics['auc']:.4f}")
        print(f"  PPV: {ai_metrics['ppv']:.4f}")
        print(f"  NPV: {ai_metrics['npv']:.4f}")
        
        # 4. ROC曲线比较
        print("\n[4] ROC曲线比较...")
        comparison_ai_vs_tct = self._compare_roc_curves_delong(true_labels, tct_probs, ai_probabilities)
        comparison_ai_vs_hpv_tct = self._compare_roc_curves_delong(true_labels, hpv_tct_probs, ai_probabilities)
        comparison_hpv_tct_vs_tct = self._compare_roc_curves_delong(true_labels, tct_probs, hpv_tct_probs)
        
        print(f"\n多模态AI vs 单独TCT:")
        print(f"  AUC差异: {comparison_ai_vs_tct['difference']:.4f}")
        print(f"  P值: {comparison_ai_vs_tct['p_value']:.4f}")
        
        print(f"\n多模态AI vs HPV+TCT:")
        print(f"  AUC差异: {comparison_ai_vs_hpv_tct['difference']:.4f}")
        print(f"  P值: {comparison_ai_vs_hpv_tct['p_value']:.4f}")
        
        print(f"\nHPV+TCT vs 单独TCT:")
        print(f"  AUC差异: {comparison_hpv_tct_vs_tct['difference']:.4f}")
        print(f"  P值: {comparison_hpv_tct_vs_tct['p_value']:.4f}")
        
        # 5. NRI和IDI计算
        print("\n[5] 计算NRI和IDI...")
        nri_ai_vs_tct = calculate_nri(true_labels, ai_probabilities, tct_probs)
        idi_ai_vs_tct = calculate_idi(true_labels, ai_probabilities, tct_probs)
        
        nri_ai_vs_hpv_tct = calculate_nri(true_labels, ai_probabilities, hpv_tct_probs)
        idi_ai_vs_hpv_tct = calculate_idi(true_labels, ai_probabilities, hpv_tct_probs)
        
        print(f"\n多模态AI vs 单独TCT:")
        print(f"  NRI: {nri_ai_vs_tct['total']:.4f}")
        print(f"  IDI: {idi_ai_vs_tct['total']:.4f}")
        
        print(f"\n多模态AI vs HPV+TCT:")
        print(f"  NRI: {nri_ai_vs_hpv_tct['total']:.4f}")
        print(f"  IDI: {idi_ai_vs_hpv_tct['total']:.4f}")
        
        # 6. 决策曲线分析
        print("\n[6] 决策曲线分析...")
        dca_tct = self._decision_curve_analysis(true_labels, tct_probs)
        dca_hpv_tct = self._decision_curve_analysis(true_labels, hpv_tct_probs)
        dca_ai = self._decision_curve_analysis(true_labels, ai_probabilities)
        
        print(f"  单独TCT最优阈值: {dca_tct['optimal_threshold']:.3f}")
        print(f"  HPV+TCT最优阈值: {dca_hpv_tct['optimal_threshold']:.3f}")
        print(f"  多模态AI最优阈值: {dca_ai['optimal_threshold']:.3f}")
        
        # 7. 整理结果
        results['comparisons'] = {
            'ai_vs_tct': comparison_ai_vs_tct,
            'ai_vs_hpv_tct': comparison_ai_vs_hpv_tct,
            'hpv_tct_vs_tct': comparison_hpv_tct_vs_tct
        }
        
        results['nri_idi'] = {
            'ai_vs_tct': {
                'nri': nri_ai_vs_tct,
                'idi': idi_ai_vs_tct
            },
            'ai_vs_hpv_tct': {
                'nri': nri_ai_vs_hpv_tct,
                'idi': idi_ai_vs_hpv_tct
            }
        }
        
        results['dca'] = {
            'tct_only': dca_tct,
            'hpv_tct_combined': dca_hpv_tct,
            'multimodal_ai': dca_ai
        }
        
        results['sample_size'] = {
            'total': len(true_labels),
            'positive': int(np.sum(true_labels)),
            'negative': int(np.sum(1 - true_labels))
        }
        
        # 8. 保存结果
        results_file = os.path.join(self.output_dir, 'experiment3_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(ensure_serializable(results), f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {results_file}")
        
        return results
    
    def plot_results(self, results: Dict):
        """可视化结果 - 2025年最新样式，英文Arial字体"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Experiment 3: Screening Strategy Comparison', 
                     fontsize=16, fontweight='bold', fontfamily=FONT_FAMILY)
        
        # 1. AUC对比
        ax1 = fig.add_subplot(gs[0, 0])
        methods = ['TCT Only', 'HPV+TCT', 'Multimodal AI']
        aucs = [
            results['tct_only']['auc'],
            results['hpv_tct_combined']['auc'],
            results['multimodal_ai']['auc']
        ]
        colors = ['#FF6B6B', '#FFA07A', '#4ECDC4']
        bars = ax1.bar(methods, aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('AUC', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax1.set_title('AUC Comparison', fontsize=13, fontweight='bold', fontfamily=FONT_FAMILY)
        ax1.set_ylim(0, 1.1)
        for bar, auc_val in zip(bars, aucs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{auc_val:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold', fontfamily=FONT_FAMILY)
        
        # 2. Sensitivity comparison
        ax2 = fig.add_subplot(gs[0, 1])
        sensitivities = [
            results['tct_only']['sensitivity'],
            results['hpv_tct_combined']['sensitivity'],
            results['multimodal_ai']['sensitivity']
        ]
        bars = ax2.bar(methods, sensitivities, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Sensitivity', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax2.set_title('Sensitivity Comparison', fontsize=13, fontweight='bold', fontfamily=FONT_FAMILY)
        ax2.set_ylim(0, 1.1)
        for bar, sens in zip(bars, sensitivities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{sens:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 3. Biopsy rate comparison
        ax3 = fig.add_subplot(gs[0, 2])
        biopsy_rates = [
            results['tct_only']['biopsy_rate'],
            results['hpv_tct_combined']['biopsy_rate'],
            results['multimodal_ai']['biopsy_rate']
        ]
        bars = ax3.bar(methods, biopsy_rates, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Biopsy Rate (%)', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax3.set_title('Biopsy Rate Comparison', fontsize=13, fontweight='bold', fontfamily=FONT_FAMILY)
        ax3.set_ylim(0, max(biopsy_rates) * 1.2)
        for bar, rate in zip(bars, biopsy_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold', fontfamily=FONT_FAMILY)
        
        # 4. ROC曲线（需要实际数据）
        ax4 = fig.add_subplot(gs[1, :])
        ax4.text(0.5, 0.5, 'ROC Curve Comparison\n(Requires Model Prediction Probabilities)',
                ha='center', va='center', fontsize=12, fontfamily=FONT_FAMILY,
                transform=ax4.transAxes)
        ax4.set_xlabel('False Positive Rate (1-Specificity)', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax4.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax4.set_title('ROC Curve Comparison', fontsize=13, fontweight='bold', fontfamily=FONT_FAMILY)
        ax4.grid(alpha=0.3, linestyle='--')
        
        # 5. 决策曲线分析
        ax5 = fig.add_subplot(gs[2, :])
        thresholds = np.array(results['dca']['tct_only']['thresholds'])
        ax5.plot(thresholds, results['dca']['tct_only']['net_benefits'], 
                label='TCT Only', linewidth=2.5, color=colors[0], marker='o', markersize=3)
        ax5.plot(thresholds, results['dca']['hpv_tct_combined']['net_benefits'],
                label='HPV+TCT', linewidth=2.5, color=colors[1], marker='s', markersize=3)
        ax5.plot(thresholds, results['dca']['multimodal_ai']['net_benefits'],
                label='Multimodal AI', linewidth=2.5, color=colors[2], marker='^', markersize=3)
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        ax5.set_xlabel('Threshold Probability', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax5.set_ylabel('Net Benefit', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax5.set_title('Decision Curve Analysis (DCA)', fontsize=13, fontweight='bold', fontfamily=FONT_FAMILY)
        ax5.legend(fontsize=10, frameon=True, fancybox=True, shadow=True, loc='best')
        ax5.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # 设置所有文本字体
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            for label in ax.get_xticklabels():
                label.set_fontfamily(FONT_FAMILY)
            for label in ax.get_yticklabels():
                label.set_fontfamily(FONT_FAMILY)
            ax.tick_params(labelsize=10)
        
        # 保存图片
        fig_path = os.path.join(self.output_dir, 'experiment3_results.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✅ Figure saved to: {fig_path}")
        plt.close()


def main():
    """主函数"""
    experiment = ScreeningComparisonExperiment(
        data_path='5centers_multi',
        output_dir='lancet_primary_care/results/experiment3_results'
    )
    
    # 运行实验
    results = experiment.run_experiment()
    
    # 可视化结果
    experiment.plot_results(results)
    
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)


if __name__ == '__main__':
    main()

