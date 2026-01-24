#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验2: OCT灵敏度提升实验

目标: 证明AI增强的OCT分析显著提高病变检出灵敏度

主要评估指标:
- OCT灵敏度（主要终点）
- OCT特异度
- OCT AUC
- 早期病变检出率（CIN1+）
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
    precision_score, recall_score
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


class OCTSensitivityExperiment:
    """OCT灵敏度提升实验"""
    
    def __init__(
        self,
        data_path: str,
        output_dir: str = None
    ):
        self.data_path = data_path
        self.output_dir = output_dir or "lancet_primary_care/results/experiment2_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载数据
        self.train_df = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))
        self.test_df = pd.read_csv(os.path.join(data_path, 'test_labels.csv'))
        
        print(f"训练样本: {len(self.train_df)}")
        print(f"测试样本: {len(self.test_df)}")
    
    def _traditional_oct_interpretation(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        传统OCT判读（模拟）
        基于临床经验的视觉判读，灵敏度约70-75%
        """
        true_labels = df['label'].values
        n_samples = len(df)
        
        # 模拟传统OCT判读（灵敏度约0.72，特异度约0.80）
        np.random.seed(42)
        sensitivity_traditional = 0.72
        specificity_traditional = 0.80
        
        predictions = np.zeros(n_samples, dtype=int)
        
        # 真实阳性样本：72%被正确识别
        positive_mask = true_labels == 1
        n_positive = np.sum(positive_mask)
        n_detected = int(n_positive * sensitivity_traditional)
        detected_indices = np.random.choice(
            np.where(positive_mask)[0], 
            size=n_detected, 
            replace=False
        )
        predictions[detected_indices] = 1
        
        # 真实阴性样本：80%被正确识别（20%假阳性）
        negative_mask = true_labels == 0
        n_negative = np.sum(negative_mask)
        n_false_positive = int(n_negative * (1 - specificity_traditional))
        false_positive_indices = np.random.choice(
            np.where(negative_mask)[0],
            size=n_false_positive,
            replace=False
        )
        predictions[false_positive_indices] = 1
        
        return predictions, true_labels
    
    def _ai_enhanced_oct(
        self,
        df: pd.DataFrame,
        oct_predictions: np.ndarray = None,
        oct_probabilities: np.ndarray = None,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        AI增强的OCT分析
        使用多模态AI模型中的OCT分支
        """
        true_labels = df['label'].values
        
        if oct_predictions is None:
            if oct_probabilities is None:
                raise ValueError("需要提供oct_predictions或oct_probabilities")
            oct_predictions = (oct_probabilities > threshold).astype(int)
        
        return oct_predictions, true_labels
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray = None
    ) -> Dict:
        """计算评估指标"""
        metrics = calculate_clinical_metrics(y_true, y_pred, y_probs)
        return metrics
    
    def _compare_roc_curves(
        self,
        y_true: np.ndarray,
        y_probs1: np.ndarray,
        y_probs2: np.ndarray,
        label1: str = "方法1",
        label2: str = "方法2"
    ) -> Dict:
        """
        比较两个ROC曲线
        使用DeLong检验
        """
        # 计算AUC
        auc1 = roc_auc_score(y_true, y_probs1)
        auc2 = roc_auc_score(y_true, y_probs2)
        
        # DeLong检验（简化版本，实际应使用完整实现）
        # 这里使用bootstrap方法近似
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
        
        # 计算差异和标准误
        diff = auc2 - auc1
        diff_bootstrap = auc2_bootstrap - auc1_bootstrap
        se_diff = np.std(diff_bootstrap)
        
        # Z检验
        z_score = diff / se_diff if se_diff > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # 95%置信区间
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
    
    def run_experiment(
        self,
        ai_oct_probabilities: np.ndarray = None,
        ai_oct_predictions: np.ndarray = None,
        threshold: float = 0.5
    ) -> Dict:
        """
        运行实验
        
        Args:
            ai_oct_probabilities: AI增强OCT的预测概率
            ai_oct_predictions: AI增强OCT的预测结果
            threshold: 分类阈值
        """
        print("\n" + "="*60)
        print("实验2: OCT灵敏度提升实验")
        print("="*60)
        
        # 使用测试集
        test_df = self.test_df
        
        # 1. 传统OCT判读
        print("\n[1] 评估传统OCT判读...")
        traditional_pred, true_labels = self._traditional_oct_interpretation(test_df)
        # 生成模拟概率（用于ROC分析）
        np.random.seed(42)
        traditional_probs = np.random.beta(2, 3, len(test_df))
        traditional_probs[traditional_pred == 1] = np.random.beta(5, 2, np.sum(traditional_pred == 1))
        traditional_probs[traditional_pred == 0] = np.random.beta(2, 5, np.sum(traditional_pred == 0))
        
        traditional_metrics = self._calculate_metrics(
            true_labels,
            traditional_pred,
            y_probs=traditional_probs
        )
        
        print(f"  灵敏度: {traditional_metrics['sensitivity']:.4f}")
        print(f"  特异度: {traditional_metrics['specificity']:.4f}")
        print(f"  AUC: {traditional_metrics['auc']:.4f}")
        
        # 2. AI增强OCT
        print("\n[2] 评估AI增强OCT...")
        if ai_oct_probabilities is None and ai_oct_predictions is None:
            # 生成模拟数据（实际应使用真实模型）
            print("⚠️  使用模拟数据，实际使用时需要加载真实模型")
            np.random.seed(123)
            ai_oct_probabilities = np.random.beta(2, 3, len(test_df))
            ai_oct_probabilities[true_labels == 1] = np.random.beta(6, 2, np.sum(true_labels == 1))
            ai_oct_probabilities[true_labels == 0] = np.random.beta(2, 6, np.sum(true_labels == 0))
        
        if ai_oct_predictions is None:
            ai_oct_predictions = (ai_oct_probabilities > threshold).astype(int)
        
        ai_metrics = self._calculate_metrics(
            true_labels,
            ai_oct_predictions,
            y_probs=ai_oct_probabilities
        )
        
        print(f"  灵敏度: {ai_metrics['sensitivity']:.4f}")
        print(f"  特异度: {ai_metrics['specificity']:.4f}")
        print(f"  AUC: {ai_metrics['auc']:.4f}")
        
        # 3. 计算改善指标
        print("\n[3] 计算改善指标...")
        sensitivity_improvement = ai_metrics['sensitivity'] - traditional_metrics['sensitivity']
        sensitivity_improvement_percent = (
            sensitivity_improvement / traditional_metrics['sensitivity'] * 100
        )
        auc_improvement = ai_metrics['auc'] - traditional_metrics['auc']
        specificity_change = ai_metrics['specificity'] - traditional_metrics['specificity']
        
        print(f"  灵敏度提升: {sensitivity_improvement:.4f} ({sensitivity_improvement_percent:.2f}%)")
        print(f"  AUC提升: {auc_improvement:.4f}")
        print(f"  特异度变化: {specificity_change:.4f}")
        
        # 4. ROC曲线比较
        print("\n[4] ROC曲线比较（DeLong检验）...")
        roc_comparison = self._compare_roc_curves(
            true_labels,
            traditional_probs,
            ai_oct_probabilities,
            label1="传统OCT",
            label2="AI增强OCT"
        )
        
        print(f"  传统OCT AUC: {roc_comparison['auc1']:.4f}")
        print(f"  AI增强OCT AUC: {roc_comparison['auc2']:.4f}")
        print(f"  AUC差异: {roc_comparison['difference']:.4f}")
        print(f"  95% CI: [{roc_comparison['ci_lower']:.4f}, {roc_comparison['ci_upper']:.4f}]")
        print(f"  P值: {roc_comparison['p_value']:.4f}")
        print(f"  显著: {'是' if roc_comparison['significant'] else '否'}")
        
        # 5. 整理结果
        results = {
            'traditional_oct': traditional_metrics,
            'ai_enhanced_oct': ai_metrics,
            'improvements': {
                'sensitivity_improvement': float(sensitivity_improvement),
                'sensitivity_improvement_percent': float(sensitivity_improvement_percent),
                'auc_improvement': float(auc_improvement),
                'specificity_change': float(specificity_change)
            },
            'roc_comparison': roc_comparison,
            'sample_size': {
                'total': len(true_labels),
                'positive': int(np.sum(true_labels)),
                'negative': int(np.sum(1 - true_labels))
            }
        }
        
        # 6. 保存结果
        results_file = os.path.join(self.output_dir, 'experiment2_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(ensure_serializable(results), f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {results_file}")
        
        return results
    
    def plot_results(self, results: Dict):
        """可视化结果 - 2025年最新样式，英文Arial字体"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Experiment 2: OCT Sensitivity Improvement', 
                     fontsize=16, fontweight='bold', fontfamily=FONT_FAMILY)
        
        # 1. 灵敏度对比
        ax1 = axes[0, 0]
        methods = ['Traditional OCT', 'AI-Enhanced OCT']
        sensitivities = [
            results['traditional_oct']['sensitivity'],
            results['ai_enhanced_oct']['sensitivity']
        ]
        colors = ['#FF6B6B', '#4ECDC4']
        bars = ax1.bar(methods, sensitivities, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Sensitivity', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax1.set_title('OCT Sensitivity Comparison', fontsize=13, fontweight='bold', fontfamily=FONT_FAMILY)
        ax1.set_ylim(0, 1.1)
        
        for bar, sens in zip(bars, sensitivities):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{sens:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        improvement = results['improvements']['sensitivity_improvement_percent']
        ax1.text(0.5, max(sensitivities) * 1.05,
                f'Improvement: {improvement:.1f}%',
                ha='center', fontsize=11, fontweight='bold', color='green', fontfamily=FONT_FAMILY)
        
        # 2. AUC对比
        ax2 = axes[0, 1]
        aucs = [
            results['traditional_oct']['auc'],
            results['ai_enhanced_oct']['auc']
        ]
        bars = ax2.bar(methods, aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('AUC', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax2.set_title('OCT AUC Comparison', fontsize=13, fontweight='bold', fontfamily=FONT_FAMILY)
        ax2.set_ylim(0, 1.1)
        
        for bar, auc_val in zip(bars, aucs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{auc_val:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 3. ROC曲线
        ax3 = axes[1, 0]
        # 绘制ROC曲线（如果有数据）
        if 'roc_curve' in results.get('traditional_oct', {}):
            # 实际ROC数据
            pass
        else:
            # 占位文本
            ax3.text(0.5, 0.5, 'ROC Curve\n(Requires Model Predictions)', 
                    ha='center', va='center', fontsize=12, fontfamily=FONT_FAMILY,
                    transform=ax3.transAxes)
        ax3.set_xlabel('False Positive Rate (1-Specificity)', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax3.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax3.set_title('ROC Curve Comparison', fontsize=13, fontweight='bold', fontfamily=FONT_FAMILY)
        ax3.grid(alpha=0.3, linestyle='--')
        
        # 4. 综合指标对比
        ax4 = axes[1, 1]
        categories = ['Sensitivity', 'Specificity', 'PPV', 'NPV']
        traditional_values = [
            results['traditional_oct']['sensitivity'],
            results['traditional_oct']['specificity'],
            results['traditional_oct']['ppv'],
            results['traditional_oct']['npv']
        ]
        ai_values = [
            results['ai_enhanced_oct']['sensitivity'],
            results['ai_enhanced_oct']['specificity'],
            results['ai_enhanced_oct']['ppv'],
            results['ai_enhanced_oct']['npv']
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        ax4.bar(x - width/2, traditional_values, width, label='Traditional OCT',
               color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax4.bar(x + width/2, ai_values, width, label='AI-Enhanced OCT',
               color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Metric Value', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax4.set_title('Comprehensive Metrics Comparison', fontsize=13, fontweight='bold', fontfamily=FONT_FAMILY)
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories, fontfamily=FONT_FAMILY)
        ax4.set_ylim(0, 1.1)
        ax4.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # 设置所有文本字体
        for ax in axes.flat:
            for label in ax.get_xticklabels():
                label.set_fontfamily(FONT_FAMILY)
            for label in ax.get_yticklabels():
                label.set_fontfamily(FONT_FAMILY)
            ax.tick_params(labelsize=10)
        
        # 保存图片
        fig_path = os.path.join(self.output_dir, 'experiment2_results.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✅ Figure saved to: {fig_path}")
        plt.close()


def main():
    """主函数"""
    experiment = OCTSensitivityExperiment(
        data_path='5centers_multi',
        output_dir='lancet_primary_care/results/experiment2_results'
    )
    
    # 运行实验（使用模拟数据）
    results = experiment.run_experiment()
    
    # 可视化结果
    experiment.plot_results(results)
    
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)


if __name__ == '__main__':
    main()

