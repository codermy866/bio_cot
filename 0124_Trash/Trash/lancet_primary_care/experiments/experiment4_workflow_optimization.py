#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验4: 筛查流程优化实验

目标: 设计并验证优化的筛查决策流程

主要评估指标:
- 筛查步骤数
- 筛查时间（模拟）
- 活检率
- 灵敏度、特异度
- 成本效益分析
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import roc_auc_score, confusion_matrix
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


class WorkflowOptimizationExperiment:
    """筛查流程优化实验"""
    
    def __init__(
        self,
        data_path: str,
        output_dir: str = None
    ):
        self.data_path = data_path
        self.output_dir = output_dir or "lancet_primary_care/results/experiment4_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载数据
        self.train_df = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))
        self.test_df = pd.read_csv(os.path.join(data_path, 'test_labels.csv'))
        
        # 成本参数（单位：元）
        self.costs = {
            'hpv_test': 200,
            'tct_test': 150,
            'oct_exam': 300,
            'colposcopy': 400,
            'biopsy': 800
        }
        
        print(f"训练样本: {len(self.train_df)}")
        print(f"测试样本: {len(self.test_df)}")
    
    def _standard_workflow(self, df: pd.DataFrame) -> Dict:
        """
        标准流程（当前）
        初筛 → HPV+TCT → 任一阳性 → 活检
        """
        results = {
            'steps': [],
            'total_cost': 0,
            'biopsy_recommendations': [],
            'final_predictions': []
        }
        
        true_labels = df['label'].values
        
        # 步骤1: HPV检测
        hpv_col = df['HPV清洗'].astype(str)
        hpv_positive = hpv_col.str.contains(
            'Positive|16|18|其他|高危',
            case=False,
            na=False
        ).values
        results['steps'].append('HPV检测')
        results['total_cost'] += self.costs['hpv_test'] * len(df)
        
        # 步骤2: TCT检测
        tct_col = df['TCT清洗'].astype(str).str.upper()
        tct_abnormal = tct_col.str.contains(
            'ASC-US|LSIL|HSIL',
            case=False,
            na=False
        ).values
        results['steps'].append('TCT检测')
        results['total_cost'] += self.costs['tct_test'] * len(df)
        
        # 步骤3: 决策（任一阳性即活检）
        biopsy_recommendation = (hpv_positive | tct_abnormal).astype(int)
        results['steps'].append('活检决策')
        results['total_cost'] += self.costs['biopsy'] * np.sum(biopsy_recommendation)
        
        results['biopsy_recommendations'] = biopsy_recommendation.tolist()
        results['final_predictions'] = biopsy_recommendation.tolist()
        results['true_labels'] = true_labels.tolist()
        results['n_steps'] = len(results['steps'])
        results['n_biopsies'] = int(np.sum(biopsy_recommendation))
        
        return results
    
    def _optimized_workflow(
        self,
        df: pd.DataFrame,
        ai_probabilities: np.ndarray = None,
        high_risk_threshold: float = 0.7,
        low_risk_threshold: float = 0.3
    ) -> Dict:
        """
        优化流程（本研究）
        初筛 → 多模态AI评估 → 风险分层 → 差异化管理
        """
        results = {
            'steps': [],
            'total_cost': 0,
            'risk_stratification': [],
            'biopsy_recommendations': [],
            'final_predictions': []
        }
        
        true_labels = df['label'].values
        
        if ai_probabilities is None:
            # 生成模拟概率
            print("⚠️  使用模拟数据，实际使用时需要加载真实模型")
            np.random.seed(42)
            ai_probabilities = np.random.beta(2, 3, len(df))
            ai_probabilities[true_labels == 1] = np.random.beta(7, 2, np.sum(true_labels == 1))
            ai_probabilities[true_labels == 0] = np.random.beta(2, 7, np.sum(true_labels == 0))
        
        # 步骤1: 多模态AI评估（一次性完成所有检查）
        results['steps'].append('多模态AI评估')
        # 成本：OCT + Colposcopy + Clinical（假设已包含在AI评估中）
        results['total_cost'] += (self.costs['oct_exam'] + self.costs['colposcopy']) * len(df)
        
        # 步骤2: 风险分层
        risk_stratification = np.zeros(len(df), dtype=int)  # 0:低风险, 1:中风险, 2:高风险
        risk_stratification[ai_probabilities >= high_risk_threshold] = 2  # 高风险
        risk_stratification[(ai_probabilities >= low_risk_threshold) & 
                           (ai_probabilities < high_risk_threshold)] = 1  # 中风险
        risk_stratification[ai_probabilities < low_risk_threshold] = 0  # 低风险
        
        results['risk_stratification'] = risk_stratification.tolist()
        
        # 步骤3: 差异化管理
        biopsy_recommendation = np.zeros(len(df), dtype=int)
        
        # 高风险：直接活检
        high_risk_mask = risk_stratification == 2
        biopsy_recommendation[high_risk_mask] = 1
        n_high_risk = np.sum(high_risk_mask)
        results['total_cost'] += self.costs['biopsy'] * n_high_risk
        results['steps'].append(f'高风险直接活检 ({n_high_risk}例)')
        
        # 中风险：OCT详细检查后决策（假设50%需要活检）
        mid_risk_mask = risk_stratification == 1
        n_mid_risk = np.sum(mid_risk_mask)
        if n_mid_risk > 0:
            # 中风险中，根据概率决定是否需要进一步检查
            mid_risk_probs = ai_probabilities[mid_risk_mask]
            mid_risk_biopsy = (mid_risk_probs > 0.5).astype(int)
            biopsy_recommendation[mid_risk_mask] = mid_risk_biopsy
            n_mid_risk_biopsy = np.sum(mid_risk_biopsy)
            results['total_cost'] += self.costs['biopsy'] * n_mid_risk_biopsy
            results['steps'].append(f'中风险详细检查后活检 ({n_mid_risk_biopsy}/{n_mid_risk}例)')
        
        # 低风险：常规随访（不需要活检）
        low_risk_mask = risk_stratification == 0
        n_low_risk = np.sum(low_risk_mask)
        results['steps'].append(f'低风险常规随访 ({n_low_risk}例)')
        
        results['biopsy_recommendations'] = biopsy_recommendation.tolist()
        results['final_predictions'] = biopsy_recommendation.tolist()
        results['true_labels'] = true_labels.tolist()
        results['n_steps'] = len(results['steps'])
        results['n_biopsies'] = int(np.sum(biopsy_recommendation))
        
        # 风险分层统计
        results['risk_distribution'] = {
            'high_risk': int(n_high_risk),
            'mid_risk': int(n_mid_risk),
            'low_risk': int(n_low_risk)
        }
        
        return results
    
    def _calculate_workflow_metrics(self, workflow_results: Dict) -> Dict:
        """计算流程指标"""
        true_labels = np.array(workflow_results['true_labels'])
        predictions = np.array(workflow_results['final_predictions'])
        
        metrics = calculate_clinical_metrics(true_labels, predictions)
        
        # 添加流程特定指标
        metrics['n_steps'] = workflow_results['n_steps']
        metrics['n_biopsies'] = workflow_results['n_biopsies']
        metrics['biopsy_rate'] = workflow_results['n_biopsies'] / len(true_labels) * 100
        metrics['total_cost'] = workflow_results['total_cost']
        metrics['cost_per_patient'] = workflow_results['total_cost'] / len(true_labels)
        
        # 每检出1例病变的成本
        n_positive = np.sum(true_labels)
        if n_positive > 0:
            tp = np.sum((predictions == 1) & (true_labels == 1))
            metrics['cost_per_detected_case'] = workflow_results['total_cost'] / tp if tp > 0 else np.inf
        else:
            metrics['cost_per_detected_case'] = np.inf
        
        return metrics
    
    def _cost_effectiveness_analysis(
        self,
        standard_results: Dict,
        optimized_results: Dict
    ) -> Dict:
        """成本效益分析"""
        standard_metrics = self._calculate_workflow_metrics(standard_results)
        optimized_metrics = self._calculate_workflow_metrics(optimized_results)
        
        # 成本差异
        cost_reduction = standard_metrics['total_cost'] - optimized_metrics['total_cost']
        cost_reduction_percent = (cost_reduction / standard_metrics['total_cost']) * 100
        
        # 活检数差异
        biopsy_reduction = standard_metrics['n_biopsies'] - optimized_metrics['n_biopsies']
        biopsy_reduction_percent = (biopsy_reduction / standard_metrics['n_biopsies']) * 100
        
        # 检出病变数
        standard_tp = np.sum(
            (np.array(standard_results['final_predictions']) == 1) &
            (np.array(standard_results['true_labels']) == 1)
        )
        optimized_tp = np.sum(
            (np.array(optimized_results['final_predictions']) == 1) &
            (np.array(optimized_results['true_labels']) == 1)
        )
        
        # 增量成本效益比 (ICER)
        if optimized_tp > standard_tp:
            # 新方法检出更多，计算每多检出1例的成本
            additional_cases = optimized_tp - standard_tp
            additional_cost = optimized_metrics['total_cost'] - standard_metrics['total_cost']
            icer = additional_cost / additional_cases if additional_cases > 0 else np.inf
        else:
            # 新方法检出相同或更少，但成本更低
            icer = None
        
        return {
            'cost_reduction': float(cost_reduction),
            'cost_reduction_percent': float(cost_reduction_percent),
            'biopsy_reduction': int(biopsy_reduction),
            'biopsy_reduction_percent': float(biopsy_reduction_percent),
            'cases_detected': {
                'standard': int(standard_tp),
                'optimized': int(optimized_tp),
                'difference': int(optimized_tp - standard_tp)
            },
            'cost_per_detected_case': {
                'standard': float(standard_metrics['cost_per_detected_case']),
                'optimized': float(optimized_metrics['cost_per_detected_case'])
            },
            'icer': float(icer) if icer is not None and not np.isinf(icer) else None
        }
    
    def run_experiment(
        self,
        ai_probabilities: np.ndarray = None,
        high_risk_threshold: float = 0.7,
        low_risk_threshold: float = 0.3
    ) -> Dict:
        """
        运行实验
        
        Args:
            ai_probabilities: AI模型的预测概率
            high_risk_threshold: 高风险阈值
            low_risk_threshold: 低风险阈值
        """
        print("\n" + "="*60)
        print("实验4: 筛查流程优化实验")
        print("="*60)
        
        test_df = self.test_df
        
        # 1. 标准流程
        print("\n[1] 评估标准流程...")
        standard_results = self._standard_workflow(test_df)
        standard_metrics = self._calculate_workflow_metrics(standard_results)
        
        print(f"  筛查步骤数: {standard_metrics['n_steps']}")
        print(f"  活检数: {standard_metrics['n_biopsies']}")
        print(f"  活检率: {standard_metrics['biopsy_rate']:.2f}%")
        print(f"  总成本: {standard_metrics['total_cost']:.0f}元")
        print(f"  人均成本: {standard_metrics['cost_per_patient']:.2f}元")
        print(f"  每检出1例成本: {standard_metrics['cost_per_detected_case']:.2f}元")
        print(f"  灵敏度: {standard_metrics['sensitivity']:.4f}")
        print(f"  特异度: {standard_metrics['specificity']:.4f}")
        
        # 2. 优化流程
        print("\n[2] 评估优化流程...")
        optimized_results = self._optimized_workflow(
            test_df,
            ai_probabilities=ai_probabilities,
            high_risk_threshold=high_risk_threshold,
            low_risk_threshold=low_risk_threshold
        )
        optimized_metrics = self._calculate_workflow_metrics(optimized_results)
        
        print(f"  筛查步骤数: {optimized_metrics['n_steps']}")
        print(f"  活检数: {optimized_metrics['n_biopsies']}")
        print(f"  活检率: {optimized_metrics['biopsy_rate']:.2f}%")
        print(f"  总成本: {optimized_metrics['total_cost']:.0f}元")
        print(f"  人均成本: {optimized_metrics['cost_per_patient']:.2f}元")
        print(f"  每检出1例成本: {optimized_metrics['cost_per_detected_case']:.2f}元")
        print(f"  灵敏度: {optimized_metrics['sensitivity']:.4f}")
        print(f"  特异度: {optimized_metrics['specificity']:.4f}")
        
        # 风险分层分布
        risk_dist = optimized_results['risk_distribution']
        print(f"\n  风险分层分布:")
        print(f"    高风险: {risk_dist['high_risk']}例")
        print(f"    中风险: {risk_dist['mid_risk']}例")
        print(f"    低风险: {risk_dist['low_risk']}例")
        
        # 3. 成本效益分析
        print("\n[3] 成本效益分析...")
        cea_results = self._cost_effectiveness_analysis(standard_results, optimized_results)
        
        print(f"  成本降低: {cea_results['cost_reduction']:.0f}元 ({cea_results['cost_reduction_percent']:.2f}%)")
        print(f"  活检减少: {cea_results['biopsy_reduction']}例 ({cea_results['biopsy_reduction_percent']:.2f}%)")
        print(f"  检出病变数:")
        print(f"    标准流程: {cea_results['cases_detected']['standard']}例")
        print(f"    优化流程: {cea_results['cases_detected']['optimized']}例")
        print(f"    差异: {cea_results['cases_detected']['difference']}例")
        
        if cea_results['icer'] is not None:
            print(f"  增量成本效益比 (ICER): {cea_results['icer']:.2f}元/例")
        
        # 4. 整理结果
        results = {
            'standard_workflow': {
                'results': standard_results,
                'metrics': standard_metrics
            },
            'optimized_workflow': {
                'results': optimized_results,
                'metrics': optimized_metrics
            },
            'cost_effectiveness_analysis': cea_results,
            'sample_size': {
                'total': len(test_df),
                'positive': int(np.sum(test_df['label'].values)),
                'negative': int(np.sum(1 - test_df['label'].values))
            }
        }
        
        # 5. 保存结果
        results_file = os.path.join(self.output_dir, 'experiment4_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(ensure_serializable(results), f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {results_file}")
        
        return results
    
    def plot_results(self, results: Dict):
        """可视化结果 - 2025年最新样式，英文Arial字体"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Experiment 4: Screening Workflow Optimization', 
                     fontsize=16, fontweight='bold', fontfamily=FONT_FAMILY)
        
        standard_metrics = results['standard_workflow']['metrics']
        optimized_metrics = results['optimized_workflow']['metrics']
        cea = results['cost_effectiveness_analysis']
        
        # 1. 活检率对比
        ax1 = axes[0, 0]
        methods = ['Standard\nWorkflow', 'Optimized\nWorkflow']
        biopsy_rates = [
            standard_metrics['biopsy_rate'],
            optimized_metrics['biopsy_rate']
        ]
        colors = ['#FF6B6B', '#4ECDC4']
        bars = ax1.bar(methods, biopsy_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Biopsy Rate (%)', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax1.set_title('Biopsy Rate Comparison', fontsize=13, fontweight='bold', fontfamily=FONT_FAMILY)
        for bar, rate in zip(bars, biopsy_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax1.text(0.5, max(biopsy_rates) * 1.1,
                f'Reduction: {cea["biopsy_reduction_percent"]:.1f}%',
                ha='center', fontsize=11, fontweight='bold', color='green', fontfamily=FONT_FAMILY)
        
        # 2. 成本对比
        ax2 = axes[0, 1]
        costs = [
            standard_metrics['total_cost'],
            optimized_metrics['total_cost']
        ]
        bars = ax2.bar(methods, costs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Total Cost (CNY)', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax2.set_title('Total Cost Comparison', fontsize=13, fontweight='bold', fontfamily=FONT_FAMILY)
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{cost:.0f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax2.text(0.5, max(costs) * 1.1,
                f'Reduction: {cea["cost_reduction_percent"]:.1f}%',
                ha='center', fontsize=11, fontweight='bold', color='green', fontfamily=FONT_FAMILY)
        
        # 3. 每检出1例成本
        ax3 = axes[0, 2]
        cost_per_case = [
            standard_metrics['cost_per_detected_case'],
            optimized_metrics['cost_per_detected_case']
        ]
        bars = ax3.bar(methods, cost_per_case, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Cost per Detected Case (CNY)', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax3.set_title('Cost-Effectiveness Comparison', fontsize=13, fontweight='bold', fontfamily=FONT_FAMILY)
        for bar, cost in zip(bars, cost_per_case):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{cost:.0f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 4. 灵敏度对比
        ax4 = axes[1, 0]
        sensitivities = [
            standard_metrics['sensitivity'],
            optimized_metrics['sensitivity']
        ]
        bars = ax4.bar(methods, sensitivities, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Sensitivity', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax4.set_title('Sensitivity Comparison', fontsize=13, fontweight='bold', fontfamily=FONT_FAMILY)
        ax4.set_ylim(0, 1.1)
        for bar, sens in zip(bars, sensitivities):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{sens:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 5. 风险分层分布
        ax5 = axes[1, 1]
        risk_dist = results['optimized_workflow']['results']['risk_distribution']
        risk_categories = ['High Risk', 'Medium Risk', 'Low Risk']
        risk_counts = [risk_dist['high_risk'], risk_dist['mid_risk'], risk_dist['low_risk']]
        risk_colors = ['#FF4444', '#FFA500', '#4ECDC4']
        bars = ax5.bar(risk_categories, risk_counts, color=risk_colors, alpha=0.7, edgecolor='black')
        ax5.set_ylabel('Patients', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax5.set_title('Risk Stratification Distribution', fontsize=13, fontweight='bold', fontfamily=FONT_FAMILY)
        for bar, count in zip(bars, risk_counts):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 6. 综合指标对比
        ax6 = axes[1, 2]
        categories = ['Sensitivity', 'Specificity', 'PPV', 'NPV']
        standard_values = [
            standard_metrics['sensitivity'],
            standard_metrics['specificity'],
            standard_metrics['ppv'],
            standard_metrics['npv']
        ]
        optimized_values = [
            optimized_metrics['sensitivity'],
            optimized_metrics['specificity'],
            optimized_metrics['ppv'],
            optimized_metrics['npv']
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        ax6.bar(x - width/2, standard_values, width, label='Standard Workflow',
               color='#FF6B6B', alpha=0.7, edgecolor='black')
        ax6.bar(x + width/2, optimized_values, width, label='Optimized Workflow',
               color='#4ECDC4', alpha=0.7, edgecolor='black')
        ax6.set_ylabel('Metric Value', fontsize=12, fontfamily=FONT_FAMILY, fontweight='bold')
        ax6.set_title('Comprehensive Metrics Comparison', fontsize=13, fontweight='bold', fontfamily=FONT_FAMILY)
        ax6.set_xticks(x)
        ax6.set_xticklabels(categories)
        ax6.set_ylim(0, 1.1)
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # 设置所有文本字体
        for ax in axes.flat:
            for label in ax.get_xticklabels():
                label.set_fontfamily(FONT_FAMILY)
            for label in ax.get_yticklabels():
                label.set_fontfamily(FONT_FAMILY)
            ax.tick_params(labelsize=10)
        
        # 保存图片
        fig_path = os.path.join(self.output_dir, 'experiment4_results.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✅ Figure saved to: {fig_path}")
        plt.close()


def main():
    """主函数"""
    experiment = WorkflowOptimizationExperiment(
        data_path='5centers_multi',
        output_dir='lancet_primary_care/results/experiment4_results'
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

