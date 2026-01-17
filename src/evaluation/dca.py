#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
决策曲线分析 (Decision Curve Analysis, DCA)
用于评估临床实用性和最优决策阈值
Lancet期刊要求：必须包含DCA分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
from typing import Dict, Tuple, Optional
import json
import os

# 设置字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


class DecisionCurveAnalyzer:
    """决策曲线分析器"""
    
    def __init__(self, thresholds: Optional[np.ndarray] = None):
        """
        Args:
            thresholds: 决策阈值范围，默认0.01-0.99
        """
        if thresholds is None:
            self.thresholds = np.arange(0.01, 0.99, 0.01)
        else:
            self.thresholds = thresholds
    
    def calculate_net_benefit(self, y_true: np.ndarray, y_probs: np.ndarray, 
                              threshold: float) -> float:
        """
        计算净收益 (Net Benefit)
        
        Args:
            y_true: 真实标签 (0/1)
            y_probs: 预测概率
            threshold: 决策阈值
        
        Returns:
            净收益值
        """
        # 将概率转换为二元决策
        decisions = (y_probs >= threshold).astype(int)
        n = len(y_true)
        
        if n == 0:
            return 0.0
        
        # True positives, False positives, False negatives
        tp = np.sum((decisions == 1) & (y_true == 1))
        fp = np.sum((decisions == 1) & (y_true == 0))
        fn = np.sum((decisions == 0) & (y_true == 1))
        
        # Net Benefit = (TP - w*FP) / N
        # w = threshold / (1 - threshold)  (harm-to-benefit ratio)
        if threshold < 1.0 and threshold > 0:
            w = threshold / (1 - threshold)
            net_benefit = (tp - w * fp) / n
        else:
            net_benefit = 0.0
        
        return net_benefit
    
    def calculate_treat_all_benefit(self, y_true: np.ndarray) -> float:
        """计算"全部治疗"策略的净收益"""
        n = len(y_true)
        if n == 0:
            return 0.0
        
        tp = np.sum(y_true == 1)
        fp = np.sum(y_true == 0)
        
        # Treat all: 所有患者都治疗，w=1 (equal harm and benefit)
        w = 1.0
        net_benefit = (tp - w * fp) / n
        
        return net_benefit
    
    def calculate_treat_none_benefit(self) -> float:
        """计算"全部不治疗"策略的净收益（始终为0）"""
        return 0.0
    
    def analyze(self, y_true: np.ndarray, y_probs: np.ndarray, 
                model_name: str = "Model") -> pd.DataFrame:
        """
        完整的DCA分析
        
        Args:
            y_true: 真实标签
            y_probs: 预测概率
            model_name: 模型名称
        
        Returns:
            包含所有阈值和净收益的DataFrame
        """
        results = []
        
        # 计算模型在不同阈值下的净收益
        for threshold in self.thresholds:
            net_benefit = self.calculate_net_benefit(y_true, y_probs, threshold)
            results.append({
                'threshold': threshold,
                'net_benefit': net_benefit,
                'strategy': model_name
            })
        
        # 添加"全部治疗"策略
        treat_all_benefit = self.calculate_treat_all_benefit(y_true)
        for threshold in self.thresholds:
            results.append({
                'threshold': threshold,
                'net_benefit': treat_all_benefit,
                'strategy': 'Treat All'
            })
        
        # 添加"全部不治疗"策略
        treat_none_benefit = self.calculate_treat_none_benefit()
        for threshold in self.thresholds:
            results.append({
                'threshold': threshold,
                'net_benefit': treat_none_benefit,
                'strategy': 'Treat None'
            })
        
        return pd.DataFrame(results)
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
        """
        找到最优决策阈值（净收益最大）
        
        Returns:
            (optimal_threshold, max_net_benefit)
        """
        net_benefits = []
        for threshold in self.thresholds:
            net_benefit = self.calculate_net_benefit(y_true, y_probs, threshold)
            net_benefits.append(net_benefit)
        
        max_idx = np.argmax(net_benefits)
        optimal_threshold = self.thresholds[max_idx]
        max_net_benefit = net_benefits[max_idx]
        
        return optimal_threshold, max_net_benefit
    
    def plot_decision_curve(self, dca_results: pd.DataFrame, 
                           output_path: str, title: str = "Decision Curve Analysis"):
        """
        绘制决策曲线
        
        Args:
            dca_results: DCA分析结果DataFrame
            output_path: 输出路径
            title: 图表标题
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 为每个策略绘制曲线
        strategies = dca_results['strategy'].unique()
        colors = {'Model': '#2E86AB', 'Treat All': '#A23B72', 'Treat None': '#F18F01'}
        linestyles = {'Model': '-', 'Treat All': '--', 'Treat None': '--'}
        
        for strategy in strategies:
            strategy_data = dca_results[dca_results['strategy'] == strategy]
            ax.plot(strategy_data['threshold'], strategy_data['net_benefit'],
                   label=strategy, color=colors.get(strategy, '#000000'),
                   linestyle=linestyles.get(strategy, '-'), linewidth=2)
        
        ax.set_xlabel('Threshold Probability', fontsize=14, fontweight='bold')
        ax.set_ylabel('Net Benefit', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ 决策曲线已保存: {output_path}")


def load_predictions_from_metrics(metrics_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从metrics.json文件加载预测结果
    
    Args:
        metrics_file: metrics.json文件路径
    
    Returns:
        (y_true, y_probs) - 需要从模型重新预测或从保存的数据加载
    """
    # TODO: 实现从metrics.json或保存的预测结果加载
    # 目前需要从模型重新预测
    return None, None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='决策曲线分析 (DCA)')
    parser.add_argument('--metrics_file', type=str,
                       help='metrics.json文件路径（包含预测结果）')
    parser.add_argument('--y_true', type=str,
                       help='真实标签文件路径（CSV或npy）')
    parser.add_argument('--y_probs', type=str,
                       help='预测概率文件路径（CSV或npy）')
    parser.add_argument('--output_dir', default='analysis/dca_results',
                       help='输出目录')
    parser.add_argument('--model_name', default='SwinT',
                       help='模型名称')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    if args.y_true and args.y_probs:
        if args.y_true.endswith('.npy'):
            y_true = np.load(args.y_true)
        else:
            y_true = pd.read_csv(args.y_true).values.flatten()
        
        if args.y_probs.endswith('.npy'):
            y_probs = np.load(args.y_probs)
        else:
            y_probs = pd.read_csv(args.y_probs).values.flatten()
    else:
        print("⚠️  需要提供y_true和y_probs文件，或从模型重新预测")
        print("   建议：在训练脚本中保存所有预测概率")
        return
    
    # 执行DCA分析
    dca_analyzer = DecisionCurveAnalyzer()
    dca_results = dca_analyzer.analyze(y_true, y_probs, args.model_name)
    
    # 找到最优阈值
    optimal_threshold, max_net_benefit = dca_analyzer.find_optimal_threshold(y_true, y_probs)
    print(f"\n📊 DCA分析结果:")
    print(f"  最优阈值: {optimal_threshold:.4f}")
    print(f"  最大净收益: {max_net_benefit:.4f}")
    
    # 保存结果
    dca_results.to_csv(os.path.join(args.output_dir, 'dca_results.csv'), index=False)
    
    # 保存最优阈值
    optimal_info = {
        'optimal_threshold': float(optimal_threshold),
        'max_net_benefit': float(max_net_benefit)
    }
    with open(os.path.join(args.output_dir, 'optimal_threshold.json'), 'w') as f:
        json.dump(optimal_info, f, indent=2)
    
    # 绘制决策曲线
    output_plot = os.path.join(args.output_dir, 'decision_curve.png')
    dca_analyzer.plot_decision_curve(dca_results, output_plot, 
                                    f"Decision Curve Analysis - {args.model_name}")
    
    print(f"\n✅ DCA分析完成！结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()

