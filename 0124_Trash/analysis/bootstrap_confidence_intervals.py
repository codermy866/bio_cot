#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap置信区间计算
用于Lancet期刊发表要求：所有指标必须报告95%置信区间
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, 
    precision_score, recall_score, confusion_matrix
)
from tqdm import tqdm
import json
import os
from typing import Dict, Tuple, List, Callable


class BootstrapCI:
    """Bootstrap置信区间计算器"""
    
    def __init__(self, n_bootstrap=2000, confidence=0.95, random_state=42):
        """
        Args:
            n_bootstrap: Bootstrap重采样次数
            confidence: 置信水平 (0.95 for 95% CI)
            random_state: 随机种子
        """
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.alpha = 1 - confidence
        self.random_state = random_state
        np.random.seed(random_state)
    
    def calculate_ci(self, metric_func: Callable, y_true: np.ndarray, 
                     y_pred: np.ndarray = None, y_probs: np.ndarray = None) -> Tuple[float, float, float]:
        """
        计算指标的Bootstrap置信区间
        
        Args:
            metric_func: 指标计算函数
            y_true: 真实标签
            y_pred: 预测标签 (用于accuracy, precision, recall, f1)
            y_probs: 预测概率 (用于AUC)
        
        Returns:
            (mean, lower_bound, upper_bound)
        """
        metric_values = []
        
        for _ in tqdm(range(self.n_bootstrap), desc="Bootstrap sampling"):
            # 重采样
            indices = np.random.choice(
                len(y_true),
                size=len(y_true),
                replace=True
            )
            
            y_true_boot = y_true[indices]
            
            try:
                if y_probs is not None:
                    y_probs_boot = y_probs[indices]
                    metric = metric_func(y_true_boot, y_probs_boot)
                elif y_pred is not None:
                    y_pred_boot = y_pred[indices]
                    metric = metric_func(y_true_boot, y_pred_boot)
                else:
                    raise ValueError("Either y_pred or y_probs must be provided")
                
                if not np.isnan(metric) and not np.isinf(metric):
                    metric_values.append(metric)
            except Exception as e:
                continue
        
        if not metric_values:
            return np.nan, np.nan, np.nan
        
        metric_values = np.array(metric_values)
        mean = np.mean(metric_values)
        lower = np.percentile(metric_values, 100 * self.alpha / 2)
        upper = np.percentile(metric_values, 100 * (1 - self.alpha / 2))
        
        return mean, lower, upper
    
    def calculate_all_metrics_ci(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_probs: np.ndarray) -> Dict:
        """
        计算所有指标的置信区间
        
        Returns:
            包含所有指标及其CI的字典
        """
        results = {}
        
        # AUC
        def auc_func(y_true, y_probs):
            try:
                return roc_auc_score(y_true, y_probs)
            except:
                return np.nan
        
        auc_mean, auc_low, auc_high = self.calculate_ci(auc_func, y_true, y_probs=y_probs)
        results['auc'] = {
            'mean': float(auc_mean),
            'ci_lower': float(auc_low),
            'ci_upper': float(auc_high),
            'formatted': f"{auc_mean:.4f} ({auc_low:.4f}-{auc_high:.4f})"
        }
        
        # Accuracy
        def acc_func(y_true, y_pred):
            return accuracy_score(y_true, y_pred)
        
        acc_mean, acc_low, acc_high = self.calculate_ci(acc_func, y_true, y_pred=y_pred)
        results['accuracy'] = {
            'mean': float(acc_mean),
            'ci_lower': float(acc_low),
            'ci_upper': float(acc_high),
            'formatted': f"{acc_mean:.4f} ({acc_low:.4f}-{acc_high:.4f})"
        }
        
        # F1-Score
        def f1_func(y_true, y_pred):
            return f1_score(y_true, y_pred, zero_division=0)
        
        f1_mean, f1_low, f1_high = self.calculate_ci(f1_func, y_true, y_pred=y_pred)
        results['f1_score'] = {
            'mean': float(f1_mean),
            'ci_lower': float(f1_low),
            'ci_upper': float(f1_high),
            'formatted': f"{f1_mean:.4f} ({f1_low:.4f}-{f1_high:.4f})"
        }
        
        # Sensitivity (Recall)
        def sens_func(y_true, y_pred):
            return recall_score(y_true, y_pred, zero_division=0)
        
        sens_mean, sens_low, sens_high = self.calculate_ci(sens_func, y_true, y_pred=y_pred)
        results['sensitivity'] = {
            'mean': float(sens_mean),
            'ci_lower': float(sens_low),
            'ci_upper': float(sens_high),
            'formatted': f"{sens_mean:.4f} ({sens_low:.4f}-{sens_high:.4f})"
        }
        
        # Specificity
        def spec_func(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        spec_mean, spec_low, spec_high = self.calculate_ci(spec_func, y_true, y_pred=y_pred)
        results['specificity'] = {
            'mean': float(spec_mean),
            'ci_lower': float(spec_low),
            'ci_upper': float(spec_high),
            'formatted': f"{spec_mean:.4f} ({spec_low:.4f}-{spec_high:.4f})"
        }
        
        # Precision (PPV)
        def prec_func(y_true, y_pred):
            return precision_score(y_true, y_pred, zero_division=0)
        
        prec_mean, prec_low, prec_high = self.calculate_ci(prec_func, y_true, y_pred=y_pred)
        results['precision'] = {
            'mean': float(prec_mean),
            'ci_lower': float(prec_low),
            'ci_upper': float(prec_high),
            'formatted': f"{prec_mean:.4f} ({prec_low:.4f}-{prec_high:.4f})"
        }
        
        # NPV
        def npv_func(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        npv_mean, npv_low, npv_high = self.calculate_ci(npv_func, y_true, y_pred=y_pred)
        results['npv'] = {
            'mean': float(npv_mean),
            'ci_lower': float(npv_low),
            'ci_upper': float(npv_high),
            'formatted': f"{npv_mean:.4f} ({npv_low:.4f}-{npv_high:.4f})"
        }
        
        return results


def load_predictions_from_results(result_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从训练结果目录加载预测结果
    
    Args:
        result_dir: 结果目录路径 (包含metrics.json或training_history.json)
    
    Returns:
        (y_true, y_pred, y_probs)
    """
    import glob
    
    # 尝试从metrics.json加载
    metrics_file = os.path.join(result_dir, 'metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # 从ROC曲线重建预测概率
        if 'val' in metrics and 'roc_curve' in metrics['val']:
            roc_data = metrics['val']['roc_curve']
            # 这里需要从ROC曲线重建，实际应该保存原始预测
            # 暂时返回None，需要从模型重新预测
            pass
    
    # 如果无法加载，返回None，需要重新预测
    return None, None, None


def main():
    """主函数：为所有模型结果计算Bootstrap CI"""
    import argparse
    
    parser = argparse.ArgumentParser(description='计算Bootstrap置信区间')
    parser.add_argument('--result_dirs', nargs='+', 
                       default=['models/SwinT/_results/multimodal', 
                               'cnn_result_unified', 
                               'vmamba_result_unified'],
                       help='结果目录列表')
    parser.add_argument('--output_dir', default='analysis/bootstrap_ci_results',
                       help='输出目录')
    parser.add_argument('--n_bootstrap', type=int, default=2000,
                       help='Bootstrap重采样次数')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    bootstrap_ci = BootstrapCI(n_bootstrap=args.n_bootstrap)
    
    all_results = {}
    
    for result_dir in args.result_dirs:
        if not os.path.exists(result_dir):
            print(f"⚠️  跳过不存在的目录: {result_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"处理: {result_dir}")
        print(f"{'='*60}")
        
        # 这里需要实际加载模型并重新预测，或者从保存的预测结果加载
        # 暂时使用占位符
        print("⚠️  需要从模型重新预测或加载保存的预测结果")
        print("   建议：在训练脚本中保存所有预测概率和标签")
        
        # TODO: 实现实际的预测加载逻辑
    
    # 保存结果
    output_file = os.path.join(args.output_dir, 'bootstrap_ci_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Bootstrap CI结果已保存到: {output_file}")


if __name__ == '__main__':
    main()

