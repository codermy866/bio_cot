#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最优阈值查找脚本
使用多种方法找到最优阈值：
1. Youden指数（敏感性+特异性-1）
2. F1-Score最大化
3. 平衡敏感性和特异性
4. ROC曲线最优点
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import (
    roc_curve, roc_auc_score, f1_score, accuracy_score,
    recall_score, precision_score, confusion_matrix
)

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analysis.bootstrap_confidence_intervals import BootstrapCI


def find_optimal_threshold_youden(y_true, y_probs):
    """
    使用Youden指数找到最优阈值
    Youden指数 = 敏感性 + 特异性 - 1
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    optimal_youden = youden_index[optimal_idx]
    
    return optimal_threshold, optimal_youden


def find_optimal_threshold_f1(y_true, y_probs):
    """
    使用F1-Score最大化找到最优阈值
    """
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.01, 0.99, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def find_optimal_threshold_balance(y_true, y_probs):
    """
    平衡敏感性和特异性（使两者差值最小）
    """
    best_balance = float('inf')
    best_threshold = 0.5
    
    for threshold in np.arange(0.01, 0.99, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        sens = recall_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balance = abs(sens - spec)
        if balance < best_balance:
            best_balance = balance
            best_threshold = threshold
    
    return best_threshold, best_balance


def find_optimal_threshold_roc(y_true, y_probs):
    """
    ROC曲线最优点（最接近左上角的点）
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    distances = np.sqrt((1 - tpr)**2 + fpr**2)
    optimal_idx = np.argmin(distances)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold


def calculate_metrics_at_threshold(y_true, y_probs, threshold):
    """计算指定阈值下的所有指标"""
    y_pred = (y_probs >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    youden = sensitivity + specificity - 1
    
    return {
        'threshold': float(threshold),
        'accuracy': float(acc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'npv': float(npv),
        'f1_score': float(f1),
        'youden_index': float(youden),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def convert_to_serializable(obj):
    """将numpy类型转换为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='查找最优阈值')
    parser.add_argument('--result_dir', type=str,
                       default='analysis/external_validation_results',
                       help='结果目录')
    parser.add_argument('--method', type=str, default='all',
                       choices=['youden', 'f1', 'balance', 'roc', 'all'],
                       help='阈值选择方法')
    
    args = parser.parse_args()
    
    # 加载预测结果
    result_dir = Path(args.result_dir)
    
    val_labels_file = result_dir / 'val_labels.npy'
    val_probs_file = result_dir / 'val_probs.npy'
    
    if not val_labels_file.exists() or not val_probs_file.exists():
        print("❌ 未找到保存的预测结果文件")
        print("   需要先运行外部验证评估生成预测结果")
        return
    
    y_true = np.load(val_labels_file)
    y_probs = np.load(val_probs_file)
    
    print("=" * 80)
    print("📊 最优阈值分析 - 多种方法对比")
    print("=" * 80)
    print()
    
    results = {}
    
    # 方法1: Youden指数
    if args.method in ['youden', 'all']:
        print("方法1: Youden指数（敏感性+特异性-1）")
        threshold_youden, youden_value = find_optimal_threshold_youden(y_true, y_probs)
        metrics_youden = calculate_metrics_at_threshold(y_true, y_probs, threshold_youden)
        results['youden'] = {
            'threshold': threshold_youden,
            'youden_index': youden_value,
            'metrics': metrics_youden
        }
        
        print(f"  最优阈值: {threshold_youden:.4f}")
        print(f"  Youden指数: {youden_value:.4f}")
        print(f"  敏感性: {metrics_youden['sensitivity']:.4f}")
        print(f"  特异性: {metrics_youden['specificity']:.4f}")
        print(f"  准确率: {metrics_youden['accuracy']:.4f}")
        print(f"  F1-Score: {metrics_youden['f1_score']:.4f}")
        print()
    
    # 方法2: F1-Score最大化
    if args.method in ['f1', 'all']:
        print("方法2: F1-Score最大化")
        threshold_f1, f1_value = find_optimal_threshold_f1(y_true, y_probs)
        metrics_f1 = calculate_metrics_at_threshold(y_true, y_probs, threshold_f1)
        results['f1'] = {
            'threshold': threshold_f1,
            'f1_score': f1_value,
            'metrics': metrics_f1
        }
        
        print(f"  最优阈值: {threshold_f1:.4f}")
        print(f"  F1-Score: {f1_value:.4f}")
        print(f"  敏感性: {metrics_f1['sensitivity']:.4f}")
        print(f"  特异性: {metrics_f1['specificity']:.4f}")
        print(f"  准确率: {metrics_f1['accuracy']:.4f}")
        print()
    
    # 方法3: 平衡敏感性和特异性
    if args.method in ['balance', 'all']:
        print("方法3: 平衡敏感性和特异性（使两者差值最小）")
        threshold_balance, balance_value = find_optimal_threshold_balance(y_true, y_probs)
        metrics_balance = calculate_metrics_at_threshold(y_true, y_probs, threshold_balance)
        results['balance'] = {
            'threshold': threshold_balance,
            'balance': balance_value,
            'metrics': metrics_balance
        }
        
        print(f"  最优阈值: {threshold_balance:.4f}")
        print(f"  敏感性-特异性差值: {balance_value:.4f}")
        print(f"  敏感性: {metrics_balance['sensitivity']:.4f}")
        print(f"  特异性: {metrics_balance['specificity']:.4f}")
        print(f"  准确率: {metrics_balance['accuracy']:.4f}")
        print(f"  F1-Score: {metrics_balance['f1_score']:.4f}")
        print()
    
    # 方法4: ROC曲线最优点
    if args.method in ['roc', 'all']:
        print("方法4: ROC曲线最优点（最接近左上角的点）")
        threshold_roc = find_optimal_threshold_roc(y_true, y_probs)
        metrics_roc = calculate_metrics_at_threshold(y_true, y_probs, threshold_roc)
        results['roc'] = {
            'threshold': threshold_roc,
            'metrics': metrics_roc
        }
        
        print(f"  最优阈值: {threshold_roc:.4f}")
        print(f"  敏感性: {metrics_roc['sensitivity']:.4f}")
        print(f"  特异性: {metrics_roc['specificity']:.4f}")
        print(f"  准确率: {metrics_roc['accuracy']:.4f}")
        print(f"  F1-Score: {metrics_roc['f1_score']:.4f}")
        print()
    
    # 推荐方案
    print("=" * 80)
    print("💡 推荐方案")
    print("=" * 80)
    
    if 'youden' in results:
        print("✅ 推荐方法1: Youden指数（平衡敏感性和特异性）")
        print(f"   阈值: {results['youden']['threshold']:.4f}")
        print(f"   敏感性: {results['youden']['metrics']['sensitivity']:.4f}")
        print(f"   特异性: {results['youden']['metrics']['specificity']:.4f}")
        print(f"   F1-Score: {results['youden']['metrics']['f1_score']:.4f}")
        print()
    
    if 'f1' in results:
        print("✅ 推荐方法2: F1-Score最大化（平衡精确率和敏感性）")
        print(f"   阈值: {results['f1']['threshold']:.4f}")
        print(f"   敏感性: {results['f1']['metrics']['sensitivity']:.4f}")
        print(f"   特异性: {results['f1']['metrics']['specificity']:.4f}")
        print(f"   F1-Score: {results['f1']['metrics']['f1_score']:.4f}")
        print()
    
    # 保存结果
    output_file = result_dir / 'optimal_threshold_analysis.json'
    results_serializable = convert_to_serializable(results)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"📁 结果已保存: {output_file}")
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()

