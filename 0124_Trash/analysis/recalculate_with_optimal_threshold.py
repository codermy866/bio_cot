#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用最优阈值重新计算性能指标
从决策曲线分析中找到最优阈值，然后重新计算所有指标
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analysis.bootstrap_confidence_intervals import BootstrapCI


def recalculate_metrics_with_threshold(y_true, y_probs, threshold=0.5):
    """
    使用指定阈值重新计算性能指标
    
    Args:
        y_true: 真实标签
        y_probs: 预测概率
        threshold: 决策阈值
    
    Returns:
        包含所有指标的字典
    """
    # 使用阈值进行预测
    y_pred = (y_probs >= threshold).astype(int)
    
    # 计算基础指标
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # AUC（不受阈值影响）
    try:
        auc = roc_auc_score(y_true, y_probs)
    except:
        auc = np.nan
    
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    return {
        'threshold': threshold,
        'auc': auc,
        'accuracy': acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'f1_score': f1,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def find_optimal_threshold_from_dca(dca_file):
    """从DCA结果中找到最优阈值"""
    df = pd.read_csv(dca_file)
    
    # 找到模型的最优阈值（净收益最大）
    model_data = df[df['strategy'] == 'SWINT']
    if len(model_data) > 0:
        max_idx = model_data['net_benefit'].idxmax()
        optimal_threshold = model_data.loc[max_idx, 'threshold']
        max_net_benefit = model_data.loc[max_idx, 'net_benefit']
        return optimal_threshold, max_net_benefit
    return None, None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='使用最优阈值重新计算性能指标')
    parser.add_argument('--result_dir', type=str,
                       default='analysis/external_validation_results',
                       help='结果目录')
    parser.add_argument('--threshold', type=float, default=None,
                       help='决策阈值（如果为None，则从DCA结果中获取）')
    
    args = parser.parse_args()
    
    # 加载预测结果
    result_dir = Path(args.result_dir)
    
    # 尝试从保存的预测结果加载
    val_labels_file = result_dir / 'val_labels.npy'
    val_probs_file = result_dir / 'val_probs.npy'
    
    if val_labels_file.exists() and val_probs_file.exists():
        y_true = np.load(val_labels_file)
        y_probs = np.load(val_probs_file)
        print(f"✅ 从保存的文件加载预测结果: {len(y_true)} 个样本")
    else:
        print("❌ 未找到保存的预测结果文件")
        print("   需要先运行外部验证评估生成预测结果")
        return
    
    # 获取最优阈值
    if args.threshold is None:
        dca_file = result_dir / 'dca_results.csv'
        if dca_file.exists():
            optimal_threshold, max_net_benefit = find_optimal_threshold_from_dca(dca_file)
            if optimal_threshold is not None:
                print(f"✅ 从DCA分析中找到最优阈值: {optimal_threshold:.4f}")
                print(f"   最大净收益: {max_net_benefit:.4f}")
                threshold = optimal_threshold
            else:
                print("⚠️  无法从DCA结果中找到最优阈值，使用默认阈值0.5")
                threshold = 0.5
        else:
            print("⚠️  未找到DCA结果文件，使用默认阈值0.5")
            threshold = 0.5
    else:
        threshold = args.threshold
        print(f"✅ 使用指定阈值: {threshold:.4f}")
    
    print()
    
    # 使用最优阈值重新计算指标
    print("📊 使用最优阈值重新计算性能指标...")
    metrics = recalculate_metrics_with_threshold(y_true, y_probs, threshold)
    
    # 计算Bootstrap CI
    print("📊 计算Bootstrap置信区间...")
    bootstrap_ci = BootstrapCI(n_bootstrap=2000)
    
    # 使用最优阈值进行预测
    y_pred_optimal = (y_probs >= threshold).astype(int)
    
    # 计算所有指标的Bootstrap CI
    ci_results = bootstrap_ci.calculate_all_metrics_ci(y_true, y_pred_optimal, y_probs)
    
    # 合并结果
    results = {
        'threshold': float(threshold),
        'metrics': metrics,
        'bootstrap_ci': ci_results
    }
    
    # 保存结果
    output_file = result_dir / 'optimal_threshold_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print()
    print("=" * 80)
    print("✅ 使用最优阈值重新计算完成！")
    print("=" * 80)
    print()
    print(f"📊 性能指标（阈值={threshold:.4f}）:")
    print(f"  AUC: {ci_results['auc']['formatted']}")
    print(f"  准确率: {ci_results['accuracy']['formatted']}")
    print(f"  敏感性: {ci_results['sensitivity']['formatted']}")
    print(f"  特异性: {ci_results['specificity']['formatted']}")
    print(f"  精确率: {ci_results['precision']['formatted']}")
    print(f"  NPV: {ci_results['npv']['formatted']}")
    print(f"  F1-Score: {ci_results['f1_score']['formatted']}")
    print()
    print(f"📁 结果已保存: {output_file}")
    print()


if __name__ == '__main__':
    main()

