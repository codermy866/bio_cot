#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最优阈值配置
从外部验证结果中获取最优阈值，用于训练和评估
"""

import os
import json
from pathlib import Path

# 最优阈值（Youden指数方法，从外部验证结果中获得）
OPTIMAL_THRESHOLD_YOUDEN = 0.2761

# 备用阈值（如果无法从外部验证结果中获取）
DEFAULT_OPTIMAL_THRESHOLD = 0.2761


def get_optimal_threshold(method='youden', result_dir='analysis/external_validation_results'):
    """
    获取最优阈值
    
    Args:
        method: 阈值选择方法 ('youden', 'f1', 'balance', 'roc')
        result_dir: 结果目录
    
    Returns:
        最优阈值
    """
    result_dir = Path(result_dir)
    analysis_file = result_dir / 'optimal_threshold_analysis.json'
    
    if analysis_file.exists():
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if method in data:
                return float(data[method]['threshold'])
            elif 'youden' in data:
                # 默认使用Youden指数方法
                return float(data['youden']['threshold'])
        except Exception as e:
            print(f"⚠️  无法从分析文件中读取最优阈值: {e}")
            print(f"   使用默认阈值: {DEFAULT_OPTIMAL_THRESHOLD}")
    
    return DEFAULT_OPTIMAL_THRESHOLD


def find_optimal_threshold_youden(y_true, y_probs):
    """
    使用Youden指数找到最优阈值
    
    Args:
        y_true: 真实标签
        y_probs: 预测概率
    
    Returns:
        (最优阈值, Youden指数)
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    youden_index = tpr - fpr
    optimal_idx = youden_index.argmax()
    optimal_threshold = thresholds[optimal_idx]
    optimal_youden = youden_index[optimal_idx]
    
    return optimal_threshold, optimal_youden


# 导出最优阈值常量
OPTIMAL_THRESHOLD = get_optimal_threshold()

