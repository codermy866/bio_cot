#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临床相关评估指标计算
包括：灵敏度、特异度、PPV、NPV、NRI、IDI等
"""

import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy import stats
from typing import Dict, Tuple, Optional


def calculate_clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: Optional[np.ndarray] = None, 
                               y_baseline: Optional[np.ndarray] = None, y_baseline_probs: Optional[np.ndarray] = None) -> Dict:
    """
    计算临床相关评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_probs: 预测概率（用于AUC、NRI、IDI等）
        y_baseline: 基线模型的预测标签（用于NRI、IDI）
        y_baseline_probs: 基线模型的预测概率（用于NRI、IDI）
    
    Returns:
        包含所有指标的字典
    """
    metrics = {}
    
    # 基础混淆矩阵
    tn, fp, fn, tp = calculate_confusion_matrix(y_true, y_pred)
    
    # 1. 基础分类指标
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    # 2. 灵敏度（Sensitivity, Recall, TPR）
    if (tp + fn) > 0:
        metrics['sensitivity'] = tp / (tp + fn)
    else:
        # 如果没有真实正类样本，灵敏度未定义，设为0.5（中性值）
        metrics['sensitivity'] = 0.5 if tp > 0 else 0.0
    metrics['recall'] = metrics['sensitivity']
    metrics['tpr'] = metrics['sensitivity']
    
    # 3. 特异度（Specificity, TNR）
    if (tn + fp) > 0:
        metrics['specificity'] = tn / (tn + fp)
    else:
        # 如果没有真实负类样本，特异度未定义，设为0.5（中性值）
        metrics['specificity'] = 0.5 if tn > 0 else 0.0
    metrics['tnr'] = metrics['specificity']
    
    # 4. 阳性预测值（Positive Predictive Value, PPV, Precision）
    if (tp + fp) > 0:
        metrics['ppv'] = tp / (tp + fp)
    else:
        # 如果没有预测正类样本，PPV未定义，设为0.0
        metrics['ppv'] = 0.0
    metrics['precision'] = metrics['ppv']
    
    # 5. 阴性预测值（Negative Predictive Value, NPV）
    if (tn + fn) > 0:
        metrics['npv'] = tn / (tn + fn)
    else:
        # 如果没有预测负类样本，NPV未定义
        # 如果TN=0且FN=0（所有样本都被预测为正类），NPV设为0.0
        # 但如果实际有负类样本（FN应该>0），这种情况不应该发生
        metrics['npv'] = 0.0 if fn == 0 and tn == 0 else 0.5
    
    # 6. 假阳性率（False Positive Rate, FPR, 1-Specificity）
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # 7. 假阴性率（False Negative Rate, FNR, 1-Sensitivity）
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # 8. 阳性似然比（Positive Likelihood Ratio, LR+）
    if (1 - metrics['specificity']) > 1e-10:  # 避免除零，使用小的阈值
        metrics['lr_plus'] = metrics['sensitivity'] / (1 - metrics['specificity'])
    else:
        metrics['lr_plus'] = float('inf') if metrics['sensitivity'] > 1e-10 else 1.0  # 特殊情况处理
    
    # 9. 阴性似然比（Negative Likelihood Ratio, LR-）
    if metrics['specificity'] > 1e-10:  # 避免除零
        metrics['lr_minus'] = (1 - metrics['sensitivity']) / metrics['specificity']
    else:
        metrics['lr_minus'] = float('inf') if (1 - metrics['sensitivity']) > 1e-10 else 1.0  # 特殊情况处理
    
    # 10. 优势比（Odds Ratio, OR）
    if fp > 0 and fn > 0:
        metrics['odds_ratio'] = (tp * tn) / (fp * fn)
    elif tp > 0 and tn > 0:
        metrics['odds_ratio'] = float('inf')  # 没有假阳性和假阴性，完全正确
    elif tp > 0 or tn > 0:
        metrics['odds_ratio'] = 0.0  # 部分预测错误
    else:
        metrics['odds_ratio'] = 0.0  # 极端情况
    
    # 11. Youden指数（Youden's Index）
    metrics['youden_index'] = metrics['sensitivity'] + metrics['specificity'] - 1.0
    
    # 12. F1-Score
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1_score'] = 0.0
    
    # 13. 平衡准确率（Balanced Accuracy）
    metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2.0
    
    # 14. Matthews相关系数（MCC）
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator > 0:
        metrics['mcc'] = (tp * tn - fp * fn) / denominator
    else:
        metrics['mcc'] = 0.0
    
    # 15. AUC (如果有概率值)
    if y_probs is not None:
        try:
            fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_probs)
            metrics['auc'] = auc(fpr_curve, tpr_curve)
            metrics['roc_curve'] = {'fpr': fpr_curve.tolist(), 'tpr': tpr_curve.tolist(), 'thresholds': thresholds.tolist()}
            
            # 找到Youden指数最大的阈值（最优阈值）
            youden_scores = tpr_curve - fpr_curve
            optimal_idx = np.argmax(youden_scores)
            metrics['optimal_threshold'] = thresholds[optimal_idx]
            metrics['optimal_sensitivity'] = float(tpr_curve[optimal_idx])
            metrics['optimal_specificity'] = float(1 - fpr_curve[optimal_idx])
        except:
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0
    
    # 16. 净重分类改善指数（Net Reclassification Improvement, NRI）
    if y_baseline_probs is not None and y_probs is not None:
        metrics['nri'] = calculate_nri(y_true, y_probs, y_baseline_probs)
    else:
        metrics['nri'] = None
    
    # 17. 综合判别改善指数（Integrated Discrimination Improvement, IDI）
    if y_baseline_probs is not None and y_probs is not None:
        metrics['idi'] = calculate_idi(y_true, y_probs, y_baseline_probs)
    else:
        metrics['idi'] = None
    
    # 18. 混淆矩阵
    metrics['confusion_matrix'] = {
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }
    
    return metrics


def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """计算混淆矩阵的四个值：TN, FP, FN, TP"""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return tn, fp, fn, tp


def calculate_nri(y_true: np.ndarray, y_new_probs: np.ndarray, y_baseline_probs: np.ndarray) -> Dict:
    """
    计算净重分类改善指数（Net Reclassification Improvement, NRI）
    
    NRI衡量新模型相比基线模型在正确分类方面的改善程度
    """
    # 将概率转换为分类（使用0.5作为阈值，也可以使用Youden指数最优阈值）
    threshold = 0.5
    y_new_pred = (y_new_probs >= threshold).astype(int)
    y_baseline_pred = (y_baseline_probs >= threshold).astype(int)
    
    # 事件组（y_true == 1）的NRI
    event_mask = y_true == 1
    if np.sum(event_mask) > 0:
        # 新模型正确分类但基线模型错误分类的事件
        up_event = np.sum((y_new_pred[event_mask] == 1) & (y_baseline_pred[event_mask] == 0))
        # 新模型错误分类但基线模型正确分类的事件
        down_event = np.sum((y_new_pred[event_mask] == 0) & (y_baseline_pred[event_mask] == 1))
        nri_event = (up_event - down_event) / np.sum(event_mask)
    else:
        nri_event = 0.0
    
    # 非事件组（y_true == 0）的NRI
    nonevent_mask = y_true == 0
    if np.sum(nonevent_mask) > 0:
        # 新模型正确分类但基线模型错误分类的非事件
        up_nonevent = np.sum((y_new_pred[nonevent_mask] == 0) & (y_baseline_pred[nonevent_mask] == 1))
        # 新模型错误分类但基线模型正确分类的非事件
        down_nonevent = np.sum((y_new_pred[nonevent_mask] == 1) & (y_baseline_pred[nonevent_mask] == 0))
        nri_nonevent = (up_nonevent - down_nonevent) / np.sum(nonevent_mask)
    else:
        nri_nonevent = 0.0
    
    # 总NRI
    nri_total = nri_event + nri_nonevent
    
    return {
        'total': float(nri_total),
        'event': float(nri_event),
        'nonevent': float(nri_nonevent)
    }


def calculate_idi(y_true: np.ndarray, y_new_probs: np.ndarray, y_baseline_probs: np.ndarray) -> Dict:
    """
    计算综合判别改善指数（Integrated Discrimination Improvement, IDI）
    
    IDI衡量新模型相比基线模型在概率预测方面的改善程度
    """
    event_mask = y_true == 1
    nonevent_mask = y_true == 0
    
    # 事件组的平均概率改善
    if np.sum(event_mask) > 0:
        idi_event = np.mean(y_new_probs[event_mask]) - np.mean(y_baseline_probs[event_mask])
    else:
        idi_event = 0.0
    
    # 非事件组的平均概率改善（应该是负的，因为非事件应该预测为低概率）
    if np.sum(nonevent_mask) > 0:
        idi_nonevent = np.mean(y_baseline_probs[nonevent_mask]) - np.mean(y_new_probs[nonevent_mask])
    else:
        idi_nonevent = 0.0
    
    # 总IDI
    idi_total = idi_event + idi_nonevent
    
    return {
        'total': float(idi_total),
        'event': float(idi_event),
        'nonevent': float(idi_nonevent)
    }


def calculate_calibration_metrics(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> Dict:
    """
    计算校准指标（Calibration Metrics）
    用于评估模型预测概率的准确性
    """
    from sklearn.calibration import calibration_curve
    
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_probs, n_bins=n_bins)
    
    # Brier Score（越小越好，范围0-1）
    brier_score = np.mean((y_probs - y_true) ** 2)
    
    # ECE (Expected Calibration Error)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_probs > bin_lower) & (y_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return {
        'brier_score': float(brier_score),
        'ece': float(ece),
        'calibration_curve': {
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist()
        }
    }

