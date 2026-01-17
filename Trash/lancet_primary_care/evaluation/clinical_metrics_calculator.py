#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临床指标计算器
专门用于The Lancet Primary Care研究的临床相关指标计算
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

# Import from project utils
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))
from utils.advanced_clinical_metrics import calculate_clinical_metrics


class ClinicalMetricsCalculator:
    """临床指标计算器"""
    
    def __init__(self):
        pass
    
    def calculate_biopsy_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: Optional[np.ndarray] = None
    ) -> Dict:
        """
        计算活检相关指标
        
        Args:
            y_true: 真实标签（0=正常, 1=病变）
            y_pred: 预测标签（0=不活检, 1=建议活检）
            y_probs: 预测概率
        
        Returns:
            包含活检相关指标的字典
        """
        metrics = calculate_clinical_metrics(y_true, y_pred, y_probs)
        
        # 活检率
        biopsy_rate = np.mean(y_pred) * 100
        
        # 活检阳性预测值（PPV）
        ppv = metrics['ppv']
        
        # 活检阴性预测值（NPV）
        npv = metrics['npv']
        
        # 避免的不必要活检数（假阳性数）
        fp = np.sum((y_pred == 1) & (y_true == 0))
        unnecessary_biopsies = fp
        
        # 漏诊数（假阴性数）
        fn = np.sum((y_pred == 0) & (y_true == 1))
        missed_cases = fn
        
        return {
            'biopsy_rate': float(biopsy_rate),
            'ppv': float(ppv),
            'npv': float(npv),
            'sensitivity': float(metrics['sensitivity']),
            'specificity': float(metrics['specificity']),
            'unnecessary_biopsies': int(unnecessary_biopsies),
            'missed_cases': int(missed_cases),
            'true_positives': int(np.sum((y_pred == 1) & (y_true == 1))),
            'true_negatives': int(np.sum((y_pred == 0) & (y_true == 0))),
            'false_positives': int(unnecessary_biopsies),
            'false_negatives': int(missed_cases)
        }
    
    def calculate_screening_comparison(
        self,
        y_true: np.ndarray,
        method1_pred: np.ndarray,
        method1_probs: np.ndarray,
        method2_pred: np.ndarray,
        method2_probs: np.ndarray,
        method1_name: str = "方法1",
        method2_name: str = "方法2"
    ) -> Dict:
        """
        计算两种筛查方法的对比指标
        
        Returns:
            包含对比指标的字典
        """
        metrics1 = self.calculate_biopsy_metrics(y_true, method1_pred, method1_probs)
        metrics2 = self.calculate_biopsy_metrics(y_true, method2_pred, method2_probs)
        
        # 计算改善
        improvements = {
            'biopsy_rate_reduction': metrics1['biopsy_rate'] - metrics2['biopsy_rate'],
            'biopsy_rate_reduction_percent': (
                (metrics1['biopsy_rate'] - metrics2['biopsy_rate']) / 
                metrics1['biopsy_rate'] * 100
            ) if metrics1['biopsy_rate'] > 0 else 0,
            'ppv_improvement': metrics2['ppv'] - metrics1['ppv'],
            'sensitivity_change': metrics2['sensitivity'] - metrics1['sensitivity'],
            'specificity_change': metrics2['specificity'] - metrics1['specificity'],
            'unnecessary_biopsies_reduction': (
                metrics1['unnecessary_biopsies'] - metrics2['unnecessary_biopsies']
            )
        }
        
        # AUC比较
        auc1 = roc_auc_score(y_true, method1_probs)
        auc2 = roc_auc_score(y_true, method2_probs)
        auc_improvement = auc2 - auc1
        
        return {
            method1_name: metrics1,
            method2_name: metrics2,
            'improvements': improvements,
            'auc_comparison': {
                method1_name: float(auc1),
                method2_name: float(auc2),
                'improvement': float(auc_improvement)
            }
        }
    
    def calculate_risk_stratification_metrics(
        self,
        y_true: np.ndarray,
        risk_scores: np.ndarray,
        high_risk_threshold: float = 0.7,
        low_risk_threshold: float = 0.3
    ) -> Dict:
        """
        计算风险分层指标
        
        Args:
            y_true: 真实标签
            risk_scores: 风险评分（0-1）
            high_risk_threshold: 高风险阈值
            low_risk_threshold: 低风险阈值
        
        Returns:
            风险分层指标
        """
        # 风险分层
        high_risk_mask = risk_scores >= high_risk_threshold
        mid_risk_mask = (risk_scores >= low_risk_threshold) & (risk_scores < high_risk_threshold)
        low_risk_mask = risk_scores < low_risk_threshold
        
        n_high = np.sum(high_risk_mask)
        n_mid = np.sum(mid_risk_mask)
        n_low = np.sum(low_risk_mask)
        
        # 各风险层的病变率
        lesion_rate_high = np.mean(y_true[high_risk_mask]) if n_high > 0 else 0
        lesion_rate_mid = np.mean(y_true[mid_risk_mask]) if n_mid > 0 else 0
        lesion_rate_low = np.mean(y_true[low_risk_mask]) if n_low > 0 else 0
        
        return {
            'risk_distribution': {
                'high_risk': int(n_high),
                'mid_risk': int(n_mid),
                'low_risk': int(n_low)
            },
            'lesion_rate_by_risk': {
                'high_risk': float(lesion_rate_high),
                'mid_risk': float(lesion_rate_mid),
                'low_risk': float(lesion_rate_low)
            },
            'risk_stratification_accuracy': {
                'high_risk_ppv': float(lesion_rate_high),
                'low_risk_npv': float(1 - lesion_rate_low)
            }
        }
    
    def calculate_cost_effectiveness(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        costs: Dict[str, float],
        n_patients: int
    ) -> Dict:
        """
        计算成本效益指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签（活检建议）
            costs: 成本字典，包含各项检查的成本
            n_patients: 患者总数
        
        Returns:
            成本效益指标
        """
        # 计算各项成本
        n_biopsies = np.sum(y_pred)
        n_positive = np.sum(y_true)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        
        # 总成本（假设所有患者都进行了初筛）
        total_cost = costs.get('screening', 0) * n_patients
        total_cost += costs.get('biopsy', 0) * n_biopsies
        
        # 每检出1例病变的成本
        cost_per_case = total_cost / tp if tp > 0 else np.inf
        
        # 避免的不必要活检节省的成本
        fp = np.sum((y_pred == 1) & (y_true == 0))
        cost_saved = costs.get('biopsy', 0) * fp
        
        return {
            'total_cost': float(total_cost),
            'cost_per_patient': float(total_cost / n_patients),
            'cost_per_detected_case': float(cost_per_case),
            'cost_saved_from_unnecessary_biopsies': float(cost_saved),
            'n_biopsies': int(n_biopsies),
            'n_detected_cases': int(tp)
        }


def main():
    """示例用法"""
    calculator = ClinicalMetricsCalculator()
    
    # 示例数据
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 1, 0, 1, 1, 0])
    y_probs = np.array([0.2, 0.6, 0.8, 0.9, 0.3, 0.85, 0.25, 0.55, 0.75, 0.35])
    
    # 计算活检指标
    biopsy_metrics = calculator.calculate_biopsy_metrics(y_true, y_pred, y_probs)
    print("活检指标:", biopsy_metrics)
    
    # 计算风险分层
    risk_metrics = calculator.calculate_risk_stratification_metrics(y_true, y_probs)
    print("\n风险分层指标:", risk_metrics)


if __name__ == '__main__':
    main()

