#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.0 补充可视化生成脚本
包括：ROC曲线、PR曲线、混淆矩阵、校准曲线、特征重要性、错误分析等
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, roc_auc_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind
import json
from datetime import datetime
import pickle

# 设置matplotlib后端
matplotlib.use('Agg')

# SCI顶刊级图表风格配置
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'SimSun'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# 添加项目路径
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from config import BioCOT_v3_Config


def load_visualization_data(config):
    """加载已保存的可视化数据"""
    data_file = Path(config.log_dir) / 'visualization_data_20260113_145816.pkl'
    csv_file = Path(config.log_dir) / 'visualization_data_20260113_145816.csv'
    
    if data_file.exists():
        with open(data_file, 'rb') as f:
            features_dict = pickle.load(f)
        print(f"✅ 加载特征数据: {data_file}")
    else:
        raise FileNotFoundError(f"未找到数据文件: {data_file}")
    
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        print(f"✅ 加载CSV数据: {csv_file}")
    else:
        df = None
    
    return features_dict, df


def visualize_roc_pr_curves(features_dict, output_dir, timestamp):
    """生成ROC曲线和PR曲线"""
    print("📊 生成ROC和PR曲线...")
    
    labels = features_dict['labels']
    probs = features_dict['probabilities'][:, 1] if 'probabilities' in features_dict else None
    
    if probs is None:
        print("⚠️ 未找到预测概率，跳过ROC/PR曲线")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. ROC曲线
    ax1 = axes[0]
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, color='#2E86AB', lw=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random (AUC = 0.5000)')
    ax1.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # 添加最优阈值点
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    ax1.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, 
             label=f'Optimal Threshold = {optimal_threshold:.3f}')
    ax1.legend(loc='lower right', fontsize=11)
    
    # 2. PR曲线
    ax2 = axes[1]
    precision, recall, pr_thresholds = precision_recall_curve(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    
    ax2.plot(recall, precision, color='#A23B72', lw=3, label=f'PR Curve (AP = {pr_auc:.4f})')
    
    # 基线（随机分类器）
    baseline = np.sum(labels) / len(labels)
    ax2.axhline(y=baseline, color='gray', lw=2, linestyle='--', label=f'Baseline (AP = {baseline:.4f})')
    
    ax2.set_xlabel('Recall', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Precision-Recall (PR) Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # 保存PDF和PNG
    output_path_pdf = output_dir / f"roc_pr_curves_{timestamp}.pdf"
    output_path_png = output_dir / f"roc_pr_curves_{timestamp}.png"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ ROC/PR曲线已保存: {output_path_pdf}")
    return output_path_pdf


def visualize_confusion_matrix(features_dict, output_dir, timestamp):
    """生成混淆矩阵热图"""
    print("📊 生成混淆矩阵...")
    
    labels = features_dict['labels']
    predictions = features_dict['predictions']
    
    cm = confusion_matrix(labels, predictions)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. 原始混淆矩阵
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                cbar_kws={'label': 'Count'}, square=True, linewidths=1, linecolor='black')
    ax1.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(['Negative', 'Positive'])
    ax1.set_yticklabels(['Negative', 'Positive'])
    
    # 2. 归一化混淆矩阵（百分比）
    ax2 = axes[1]
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax2,
                cbar_kws={'label': 'Percentage'}, square=True, linewidths=1, linecolor='black')
    ax2.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Normalized Confusion Matrix (%)', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(['Negative', 'Positive'])
    ax2.set_yticklabels(['Negative', 'Positive'])
    
    plt.tight_layout()
    
    # 保存
    output_path_pdf = output_dir / f"confusion_matrix_{timestamp}.pdf"
    output_path_png = output_dir / f"confusion_matrix_{timestamp}.png"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 混淆矩阵已保存: {output_path_pdf}")
    return output_path_pdf


def visualize_calibration_curve(features_dict, output_dir, timestamp):
    """生成校准曲线"""
    print("📊 生成校准曲线...")
    
    labels = features_dict['labels']
    probs = features_dict['probabilities'][:, 1] if 'probabilities' in features_dict else None
    
    if probs is None:
        print("⚠️ 未找到预测概率，跳过校准曲线")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. 校准曲线
    ax1 = axes[0]
    fraction_of_positives, mean_predicted_value = calibration_curve(labels, probs, n_bins=10)
    brier_score = brier_score_loss(labels, probs)
    
    ax1.plot(mean_predicted_value, fraction_of_positives, 's-', color='#2E86AB', 
             lw=3, markersize=8, label='Bio-COT 3.0')
    ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration')
    ax1.set_xlabel('Mean Predicted Probability', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Fraction of Positives', fontsize=13, fontweight='bold')
    ax1.set_title(f'(a) Calibration Curve (Brier Score = {brier_score:.4f})', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # 2. 预测概率分布直方图（按标签）
    ax2 = axes[1]
    neg_probs = probs[labels == 0]
    pos_probs = probs[labels == 1]
    
    ax2.hist(neg_probs, bins=30, alpha=0.6, label='Negative (True)', color='red', 
             edgecolor='black', linewidth=1)
    ax2.hist(pos_probs, bins=30, alpha=0.6, label='Positive (True)', color='green', 
             edgecolor='black', linewidth=1)
    ax2.axvline(x=0.5, color='blue', linestyle='--', lw=2, label='Decision Threshold (0.5)')
    ax2.set_xlabel('Predicted Probability', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Prediction Probability Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存
    output_path_pdf = output_dir / f"calibration_curve_{timestamp}.pdf"
    output_path_png = output_dir / f"calibration_curve_{timestamp}.png"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 校准曲线已保存: {output_path_pdf}")
    return output_path_pdf


def visualize_feature_importance(features_dict, output_dir, timestamp):
    """生成特征重要性分析"""
    print("📊 生成特征重要性分析...")
    
    z_causal = features_dict['z_causal']
    labels = features_dict['labels']
    
    # 计算每个特征的重要性（使用t-test统计量）
    feature_importance = []
    for i in range(z_causal.shape[1]):
        neg_values = z_causal[labels == 0, i]
        pos_values = z_causal[labels == 1, i]
        t_stat, p_val = ttest_ind(pos_values, neg_values)
        feature_importance.append({
            'feature_idx': i,
            't_statistic': abs(t_stat),
            'p_value': p_val,
            'mean_diff': np.mean(pos_values) - np.mean(neg_values)
        })
    
    df_importance = pd.DataFrame(feature_importance)
    df_importance = df_importance.sort_values('t_statistic', ascending=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Top特征重要性（t-statistic）
    ax1 = axes[0, 0]
    top_n = 50
    top_features = df_importance.head(top_n)
    ax1.barh(range(top_n), top_features['t_statistic'].values, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels([f'Feature {idx}' for idx in top_features['feature_idx'].values], fontsize=8)
    ax1.set_xlabel('|t-statistic|', fontsize=12, fontweight='bold')
    ax1.set_title(f'(a) Top {top_n} Feature Importance (t-statistic)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # 2. 特征重要性分布直方图
    ax2 = axes[0, 1]
    ax2.hist(df_importance['t_statistic'], bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('|t-statistic|', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Feature Importance Distribution', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. p值分布
    ax3 = axes[1, 0]
    significant = df_importance['p_value'] < 0.05
    ax3.hist(df_importance[significant]['p_value'], bins=30, color='green', alpha=0.6, 
             label=f'Significant (n={significant.sum()})', edgecolor='black')
    ax3.hist(df_importance[~significant]['p_value'], bins=30, color='red', alpha=0.6, 
             label=f'Non-significant (n={(~significant).sum()})', edgecolor='black')
    ax3.axvline(x=0.05, color='blue', linestyle='--', lw=2, label='p=0.05 threshold')
    ax3.set_xlabel('p-value', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('(c) p-value Distribution', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 特征重要性vs均值差异散点图
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df_importance['mean_diff'], df_importance['t_statistic'], 
                         c=-np.log10(df_importance['p_value'] + 1e-10), 
                         cmap='RdYlGn', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Mean Difference (Positive - Negative)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('|t-statistic|', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Feature Importance vs Mean Difference', fontsize=13, fontweight='bold')
    plt.colorbar(scatter, ax=ax4, label='-Log10(p-value)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path_pdf = output_dir / f"feature_importance_{timestamp}.pdf"
    output_path_png = output_dir / f"feature_importance_{timestamp}.png"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存特征重要性数据
    importance_file = output_dir / f"feature_importance_data_{timestamp}.csv"
    df_importance.to_csv(importance_file, index=False)
    print(f"✅ 特征重要性数据已保存: {importance_file}")
    
    print(f"✅ 特征重要性分析已保存: {output_path_pdf}")
    return output_path_pdf


def visualize_center_comparison(features_dict, output_dir, timestamp):
    """生成中心性能对比（森林图风格）"""
    print("📊 生成中心性能对比...")
    
    labels = features_dict['labels']
    predictions = features_dict['predictions']
    probs = features_dict['probabilities'][:, 1] if 'probabilities' in features_dict else None
    centers = features_dict['centers']
    
    if probs is None:
        print("⚠️ 未找到预测概率，跳过中心对比")
        return None
    
    # 计算每个中心的性能指标
    center_results = []
    unique_centers = np.unique(centers)
    
    for center_id in unique_centers:
        mask = centers == center_id
        center_labels = labels[mask]
        center_preds = predictions[mask]
        center_probs = probs[mask]
        
        # 计算指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        acc = accuracy_score(center_labels, center_preds)
        prec = precision_score(center_labels, center_preds, zero_division=0)
        rec = recall_score(center_labels, center_preds, zero_division=0)
        f1 = f1_score(center_labels, center_preds, zero_division=0)
        auc_score = roc_auc_score(center_labels, center_probs) if len(np.unique(center_labels)) > 1 else 0.5
        
        center_results.append({
            'center': center_id,
            'n_samples': mask.sum(),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'auc': auc_score
        })
    
    df_centers = pd.DataFrame(center_results)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 3, idx % 3]
        
        # 计算置信区间（使用Bootstrap）
        values = df_centers[metric].values
        centers_list = df_centers['center'].values
        
        # 绘制柱状图
        bars = ax.bar(range(len(centers_list)), values, color='#2E86AB', alpha=0.7, 
                     edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(len(centers_list)))
        ax.set_xticklabels([f'Center {c}' for c in centers_list], fontsize=11)
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {label} by Center', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
    
    # 最后一个子图：样本数量
    ax6 = axes[1, 2]
    bars = ax6.bar(range(len(centers_list)), df_centers['n_samples'].values, 
                   color='#A23B72', alpha=0.7, edgecolor='black', linewidth=1.5)
    for i, (bar, val) in enumerate(zip(bars, df_centers['n_samples'].values)):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{int(val)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax6.set_xticks(range(len(centers_list)))
    ax6.set_xticklabels([f'Center {c}' for c in centers_list], fontsize=11)
    ax6.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax6.set_title('(f) Sample Size by Center', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存
    output_path_pdf = output_dir / f"center_comparison_{timestamp}.pdf"
    output_path_png = output_dir / f"center_comparison_{timestamp}.png"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存中心对比数据
    center_file = output_dir / f"center_comparison_data_{timestamp}.csv"
    df_centers.to_csv(center_file, index=False)
    print(f"✅ 中心对比数据已保存: {center_file}")
    
    print(f"✅ 中心性能对比已保存: {output_path_pdf}")
    return output_path_pdf


def visualize_error_analysis(features_dict, output_dir, timestamp):
    """生成错误分析可视化"""
    print("📊 生成错误分析...")
    
    labels = features_dict['labels']
    predictions = features_dict['predictions']
    probs = features_dict['probabilities'][:, 1] if 'probabilities' in features_dict else None
    
    if probs is None:
        print("⚠️ 未找到预测概率，跳过错误分析")
        return None
    
    # 识别错误样本
    errors = predictions != labels
    false_positives = (predictions == 1) & (labels == 0)
    false_negatives = (predictions == 0) & (labels == 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. 错误类型分布
    ax1 = axes[0, 0]
    error_types = ['Correct', 'False Positive', 'False Negative']
    error_counts = [
        np.sum(~errors),
        np.sum(false_positives),
        np.sum(false_negatives)
    ]
    colors = ['green', 'orange', 'red']
    bars = ax1.bar(error_types, error_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    for bar, count in zip(bars, error_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Error Type Distribution', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 错误样本的预测概率分布
    ax2 = axes[0, 1]
    fp_probs = probs[false_positives]
    fn_probs = probs[false_negatives]
    correct_probs = probs[~errors]
    
    ax2.hist(correct_probs, bins=30, alpha=0.5, label='Correct', color='green', edgecolor='black')
    if len(fp_probs) > 0:
        ax2.hist(fp_probs, bins=30, alpha=0.7, label='False Positive', color='orange', edgecolor='black')
    if len(fn_probs) > 0:
        ax2.hist(fn_probs, bins=30, alpha=0.7, label='False Negative', color='red', edgecolor='black')
    ax2.axvline(x=0.5, color='blue', linestyle='--', lw=2, label='Decision Threshold')
    ax2.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Prediction Probability Distribution by Error Type', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 错误样本的特征空间分布（使用PCA）
    ax3 = axes[1, 0]
    from sklearn.decomposition import PCA
    z_causal = features_dict['z_causal']
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_causal)
    
    correct_mask = ~errors
    ax3.scatter(z_pca[correct_mask, 0], z_pca[correct_mask, 1], 
               c='green', label='Correct', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    if np.sum(false_positives) > 0:
        ax3.scatter(z_pca[false_positives, 0], z_pca[false_positives, 1], 
                   c='orange', label='False Positive', alpha=0.8, s=100, 
                   edgecolors='black', linewidth=1, marker='^')
    if np.sum(false_negatives) > 0:
        ax3.scatter(z_pca[false_negatives, 0], z_pca[false_negatives, 1], 
                   c='red', label='False Negative', alpha=0.8, s=100, 
                   edgecolors='black', linewidth=1, marker='s')
    ax3.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax3.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Error Samples in Feature Space', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. 错误率统计
    ax4 = axes[1, 1]
    error_rate = np.sum(errors) / len(errors)
    fp_rate = np.sum(false_positives) / len(labels)
    fn_rate = np.sum(false_negatives) / len(labels)
    
    stats_data = {
        'Overall Error Rate': error_rate,
        'False Positive Rate': fp_rate,
        'False Negative Rate': fn_rate
    }
    
    bars = ax4.bar(stats_data.keys(), stats_data.values(), color=['#2E86AB', '#F18F01', '#C73E1D'], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, stats_data.values()):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Rate', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Error Rate Statistics', fontsize=13, fontweight='bold')
    ax4.set_ylim([0, max(stats_data.values()) * 1.2])
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存
    output_path_pdf = output_dir / f"error_analysis_{timestamp}.pdf"
    output_path_png = output_dir / f"error_analysis_{timestamp}.png"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 错误分析已保存: {output_path_pdf}")
    return output_path_pdf


def visualize_causal_vs_noise(features_dict, output_dir, timestamp):
    """生成因果特征vs噪声特征的对比可视化"""
    print("📊 生成因果特征vs噪声特征对比...")
    
    z_causal = features_dict['z_causal']
    z_noise = features_dict.get('z_noise', None)
    labels = features_dict['labels']
    centers = features_dict['centers']
    
    if z_noise is None:
        print("⚠️ 未找到噪声特征，跳过对比可视化")
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 使用PCA降维
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    z_causal_2d = pca.fit_transform(z_causal)
    z_noise_2d = pca.fit_transform(z_noise)
    
    # 1. 因果特征：按标签着色
    ax1 = axes[0, 0]
    neg_mask = labels == 0
    pos_mask = labels == 1
    ax1.scatter(z_causal_2d[neg_mask, 0], z_causal_2d[neg_mask, 1], 
               c='red', label='Negative', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax1.scatter(z_causal_2d[pos_mask, 0], z_causal_2d[pos_mask, 1], 
               c='green', label='Positive', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Causal Features by Label', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 噪声特征：按标签着色
    ax2 = axes[0, 1]
    ax2.scatter(z_noise_2d[neg_mask, 0], z_noise_2d[neg_mask, 1], 
               c='red', label='Negative', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax2.scatter(z_noise_2d[pos_mask, 0], z_noise_2d[pos_mask, 1], 
               c='green', label='Positive', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Noise Features by Label', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. 因果特征：按中心着色
    ax3 = axes[0, 2]
    unique_centers = np.unique(centers)
    colors_center = plt.cm.tab10(np.linspace(0, 1, len(unique_centers)))
    for i, center_id in enumerate(unique_centers):
        mask = centers == center_id
        ax3.scatter(z_causal_2d[mask, 0], z_causal_2d[mask, 1], 
                   c=[colors_center[i]], label=f'Center {center_id}', alpha=0.6, s=50, 
                   edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax3.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Causal Features by Center', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. 噪声特征：按中心着色
    ax4 = axes[1, 0]
    for i, center_id in enumerate(unique_centers):
        mask = centers == center_id
        ax4.scatter(z_noise_2d[mask, 0], z_noise_2d[mask, 1], 
                   c=[colors_center[i]], label=f'Center {center_id}', alpha=0.6, s=50, 
                   edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax4.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Noise Features by Center', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. 分离度对比（类间距离/类内距离）
    ax5 = axes[1, 1]
    
    # 计算因果特征的分离度
    neg_causal = z_causal[neg_mask]
    pos_causal = z_causal[pos_mask]
    causal_separation = np.linalg.norm(neg_causal.mean(axis=0) - pos_causal.mean(axis=0))
    causal_within = (np.std(neg_causal, axis=0).mean() + np.std(pos_causal, axis=0).mean()) / 2
    causal_ratio = causal_separation / (causal_within + 1e-8)
    
    # 计算噪声特征的分离度
    neg_noise = z_noise[neg_mask]
    pos_noise = z_noise[pos_mask]
    noise_separation = np.linalg.norm(neg_noise.mean(axis=0) - pos_noise.mean(axis=0))
    noise_within = (np.std(neg_noise, axis=0).mean() + np.std(pos_noise, axis=0).mean()) / 2
    noise_ratio = noise_separation / (noise_within + 1e-8)
    
    metrics = ['Causal Features', 'Noise Features']
    ratios = [causal_ratio, noise_ratio]
    bars = ax5.bar(metrics, ratios, color=['#2E86AB', '#A23B72'], alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    for bar, ratio in zip(bars, ratios):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{ratio:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Separation Ratio (Between/Within)', fontsize=12, fontweight='bold')
    ax5.set_title('(e) Feature Separation Comparison', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 中心重叠度对比
    ax6 = axes[1, 2]
    
    # 计算不同中心在因果特征空间的重叠度（使用Jaccard相似度近似）
    def compute_overlap_ratio(features, centers):
        unique_centers = np.unique(centers)
        overlaps = []
        for i, c1 in enumerate(unique_centers):
            for j, c2 in enumerate(unique_centers):
                if i < j:
                    mask1 = centers == c1
                    mask2 = centers == c2
                    mean1 = features[mask1].mean(axis=0)
                    mean2 = features[mask2].mean(axis=0)
                    # 使用余弦相似度作为重叠度
                    cos_sim = np.dot(mean1, mean2) / (np.linalg.norm(mean1) * np.linalg.norm(mean2) + 1e-8)
                    overlaps.append(cos_sim)
        return np.mean(overlaps) if overlaps else 0
    
    causal_overlap = compute_overlap_ratio(z_causal, centers)
    noise_overlap = compute_overlap_ratio(z_noise, centers)
    
    metrics_overlap = ['Causal Features', 'Noise Features']
    overlaps = [causal_overlap, noise_overlap]
    bars = ax6.bar(metrics_overlap, overlaps, color=['#2E86AB', '#A23B72'], alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    for bar, overlap in zip(bars, overlaps):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{overlap:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Center Overlap (Cosine Similarity)', fontsize=12, fontweight='bold')
    ax6.set_title('(f) Cross-Center Overlap Comparison', fontsize=13, fontweight='bold')
    ax6.set_ylim([0, 1.1])
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Causal Features vs Noise Features: Domain-Invariance Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存
    output_path_pdf = output_dir / f"causal_vs_noise_{timestamp}.pdf"
    output_path_png = output_dir / f"causal_vs_noise_{timestamp}.png"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 因果vs噪声特征对比已保存: {output_path_pdf}")
    return output_path_pdf


def visualize_training_curves_detailed(history_file, output_dir, timestamp):
    """生成详细的训练曲线"""
    print("📊 生成详细训练曲线...")
    
    if not Path(history_file).exists():
        print(f"⚠️ 未找到训练历史文件: {history_file}")
        return None
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = np.array(range(1, len(history['train_loss']) + 1))
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    
    # 1. Loss曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2.5, alpha=0.8)
    ax1.plot(epochs, history['val_loss'], 'r--', label='Val Loss', linewidth=2.5, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Training and Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy曲线
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2.5, alpha=0.8)
    ax2.plot(epochs, history['val_acc'], 'r--', label='Val Acc', linewidth=2.5, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Training and Validation Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. AUC曲线
    ax3 = axes[0, 2]
    best_auc = max(history['val_auc'])
    ax3.plot(epochs, history['val_auc'], 'g-', label='Val AUC', linewidth=2.5, marker='o', markersize=5, alpha=0.8)
    ax3.axhline(y=best_auc, color='r', linestyle='--', linewidth=2, label=f'Best: {best_auc:.4f}')
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Validation AUC', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4-9. 损失组件详细分析
    loss_components = ['cls_loss', 'ot_loss', 'sparse_loss', 'consist_loss', 'adv_loss']
    loss_labels = ['Classification', 'OT Loss', 'Sparse Loss', 'Consistency', 'Adversarial']
    
    for idx, (comp, label) in enumerate(zip(loss_components, loss_labels)):
        ax = axes[1 + idx // 3, idx % 3]
        if comp in history and any(v > 0 for v in history[comp]):
            ax.plot(epochs, history[comp], linewidth=2.5, alpha=0.8, label=label)
            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
            ax.set_title(f'({chr(100+idx)}) {label}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            if comp in ['ot_loss', 'sparse_loss', 'consist_loss', 'adv_loss']:
                ax.set_yscale('log')
    
    plt.tight_layout()
    
    # 保存
    output_path_pdf = output_dir / f"training_curves_detailed_{timestamp}.pdf"
    output_path_png = output_dir / f"training_curves_detailed_{timestamp}.png"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 详细训练曲线已保存: {output_path_pdf}")
    return output_path_pdf


def visualize_attention_evolution(features_dict, output_dir, timestamp):
    """生成注意力演化分析（如果有多个epoch的数据）"""
    print("📊 生成注意力演化分析...")
    
    if 'attn_oct' not in features_dict or len(features_dict['attn_oct']) == 0:
        print("⚠️ 未找到注意力数据，跳过注意力演化分析")
        return None
    
    attn_oct = features_dict['attn_oct']
    attn_colpo = features_dict['attn_colpo']
    labels = features_dict['labels']
    
    # 计算每个样本的平均注意力
    attn_oct_mean = np.mean(attn_oct.reshape(attn_oct.shape[0], -1), axis=1)
    attn_colpo_mean = np.mean(attn_colpo.reshape(attn_colpo.shape[0], -1), axis=1)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 注意力值分布（按标签）
    ax1 = axes[0, 0]
    neg_mask = labels == 0
    pos_mask = labels == 1
    
    ax1.hist(attn_oct_mean[neg_mask], bins=30, alpha=0.6, label='Negative (OCT)', 
             color='red', edgecolor='black')
    ax1.hist(attn_oct_mean[pos_mask], bins=30, alpha=0.6, label='Positive (OCT)', 
             color='green', edgecolor='black')
    ax1.set_xlabel('Mean Attention Value', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('(a) OCT Attention Distribution by Label', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 注意力值分布（Colposcopy）
    ax2 = axes[0, 1]
    ax2.hist(attn_colpo_mean[neg_mask], bins=30, alpha=0.6, label='Negative (Colpo)', 
             color='red', edgecolor='black')
    ax2.hist(attn_colpo_mean[pos_mask], bins=30, alpha=0.6, label='Positive (Colpo)', 
             color='green', edgecolor='black')
    ax2.set_xlabel('Mean Attention Value', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Colposcopy Attention Distribution by Label', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 注意力相关性（OCT vs Colposcopy）
    ax3 = axes[1, 0]
    ax3.scatter(attn_oct_mean, attn_colpo_mean, c=labels, cmap='RdYlGn', 
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('OCT Mean Attention', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Colposcopy Mean Attention', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Attention Correlation: OCT vs Colposcopy', fontsize=13, fontweight='bold')
    # 计算相关系数
    corr = np.corrcoef(attn_oct_mean, attn_colpo_mean)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax3.transAxes,
            fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax3.grid(True, alpha=0.3)
    
    # 4. 注意力稀疏性分析
    ax4 = axes[1, 1]
    # 计算每个样本的注意力熵（稀疏性指标）
    def compute_entropy(attn_map):
        attn_flat = attn_map.flatten()
        attn_norm = attn_flat / (attn_flat.sum() + 1e-8)
        entropy = -np.sum(attn_norm * np.log(attn_norm + 1e-8))
        return entropy
    
    entropies_oct = [compute_entropy(attn_oct[i]) for i in range(len(attn_oct))]
    entropies_colpo = [compute_entropy(attn_colpo[i]) for i in range(len(attn_colpo))]
    
    ax4.scatter(entropies_oct, entropies_colpo, c=labels, cmap='RdYlGn', 
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('OCT Attention Entropy', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Colposcopy Attention Entropy', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Attention Sparsity Analysis', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path_pdf = output_dir / f"attention_evolution_{timestamp}.pdf"
    output_path_png = output_dir / f"attention_evolution_{timestamp}.png"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 注意力演化分析已保存: {output_path_pdf}")
    return output_path_pdf


def visualize_feature_correlation(features_dict, output_dir, timestamp):
    """生成特征相关性热图"""
    print("📊 生成特征相关性热图...")
    
    z_causal = features_dict['z_causal']
    z_sem = features_dict.get('z_sem', None)
    z_noise = features_dict.get('z_noise', None)
    
    # 选择前50个最重要的特征（避免计算量过大）
    if z_causal.shape[1] > 50:
        # 使用方差选择
        variances = np.var(z_causal, axis=0)
        top_indices = np.argsort(variances)[-50:]
        z_causal_selected = z_causal[:, top_indices]
    else:
        z_causal_selected = z_causal
        top_indices = np.arange(z_causal.shape[1])
    
    # 计算相关性矩阵
    corr_matrix = np.corrcoef(z_causal_selected.T)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. 特征相关性热图
    ax1 = axes[0]
    im1 = ax1.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xlabel('Feature Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Feature Index', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Feature Correlation Heatmap (Top 50 Features)', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Correlation Coefficient')
    
    # 2. 相关性分布直方图
    ax2 = axes[1]
    # 只取上三角（避免重复）
    triu_indices = np.triu_indices_from(corr_matrix, k=1)
    corr_values = corr_matrix[triu_indices]
    ax2.hist(corr_values, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Correlation Coefficient Distribution', fontsize=13, fontweight='bold')
    ax2.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Correlation')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存
    output_path_pdf = output_dir / f"feature_correlation_{timestamp}.pdf"
    output_path_png = output_dir / f"feature_correlation_{timestamp}.png"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 特征相关性热图已保存: {output_path_pdf}")
    return output_path_pdf


def visualize_decision_boundary(features_dict, output_dir, timestamp):
    """生成决策边界可视化（2D投影）"""
    print("📊 生成决策边界可视化...")
    
    z_causal = features_dict['z_causal']
    labels = features_dict['labels']
    probs = features_dict['probabilities'][:, 1] if 'probabilities' in features_dict else None
    
    if probs is None:
        print("⚠️ 未找到预测概率，跳过决策边界可视化")
        return None
    
    # 使用PCA降维到2D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_causal)
    
    # 创建网格用于绘制决策边界
    x_min, x_max = z_2d[:, 0].min() - 1, z_2d[:, 0].max() + 1
    y_min, y_max = z_2d[:, 1].min() - 1, z_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # 使用KNN分类器近似决策边界
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(z_2d, labels)
    Z = knn.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. 决策边界 + 数据点
    ax1 = axes[0]
    contour = ax1.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.6)
    ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    scatter = ax1.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='RdYlGn', 
                         alpha=0.8, s=50, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Decision Boundary (KNN Approximation)', fontsize=13, fontweight='bold')
    plt.colorbar(contour, ax=ax1, label='Predicted Probability')
    
    # 2. 预测概率等高线
    ax2 = axes[1]
    # 使用实际预测概率进行插值
    from scipy.interpolate import griddata
    Z_probs = griddata(z_2d, probs, (xx, yy), method='cubic')
    contour2 = ax2.contourf(xx, yy, Z_probs, levels=20, cmap='RdYlGn', alpha=0.6)
    ax2.contour(xx, yy, Z_probs, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    scatter2 = ax2.scatter(z_2d[:, 0], z_2d[:, 1], c=probs, cmap='RdYlGn', 
                          alpha=0.8, s=50, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Prediction Probability Contour', fontsize=13, fontweight='bold')
    plt.colorbar(contour2, ax=ax2, label='Predicted Probability')
    
    plt.tight_layout()
    
    # 保存
    output_path_pdf = output_dir / f"decision_boundary_{timestamp}.pdf"
    output_path_png = output_dir / f"decision_boundary_{timestamp}.png"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 决策边界可视化已保存: {output_path_pdf}")
    return output_path_pdf


def main():
    """主函数"""
    print("=" * 80)
    print("Bio-COT 3.0 补充可视化生成")
    print("=" * 80)
    
    # 配置
    config = BioCOT_v3_Config()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("\n📥 加载可视化数据...")
    features_dict, df = load_visualization_data(config)
    
    # 生成所有补充可视化
    print("\n" + "=" * 80)
    print("生成补充可视化图表...")
    print("=" * 80)
    
    visualizations = []
    
    # 1. ROC和PR曲线
    try:
        vis = visualize_roc_pr_curves(features_dict, output_dir, timestamp)
        if vis:
            visualizations.append(vis)
    except Exception as e:
        print(f"⚠️ ROC/PR曲线生成失败: {e}")
    
    # 2. 混淆矩阵
    try:
        vis = visualize_confusion_matrix(features_dict, output_dir, timestamp)
        if vis:
            visualizations.append(vis)
    except Exception as e:
        print(f"⚠️ 混淆矩阵生成失败: {e}")
    
    # 3. 校准曲线
    try:
        vis = visualize_calibration_curve(features_dict, output_dir, timestamp)
        if vis:
            visualizations.append(vis)
    except Exception as e:
        print(f"⚠️ 校准曲线生成失败: {e}")
    
    # 4. 特征重要性
    try:
        vis = visualize_feature_importance(features_dict, output_dir, timestamp)
        if vis:
            visualizations.append(vis)
    except Exception as e:
        print(f"⚠️ 特征重要性生成失败: {e}")
    
    # 5. 中心性能对比
    try:
        vis = visualize_center_comparison(features_dict, output_dir, timestamp)
        if vis:
            visualizations.append(vis)
    except Exception as e:
        print(f"⚠️ 中心对比生成失败: {e}")
    
    # 6. 错误分析
    try:
        vis = visualize_error_analysis(features_dict, output_dir, timestamp)
        if vis:
            visualizations.append(vis)
    except Exception as e:
        print(f"⚠️ 错误分析生成失败: {e}")
    
    # 7. 因果vs噪声特征对比
    try:
        vis = visualize_causal_vs_noise(features_dict, output_dir, timestamp)
        if vis:
            visualizations.append(vis)
    except Exception as e:
        print(f"⚠️ 因果vs噪声对比生成失败: {e}")
    
    # 8. 详细训练曲线
    try:
        history_file = output_dir / 'training_history_20260113_093929.json'
        vis = visualize_training_curves_detailed(str(history_file), output_dir, timestamp)
        if vis:
            visualizations.append(vis)
    except Exception as e:
        print(f"⚠️ 详细训练曲线生成失败: {e}")
    
    # 9. 注意力演化分析
    try:
        vis = visualize_attention_evolution(features_dict, output_dir, timestamp)
        if vis:
            visualizations.append(vis)
    except Exception as e:
        print(f"⚠️ 注意力演化生成失败: {e}")
    
    # 10. 特征相关性热图
    try:
        vis = visualize_feature_correlation(features_dict, output_dir, timestamp)
        if vis:
            visualizations.append(vis)
    except Exception as e:
        print(f"⚠️ 特征相关性生成失败: {e}")
    
    # 11. 决策边界可视化
    try:
        vis = visualize_decision_boundary(features_dict, output_dir, timestamp)
        if vis:
            visualizations.append(vis)
    except Exception as e:
        print(f"⚠️ 决策边界生成失败: {e}")
    
    print("\n" + "=" * 80)
    print("✅ 所有补充可视化已完成！")
    print("=" * 80)
    print(f"输出目录: {output_dir}")
    print(f"时间戳: {timestamp}")
    print(f"\n成功生成 {len(visualizations)} 个可视化文件")


if __name__ == '__main__':
    main()

