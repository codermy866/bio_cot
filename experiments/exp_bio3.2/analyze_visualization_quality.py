#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化图表质量分析报告
分析 Linear_Regression_Marginal 和 Scatterplot_Matrix 两张图
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
DATA_FILE = ROOT / 'visualization' / 'data' / 'Complete_Dataset.csv'

print("=" * 80)
print("📊 可视化图表质量分析报告")
print("=" * 80)

# 加载数据进行分析
if DATA_FILE.exists():
    df = pd.read_csv(DATA_FILE)
    feature_cols = [f'feature_{i}' for i in range(10) if f'feature_{i}' in df.columns]
    
    if feature_cols:
        features = df[feature_cols].values
        labels = df['label'].values if 'label' in df.columns else None
        
        print(f"\n📈 数据概览:")
        print(f"  样本数量: {len(features)}")
        print(f"  特征数量: {len(feature_cols)}")
        if labels is not None:
            print(f"  标签分布: Negative={np.sum(labels==0)}, Positive={np.sum(labels==1)}")
        
        # 分析特征对的相关性
        if len(feature_cols) >= 2:
            feat0 = features[:, 0]
            feat1 = features[:, 1]
            
            from scipy.stats import pearsonr, linregress
            r, p = pearsonr(feat0, feat1)
            slope, intercept, r_value, p_value, std_err = linregress(feat0, feat1)
            
            print(f"\n📊 Linear_Regression_Marginal 分析:")
            print(f"  特征对: feature_0 vs feature_1")
            print(f"  Pearson相关系数: r = {r:.4f}, p = {p:.4f}")
            print(f"  线性回归: R² = {r_value**2:.4f}, p = {p_value:.4f}")
            print(f"  回归系数: slope = {slope:.4f}, intercept = {intercept:.4f}")
            print(f"  标准误差: {std_err:.4f}")
            
            # 评估回归质量
            if r_value**2 > 0.7:
                quality = "优秀"
            elif r_value**2 > 0.5:
                quality = "良好"
            elif r_value**2 > 0.3:
                quality = "中等"
            else:
                quality = "较弱"
            
            print(f"  回归质量评估: {quality} (R² = {r_value**2:.3f})")
            
            # 分析特征分布
            print(f"\n  特征分布分析:")
            print(f"    feature_0: 均值={feat0.mean():.4f}, 标准差={feat0.std():.4f}, 范围=[{feat0.min():.4f}, {feat0.max():.4f}]")
            print(f"    feature_1: 均值={feat1.mean():.4f}, 标准差={feat1.std():.4f}, 范围=[{feat1.min():.4f}, {feat1.max():.4f}]")
            
            # 检查异常值
            from scipy import stats
            z_scores_0 = np.abs(stats.zscore(feat0))
            z_scores_1 = np.abs(stats.zscore(feat1))
            outliers_0 = np.sum(z_scores_0 > 3)
            outliers_1 = np.sum(z_scores_1 > 3)
            print(f"  异常值检测 (|z| > 3): feature_0有{outliers_0}个, feature_1有{outliers_1}个")
        
        # 分析Scatterplot Matrix
        if len(feature_cols) >= 6:
            print(f"\n📊 Scatterplot_Matrix 分析:")
            print(f"  包含特征数量: 6个")
            print(f"  矩阵大小: 6x6 = 36个子图")
            
            # 计算特征间的相关性矩阵
            corr_matrix = np.corrcoef(features[:, :6].T)
            print(f"\n  特征相关性矩阵 (前6个特征):")
            print(f"    平均绝对相关性: {np.abs(corr_matrix[np.triu_indices(6, k=1)]).mean():.4f}")
            print(f"    最大相关性: {np.abs(corr_matrix[np.triu_indices(6, k=1)]).max():.4f}")
            print(f"    最小相关性: {np.abs(corr_matrix[np.triu_indices(6, k=1)]).min():.4f}")
            
            # 检查多重共线性
            high_corr_pairs = []
            for i in range(6):
                for j in range(i+1, 6):
                    if abs(corr_matrix[i, j]) > 0.7:
                        high_corr_pairs.append((i, j, corr_matrix[i, j]))
            
            if high_corr_pairs:
                print(f"  高相关性特征对 (|r| > 0.7): {len(high_corr_pairs)}对")
                for i, j, r in high_corr_pairs[:3]:
                    print(f"    feature_{i} vs feature_{j}: r = {r:.4f}")
            else:
                print(f"  高相关性特征对: 无 (所有特征对相关性 < 0.7)")
            
            # 分析标签分离度
            if labels is not None:
                print(f"\n  标签分离度分析:")
                for i in range(min(6, len(feature_cols))):
                    feat_neg = features[labels == 0, i]
                    feat_pos = features[labels == 1, i]
                    if len(feat_neg) > 0 and len(feat_pos) > 0:
                        from scipy.stats import ttest_ind
                        t_stat, p_val = ttest_ind(feat_neg, feat_pos)
                        mean_diff = feat_pos.mean() - feat_neg.mean()
                        print(f"    feature_{i}: 均值差={mean_diff:.4f}, t={t_stat:.4f}, p={p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''}")

print("\n" + "=" * 80)
print("✅ 分析完成")
print("=" * 80)

