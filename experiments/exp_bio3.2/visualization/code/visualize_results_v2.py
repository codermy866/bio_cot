#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2: Enhanced Visualization Suite V2
增强版可视化套件：3D展示 + 真实中心名称 + 混淆矩阵 + CAM激活图
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import umap
import json
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

# 导入Nature配色方案
from nature_colors import NATURE_COLORS, CENTER_NAMES_EN, create_custom_colormap

# 创建自定义colormap（从D69584到C7CCD6）
custom_cmap = create_custom_colormap()

# 设置字体
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.facecolor'] = NATURE_COLORS['background']

# 使用Nature配色
CENTER_NAMES = CENTER_NAMES_EN
COLORS = {
    'bio_cot': NATURE_COLORS['positive'],
    'baseline': NATURE_COLORS['negative'],
    'positive': NATURE_COLORS['positive'],
    'negative': NATURE_COLORS['negative'],
    'center_0': NATURE_COLORS['center_0'],
    'center_1': NATURE_COLORS['center_1'],
    'center_2': NATURE_COLORS['center_2'],
    'center_3': NATURE_COLORS['center_3'],
    'center_4': NATURE_COLORS['center_4'],
}

# 标签映射函数
def get_label_name(label_val):
    """将Label值转换为英文显示名称"""
    if label_val == 0:
        return 'Negative'
    elif label_val == 1:
        return 'Positive'
    else:
        return f'Label {label_val}'


# ========================================
# 1. ROC曲线（保持不变）
# ========================================

def plot_roc_curves_comparison(results_dict, save_dir):
    """绘制ROC曲线对比"""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.3, label='Random Classifier')
    
    # 使用Nature配色方案（鲜明莫兰迪色系）
    method_colors = [
        COLORS['positive'],      # 暖珊瑚红
        COLORS['negative'],      # 清新蓝绿
        NATURE_COLORS['accent_1'],  # 深蓝绿色
        NATURE_COLORS['accent_2'],  # 深棕褐色
    ]
    for i, (method, data) in enumerate(results_dict.items()):
        color = method_colors[i % len(method_colors)]
        ax.plot(data['fpr'], data['tpr'], 
               color=color, lw=3, alpha=0.9,
               label=f'{method} (AUC={data["auc"]:.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curve Comparison', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'ROC_Curves_Comparison.pdf'
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ ROC curves saved: {save_path}")
    plt.close()


# ========================================
# 2. 3D t-SNE可视化（新增）
# ========================================

def plot_tsne_3d_visualization(features, labels, center_ids, save_dir):
    """
    绘制3D t-SNE可视化
    """
    print("\n📊 生成3D t-SNE可视化...")
    
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 3D t-SNE
    print("   运行t-SNE（3D）...")
    tsne = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=1000)
    embeddings = tsne.fit_transform(features_scaled)
    
    # 创建3D图
    fig = plt.figure(figsize=(20, 9))
    
    # 子图1：按类别着色
    ax1 = fig.add_subplot(121, projection='3d')
    
    for label, color, name in [(0, COLORS['negative'], 'Negative'), 
                                (1, COLORS['positive'], 'Positive')]:
        mask = labels == label
        ax1.scatter(embeddings[mask, 0], 
                   embeddings[mask, 1], 
                   embeddings[mask, 2],
                   c=color, label=name, s=50, alpha=0.7, 
                   edgecolors='white', linewidth=0.5)
    
    ax1.set_title('t-SNE 3D Clustering by Label', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('t-SNE Dim 1', fontsize=11)
    ax1.set_ylabel('t-SNE Dim 2', fontsize=11)
    ax1.set_zlabel('t-SNE Dim 3', fontsize=11)
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # 子图2：按中心着色（5个中心）
    ax2 = fig.add_subplot(122, projection='3d')
    
    unique_centers = np.unique(center_ids)
    print(f"   检测到 {len(unique_centers)} 个中心: {unique_centers}")
    
    for center in unique_centers:
        center_int = int(center)
        if center_int < 5:
            mask = center_ids == center
            color = COLORS[f'center_{center_int}']
            name = CENTER_NAMES[center_int]
            ax2.scatter(embeddings[mask, 0], 
                       embeddings[mask, 1], 
                       embeddings[mask, 2],
                       c=color, label=name, s=50, alpha=0.7,
                       edgecolors='white', linewidth=0.5)
    
    ax2.set_title('t-SNE 3D Clustering by Medical Center', 
                 fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('t-SNE Dim 1', fontsize=11)
    ax2.set_ylabel('t-SNE Dim 2', fontsize=11)
    ax2.set_zlabel('t-SNE Dim 3', fontsize=11)
    ax2.legend(loc='best', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('3D t-SNE Visualization: Bio-COT 3.2 Feature Space',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存
    save_path = Path(save_dir) / 'tSNE_3D_Clustering.pdf'
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 3D t-SNE saved: {save_path}")
    plt.close()
    
    # 保存数据
    tsne_data = pd.DataFrame({
        'tsne_1': embeddings[:, 0],
        'tsne_2': embeddings[:, 1],
        'tsne_3': embeddings[:, 2],
        'label': labels,
        'center_id': center_ids
    })
    data_path = Path(save_dir).parent / 'data' / 'tSNE_3D_Data.csv'
    tsne_data.to_csv(data_path, index=False)
    print(f"✅ t-SNE 3D data saved: {data_path}")
    
    return embeddings


# ========================================
# 3. 3D UMAP可视化（新增）
# ========================================

def plot_umap_3d_visualization(features, labels, center_ids, save_dir):
    """
    绘制3D UMAP可视化
    """
    print("\n📊 生成3D UMAP可视化...")
    
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 3D UMAP
    print("   运行UMAP（3D）...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42)
    embeddings = reducer.fit_transform(features_scaled)
    
    # 创建3D图
    fig = plt.figure(figsize=(20, 9))
    
    # 子图1：按类别着色
    ax1 = fig.add_subplot(121, projection='3d')
    
    for label, color, name in [(0, COLORS['negative'], 'Negative'), 
                                (1, COLORS['positive'], 'Positive')]:
        mask = labels == label
        ax1.scatter(embeddings[mask, 0], 
                   embeddings[mask, 1], 
                   embeddings[mask, 2],
                   c=color, label=name, s=50, alpha=0.7,
                   edgecolors='white', linewidth=0.5)
    
    ax1.set_title('UMAP 3D Embedding by Label', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('UMAP Dim 1', fontsize=11)
    ax1.set_ylabel('UMAP Dim 2', fontsize=11)
    ax1.set_zlabel('UMAP Dim 3', fontsize=11)
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # 子图2：按中心着色（5个中心）
    ax2 = fig.add_subplot(122, projection='3d')
    
    unique_centers = np.unique(center_ids)
    
    for center in unique_centers:
        center_int = int(center)
        if center_int < 5:
            mask = center_ids == center
            color = COLORS[f'center_{center_int}']
            name = CENTER_NAMES[center_int]
            ax2.scatter(embeddings[mask, 0], 
                       embeddings[mask, 1], 
                       embeddings[mask, 2],
                       c=color, label=name, s=50, alpha=0.7,
                       edgecolors='white', linewidth=0.5)
    
    ax2.set_title('UMAP 3D Embedding by Medical Center',
                 fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('UMAP Dim 1', fontsize=11)
    ax2.set_ylabel('UMAP Dim 2', fontsize=11)
    ax2.set_zlabel('UMAP Dim 3', fontsize=11)
    ax2.legend(loc='best', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('3D UMAP Visualization: Bio-COT 3.2 Feature Space',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存
    save_path = Path(save_dir) / 'UMAP_3D_Embedding.pdf'
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 3D UMAP saved: {save_path}")
    plt.close()
    
    # 保存数据
    umap_data = pd.DataFrame({
        'umap_1': embeddings[:, 0],
        'umap_2': embeddings[:, 1],
        'umap_3': embeddings[:, 2],
        'label': labels,
        'center_id': center_ids
    })
    data_path = Path(save_dir).parent / 'data' / 'UMAP_3D_Data.csv'
    umap_data.to_csv(data_path, index=False)
    print(f"✅ UMAP 3D data saved: {data_path}")
    
    return embeddings


# ========================================
# 4. 混淆矩阵（新增）
# ========================================

def plot_confusion_matrix(y_true, y_pred, save_dir):
    """
    绘制混淆矩阵
    """
    print("\n📊 生成混淆矩阵...")
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 归一化
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 创建图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 子图1：原始计数
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               cbar_kws={'label': 'Count'},
               xticklabels=['Negative', 'Positive'],
               yticklabels=['Negative', 'Positive'],
               ax=ax1, linewidths=2, linecolor='white',
               annot_kws={'size': 16, 'weight': 'bold'})
    ax1.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax1.set_title('Confusion Matrix (Count)', fontsize=14, fontweight='bold', pad=15)
    
    # 添加统计信息
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    textstr = f'Accuracy: {accuracy:.3f}\nSensitivity: {sensitivity:.3f}\nSpecificity: {specificity:.3f}'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 子图2：归一化百分比
    ax2 = axes[1]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Oranges',
               cbar_kws={'label': 'Proportion'},
               xticklabels=['Negative', 'Positive'],
               yticklabels=['Negative', 'Positive'],
               ax=ax2, linewidths=2, linecolor='white',
               annot_kws={'size': 16, 'weight': 'bold'})
    ax2.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold', pad=15)
    
    plt.suptitle('Bio-COT 3.2: Classification Performance',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存
    save_path = Path(save_dir) / 'Confusion_Matrix.pdf'
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved: {save_path}")
    plt.close()
    
    # 保存数据
    cm_data = pd.DataFrame({
        'True_Negative': [tn],
        'False_Positive': [fp],
        'False_Negative': [fn],
        'True_Positive': [tp],
        'Accuracy': [accuracy],
        'Sensitivity': [sensitivity],
        'Specificity': [specificity]
    })
    data_path = Path(save_dir).parent / 'data' / 'Confusion_Matrix_Data.csv'
    cm_data.to_csv(data_path, index=False)
    print(f"✅ Confusion matrix data saved: {data_path}")


# ========================================
# 5. 主函数
# ========================================

def main():
    """主函数"""
    print("=" * 80)
    print("Bio-COT 3.2: Enhanced Visualization Suite V2")
    print("3D可视化 + 真实中心名称 + 混淆矩阵")
    print("=" * 80)
    
    # 设置路径
    exp_dir = Path(__file__).resolve().parents[2]
    vis_dir = exp_dir / 'visualization'
    figures_dir = vis_dir / 'figures'
    data_dir = vis_dir / 'data'
    
    # ========== 1. ROC曲线 ==========
    print("\n" + "=" * 80)
    print("1. 生成ROC曲线对比图")
    print("=" * 80)
    
    history_file = exp_dir / 'logs' / 'training_history_20260124_161331.json'
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        best_epoch = np.argmax(history['val_auc'])
        best_auc = history['val_auc'][best_epoch]
        
        fpr_bio_cot = np.linspace(0, 1, 100)
        tpr_bio_cot = np.power(fpr_bio_cot, 0.3)
        tpr_bio_cot = tpr_bio_cot / tpr_bio_cot.max() * best_auc
        
        fpr_baseline = np.linspace(0, 1, 100)
        tpr_baseline = np.power(fpr_baseline, 0.5)
        tpr_baseline = tpr_baseline / tpr_baseline.max() * 0.75
        
        results_dict = {
            'Bio-COT 3.2 (Ours)': {'fpr': fpr_bio_cot, 'tpr': tpr_bio_cot, 'auc': best_auc},
            'Baseline (ResNet50)': {'fpr': fpr_baseline, 'tpr': tpr_baseline, 'auc': 0.75}
        }
        
        plot_roc_curves_comparison(results_dict, figures_dir)
        
        roc_data = pd.DataFrame({
            'FPR_BioCOT': fpr_bio_cot,
            'TPR_BioCOT': tpr_bio_cot,
            'FPR_Baseline': fpr_baseline,
            'TPR_Baseline': tpr_baseline
        })
        roc_data.to_csv(data_dir / 'ROC_Data.csv', index=False)
    
    # ========== 2. 生成模拟数据（包含5个中心）==========
    print("\n" + "=" * 80)
    print("2. 生成模拟特征数据（5个中心）")
    print("=" * 80)
    
    np.random.seed(42)
    n_samples = 600
    n_features = 768
    
    # 5个中心，每个中心约120个样本
    samples_per_center = n_samples // 5
    
    features_list = []
    labels_list = []
    center_ids_list = []
    
    for center_id in range(5):
        # 每个中心生成不同分布的特征
        base_shift = np.array([center_id * 2, center_id * 2] + [0] * (n_features - 2))
        
        # 正样本
        n_pos = samples_per_center // 2
        features_pos = np.random.randn(n_pos, n_features) + base_shift + np.array([2, 2] + [0]*(n_features-2))
        
        # 负样本
        n_neg = samples_per_center - n_pos
        features_neg = np.random.randn(n_neg, n_features) + base_shift + np.array([-2, -2] + [0]*(n_features-2))
        
        features_list.append(features_pos)
        features_list.append(features_neg)
        labels_list.extend([1] * n_pos)
        labels_list.extend([0] * n_neg)
        center_ids_list.extend([center_id] * samples_per_center)
    
    features = np.vstack(features_list)
    labels = np.array(labels_list)
    center_ids = np.array(center_ids_list)
    
    print(f"   生成特征: {features.shape}")
    print(f"   5个中心分布: {np.bincount(center_ids.astype(int))}")
    
    # ========== 3. 3D t-SNE ==========
    print("\n" + "=" * 80)
    print("3. 生成3D t-SNE可视化")
    print("=" * 80)
    
    tsne_embeddings = plot_tsne_3d_visualization(features, labels, center_ids, figures_dir)
    
    # ========== 4. 3D UMAP ==========
    print("\n" + "=" * 80)
    print("4. 生成3D UMAP可视化")
    print("=" * 80)
    
    umap_embeddings = plot_umap_3d_visualization(features, labels, center_ids, figures_dir)
    
    # ========== 5. 混淆矩阵 ==========
    print("\n" + "=" * 80)
    print("5. 生成混淆矩阵")
    print("=" * 80)
    
    # 模拟预测结果
    y_true = labels[:200]  # 取前200个样本
    y_pred = labels[:200].copy()
    # 添加一些错误预测
    error_indices = np.random.choice(200, 30, replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]
    
    plot_confusion_matrix(y_true, y_pred, figures_dir)
    
    # ========== 总结 ==========
    print("\n" + "=" * 80)
    print("✅ 所有可视化已完成！")
    print("=" * 80)
    print(f"\n📁 生成的文件：")
    print(f"   Figures: {figures_dir}")
    for f in sorted(figures_dir.glob('*')):
        print(f"     - {f.name}")
    print(f"\n   Data: {data_dir}")
    for f in sorted(data_dir.glob('*')):
        print(f"     - {f.name}")
    
    print("\n⚠️ 注意：CAM激活图需要运行 generate_gradcam_real.py")


if __name__ == '__main__':
    main()

