#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成高级可视化图表（仿照参考风格）
包括：时间序列图、散点图矩阵、逻辑回归图、热图等
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.manifold import TSNE
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from scipy import stats
from scipy.stats import gaussian_kde
import json
from datetime import datetime
import pickle
from tqdm import tqdm

# 设置matplotlib后端
matplotlib.use('Agg')

# 设置seaborn风格（仿照参考图）
sns.set_theme(style="whitegrid", palette="husl")
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'pdf.fonttype': 42,
})

# 添加项目路径
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from config import BioCOT_v3_Config
from models.bio_cot_v3 import BioCOT_v3
from data.dataset_v3 import FiveCentersMultimodalDatasetV3
from training.extract_vit_patches import extract_patch_features_with_vit
from torch.utils.data import DataLoader
from torchvision import transforms


def load_model_and_features(config, device):
    """加载最新模型并提取特征"""
    print("📥 加载模型和提取特征...")
    
    # 加载最新检查点
    checkpoint_files = sorted(Path(config.checkpoint_dir).glob("best_model_v3_*.pth"),
                             key=lambda x: x.stat().st_mtime, reverse=True)
    if not checkpoint_files:
        raise FileNotFoundError("未找到模型检查点")
    
    checkpoint_path = checkpoint_files[0]
    print(f"✅ 加载模型: {checkpoint_path.name}")
    
    model = BioCOT_v3(
        embed_dim=config.embed_dim,
        num_classes=config.num_classes,
        num_centers=config.num_centers,
        input_dim=config.input_dim,
        use_visual_notes=config.use_visual_notes,
        use_ot=config.use_ot,
        use_dual=config.use_dual,
        use_cross_attn=config.use_cross_attn,
        warmup_epochs=config.warmup_epochs
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # 加载数据集
    data_root = Path(config.data_root)
    val_csv = data_root / 'internal_val' / 'labels.csv'
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = FiveCentersMultimodalDatasetV3(
        csv_path=str(val_csv),
        knowledge_embed_path=config.knowledge_embed_path,
        transform=transform,
        oct_num_frames=config.oct_frames,
        max_col_images=config.colposcopy_images,
        balance_negative_frames=True
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # 提取特征
    features_dict = {
        'z_causal': [],
        'z_noise': [],
        'z_sem': [],
        'attn_oct': [],
        'attn_colpo': [],
        'predictions': [],
        'probabilities': [],
        'labels': [],
        'centers': [],
        'knowledge_embeds': [],
        'oct_features': [],
        'colpo_features': []
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取特征"):
            oct_images = batch['oct_images'].to(device)
            colposcopy_images = batch['colposcopy_images'].to(device)
            labels = batch['label'].to(device)
            center_labels = batch['center_idx'].to(device)
            knowledge_embeddings = batch['knowledge_embedding'].to(device)
            
            # 提取patch特征
            B, F_oct = oct_images.shape[0], oct_images.shape[1]
            oct_images_flat = oct_images.view(B * F_oct, *oct_images.shape[2:])
            oct_feats_flat = extract_patch_features_with_vit(oct_images_flat, device)
            oct_feats = oct_feats_flat.view(B, F_oct, *oct_feats_flat.shape[1:]).mean(dim=1)
            
            B, N_colpo = colposcopy_images.shape[0], colposcopy_images.shape[1]
            colpo_images_flat = colposcopy_images.view(B * N_colpo, *colposcopy_images.shape[2:])
            colpo_feats_flat = extract_patch_features_with_vit(colpo_images_flat, device)
            colpo_feats = colpo_feats_flat.view(B, N_colpo, *colpo_feats_flat.shape[1:]).mean(dim=1)
            
            # 前向传播
            outputs = model(
                f_oct=oct_feats,
                f_colpo=colpo_feats,
                note_embeds=knowledge_embeddings,
                center_labels=center_labels,
                return_loss_components=False
            )
            
            # 收集特征
            if 'z_causal' in outputs:
                features_dict['z_causal'].append(outputs['z_causal'].cpu().numpy())
            if 'z_noise' in outputs:
                features_dict['z_noise'].append(outputs['z_noise'].cpu().numpy())
            if 'z_sem' in outputs:
                features_dict['z_sem'].append(outputs['z_sem'].cpu().numpy())
            if 'attn_maps' in outputs and len(outputs['attn_maps']) >= 2:
                features_dict['attn_oct'].append(outputs['attn_maps'][0].cpu().numpy())
                features_dict['attn_colpo'].append(outputs['attn_maps'][1].cpu().numpy())
            
            features_dict['oct_features'].append(oct_feats.cpu().numpy())
            features_dict['colpo_features'].append(colpo_feats.cpu().numpy())
            
            probs = torch.softmax(outputs['pred'], dim=1).cpu().numpy()
            features_dict['predictions'].append(torch.argmax(outputs['pred'], dim=1).cpu().numpy())
            features_dict['probabilities'].append(probs)
            features_dict['labels'].append(labels.cpu().numpy())
            features_dict['centers'].append(center_labels.cpu().numpy())
            features_dict['knowledge_embeds'].append(knowledge_embeddings.cpu().numpy())
    
    # 合并所有batch
    for key in features_dict:
        if features_dict[key]:
            if len(features_dict[key][0].shape) > 1:
                features_dict[key] = np.vstack(features_dict[key])
            else:
                features_dict[key] = np.hstack(features_dict[key])
    
    return features_dict, model


def visualize_timeseries_with_error_bands(features_dict, history_file, output_dir, timestamp):
    """生成时间序列图（带误差带）- 仿照参考图1"""
    print("📊 生成时间序列图（带误差带）...")
    
    if not Path(history_file).exists():
        print("⚠️ 未找到训练历史文件")
        return None
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = np.array(range(1, len(history['train_loss']) + 1))
    
    # 创建DataFrame
    df = pd.DataFrame({
        'epoch': np.tile(epochs, 2),
        'metric': ['train'] * len(epochs) + ['val'] * len(epochs),
        'loss': history['train_loss'] + history['val_loss'],
        'accuracy': history['train_acc'] + history['val_acc'],
        'auc': [0] * len(epochs) + history['val_auc']
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Loss时间序列
    ax1 = axes[0]
    for metric in ['train', 'val']:
        data = df[df['metric'] == metric]
        mean_vals = data['loss'].values
        # 计算误差带（使用移动标准差）
        window = 5
        std_vals = pd.Series(mean_vals).rolling(window=window, center=True).std().fillna(0)
        
        ax1.plot(data['epoch'], mean_vals, label=f'{metric.capitalize()} Loss', linewidth=2)
        ax1.fill_between(data['epoch'], 
                        mean_vals - std_vals, 
                        mean_vals + std_vals, 
                        alpha=0.3)
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Timeseries plot with error bands: Loss', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy时间序列
    ax2 = axes[1]
    for metric in ['train', 'val']:
        data = df[df['metric'] == metric]
        mean_vals = data['accuracy'].values
        std_vals = pd.Series(mean_vals).rolling(window=window, center=True).std().fillna(0)
        
        ax2.plot(data['epoch'], mean_vals, label=f'{metric.capitalize()} Accuracy', linewidth=2)
        ax2.fill_between(data['epoch'], 
                        mean_vals - std_vals, 
                        mean_vals + std_vals, 
                        alpha=0.3)
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Timeseries plot with error bands: Accuracy', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. AUC时间序列
    ax3 = axes[2]
    data = df[df['metric'] == 'val']
    mean_vals = data['auc'].values
    std_vals = pd.Series(mean_vals).rolling(window=window, center=True).std().fillna(0)
    
    ax3.plot(data['epoch'], mean_vals, label='Validation AUC', linewidth=2, color='green')
    ax3.fill_between(data['epoch'], 
                     mean_vals - std_vals, 
                     mean_vals + std_vals, 
                     alpha=0.3, color='green')
    
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax3.set_title('Timeseries plot with error bands: AUC', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    plt.tight_layout()
    
    output_path_pdf = output_dir / f"timeseries_error_bands_{timestamp}.pdf"
    output_path_png = output_dir / f"timeseries_error_bands_{timestamp}.png"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 时间序列图已保存: {output_path_pdf}")
    return output_path_pdf


def visualize_multiple_linear_regression(features_dict, output_dir, timestamp):
    """生成多元线性回归图 - 仿照参考图2"""
    print("📊 生成多元线性回归图...")
    
    z_causal = features_dict['z_causal']
    labels = features_dict['labels']
    centers = features_dict['centers']
    probs = features_dict['probabilities'][:, 1] if 'probabilities' in features_dict else None
    
    # 使用PCA降维到2D
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_causal)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'PC1': z_2d[:, 0],
        'PC2': z_2d[:, 1],
        'label': labels,
        'center': centers,
        'probability': probs if probs is not None else np.zeros(len(labels))
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. 按标签分组
    ax1 = axes[0]
    for label_val, color, label_name in [(0, 'red', 'Negative'), (1, 'green', 'Positive')]:
        data = df[df['label'] == label_val]
        ax1.scatter(data['PC1'], data['PC2'], c=color, label=label_name, 
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # 添加回归线
        if len(data) > 1:
            z = np.polyfit(data['PC1'], data['PC2'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data['PC1'].min(), data['PC1'].max(), 100)
            ax1.plot(x_line, p(x_line), color=color, linestyle='--', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('PC1 (Principal Component 1)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PC2 (Principal Component 2)', fontsize=12, fontweight='bold')
    ax1.set_title('Multiple linear regression: By Label', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 按中心分组
    ax2 = axes[1]
    unique_centers = np.unique(centers)
    colors_center = sns.color_palette("husl", len(unique_centers))
    
    for i, center_id in enumerate(unique_centers):
        data = df[df['center'] == center_id]
        ax2.scatter(data['PC1'], data['PC2'], c=[colors_center[i]], 
                   label=f'Center {center_id}', alpha=0.6, s=50, 
                   edgecolors='black', linewidth=0.5)
        
        # 添加回归线
        if len(data) > 1:
            z = np.polyfit(data['PC1'], data['PC2'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data['PC1'].min(), data['PC1'].max(), 100)
            ax2.plot(x_line, p(x_line), color=colors_center[i], linestyle='--', 
                    linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('PC1 (Principal Component 1)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PC2 (Principal Component 2)', fontsize=12, fontweight='bold')
    ax2.set_title('Multiple linear regression: By Center', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path_pdf = output_dir / f"multiple_regression_{timestamp}.pdf"
    output_path_png = output_dir / f"multiple_regression_{timestamp}.png"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 多元线性回归图已保存: {output_path_pdf}")
    return output_path_pdf


def visualize_scatterplot_matrix(features_dict, output_dir, timestamp):
    """生成散点图矩阵 - 仿照参考图3"""
    print("📊 生成散点图矩阵...")
    
    z_causal = features_dict['z_causal']
    labels = features_dict['labels']
    centers = features_dict['centers']
    probs = features_dict['probabilities'][:, 1] if 'probabilities' in features_dict else None
    
    # 选择前4个最重要的特征（使用方差）
    variances = np.var(z_causal, axis=0)
    top_indices = np.argsort(variances)[-4:]
    z_selected = z_causal[:, top_indices]
    
    # 创建DataFrame
    df = pd.DataFrame({
        f'Feature_{i+1}': z_selected[:, i] for i in range(4)
    })
    df['label'] = labels
    df['center'] = centers
    df['probability'] = probs if probs is not None else np.zeros(len(labels))
    
    # 创建散点图矩阵
    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    feature_names = [f'Feature_{i+1}' for i in range(4)]
    
    for i in range(4):
        for j in range(4):
            ax = fig.add_subplot(gs[i, j])
            
            if i == j:
                # 对角线：KDE图
                for label_val, color in [(0, 'red'), (1, 'green')]:
                    data = df[df['label'] == label_val][feature_names[i]]
                    if len(data) > 0:
                        ax.hist(data, bins=30, alpha=0.6, color=color, 
                               label='Positive' if label_val == 1 else 'Negative',
                               density=True, edgecolor='black')
                ax.set_xlabel(feature_names[i], fontsize=10)
                ax.set_ylabel('Density', fontsize=10)
                if i == 0:
                    ax.legend(fontsize=8)
            else:
                # 非对角线：散点图
                scatter = ax.scatter(df[feature_names[j]], df[feature_names[i]], 
                                   c=df['label'], cmap='RdYlGn', alpha=0.6, s=30,
                                   edgecolors='black', linewidth=0.3)
                ax.set_xlabel(feature_names[j], fontsize=10)
                ax.set_ylabel(feature_names[i], fontsize=10)
            
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.set_title('Scatterplot Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_path_pdf = output_dir / f"scatterplot_matrix_{timestamp}.pdf"
    output_path_png = output_dir / f"scatterplot_matrix_{timestamp}.png"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 散点图矩阵已保存: {output_path_pdf}")
    return output_path_pdf


def visualize_faceted_logistic_regression(features_dict, output_dir, timestamp):
    """生成分面逻辑回归图 - 仿照参考图4"""
    print("📊 生成分面逻辑回归图...")
    
    z_causal = features_dict['z_causal']
    labels = features_dict['labels']
    centers = features_dict['centers']
    probs = features_dict['probabilities'][:, 1] if 'probabilities' in features_dict else None
    
    if probs is None:
        print("⚠️ 未找到预测概率，跳过逻辑回归图")
        return None
    
    # 使用PCA降维到1D（用于x轴）
    pca = PCA(n_components=1)
    z_1d = pca.fit_transform(z_causal).flatten()
    
    # 创建DataFrame
    df = pd.DataFrame({
        'feature': z_1d,
        'label': labels,
        'center': centers,
        'probability': probs
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. 按中心分面
    unique_centers = np.unique(centers)
    colors_center = sns.color_palette("husl", len(unique_centers))
    
    for idx, center_id in enumerate(unique_centers):
        ax = axes[0]
        data = df[df['center'] == center_id]
        
        # 散点图
        ax.scatter(data['feature'], data['probability'], 
                  c=[colors_center[idx]], alpha=0.6, s=50,
                  edgecolors='black', linewidth=0.5,
                  label=f'Center {center_id}')
        
        # 逻辑回归拟合
        if len(data) > 1:
            X = data[['feature']].values
            y = data['probability'].values
            
            # 使用多项式拟合（模拟逻辑回归曲线）
            z = np.polyfit(data['feature'], data['probability'], 3)
            p = np.poly1d(z)
            x_line = np.linspace(data['feature'].min(), data['feature'].max(), 100)
            y_line = p(x_line)
            y_line = np.clip(y_line, 0, 1)  # 限制在[0,1]
            
            ax.plot(x_line, y_line, color=colors_center[idx], 
                   linestyle='-', linewidth=2, alpha=0.8)
    
    axes[0].set_xlabel('Feature (PC1)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Prediction Probability', fontsize=12, fontweight='bold')
    axes[0].set_title('Faceted logistic regression: By Center', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # 2. 按标签分面
    for label_val, color, label_name in [(0, 'red', 'Negative'), (1, 'green', 'Positive')]:
        ax = axes[1]
        data = df[df['label'] == label_val]
        
        # 散点图
        ax.scatter(data['feature'], data['probability'], 
                  c=color, alpha=0.6, s=50,
                  edgecolors='black', linewidth=0.5,
                  label=label_name)
        
        # 逻辑回归拟合
        if len(data) > 1:
            z = np.polyfit(data['feature'], data['probability'], 3)
            p = np.poly1d(z)
            x_line = np.linspace(data['feature'].min(), data['feature'].max(), 100)
            y_line = p(x_line)
            y_line = np.clip(y_line, 0, 1)
            
            ax.plot(x_line, y_line, color=color, 
                   linestyle='-', linewidth=2, alpha=0.8)
    
    axes[1].set_xlabel('Feature (PC1)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Prediction Probability', fontsize=12, fontweight='bold')
    axes[1].set_title('Faceted logistic regression: By Label', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    output_path_pdf = output_dir / f"faceted_logistic_{timestamp}.pdf"
    output_path_png = output_dir / f"faceted_logistic_{timestamp}.png"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 分面逻辑回归图已保存: {output_path_pdf}")
    return output_path_pdf


def visualize_heatmap_with_clustering(features_dict, output_dir, timestamp):
    """生成带聚类的热图 - 仿照参考图5"""
    print("📊 生成带聚类的热图...")
    
    z_causal = features_dict['z_causal']
    labels = features_dict['labels']
    centers = features_dict['centers']
    
    # 选择前30个最重要的特征（避免计算量过大）
    variances = np.var(z_causal, axis=0)
    top_indices = np.argsort(variances)[-30:]
    z_selected = z_causal[:, top_indices]
    
    # 计算相关性矩阵
    corr_matrix = np.corrcoef(z_selected.T)
    
    # 使用seaborn的clustermap
    fig = plt.figure(figsize=(14, 12))
    
    # 创建聚类热图
    g = sns.clustermap(corr_matrix, 
                      cmap='RdBu_r', 
                      center=0,
                      vmin=-1, vmax=1,
                      figsize=(14, 12),
                      cbar_kws={'label': 'Correlation Coefficient'},
                      linewidths=0.5,
                      method='ward',
                      metric='euclidean')
    
    g.fig.suptitle('Discovering structure in heatmap data', 
                   fontsize=14, fontweight='bold', y=0.98)
    
    # 保存
    output_path_pdf = output_dir / f"heatmap_clustered_{timestamp}.pdf"
    output_path_png = output_dir / f"heatmap_clustered_{timestamp}.png"
    g.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    g.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 聚类热图已保存: {output_path_pdf}")
    return output_path_pdf


def visualize_scatterplot_multiple_semantics(features_dict, output_dir, timestamp):
    """生成多语义散点图 - 仿照参考图6"""
    print("📊 生成多语义散点图...")
    
    z_causal = features_dict['z_causal']
    labels = features_dict['labels']
    centers = features_dict['centers']
    probs = features_dict['probabilities'][:, 1] if 'probabilities' in features_dict else None
    attn_oct = features_dict.get('attn_oct', None)
    
    # 使用PCA降维到2D
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_causal)
    
    # 计算注意力均值（作为大小）
    if attn_oct is not None and len(attn_oct) > 0:
        attn_mean = np.mean(attn_oct.reshape(attn_oct.shape[0], -1), axis=1)
        # 归一化到合理范围
        attn_mean = (attn_mean - attn_mean.min()) / (attn_mean.max() - attn_mean.min() + 1e-8)
        attn_mean = attn_mean * 100 + 20  # 映射到20-120
    else:
        attn_mean = np.ones(len(labels)) * 50
    
    # 创建DataFrame
    df = pd.DataFrame({
        'PC1': z_2d[:, 0],
        'PC2': z_2d[:, 1],
        'label': labels,
        'center': centers,
        'probability': probs if probs is not None else np.zeros(len(labels)),
        'attention': attn_mean
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. 颜色=标签，大小=注意力
    ax1 = axes[0]
    scatter1 = ax1.scatter(df['PC1'], df['PC2'], 
                          c=df['label'], cmap='RdYlGn',
                          s=df['attention'], alpha=0.6,
                          edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('PC1 (Principal Component 1)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PC2 (Principal Component 2)', fontsize=12, fontweight='bold')
    ax1.set_title('Scatterplot with multiple semantics: Label (color) & Attention (size)', 
                 fontsize=13, fontweight='bold')
    plt.colorbar(scatter1, ax=ax1, label='Label (0=Neg, 1=Pos)')
    ax1.grid(True, alpha=0.3)
    
    # 2. 颜色=概率，大小=中心
    ax2 = axes[1]
    if probs is not None:
        scatter2 = ax2.scatter(df['PC1'], df['PC2'], 
                              c=df['probability'], cmap='RdYlGn',
                              s=df['center']*20 + 30, alpha=0.6,
                              edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('PC1 (Principal Component 1)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('PC2 (Principal Component 2)', fontsize=12, fontweight='bold')
        ax2.set_title('Scatterplot with multiple semantics: Probability (color) & Center (size)', 
                     fontsize=13, fontweight='bold')
        plt.colorbar(scatter2, ax=ax2, label='Prediction Probability')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path_pdf = output_dir / f"scatterplot_multiple_semantics_{timestamp}.pdf"
    output_path_png = output_dir / f"scatterplot_multiple_semantics_{timestamp}.png"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 多语义散点图已保存: {output_path_pdf}")
    return output_path_pdf


def visualize_kde_palettes(features_dict, output_dir, timestamp):
    """生成KDE调色板图 - 仿照参考图7"""
    print("📊 生成KDE调色板图...")
    
    z_causal = features_dict['z_causal']
    labels = features_dict['labels']
    
    # 使用PCA降维到2D
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_causal)
    
    # 创建不同调色板的KDE图
    palettes = ['husl', 'Set2', 'Paired', 'tab10', 'viridis', 'plasma', 
                'coolwarm', 'RdYlGn', 'Spectral']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    
    for idx, palette_name in enumerate(palettes):
        ax = axes[idx]
        
        # 为每个标签创建KDE
        for label_val in [0, 1]:
            data = z_2d[labels == label_val]
            if len(data) > 0:
                # 创建KDE
                kde = gaussian_kde(data.T)
                x_min, x_max = z_2d[:, 0].min(), z_2d[:, 0].max()
                y_min, y_max = z_2d[:, 1].min(), z_2d[:, 1].max()
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                                    np.linspace(y_min, y_max, 50))
                positions = np.vstack([xx.ravel(), yy.ravel()])
                zz = kde(positions).reshape(xx.shape)
                
                # 使用不同的调色板
                colors = sns.color_palette(palette_name, 2)
                ax.contourf(xx, yy, zz, levels=20, cmap=palette_name, alpha=0.6)
        
        ax.set_title(f'Different {palette_name} palettes', fontsize=11, fontweight='bold')
        ax.set_xlabel('PC1', fontsize=10)
        ax.set_ylabel('PC2', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Kernel Density Estimates with Different Color Palettes', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path_pdf = output_dir / f"kde_palettes_{timestamp}.pdf"
    output_path_png = output_dir / f"kde_palettes_{timestamp}.png"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ KDE调色板图已保存: {output_path_pdf}")
    return output_path_pdf


def main():
    """主函数"""
    print("=" * 80)
    print("生成高级可视化图表（仿照参考风格）")
    print("=" * 80)
    
    # 配置
    config = BioCOT_v3_Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型和特征
    features_dict, model = load_model_and_features(config, device)
    
    # 查找最新的训练历史文件
    history_files = sorted(Path(config.log_dir).glob("training_history_*.json"),
                          key=lambda x: x.stat().st_mtime, reverse=True)
    history_file = str(history_files[0]) if history_files else None
    
    # 生成所有可视化
    print("\n" + "=" * 80)
    print("生成可视化图表...")
    print("=" * 80)
    
    visualizations = []
    
    # 1. 时间序列图（带误差带）
    if history_file:
        try:
            vis = visualize_timeseries_with_error_bands(features_dict, history_file, output_dir, timestamp)
            if vis:
                visualizations.append(vis)
        except Exception as e:
            print(f"⚠️ 时间序列图生成失败: {e}")
    
    # 2. 多元线性回归图
    try:
        vis = visualize_multiple_linear_regression(features_dict, output_dir, timestamp)
        if vis:
            visualizations.append(vis)
    except Exception as e:
        print(f"⚠️ 多元线性回归图生成失败: {e}")
    
    # 3. 散点图矩阵
    try:
        vis = visualize_scatterplot_matrix(features_dict, output_dir, timestamp)
        if vis:
            visualizations.append(vis)
    except Exception as e:
        print(f"⚠️ 散点图矩阵生成失败: {e}")
    
    # 4. 分面逻辑回归图
    try:
        vis = visualize_faceted_logistic_regression(features_dict, output_dir, timestamp)
        if vis:
            visualizations.append(vis)
    except Exception as e:
        print(f"⚠️ 分面逻辑回归图生成失败: {e}")
    
    # 5. 聚类热图
    try:
        vis = visualize_heatmap_with_clustering(features_dict, output_dir, timestamp)
        if vis:
            visualizations.append(vis)
    except Exception as e:
        print(f"⚠️ 聚类热图生成失败: {e}")
    
    # 6. 多语义散点图
    try:
        vis = visualize_scatterplot_multiple_semantics(features_dict, output_dir, timestamp)
        if vis:
            visualizations.append(vis)
    except Exception as e:
        print(f"⚠️ 多语义散点图生成失败: {e}")
    
    # 7. KDE调色板图
    try:
        vis = visualize_kde_palettes(features_dict, output_dir, timestamp)
        if vis:
            visualizations.append(vis)
    except Exception as e:
        print(f"⚠️ KDE调色板图生成失败: {e}")
    
    print("\n" + "=" * 80)
    print("✅ 所有高级可视化已完成！")
    print("=" * 80)
    print(f"输出目录: {output_dir}")
    print(f"时间戳: {timestamp}")
    print(f"\n成功生成 {len(visualizations)} 个可视化文件")
    print("\n生成的文件:")
    for vis in visualizations:
        print(f"  - {Path(vis).name}")


if __name__ == '__main__':
    main()

