#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.0 3D可视化生成脚本
包括：3D t-SNE/UMAP、3D分布图，并保存为PDF格式
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import seaborn as sns
from sklearn.manifold import TSNE
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
import pickle

# 设置matplotlib后端（支持PDF）
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
    'pdf.fonttype': 42,  # TrueType字体，确保PDF中文字可编辑
    'ps.fonttype': 42,
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


def load_saved_features_or_extract(config, device, force_reload=False):
    """加载保存的特征或重新提取"""
    features_file = Path(config.log_dir) / 'visualization_features.pkl'
    
    if features_file.exists() and not force_reload:
        print(f"📥 加载已保存的特征: {features_file}")
        with open(features_file, 'rb') as f:
            features_dict = pickle.load(f)
        print(f"✅ 特征加载完成")
        return features_dict
    
    # 需要重新提取特征
    print("📊 重新提取特征...")
    
    # 加载模型
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
        'knowledge_embeds': []
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取特征"):
            oct_images = batch['oct_images'].to(device)
            colposcopy_images = batch['colposcopy_images'].to(device)
            labels = batch['label'].to(device)
            center_labels = batch['center_idx'].to(device)
            knowledge_embeddings = batch['knowledge_embedding'].to(device)
            
            # 提取patch特征（处理多帧/多图像输入）
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
            
            probs = torch.softmax(outputs['pred'], dim=1).cpu().numpy()
            features_dict['predictions'].append(torch.argmax(outputs['pred'], dim=1).cpu().numpy())
            features_dict['probabilities'].append(probs)
            features_dict['labels'].append(labels.cpu().numpy())
            features_dict['centers'].append(center_labels.cpu().numpy())
            features_dict['knowledge_embeds'].append(knowledge_embeddings.cpu().numpy())
    
    # 合并所有batch
    for key in features_dict:
        if features_dict[key]:
            features_dict[key] = np.vstack(features_dict[key]) if len(features_dict[key][0].shape) > 1 else np.hstack(features_dict[key])
    
    # 保存特征
    print(f"💾 保存特征到: {features_file}")
    with open(features_file, 'wb') as f:
        pickle.dump(features_dict, f)
    
    return features_dict


def visualize_3d_tsne_umap(features_dict, output_dir, timestamp):
    """生成3D t-SNE和UMAP可视化"""
    print("📊 生成3D t-SNE和UMAP可视化...")
    
    z_causal = features_dict['z_causal']
    labels = features_dict['labels']
    centers = features_dict['centers']
    probs = features_dict['probabilities'][:, 1] if 'probabilities' in features_dict else None
    
    fig = plt.figure(figsize=(24, 12))
    
    # t-SNE 3D
    print("  计算3D t-SNE...")
    try:
        tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=1000)
    except TypeError:
        tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
    z_tsne_3d = tsne_3d.fit_transform(z_causal)
    
    # 1. 3D t-SNE by Label
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    neg_mask = labels == 0
    pos_mask = labels == 1
    ax1.scatter(z_tsne_3d[neg_mask, 0], z_tsne_3d[neg_mask, 1], z_tsne_3d[neg_mask, 2], 
               c='red', label='Negative', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax1.scatter(z_tsne_3d[pos_mask, 0], z_tsne_3d[pos_mask, 1], z_tsne_3d[pos_mask, 2], 
               c='green', label='Positive', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('t-SNE Dim 1', fontsize=11, fontweight='bold')
    ax1.set_ylabel('t-SNE Dim 2', fontsize=11, fontweight='bold')
    ax1.set_zlabel('t-SNE Dim 3', fontsize=11, fontweight='bold')
    ax1.set_title('(a) 3D t-SNE by Label', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(fontsize=10)
    
    # 2. 3D t-SNE by Center
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    unique_centers = np.unique(centers)
    colors_center = plt.cm.tab10(np.linspace(0, 1, len(unique_centers)))
    for i, center_id in enumerate(unique_centers):
        mask = centers == center_id
        ax2.scatter(z_tsne_3d[mask, 0], z_tsne_3d[mask, 1], z_tsne_3d[mask, 2], 
                   c=[colors_center[i]], label=f'Center {center_id}', alpha=0.6, s=50, 
                   edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('t-SNE Dim 1', fontsize=11, fontweight='bold')
    ax2.set_ylabel('t-SNE Dim 2', fontsize=11, fontweight='bold')
    ax2.set_zlabel('t-SNE Dim 3', fontsize=11, fontweight='bold')
    ax2.set_title('(b) 3D t-SNE by Center', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(fontsize=9)
    
    # 3. 3D t-SNE by Prediction Probability
    if probs is not None:
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        scatter3 = ax3.scatter(z_tsne_3d[:, 0], z_tsne_3d[:, 1], z_tsne_3d[:, 2], 
                              c=probs, cmap='RdYlGn', alpha=0.6, s=50, 
                              edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('t-SNE Dim 1', fontsize=11, fontweight='bold')
        ax3.set_ylabel('t-SNE Dim 2', fontsize=11, fontweight='bold')
        ax3.set_zlabel('t-SNE Dim 3', fontsize=11, fontweight='bold')
        ax3.set_title('(c) 3D t-SNE by Prediction Probability', fontsize=12, fontweight='bold', pad=10)
        plt.colorbar(scatter3, ax=ax3, label='Positive Probability', shrink=0.8)
    
    # UMAP 3D (if available)
    if HAS_UMAP:
        print("  计算3D UMAP...")
        umap_3d = UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
        z_umap_3d = umap_3d.fit_transform(z_causal)
        
        # 4. 3D UMAP by Label
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        ax4.scatter(z_umap_3d[neg_mask, 0], z_umap_3d[neg_mask, 1], z_umap_3d[neg_mask, 2], 
                   c='red', label='Negative', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax4.scatter(z_umap_3d[pos_mask, 0], z_umap_3d[pos_mask, 1], z_umap_3d[pos_mask, 2], 
                   c='green', label='Positive', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('UMAP Dim 1', fontsize=11, fontweight='bold')
        ax4.set_ylabel('UMAP Dim 2', fontsize=11, fontweight='bold')
        ax4.set_zlabel('UMAP Dim 3', fontsize=11, fontweight='bold')
        ax4.set_title('(d) 3D UMAP by Label', fontsize=12, fontweight='bold', pad=10)
        ax4.legend(fontsize=10)
        
        # 5. 3D UMAP by Center
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        for i, center_id in enumerate(unique_centers):
            mask = centers == center_id
            ax5.scatter(z_umap_3d[mask, 0], z_umap_3d[mask, 1], z_umap_3d[mask, 2], 
                       c=[colors_center[i]], label=f'Center {center_id}', alpha=0.6, s=50, 
                       edgecolors='black', linewidth=0.5)
        ax5.set_xlabel('UMAP Dim 1', fontsize=11, fontweight='bold')
        ax5.set_ylabel('UMAP Dim 2', fontsize=11, fontweight='bold')
        ax5.set_zlabel('UMAP Dim 3', fontsize=11, fontweight='bold')
        ax5.set_title('(e) 3D UMAP by Center', fontsize=12, fontweight='bold', pad=10)
        ax5.legend(fontsize=9)
        
        # 6. 3D UMAP by Prediction Probability
        if probs is not None:
            ax6 = fig.add_subplot(2, 3, 6, projection='3d')
            scatter6 = ax6.scatter(z_umap_3d[:, 0], z_umap_3d[:, 1], z_umap_3d[:, 2], 
                                  c=probs, cmap='RdYlGn', alpha=0.6, s=50, 
                                  edgecolors='black', linewidth=0.5)
            ax6.set_xlabel('UMAP Dim 1', fontsize=11, fontweight='bold')
            ax6.set_ylabel('UMAP Dim 2', fontsize=11, fontweight='bold')
            ax6.set_zlabel('UMAP Dim 3', fontsize=11, fontweight='bold')
            ax6.set_title('(f) 3D UMAP by Prediction Probability', fontsize=12, fontweight='bold', pad=10)
            plt.colorbar(scatter6, ax=ax6, label='Positive Probability', shrink=0.8)
    else:
        # 如果没有UMAP，显示PCA 3D
        print("  计算3D PCA...")
        pca_3d = PCA(n_components=3)
        z_pca_3d = pca_3d.fit_transform(z_causal)
        
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        ax4.scatter(z_pca_3d[neg_mask, 0], z_pca_3d[neg_mask, 1], z_pca_3d[neg_mask, 2], 
                   c='red', label='Negative', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax4.scatter(z_pca_3d[pos_mask, 0], z_pca_3d[pos_mask, 1], z_pca_3d[pos_mask, 2], 
                   c='green', label='Positive', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('PC1', fontsize=11, fontweight='bold')
        ax4.set_ylabel('PC2', fontsize=11, fontweight='bold')
        ax4.set_zlabel('PC3', fontsize=11, fontweight='bold')
        ax4.set_title('(d) 3D PCA by Label', fontsize=12, fontweight='bold', pad=10)
        ax4.legend(fontsize=10)
    
    plt.suptitle('3D Feature Space Visualization: t-SNE and UMAP', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存为PDF
    output_path_pdf = output_dir / f"tsne_umap_3d_{timestamp}.pdf"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    
    # 同时保存PNG
    output_path_png = output_dir / f"tsne_umap_3d_{timestamp}.png"
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 3D t-SNE/UMAP可视化已保存: {output_path_pdf}")
    return output_path_pdf


def visualize_3d_distribution(features_dict, output_dir, timestamp):
    """生成3D分布可视化（替代小提琴图）"""
    print("📊 生成3D分布可视化...")
    
    probs = features_dict['probabilities'][:, 1] if 'probabilities' in features_dict else None
    labels = features_dict['labels']
    centers = features_dict['centers']
    z_causal = features_dict['z_causal']
    
    if probs is None:
        print("⚠️ 未找到预测概率，跳过3D分布可视化")
        return None
    
    # 使用PCA降维到3D
    pca_3d = PCA(n_components=3)
    z_pca_3d = pca_3d.fit_transform(z_causal)
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 3D分布：按标签着色
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    neg_mask = labels == 0
    pos_mask = labels == 1
    ax1.scatter(z_pca_3d[neg_mask, 0], z_pca_3d[neg_mask, 1], probs[neg_mask], 
               c='red', label='Negative', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax1.scatter(z_pca_3d[pos_mask, 0], z_pca_3d[pos_mask, 1], probs[pos_mask], 
               c='green', label='Positive', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('PC1', fontsize=11, fontweight='bold')
    ax1.set_ylabel('PC2', fontsize=11, fontweight='bold')
    ax1.set_zlabel('Prediction Probability', fontsize=11, fontweight='bold')
    ax1.set_title('(a) 3D Distribution: Features vs Probability by Label', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(fontsize=10)
    
    # 2. 3D分布：按中心着色
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    unique_centers = np.unique(centers)
    colors_center = plt.cm.tab10(np.linspace(0, 1, len(unique_centers)))
    for i, center_id in enumerate(unique_centers):
        mask = centers == center_id
        ax2.scatter(z_pca_3d[mask, 0], z_pca_3d[mask, 1], probs[mask], 
                   c=[colors_center[i]], label=f'Center {center_id}', alpha=0.6, s=50, 
                   edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('PC1', fontsize=11, fontweight='bold')
    ax2.set_ylabel('PC2', fontsize=11, fontweight='bold')
    ax2.set_zlabel('Prediction Probability', fontsize=11, fontweight='bold')
    ax2.set_title('(b) 3D Distribution: Features vs Probability by Center', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(fontsize=9)
    
    # 3. 3D柱状图：按标签和中心分组的平均概率
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    df = pd.DataFrame({
        'Center': centers,
        'Label': labels,
        'Probability': probs
    })
    
    # 计算每个组合的平均概率
    grouped = df.groupby(['Center', 'Label'])['Probability'].mean().reset_index()
    
    x_pos = grouped['Center'].values
    y_pos = grouped['Label'].values
    z_pos = np.zeros(len(grouped))
    dx = 0.3
    dy = 0.3
    dz = grouped['Probability'].values
    
    colors_bar = ['red' if l == 0 else 'green' for l in y_pos]
    ax3.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Center ID', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Label (0: Neg, 1: Pos)', fontsize=11, fontweight='bold')
    ax3.set_zlabel('Mean Probability', fontsize=11, fontweight='bold')
    ax3.set_title('(c) 3D Bar Chart: Mean Probability by Center and Label', fontsize=12, fontweight='bold', pad=10)
    
    # 4. 3D散点图：特征空间 + 预测概率（颜色映射）
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    scatter4 = ax4.scatter(z_pca_3d[:, 0], z_pca_3d[:, 1], z_pca_3d[:, 2], 
                          c=probs, cmap='RdYlGn', alpha=0.6, s=50, 
                          edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('PC1', fontsize=11, fontweight='bold')
    ax4.set_ylabel('PC2', fontsize=11, fontweight='bold')
    ax4.set_zlabel('PC3', fontsize=11, fontweight='bold')
    ax4.set_title('(d) 3D Feature Space Colored by Probability', fontsize=12, fontweight='bold', pad=10)
    plt.colorbar(scatter4, ax=ax4, label='Positive Probability', shrink=0.8)
    
    # 5. 3D表面图：预测概率分布（使用插值）
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    # 创建网格
    x_grid = np.linspace(z_pca_3d[:, 0].min(), z_pca_3d[:, 0].max(), 20)
    y_grid = np.linspace(z_pca_3d[:, 1].min(), z_pca_3d[:, 1].max(), 20)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # 使用插值创建Z值
    from scipy.interpolate import griddata
    Z_grid = griddata((z_pca_3d[:, 0], z_pca_3d[:, 1]), probs, (X_grid, Y_grid), method='cubic')
    
    # 绘制表面
    surf = ax5.plot_surface(X_grid, Y_grid, Z_grid, cmap='RdYlGn', alpha=0.7, 
                           linewidth=0, antialiased=True)
    ax5.scatter(z_pca_3d[:, 0], z_pca_3d[:, 1], probs, c=probs, cmap='RdYlGn', 
               s=30, edgecolors='black', linewidth=0.5, alpha=0.8)
    ax5.set_xlabel('PC1', fontsize=11, fontweight='bold')
    ax5.set_ylabel('PC2', fontsize=11, fontweight='bold')
    ax5.set_zlabel('Prediction Probability', fontsize=11, fontweight='bold')
    ax5.set_title('(e) 3D Surface: Probability Distribution', fontsize=12, fontweight='bold', pad=10)
    plt.colorbar(surf, ax=ax5, label='Positive Probability', shrink=0.8)
    
    # 6. 3D密度图：按标签分组的概率分布
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    
    # 为每个标签创建直方图
    neg_probs = probs[neg_mask]
    pos_probs = probs[pos_mask]
    
    # 创建直方图数据
    bins = np.linspace(0, 1, 21)
    neg_hist, _ = np.histogram(neg_probs, bins=bins)
    pos_hist, _ = np.histogram(pos_probs, bins=bins)
    
    x_pos_neg = np.array([0] * len(neg_hist))
    x_pos_pos = np.array([1] * len(pos_hist))
    y_pos = (bins[:-1] + bins[1:]) / 2
    
    # 绘制柱状图
    ax6.bar3d(x_pos_neg, y_pos, np.zeros(len(neg_hist)), 0.3, (bins[1] - bins[0]) * 0.9, neg_hist, 
             color='red', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax6.bar3d(x_pos_pos, y_pos, np.zeros(len(pos_hist)), 0.3, (bins[1] - bins[0]) * 0.9, pos_hist, 
             color='green', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax6.set_xlabel('Label (0: Neg, 1: Pos)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Probability', fontsize=11, fontweight='bold')
    ax6.set_zlabel('Frequency', fontsize=11, fontweight='bold')
    ax6.set_title('(f) 3D Histogram: Probability Distribution by Label', fontsize=12, fontweight='bold', pad=10)
    ax6.set_xticks([0, 1])
    ax6.set_xticklabels(['Negative', 'Positive'])
    
    plt.suptitle('3D Distribution Visualization: Prediction Probability Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存为PDF
    output_path_pdf = output_dir / f"distribution_3d_{timestamp}.pdf"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    
    # 同时保存PNG
    output_path_png = output_dir / f"distribution_3d_{timestamp}.png"
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 3D分布可视化已保存: {output_path_pdf}")
    return output_path_pdf


def save_visualization_data_and_code(features_dict, output_dir, timestamp):
    """保存可视化数据和代码"""
    print("💾 保存可视化数据和代码...")
    
    # 1. 保存特征数据
    data_file = output_dir / f"visualization_data_{timestamp}.pkl"
    with open(data_file, 'wb') as f:
        pickle.dump(features_dict, f)
    print(f"✅ 特征数据已保存: {data_file}")
    
    # 2. 保存为CSV（便于查看）
    csv_file = output_dir / f"visualization_data_{timestamp}.csv"
    df_data = pd.DataFrame({
        'label': features_dict['labels'],
        'center': features_dict['centers'],
        'prediction': features_dict['predictions'],
        'probability_negative': features_dict['probabilities'][:, 0] if 'probabilities' in features_dict else None,
        'probability_positive': features_dict['probabilities'][:, 1] if 'probabilities' in features_dict else None,
    })
    df_data.to_csv(csv_file, index=False)
    print(f"✅ CSV数据已保存: {csv_file}")
    
    # 3. 保存代码
    code_file = output_dir / f"visualization_code_{timestamp}.py"
    with open(__file__, 'r', encoding='utf-8') as f:
        code_content = f.read()
    
    with open(code_file, 'w', encoding='utf-8') as f:
        f.write(code_content)
    print(f"✅ 可视化代码已保存: {code_file}")
    
    # 4. 保存配置信息
    config_file = output_dir / f"visualization_config_{timestamp}.json"
    config_info = {
        'timestamp': timestamp,
        'num_samples': len(features_dict['labels']),
        'label_distribution': {
            'negative': int(np.sum(features_dict['labels'] == 0)),
            'positive': int(np.sum(features_dict['labels'] == 1))
        },
        'center_distribution': {int(k): int(v) for k, v in zip(*np.unique(features_dict['centers'], return_counts=True))},
        'feature_shapes': {k: list(v.shape) if isinstance(v, np.ndarray) else str(type(v)) 
                          for k, v in features_dict.items() if isinstance(v, np.ndarray)}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    print(f"✅ 配置信息已保存: {config_file}")
    
    return data_file, csv_file, code_file, config_file


def main():
    """主函数"""
    print("=" * 80)
    print("Bio-COT 3.0 3D可视化生成（PDF格式）")
    print("=" * 80)
    
    # 配置
    config = BioCOT_v3_Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载或提取特征
    features_dict = load_saved_features_or_extract(config, device, force_reload=False)
    
    # 保存数据和代码
    save_visualization_data_and_code(features_dict, output_dir, timestamp)
    
    # 生成3D可视化
    print("\n" + "=" * 80)
    print("生成3D可视化图表...")
    print("=" * 80)
    
    # 1. 3D t-SNE/UMAP
    visualize_3d_tsne_umap(features_dict, output_dir, timestamp)
    
    # 2. 3D分布可视化
    visualize_3d_distribution(features_dict, output_dir, timestamp)
    
    print("\n" + "=" * 80)
    print("✅ 所有3D可视化已完成！")
    print("=" * 80)
    print(f"输出目录: {output_dir}")
    print(f"时间戳: {timestamp}")
    print("\n生成的文件:")
    print(f"  - tsne_umap_3d_{timestamp}.pdf (3D t-SNE/UMAP)")
    print(f"  - distribution_3d_{timestamp}.pdf (3D分布可视化)")
    print(f"  - visualization_data_{timestamp}.pkl (特征数据)")
    print(f"  - visualization_data_{timestamp}.csv (CSV数据)")
    print(f"  - visualization_code_{timestamp}.py (可视化代码)")
    print(f"  - visualization_config_{timestamp}.json (配置信息)")


if __name__ == '__main__':
    main()

