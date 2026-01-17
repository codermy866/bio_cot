#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.0 SCI顶刊级可视化生成脚本
包括：Knowledge Notes、Visual Notes、小提琴图、火山图、t-SNE、UMAP、CAM图
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.manifold import TSNE
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️ UMAP未安装，将跳过UMAP可视化。安装: pip install umap-learn")
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
from PIL import Image
import cv2
from scipy import stats
from scipy.stats import ttest_ind

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
from models.bio_cot_v3 import BioCOT_v3
from data.dataset_v3 import FiveCentersMultimodalDatasetV3
from training.extract_vit_patches import extract_patch_features_with_vit
from torch.utils.data import DataLoader
from torchvision import transforms


class BioCOT3_GradCAM:
    """Bio-COT 3.0的GradCAM可视化工具"""
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.handlers = []
        self._register_hooks()
    
    def _register_hooks(self):
        """注册hook"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
        
        # 找到分类器的第一个线性层
        if hasattr(self.model, 'classifier'):
            target_layer = list(self.model.classifier.children())[0]
            if isinstance(target_layer, nn.Linear):
                self.handlers.append(target_layer.register_forward_hook(forward_hook))
                self.handlers.append(target_layer.register_full_backward_hook(backward_hook))
    
    def remove_hooks(self):
        for handle in self.handlers:
            handle.remove()
    
    def generate_cam(self, oct_feats, colpo_feats, knowledge_embeds, center_labels, target_class=None):
        """生成CAM图"""
        self.model.zero_grad()
        self.gradients = None
        self.activations = None
        
        # 确保输入需要梯度
        if not oct_feats.requires_grad:
            oct_feats = oct_feats.clone().detach().requires_grad_(True)
        if not colpo_feats.requires_grad:
            colpo_feats = colpo_feats.clone().detach().requires_grad_(True)
        if not knowledge_embeds.requires_grad:
            knowledge_embeds = knowledge_embeds.clone().detach().requires_grad_(True)
        
        # 前向传播（不使用no_grad）
        outputs = self.model(
            f_oct=oct_feats,
            f_colpo=colpo_feats,
            note_embeds=knowledge_embeds,
            center_labels=center_labels,
            return_loss_components=False
        )
        
        logits = outputs['pred']
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
        
        # 反向传播
        one_hot = torch.zeros_like(logits)
        one_hot[0][target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            return None
        
        # 计算CAM
        grads = self.gradients
        fmaps = self.activations
        
        # Global Average Pooling
        weights = torch.mean(grads, dim=1, keepdim=True)
        cam = torch.sum(weights * fmaps, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # 归一化
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        return cam.cpu().numpy()


def extract_all_features(model, dataloader, device, config):
    """提取所有特征用于可视化"""
    model.eval()
    
    features_dict = {
        'z_causal': [],
        'z_noise': [],
        'z_sem': [],
        'fused': [],
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
            # OCT: [B, F, C, H, W] -> 需要flatten到 [B*F, C, H, W]
            B, F_oct = oct_images.shape[0], oct_images.shape[1]
            oct_images_flat = oct_images.view(B * F_oct, *oct_images.shape[2:])
            oct_feats_flat = extract_patch_features_with_vit(oct_images_flat, device)  # [B*F, N, D]
            # Reshape back: [B*F, N, D] -> [B, F, N, D] -> average over frames: [B, N, D]
            oct_feats = oct_feats_flat.view(B, F_oct, *oct_feats_flat.shape[1:]).mean(dim=1)  # [B, N, D]
            
            # Colposcopy: [B, N_colpo, C, H, W] -> 需要flatten到 [B*N_colpo, C, H, W]
            B, N_colpo = colposcopy_images.shape[0], colposcopy_images.shape[1]
            colpo_images_flat = colposcopy_images.view(B * N_colpo, *colposcopy_images.shape[2:])
            colpo_feats_flat = extract_patch_features_with_vit(colpo_images_flat, device)  # [B*N_colpo, N, D]
            # Reshape back: [B*N_colpo, N, D] -> [B, N_colpo, N, D] -> average over images: [B, N, D]
            colpo_feats = colpo_feats_flat.view(B, N_colpo, *colpo_feats_flat.shape[1:]).mean(dim=1)  # [B, N, D]
            
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
    
    return features_dict


def visualize_knowledge_notes_distribution(features_dict, output_dir, timestamp):
    """可视化Knowledge Notes分布"""
    print("📊 生成Knowledge Notes分布图...")
    
    z_sem = features_dict['z_sem']
    labels = features_dict['labels']
    
    # 使用PCA降维到2D用于可视化
    pca = PCA(n_components=2)
    z_sem_2d = pca.fit_transform(z_sem)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # 1. 按标签着色
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(z_sem_2d[:, 0], z_sem_2d[:, 1], c=labels, cmap='RdYlGn', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('PC1 (Explained Variance: {:.2f}%)'.format(pca.explained_variance_ratio_[0]*100), fontsize=12, fontweight='bold')
    ax1.set_ylabel('PC2 (Explained Variance: {:.2f}%)'.format(pca.explained_variance_ratio_[1]*100), fontsize=12, fontweight='bold')
    ax1.set_title('(a) Knowledge Notes Distribution by Label', fontsize=13, fontweight='bold')
    plt.colorbar(scatter1, ax=ax1, label='Label (0: Negative, 1: Positive)')
    ax1.grid(True, alpha=0.3)
    
    # 2. 小提琴图：按标签分组
    ax2 = axes[0, 1]
    df = pd.DataFrame({
        'Knowledge Note Embedding (PC1)': z_sem_2d[:, 0],
        'Label': ['Positive' if l == 1 else 'Negative' for l in labels]
    })
    sns.violinplot(data=df, x='Label', y='Knowledge Note Embedding (PC1)', ax=ax2, inner='box', palette='Set2')
    ax2.set_title('(b) Knowledge Notes Distribution (Violin Plot)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 箱线图：按标签分组
    ax3 = axes[1, 0]
    sns.boxplot(data=df, x='Label', y='Knowledge Note Embedding (PC1)', ax=ax3, palette='Set2')
    ax3.set_title('(c) Knowledge Notes Distribution (Box Plot)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 统计显著性检验
    ax4 = axes[1, 1]
    neg_values = z_sem_2d[labels == 0, 0]
    pos_values = z_sem_2d[labels == 1, 0]
    t_stat, p_value = ttest_ind(neg_values, pos_values)
    
    ax4.bar(['Negative', 'Positive'], [neg_values.mean(), pos_values.mean()], 
            yerr=[neg_values.std(), pos_values.std()], capsize=10, 
            color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Mean PC1 Value', fontsize=12, fontweight='bold')
    ax4.set_title(f'(d) Statistical Comparison\n(t-test: p={p_value:.4f})', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / f"knowledge_notes_distribution_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Knowledge Notes分布图已保存: {output_path}")
    return output_path


def visualize_visual_notes_attention(features_dict, output_dir, timestamp):
    """可视化Visual Notes注意力分布"""
    print("📊 生成Visual Notes注意力分布图...")
    
    if 'attn_oct' not in features_dict or len(features_dict['attn_oct']) == 0:
        print("⚠️ 未找到注意力数据，跳过Visual Notes可视化")
        return None
    
    attn_oct = features_dict['attn_oct']  # [B, N, 1]
    attn_colpo = features_dict['attn_colpo']  # [B, N, 1]
    labels = features_dict['labels']
    
    # 计算每个样本的平均注意力
    attn_oct_mean = np.mean(attn_oct.reshape(attn_oct.shape[0], -1), axis=1)
    attn_colpo_mean = np.mean(attn_colpo.reshape(attn_colpo.shape[0], -1), axis=1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. OCT注意力小提琴图
    ax1 = axes[0, 0]
    df_oct = pd.DataFrame({
        'Attention Value': attn_oct_mean,
        'Label': ['Positive' if l == 1 else 'Negative' for l in labels]
    })
    sns.violinplot(data=df_oct, x='Label', y='Attention Value', ax=ax1, inner='box', palette='Blues')
    ax1.set_title('(a) OCT Attention Distribution', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Colposcopy注意力小提琴图
    ax2 = axes[0, 1]
    df_colpo = pd.DataFrame({
        'Attention Value': attn_colpo_mean,
        'Label': ['Positive' if l == 1 else 'Negative' for l in labels]
    })
    sns.violinplot(data=df_colpo, x='Label', y='Attention Value', ax=ax2, inner='box', palette='Reds')
    ax2.set_title('(b) Colposcopy Attention Distribution', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 注意力对比箱线图
    ax3 = axes[0, 2]
    df_combined = pd.DataFrame({
        'Attention': np.concatenate([attn_oct_mean, attn_colpo_mean]),
        'Modality': ['OCT'] * len(attn_oct_mean) + ['Colposcopy'] * len(attn_colpo_mean),
        'Label': (['Positive' if l == 1 else 'Negative' for l in labels] * 2)
    })
    sns.boxplot(data=df_combined, x='Modality', y='Attention', hue='Label', ax=ax3, palette='Set2')
    ax3.set_title('(c) Attention Comparison by Modality', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(title='Label')
    
    # 4. 注意力热图（平均注意力值）
    ax4 = axes[1, 0]
    # 选择几个代表性样本
    n_samples = min(10, len(attn_oct))
    sample_indices = np.linspace(0, len(attn_oct)-1, n_samples, dtype=int)
    attn_samples = attn_oct[sample_indices].reshape(n_samples, 14, 14)  # 假设14x14的patch grid
    
    im = ax4.imshow(np.mean(attn_samples, axis=0), cmap='hot', interpolation='bilinear')
    ax4.set_title('(d) Average Attention Heatmap (OCT)', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Patch X', fontsize=11)
    ax4.set_ylabel('Patch Y', fontsize=11)
    plt.colorbar(im, ax=ax4, label='Attention Value')
    
    # 5. 注意力热图（Colposcopy）
    ax5 = axes[1, 1]
    attn_colpo_samples = attn_colpo[sample_indices].reshape(n_samples, 14, 14)
    im2 = ax5.imshow(np.mean(attn_colpo_samples, axis=0), cmap='hot', interpolation='bilinear')
    ax5.set_title('(e) Average Attention Heatmap (Colposcopy)', fontsize=13, fontweight='bold')
    ax5.set_xlabel('Patch X', fontsize=11)
    ax5.set_ylabel('Patch Y', fontsize=11)
    plt.colorbar(im2, ax=ax5, label='Attention Value')
    
    # 6. 注意力统计
    ax6 = axes[1, 2]
    stats_data = {
        'OCT': [attn_oct_mean.mean(), attn_oct_mean.std()],
        'Colposcopy': [attn_colpo_mean.mean(), attn_colpo_mean.std()]
    }
    x_pos = np.arange(len(stats_data))
    means = [stats_data[k][0] for k in stats_data.keys()]
    stds = [stats_data[k][1] for k in stats_data.keys()]
    ax6.bar(x_pos, means, yerr=stds, capsize=10, color=['#4A90E2', '#E24A4A'], 
            alpha=0.7, edgecolor='black', linewidth=1.5)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(list(stats_data.keys()))
    ax6.set_ylabel('Mean Attention Value', fontsize=12, fontweight='bold')
    ax6.set_title('(f) Attention Statistics', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / f"visual_notes_attention_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visual Notes注意力分布图已保存: {output_path}")
    return output_path


def visualize_volcano_plot(features_dict, output_dir, timestamp):
    """生成火山图（Volcano Plot）"""
    print("📊 生成火山图...")
    
    z_causal = features_dict['z_causal']
    labels = features_dict['labels']
    
    # 计算每个特征的fold change和p值
    neg_mask = labels == 0
    pos_mask = labels == 1
    
    neg_mean = np.mean(z_causal[neg_mask], axis=0)
    pos_mean = np.mean(z_causal[pos_mask], axis=0)
    
    # Fold change (log2)
    fold_change = np.log2((pos_mean + 1e-8) / (neg_mean + 1e-8))
    
    # t-test for each feature
    p_values = []
    for i in range(z_causal.shape[1]):
        _, p_val = ttest_ind(z_causal[pos_mask, i], z_causal[neg_mask, i])
        p_values.append(p_val)
    p_values = np.array(p_values)
    neg_log10_p = -np.log10(p_values + 1e-10)
    
    # 标记显著特征
    significant = (p_values < 0.05) & (np.abs(fold_change) > 0.5)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制散点
    ax.scatter(fold_change[~significant], neg_log10_p[~significant], 
               c='gray', alpha=0.5, s=50, label='Non-significant')
    ax.scatter(fold_change[significant], neg_log10_p[significant], 
               c='red', alpha=0.7, s=80, label='Significant (p<0.05, |FC|>0.5)')
    
    # 添加阈值线
    ax.axhline(y=-np.log10(0.05), color='blue', linestyle='--', linewidth=2, alpha=0.7, label='p=0.05')
    ax.axvline(x=0.5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='FC=0.5')
    ax.axvline(x=-0.5, color='green', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Log2 Fold Change (Positive vs Negative)', fontsize=13, fontweight='bold')
    ax.set_ylabel('-Log10(p-value)', fontsize=13, fontweight='bold')
    ax.set_title('Volcano Plot: Causal Features Differential Analysis', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"volcano_plot_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 火山图已保存: {output_path}")
    return output_path


def visualize_tsne_umap(features_dict, output_dir, timestamp):
    """生成t-SNE和UMAP可视化"""
    print("📊 生成t-SNE和UMAP可视化...")
    
    z_causal = features_dict['z_causal']
    labels = features_dict['labels']
    centers = features_dict['centers']
    probs = features_dict['probabilities'][:, 1] if 'probabilities' in features_dict else None
    
    fig = plt.figure(figsize=(20, 10))
    
    # t-SNE
    print("  计算t-SNE...")
    # 新版本scikit-learn使用max_iter而不是n_iter
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    except TypeError:
        # 旧版本兼容
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    z_tsne = tsne.fit_transform(z_causal)
    
    # 1. t-SNE by Label
    ax1 = plt.subplot(2, 3, 1)
    scatter1 = ax1.scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels, cmap='RdYlGn', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax1.set_title('(a) t-SNE by Label', fontsize=13, fontweight='bold')
    plt.colorbar(scatter1, ax=ax1, label='Label')
    ax1.grid(True, alpha=0.3)
    
    # 2. t-SNE by Center
    ax2 = plt.subplot(2, 3, 2)
    scatter2 = ax2.scatter(z_tsne[:, 0], z_tsne[:, 1], c=centers, cmap='tab10', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax2.set_title('(b) t-SNE by Center', fontsize=13, fontweight='bold')
    plt.colorbar(scatter2, ax=ax2, label='Center ID')
    ax2.grid(True, alpha=0.3)
    
    # 3. t-SNE by Prediction Probability
    if probs is not None:
        ax3 = plt.subplot(2, 3, 3)
        scatter3 = ax3.scatter(z_tsne[:, 0], z_tsne[:, 1], c=probs, cmap='RdYlGn', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
        ax3.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
        ax3.set_title('(c) t-SNE by Prediction Probability', fontsize=13, fontweight='bold')
        plt.colorbar(scatter3, ax=ax3, label='Positive Probability')
        ax3.grid(True, alpha=0.3)
    
    # UMAP (if available)
    if HAS_UMAP:
        print("  计算UMAP...")
        umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        z_umap = umap_model.fit_transform(z_causal)
        
        # 4. UMAP by Label
        ax4 = plt.subplot(2, 3, 4)
        scatter4 = ax4.scatter(z_umap[:, 0], z_umap[:, 1], c=labels, cmap='RdYlGn', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
        ax4.set_ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')
        ax4.set_title('(d) UMAP by Label', fontsize=13, fontweight='bold')
        plt.colorbar(scatter4, ax=ax4, label='Label')
        ax4.grid(True, alpha=0.3)
        
        # 5. UMAP by Center
        ax5 = plt.subplot(2, 3, 5)
        scatter5 = ax5.scatter(z_umap[:, 0], z_umap[:, 1], c=centers, cmap='tab10', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax5.set_xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
        ax5.set_ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')
        ax5.set_title('(e) UMAP by Center', fontsize=13, fontweight='bold')
        plt.colorbar(scatter5, ax=ax5, label='Center ID')
        ax5.grid(True, alpha=0.3)
        
        # 6. UMAP by Prediction Probability
        if probs is not None:
            ax6 = plt.subplot(2, 3, 6)
            scatter6 = ax6.scatter(z_umap[:, 0], z_umap[:, 1], c=probs, cmap='RdYlGn', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            ax6.set_xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
            ax6.set_ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')
            ax6.set_title('(f) UMAP by Prediction Probability', fontsize=13, fontweight='bold')
            plt.colorbar(scatter6, ax=ax6, label='Positive Probability')
            ax6.grid(True, alpha=0.3)
    else:
        # 如果没有UMAP，显示其他可视化
        ax4 = plt.subplot(2, 3, 4)
        ax4.text(0.5, 0.5, 'UMAP not available', ha='center', va='center', fontsize=14)
        ax4.axis('off')
    
    plt.tight_layout()
    output_path = output_dir / f"tsne_umap_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ t-SNE/UMAP可视化已保存: {output_path}")
    return output_path


def visualize_cam_samples(model, dataloader, device, config, output_dir, timestamp, num_samples=12):
    """生成CAM图样本"""
    print("📊 生成CAM图...")
    
    gradcam = BioCOT3_GradCAM(model)
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    count = 0
    for batch in dataloader:
        if count >= num_samples:
            break
        
        oct_images = batch['oct_images'].to(device)
        colposcopy_images = batch['colposcopy_images'].to(device)
        labels = batch['label'].to(device)
        center_labels = batch['center_idx'].to(device)
        knowledge_embeddings = batch['knowledge_embedding'].to(device)
        
        # 只处理第一个样本
        oct_img = oct_images[0:1]
        colpo_img = colposcopy_images[0:1]
        label = labels[0].item()
        center = center_labels[0].item()
        knowledge = knowledge_embeddings[0:1]
        center_label = center_labels[0:1]
        
        # 提取patch特征（处理多帧/多图像输入）
        # 注意：CAM需要梯度，所以不使用no_grad
        # OCT: [1, F, C, H, W] -> flatten到 [F, C, H, W]
        F_oct = oct_img.shape[1]
        oct_img_flat = oct_img.view(F_oct, *oct_img.shape[2:])
        with torch.no_grad():
            oct_feats_flat = extract_patch_features_with_vit(oct_img_flat, device)  # [F, N, D]
        oct_feats = oct_feats_flat.mean(dim=0, keepdim=True).requires_grad_(True)  # [1, N, D]
        
        # Colposcopy: [1, N_colpo, C, H, W] -> flatten到 [N_colpo, C, H, W]
        N_colpo = colpo_img.shape[1]
        colpo_img_flat = colpo_img.view(N_colpo, *colpo_img.shape[2:])
        with torch.no_grad():
            colpo_feats_flat = extract_patch_features_with_vit(colpo_img_flat, device)  # [N_colpo, N, D]
        colpo_feats = colpo_feats_flat.mean(dim=0, keepdim=True).requires_grad_(True)  # [1, N, D]
        
        # 确保knowledge也需要梯度
        knowledge = knowledge.requires_grad_(True)
        
        # 生成CAM
        cam = gradcam.generate_cam(oct_feats, colpo_feats, knowledge, center_label)
        
        if cam is not None:
            # 可视化（使用OCT图像）
            # oct_img是[1, F, C, H, W]，需要取第一帧或平均
            if len(oct_img.shape) == 5:
                img = oct_img[0, 0].cpu().permute(1, 2, 0).numpy()  # 取第一帧
            else:
                img = oct_img[0].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
            
            # 上采样CAM到图像尺寸
            # CAM可能是1D或2D，需要reshape
            if len(cam.shape) == 4:
                cam_2d = cam[0, 0]
            elif len(cam.shape) == 3:
                cam_2d = cam[0]
            else:
                cam_2d = cam
            
            # 如果CAM是1D，reshape到14x14（假设patch grid）
            if len(cam_2d.shape) == 1:
                cam_2d = cam_2d.reshape(14, 14)
            
            cam_resized = cv2.resize(cam_2d, (img.shape[1], img.shape[0]))
            cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min())
            
            # 叠加
            ax = axes[count]
            ax.imshow(img)
            ax.imshow(cam_resized, alpha=0.5, cmap='jet')
            ax.set_title(f'Sample {count+1}\nLabel: {label}, Center: {center}', fontsize=10, fontweight='bold')
            ax.axis('off')
            
            count += 1
    
    gradcam.remove_hooks()
    
    # 隐藏多余的子图
    for i in range(count, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Grad-CAM Visualization: Class Activation Maps', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = output_dir / f"cam_samples_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ CAM图已保存: {output_path}")
    return output_path


def visualize_violin_plots_comprehensive(features_dict, output_dir, timestamp):
    """生成综合小提琴图"""
    print("📊 生成综合小提琴图...")
    
    probs = features_dict['probabilities'][:, 1] if 'probabilities' in features_dict else None
    labels = features_dict['labels']
    centers = features_dict['centers']
    
    if probs is None:
        print("⚠️ 未找到预测概率，跳过小提琴图")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 按标签分组的小提琴图
    ax1 = axes[0, 0]
    df = pd.DataFrame({
        'Prediction Probability': probs,
        'Label': ['Positive' if l == 1 else 'Negative' for l in labels]
    })
    sns.violinplot(data=df, x='Label', y='Prediction Probability', ax=ax1, inner='box', palette='Set2')
    ax1.set_title('(a) Prediction Probability by True Label', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1])
    
    # 2. 按中心分组的小提琴图
    ax2 = axes[0, 1]
    df2 = pd.DataFrame({
        'Prediction Probability': probs,
        'Center': [f'Center {c}' for c in centers]
    })
    sns.violinplot(data=df2, x='Center', y='Prediction Probability', ax=ax2, inner='box', palette='tab10')
    ax2.set_title('(b) Prediction Probability by Center', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1])
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. 按标签和中心组合
    ax3 = axes[1, 0]
    df3 = pd.DataFrame({
        'Prediction Probability': probs,
        'Label': ['Positive' if l == 1 else 'Negative' for l in labels],
        'Center': [f'C{c}' for c in centers]
    })
    sns.violinplot(data=df3, x='Center', y='Prediction Probability', hue='Label', ax=ax3, inner='box', palette='Set2')
    ax3.set_title('(c) Prediction Probability by Center and Label', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 1])
    ax3.legend(title='Label', loc='upper right')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. 预测概率分布直方图
    ax4 = axes[1, 1]
    ax4.hist(probs[labels == 0], bins=30, alpha=0.6, label='Negative', color='red', edgecolor='black')
    ax4.hist(probs[labels == 1], bins=30, alpha=0.6, label='Positive', color='green', edgecolor='black')
    ax4.set_xlabel('Prediction Probability', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Prediction Probability Distribution', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / f"violin_plots_comprehensive_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 综合小提琴图已保存: {output_path}")
    return output_path


def main():
    """主函数"""
    print("=" * 80)
    print("Bio-COT 3.0 SCI顶刊级可视化生成")
    print("=" * 80)
    
    # 配置
    config = BioCOT_v3_Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载最佳模型
    checkpoint_files = sorted(Path(config.checkpoint_dir).glob("best_model_v3_*.pth"),
                             key=lambda x: x.stat().st_mtime, reverse=True)
    if not checkpoint_files:
        print("❌ 未找到模型检查点")
        return
    
    checkpoint_path = checkpoint_files[0]
    print(f"✅ 加载模型: {checkpoint_path.name}")
    
    # 创建模型
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
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # 加载数据集
    print("\n📥 加载数据集...")
    # 构建CSV路径（与训练脚本保持一致）
    data_root = Path(config.data_root)
    val_csv = data_root / 'internal_val' / 'labels.csv'
    
    if not val_csv.exists():
        raise FileNotFoundError(f"未找到验证集CSV文件: {val_csv}")
    
    print(f"   使用CSV文件: {val_csv}")
    
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
    print("\n📊 提取特征用于可视化...")
    features_dict = extract_all_features(model, dataloader, device, config)
    
    # 生成所有可视化
    print("\n" + "=" * 80)
    print("生成可视化图表...")
    print("=" * 80)
    
    # 1. Knowledge Notes可视化
    visualize_knowledge_notes_distribution(features_dict, output_dir, timestamp)
    
    # 2. Visual Notes可视化
    visualize_visual_notes_attention(features_dict, output_dir, timestamp)
    
    # 3. 火山图
    visualize_volcano_plot(features_dict, output_dir, timestamp)
    
    # 4. t-SNE和UMAP
    visualize_tsne_umap(features_dict, output_dir, timestamp)
    
    # 5. 综合小提琴图
    visualize_violin_plots_comprehensive(features_dict, output_dir, timestamp)
    
    # 6. CAM图
    visualize_cam_samples(model, dataloader, device, config, output_dir, timestamp, num_samples=12)
    
    print("\n" + "=" * 80)
    print("✅ 所有可视化已完成！")
    print("=" * 80)
    print(f"输出目录: {output_dir}")
    print(f"时间戳: {timestamp}")
    print("\n生成的文件:")
    print(f"  - knowledge_notes_distribution_{timestamp}.png")
    print(f"  - visual_notes_attention_{timestamp}.png")
    print(f"  - volcano_plot_{timestamp}.png")
    print(f"  - tsne_umap_{timestamp}.png")
    print(f"  - violin_plots_comprehensive_{timestamp}.png")
    print(f"  - cam_samples_{timestamp}.png")


if __name__ == '__main__':
    main()

