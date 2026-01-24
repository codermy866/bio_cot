#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成Nature/Science风格的t-SNE可视化图表
- 2x2网格布局（类似用户提供的图片）
- 按不同条件分组展示
- 符合顶级期刊的配色和布局
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import tqdm

# 设置matplotlib后端
matplotlib.use('Agg')

# Nature/Science风格配置
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'STSong'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.6,
    'lines.linewidth': 2.0,
    'lines.markersize': 4,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Nature风格配色（色盲友好）
COLORS_LABEL = {
    0: '#2E86AB',  # 深蓝色（阴性）
    1: '#A23B72',  # 深紫红色（阳性）
}

COLORS_CENTER = [
    '#2E86AB',  # 深蓝色
    '#A23B72',  # 深紫红色
    '#F18F01',  # 深橙色
    '#C73E1D',  # 深红色
    '#6A994E',  # 深绿色
]

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.bida.bio_cot_model import BioCOTModel
from experiments.exp_5centers.train_bio_cot_5centers_multimodal import extract_features_with_vit
from torch.utils.data import DataLoader
from torchvision import transforms


def extract_features_v1(model, dataloader, device):
    """从BioCOTModel (v1)中提取特征"""
    model.eval()
    
    z_causal_list = []
    z_sem_list = []
    fused_list = []
    labels_list = []
    centers_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取特征"):
            oct_images = batch['oct_images'].to(device)
            colposcopy_images = batch['colposcopy_images'].to(device)
            
            oct_feat = extract_features_with_vit(oct_images, device)
            colpo_feat = extract_features_with_vit(colposcopy_images, device)
            
            # 获取临床数据
            clinical_data = batch.get('clinical_data', None)
            if clinical_data is None:
                # 如果没有clinical_data，尝试从clinical_features构建
                if 'clinical_features' in batch:
                    clinical_features = batch['clinical_features']
                    # 构建clinical_data字典
                    clinical_data = {
                        'age': clinical_features[:, 0].cpu().numpy() * 100,  # 反归一化
                        'hpv': (clinical_features[:, 1] > 0.5).cpu().numpy().astype(int),
                        'tct': np.argmax(clinical_features[:, 2:].cpu().numpy(), axis=1) if clinical_features.shape[1] > 2 else np.zeros(len(clinical_features)),
                    }
            
            center_labels = batch['center_idx'].to(device)
            
            output = model(
                oct_features=oct_feat,
                colpo_features=colpo_feat,
                clinical_features=None,
                clinical_data=clinical_data,
                center_labels=center_labels,
                return_loss_components=False
            )
            
            if 'z_causal' in output:
                z_causal_list.append(output['z_causal'].cpu().numpy())
            if 'z_sem' in output:
                z_sem_list.append(output['z_sem'].cpu().numpy())
            
            if 'z_causal' in output and 'z_sem' in output:
                z_causal = output['z_causal']
                z_sem = output['z_sem']
                multimodal_feat = torch.cat([z_causal, z_sem], dim=-1)
                if hasattr(model, 'multimodal_fusion'):
                    fused = model.multimodal_fusion(multimodal_feat)
                else:
                    fused = (z_causal + z_sem) / 2.0
                fused_list.append(fused.cpu().numpy())
            
            labels_list.append(batch['label'].numpy())
            centers_list.append(batch['center_idx'].numpy())
    
    features_dict = {}
    if z_causal_list:
        features_dict['z_causal'] = np.concatenate(z_causal_list, axis=0)
    if z_sem_list:
        features_dict['z_sem'] = np.concatenate(z_sem_list, axis=0)
    if fused_list:
        features_dict['fused'] = np.concatenate(fused_list, axis=0)
    
    labels = np.concatenate(labels_list, axis=0)
    centers = np.concatenate(centers_list, axis=0)
    
    return features_dict, labels, centers


def visualize_tsne_nature_style(features_dict, labels, centers, output_dir, timestamp, feature_name='fused'):
    """
    生成Nature/Science风格的t-SNE可视化（2x2网格布局）
    类似用户提供的图片样式
    """
    features = features_dict[feature_name]
    
    print(f"📊 特征形状: {features.shape}")
    print(f"📊 标签分布: {np.bincount(labels)}")
    print(f"📊 中心分布: {np.bincount(centers)}")
    
    # PCA预降维
    if features.shape[1] > 50:
        print(f"⚠️ 特征维度 {features.shape[1]} 较高，先用PCA降维到50维...")
        pca = PCA(n_components=50, random_state=42)
        features = pca.fit_transform(features)
        print(f"✅ PCA降维完成，保留方差: {pca.explained_variance_ratio_.sum():.4f}")
    
    # t-SNE降维
    print("🔄 正在进行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=1)
    features_2d = tsne.fit_transform(features)
    print("✅ t-SNE降维完成")
    
    # ========== 图1: 2x2网格布局（按标签和中心组合） ==========
    fig1, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig1.suptitle('t-SNE Visualization of Feature Space', 
                  fontsize=16, fontweight='bold', y=0.98)
    
    # 准备数据：按标签和中心分组
    unique_centers = np.unique(centers)
    
    # 左上：Negative + Center 0
    ax1 = axes[0, 0]
    for center_id in unique_centers:
        mask = (labels == 0) & (centers == center_id)
        if np.sum(mask) > 0:
            color = COLORS_CENTER[int(center_id) % len(COLORS_CENTER)]
            ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=color, label=f'Center {int(center_id)}', 
                       alpha=0.7, s=40, edgecolors='white', linewidths=0.5, zorder=2)
    ax1.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Negative Samples by Center', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 右上：Positive + Center 0
    ax2 = axes[0, 1]
    for center_id in unique_centers:
        mask = (labels == 1) & (centers == center_id)
        if np.sum(mask) > 0:
            color = COLORS_CENTER[int(center_id) % len(COLORS_CENTER)]
            ax2.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=color, label=f'Center {int(center_id)}', 
                       alpha=0.7, s=40, edgecolors='white', linewidths=0.5, zorder=2)
    ax2.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Positive Samples by Center', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 左下：按标签着色（所有样本）
    ax3 = axes[1, 0]
    for label_val in [0, 1]:
        mask = labels == label_val
        color = COLORS_LABEL[label_val]
        label_name = 'Negative' if label_val == 0 else 'Positive'
        ax3.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=color, label=label_name, alpha=0.7, s=40,
                   edgecolors='white', linewidths=0.5, zorder=2)
    ax3.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
    ax3.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
    ax3.set_title('(c) All Samples by Label', fontsize=13, fontweight='bold', pad=10)
    ax3.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax3.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 右下：按中心着色（所有样本）
    ax4 = axes[1, 1]
    for center_id in unique_centers:
        mask = centers == center_id
        color = COLORS_CENTER[int(center_id) % len(COLORS_CENTER)]
        ax4.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=color, label=f'Center {int(center_id)}', 
                   alpha=0.7, s=40, edgecolors='white', linewidths=0.5, zorder=2)
    ax4.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
    ax4.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
    ax4.set_title('(d) All Samples by Center', fontsize=13, fontweight='bold', pad=10)
    ax4.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax4.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plot_file1 = output_dir / f"tsne_nature_style_grid_{timestamp}.png"
    fig1.savefig(plot_file1, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 2x2网格图已保存: {plot_file1}")
    
    plot_file1_pdf = output_dir / f"tsne_nature_style_grid_{timestamp}.pdf"
    fig1.savefig(plot_file1_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 2x2网格图PDF已保存: {plot_file1_pdf}")
    
    plt.close(fig1)
    
    # ========== 图2: 并排对比（类似用户提供的UMAP图） ==========
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('t-SNE Feature Space Comparison', 
                  fontsize=16, fontweight='bold', y=0.98)
    
    # 左图：按标签着色
    ax5 = axes[0]
    for label_val in [0, 1]:
        mask = labels == label_val
        color = COLORS_LABEL[label_val]
        label_name = 'Negative' if label_val == 0 else 'Positive'
        ax5.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=color, label=label_name, alpha=0.7, s=50,
                   edgecolors='white', linewidths=0.8, zorder=2)
    ax5.set_xlabel('t-SNE 1', fontsize=13, fontweight='bold')
    ax5.set_ylabel('t-SNE 2', fontsize=13, fontweight='bold')
    ax5.set_title('(a) Colored by Label', fontsize=14, fontweight='bold', pad=12)
    ax5.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax5.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    # 右图：按中心着色
    ax6 = axes[1]
    for center_id in unique_centers:
        mask = centers == center_id
        color = COLORS_CENTER[int(center_id) % len(COLORS_CENTER)]
        ax6.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=color, label=f'Center {int(center_id)}', 
                   alpha=0.7, s=50, edgecolors='white', linewidths=0.8, zorder=2)
    ax6.set_xlabel('t-SNE 1', fontsize=13, fontweight='bold')
    ax6.set_ylabel('t-SNE 2', fontsize=13, fontweight='bold')
    ax6.set_title('(b) Colored by Center', fontsize=14, fontweight='bold', pad=12)
    ax6.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax6.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_file2 = output_dir / f"tsne_nature_style_sidebyside_{timestamp}.png"
    fig2.savefig(plot_file2, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 并排对比图已保存: {plot_file2}")
    
    plot_file2_pdf = output_dir / f"tsne_nature_style_sidebyside_{timestamp}.pdf"
    fig2.savefig(plot_file2_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 并排对比图PDF已保存: {plot_file2_pdf}")
    
    plt.close(fig2)
    
    # ========== 图3: 特征对比（z_causal vs z_sem，类似用户提供的对比图） ==========
    if 'z_causal' in features_dict and 'z_sem' in features_dict:
        fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig3.suptitle('t-SNE Comparison: Causal vs Semantic Features', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # z_causal的t-SNE
        z_causal = features_dict['z_causal']
        if z_causal.shape[1] > 50:
            pca = PCA(n_components=50, random_state=42)
            z_causal = pca.fit_transform(z_causal)
        tsne_causal = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=0)
        z_causal_2d = tsne_causal.fit_transform(z_causal)
        
        ax7 = axes[0]
        for label_val in [0, 1]:
            mask = labels == label_val
            color = COLORS_LABEL[label_val]
            label_name = 'Negative' if label_val == 0 else 'Positive'
            ax7.scatter(z_causal_2d[mask, 0], z_causal_2d[mask, 1], 
                       c=color, label=label_name, alpha=0.7, s=50,
                       edgecolors='white', linewidths=0.8, zorder=2)
        ax7.set_xlabel('t-SNE 1', fontsize=13, fontweight='bold')
        ax7.set_ylabel('t-SNE 2', fontsize=13, fontweight='bold')
        ax7.set_title('(a) Causal Features (z_causal)', fontsize=14, fontweight='bold', pad=12)
        ax7.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=11)
        ax7.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
        ax7.spines['top'].set_visible(False)
        ax7.spines['right'].set_visible(False)
        
        # z_sem的t-SNE
        z_sem = features_dict['z_sem']
        if z_sem.shape[1] > 50:
            pca = PCA(n_components=50, random_state=42)
            z_sem = pca.fit_transform(z_sem)
        tsne_sem = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=0)
        z_sem_2d = tsne_sem.fit_transform(z_sem)
        
        ax8 = axes[1]
        for label_val in [0, 1]:
            mask = labels == label_val
            color = COLORS_LABEL[label_val]
            label_name = 'Negative' if label_val == 0 else 'Positive'
            ax8.scatter(z_sem_2d[mask, 0], z_sem_2d[mask, 1], 
                       c=color, label=label_name, alpha=0.7, s=50,
                       edgecolors='white', linewidths=0.8, zorder=2)
        ax8.set_xlabel('t-SNE 1', fontsize=13, fontweight='bold')
        ax8.set_ylabel('t-SNE 2', fontsize=13, fontweight='bold')
        ax8.set_title('(b) Semantic Features (z_sem)', fontsize=14, fontweight='bold', pad=12)
        ax8.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=11)
        ax8.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
        ax8.spines['top'].set_visible(False)
        ax8.spines['right'].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        plot_file3 = output_dir / f"tsne_nature_style_comparison_{timestamp}.png"
        fig3.savefig(plot_file3, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"📊 特征对比图已保存: {plot_file3}")
        
        plot_file3_pdf = output_dir / f"tsne_nature_style_comparison_{timestamp}.pdf"
        fig3.savefig(plot_file3_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"📊 特征对比图PDF已保存: {plot_file3_pdf}")
        
        plt.close(fig3)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='生成Nature/Science风格的t-SNE可视化')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, 
                       default='/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--feature', type=str, default='fused', 
                       choices=['z_causal', 'z_sem', 'z_noise', 'fused'])
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备: {device}")
    
    # 加载数据集
    from experiments.exp_5centers.dataset_v2 import FiveCentersMultimodalDatasetV2
    
    data_root = Path(args.data_root)
    if args.split == 'train':
        csv_path = data_root / 'internal_train' / 'labels.csv'
    elif args.split == 'val':
        csv_path = data_root / 'internal_val' / 'labels.csv'
    else:
        csv_path = data_root / 'internal_test' / 'labels.csv'
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = FiveCentersMultimodalDatasetV2(
        csv_path=str(csv_path),
        clinical_embed_path=None,
        transform=transform,
        oct_num_frames=20,
        max_col_images=3,
        balance_negative_frames=True,
        use_llm=False
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"✅ 数据集: {len(dataset)} 个样本")
    
    # 加载模型
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint.get('args', {})
    
    sample = dataset[0]
    oct_images = sample['oct_images'].unsqueeze(0).to(device)
    oct_feat = extract_features_with_vit(oct_images, device)
    input_dim = oct_feat.shape[-1]
    
    model = BioCOTModel(
        embed_dim=model_args.get('embed_dim', 768),
        num_classes=model_args.get('num_classes', 2),
        num_centers=model_args.get('num_centers', 4),
        input_dim=input_dim,
        use_vlm_encoder=model_args.get('use_vlm_encoder', False)
    )
    
    model = model.to(device)
    
    # 创建动态层
    with torch.no_grad():
        clinical_data = {
            'age': np.array([sample['clinical_data'].get('age', 50)]),
            'hpv': np.array([sample['clinical_data'].get('hpv', 0)]),
            'tct': np.array([sample['clinical_data'].get('tct', 0)]),
        }
        center_labels = torch.tensor([sample['center_idx']], dtype=torch.long).to(device)
        _ = model(oct_features=oct_feat, colpo_features=oct_feat, clinical_features=None,
                 clinical_data=clinical_data, center_labels=center_labels, return_loss_components=False)
    
    state_dict = checkpoint['model_state_dict']
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    print(f"✅ 模型加载完成，最佳AUC: {checkpoint.get('best_auc', 0):.4f}\n")
    
    # 提取特征
    features_dict, labels, centers = extract_features_v1(model, dataloader, device)
    
    # 生成可视化
    timestamp = Path(args.checkpoint).stem.replace('best_model_5centers_', '').replace('best_model_', '')
    output_dir = Path(args.output) if args.output else Path(args.checkpoint).parent.parent / 'logs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_tsne_nature_style(features_dict, labels, centers, output_dir, timestamp, feature_name=args.feature)
    
    print(f"\n✅ Nature/Science风格t-SNE可视化完成！")


if __name__ == '__main__':
    main()

