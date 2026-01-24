#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成t-SNE可视化图表（适合论文发表）
- 可视化特征空间分布
- 按标签和中心着色
- 高分辨率输出
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

# 设置学术论文风格
plt.style.use('seaborn-v0_8-whitegrid')
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
    'lines.markersize': 5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.bida.bio_cot_v2 import BioCOT_v2
from experiments.exp_5centers.dataset_v2 import FiveCentersMultimodalDatasetV2
from experiments.exp_5centers.train_bio_cot_5centers_multimodal import extract_features_with_vit
from torch.utils.data import DataLoader
from torchvision import transforms


def extract_features(model, dataloader, device, use_llm=True):
    """
    从模型中提取特征
    
    Returns:
        features_dict: 包含各种特征的字典
        labels: 标签
        centers: 中心ID
    """
    model.eval()
    
    z_causal_list = []
    z_sem_list = []
    z_noise_list = []
    fused_list = []
    labels_list = []
    centers_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取特征"):
            # 提取图像特征
            oct_images = batch['oct_images'].to(device)
            colposcopy_images = batch['colposcopy_images'].to(device)
            
            # 提取ViT特征
            oct_feat = extract_features_with_vit(oct_images, device)
            colpo_feat = extract_features_with_vit(colposcopy_images, device)
            
            # 融合图像特征
            img_feat = (oct_feat + colpo_feat) / 2.0
            
            # 获取临床嵌入
            if use_llm:
                clinical_embeddings = batch['clinical_embedding'].to(device)
                clinical_data = None
            else:
                clinical_embeddings = None
                clinical_data = batch['clinical_data']
            
            center_labels = batch['center_idx'].to(device)
            
            # 前向传播
            output = model(
                oct_features=oct_feat,
                colpo_features=colpo_feat,
                clinical_embeddings=clinical_embeddings,
                clinical_data=clinical_data,
                center_labels=center_labels,
                return_loss_components=False
            )
            
            # 收集特征
            z_causal_list.append(output['z_causal'].cpu().numpy())
            z_sem_list.append(output['z_sem'].cpu().numpy())
            
            if 'z_noise' in output:
                z_noise_list.append(output['z_noise'].cpu().numpy())
            
            # 获取融合后的特征（用于分类的特征）
            # 需要手动计算融合特征
            if model.use_cross_attn:
                # Cross-Attention融合
                fused = model.fusion_module(output['z_causal'], output['z_sem'])
            else:
                # Concat融合
                multimodal_feat = torch.cat([output['z_causal'], output['z_sem']], dim=-1)
                fused = model.fusion_module(multimodal_feat)
            fused_list.append(fused.cpu().numpy())
            
            labels_list.append(batch['label'].numpy())
            centers_list.append(batch['center_idx'].numpy())
    
    # 合并所有batch
    features_dict = {
        'z_causal': np.concatenate(z_causal_list, axis=0),
        'z_sem': np.concatenate(z_sem_list, axis=0),
        'fused': np.concatenate(fused_list, axis=0),
    }
    
    if z_noise_list:
        features_dict['z_noise'] = np.concatenate(z_noise_list, axis=0)
    
    labels = np.concatenate(labels_list, axis=0)
    centers = np.concatenate(centers_list, axis=0)
    
    return features_dict, labels, centers


def visualize_tsne(features_dict, labels, centers, output_dir, timestamp, feature_name='fused'):
    """
    生成t-SNE可视化图表
    
    Args:
        features_dict: 特征字典
        labels: 标签数组
        centers: 中心ID数组
        output_dir: 输出目录
        timestamp: 时间戳
        feature_name: 要可视化的特征名称
    """
    features = features_dict[feature_name]
    
    print(f"📊 特征形状: {features.shape}")
    print(f"📊 标签分布: {np.bincount(labels)}")
    print(f"📊 中心分布: {np.bincount(centers)}")
    
    # 如果特征维度太高，先用PCA降维到50维（t-SNE建议）
    if features.shape[1] > 50:
        print(f"⚠️ 特征维度 {features.shape[1]} 较高，先用PCA降维到50维...")
        pca = PCA(n_components=50, random_state=42)
        features = pca.fit_transform(features)
        print(f"✅ PCA降维完成，保留方差: {pca.explained_variance_ratio_.sum():.4f}")
    
    # t-SNE降维
    print("🔄 正在进行t-SNE降维（这可能需要几分钟）...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=1)
    features_2d = tsne.fit_transform(features)
    print("✅ t-SNE降维完成")
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f't-SNE Visualization of {feature_name.upper()} Features', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. 按标签着色
    ax1 = axes[0]
    colors = ['#1f77b4', '#ff7f0e']  # 蓝色（阴性），橙色（阳性）
    labels_names = ['Negative', 'Positive']
    
    for i, (label_val, label_name, color) in enumerate(zip([0, 1], labels_names, colors)):
        mask = labels == label_val
        ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=color, label=label_name, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
    
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Colored by Label', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 2. 按中心着色
    ax2 = axes[1]
    center_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 不同颜色代表不同中心
    unique_centers = np.unique(centers)
    
    for center_id in unique_centers:
        mask = centers == center_id
        color = center_colors[int(center_id) % len(center_colors)]
        ax2.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=color, label=f'Center {int(center_id)}', alpha=0.6, s=30, 
                   edgecolors='black', linewidths=0.5)
    
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Colored by Center', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图表
    plot_file = output_dir / f"tsne_{feature_name}_{timestamp}.png"
    fig.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 t-SNE图已保存: {plot_file}")
    
    plot_file_pdf = output_dir / f"tsne_{feature_name}_{timestamp}.pdf"
    fig.savefig(plot_file_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 t-SNE图PDF已保存: {plot_file_pdf}")
    
    plt.close(fig)
    
    # 创建单独的特征对比图（z_causal vs z_sem）
    if 'z_causal' in features_dict and 'z_sem' in features_dict:
        fig2, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig2.suptitle('t-SNE Comparison: Causal Features vs Semantic Features', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # z_causal的t-SNE
        z_causal = features_dict['z_causal']
        if z_causal.shape[1] > 50:
            pca = PCA(n_components=50, random_state=42)
            z_causal = pca.fit_transform(z_causal)
        tsne_causal = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=0)
        z_causal_2d = tsne_causal.fit_transform(z_causal)
        
        ax3 = axes[0]
        for i, (label_val, label_name, color) in enumerate(zip([0, 1], labels_names, colors)):
            mask = labels == label_val
            ax3.scatter(z_causal_2d[mask, 0], z_causal_2d[mask, 1], 
                       c=color, label=label_name, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
        ax3.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
        ax3.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
        ax3.set_title('(a) Causal Features (z_causal)', fontsize=13, fontweight='bold', pad=10)
        ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # z_sem的t-SNE
        z_sem = features_dict['z_sem']
        if z_sem.shape[1] > 50:
            pca = PCA(n_components=50, random_state=42)
            z_sem = pca.fit_transform(z_sem)
        tsne_sem = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=0)
        z_sem_2d = tsne_sem.fit_transform(z_sem)
        
        ax4 = axes[1]
        for i, (label_val, label_name, color) in enumerate(zip([0, 1], labels_names, colors)):
            mask = labels == label_val
            ax4.scatter(z_sem_2d[mask, 0], z_sem_2d[mask, 1], 
                       c=color, label=label_name, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
        ax4.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
        ax4.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
        ax4.set_title('(b) Semantic Features (z_sem)', fontsize=13, fontweight='bold', pad=10)
        ax4.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        plot_file_comp = output_dir / f"tsne_comparison_{timestamp}.png"
        fig2.savefig(plot_file_comp, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"📊 特征对比图已保存: {plot_file_comp}")
        
        plot_file_comp_pdf = output_dir / f"tsne_comparison_{timestamp}.pdf"
        fig2.savefig(plot_file_comp_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"📊 特征对比图PDF已保存: {plot_file_comp_pdf}")
        
        plt.close(fig2)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='生成t-SNE可视化图表')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--data_root', type=str, 
                       default='/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal',
                       help='数据根目录')
    parser.add_argument('--clinical_embed_path', type=str,
                       default='/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_5centers/data/clinical_embeddings_from_csv.pkl',
                       help='LLM嵌入路径')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                       help='使用哪个数据集')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--feature', type=str, default='fused', 
                       choices=['z_causal', 'z_sem', 'z_noise', 'fused'],
                       help='要可视化的特征')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备: {device}")
    
    # 加载数据集
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
        clinical_embed_path=args.clinical_embed_path,
        transform=transform,
        oct_num_frames=20,
        max_col_images=3,
        balance_negative_frames=True,
        use_llm=True
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"✅ 数据集加载完成: {len(dataset)} 个样本")
    
    # 加载模型
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint.get('args', {})
    
    # 检测特征维度
    sample = dataset[0]
    oct_images = sample['oct_images'].unsqueeze(0).to(device)
    oct_feat = extract_features_with_vit(oct_images, device)
    input_dim = oct_feat.shape[-1]
    llm_embed_dim = sample['clinical_embedding'].shape[-1] if 'clinical_embedding' in sample else 768
    
    model = BioCOT_v2(
        embed_dim=768,
        num_classes=2,
        num_centers=4,
        input_dim=input_dim,
        llm_embed_dim=llm_embed_dim,
        use_ot=model_args.get('use_ot', True),
        use_dual=model_args.get('use_dual', True),
        use_llm=model_args.get('use_llm', True),
        use_cross_attn=model_args.get('use_cross_attn', False)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"✅ 模型加载完成，最佳AUC: {checkpoint.get('best_auc', 0):.4f}")
    
    # 提取特征
    features_dict, labels, centers = extract_features(model, dataloader, device, use_llm=True)
    
    # 生成可视化
    timestamp = Path(args.checkpoint).stem.replace('best_model_v2_', '')
    output_dir = Path(args.output) if args.output else Path(args.checkpoint).parent.parent / 'logs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_tsne(features_dict, labels, centers, output_dir, timestamp, feature_name=args.feature)
    
    print(f"\n✅ t-SNE可视化完成！")


if __name__ == '__main__':
    main()

