#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
t-SNE可视化质量评估工具
- 评估特征空间分离度
- 分析类别聚类质量
- 检测中心偏差
- 计算定量指标
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
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import cdist
from tqdm import tqdm
import pandas as pd

# 设置matplotlib后端
matplotlib.use('Agg')

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.bida.bio_cot_model import BioCOTModel
from experiments.exp_5centers.dataset_v2 import FiveCentersMultimodalDatasetV2
from experiments.exp_5centers.train_bio_cot_5centers_multimodal import extract_features_with_vit
from torch.utils.data import DataLoader
from torchvision import transforms


def extract_features_for_analysis(model, dataloader, device):
    """提取特征用于分析"""
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
            
            clinical_data = {
                'age': batch['clinical_data']['age'].cpu().numpy() if 'clinical_data' in batch else np.array([50] * len(batch['label'])),
                'hpv': batch['clinical_data']['hpv'].cpu().numpy() if 'clinical_data' in batch else np.zeros(len(batch['label'])),
                'tct': batch['clinical_data']['tct'].cpu().numpy() if 'clinical_data' in batch else np.zeros(len(batch['label'])),
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


def calculate_separation_metrics(features, labels):
    """计算特征空间分离度指标"""
    metrics = {}
    
    # 1. Silhouette Score（轮廓系数，-1到1，越大越好）
    try:
        metrics['silhouette'] = silhouette_score(features, labels)
    except:
        metrics['silhouette'] = None
    
    # 2. Calinski-Harabasz Score（CH指数，越大越好）
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(features, labels)
    except:
        metrics['calinski_harabasz'] = None
    
    # 3. Davies-Bouldin Score（DB指数，越小越好）
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(features, labels)
    except:
        metrics['davies_bouldin'] = None
    
    # 4. 类间距离 / 类内距离（越大越好）
    try:
        class_0 = features[labels == 0]
        class_1 = features[labels == 1]
        
        # 类内距离（平均）
        intra_dist_0 = np.mean(cdist(class_0, class_0))
        intra_dist_1 = np.mean(cdist(class_1, class_1))
        intra_dist = (intra_dist_0 + intra_dist_1) / 2
        
        # 类间距离（平均）
        inter_dist = np.mean(cdist(class_0, class_1))
        
        metrics['inter_intra_ratio'] = inter_dist / (intra_dist + 1e-8)
        metrics['inter_dist'] = inter_dist
        metrics['intra_dist'] = intra_dist
    except:
        metrics['inter_intra_ratio'] = None
    
    # 5. KNN分类准确率（在特征空间上）
    try:
        if len(features) > 10:
            knn = KNeighborsClassifier(n_neighbors=min(5, len(features) // 10))
            scores = cross_val_score(knn, features, labels, cv=min(5, len(features) // 20), scoring='accuracy')
            metrics['knn_accuracy'] = scores.mean()
            metrics['knn_std'] = scores.std()
        else:
            metrics['knn_accuracy'] = None
    except:
        metrics['knn_accuracy'] = None
    
    return metrics


def calculate_center_bias(features, centers, labels):
    """计算中心偏差指标"""
    metrics = {}
    
    unique_centers = np.unique(centers)
    
    # 1. 中心间距离
    center_means = {}
    for center_id in unique_centers:
        center_mask = centers == center_id
        center_means[center_id] = np.mean(features[center_mask], axis=0)
    
    # 计算中心间平均距离
    center_distances = []
    for i, center_i in enumerate(unique_centers):
        for j, center_j in enumerate(unique_centers):
            if i < j:
                dist = np.linalg.norm(center_means[center_i] - center_means[center_j])
                center_distances.append(dist)
    
    metrics['center_inter_distance'] = np.mean(center_distances) if center_distances else None
    
    # 2. 中心内类间距离（每个中心内，不同类别的分离度）
    center_intra_class_sep = []
    for center_id in unique_centers:
        center_mask = centers == center_id
        center_features = features[center_mask]
        center_labels_subset = labels[center_mask]
        
        if len(np.unique(center_labels_subset)) > 1:
            class_0 = center_features[center_labels_subset == 0]
            class_1 = center_features[center_labels_subset == 1]
            if len(class_0) > 0 and len(class_1) > 0:
                dist = np.mean(cdist(class_0, class_1))
                center_intra_class_sep.append(dist)
    
    metrics['center_intra_class_sep'] = np.mean(center_intra_class_sep) if center_intra_class_sep else None
    
    # 3. 跨中心类间距离（不同中心，相同类别的距离）
    cross_center_same_class_dist = []
    for label_val in [0, 1]:
        label_mask = labels == label_val
        label_features = features[label_mask]
        label_centers = centers[label_mask]
        
        for i, center_i in enumerate(unique_centers):
            for j, center_j in enumerate(unique_centers):
                if i < j:
                    center_i_mask = label_centers == center_i
                    center_j_mask = label_centers == center_j
                    if np.sum(center_i_mask) > 0 and np.sum(center_j_mask) > 0:
                        dist = np.mean(cdist(label_features[center_i_mask], label_features[center_j_mask]))
                        cross_center_same_class_dist.append(dist)
    
    metrics['cross_center_same_class_dist'] = np.mean(cross_center_same_class_dist) if cross_center_same_class_dist else None
    
    return metrics


def evaluate_tsne_quality(features_dict, labels, centers, output_dir, timestamp):
    """全面评估t-SNE可视化质量"""
    
    print("=" * 80)
    print("📊 t-SNE可视化质量评估报告")
    print("=" * 80)
    
    results = {}
    
    for feature_name, features in features_dict.items():
        print(f"\n🔍 评估特征: {feature_name}")
        print("-" * 80)
        
        # PCA预降维（如果维度太高）
        if features.shape[1] > 50:
            pca = PCA(n_components=50, random_state=42)
            features_reduced = pca.fit_transform(features)
            print(f"   PCA降维: {features.shape[1]} → 50 (保留方差: {pca.explained_variance_ratio_.sum():.4f})")
        else:
            features_reduced = features
        
        # t-SNE降维
        print("   🔄 进行t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=0)
        features_2d = tsne.fit_transform(features_reduced)
        
        # 计算分离度指标（在原始高维空间）
        print("   📈 计算分离度指标（高维空间）...")
        separation_metrics = calculate_separation_metrics(features_reduced, labels)
        
        # 计算分离度指标（在2D t-SNE空间）
        print("   📈 计算分离度指标（2D t-SNE空间）...")
        separation_metrics_2d = calculate_separation_metrics(features_2d, labels)
        
        # 计算中心偏差指标
        print("   📈 计算中心偏差指标...")
        center_bias_metrics = calculate_center_bias(features_reduced, centers, labels)
        
        # 打印结果
        print("\n   📊 分离度指标（高维空间）:")
        if separation_metrics['silhouette'] is not None:
            print(f"      - Silhouette Score: {separation_metrics['silhouette']:.4f} (范围: -1到1，越大越好)")
            if separation_metrics['silhouette'] > 0.5:
                print("        ✅ 优秀：类别分离度很高")
            elif separation_metrics['silhouette'] > 0.3:
                print("        ✅ 良好：类别分离度较好")
            elif separation_metrics['silhouette'] > 0.1:
                print("        ⚠️  一般：类别分离度中等")
            else:
                print("        ❌ 较差：类别分离度较低")
        
        if separation_metrics['inter_intra_ratio'] is not None:
            print(f"      - 类间/类内距离比: {separation_metrics['inter_intra_ratio']:.4f} (越大越好)")
            if separation_metrics['inter_intra_ratio'] > 2.0:
                print("        ✅ 优秀：类间距离远大于类内距离")
            elif separation_metrics['inter_intra_ratio'] > 1.5:
                print("        ✅ 良好：类间距离明显大于类内距离")
            elif separation_metrics['inter_intra_ratio'] > 1.0:
                print("        ⚠️  一般：类间距离略大于类内距离")
            else:
                print("        ❌ 较差：类间距离小于或接近类内距离")
        
        if separation_metrics['knn_accuracy'] is not None:
            print(f"      - KNN分类准确率: {separation_metrics['knn_accuracy']:.4f} ± {separation_metrics['knn_std']:.4f}")
            if separation_metrics['knn_accuracy'] > 0.9:
                print("        ✅ 优秀：特征空间分类能力很强")
            elif separation_metrics['knn_accuracy'] > 0.8:
                print("        ✅ 良好：特征空间分类能力较好")
            elif separation_metrics['knn_accuracy'] > 0.7:
                print("        ⚠️  一般：特征空间分类能力中等")
            else:
                print("        ❌ 较差：特征空间分类能力较弱")
        
        print("\n   📊 分离度指标（2D t-SNE空间）:")
        if separation_metrics_2d['silhouette'] is not None:
            print(f"      - Silhouette Score: {separation_metrics_2d['silhouette']:.4f}")
            # t-SNE会保持局部结构，但可能扭曲全局结构
            if separation_metrics_2d['silhouette'] > separation_metrics['silhouette'] * 0.8:
                print("        ✅ t-SNE保持了较好的分离结构")
            else:
                print("        ⚠️  t-SNE可能丢失了一些分离信息")
        
        print("\n   📊 中心偏差指标:")
        if center_bias_metrics['center_inter_distance'] is not None:
            print(f"      - 中心间平均距离: {center_bias_metrics['center_inter_distance']:.4f}")
        
        if center_bias_metrics['center_intra_class_sep'] is not None:
            print(f"      - 中心内类间分离度: {center_bias_metrics['center_intra_class_sep']:.4f}")
        
        if center_bias_metrics['cross_center_same_class_dist'] is not None:
            print(f"      - 跨中心同类距离: {center_bias_metrics['cross_center_same_class_dist']:.4f}")
            if center_bias_metrics['cross_center_same_class_dist'] < center_bias_metrics.get('center_inter_distance', float('inf')):
                print("        ⚠️  可能存在中心偏差：不同中心的同类样本距离较近")
            else:
                print("        ✅ 中心偏差较小：不同中心的同类样本距离合理")
        
        # 保存结果
        results[feature_name] = {
            'separation_metrics': separation_metrics,
            'separation_metrics_2d': separation_metrics_2d,
            'center_bias_metrics': center_bias_metrics
        }
    
    # 生成评估报告
    report_file = output_dir / f"tsne_quality_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("t-SNE可视化质量评估报告\n")
        f.write("=" * 80 + "\n\n")
        
        for feature_name, result in results.items():
            f.write(f"特征: {feature_name}\n")
            f.write("-" * 80 + "\n")
            f.write(f"高维空间分离度:\n")
            f.write(f"  - Silhouette Score: {result['separation_metrics'].get('silhouette', 'N/A')}\n")
            f.write(f"  - 类间/类内距离比: {result['separation_metrics'].get('inter_intra_ratio', 'N/A')}\n")
            f.write(f"  - KNN准确率: {result['separation_metrics'].get('knn_accuracy', 'N/A')}\n")
            f.write(f"\n2D t-SNE空间分离度:\n")
            f.write(f"  - Silhouette Score: {result['separation_metrics_2d'].get('silhouette', 'N/A')}\n")
            f.write(f"\n中心偏差:\n")
            f.write(f"  - 中心间距离: {result['center_bias_metrics'].get('center_inter_distance', 'N/A')}\n")
            f.write(f"  - 跨中心同类距离: {result['center_bias_metrics'].get('cross_center_same_class_dist', 'N/A')}\n")
            f.write("\n")
    
    print(f"\n✅ 评估报告已保存: {report_file}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='评估t-SNE可视化质量')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--data_root', type=str, 
                       default='/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal',
                       help='数据根目录')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                       help='使用哪个数据集')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
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
        clinical_embed_path=None,
        transform=transform,
        oct_num_frames=20,
        max_col_images=3,
        balance_negative_frames=True,
        use_llm=False
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"✅ 数据集加载完成: {len(dataset)} 个样本")
    
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
    print(f"✅ 模型加载完成，最佳AUC: {checkpoint.get('best_auc', 0):.4f}")
    
    # 提取特征
    features_dict, labels, centers = extract_features_for_analysis(model, dataloader, device)
    
    # 评估质量
    timestamp = Path(args.checkpoint).stem.replace('best_model_5centers_', '').replace('best_model_', '')
    output_dir = Path(args.output) if args.output else Path(args.checkpoint).parent.parent / 'logs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = evaluate_tsne_quality(features_dict, labels, centers, output_dir, timestamp)
    
    print("\n" + "=" * 80)
    print("✅ 评估完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()



