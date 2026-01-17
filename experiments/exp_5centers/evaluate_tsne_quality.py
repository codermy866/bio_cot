#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
t-SNE可视化质量评估工具（简化版）
基于已生成的t-SNE结果进行定量评估
"""

import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib

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


def extract_features_simple(model, dataloader, device):
    """提取特征"""
    model.eval()
    
    fused_list = []
    z_causal_list = []
    z_sem_list = []
    labels_list = []
    centers_list = []
    
    with torch.no_grad():
        for batch in dataloader:
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


def evaluate_separation(features, labels):
    """评估类别分离度"""
    metrics = {}
    
    # 1. Silhouette Score
    try:
        metrics['silhouette'] = silhouette_score(features, labels)
    except:
        metrics['silhouette'] = None
    
    # 2. 类间/类内距离比
    try:
        class_0 = features[labels == 0]
        class_1 = features[labels == 1]
        
        intra_dist_0 = np.mean(cdist(class_0, class_0))
        intra_dist_1 = np.mean(cdist(class_1, class_1))
        intra_dist = (intra_dist_0 + intra_dist_1) / 2
        inter_dist = np.mean(cdist(class_0, class_1))
        
        metrics['inter_intra_ratio'] = inter_dist / (intra_dist + 1e-8)
        metrics['inter_dist'] = inter_dist
        metrics['intra_dist'] = intra_dist
    except:
        metrics['inter_intra_ratio'] = None
    
    # 3. KNN分类准确率
    try:
        if len(features) > 10:
            knn = KNeighborsClassifier(n_neighbors=min(5, len(features) // 10))
            scores = cross_val_score(knn, features, labels, cv=min(5, len(features) // 20), scoring='accuracy')
            metrics['knn_accuracy'] = scores.mean()
        else:
            metrics['knn_accuracy'] = None
    except:
        metrics['knn_accuracy'] = None
    
    return metrics


def evaluate_center_bias(features, centers, labels):
    """评估中心偏差"""
    unique_centers = np.unique(centers)
    
    # 计算每个中心的特征中心
    center_means = {}
    for center_id in unique_centers:
        center_mask = centers == center_id
        center_means[center_id] = np.mean(features[center_mask], axis=0)
    
    # 中心间距离
    center_distances = []
    for i, center_i in enumerate(unique_centers):
        for j, center_j in enumerate(unique_centers):
            if i < j:
                dist = np.linalg.norm(center_means[center_i] - center_means[center_j])
                center_distances.append(dist)
    
    # 跨中心同类距离
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
    
    return {
        'center_inter_distance': np.mean(center_distances) if center_distances else None,
        'cross_center_same_class_dist': np.mean(cross_center_same_class_dist) if cross_center_same_class_dist else None
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='评估t-SNE可视化质量')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, 
                       default='/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备: {device}")
    
    # 加载数据集
    data_root = Path(args.data_root)
    csv_path = data_root / f'internal_{args.split}' / 'labels.csv'
    
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
    features_dict, labels, centers = extract_features_simple(model, dataloader, device)
    
    # 评估
    print("=" * 80)
    print("📊 t-SNE可视化质量评估报告")
    print("=" * 80)
    
    for feature_name, features in features_dict.items():
        print(f"\n🔍 特征: {feature_name}")
        print("-" * 80)
        
        # PCA预降维
        if features.shape[1] > 50:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50, random_state=42)
            features_reduced = pca.fit_transform(features)
            print(f"   特征维度: {features.shape[1]} → 50 (PCA保留方差: {pca.explained_variance_ratio_.sum():.4f})")
        else:
            features_reduced = features
        
        # 评估分离度
        separation = evaluate_separation(features_reduced, labels)
        
        print(f"\n   📈 类别分离度指标:")
        if separation['silhouette'] is not None:
            score = separation['silhouette']
            print(f"      - Silhouette Score: {score:.4f} (范围: -1到1，越大越好)")
            if score > 0.5:
                print("        ✅ 优秀：类别分离度很高，特征空间区分能力强")
            elif score > 0.3:
                print("        ✅ 良好：类别分离度较好，特征空间有一定区分能力")
            elif score > 0.1:
                print("        ⚠️  一般：类别分离度中等，特征空间区分能力有限")
            else:
                print("        ❌ 较差：类别分离度较低，特征空间区分能力弱")
        
        if separation['inter_intra_ratio'] is not None:
            ratio = separation['inter_intra_ratio']
            print(f"      - 类间/类内距离比: {ratio:.4f} (越大越好)")
            print(f"        (类间距离: {separation['inter_dist']:.4f}, 类内距离: {separation['intra_dist']:.4f})")
            if ratio > 2.0:
                print("        ✅ 优秀：类间距离远大于类内距离，类别分离明显")
            elif ratio > 1.5:
                print("        ✅ 良好：类间距离明显大于类内距离，类别分离较好")
            elif ratio > 1.0:
                print("        ⚠️  一般：类间距离略大于类内距离，类别分离一般")
            else:
                print("        ❌ 较差：类间距离小于或接近类内距离，类别重叠严重")
        
        if separation['knn_accuracy'] is not None:
            acc = separation['knn_accuracy']
            print(f"      - KNN分类准确率: {acc:.4f}")
            if acc > 0.9:
                print("        ✅ 优秀：特征空间分类能力很强")
            elif acc > 0.8:
                print("        ✅ 良好：特征空间分类能力较好")
            elif acc > 0.7:
                print("        ⚠️  一般：特征空间分类能力中等")
            else:
                print("        ❌ 较差：特征空间分类能力较弱")
        
        # 评估中心偏差
        center_bias = evaluate_center_bias(features_reduced, centers, labels)
        
        print(f"\n   📊 中心偏差指标:")
        if center_bias['center_inter_distance'] is not None:
            print(f"      - 中心间平均距离: {center_bias['center_inter_distance']:.4f}")
        
        if center_bias['cross_center_same_class_dist'] is not None:
            cross_dist = center_bias['cross_center_same_class_dist']
            center_dist = center_bias['center_inter_distance']
            print(f"      - 跨中心同类距离: {cross_dist:.4f}")
            if center_dist and cross_dist < center_dist * 0.8:
                print("        ⚠️  可能存在中心偏差：不同中心的同类样本距离较近")
                print("        💡 建议：加强对抗训练或中心对齐损失")
            else:
                print("        ✅ 中心偏差较小：不同中心的同类样本距离合理")
    
    print("\n" + "=" * 80)
    print("✅ 评估完成！")
    print("=" * 80)
    
    # 生成总结
    print("\n📝 评估总结:")
    print("-" * 80)
    best_feature = max(features_dict.keys(), 
                      key=lambda k: evaluate_separation(
                          PCA(n_components=50, random_state=42).fit_transform(features_dict[k]) 
                          if features_dict[k].shape[1] > 50 else features_dict[k], 
                          labels
                      ).get('silhouette', 0) or 0)
    
    best_separation = evaluate_separation(
        PCA(n_components=50, random_state=42).fit_transform(features_dict[best_feature]) 
        if features_dict[best_feature].shape[1] > 50 else features_dict[best_feature], 
        labels
    )
    
    print(f"   最佳特征: {best_feature}")
    if best_separation.get('silhouette'):
        print(f"   Silhouette Score: {best_separation['silhouette']:.4f}")
    if best_separation.get('inter_intra_ratio'):
        print(f"   类间/类内距离比: {best_separation['inter_intra_ratio']:.4f}")
    if best_separation.get('knn_accuracy'):
        print(f"   KNN准确率: {best_separation['knn_accuracy']:.4f}")


if __name__ == '__main__':
    main()



