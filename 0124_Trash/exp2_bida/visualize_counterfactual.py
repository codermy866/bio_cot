#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 4: 反事实轨迹可视化
绘制"反事实轨迹图"，展示模型学会了"以不变（因果）应万变（噪声）"
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
from src.models.bida.bio_cot_model import BioCOTModel


def visualize_counterfactual_trajectory(
    model, test_dataset, device, 
    num_samples=10, num_centers=5,
    output_file='counterfactual_trajectory.pdf'
):
    """
    可视化反事实轨迹
    
    流程：
    1. 选取测试样本A
    2. 获取z_causal^A
    3. 遍历所有5个中心的典型噪声
    4. 生成合成特征序列：z_mix^Ci = z_causal^A + z_noise^Ci
    5. 使用t-SNE降维
    6. 画图：星号表示纯因果特征，圆点表示合成特征，虚线连接
    """
    model.eval()
    
    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # 收集特征
    all_z_causal = []
    all_z_mix = []
    all_labels = []
    all_center_ids = []
    sample_indices = []
    
    print("🔄 收集特征...")
    
    with torch.no_grad():
        # 随机选择几个样本
        selected_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        
        for idx in selected_indices:
            batch = test_dataset[idx]
            
            # 转换为batch格式
            if isinstance(batch, dict):
                oct_feat = batch['oct_features'].unsqueeze(0).to(device)
                colpo_feat = batch['colposcopy_features'].unsqueeze(0).to(device)
                clinical_feat = batch['clinical_features'].unsqueeze(0).to(device)
                label = batch['label']
                center_id = batch.get('center_id', 0)
                clinical_data = batch.get('clinical_data', None)
            else:
                continue
            
            # 获取因果特征
            if oct_feat.size(-1) != colpo_feat.size(-1):
                min_dim = min(oct_feat.size(-1), colpo_feat.size(-1))
                image_feat = (oct_feat[..., :min_dim] + colpo_feat[..., :min_dim]) / 2
            else:
                image_feat = (oct_feat + colpo_feat) / 2
            
            z_causal, _ = model.image_encoder(image_feat)  # [1, 768]
            z_causal_np = z_causal.cpu().numpy()[0]  # [768]
            
            all_z_causal.append(z_causal_np)
            all_labels.append(label.item())
            sample_indices.append(idx)
            
            # 为每个中心生成合成特征
            for center_id_cf in range(num_centers):
                # 从Memory Bank获取该中心的噪声
                center_id_tensor = torch.tensor([center_id_cf], device=device)
                z_noise_cf = model.memory_bank.get_counterfactual_noise(center_id_tensor, strategy='random')
                z_noise_cf_np = z_noise_cf.cpu().numpy()[0]  # [768]
                
                # 合成特征
                z_mix = z_causal_np + z_noise_cf_np
                all_z_mix.append(z_mix)
                all_center_ids.append(center_id_cf)
    
    # 合并所有特征
    all_features = np.array(all_z_causal + all_z_mix)  # [N, 768]
    
    print(f"✅ 收集完成: {len(all_z_causal)} 个因果特征, {len(all_z_mix)} 个合成特征")
    
    # t-SNE降维
    print("🔄 进行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features)-1))
    features_2d = tsne.fit_transform(all_features)
    
    # 分离因果特征和合成特征
    n_causal = len(all_z_causal)
    causal_2d = features_2d[:n_causal]
    mix_2d = features_2d[n_causal:]
    
    # 绘制
    print("🔄 绘制图像...")
    plt.figure(figsize=(12, 10))
    
    # 设置颜色
    colors = plt.cm.Set3(np.linspace(0, 1, num_centers))
    
    # 绘制合成特征（按中心分组）
    for center_id in range(num_centers):
        center_mask = np.array(all_center_ids) == center_id
        if center_mask.sum() > 0:
            center_mix_2d = mix_2d[center_mask]
            plt.scatter(center_mix_2d[:, 0], center_mix_2d[:, 1], 
                       c=[colors[center_id]], s=50, alpha=0.6,
                       label=f'Center {center_id} (Synthetic)', marker='o')
    
    # 绘制因果特征（星号）
    for i, (causal_point, label) in enumerate(zip(causal_2d, all_labels)):
        color = 'red' if label == 1 else 'blue'
        marker = '*' if label == 1 else '^'
        plt.scatter(causal_point[0], causal_point[1], 
                   c=color, s=200, marker=marker, 
                   edgecolors='black', linewidths=2,
                   label='Causal Feature' if i == 0 else '')
    
    # 绘制连接线（从因果特征到合成特征）
    for i, causal_point in enumerate(causal_2d):
        start_idx = i * num_centers
        end_idx = start_idx + num_centers
        for mix_point in mix_2d[start_idx:end_idx]:
            plt.plot([causal_point[0], mix_point[0]], 
                    [causal_point[1], mix_point[1]], 
                    'k--', alpha=0.2, linewidth=0.5)
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('Counterfactual Trajectory Visualization\n(Causal Features + Center-Specific Noise)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图像已保存: {output_path}")
    
    plt.close()


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='反事实轨迹可视化')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--data_root', type=str, 
                       default='/data2/hmy/5Center_datas/5centers_multi_leave_centers_out',
                       help='数据根目录')
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='val', help='数据集')
    parser.add_argument('--device', type=str, default='cuda:1', help='设备')
    parser.add_argument('--num_samples', type=int, default=10, help='选择的样本数')
    parser.add_argument('--output_file', type=str, 
                       default='counterfactual_trajectory.pdf', help='输出文件')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 加载模型
    print(f"📂 加载模型: {args.model_path}")
    model = BioCOTModel(embed_dim=768, num_classes=2, num_centers=5, input_dim=512).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("✅ 模型加载完成")
    
    # 加载数据集
    print(f"📂 加载数据集: {args.split}")
    if args.split == 'val':
        test_dataset = EnhancedMultimodalCervicalDataset(
            root=Path(args.data_root) / 'internal_train' / 'val',
            labels_file=Path(args.data_root) / 'val_labels.csv',
            use_pretrained_backbones=True
        )
    else:
        test_dataset = EnhancedMultimodalCervicalDataset(
            root=Path(args.data_root) / 'external_test',
            labels_file=Path(args.data_root) / 'external_test_labels.csv',
            use_pretrained_backbones=True
        )
    print(f"✅ 数据集加载完成: {len(test_dataset)} 个样本")
    
    # 可视化
    visualize_counterfactual_trajectory(
        model, test_dataset, device,
        num_samples=args.num_samples,
        output_file=args.output_file
    )


if __name__ == '__main__':
    main()


