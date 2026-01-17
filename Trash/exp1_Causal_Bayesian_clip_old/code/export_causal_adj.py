#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出最佳模型的因果邻接可视化（热力图 + npy）
使用验证集前若干batch估计平均邻接
"""

import os
import json
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
import sys
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from training.train_enhanced_causal_clip import FeatureExtractor, prepare_data_loaders
from src.models.enhanced_causal_clip import EnhancedCausalBayesianCLIP


def export_causal_adj(checkpoint_path, data_path, output_dir, batches=3, device='cuda'):
    device = device if torch.cuda.is_available() else 'cpu'
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据（只需验证集）
    _, val_loader = prepare_data_loaders(data_path, batch_size=8, num_workers=2)

    # 创建模型与特征提取器
    model = EnhancedCausalBayesianCLIP(use_learnable_causal=True, use_uncertainty_decomposition=True)
    feature_extractor = FeatureExtractor(device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # 允许缺少/多余键（投影层可能动态创建）
    feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'], strict=False)
    model.to(device).eval()
    feature_extractor.to(device).eval()

    causal_adjs = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= batches:
                break
            # 只支持图像输入形式
            oct_images = batch['oct_images'].to(device)
            col_images = batch['col_images'].to(device)
            clinical_features = batch['clinical_features'].to(device)
            oct_feat, col_feat = feature_extractor(oct_images, col_images)
            output = model(oct_feat, col_feat, clinical_features, return_causal_penalty=False)
            if output.get('causal_adj') is not None:
                causal_adjs.append(output['causal_adj'].detach().cpu())

    if not causal_adjs:
        print("未得到因果邻接矩阵，退出")
        return

    causal_adj_mean = torch.cat(causal_adjs, dim=0).mean(dim=0).numpy()  # [3,3]

    # 保存 npy
    npy_path = os.path.join(output_dir, 'causal_adj_mean.npy')
    np.save(npy_path, causal_adj_mean)

    # 绘制热力图
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Helvetica']
    plt.rcParams['font.family'] = 'sans-serif'
    modalities = ['OCT', 'Colposcopy', 'Clinical']
    plt.figure(figsize=(4, 3))
    sns.heatmap(causal_adj_mean, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=modalities, yticklabels=modalities)
    plt.title("Mean Causal Adjacency (best model)")
    heatmap_path = os.path.join(output_dir, 'causal_adj_heatmap.png')
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    print(f"Saved causal adjacency to:\n  npy: {npy_path}\n  png: {heatmap_path}")


def main():
    parser = argparse.ArgumentParser(description="Export causal adjacency heatmap from best model checkpoint")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data root (5centers_multi)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--batches', type=int, default=3, help='Number of val batches to average')
    args = parser.parse_args()

    export_causal_adj(args.checkpoint, args.data_path, args.output_dir, batches=args.batches)


if __name__ == '__main__':
    main()

