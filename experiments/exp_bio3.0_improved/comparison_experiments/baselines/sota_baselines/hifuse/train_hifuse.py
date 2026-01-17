#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SOTA Baseline: HiFuse
层次多尺度特征融合网络 - 结合Transformer和CNN的优势
基于论文: "HiFuse: Hierarchical Multi-Scale Feature Fusion Network for Medical Image Classification"
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from tqdm import tqdm

# 添加项目路径
ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.dataset_v3 import FiveCentersMultimodalDatasetV3
from torchvision import transforms
from utils.experiment_manager import ExperimentConfig
from baselines.sota_baselines.common.trainer_base import SOTABaselineTrainer


class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器"""
    def __init__(self, embed_dim=768):
        super().__init__()
        try:
            import timm
            # 使用ResNet作为backbone（多尺度特征）
            self.backbone = timm.create_model(
                'resnet50',
                pretrained=True,
                features_only=True,
                out_indices=(1, 2, 3, 4)  # 多尺度特征
            )
            # 特征融合
            self.fusion = nn.Sequential(
                nn.Conv2d(2048 + 1024 + 512 + 256, embed_dim, 1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        except:
            # 简化版本
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            self.fusion = nn.Linear(64, embed_dim)
    
    def forward(self, images):
        if len(images.shape) == 5:  # [B, F, C, H, W]
            B, F, C, H, W = images.shape
            images = images.view(B * F, C, H, W)
            
            if hasattr(self.backbone, 'forward_features'):
                features = self.backbone.forward_features(images)
                if isinstance(features, (list, tuple)):
                    # 多尺度特征融合
                    features = torch.cat([F.adaptive_avg_pool2d(f, 1) for f in features], dim=1)
                    features = features.squeeze(-1).squeeze(-1)
                else:
                    features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
            else:
                features = self.backbone(images)
            
            features = features.view(B, F, -1).mean(dim=1)
        else:
            if hasattr(self.backbone, 'forward_features'):
                features = self.backbone.forward_features(images)
                if isinstance(features, (list, tuple)):
                    features = torch.cat([F.adaptive_avg_pool2d(f, 1) for f in features], dim=1)
                    features = features.squeeze(-1).squeeze(-1)
                else:
                    features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
            else:
                features = self.backbone(images)
        
        return self.fusion(features) if features.dim() > 2 else features


class HierarchicalFusion(nn.Module):
    """层次融合模块"""
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        # 自注意力层
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 跨模态注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
    
    def forward(self, modality_features):
        """
        Args:
            modality_features: List of [B, embed_dim] tensors
        Returns:
            fused: [B, embed_dim]
        """
        # Stack: [B, num_modalities, embed_dim]
        stacked = torch.stack(modality_features, dim=1)
        
        # 自注意力
        attn_out, _ = self.self_attn(stacked, stacked, stacked)
        stacked = self.norm1(stacked + attn_out)
        
        # 跨模态注意力（使用第一个模态作为query）
        cross_out, _ = self.cross_attn(
            stacked[:, 0:1],  # query
            stacked,  # key
            stacked  # value
        )
        stacked = self.norm2(stacked + cross_out)
        
        # FFN
        ffn_out = self.ffn(stacked)
        stacked = self.norm3(stacked + ffn_out)
        
        # 全局平均池化
        fused = stacked.mean(dim=1)  # [B, embed_dim]
        
        return fused


class HiFuseModel(nn.Module):
    """
    HiFuse模型 - 层次多尺度特征融合
    """
    
    def __init__(self, embed_dim=768, num_classes=2, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 多尺度特征提取器
        self.oct_extractor = MultiScaleFeatureExtractor(embed_dim)
        self.col_extractor = MultiScaleFeatureExtractor(embed_dim)
        self.clinical_encoder = nn.Sequential(
            nn.Linear(7, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        
        # 层次融合
        self.hierarchical_fusion = HierarchicalFusion(embed_dim, num_heads)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def forward(self, oct_images, col_images, clinical_features):
        """前向传播"""
        # 多尺度特征提取
        oct_feat = self.oct_extractor(oct_images)
        col_feat = self.col_extractor(col_images)
        clinical_feat = self.clinical_encoder(clinical_features)
        
        # 层次融合
        modality_features = [oct_feat, col_feat, clinical_feat]
        fused = self.hierarchical_fusion(modality_features)
        
        # 分类
        logits = self.classifier(fused)
        
        return logits


class HiFuseTrainer(SOTABaselineTrainer):
    """HiFuse训练器"""
    
    def create_model(self):
        """创建HiFuse模型"""
        return HiFuseModel(
            embed_dim=768,
            num_classes=2,
            num_heads=8
        )
    
    def prepare_data(self, data_root: str):
        """准备数据"""
        train_csv = Path(data_root) / 'internal_train' / 'labels.csv'
        val_csv = Path(data_root) / 'internal_val' / 'labels.csv'
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = FiveCentersMultimodalDatasetV3(
            csv_path=str(train_csv),
            transform=transform,
            oct_num_frames=20,
            max_col_images=3
        )
        
        val_dataset = FiveCentersMultimodalDatasetV3(
            csv_path=str(val_csv),
            transform=transform,
            oct_num_frames=20,
            max_col_images=3
        )
        
        # 加权采样
        train_labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
        class_counts = pd.Series(train_labels).value_counts().sort_index()
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def forward_pass(self, batch):
        """前向传播"""
        oct_images = batch['oct_images'].to(self.device)
        colposcopy_images = batch['colposcopy_images'].to(self.device)
        labels = batch['label'].to(self.device)
        
        # 处理图像形状
        if len(oct_images.shape) == 4:
            oct_images = oct_images.unsqueeze(1)
        if len(colposcopy_images.shape) == 4:
            colposcopy_images = colposcopy_images.unsqueeze(1)
        
        # 临床特征
        if 'clinical_features' in batch:
            clinical_features = batch['clinical_features'].to(self.device)
        else:
            clinical_features = torch.zeros(len(oct_images), 7, device=self.device)
        
        # 前向传播
        logits = self.model(oct_images, colposcopy_images, clinical_features)
        loss = self.criterion(logits, labels)
        
        return logits, loss


def main():
    parser = argparse.ArgumentParser(description='训练HiFuse Baseline')
    parser.add_argument('--experiment_name', type=str, default='baseline_hifuse',
                       help='实验名称')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='运行次数')
    parser.add_argument('--output_dir', type=str, default='comparison_experiments/results',
                       help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                       help='学习率')
    parser.add_argument('--data_root', type=str, 
                       default='/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal',
                       help='数据根目录')
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        method='hifuse',
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=1e-5,
        use_knowledge_notes=False,
        use_visual_notes=False,
        use_ot=False,
        use_dual=False,
        output_dir=args.output_dir,
        data_root=args.data_root
    )
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    trainer = HiFuseTrainer(config, device)
    trainer.run_experiment(num_runs=args.num_runs)
    
    print(f"\n✅ HiFuse实验完成!")


if __name__ == '__main__':
    main()

