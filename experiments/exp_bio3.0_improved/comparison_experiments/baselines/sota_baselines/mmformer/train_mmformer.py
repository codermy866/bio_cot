#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SOTA Baseline: mmFormer (简化实现)
多模态医学Transformer - 基于mmFormer思想实现
注意：这是基于mmFormer论文思想的简化实现，不是完整克隆
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
ROOT = Path(__file__).resolve().parents[4]  # 到exp_bio3.0_improved
sys.path.insert(0, str(ROOT))

from data.dataset_v3 import FiveCentersMultimodalDatasetV3
from torchvision import transforms
from utils.experiment_manager import ExperimentConfig
from comparison_experiments.baselines.sota_baselines.common.trainer_base import SOTABaselineTrainer


class MultiModalTransformerBlock(nn.Module):
    """多模态Transformer块"""
    
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 自注意力
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 跨模态注意力
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, modality_mask=None):
        """前向传播"""
        # 自注意力
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # 跨模态注意力（如果有多个模态）
        if modality_mask is not None:
            cross_out, _ = self.cross_attn(x, x, x, key_padding_mask=modality_mask)
            x = self.norm2(x + cross_out)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        return x


class MMFormerModel(nn.Module):
    """
    mmFormer模型 - 多模态医学Transformer
    基于mmFormer思想：处理不完整多模态数据
    """
    
    def __init__(self, embed_dim=768, num_classes=2, num_layers=4):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 图像编码器（使用ResNet18以节省显存）
        try:
            import torchvision.models as models
            resnet = models.resnet18(pretrained=True)
            self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
            self.image_proj = nn.Linear(512, embed_dim)
        except:
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, embed_dim)
            )
            self.image_proj = nn.Identity()
        
        # 临床特征编码器
        self.clinical_encoder = nn.Sequential(
            nn.Linear(7, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        
        # 模态嵌入（区分不同模态）
        self.modal_embeddings = nn.Parameter(torch.randn(3, embed_dim))  # OCT, Colpo, Clinical
        
        # 多模态Transformer层
        self.transformer_layers = nn.ModuleList([
            MultiModalTransformerBlock(embed_dim, num_heads=8, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def encode_image(self, images):
        """编码图像（优化显存使用）"""
        if len(images.shape) == 5:  # [B, F, C, H, W]
            # 先平均帧，减少显存占用
            images = images.mean(dim=1)  # [B, C, H, W]
            features = self.image_encoder(images)
            if features.dim() > 2:
                features = features.view(features.size(0), -1)  # [B, 512]
        else:
            features = self.image_encoder(images)
            if features.dim() > 2:
                features = features.view(features.size(0), -1)  # [B, 512]
        
        features = self.image_proj(features)
        return features
    
    def forward(self, oct_images, col_images, clinical_features):
        """前向传播"""
        B = len(oct_images)
        
        # 编码各模态
        oct_feat = self.encode_image(oct_images)  # [B, embed_dim]
        col_feat = self.encode_image(col_images)  # [B, embed_dim]
        clinical_feat = self.clinical_encoder(clinical_features)  # [B, embed_dim]
        
        # 添加模态嵌入
        oct_feat = oct_feat + self.modal_embeddings[0:1]
        col_feat = col_feat + self.modal_embeddings[1:2]
        clinical_feat = clinical_feat + self.modal_embeddings[2:3]
        
        # 堆叠为序列 [B, 3, embed_dim]
        multi_modal_tokens = torch.stack([oct_feat, col_feat, clinical_feat], dim=1)
        
        # 通过Transformer层
        x = multi_modal_tokens
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 全局平均池化
        x = x.mean(dim=1)  # [B, embed_dim]
        
        # 分类
        logits = self.classifier(x)
        
        return logits


class MMFormerTrainer(SOTABaselineTrainer):
    """mmFormer训练器"""
    
    def create_model(self):
        """创建mmFormer模型"""
        return MMFormerModel(
            embed_dim=768,
            num_classes=2,
            num_layers=4
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
        
        # 查找Knowledge Note Embeddings路径
        knowledge_embed_path = None
        possible_paths = [
            ROOT / 'data' / 'knowledge_embeddings.pt',
            ROOT.parent / 'exp_bio3.0' / 'data' / 'knowledge_embeddings.pt',
            ROOT.parent / 'exp_bio3.0_improved' / 'data' / 'knowledge_embeddings.pt',
        ]
        for path in possible_paths:
            if path.exists():
                knowledge_embed_path = str(path)
                print(f"✅ 找到Knowledge Note Embeddings: {knowledge_embed_path}")
                break
        
        if knowledge_embed_path is None:
            print(f"⚠️  未找到Knowledge Note Embeddings，尝试的路径:")
            for path in possible_paths:
                print(f"   - {path}")
            print(f"⚠️  将使用零向量")
        
        # 减少帧数以节省显存
        train_dataset = FiveCentersMultimodalDatasetV3(
            csv_path=str(train_csv),
            transform=transform,
            oct_num_frames=10,  # 从20减少到10
            max_col_images=3,
            knowledge_embed_path=knowledge_embed_path  # 添加Knowledge Note Embeddings路径
        )
        
        val_dataset = FiveCentersMultimodalDatasetV3(
            csv_path=str(val_csv),
            transform=transform,
            oct_num_frames=10,  # 从20减少到10
            max_col_images=3,
            knowledge_embed_path=knowledge_embed_path  # 添加Knowledge Note Embeddings路径
        )
        
        # 加权采样 - 直接从CSV读取标签，避免遍历数据集（更快）
        print("正在准备加权采样器...")
        train_df = pd.read_csv(train_csv)
        if 'label' in train_df.columns:
            train_labels = train_df['label'].values
        else:
            # 如果没有label列，从数据集读取（较慢）
            print("⚠️  CSV中没有label列，从数据集读取标签（可能较慢）...")
            train_labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
        
        class_counts = pd.Series(train_labels).value_counts().sort_index()
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        print(f"✅ 加权采样器准备完成 (类别分布: {dict(class_counts)})")
        
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
    parser = argparse.ArgumentParser(description='训练mmFormer Baseline')
    parser.add_argument('--experiment_name', type=str, default='baseline_mmformer',
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
        method='mmformer',
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
    
    device = torch.device('cuda:1' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 'cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(device.index if hasattr(device, 'index') else 1)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(device.index if hasattr(device, 'index') else 1).total_memory / 1024**3:.2f} GB")
    
    trainer = MMFormerTrainer(config, device)
    trainer.run_experiment(num_runs=args.num_runs)
    
    print(f"\n✅ mmFormer实验完成!")


if __name__ == '__main__':
    main()
