#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SOTA Baseline: MedCLIP
医学领域专用的CLIP模型 - 基于标准CLIP实现医学版本
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
# train_medclip.py -> medclip -> sota_baselines -> baselines -> comparison_experiments -> exp_bio3.0_improved
ROOT = Path(__file__).resolve().parents[4]  # 到exp_bio3.0_improved
sys.path.insert(0, str(ROOT))

from data.dataset_v3 import FiveCentersMultimodalDatasetV3
from torchvision import transforms
from utils.experiment_manager import ExperimentConfig
from comparison_experiments.baselines.sota_baselines.common.trainer_base import SOTABaselineTrainer


class MedCLIPModel(nn.Module):
    """
    MedCLIP模型 - 医学领域CLIP
    基于标准CLIP架构，但使用医学相关的文本提示
    """
    
    def __init__(self, embed_dim=768, num_classes=2, temperature=0.07):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        
        # 图像编码器（使用更小的ResNet18以节省显存）
        try:
            import torchvision.models as models
            # 使用ResNet18代替ViT base，显存占用更小
            resnet = models.resnet18(pretrained=True)
            # 移除最后的全连接层
            self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
            # ResNet18输出512维，投影到embed_dim
            self.image_proj = nn.Linear(512, embed_dim)
        except:
            # 如果不可用，使用简单的CNN
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, embed_dim)
            )
            self.image_proj = nn.Identity()
        
        # 文本编码器（简单的MLP，模拟医学文本特征）
        # 在实际MedCLIP中，应该使用BERT等文本编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(7, 256),  # 临床特征作为"文本"
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_classes)
        )
    
    def encode_image(self, images):
        """编码图像（优化显存使用）"""
        # 处理多帧OCT图像 - 使用更小的batch处理
        if len(images.shape) == 5:  # [B, num_frames, C, H, W]
            B, num_frames, C, H, W = images.shape  # 修复：使用num_frames避免覆盖F
            # 先平均帧，减少显存占用
            images = images.mean(dim=1)  # [B, C, H, W] - 帧间平均
            features = self.image_encoder(images)
            if features.dim() > 2:
                features = features.view(features.size(0), -1)  # [B, 512]
        else:
            features = self.image_encoder(images)
            if features.dim() > 2:
                features = features.view(features.size(0), -1)  # [B, 512]
        
        features = self.image_proj(features)
        return F.normalize(features, dim=-1)  # 使用全局导入的F
    
    def encode_text(self, clinical_features):
        """编码文本（使用临床特征作为文本表示）"""
        features = self.text_encoder(clinical_features)
        return F.normalize(features, dim=-1)
    
    def forward(self, oct_images, col_images, clinical_features):
        """前向传播"""
        # 编码OCT图像
        oct_feat = self.encode_image(oct_images)
        
        # 编码Colposcopy图像
        col_feat = self.encode_image(col_images)
        
        # 编码临床特征（作为"文本"）
        text_feat = self.encode_text(clinical_features)
        
        # CLIP风格的对比学习
        # 图像特征：OCT和Colposcopy的平均
        img_feat = (oct_feat + col_feat) / 2
        
        # 计算图像-文本相似度
        similarity = (img_feat * text_feat).sum(dim=-1, keepdim=True) / self.temperature
        
        # 融合特征用于分类
        fused_feat = torch.cat([img_feat, text_feat], dim=-1)
        logits = self.classifier(fused_feat)
        
        return logits


class MedCLIPTrainer(SOTABaselineTrainer):
    """MedCLIP训练器"""
    
    def create_model(self):
        """创建MedCLIP模型"""
        return MedCLIPModel(
            embed_dim=768,
            num_classes=2,
            temperature=0.07
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
    parser = argparse.ArgumentParser(description='训练MedCLIP Baseline')
    parser.add_argument('--experiment_name', type=str, default='baseline_medclip',
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
        method='medclip',
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
    
    trainer = MedCLIPTrainer(config, device)
    trainer.run_experiment(num_runs=args.num_runs)
    
    print(f"\n✅ MedCLIP实验完成!")


if __name__ == '__main__':
    main()

