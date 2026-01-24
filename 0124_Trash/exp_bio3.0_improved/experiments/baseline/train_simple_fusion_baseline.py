#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline 1: Simple Fusion Baseline
最简单的多模态融合方法 - 特征拼接 + MLP分类器
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# 添加项目路径
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.experiment_manager import ExperimentManager, ExperimentConfig, ExperimentResult
from data.dataset_v3 import FiveCentersMultimodalDatasetV3
from training.extract_vit_patches import extract_patch_features_with_vit
from torchvision import transforms


class SimpleFusionModel(nn.Module):
    """Simple Fusion Baseline模型"""
    
    def __init__(self, embed_dim=768, num_classes=2):
        super().__init__()
        
        # 临床特征编码器
        self.clinical_encoder = nn.Sequential(
            nn.Linear(3, 256),  # HPV, TCT, Age
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim)
        )
        
        # 特征融合（简单拼接）
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),  # OCT + Colpo + Clinical
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def forward(self, f_oct, f_colpo, clinical_data):
        """
        Args:
            f_oct: [B, 768] OCT特征
            f_colpo: [B, 768] Colposcopy特征
            clinical_data: [B, 3] 临床数据 (HPV, TCT, Age)
        """
        # 编码临床特征
        f_clinical = self.clinical_encoder(clinical_data)  # [B, 768]
        
        # 简单拼接
        f_concat = torch.cat([f_oct, f_colpo, f_clinical], dim=-1)  # [B, 2304]
        
        # 融合
        f_fused = self.fusion(f_concat)  # [B, 768]
        
        # 分类
        logits = self.classifier(f_fused)  # [B, 2]
        
        return logits


def train_simple_fusion(config: ExperimentConfig, device, num_runs=5):
    """训练Simple Fusion Baseline"""
    
    # 数据加载
    data_root = config.data_root
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
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    # 使用ExperimentManager运行多次实验
    manager = ExperimentManager(config.experiment_name, config, output_dir=config.output_dir)
    
    # 重写_run_single方法以适配Simple Fusion
    def run_single_simple_fusion(run_id, seed, verbose=True):
        manager.set_seed(seed)
        
        # 创建模型
        model = SimpleFusionModel(embed_dim=768, num_classes=2).to(device)
        
        # 优化器和损失函数
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        best_auc = 0.0
        best_epoch = 0
        
        for epoch in range(1, config.num_epochs + 1):
            # 训练
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                oct_images = batch['oct_images'].to(device)
                colposcopy_images = batch['colposcopy_images'].to(device)
                labels = batch['label'].to(device)
                
                # 提取特征
                f_oct = extract_patch_features_with_vit(oct_images.view(-1, *oct_images.shape[2:]), device)
                f_oct = f_oct.view(len(oct_images), -1, 768).mean(dim=1)  # [B, 768]
                
                f_colpo = extract_patch_features_with_vit(colposcopy_images.view(-1, *colposcopy_images.shape[2:]), device)
                f_colpo = f_colpo.view(len(colposcopy_images), -1, 768).mean(dim=1)  # [B, 768]
                
                # 临床数据
                clinical_data = torch.stack([
                    batch['hpv'].float(),
                    batch['tct'].float(),
                    batch['age'].float()
                ], dim=1).to(device)  # [B, 3]
                
                # 前向传播
                optimizer.zero_grad()
                logits = model(f_oct, f_colpo, clinical_data)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证
            model.eval()
            all_preds = []
            all_labels = []
            all_probs = []
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    oct_images = batch['oct_images'].to(device)
                    colposcopy_images = batch['colposcopy_images'].to(device)
                    labels = batch['label'].to(device)
                    
                    # 提取特征
                    f_oct = extract_patch_features_with_vit(oct_images.view(-1, *oct_images.shape[2:]), device)
                    f_oct = f_oct.view(len(oct_images), -1, 768).mean(dim=1)
                    
                    f_colpo = extract_patch_features_with_vit(colposcopy_images.view(-1, *colposcopy_images.shape[2:]), device)
                    f_colpo = f_colpo.view(len(colposcopy_images), -1, 768).mean(dim=1)
                    
                    clinical_data = torch.stack([
                        batch['hpv'].float(),
                        batch['tct'].float(),
                        batch['age'].float()
                    ], dim=1).to(device)
                    
                    logits = model(f_oct, f_colpo, clinical_data)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    
                    probs = torch.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())
            
            # 计算指标
            acc = accuracy_score(all_labels, all_preds)
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except:
                auc = 0.0
            
            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, "
                      f"Val Loss={val_loss/len(val_loader):.4f}, Acc={acc:.4f}, AUC={auc:.4f}")
        
        # 最终评估
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # 计算特异性
        tn = np.sum((np.array(all_labels) == 0) & (np.array(all_preds) == 0))
        fp = np.sum((np.array(all_labels) == 0) & (np.array(all_preds) == 1))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return {
            'auc': best_auc,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'best_epoch': best_epoch,
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader)
        }
    
    # 运行多次实验
    results = []
    seeds = [42, 123, 456, 789, 2024][:num_runs]
    
    for run_id, seed in enumerate(seeds, 1):
        print(f"\n运行 {run_id}/{num_runs} (seed={seed})...")
        result_dict = run_single_simple_fusion(run_id, seed, verbose=True)
        
        result = ExperimentResult(
            experiment_name=config.experiment_name,
            run_id=run_id,
            random_seed=seed,
            **result_dict,
            training_time=0.0  # 可以添加计时
        )
        results.append(result)
        
        print(f"完成! AUC: {result.auc:.4f}, Acc: {result.accuracy:.4f}")
    
    # 保存结果
    manager.results = results
    manager._save_all_results()
    
    # 计算统计信息
    stats = manager._compute_statistics()
    manager._save_statistics(stats)
    manager._print_statistics(stats)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='训练Simple Fusion Baseline')
    parser.add_argument('--experiment_name', type=str, default='baseline_simple_fusion',
                       help='实验名称')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='运行次数')
    parser.add_argument('--output_dir', type=str, default='experiments/results',
                       help='输出目录')
    parser.add_argument('--batch_size', type=int, default=48,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                       help='学习率')
    
    args = parser.parse_args()
    
    # 创建配置
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        method='simple_fusion',
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        use_knowledge_notes=False,
        use_visual_notes=False,
        use_ot=False,
        use_dual=False,
        output_dir=args.output_dir
    )
    
    # 设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 运行实验
    results = train_simple_fusion(config, device, num_runs=args.num_runs)
    
    print(f"\n✅ 实验完成! 结果保存在: {args.output_dir}/{args.experiment_name}")


if __name__ == '__main__':
    main()

