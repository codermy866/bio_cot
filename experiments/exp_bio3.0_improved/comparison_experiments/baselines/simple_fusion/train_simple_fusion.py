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
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import json
from datetime import datetime
import time

# 添加项目路径
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.dataset_v3 import FiveCentersMultimodalDatasetV3
from training.extract_vit_patches import extract_patch_features_with_vit
from torchvision import transforms
from utils.experiment_manager import ExperimentManager, ExperimentConfig, ExperimentResult


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


def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_single_run(config, device, seed, run_id, output_dir):
    """训练单次运行"""
    set_seed(seed)
    
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
        oct_num_frames=10,  # 减少帧数以节省显存
        max_col_images=3
    )
    
    val_dataset = FiveCentersMultimodalDatasetV3(
        csv_path=str(val_csv),
        transform=transform,
        oct_num_frames=10,  # 减少帧数以节省显存
        max_col_images=3
    )
    
    # 加权采样
    train_labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
    class_counts = pd.Series(train_labels).value_counts().sort_index()
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 创建模型
    print("正在创建模型...")
    sys.stdout.flush()
    model = SimpleFusionModel(embed_dim=768, num_classes=2).to(device)
    print(f"模型已移动到 {next(model.parameters()).device}")
    sys.stdout.flush()
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    print("优化器和损失函数已创建")
    sys.stdout.flush()
    
    # 训练循环
    best_auc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    print(f"\n开始训练，共 {config.num_epochs} 个epochs...")
    sys.stdout.flush()
    
    for epoch in range(1, config.num_epochs + 1):
        # 训练
        model.train()
        train_loss = 0.0
        print(f"\nEpoch {epoch}/{config.num_epochs} - 开始训练...")
        sys.stdout.flush()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}/{config.num_epochs} [Train]', leave=False)):
            if batch_idx == 0:
                print(f"  第一个batch - 数据形状: OCT={batch['oct_images'].shape}, Colpo={batch['colposcopy_images'].shape}")
                sys.stdout.flush()
            oct_images = batch['oct_images'].to(device)
            colposcopy_images = batch['colposcopy_images'].to(device)
            labels = batch['label'].to(device)
            
            # 提取特征
            B = len(oct_images)
            if len(oct_images.shape) == 5:  # [B, F, C, H, W]
                F = oct_images.shape[1]
                oct_images_flat = oct_images.view(B * F, *oct_images.shape[2:])
                f_oct_patch = extract_patch_features_with_vit(oct_images_flat, device)
                f_oct = f_oct_patch.view(B, F, -1, 768).mean(dim=(1, 2))  # [B, 768]
            else:
                f_oct_patch = extract_patch_features_with_vit(oct_images, device)
                f_oct = f_oct_patch.mean(dim=1)  # [B, 768]
            
            if len(colposcopy_images.shape) == 5:  # [B, N, C, H, W]
                N = colposcopy_images.shape[1]
                colpo_images_flat = colposcopy_images.view(B * N, *colposcopy_images.shape[2:])
                f_colpo_patch = extract_patch_features_with_vit(colpo_images_flat, device)
                f_colpo = f_colpo_patch.view(B, N, -1, 768).mean(dim=(1, 2))  # [B, 768]
            else:
                f_colpo_patch = extract_patch_features_with_vit(colposcopy_images, device)
                f_colpo = f_colpo_patch.mean(dim=1)  # [B, 768]
            
            # 临床数据 - 从clinical_features或clinical_data获取
            if 'clinical_features' in batch:
                # 使用clinical_features [B, 7] -> 提取hpv, tct, age
                clinical_features = batch['clinical_features'].to(device)  # [B, 7]
                # clinical_features格式: [age/100, hpv, tct_5dim]
                clinical_data = torch.stack([
                    clinical_features[:, 1],  # hpv
                    clinical_features[:, 2],  # tct第一个维度
                    clinical_features[:, 0] * 100  # age (反归一化)
                ], dim=1)  # [B, 3]
            elif 'clinical_data' in batch:
                # 使用clinical_data字典
                clinical_data = torch.stack([
                    torch.tensor([d['hpv'] for d in batch['clinical_data']], dtype=torch.float32),
                    torch.tensor([d.get('tct', 0.0) if isinstance(d.get('tct'), (int, float)) else 0.0 for d in batch['clinical_data']], dtype=torch.float32),
                    torch.tensor([d.get('age', 50.0) for d in batch['clinical_data']], dtype=torch.float32)
                ], dim=1).to(device)  # [B, 3]
            else:
                # 默认值
                clinical_data = torch.zeros(B, 3, device=device)
            
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
            for batch in tqdm(val_loader, desc=f'Epoch {epoch}/{config.num_epochs} [Val]', leave=False):
                oct_images = batch['oct_images'].to(device)
                colposcopy_images = batch['colposcopy_images'].to(device)
                labels = batch['label'].to(device)
                
                # 提取特征
                B = len(oct_images)
                if len(oct_images.shape) == 5:
                    F = oct_images.shape[1]
                    oct_images_flat = oct_images.view(B * F, *oct_images.shape[2:])
                    f_oct_patch = extract_patch_features_with_vit(oct_images_flat, device)
                    f_oct = f_oct_patch.view(B, F, -1, 768).mean(dim=(1, 2))
                else:
                    f_oct_patch = extract_patch_features_with_vit(oct_images, device)
                    f_oct = f_oct_patch.mean(dim=1)
                
                if len(colposcopy_images.shape) == 5:
                    N = colposcopy_images.shape[1]
                    colpo_images_flat = colposcopy_images.view(B * N, *colposcopy_images.shape[2:])
                    f_colpo_patch = extract_patch_features_with_vit(colpo_images_flat, device)
                    f_colpo = f_colpo_patch.view(B, N, -1, 768).mean(dim=(1, 2))
                else:
                    f_colpo_patch = extract_patch_features_with_vit(colposcopy_images, device)
                    f_colpo = f_colpo_patch.mean(dim=1)
                
                # 临床数据
                if 'clinical_features' in batch:
                    clinical_features = batch['clinical_features'].to(device)
                    clinical_data = torch.stack([
                        clinical_features[:, 1],
                        clinical_features[:, 2],
                        clinical_features[:, 0] * 100
                    ], dim=1)
                elif 'clinical_data' in batch:
                    clinical_data = torch.stack([
                        torch.tensor([d['hpv'] for d in batch['clinical_data']], dtype=torch.float32),
                        torch.tensor([d.get('tct', 0.0) if isinstance(d.get('tct'), (int, float)) else 0.0 for d in batch['clinical_data']], dtype=torch.float32),
                        torch.tensor([d.get('age', 50.0) for d in batch['clinical_data']], dtype=torch.float32)
                    ], dim=1).to(device)
                else:
                    clinical_data = torch.zeros(B, 3, device=device)
                
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
        
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # 计算特异性
        cm = confusion_matrix(all_labels, all_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0
        
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Val Loss={val_loss/len(val_loader):.4f}, Acc={acc:.4f}, AUC={auc:.4f}")
    
    training_time = time.time() - start_time
    
    # 最终评估
    return {
        'auc': best_auc,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'best_epoch': best_epoch,
        'train_loss': train_loss / len(train_loader),
        'val_loss': val_loss / len(val_loader),
        'training_time': training_time
    }


def main():
    parser = argparse.ArgumentParser(description='训练Simple Fusion Baseline')
    parser.add_argument('--experiment_name', type=str, default='baseline_simple_fusion',
                       help='实验名称')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='运行次数')
    parser.add_argument('--output_dir', type=str, default='comparison_experiments/results',
                       help='输出目录')
    parser.add_argument('--batch_size', type=int, default=48,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                       help='学习率')
    parser.add_argument('--data_root', type=str, 
                       default='/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal',
                       help='数据根目录')
    
    args = parser.parse_args()
    
    # 创建配置
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        method='simple_fusion',
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
    
    # 设备
    device = torch.device('cuda:1' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 'cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(device.index if hasattr(device, 'index') else 1)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(device.index if hasattr(device, 'index') else 1).total_memory / 1024**3:.2f} GB")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    import sys
    sys.stdout.flush()
    
    # 运行多次实验
    results = []
    seeds = [42, 123, 456, 789, 2024][:args.num_runs]
    
    print(f"\n{'='*80}")
    print(f"开始运行实验: {args.experiment_name}")
    print(f"运行次数: {args.num_runs}")
    print(f"随机种子: {seeds}")
    print(f"{'='*80}\n")
    
    for run_id, seed in enumerate(seeds, 1):
        print(f"\n运行 {run_id}/{args.num_runs} (seed={seed})...")
        result_dict = train_single_run(config, device, seed, run_id, args.output_dir)
        
        result = ExperimentResult(
            experiment_name=args.experiment_name,
            run_id=run_id,
            random_seed=seed,
            **result_dict,
            checkpoint_path=""
        )
        results.append(result)
        
        print(f"完成! AUC: {result.auc:.4f}, Acc: {result.accuracy:.4f}")
    
    # 保存结果
    output_path = Path(args.output_dir) / args.experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    results_dir = output_path / "results"
    results_dir.mkdir(exist_ok=True)
    logs_dir = output_path / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # 保存所有结果
    all_results = [r.to_dict() for r in results]
    with open(results_dir / "all_results.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 保存CSV
    df = pd.DataFrame(all_results)
    csv_path = results_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ 结果已保存:")
    print(f"  - JSON: {results_dir / 'all_results.json'}")
    print(f"  - CSV: {csv_path}")
    
    # 计算统计信息
    from utils.statistics import compute_statistics
    stats = {}
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score']
    for metric in metrics:
        values = [getattr(r, metric) for r in results]
        stats[metric] = compute_statistics(values)
    
    # 保存统计信息
    stats_clean = {}
    for key, value in stats.items():
        stats_clean[key] = {k: v for k, v in value.items() if k != 'values'}
    
    with open(results_dir / "statistics.json", 'w', encoding='utf-8') as f:
        json.dump(stats_clean, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    print(f"\n{'='*80}")
    print(f"实验统计信息: {args.experiment_name}")
    print(f"{'='*80}")
    print(f"{'指标':<15} {'均值':<10} {'标准差':<10} {'95% CI':<20}")
    print(f"{'-'*80}")
    
    for metric, stat in stats.items():
        mean = stat['mean']
        std = stat['std']
        ci = stat['ci_95']
        print(f"{metric:<15} {mean:.4f}     {std:.4f}     [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    print(f"{'='*80}\n")
    
    print(f"\n✅ 实验完成! 结果保存在: {output_path}")


if __name__ == '__main__':
    main()
