#!/usr/bin/env python3
"""
Training script for the hierarchical multi-granularity multimodal model (Paper1).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Add project root to PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from paper1_hierarchical_multimodal.models import HierarchicalMultimodalModel
from paper1_hierarchical_multimodal.models.hierarchical_multimodal_model_vit import HierarchicalMultimodalModelViT
from paper1_hierarchical_multimodal.models.losses import (
    CombinedLoss, 
    FocalLoss, 
    ClassWeightedCrossEntropy
)
from src.data.enhanced_multimodal_dataset import build_enhanced_dataset


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_dataset_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        data_path=args.data_path,
        input_size=args.input_size,
        oct_num_frames=args.oct_num_frames,
        oct_points=args.oct_points,
        oct_frames_per_point=args.oct_frames_per_point,
        col_num_frames=args.col_num_frames,
        oct_cache_dir=args.oct_cache_dir,
        use_pretrained_backbones=True,
        use_text_contrastive=False,
        num_classes=args.num_classes
    )


def collate_fn(batch):
    if isinstance(batch[0], dict):
        def to_tensor(x):
            return x if torch.is_tensor(x) else torch.tensor(x)
        oct_images = torch.stack([to_tensor(sample['oct_images']) for sample in batch]).float()
        col_images = torch.stack([to_tensor(sample['col_images']) for sample in batch]).float()
        clinical = torch.stack([to_tensor(sample['clinical_features']) for sample in batch]).float()
        labels = torch.stack([to_tensor(sample['label']) for sample in batch]).long()
    else:
        oct_images = torch.stack([sample[0] for sample in batch]).float()
        col_images = torch.stack([sample[1] for sample in batch]).float()
        clinical = torch.stack([sample[2] for sample in batch]).float()
        labels = torch.stack([sample[3] for sample in batch]).long()
    return {
        'oct_images': oct_images,
        'col_images': col_images,
        'clinical_features': clinical,
        'labels': labels
    }


def prepare_dataloaders(args):
    dataset_args = build_dataset_args(args)
    train_dataset = build_enhanced_dataset('train', dataset_args)
    val_dataset = build_enhanced_dataset('test', dataset_args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs,
                   log_interval=20, max_grad_norm=None, scaler=None, use_amp=False,
                   grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []
    num_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(loader, start=1):
        oct_images = batch['oct_images'].to(device)
        col_images = batch['col_images'].to(device)
        clinical = batch['clinical_features'].to(device)
        labels = batch['labels'].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(oct_images, col_images, clinical, labels)
            logits = outputs['logits']
            contrastive_loss = outputs['contrastive_loss'] if outputs['contrastive_loss'] is not None else 0.0
            loss = criterion(logits, labels) + contrastive_loss

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if max_grad_norm is not None:
            if scaler is not None and use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        if batch_idx % grad_accum_steps == 0 or batch_idx == num_batches:
            if scaler is not None and use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * labels.size(0)
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        y_prob.extend(probs.tolist())
        y_pred.extend(preds.tolist())
        y_true.extend(labels.detach().cpu().numpy().tolist())

        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == num_batches):
            percent = 100.0 * batch_idx / num_batches
            contr_val = float(contrastive_loss) if isinstance(contrastive_loss, (float, int)) else 0.0
            print(f"[Epoch {epoch}/{total_epochs}] Batch {batch_idx}/{num_batches} ({percent:.1f}%) "
                  f"Loss: {loss.item():.4f} | Contrastive: {contr_val:.6f}",
                  flush=True)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics['loss'] = total_loss / len(loader.dataset)
    return metrics


def evaluate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for batch in loader:
            oct_images = batch['oct_images'].to(device)
            col_images = batch['col_images'].to(device)
            clinical = batch['clinical_features'].to(device)
            labels = batch['labels'].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(oct_images, col_images, clinical, labels)
                logits = outputs['logits']
                contrastive_loss = outputs['contrastive_loss'] if outputs['contrastive_loss'] is not None else 0.0
                loss = criterion(logits, labels) + contrastive_loss

            total_loss += loss.item() * labels.size(0)
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            y_prob.extend(probs.tolist())
            y_pred.extend(preds.tolist())
            y_true.extend(labels.detach().cpu().numpy().tolist())

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics['loss'] = total_loss / len(loader.dataset)
    return metrics


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)
    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['auc'] = float('nan')
    return metrics


def save_metrics(metrics, output_dir):
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

def save_training_history(history, output_dir):
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)


def get_warmup_cosine_scheduler(optimizer, num_warmup_epochs, num_training_epochs):
    """创建带预热的余弦退火学习率调度器"""
    def lr_lambda_warmup(epoch):
        if epoch < num_warmup_epochs:
            return float(epoch) / float(max(1, num_warmup_epochs))
        else:
            progress = float(epoch - num_warmup_epochs) / float(max(1, num_training_epochs - num_warmup_epochs))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda=lr_lambda_warmup)


def main():
    parser = argparse.ArgumentParser(description='Train Hierarchical Multimodal Model (Paper1) - Optimized')
    parser.add_argument('--data_path', type=str, default='5centers_multi')
    parser.add_argument('--output_dir', type=str, default='paper1_hierarchical_multimodal/results')
    parser.add_argument('--batch_size', type=int, default=8, help='Increased batch size for better contrastive learning')
    parser.add_argument('--num_epochs', type=int, default=30, help='More epochs for better convergence')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Learning rate warmup epochs')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--oct_num_frames', type=int, default=32, help='Reduced for memory efficiency')
    parser.add_argument('--oct_points', type=int, default=12)
    parser.add_argument('--oct_frames_per_point', type=int, default=5)
    parser.add_argument('--col_num_frames', type=int, default=3)
    parser.add_argument('--oct_cache_dir', type=str, default='oct_cache_optimized')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--contrastive_weight', type=float, default=0.3, help='Increased contrastive weight')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=10, help='Batches between logging progress')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Gradient clipping norm (reduced for stability)')
    parser.add_argument('--use_amp', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--use_focal_loss', action='store_true', default=True, help='Use Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=1.0, help='Focal loss alpha (can be list for per-class weights)')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--use_class_weights', action='store_true', default=True, help='Use class weights for imbalanced data')
    parser.add_argument('--class_weights', type=str, default=None, help='Class weights as comma-separated values (e.g., "1.0,2.08")')
    parser.add_argument('--use_weighted_ce', action='store_true', help='Use weighted CrossEntropy instead of Focal Loss')
    parser.add_argument('--use_vit_backbone', action='store_true', help='Use Vision Transformer as backbone')
    parser.add_argument('--backbone_type', type=str, default='vit', choices=['vit', 'swin'], help='Backbone type: vit or swin')
    parser.add_argument('--vit_model_name', type=str, default='vit_base_patch16_224', help='ViT model name from timm')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = prepare_dataloaders(args)

    # 计算类别权重（处理类别不平衡）
    class_weights = None
    if args.use_class_weights:
        if args.class_weights:
            # 从命令行参数解析
            class_weights = [float(w) for w in args.class_weights.split(',')]
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            print(f"Using provided class weights: {class_weights.tolist()}")
        else:
            # 自动计算类别权重（逆频率）
            import pandas as pd
            train_df = pd.read_csv(os.path.join(args.data_path, 'train_labels.csv'))
            label_counts = train_df['label'].value_counts().sort_index()
            total = len(train_df)
            num_classes = len(label_counts)
            class_weights = torch.tensor([
                total / (num_classes * count) for count in label_counts
            ], dtype=torch.float32).to(device)
            print(f"Auto-computed class weights (inverse frequency): {class_weights.tolist()}")
            print(f"  Class 0: {label_counts[0]} samples -> weight {class_weights[0]:.4f}")
            print(f"  Class 1: {label_counts[1]} samples -> weight {class_weights[1]:.4f}")

    # 根据参数选择模型
    if args.use_vit_backbone:
        print(f"Using {args.backbone_type.upper()} Transformer as backbone: {args.vit_model_name}")
        model = HierarchicalMultimodalModelViT(
            embed_dim=768,
            num_classes=args.num_classes,
            clinical_dim=7,
            contrastive_weight=args.contrastive_weight,
            backbone_type=args.backbone_type,
            model_name=args.vit_model_name,
            use_pretrained=True,
            input_size=args.input_size
        ).to(device)
    else:
        print("Using ResNet as backbone")
        model = HierarchicalMultimodalModel(
            embed_dim=768,
            num_classes=args.num_classes,
            clinical_dim=7,
            contrastive_weight=args.contrastive_weight
        ).to(device)

    # 使用优化的损失函数（处理类别不平衡）
    if args.use_weighted_ce:
        # 使用类别加权的交叉熵（简单有效）
        print("Using Class-Weighted CrossEntropy Loss")
        criterion = ClassWeightedCrossEntropy(
            class_weights=class_weights,
            label_smoothing=args.label_smoothing
        )
    elif args.use_focal_loss:
        # 使用Focal Loss（更关注难样本）
        # 对于类别不平衡，调整alpha：少数类权重更高
        if isinstance(args.focal_alpha, float) and args.focal_alpha == 1.0:
            # 自动计算alpha：少数类权重 = 1 / (1 + 类别比例)
            # 类别0:530, 类别1:255 -> ratio=2.08
            # alpha for class 1 = 1 / (1 + 2.08) ≈ 0.32
            focal_alpha = [1.0, 0.32]  # [majority, minority]
            print(f"Auto-adjusted Focal Loss alpha for class imbalance: {focal_alpha}")
        else:
            focal_alpha = args.focal_alpha
        
        # 提高gamma以更关注难样本（医学图像通常需要gamma=3-5）
        focal_gamma = max(args.focal_gamma, 4.0) if args.focal_gamma < 4.0 else args.focal_gamma
        print(f"Using Focal Loss with gamma={focal_gamma}, alpha={focal_alpha}")
        
        criterion = CombinedLoss(
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            label_smoothing=args.label_smoothing,
            focal_weight=0.7,  # 增加Focal Loss权重
            smoothing_weight=0.3,
            class_weights=class_weights  # 同时使用类别权重
        )
    else:
        # 标准交叉熵（带类别权重）
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
            print(f"Using CrossEntropy with class weights: {class_weights.tolist()}")
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
    # 使用带预热的余弦退火调度器
    scheduler = get_warmup_cosine_scheduler(optimizer, args.warmup_epochs, args.num_epochs)
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == 'cuda')

    best_auc = 0.0
    history = {'train': [], 'val': []}

    for epoch in range(1, args.num_epochs + 1):
        start_time = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            args.num_epochs,
            log_interval=args.log_interval,
            max_grad_norm=args.max_grad_norm,
            scaler=scaler,
            use_amp=args.use_amp,
            grad_accum_steps=args.grad_accum_steps
        )
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp=args.use_amp)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']

        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{args.num_epochs} - Train ACC: {train_metrics['accuracy']:.4f}, Val ACC: {val_metrics['accuracy']:.4f}, Val AUC: {val_metrics['auc']:.4f}, LR: {current_lr:.6f}, Time: {elapsed:.1f}s")

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            save_metrics({'best_val_auc': best_auc, 'val_metrics': val_metrics}, args.output_dir)

    save_training_history(history, args.output_dir)
    print('Training complete. Best Val AUC:', best_auc)


if __name__ == '__main__':
    main()
