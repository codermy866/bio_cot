#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的CLIP训练脚本 - 采用成功的Swin-T架构和配置
目标：AUC达到70-80%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, 
    recall_score, confusion_matrix, roc_curve
)
import time
from datetime import datetime
import os
import json
import sys
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入成功的模型架构
from models.SwinT.swin_multimodal_model import SwinTMultimodalTransformer
from utils.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
from utils.advanced_clinical_metrics import calculate_clinical_metrics

# 导入超强数据增强
try:
    from utils.ultra_strong_augmentation import create_ultra_strong_transform
except ImportError:
    def create_ultra_strong_transform(input_size=224, is_training=True, use_albumentations=True):
        from timm.data import create_transform
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        if is_training:
            return create_transform(
                input_size=input_size,
                is_training=True,
                auto_augment='rand-m9-mstd0.5-inc1',
                interpolation='bicubic',
                re_prob=0.25,
                re_mode='pixel',
                re_count=1,
            )
        else:
            from torchvision import transforms
            return transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            ])


class FocalLoss(nn.Module):
    """Focal Loss with Label Smoothing - 参考成功模型"""
    
    def __init__(self, alpha=None, gamma=3.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        # 应用Label Smoothing
        if self.label_smoothing > 0:
            confidence = 1.0 - self.label_smoothing
            log_probs = nn.functional.log_softmax(inputs, dim=1)
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(self.label_smoothing / (num_classes - 1))
                true_dist.scatter_(1, targets.data.unsqueeze(1), confidence)
            ce_loss = -torch.sum(true_dist * log_probs, dim=1)
        else:
            ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


def prepare_loaders(data_path='5centers_multi', batch_size=5, input_size=192, oct_frames=48, use_strong_aug=True):
    """准备数据加载器 - 参考成功模型"""
    print("📥 加载数据集...")
    
    class Args:
        def __init__(self, data_path):
            self.data_path = data_path
            self.input_size = input_size
            self.oct_num_frames = oct_frames
            self.col_num_frames = 3
            self.oct_cache_dir = 'oct_cache_optimized'
            self.use_text_contrastive = False
            self.oct_points = 12
            self.oct_frames_per_point = 10
            self.use_pretrained_backbones = True
    
    args = Args(data_path)
    
    # 创建数据增强（使用timm的增强，避免albumentations版本问题）
    if use_strong_aug:
        try:
            from timm.data import create_transform
            from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
            train_transform = create_transform(
                input_size=input_size,
                is_training=True,
                auto_augment='rand-m9-mstd0.5-inc1',
                interpolation='bicubic',
                re_prob=0.25,
                re_mode='pixel',
                re_count=1,
            )
            val_transform = create_transform(
                input_size=input_size,
                is_training=False,
                interpolation='bicubic',
            )
        except Exception as e:
            print(f"⚠️  使用timm增强失败: {e}，使用标准增强")
            from torchvision import transforms
            train_transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            val_transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    else:
        train_transform = None
        val_transform = None
    
    # 训练集
    train_dataset = EnhancedMultimodalCervicalDataset(
        root=os.path.join(data_path, 'train'),
        is_train='train',
        args=args,
        transform=train_transform,
        use_enhanced_oct=True,
        cache_oct_features=True
    )
    
    # 验证集
    val_dataset = EnhancedMultimodalCervicalDataset(
        root=os.path.join(data_path, 'test'),
        is_train='test',
        args=args,
        transform=val_transform,
        use_enhanced_oct=True,
        cache_oct_features=True
    )
    
    def collate_fn(batch):
        """自定义collate函数 - 处理字典和元组两种返回格式"""
        oct_images = []
        col_images = []
        clinical_features = []
        labels = []
        
        for item in batch:
            if isinstance(item, dict):
                # 字典格式（use_pretrained_backbones=True）
                oct_img = item.get('oct_images', item.get('oct_features'))
                col_img = item.get('col_images', item.get('colposcopy_features'))
                clinical_feat = item.get('clinical_features')
                label = item.get('label', item.get('labels'))
            elif isinstance(item, (list, tuple)) and len(item) >= 4:
                # 元组格式
                oct_img, col_img, clinical_feat, label = item[:4]
            else:
                continue  # 跳过无效项
            
            # 确保是tensor
            if not isinstance(oct_img, torch.Tensor):
                oct_img = torch.tensor(oct_img) if oct_img is not None else torch.zeros(1)
            if not isinstance(col_img, torch.Tensor):
                col_img = torch.tensor(col_img) if col_img is not None else torch.zeros(1)
            if not isinstance(clinical_feat, torch.Tensor):
                clinical_feat = torch.tensor(clinical_feat) if clinical_feat is not None else torch.zeros(7)
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label, dtype=torch.long) if label is not None else torch.tensor(0, dtype=torch.long)
            
            # 确保维度正确
            if oct_img.dim() == 4:  # [C, H, W] -> [1, C, H, W] for stack
                oct_img = oct_img.unsqueeze(0)
            if col_img.dim() == 4:
                col_img = col_img.unsqueeze(0)
            if clinical_feat.dim() == 1:
                clinical_feat = clinical_feat.unsqueeze(0)
            if label.dim() == 0:
                label = label.unsqueeze(0)
            
            oct_images.append(oct_img)
            col_images.append(col_img)
            clinical_features.append(clinical_feat)
            labels.append(label)
        
        if len(oct_images) == 0:
            raise ValueError("Batch is empty after processing")
        
        # 处理OCT图像（可能是多帧）
        try:
            if oct_images[0].dim() == 5:  # [B, T, C, H, W]
                oct_images = torch.cat(oct_images, dim=0)
            else:
                oct_images = torch.stack(oct_images)
        except Exception:
            oct_images = torch.stack([img.squeeze(0) if img.dim() > 4 else img for img in oct_images])
        
        # 处理COL图像
        try:
            if col_images[0].dim() == 5:  # [B, T, C, H, W]
                col_images = torch.cat(col_images, dim=0)
            else:
                col_images = torch.stack(col_images)
        except Exception:
            col_images = torch.stack([img.squeeze(0) if img.dim() > 4 else img for img in col_images])
        
        # 处理临床特征
        clinical_features = torch.cat(clinical_features, dim=0) if clinical_features[0].dim() == 2 else torch.stack(clinical_features)
        
        # 处理标签
        labels = torch.cat(labels, dim=0) if labels[0].dim() == 1 else torch.stack(labels)
        labels = labels.squeeze() if labels.dim() > 1 else labels
        
        return {
            'oct_images': oct_images,
            'col_images': col_images,
            'clinical_features': clinical_features,
            'labels': labels
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"✅ 训练集: {len(train_dataset)} 样本")
    print(f"✅ 验证集: {len(val_dataset)} 样本")
    
    return train_loader, val_loader


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, num_epochs):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    
    for batch_idx, batch in enumerate(pbar):
        if isinstance(batch, dict):
            oct_images = batch.get('oct_images')
            col_images = batch.get('col_images')
            clinical_features = batch.get('clinical_features')
            labels = batch.get('labels')
        else:
            if len(batch) >= 4:
                oct_images, col_images, clinical_features, labels = batch[:4]
            else:
                continue
        
        oct_images = oct_images.to(device)
        col_images = col_images.to(device)
        clinical_features = clinical_features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            logits = model(oct_images, col_images, clinical_features)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy_score(all_labels, all_preds):.2%}'
        })
    
    avg_loss = total_loss / len(train_loader)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='验证中'):
            if isinstance(batch, dict):
                oct_images = batch.get('oct_images')
                col_images = batch.get('col_images')
                clinical_features = batch.get('clinical_features')
                labels = batch.get('labels')
            else:
                if len(batch) >= 4:
                    oct_images, col_images, clinical_features, labels = batch[:4]
                else:
                    continue
            
            oct_images = oct_images.to(device)
            col_images = col_images.to(device)
            clinical_features = clinical_features.to(device)
            labels = labels.to(device)
            
            with autocast():
                logits = model(oct_images, col_images, clinical_features)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].detach().cpu().numpy().tolist())
    
    avg_loss = total_loss / len(val_loader)
    acc = accuracy_score(all_labels, all_preds)
    
    # 计算AUC
    auc = 0.5
    if len(all_labels) > 0 and len(set(all_labels)) > 1:
        probs_np = np.asarray(all_probs, dtype=float)
        if np.std(probs_np) > 1e-8:
            try:
                auc = roc_auc_score(all_labels, probs_np)
            except Exception:
                auc = 0.5
    
    # 计算最优阈值
    optimal_threshold = 0.5
    optimal_acc = acc
    optimal_f1 = f1_score(all_labels, all_preds, zero_division=0)
    optimal_sensitivity = recall_score(all_labels, all_preds, zero_division=0)
    optimal_specificity = 0.0
    
    if len(all_labels) > 0 and len(set(all_labels)) > 1:
        probs_np = np.asarray(all_probs, dtype=float)
        if np.std(probs_np) > 1e-8:
            fpr, tpr, thresholds = roc_curve(all_labels, probs_np)
            youden_index = tpr - fpr
            optimal_idx = int(np.argmax(youden_index))
            optimal_threshold = float(thresholds[optimal_idx])
        
        opt_preds = (probs_np >= optimal_threshold).astype(int)
        optimal_acc = accuracy_score(all_labels, opt_preds)
        optimal_f1 = f1_score(all_labels, opt_preds, zero_division=0)
        optimal_sensitivity = recall_score(all_labels, opt_preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(all_labels, opt_preds, labels=[0,1]).ravel()
        denom = (tn + fp)
        optimal_specificity = float(tn / denom) if denom > 0 else 0.0
    
    try:
        clinical_metrics = calculate_clinical_metrics(all_labels, all_preds, all_probs)
    except Exception:
        clinical_metrics = {}
    
    return {
        'loss': avg_loss,
        'accuracy': acc,
        'auc': float(auc),
        'f1': optimal_f1,
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': optimal_sensitivity,
        'optimal_threshold': float(optimal_threshold),
        'optimal_accuracy': optimal_acc,
        'optimal_f1': optimal_f1,
        'optimal_sensitivity': optimal_sensitivity,
        'optimal_specificity': optimal_specificity,
        'clinical_metrics': clinical_metrics
    }


def main():
    """主训练函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='优化的CLIP训练 - 采用成功Swin-T架构')
    parser.add_argument('--data_path', type=str, default='5centers_multi', help='数据路径')
    parser.add_argument('--output_dir', type=str, default='adaptive_causal_intervention_results', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=5, help='批次大小（参考成功模型）')
    parser.add_argument('--num_epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2.1e-5, help='学习率（参考成功模型）')
    parser.add_argument('--input_size', type=int, default=192, help='输入尺寸（参考成功模型）')
    parser.add_argument('--oct_frames', type=int, default=48, help='OCT帧数（参考成功模型）')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()
    
    data_path = args.data_path
    output_dir = args.output_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    input_size = args.input_size
    oct_frames = args.oct_frames
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print("=" * 80)
    print("🚀 优化的CLIP训练 - 采用成功Swin-T架构")
    print("=" * 80)
    print(f"📁 数据路径: {data_path}")
    print(f"💻 使用设备: {device}")
    print(f"📦 批次大小: {batch_size}（参考成功模型）")
    print(f"🎯 训练轮数: {num_epochs}")
    print(f"📚 学习率: {learning_rate}（参考成功模型）")
    print(f"🖼️  输入尺寸: {input_size}（参考成功模型）")
    print(f"🎬 OCT帧数: {oct_frames}（参考成功模型）")
    print(f"🔬 使用超强数据增强")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    train_loader, val_loader = prepare_loaders(
        data_path, batch_size, input_size, oct_frames, use_strong_aug=True
    )
    
    # 创建模型（使用成功的Swin-T架构）
    print("\n🔧 创建模型（Swin-T架构）...")
    model = SwinTMultimodalTransformer(
        num_classes=2,
        embed_dim=768,
        num_heads=8,
        dropout=0.25,  # 参考成功模型
        clinical_dim=7,
        oct_num_frames=oct_frames,
        col_num_frames=3,
        swin_name='swin_tiny_patch4_window7_224',
        pretrained=True,
        input_size=input_size,
        use_frame_attention=False,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 模型创建成功")
    print(f"   总参数量: {total_params / 1e6:.2f}M")
    print(f"   可训练参数: {trainable_params / 1e6:.2f}M")
    
    # 计算类别权重
    class_alpha = torch.tensor([0.325, 0.675]).to(device)
    
    # 创建损失函数（参考成功模型）
    criterion = FocalLoss(
        alpha=class_alpha,
        gamma=3.0,  # 参考成功模型
        label_smoothing=0.1  # 参考成功模型
    )
    
    # 创建优化器（参考成功模型）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=6e-4,  # 参考成功模型
        betas=(0.9, 0.999)
    )
    
    # 学习率调度（参考成功模型）
    warmup_epochs = max(1, num_epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scaler = GradScaler()
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_f1': [],
        'best_auc': 0.0,
        'best_epoch': 0
    }
    
    print("\n🎓 开始训练...\n")
    best_auc = 0.0
    patience = 7
    patience_counter = 0
    min_delta = 0.0005
    
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, num_epochs
        )
        
        # 验证
        val_results = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_results['loss'])
        history['val_acc'].append(val_results['accuracy'])
        history['val_auc'].append(val_results['auc'])
        history['val_f1'].append(val_results['f1'])
        
        # 打印结果
        print(f"\n{'='*80}")
        print(f"📊 Epoch [{epoch+1}/{num_epochs}] 结果")
        print(f"{'='*80}")
        print(f"训练集:")
        print(f"  Loss: {train_loss:.4f}  |  Accuracy: {train_acc:.2%}")
        print(f"验证集:")
        print(f"  Loss: {val_results['loss']:.4f}")
        print(f"  Accuracy: {val_results['accuracy']:.2%}")
        print(f"  AUC: {val_results['auc']:.4f}")
        print(f"  F1-Score: {val_results['f1']:.4f}")
        print(f"  最优阈值: {val_results['optimal_threshold']:.4f}")
        print(f"  最优准确率: {val_results['optimal_accuracy']:.2%}")
        print(f"  最优敏感性: {val_results['optimal_sensitivity']:.4f}")
        print(f"  最优特异性: {val_results['optimal_specificity']:.4f}")
        
        # Early Stopping
        if val_results['auc'] > best_auc + min_delta:
            best_auc = val_results['auc']
            patience_counter = 0
            history['best_auc'] = best_auc
            history['best_epoch'] = epoch + 1
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_auc': best_auc,
                'val_results': val_results,
                'history': history
            }
            torch.save(checkpoint, os.path.join(output_dir, 'best_model_swin_style.pth'))
            print(f"\n💾 保存最佳模型 (Epoch {epoch+1}, AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
        
        # Early Stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early Stopping触发 (patience={patience})")
            print(f"   最佳AUC: {best_auc:.4f} (Epoch {history['best_epoch']})")
            break
        
        # 保存训练历史
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        with open(os.path.join(output_dir, 'training_history_swin_style.json'), 'w') as f:
            json.dump(convert_to_serializable(history), f, indent=2)
    
    print(f"\n✅ 训练完成！最佳AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()

