#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ResNet多模态训练脚本（襄阳数据集，作为baseline）
"""

import sys
from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from xiangyang_multimodal_dataset import XiangyangMultimodalDataset


class ResNetMultimodalArgs:
    def __init__(self):
        self.data_root = '/data2/hmy/5Center_datas/襄阳按点图片分类_multimodal'
        self.output_dir = Path(__file__).parent / 'results_multimodal'
        self.checkpoint_dir = Path(__file__).parent / 'checkpoints_multimodal'
        self.log_dir = Path(__file__).parent / 'logs'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = 'cuda:1'
        self.num_epochs = 50
        self.batch_size = 8
        self.num_workers = 4
        self.pin_memory = True
        
        self.oct_num_frames = 60
        self.num_classes = 2
        
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.step_size = 15
        self.gamma = 0.1


def get_data_transforms(args, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class MultimodalResNet(nn.Module):
    """多模态ResNet模型"""
    def __init__(self, num_classes=2):
        super().__init__()
        # OCT编码器
        self.oct_backbone = models.resnet50(pretrained=True)
        self.oct_backbone.fc = nn.Identity()
        
        # Colposcopy编码器
        self.col_backbone = models.resnet50(pretrained=True)
        self.col_backbone.fc = nn.Identity()
        
        # 临床特征编码器
        self.clinical_encoder = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 2048 + 256, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, oct_images, col_images, clinical_features):
        # OCT特征（平均池化多帧）
        B, F, C, H, W = oct_images.shape
        oct_flat = oct_images.view(B * F, C, H, W)
        oct_feat = self.oct_backbone(oct_flat)  # [B*F, 2048]
        oct_feat = oct_feat.view(B, F, -1).mean(dim=1)  # [B, 2048]
        
        # Colposcopy特征（平均池化3张图像）
        B, N, C, H, W = col_images.shape
        col_flat = col_images.view(B * N, C, H, W)
        col_feat = self.col_backbone(col_flat)  # [B*N, 2048]
        col_feat = col_feat.view(B, N, -1).mean(dim=1)  # [B, 2048]
        
        # 临床特征
        clinical_feat = self.clinical_encoder(clinical_features)  # [B, 256]
        
        # 融合
        fused = torch.cat([oct_feat, col_feat, clinical_feat], dim=1)  # [B, 4352]
        logits = self.fusion(fused)  # [B, num_classes]
        
        return logits


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, args):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Train]')
    for batch in pbar:
        oct_images = batch['oct_images'].to(device)
        col_images = batch['colposcopy_images'].to(device)
        clinical_features = batch['clinical_features'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(oct_images, col_images, clinical_features)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return {'loss': epoch_loss, 'acc': epoch_acc}


def validate(model, dataloader, criterion, device, epoch, args):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Val]')
        for batch in pbar:
            oct_images = batch['oct_images'].to(device)
            col_images = batch['colposcopy_images'].to(device)
            clinical_features = batch['clinical_features'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(oct_images, col_images, clinical_features)
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
            acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except:
        epoch_auc = 0.0
    
    return {
        'loss': epoch_loss,
        'acc': epoch_acc,
        'auc': epoch_auc,
        'preds': all_preds,
        'labels': all_labels,
        'probs': all_probs
    }


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 训练曲线已保存: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['阴性', '阳性'], 
                yticklabels=['阴性', '阳性'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存: {save_path}")


def main():
    args = ResNetMultimodalArgs()
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA不可用，使用CPU")
        args.device = 'cpu'
        device = torch.device('cpu')
    else:
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if visible_devices:
            device = torch.device('cuda:0')
            args.device = 'cuda:0'
            print(f"✅ 检测到CUDA_VISIBLE_DEVICES={visible_devices}，使用设备: cuda:0")
        else:
            device_idx = int(args.device.split(':')[1]) if ':' in args.device else 0
            if device_idx >= torch.cuda.device_count():
                print(f"⚠️ GPU {device_idx} 不可用，使用GPU 0")
                device_idx = 0
            device = torch.device(f'cuda:{device_idx}')
            args.device = f'cuda:{device_idx}'
            print(f"✅ 使用设备: {args.device}")
    
    # 创建数据集
    print("📂 加载数据集...")
    train_dataset = XiangyangMultimodalDataset(
        data_root=args.data_root,
        split='train',
        transform=get_data_transforms(args, is_train=True),
        oct_num_frames=args.oct_num_frames,
        oct_frames_per_point=5
    )
    val_dataset = XiangyangMultimodalDataset(
        data_root=args.data_root,
        split='val',
        transform=get_data_transforms(args, is_train=False),
        oct_num_frames=args.oct_num_frames,
        oct_frames_per_point=5
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    print(f"✅ 数据加载完成: 训练集 {len(train_dataset)} 个样本, 验证集 {len(val_dataset)} 个样本")
    
    # 计算类别权重
    train_labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    class_weights = total_samples / (args.num_classes * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"📊 类别分布: {dict(enumerate(class_counts))}")
    print(f"📊 类别权重: {dict(enumerate(class_weights.cpu().numpy()))}")
    
    # 创建ResNet模型
    print("🏗️ 创建ResNet多模态模型...")
    model = MultimodalResNet(num_classes=args.num_classes)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }
    
    best_val_acc = 0.0
    best_val_auc = 0.0
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("🚀 开始训练...")
    for epoch in range(args.num_epochs):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args)
        val_metrics = validate(model, val_loader, criterion, device, epoch, args)
        
        scheduler.step()
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['acc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_auc'].append(val_metrics['auc'])
        
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['acc']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        
        # 保存最佳模型
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            best_val_auc = val_metrics['auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['acc'],
                'val_auc': val_metrics['auc']
            }, args.checkpoint_dir / f'best_model_resnet_{timestamp}.pth')
            print(f"  ✅ 保存最佳模型 (Val Acc: {best_val_acc:.4f}, Val AUC: {best_val_auc:.4f})")
    
    # 最终评估
    print("📊 使用最佳模型进行最终评估...")
    checkpoint = torch.load(args.checkpoint_dir / f'best_model_resnet_{timestamp}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    final_metrics = validate(model, val_loader, criterion, device, args.num_epochs, args)
    
    # 分类报告
    report = classification_report(
        final_metrics['labels'],
        final_metrics['preds'],
        target_names=['阴性', '阳性'],
        output_dict=True
    )
    
    # 保存结果
    results = {
        'model': 'ResNet (多模态baseline)',
        'best_val_acc': best_val_acc,
        'best_val_auc': best_val_auc,
        'final_metrics': {
            'loss': final_metrics['loss'],
            'acc': final_metrics['acc'],
            'auc': final_metrics['auc']
        },
        'classification_report': report
    }
    
    results_file = args.output_dir / f'results_resnet_{timestamp}.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ 结果已保存: {results_file}")
    
    # 可视化
    plot_training_curves(history, args.output_dir / f'training_curves_resnet_{timestamp}.png')
    plot_confusion_matrix(
        final_metrics['labels'],
        final_metrics['preds'],
        args.output_dir / f'confusion_matrix_resnet_{timestamp}.png'
    )
    
    print("✅ 训练完成!")


if __name__ == '__main__':
    main()

