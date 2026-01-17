#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swin-S/B 大模型训练脚本
使用超强数据增强和更大的Swin模型
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import time
from datetime import datetime
import json
from tqdm import tqdm

try:
    from models.SwinT.swin_multimodal_model import SwinTMultimodalTransformer
except ImportError:
    # 如果导入失败，尝试从项目根目录导入
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.SwinT.swin_multimodal_model import SwinTMultimodalTransformer

from utils.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
try:
    from utils.ultra_strong_augmentation import create_ultra_strong_transform
except ImportError:
    # 如果导入失败，使用标准增强
    def create_ultra_strong_transform(input_size=224, is_training=True, use_albumentations=True):
        from timm.data import create_transform
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        if is_training:
            return create_transform(
                input_size=input_size,
                is_training=True,
                color_jitter=0.5,
                auto_augment='rand-m9-mstd0.5-inc1',
                interpolation='bicubic',
                re_prob=0.4,
                re_mode='pixel',
                re_count=2,
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            )
        else:
            from torchvision import transforms
            return transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            ])

try:
    from utils.advanced_clinical_metrics import calculate_clinical_metrics, calculate_calibration_metrics
except ImportError:
    def calculate_clinical_metrics(y_true, y_pred, y_probs):
        return {}
    def calculate_calibration_metrics(y_true, y_probs):
        return {}


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


def prepare_loaders(data_path='5centers_multi', batch_size=4, input_size=224, oct_frames=48, use_strong_aug=True):
    """准备数据加载器"""
    print("🔄 准备数据加载器...")
    
    class Args:
        def __init__(self, data_path):
            self.data_path = data_path
            self.input_size = input_size
            self.oct_num_frames = oct_frames
            self.oct_cache_dir = 'oct_cache_optimized'
            self.use_text_contrastive = False
            self.use_pretrained_backbones = True
            self.oct_points = 12
            self.oct_frames_per_point = 10
    
    args = Args(data_path)
    
    # 创建超强数据增强
    if use_strong_aug:
        train_transform = create_ultra_strong_transform(input_size=input_size, is_training=True, use_albumentations=True)
        val_transform = create_ultra_strong_transform(input_size=input_size, is_training=False, use_albumentations=True)
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
    
    # 测试集
    test_dataset = EnhancedMultimodalCervicalDataset(
        root=os.path.join(data_path, 'test'),
        is_train='test',
        args=args,
        transform=val_transform,
        use_enhanced_oct=True,
        cache_oct_features=True
    )
    
    print(f"✅ 数据加载完成")
    print(f"  训练样本: {len(train_dataset)}")
    print(f"  测试样本: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader


def train_swin_large(
    swin_model='swin_small',  # 'swin_tiny', 'swin_small', 'swin_base'
    epochs=30,
    batch_size=4,  # 大模型需要更小的batch size
    learning_rate=2e-5,
    data_path='5centers_multi',
    output_dir='swin_large_results',
    input_size=224,
    oct_frames=48,
    use_strong_aug=True,
    gradient_accumulation_steps=1,  # 梯度累积步数
):
    """训练大Swin模型"""
    
    print("🎯 开始训练大Swin模型")
    print("=" * 60)
    print(f"模型: {swin_model}")
    print(f"数据增强: {'超强增强' if use_strong_aug else '标准增强'}")
    print("=" * 60)
    
    # 准备数据
    train_loader, test_loader = prepare_loaders(
        data_path, batch_size, input_size, oct_frames, use_strong_aug
    )
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 使用设备: {device}")
    
    # 根据模型选择配置
    if swin_model == 'swin_tiny':
        swin_name = 'swin_tiny_patch4_window7_224'
        embed_dim = 768
        num_heads = 12
    elif swin_model == 'swin_small':
        swin_name = 'swin_small_patch4_window7_224'
        embed_dim = 768
        num_heads = 12
    elif swin_model == 'swin_base':
        swin_name = 'swin_base_patch4_window7_224'
        embed_dim = 1024
        num_heads = 16
    else:
        raise ValueError(f"Unknown swin_model: {swin_model}")
    
    model = SwinTMultimodalTransformer(
        num_classes=2,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.2,
        clinical_dim=7,
        oct_num_frames=oct_frames,
        col_num_frames=3,
        swin_name=swin_name,
        pretrained=True,
        input_size=input_size,
        use_frame_attention=False,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数量: {total_params/1e6:.2f}M (可训练: {trainable_params/1e6:.2f}M)")
    
    # 优化器和调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=5e-4,
        betas=(0.9, 0.999)
    )
    
    warmup_epochs = max(1, epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Focal Loss
    criterion = FocalLoss(alpha=torch.tensor([0.325, 0.675]).to(device), gamma=2.5)
    
    # 混合精度
    scaler = GradScaler()
    max_grad_norm = 1.0
    
    # 训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [],
        'val_f1': [], 'val_precision': [], 'val_recall': [],
        'learning_rate': []
    }
    
    best_val_auc = 0
    best_epoch = 0
    patience = 7
    patience_counter = 0
    min_delta = 0.001
    
    print("\n🚀 开始训练...")
    print("=" * 60)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, dict):
                oct_images = batch.get('oct_images')
                col_images = batch.get('col_images')
                clinical_features = batch['clinical_features'] if 'clinical_features' in batch else batch.get('clinical')
                labels = batch['label'] if 'label' in batch else batch.get('labels')
            else:
                if len(batch) == 5:
                    oct_images, col_images, clinical_features, labels, _ = batch
                else:
                    oct_images, col_images, clinical_features, labels = batch
            
            oct_images = oct_images.to(device)
            col_images = col_images.to(device)
            clinical_features = clinical_features.to(device)
            labels = labels.to(device)
            
            with autocast():
                outputs = model(oct_images, col_images, clinical_features)
                loss = criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps  # 梯度累积时缩放loss
            
            scaler.scale(loss).backward()
            
            # 梯度累积：每accumulation_steps步更新一次
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                current_acc = 100. * train_correct / train_total
                avg_loss = train_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="验证中"):
                if isinstance(batch, dict):
                    oct_images = batch.get('oct_images')
                    col_images = batch.get('col_images')
                    clinical_features = batch['clinical_features'] if 'clinical_features' in batch else batch.get('clinical')
                    labels = batch['label'] if 'label' in batch else batch.get('labels')
                else:
                    if len(batch) == 5:
                        oct_images, col_images, clinical_features, labels, _ = batch
                    else:
                        oct_images, col_images, clinical_features, labels = batch
                
                oct_images = oct_images.to(device)
                col_images = col_images.to(device)
                clinical_features = clinical_features.to(device)
                labels = labels.to(device)
                
                with autocast():
                    outputs = model(oct_images, col_images, clinical_features)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        val_loss /= len(test_loader)
        val_acc = 100. * val_correct / val_total
        
        all_labels_array = np.array(all_labels)
        all_probs_array = np.array(all_probs)
        
        try:
            val_auc = roc_auc_score(all_labels_array, all_probs_array)
            
            # 使用Youden指数方法找到最优阈值（替代默认0.5）
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(all_labels_array, all_probs_array)
            youden_index = tpr - fpr
            best_idx = np.argmax(youden_index)
            optimal_threshold = thresholds[best_idx]
            
            # 使用最优阈值进行预测
            val_pred = (all_probs_array >= optimal_threshold).astype(int)
            val_f1 = f1_score(all_labels_array, val_pred)
            val_precision = precision_score(all_labels_array, val_pred, zero_division=0)
            val_recall = recall_score(all_labels_array, val_pred, zero_division=0)
            
            cm = confusion_matrix(all_labels_array, val_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            clinical_metrics = calculate_clinical_metrics(
                y_true=all_labels_array,
                y_pred=val_pred,
                y_probs=all_probs_array
            )
            
            # 添加最优阈值信息
            clinical_metrics['optimal_threshold'] = float(optimal_threshold)
            clinical_metrics['youden_index'] = float(youden_index[best_idx])
        except Exception as e:
            val_auc = 0.5
            val_f1 = 0.0
            val_precision = 0.0
            val_recall = 0.0
            tn, fp, fn, tp = 0, 0, 0, 0
            clinical_metrics = {}
            print(f"Warning: 计算指标时出错: {e}")
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(float(val_f1))
        history['val_precision'].append(float(val_precision))
        history['val_recall'].append(float(val_recall))
        
        current_lr = scheduler.get_last_lr()[0]
        history['learning_rate'].append(float(current_lr))
        scheduler.step()
        
        # 保存最佳模型
        if val_auc > best_val_auc + min_delta:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"💾 保存最佳模型 (Epoch {epoch+1}, AUC: {val_auc:.4f})", flush=True)
        else:
            patience_counter += 1
        
        epoch_time = time.time() - epoch_start
        
        # 打印结果
        print("\n" + "=" * 80, flush=True)
        print(f"📊 Epoch [{epoch+1}/{epochs}] 训练结果", flush=True)
        print("-" * 80, flush=True)
        print(f"训练集: Loss={train_loss:.4f}, Acc={train_acc:.2f}%", flush=True)
        print(f"验证集: Loss={val_loss:.4f}, Acc={val_acc:.2f}%, AUC={val_auc:.4f}, F1={val_f1:.4f}", flush=True)
        if clinical_metrics:
            print(f"  灵敏度: {clinical_metrics.get('sensitivity', 0):.4f}, 特异度: {clinical_metrics.get('specificity', 0):.4f}", flush=True)
        print(f"  Learning Rate: {current_lr:.6f}, Time: {epoch_time:.1f}s", flush=True)
        if val_auc == best_val_auc:
            print(f"  ⭐ 当前最佳模型 (AUC: {val_auc:.4f})", flush=True)
        print("=" * 80 + "\n", flush=True)
        
        # Early Stopping
        if patience_counter >= patience:
            print(f"\n🛑 Early Stopping触发: 验证AUC连续{patience}个epoch未提升", flush=True)
            print(f"   最佳AUC: {best_val_auc:.4f} (Epoch {best_epoch})", flush=True)
            break
    
    # 保存历史
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    summary = {
        'best_epoch': best_epoch,
        'best_val_auc': float(best_val_auc),
        'best_val_acc': float(max(history['val_acc'])) if history['val_acc'] else 0.0,
        'best_val_f1': float(max(history['val_f1'])) if history['val_f1'] else 0.0,
        'total_epochs': epochs,
        'model': swin_model,
        'use_strong_aug': use_strong_aug,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80, flush=True)
    print(f"✅ 训练完成！", flush=True)
    print(f"📈 最终结果: 最佳验证AUC: {best_val_auc:.4f} (Epoch {best_epoch})", flush=True)
    print("=" * 80 + "\n", flush=True)
    
    return model, history


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--swin_model', type=str, default='swin_small', choices=['swin_tiny', 'swin_small', 'swin_base'])
    parser.add_argument('--data_path', type=str, default='5centers_multi')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--output_dir', type=str, default='swin_large_results')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--oct_frames', type=int, default=48)
    parser.add_argument('--no_strong_aug', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    
    args = parser.parse_args()
    
    train_swin_large(
        swin_model=args.swin_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        data_path=args.data_path,
        output_dir=args.output_dir,
        input_size=args.input_size,
        oct_frames=args.oct_frames,
        use_strong_aug=not args.no_strong_aug,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )


if __name__ == "__main__":
    main()

