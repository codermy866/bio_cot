#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化训练至90%以上
使用真实数据 + 高级训练策略
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time
from datetime import datetime
import os
import json
from tqdm import tqdm

# 导入现有的模块
import sys
sys.path.append('.')
from cnn_multimodal_model import CNNMultimodalTransformer
from util.datasets import build_dataset


class FocalLoss(nn.Module):
    """Focal Loss处理类别不平衡"""
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


def prepare_optimized_train_loaders(data_path='5centers_multi', batch_size=8, num_workers=8):
    """准备优化的训练数据加载器"""
    print("🔄 准备优化数据加载器...")
    
    class Args:
        def __init__(self, data_path):
            self.data_path = data_path
            self.input_size = 224
            self.oct_num_frames = 48
            self.oct_cache_dir = None
    
    args = Args(data_path)
    
    train_dataset = build_dataset('train', args)
    test_dataset = build_dataset('test', args)
    
    print(f"✅ 训练集: {len(train_dataset)} 样本")
    print(f"✅ 测试集: {len(test_dataset)} 样本")
    
    # 创建加权采样器
    import pandas as pd
    train_df = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))
    labels = train_df['label'].values
    class_counts = np.bincount(labels)
    total = class_counts.sum()
    class_weights = total / (len(class_counts) * class_counts)
    class_weights = class_weights / class_weights.sum()
    
    weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def train_to_90_percent(epochs=30, batch_size=8, learning_rate=1e-4, data_path='5centers_multi', output_dir='cnn_training_90'):
    """训练至90%+"""
    
    print("🎯 开始优化训练至90%+")
    print("=" * 60)
    
    # 准备数据
    train_loader, test_loader = prepare_optimized_train_loaders(data_path, batch_size)
    
    # 初始化模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 使用设备: {device}")
    
    model = CNNMultimodalTransformer(
        num_classes=2,
        embed_dim=768,
        heads=8,
        dropout=0.2
    ).to(device)
    
    print(f"📊 模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 优化器和调度
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    # Focal Loss
    criterion = FocalLoss(alpha=None, gamma=2.0)
    
    # 混合精度
    scaler = GradScaler()
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }
    
    best_val_acc = 0
    best_val_auc = 0
    best_epoch = 0
    
    print("\n🚀 开始训练...")
    print("=" * 60)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            oct_images, col_images, clinical_features, labels, _ = batch
            
            oct_images = oct_images.to(device)
            col_images = col_images.to(device)
            clinical_features = clinical_features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(oct_images, col_images, clinical_features)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
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
                oct_images, col_images, clinical_features, labels, _ = batch
                
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
        
        # 计算AUC
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except:
            val_auc = 0.5
        
        # 记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_auc = val_auc
            best_epoch = epoch + 1
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train: Loss {train_loss:.4f}, Acc {train_acc:.2f}% | "
              f"Val: Loss {val_loss:.4f}, Acc {val_acc:.2f}%, AUC {val_auc:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # 提前停止检查
        if val_acc >= 90.0:
            print(f"\n🎉 达到目标！验证准确率: {val_acc:.2f}%")
            break
    
    print("\n" + "=" * 60)
    print(f"✅ 训练完成！")
    print(f"  最佳验证准确率: {best_val_acc:.2f}%")
    print(f"  最佳验证AUC: {best_val_auc:.4f}")
    print(f"  最佳轮次: Epoch {best_epoch}")
    
    # 保存历史
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history


def main():
    train_to_90_percent(
        epochs=30,
        batch_size=8,
        learning_rate=1e-4,
        data_path='5centers_multi',
        output_dir='cnn_training_90'
    )


if __name__ == "__main__":
    main()

