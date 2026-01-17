#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5分类多模态模型训练脚本
支持OCT、Colposcopy和临床特征的5分类训练
"""

import os
import sys
import json
import time
from datetime import datetime
import logging
import argparse

class _Tee:
    """A simple tee to duplicate stdout/stderr to a file."""
    def __init__(self, stream, logfile_path: str):
        self.stream = stream
        self.log = open(logfile_path, 'a', buffering=1)

    def write(self, data):
        try:
            self.stream.write(data)
        except Exception:
            pass
        try:
            self.log.write(data)
        except Exception:
            pass

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass
        try:
            self.log.flush()
        except Exception:
            pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# 导入模型和数据集
from cnn_multimodal_model_5class import CNNMultimodalTransformer5Class, create_5class_model
from enhanced_multimodal_dataset import build_enhanced_dataset

class FocalLoss(nn.Module):
    """Focal Loss for 5分类"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        log_preds = F.log_softmax(inputs, dim=1)
        targets = targets * (1 - self.smoothing) + self.smoothing / inputs.size(1)
        loss = (-targets * log_preds).sum(dim=1)
        return loss.mean()

def prepare_loaders_5class(data_path: str, batch_size: int = 8, num_workers: int = 8, 
                          input_size: int = 160, oct_num_frames: int = 48, 
                          oct_cache_dir: str = None, use_text_contrastive: bool = False):
    """准备5分类数据加载器"""
    print("🔄 准备5分类数据加载器...")
    
    # 创建参数对象
    class Args:
        def __init__(self, data_path, input_size=160, oct_num_frames=48):
            self.data_path = data_path
            self.input_size = input_size
            self.oct_num_frames = oct_num_frames
            self.use_text_contrastive = use_text_contrastive
    
    args = Args(data_path, input_size, oct_num_frames)
    
    # 构建数据集
    train_dataset = build_enhanced_dataset('train', args)
    test_dataset = build_enhanced_dataset('test', args)
    
    # 计算类别权重（用于处理不平衡）
    train_labels = [train_dataset[i][3] for i in range(len(train_dataset))]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    
    print(f"📊 类别分布: {class_counts}")
    print(f"📊 类别权重: {class_weights}")
    
    # 创建加权采样器
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # 创建数据加载器
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
    
    print(f"✅ 训练集: {len(train_dataset)} 样本")
    print(f"✅ 测试集: {len(test_dataset)} 样本")
    
    return train_loader, test_loader, class_weights

def calculate_metrics_5class(y_true, y_pred, y_prob):
    """计算5分类指标"""
    # 基本指标
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 多分类AUC
    try:
        auc_macro = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        auc_weighted = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    except:
        auc_macro = 0.0
        auc_weighted = 0.0
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'auc_macro': auc_macro,
        'auc_weighted': auc_weighted
    }

def train_epoch_5class(model, train_loader, criterion, optimizer, device, scaler, class_names):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch_idx, (oct_data, col_data, clinical_data, labels, metadata) in enumerate(train_loader):
        oct_data = oct_data.to(device)
        col_data = col_data.to(device)
        clinical_data = clinical_data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(oct_data, col_data, clinical_data)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        # 收集预测结果
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # 计算指标
    metrics = calculate_metrics_5class(all_labels, all_preds, all_probs)
    
    return total_loss / len(train_loader), metrics

def validate_epoch_5class(model, test_loader, criterion, device, class_names):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for oct_data, col_data, clinical_data, labels, metadata in test_loader:
            oct_data = oct_data.to(device)
            col_data = col_data.to(device)
            clinical_data = clinical_data.to(device)
            labels = labels.to(device)
            
            outputs = model(oct_data, col_data, clinical_data)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # 收集预测结果
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算指标
    metrics = calculate_metrics_5class(all_labels, all_preds, all_probs)
    
    return total_loss / len(test_loader), metrics

def plot_training_history_5class(history, output_dir):
    """绘制5分类训练历史"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 损失曲线
    axes[0, 0].plot(history['train_loss'], label='训练损失')
    axes[0, 0].plot(history['val_loss'], label='验证损失')
    axes[0, 0].set_title('损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 准确率曲线
    axes[0, 1].plot(history['train_accuracy'], label='训练准确率')
    axes[0, 1].plot(history['val_accuracy'], label='验证准确率')
    axes[0, 1].set_title('准确率曲线')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1分数曲线
    axes[0, 2].plot(history['train_f1_macro'], label='训练F1(macro)')
    axes[0, 2].plot(history['val_f1_macro'], label='验证F1(macro)')
    axes[0, 2].plot(history['train_f1_weighted'], label='训练F1(weighted)')
    axes[0, 2].plot(history['val_f1_weighted'], label='验证F1(weighted)')
    axes[0, 2].set_title('F1分数曲线')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # AUC曲线
    axes[1, 0].plot(history['train_auc_macro'], label='训练AUC(macro)')
    axes[1, 0].plot(history['val_auc_macro'], label='验证AUC(macro)')
    axes[1, 0].plot(history['train_auc_weighted'], label='训练AUC(weighted)')
    axes[1, 0].plot(history['val_auc_weighted'], label='验证AUC(weighted)')
    axes[1, 0].set_title('AUC曲线')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 精确率和召回率
    axes[1, 1].plot(history['train_precision_macro'], label='训练精确率')
    axes[1, 1].plot(history['val_precision_macro'], label='验证精确率')
    axes[1, 1].plot(history['train_recall_macro'], label='训练召回率')
    axes[1, 1].plot(history['val_recall_macro'], label='验证召回率')
    axes[1, 1].set_title('精确率和召回率')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 学习率曲线
    if 'learning_rate' in history:
        axes[1, 2].plot(history['learning_rate'])
        axes[1, 2].set_title('学习率曲线')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history_5class.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train_5class_model(data_path='5centers_multi_5class', 
                       output_dir='cnn_training_5class',
                       num_epochs=20,
                       batch_size=8,
                       learning_rate=1e-4,
                       weight_decay=1e-5,
                       embed_dim=512,
                       clinical_dim=8,
                       use_focal_loss=True,
                       use_label_smoothing=True,
                       use_causal_adjustment=False,
                       use_text_contrastive=False):
    """训练5分类模型"""
    
    print("🚀 开始5分类模型训练")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(output_dir, 'training.log')
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 加载标签映射
    label_mapping_path = os.path.join(data_path, 'label_mapping.json')
    with open(label_mapping_path, 'r', encoding='utf-8') as f:
        label_info = json.load(f)
    
    class_names = label_info['class_names']
    num_classes = label_info['num_classes']
    
    print(f"📊 类别信息:")
    print(f"  类别数: {num_classes}")
    print(f"  类别名称: {class_names}")
    
    # 准备数据
    train_loader, test_loader, class_weights = prepare_loaders_5class(
        data_path=data_path,
        batch_size=batch_size,
        use_text_contrastive=use_text_contrastive
    )
    
    # 创建模型
    model = create_5class_model(
        num_classes=num_classes,
        embed_dim=embed_dim,
        clinical_dim=clinical_dim,
        use_causal_adjustment=use_causal_adjustment,
        use_text_contrastive=use_text_contrastive
    )
    
    model = model.to(device)
    print(f"🏗️ 模型创建完成")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建损失函数
    if use_focal_loss:
        # 计算Focal Loss的alpha权重
        alpha = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
        print("🎯 使用Focal Loss")
    elif use_label_smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        print("🎯 使用标签平滑交叉熵")
    else:
        # 使用加权交叉熵
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        print("🎯 使用加权交叉熵")
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 训练历史
    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_f1_macro': [], 'val_f1_macro': [],
        'train_f1_weighted': [], 'val_f1_weighted': [],
        'train_precision_macro': [], 'val_precision_macro': [],
        'train_recall_macro': [], 'val_recall_macro': [],
        'train_auc_macro': [], 'val_auc_macro': [],
        'train_auc_weighted': [], 'val_auc_weighted': [],
        'learning_rate': []
    }
    
    best_f1 = 0
    best_epoch = 0
    
    print(f"\n🎯 开始训练 ({num_epochs} epochs)")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # 训练
        train_loss, train_metrics = train_epoch_5class(
            model, train_loader, criterion, optimizer, device, scaler, class_names
        )
        
        # 验证
        val_loss, val_metrics = validate_epoch_5class(
            model, test_loader, criterion, device, class_names
        )
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['train_f1_macro'].append(train_metrics['f1_macro'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        history['train_f1_weighted'].append(train_metrics['f1_weighted'])
        history['val_f1_weighted'].append(val_metrics['f1_weighted'])
        history['train_precision_macro'].append(train_metrics['precision_macro'])
        history['val_precision_macro'].append(val_metrics['precision_macro'])
        history['train_recall_macro'].append(train_metrics['recall_macro'])
        history['val_recall_macro'].append(val_metrics['recall_macro'])
        history['train_auc_macro'].append(train_metrics['auc_macro'])
        history['val_auc_macro'].append(val_metrics['auc_macro'])
        history['train_auc_weighted'].append(train_metrics['auc_weighted'])
        history['val_auc_weighted'].append(val_metrics['auc_weighted'])
        history['learning_rate'].append(current_lr)
        
        # 打印结果
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1_macro']:.4f}")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_macro']:.4f}")
        print(f"学习率: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            best_epoch = epoch
            
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_f1': best_f1,
                'best_auc': val_metrics['auc_macro'],
                'best_accuracy': val_metrics['accuracy'],
                'class_names': class_names,
                'num_classes': num_classes
            }, os.path.join(output_dir, 'best_model_5class.pth'))
            
            print(f"✅ 保存最佳模型 (F1: {best_f1:.4f})")
    
    # 保存训练历史
    with open(os.path.join(output_dir, 'history_5class.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # 绘制训练历史
    plot_training_history_5class(history, output_dir)
    
    print(f"\n🎉 训练完成！")
    print(f"📊 最佳F1分数: {best_f1:.4f} (Epoch {best_epoch+1})")
    print(f"📁 模型保存在: {output_dir}")
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description='5分类多模态模型训练')
    parser.add_argument('--data_path', type=str, default='5centers_multi_5class', help='数据路径')
    parser.add_argument('--output_dir', type=str, default='cnn_training_5class', help='输出目录')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--embed_dim', type=int, default=512, help='嵌入维度')
    parser.add_argument('--clinical_dim', type=int, default=8, help='临床特征维度')
    parser.add_argument('--use_focal_loss', action='store_true', help='使用Focal Loss')
    parser.add_argument('--use_label_smoothing', action='store_true', help='使用标签平滑')
    parser.add_argument('--use_causal_adjustment', action='store_true', help='使用因果调整')
    parser.add_argument('--use_text_contrastive', action='store_true', help='使用文本对比学习')
    
    args = parser.parse_args()
    
    train_5class_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        embed_dim=args.embed_dim,
        clinical_dim=args.clinical_dim,
        use_focal_loss=args.use_focal_loss,
        use_label_smoothing=args.use_label_smoothing,
        use_causal_adjustment=args.use_causal_adjustment,
        use_text_contrastive=args.use_text_contrastive
    )

if __name__ == "__main__":
    main()
