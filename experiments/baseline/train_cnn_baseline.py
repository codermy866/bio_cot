#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的2分类训练脚本
目标：AUC 0.85
包含：Focal Loss、数据增强、混合精度训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report
import time
from datetime import datetime
import os
import json
import sys
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate

# ---- Safe collate (module-level, picklable for DataLoader workers) ----
def _detach_any(x):
    if torch.is_tensor(x):
        return x.detach()
    if isinstance(x, dict):
        return {k: _detach_any(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_detach_any(v) for v in x)
    return x

def safe_collate(batch):
    # detach all tensors then use default_collate to keep shapes consistent
    return default_collate([_detach_any(b) for b in batch])

# 添加项目根目录到路径
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.cnn_multimodal_model import CNNMultimodalTransformer
try:
    from models.SwinT.swin_multimodal_model import SwinTMultimodalTransformer
except Exception:
    SwinTMultimodalTransformer = None
try:
    from models.ViT.vit_multimodal_model import ViTMultimodalTransformer
except Exception:
    ViTMultimodalTransformer = None
try:
    from models.MedicalViT.medical_vit_multimodal_model import MedicalViTMultimodalTransformer
except Exception:
    MedicalViTMultimodalTransformer = None
from utils.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
from utils.enhanced_oct_processing import EnhancedOCTProcessor, OCTFeatureFusion
from utils.advanced_clinical_metrics import calculate_clinical_metrics, calculate_calibration_metrics

# 确保输出实时刷新
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None


class FocalLoss(nn.Module):
    """Focal Loss with Label Smoothing for class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
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


def prepare_optimized_loaders(
    data_path='5centers_multi',
    batch_size=8,
    num_workers=8,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    oct_frames: int = 32,
    input_size: int = 192,
    oct_cache_dir: str = 'oct_cache_optimized',
):
    """准备优化的数据加载器"""
    print("🔄 准备优化的数据加载器...")
    
    class Args:
        def __init__(self, data_path):
            self.data_path = data_path
            self.input_size = input_size
            self.oct_num_frames = oct_frames
            self.oct_cache_dir = oct_cache_dir
            self.use_text_contrastive = False
            self.use_pretrained_backbones = True
            self.oct_points = 12
            self.oct_frames_per_point = 10
    
    args = Args(data_path)
    
    # 使用模块级safe_collate
    
    # 训练集
    train_dataset = EnhancedMultimodalCervicalDataset(
        root=os.path.join(data_path, 'train'),
        is_train='train',
        args=args,
        transform=None,
        use_enhanced_oct=True,
        cache_oct_features=True
    )
    
    # 测试集
    test_dataset = EnhancedMultimodalCervicalDataset(
        root=os.path.join(data_path, 'test'),
        is_train='test',
        args=args,
        transform=None,
        use_enhanced_oct=True,
        cache_oct_features=True
    )
    
    print(f"✅ 数据加载完成")
    print(f"  训练样本: {len(train_dataset)}")
    print(f"  测试样本: {len(test_dataset)}")
    
    # 计算类别权重
    import pandas as pd
    train_df = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))
    labels = train_df['label'].values
    class_counts = np.bincount(labels)
    
    # 计算权重
    total = class_counts.sum()
    class_weights = total / (len(class_counts) * class_counts)
    class_weights = class_weights / class_weights.sum()
    
    print(f"  类别分布: {dict(enumerate(class_counts))}")
    print(f"  类别权重: {dict(enumerate(class_weights))}")
    
    # 移除加权采样器，使用普通shuffle（避免过拟合）
    # weights = class_weights[labels]
    # sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 使用普通shuffle替代加权采样器
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=safe_collate,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=safe_collate,
    )
    
    return train_loader, test_loader


def train_optimized_2class(
    epochs=20,
    batch_size=8,
    learning_rate=3e-5,  # 进一步降低学习率，配合更强的正则化
    data_path='5centers_multi',
    output_dir='cnn_training_optimized',
    backbone: str = 'cnn',  # 'cnn', 'swin_t', 'vit', 'medical_vit'
    swin_pretrained: bool = True,
    vit_model: str = 'vit_base_patch16_224',
    vit_pretrained: bool = True,
    oct_frames: int = 32,
    input_size: int = 192,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    oct_cache_dir: str = 'oct_cache_optimized',
):
    """训练优化的2分类模型"""
    
    print("🎯 开始优化2分类模型训练")
    print("=" * 60)
    
    # 准备数据
    train_loader, test_loader = prepare_optimized_loaders(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        oct_frames=oct_frames,
        input_size=input_size,
        oct_cache_dir=oct_cache_dir,
    )
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 使用设备: {device}")
    
    if backbone == 'swin_t':
        if SwinTMultimodalTransformer is None:
            raise ImportError('未找到 SwinTMultimodalTransformer，请确保 models/SwinT 已创建并安装 timm')
        model = SwinTMultimodalTransformer(
            num_classes=2,
            embed_dim=768,   # Swin-T 默认嵌入维度
            num_heads=12,    # 768/64=12
            dropout=0.25,    # 从0.2增加到0.25，增强正则化
            clinical_dim=7,
            oct_num_frames=oct_frames,  # 使用传入的oct_frames参数
            col_num_frames=3,
            swin_name='swin_tiny_patch4_window7_224',
            pretrained=swin_pretrained,
            input_size=input_size,
            use_frame_attention=True,  # 启用帧注意力以更好地利用多帧信息
        ).to(device)
    elif backbone == 'vit':
        if ViTMultimodalTransformer is None:
            raise ImportError('未找到 ViTMultimodalTransformer，请确保 models/ViT 已创建并安装 timm')
        model = ViTMultimodalTransformer(
            num_classes=2,
            embed_dim=768,
            num_heads=12,
            dropout=0.2,
            clinical_dim=7,
            oct_num_frames=oct_frames,
            col_num_frames=3,
            vit_name=vit_model,
            pretrained=vit_pretrained,
            input_size=input_size,
            use_frame_attention=False,
        ).to(device)
    elif backbone == 'medical_vit':
        if MedicalViTMultimodalTransformer is None:
            raise ImportError('未找到 MedicalViTMultimodalTransformer，请确保 models/MedicalViT 已创建并安装 timm')
        model = MedicalViTMultimodalTransformer(
            num_classes=2,
            embed_dim=768,
            num_heads=12,
            dropout=0.2,
            clinical_dim=7,
            oct_num_frames=oct_frames,
            col_num_frames=3,
            vit_name=vit_model,
            pretrained=vit_pretrained,
            input_size=input_size,
            use_frame_attention=False,
            use_medical_pretrained=True,
        ).to(device)
    else:
        model = CNNMultimodalTransformer(
            num_classes=2,
            embed_dim=1280,  # 从1024增加到1280，进一步提升模型容量
            num_heads=20,    # 从16增加到20，匹配新的embed_dim（1280/64=20）
            dropout=0.3,     # 从0.1增加到0.3，大幅提升正则化以解决过拟合
            clinical_dim=7,
            oct_num_frames=oct_frames,
            col_num_frames=3
        ).to(device)
    
    print(f"📊 模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 优化器和调度器（增强优化版本）
    # 使用更精细的学习率策略（进一步降低以稳定训练）
    adjusted_lr = learning_rate * 0.5  # 降低50%以增强训练稳定性，避免NaN
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=adjusted_lr, 
        weight_decay=6e-4,  # 从5e-4增加到6e-4，进一步强化正则化
        betas=(0.9, 0.999),
        eps=1e-8
    )
    print(f"📈 调整后学习率: {adjusted_lr:.2e} (原始: {learning_rate:.2e})", flush=True)
    print(f"📈 正则化强度: weight_decay={6e-4:.2e}", flush=True)
    
    # 使用改进的学习率调度：Warmup + CosineAnnealing + 重启
    warmup_epochs = max(2, epochs // 8)  # 12.5%的epochs用于warmup（增加warmup）
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup阶段：线性增长
            return (epoch + 1) / warmup_epochs
        else:
            # CosineAnnealing阶段（带重启机制）
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            # 使用更平滑的余弦退火
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            # 添加最小学习率保持（避免过小）
            min_lr_ratio = 0.1
            return max(min_lr_ratio, cosine_factor)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 增强的Focal Loss（带Label Smoothing，更好的类别平衡）
    # 类别权重：{0: 0.325, 1: 0.675}，所以alpha=[0.325, 0.675]
    criterion = FocalLoss(
        alpha=torch.tensor([0.325, 0.675]).to(device), 
        gamma=3.0,  # 从2.5增加到3.0，更关注难样本
        label_smoothing=0.1  # 添加Label Smoothing，提升泛化能力
    )
    
    # 混合精度
    scaler = GradScaler()
    
    # 梯度裁剪（使训练更稳定，增加阈值以允许更大梯度）
    max_grad_norm = 2.0  # 从1.0增加到2.0，允许更大的梯度流动
    
    # EMA（指数移动平均）用于平滑Loss
    ema_decay = 0.99
    
    # 训练历史（包含学习率）
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'learning_rate': []
    }
    
    best_val_auc = 0
    best_epoch = 0
    
    # Early Stopping机制（防止过拟合，但给予更多机会）
    patience = 7  # 从5增加到7，给予更多epoch让模型找到更好的解
    patience_counter = 0
    min_delta = 0.0005  # 从0.001降低到0.0005，更敏感地检测改善
    
    # 过拟合监控
    max_gap = 0.25  # 训练准确率与验证准确率的最大允许差距（25%）
    overfitting_epochs = 0
    
    # 训练循环
    print("\n🚀 开始训练...", flush=True)
    print("=" * 60, flush=True)
    print(f"📊 过拟合监控: 最大允许差距={max_gap*100:.1f}%", flush=True)
    print(f"🛑 Early Stopping: patience={patience}, min_delta={min_delta:.4f}", flush=True)
    
    # EMA（指数移动平均）用于平滑Loss（每个epoch重置）
    ema_decay = 0.99
    
    for epoch in range(epochs):
        epoch_start = time.time()
        ema_train_loss = None  # 每个epoch重置EMA
        
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # 使用tqdm，但每N个batch打印一次详细信息
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, dict):
                oct_images = batch.get('oct_images')
                col_images = batch.get('col_images')
                clinical_features = batch['clinical_features'] if 'clinical_features' in batch else batch.get('clinical')
                labels = batch['label'] if 'label' in batch else batch.get('labels')
            else:
                # tuple/list
                if len(batch) == 5:
                    oct_images, col_images, clinical_features, labels, _ = batch
                else:
                    oct_images, col_images, clinical_features, labels = batch
            # 保留多帧输入，交由模型内部注意力聚合
            oct_images = oct_images.to(device)
            col_images = col_images.to(device)
            clinical_features = clinical_features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(oct_images, col_images, clinical_features)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            # 梯度裁剪（使训练更稳定）
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # 检查NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  警告: Epoch {epoch+1}, Batch {batch_idx}: Loss为NaN/Inf，跳过此batch")
                scaler.update()
                continue
            
            scaler.step(optimizer)
            scaler.update()
            
            # EMA平滑Loss
            loss_value = loss.item()
            
            # 再次检查loss_value是否为NaN/Inf
            if np.isnan(loss_value) or np.isinf(loss_value):
                print(f"⚠️  警告: Epoch {epoch+1}, Batch {batch_idx}: Loss值为NaN/Inf，跳过统计")
                continue
            if ema_train_loss is None:
                ema_train_loss = loss_value
            else:
                ema_train_loss = ema_decay * ema_train_loss + (1 - ema_decay) * loss_value
            
            train_loss += loss_value
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # 每10个batch更新一次进度条信息
            if (batch_idx + 1) % 10 == 0:
                current_acc = 100. * train_correct / train_total
                avg_loss = train_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'EMA': f'{ema_train_loss:.4f}',
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
                # 保留多帧输入，交由模型内部注意力聚合
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
                
                # AUC和预测计算
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        val_loss /= len(test_loader)
        val_acc = 100. * val_correct / val_total
        
        # 计算详细指标（包括临床指标）
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
            
            # 混淆矩阵
            cm = confusion_matrix(all_labels_array, val_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            # 计算临床指标
            clinical_metrics = calculate_clinical_metrics(
                y_true=all_labels_array,
                y_pred=val_pred,
                y_probs=all_probs_array
            )
            
            # 添加最优阈值信息
            clinical_metrics['optimal_threshold'] = float(optimal_threshold)
            clinical_metrics['youden_index'] = float(youden_index[best_idx])
            
            # 计算校准指标
            calibration_metrics = calculate_calibration_metrics(all_labels_array, all_probs_array)
            
        except Exception as e:
            val_auc = 0.5
            val_f1 = 0.0
            val_precision = 0.0
            val_recall = 0.0
            tn, fp, fn, tp = 0, 0, 0, 0
            clinical_metrics = {}
            calibration_metrics = {}
            print(f"Warning: 计算指标时出错: {e}", flush=True)
        
        # 记录（训练Loss使用EMA平滑后的值）
        # train_loss已在上面记录（使用EMA平滑）
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(float(val_f1))
        history['val_precision'].append(float(val_precision))
        history['val_recall'].append(float(val_recall))
        
        # 记录平滑后的训练Loss（使用EMA）
        if ema_train_loss is not None:
            history['train_loss'].append(ema_train_loss)
        else:
            history['train_loss'].append(train_loss)
        
        # 更新学习率
        current_lr = scheduler.get_last_lr()[0]
        history['learning_rate'].append(float(current_lr))
        scheduler.step()
        
        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
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
        
        epoch_time = time.time() - epoch_start
        
        # 详细打印每个epoch的指标
        print("\n" + "=" * 80, flush=True)
        print(f"📊 Epoch [{epoch+1}/{epochs}] 训练结果", flush=True)
        print("-" * 80, flush=True)
        print(f"训练集:", flush=True)
        print(f"  Loss: {train_loss:.4f}  |  Accuracy: {train_acc:.2f}%", flush=True)
        print(f"验证集:", flush=True)
        print(f"  Loss:     {val_loss:.4f}", flush=True)
        print(f"  Accuracy: {val_acc:.2f}%", flush=True)
        print(f"  AUC:      {val_auc:.4f}", flush=True)
        print(f"  F1-Score: {val_f1:.4f}", flush=True)
        print(f"  Precision: {val_precision:.4f}", flush=True)
        print(f"  Recall:    {val_recall:.4f}", flush=True)
        print(f"混淆矩阵:", flush=True)
        print(f"  TN: {tn:4d}  FP: {fp:4d}", flush=True)
        print(f"  FN: {fn:4d}  TP: {tp:4d}", flush=True)
        
        # 打印临床指标
        if clinical_metrics:
            print(f"\n临床指标:", flush=True)
            print(f"  灵敏度(Sensitivity): {clinical_metrics.get('sensitivity', 0):.4f}", flush=True)
            print(f"  特异度(Specificity): {clinical_metrics.get('specificity', 0):.4f}", flush=True)
            print(f"  PPV: {clinical_metrics.get('ppv', 0):.4f}", flush=True)
            print(f"  NPV: {clinical_metrics.get('npv', 0):.4f}", flush=True)
            print(f"  阳性似然比(LR+): {clinical_metrics.get('lr_plus', 0):.4f}", flush=True)
            print(f"  阴性似然比(LR-): {clinical_metrics.get('lr_minus', 0):.4f}", flush=True)
            print(f"  Youden指数: {clinical_metrics.get('youden_index', 0):.4f}", flush=True)
            print(f"  平衡准确率: {clinical_metrics.get('balanced_accuracy', 0):.4f}", flush=True)
            print(f"  MCC: {clinical_metrics.get('mcc', 0):.4f}", flush=True)
        
        if calibration_metrics:
            print(f"\n校准指标:", flush=True)
            print(f"  Brier Score: {calibration_metrics.get('brier_score', 0):.4f}", flush=True)
            print(f"  ECE: {calibration_metrics.get('ece', 0):.4f}", flush=True)
        # 过拟合监控
        train_val_gap = abs(train_acc - val_acc) / 100.0  # 转换为比例
        is_overfitting = train_val_gap > max_gap
        
        if is_overfitting:
            overfitting_epochs += 1
            print(f"⚠️  过拟合警告: 训练-验证差距={train_val_gap*100:.1f}% (阈值: {max_gap*100:.1f}%)", flush=True)
            print(f"   连续过拟合epoch数: {overfitting_epochs}", flush=True)
        else:
            overfitting_epochs = 0
        
        print(f"其他信息:", flush=True)
        print(f"  Learning Rate: {current_lr:.6f}", flush=True)
        print(f"  Epoch Time: {epoch_time:.1f}s", flush=True)
        print(f"  训练-验证准确率差距: {train_val_gap*100:.1f}%", flush=True)
        if val_auc == best_val_auc:
            print(f"  ⭐ 当前最佳模型 (AUC: {val_auc:.4f})", flush=True)
        print("=" * 80 + "\n", flush=True)
        
        # Early Stopping检查
        if val_auc > best_val_auc + min_delta:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n🛑 Early Stopping触发: 验证AUC连续{patience}个epoch未提升", flush=True)
                print(f"   最佳AUC: {best_val_auc:.4f} (Epoch {best_epoch})", flush=True)
                break
        
        # 严重过拟合检查
        if overfitting_epochs >= 3 and train_val_gap > max_gap * 1.5:
            print(f"\n🛑 严重过拟合检测: 连续{overfitting_epochs}个epoch过拟合严重", flush=True)
            print(f"   训练-验证差距: {train_val_gap*100:.1f}%", flush=True)
            print(f"   建议: 增加正则化或降低学习率", flush=True)
        
        # 实时保存指标到JSON（包含临床指标）
        # 定义递归函数来转换numpy类型为Python原生类型
        def convert_to_serializable(obj):
            """递归转换numpy类型为Python原生类型"""
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            # 优先检查numpy标量类型（包括float16）
            elif isinstance(obj, np.generic):
                if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8, np.int_)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16, np.float_)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                else:
                    # 兜底：使用item()方法
                    return obj.item() if hasattr(obj, 'item') else float(obj) if np.issubdtype(type(obj), np.floating) else int(obj)
            elif hasattr(obj, 'item') and 'numpy' in str(type(obj)):
                # numpy标量（可能不是np.generic的子类）
                return obj.item()
            elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None  # 将NaN和Inf转换为None
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8, np.int_)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16, np.float_)):
                return float(obj)
            else:
                return obj
        
        metrics_file = os.path.join(output_dir, 'metrics.json')
        metrics_dict = {
            'epoch': epoch + 1,
            'train': {
                'loss': float(train_loss) if not (np.isnan(train_loss) or np.isinf(train_loss)) else None,
                'accuracy': float(train_acc)
            },
            'val': {
                'loss': float(val_loss) if not (np.isnan(val_loss) or np.isinf(val_loss)) else None,
                'accuracy': float(val_acc),
                'auc': float(val_auc) if not (np.isnan(val_auc) or np.isinf(val_auc)) else None,
                'f1': float(val_f1) if not (np.isnan(val_f1) or np.isinf(val_f1)) else None,
                'precision': float(val_precision) if not (np.isnan(val_precision) or np.isinf(val_precision)) else None,
                'recall': float(val_recall) if not (np.isnan(val_recall) or np.isinf(val_recall)) else None,
                'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
                'clinical_metrics': convert_to_serializable(clinical_metrics) if clinical_metrics else {},
                'calibration': convert_to_serializable(calibration_metrics) if calibration_metrics else {}
            },
            'best_auc': float(best_val_auc),
            'best_epoch': int(best_epoch),
            'learning_rate': float(current_lr)
        }
        
        # 保存ROC曲线数据
        if clinical_metrics and 'roc_curve' in clinical_metrics:
            metrics_dict['val']['roc_curve'] = convert_to_serializable(clinical_metrics['roc_curve'])
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
    
    print("\n" + "=" * 80, flush=True)
    print(f"✅ 训练完成！", flush=True)
    print("-" * 80, flush=True)
    print(f"📈 最终结果:", flush=True)
    print(f"  最佳验证AUC: {best_val_auc:.4f} (Epoch {best_epoch})", flush=True)
    if history['val_f1']:
        print(f"  最佳验证F1: {max(history['val_f1']):.4f}", flush=True)
    if history['val_acc']:
        print(f"  最佳验证Accuracy: {max(history['val_acc']):.2f}%", flush=True)
    print("=" * 80 + "\n", flush=True)
    
    # 保存历史
    os.makedirs(output_dir, exist_ok=True)
    # 递归转换numpy类型为Python原生类型
    def convert_to_serializable(obj):
        """递归转换numpy类型为Python原生类型"""
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8, np.int_)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16, np.float_)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj.item() if hasattr(obj, 'item') else float(obj) if np.issubdtype(type(obj), np.floating) else int(obj)
        elif hasattr(obj, 'item') and 'numpy' in str(type(obj)):
            return obj.item()
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16, np.float_)):
            return float(obj)
        else:
            return obj
    
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(convert_to_serializable(history), f, indent=2)
    
    # 保存最终摘要
    summary = {
        'best_epoch': best_epoch,
        'best_val_auc': float(best_val_auc),
        'best_val_acc': float(max(history['val_acc'])) if history['val_acc'] else 0.0,
        'best_val_f1': float(max(history['val_f1'])) if history['val_f1'] else 0.0,
        'total_epochs': epochs,
        'final_train_acc': float(history['train_acc'][-1]) if history['train_acc'] else 0.0,
        'final_val_acc': float(history['val_acc'][-1]) if history['val_acc'] else 0.0,
        'final_val_auc': float(history['val_auc'][-1]) if history['val_auc'] else 0.0,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 训练历史已保存到: {os.path.join(output_dir, 'training_history.json')}", flush=True)
    print(f"💾 训练摘要已保存到: {os.path.join(output_dir, 'training_summary.json')}", flush=True)
    
    # 生成可视化图表
    print("\n" + "=" * 80, flush=True)
    print("📊 开始生成训练可视化图表...", flush=True)
    print("=" * 80, flush=True)
    
    try:
        import sys as sys_module
        # 使用模块级别的 os，不要重新导入
        vis_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualization')
        if os.path.exists(vis_path) and vis_path not in sys_module.path:
            sys_module.path.insert(0, vis_path)
        from generate_training_plots import generate_training_plots
        generate_training_plots(output_dir)
    except Exception as e:
        print(f"⚠️  生成基础可视化图表时出错: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    # Generate advanced visualization plots
    try:
        import sys as sys_module
        # 使用模块级别的 os，不要重新导入
        vis_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualization')
        if os.path.exists(vis_path) and vis_path not in sys_module.path:
            sys_module.path.insert(0, vis_path)
        from generate_advanced_visualizations import generate_all_advanced_visualizations
        generate_all_advanced_visualizations(output_dir)
    except Exception as e:
        print(f"⚠️  生成高级可视化图表时出错: {e}", flush=True)
        import traceback
        traceback.print_exc()
        print("   您可以稍后手动运行可视化脚本", flush=True)
    
    return model, history


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='5centers_multi')
    parser.add_argument('--epochs', type=int, default=20)  # 增加训练轮数，配合早停
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-5)  # 降低初始学习率，配合更强的正则化
    parser.add_argument('--output_dir', type=str, default='cnn_training_optimized')
    parser.add_argument('--backbone', type=str, default='cnn', choices=['cnn', 'swin_t'])
    parser.add_argument('--swin_no_pretrained', action='store_true')
    # new performance-related args
    parser.add_argument('--oct_frames', type=int, default=32)
    parser.add_argument('--input_size', type=int, default=192)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--persistent_workers', action='store_true', default=True)
    parser.add_argument('--oct_cache_dir', type=str, default='oct_cache_optimized')
    
    args = parser.parse_args()
    
    train_optimized_2class(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        data_path=args.data_path,
        output_dir=args.output_dir,
        backbone=args.backbone,
        swin_pretrained=(not args.swin_no_pretrained),
        oct_frames=args.oct_frames,
        input_size=args.input_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        oct_cache_dir=args.oct_cache_dir,
    )


if __name__ == "__main__":
    main()



