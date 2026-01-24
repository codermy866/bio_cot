#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识蒸馏训练脚本
将大模型（教师）的知识传递给小模型（学生）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import time
from datetime import datetime
import os
import json
import sys
from tqdm import tqdm

from models.cnn_multimodal_model import CNNMultimodalTransformer
from utils.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
from utils.advanced_clinical_metrics import calculate_clinical_metrics, calculate_calibration_metrics

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None


class KnowledgeDistillationLoss(nn.Module):
    """
    知识蒸馏损失函数
    
    损失 = α * KL(软标签_教师 || 软标签_学生) + (1-α) * CE(硬标签 || 学生预测)
    
    Args:
        temperature: 温度参数，控制软标签的平滑程度（T=1时接近硬标签，T越大越平滑）
        alpha: 蒸馏权重，平衡教师知识和真实标签
    """
    
    def __init__(self, temperature=3.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, targets):
        """
        Args:
            student_logits: 学生模型的logits [B, num_classes]
            teacher_logits: 教师模型的logits [B, num_classes]
            targets: 真实标签 [B]
        """
        # 软标签蒸馏损失（KL散度）
        # 教师软标签：softmax(teacher_logits / T)
        # 学生软标签：log_softmax(student_logits / T)
        soft_loss = self.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)  # 乘以T^2以保持梯度尺度
        
        # 硬标签损失（标准交叉熵）
        hard_loss = self.ce_loss(student_logits, targets)
        
        # 组合损失
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, soft_loss, hard_loss


def prepare_distillation_loaders(data_path='5centers_multi', batch_size=8, num_workers=0):
    """准备蒸馏训练的数据加载器"""
    print("🔄 准备蒸馏训练数据加载器...")
    
    class Args:
        def __init__(self, data_path):
            self.data_path = data_path
            self.input_size = 224
            self.oct_num_frames = 120
            self.oct_cache_dir = 'oct_cache_optimized'
            self.use_text_contrastive = False
            self.use_pretrained_backbones = True
            self.oct_points = 12
            self.oct_frames_per_point = 10
    
    args = Args(data_path)
    
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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


def train_distillation(
    teacher_model_path: str,
    epochs=20,
    batch_size=8,
    learning_rate=2e-5,
    temperature=3.0,
    alpha=0.7,
    data_path='5centers_multi',
    output_dir='cnn_result_distilled'
):
    """
    知识蒸馏训练
    
    Args:
        teacher_model_path: 教师模型路径
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        temperature: 温度参数（建议3-5）
        alpha: 蒸馏权重（建议0.5-0.7）
        data_path: 数据路径
        output_dir: 输出目录
    """
    
    print("=" * 80)
    print("🎓 开始知识蒸馏训练")
    print("=" * 80)
    print(f"📚 教师模型: {teacher_model_path}")
    print(f"📊 蒸馏参数: temperature={temperature}, alpha={alpha}")
    print("=" * 80)
    
    # 准备数据
    train_loader, test_loader = prepare_distillation_loaders(data_path, batch_size)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 使用设备: {device}")
    
    # ========== 加载教师模型 ==========
    print("\n📚 加载教师模型...")
    teacher_model = CNNMultimodalTransformer(
        num_classes=2,
        embed_dim=1280,  # 大模型配置
        num_heads=20,
        dropout=0.1,
        clinical_dim=7,
        oct_num_frames=120,
        col_num_frames=3
    ).to(device)
    
    # 加载教师模型权重
    if os.path.exists(teacher_model_path):
        checkpoint = torch.load(teacher_model_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                teacher_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                teacher_model.load_state_dict(checkpoint['state_dict'])
            else:
                teacher_model.load_state_dict(checkpoint)
        else:
            teacher_model.load_state_dict(checkpoint)
        print(f"✅ 教师模型加载成功: {teacher_model_path}")
    else:
        raise FileNotFoundError(f"教师模型文件不存在: {teacher_model_path}")
    
    # 冻结教师模型
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"📊 教师模型参数量: {teacher_params/1e6:.2f}M")
    
    # ========== 初始化学生模型 ==========
    print("\n📖 初始化学生模型...")
    student_model = CNNMultimodalTransformer(
        num_classes=2,
        embed_dim=768,  # 小模型配置（比教师的1280小）
        num_heads=12,   # 比教师的20小
        dropout=0.2,   # 学生模型使用稍高的dropout
        clinical_dim=7,
        oct_num_frames=120,
        col_num_frames=3
    ).to(device)
    
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"📊 学生模型参数量: {student_params/1e6:.2f}M")
    print(f"📉 压缩比: {(1 - student_params/teacher_params)*100:.1f}%")
    
    # ========== 优化器和调度器 ==========
    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=learning_rate,
        weight_decay=3e-4,
        betas=(0.9, 0.999)
    )
    
    # 学习率warmup + CosineAnnealing
    warmup_epochs = max(1, epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # ========== 损失函数 ==========
    criterion = KnowledgeDistillationLoss(temperature=temperature, alpha=alpha)
    
    # ========== 混合精度 ==========
    scaler = GradScaler()
    
    # ========== 训练历史 ==========
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'distill_loss': [],
        'hard_loss': []
    }
    
    best_val_auc = 0
    best_epoch = 0
    
    # ========== 训练循环 ==========
    print("\n🚀 开始蒸馏训练...")
    print("=" * 80)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # ========== 训练阶段 ==========
        student_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        distill_loss_epoch = 0
        hard_loss_epoch = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", file=sys.stdout)
        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, dict):
                oct_images = batch.get('oct_images')
                col_images = batch.get('col_images')
                clinical_features = batch.get('clinical_features') or batch.get('clinical')
                labels = batch.get('label') or batch.get('labels')
            else:
                if len(batch) == 5:
                    oct_images, col_images, clinical_features, labels, _ = batch
                else:
                    oct_images, col_images, clinical_features, labels = batch
            
            oct_images = oct_images.to(device)
            col_images = col_images.to(device)
            clinical_features = clinical_features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                # 学生模型前向传播
                student_outputs = student_model(oct_images, col_images, clinical_features)
                
                # 教师模型前向传播（不计算梯度）
                with torch.no_grad():
                    teacher_outputs = teacher_model(oct_images, col_images, clinical_features)
                
                # 计算蒸馏损失
                loss, distill_loss, hard_loss = criterion(student_outputs, teacher_outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 统计
            train_loss += loss.item()
            distill_loss_epoch += distill_loss.item()
            hard_loss_epoch += hard_loss.item()
            
            _, predicted = torch.max(student_outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 更新进度条
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Distill': f'{distill_loss.item():.4f}',
                    'Hard': f'{hard_loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.1f}%'
                })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        distill_loss_epoch /= len(train_loader)
        hard_loss_epoch /= len(train_loader)
        
        # ========== 验证阶段 ==========
        student_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="验证中", leave=False):
                if isinstance(batch, dict):
                    oct_images = batch.get('oct_images')
                    col_images = batch.get('col_images')
                    clinical_features = batch.get('clinical_features') or batch.get('clinical')
                    labels = batch.get('label') or batch.get('labels')
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
                    outputs = student_model(oct_images, col_images, clinical_features)
                    loss = F.cross_entropy(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                probs = F.softmax(outputs, dim=1)
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(test_loader)
        val_acc = 100. * val_correct / val_total
        
        # 计算指标
        val_auc = roc_auc_score(all_labels, all_probs)
        val_f1 = f1_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds)
        val_recall = recall_score(all_labels, all_preds)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        
        # 临床指标
        clinical_metrics = calculate_clinical_metrics(all_labels, all_probs)
        calibration_metrics = calculate_calibration_metrics(all_labels, all_probs)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 保存历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['distill_loss'].append(distill_loss_epoch)
        history['hard_loss'].append(hard_loss_epoch)
        
        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_val_auc,
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"\n💾 保存最佳模型 (Epoch {best_epoch}, AUC: {best_val_auc:.4f})")
        
        # 打印结果
        epoch_time = time.time() - epoch_start
        print("\n" + "=" * 80)
        print(f"📊 Epoch [{epoch+1}/{epochs}] 蒸馏训练结果")
        print("-" * 80)
        print(f"训练集:")
        print(f"  Loss: {train_loss:.4f}  |  Accuracy: {train_acc:.2f}%")
        print(f"  蒸馏损失: {distill_loss_epoch:.4f}  |  硬标签损失: {hard_loss_epoch:.4f}")
        print(f"验证集:")
        print(f"  Loss:     {val_loss:.4f}")
        print(f"  Accuracy: {val_acc:.2f}%")
        print(f"  AUC:      {val_auc:.4f}")
        print(f"  F1-Score: {val_f1:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall:    {val_recall:.4f}")
        print(f"其他信息:")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Epoch Time: {epoch_time:.1f}s")
        if val_auc == best_val_auc:
            print(f"  ⭐ 当前最佳模型 (AUC: {val_auc:.4f})")
        print("=" * 80 + "\n")
        
        # 保存指标
        metrics_file = os.path.join(output_dir, 'metrics.json')
        metrics_dict = {
            'epoch': epoch + 1,
            'train': {
                'loss': float(train_loss),
                'accuracy': float(train_acc),
                'distill_loss': float(distill_loss_epoch),
                'hard_loss': float(hard_loss_epoch)
            },
            'val': {
                'loss': float(val_loss),
                'accuracy': float(val_acc),
                'auc': float(val_auc),
                'f1': float(val_f1),
                'precision': float(val_precision),
                'recall': float(val_recall),
                'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
                'clinical_metrics': clinical_metrics,
                'calibration': calibration_metrics
            },
            'best_auc': float(best_val_auc),
            'best_epoch': int(best_epoch),
            'learning_rate': float(current_lr),
            'distillation_params': {
                'temperature': float(temperature),
                'alpha': float(alpha)
            }
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
    
    # 训练完成
    print("\n" + "=" * 80)
    print("✅ 蒸馏训练完成！")
    print("=" * 80)
    print(f"📈 最终结果:")
    print(f"  最佳验证AUC: {best_val_auc:.4f} (Epoch {best_epoch})")
    print(f"  模型压缩比: {(1 - student_params/teacher_params)*100:.1f}%")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='知识蒸馏训练')
    parser.add_argument('--teacher_model', type=str, required=True,
                        help='教师模型路径')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='学习率')
    parser.add_argument('--temperature', type=float, default=3.0,
                        help='温度参数')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='蒸馏权重')
    parser.add_argument('--data_path', type=str, default='5centers_multi',
                        help='数据路径')
    parser.add_argument('--output_dir', type=str, default='cnn_result_distilled',
                        help='输出目录')
    
    args = parser.parse_args()
    
    train_distillation(
        teacher_model_path=args.teacher_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        alpha=args.alpha,
        data_path=args.data_path,
        output_dir=args.output_dir
    )

