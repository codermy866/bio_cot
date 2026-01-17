#!/usr/bin/env python3
"""
BIDA Training Script
Bio-Invariant Distributional Anchoring for Cross-Center Generalization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]  # experiments/exp2_bida -> experiments -> project_root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.enhanced_multimodal_dataset import build_enhanced_dataset
from src.models.bida.bida_model import BIDAModel


class Args:
    """训练参数"""
    def __init__(self):
        self.data_path = '/data2/hmy/5Center_datas/5centers_multi_leave_centers_out'
        self.batch_size = 80  # 从48增加到80，充分利用显存（还有39GB可用，约80%）
        self.num_workers = 12  # 从8增加到12，加快数据加载
        self.num_epochs = 50
        self.learning_rate = 2e-4  # 从1.5e-4增加到2e-4，加快收敛
        self.weight_decay = 1e-4  # 从2e-4降低到1e-4，减少正则化
        self.gradient_accumulation_steps = 1  # 梯度累积步数（如果batch_size太大可以增加）
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'  # 使用cuda:1避免显存冲突
        self.num_classes = 2
        self.num_centers = 5
        self.embed_dim = 768
        
        # 数据集参数（需要与数据集兼容）
        self.input_size = 224
        self.oct_num_frames = 48
        self.oct_points = 12
        self.oct_frames_per_point = 10
        self.oct_cache_dir = None
        self.use_pretrained_backbones = True  # BIDA需要原始图像用于VLM处理
        
        # BIDA特定参数（进一步优化权重，使loss降到0.1左右）
        self.lambda_kl = 0.005  # L_dist权重：从0.01进一步降低到0.005
        self.lambda_orth = 0.01  # L_orth权重：从0.05降低到0.01（正交损失仍然偏高）
        self.lambda_adv = 0.05  # L_noise权重：从0.1降低到0.05
        
        # 输出目录
        self.output_dir = Path(__file__).parent / 'exp_bida'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # VLM设置
        self.vlm_model = "Qwen/Qwen2-VL-2B-Instruct"
        self.use_vlm = True


def train_epoch(model, train_loader, criterion, optimizer, scaler, args, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_dist_loss = 0.0
    total_orth_loss = 0.0
    total_noise_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        # 解析batch（根据数据集返回格式）
        oct_images = None
        col_images = None
        
        if isinstance(batch, dict):
            # 字典格式（使用pretrained_backbones时）
            oct_feat = batch['oct_features']
            colpo_feat = batch['colposcopy_features']
            clinical_feat = batch['clinical_features']
            labels = batch['label']
            center_labels = batch.get('center_id', torch.zeros(len(labels), dtype=torch.long))
            oct_images = batch.get('oct_images', None)  # 原始OCT图像
            col_images = batch.get('col_images', None)  # 原始阴道镜图像
            clinical_data = None
        elif len(batch) == 5:
            oct_feat, colpo_feat, clinical_feat, labels, center_labels = batch
            clinical_data = None
        elif len(batch) == 6:
            oct_feat, colpo_feat, clinical_feat, labels, center_labels, text_desc = batch
            clinical_data = None
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}, length: {len(batch) if hasattr(batch, '__len__') else 'N/A'}")
        
        # 构建clinical_data用于VLM（从clinical_feat提取）
        if clinical_data is None and clinical_feat is not None:
            batch_size = clinical_feat.size(0)
            clinical_data = {
                'hpv': [1 if clinical_feat[i, 1].item() > 0.5 else 0 for i in range(batch_size)],
                'tct': ['NILM'] * batch_size,  # 简化处理，实际可以从one-hot解码
                'age': [clinical_feat[i, 0].item() * 100 for i in range(batch_size)]
            }
        
        oct_feat = oct_feat.to(device)
        colpo_feat = colpo_feat.to(device)
        clinical_feat = clinical_feat.to(device)
        labels = labels.to(device)
        center_labels = center_labels.to(device)
        if oct_images is not None:
            oct_images = oct_images.to(device)
        if col_images is not None:
            col_images = col_images.to(device)
        
        # 梯度累积：每gradient_accumulation_steps步才更新一次
        if batch_idx % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        with autocast():
            # 前向传播
            outputs = model(
                oct_features=oct_feat,
                colpo_features=colpo_feat,
                clinical_features=clinical_feat,
                clinical_data=clinical_data,
                center_labels=center_labels,
                oct_images=oct_images,  # 传递给VLM用于图像+文本联合理解
                colposcopy_images=col_images,  # 传递给VLM（可选）
                return_loss_components=True
            )
            
            logits = outputs['logits']
            
            # 分类损失
            cls_loss = criterion(logits, labels)
            
            # BIDA损失组件
            loss_components = outputs['loss_components']
            dist_loss = loss_components['L_dist']
            orth_loss = loss_components['L_orth']
            noise_loss = loss_components['L_noise']
            
            # 总损失（除以梯度累积步数，因为会累积多次）
            total_loss_batch = (
                cls_loss +
                args.lambda_kl * dist_loss +
                args.lambda_orth * orth_loss +
                args.lambda_adv * noise_loss
            ) / args.gradient_accumulation_steps
        
        # 反向传播
        scaler.scale(total_loss_batch).backward()
        
        # 每gradient_accumulation_steps步更新一次
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            # 梯度裁剪，防止梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        
        # 统计（注意：total_loss_batch已经除以了gradient_accumulation_steps）
        total_loss += total_loss_batch.item() * args.gradient_accumulation_steps  # 恢复原始scale
        total_cls_loss += cls_loss.item()
        total_dist_loss += dist_loss.item()
        total_orth_loss += orth_loss.item()
        total_noise_loss += noise_loss.item()
        
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{total_loss_batch.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%',
            'Cls': f'{cls_loss.item():.4f}',
            'Dist': f'{dist_loss.item():.4f}',
            'Orth': f'{orth_loss.item():.4f}',
            'Noise': f'{noise_loss.item():.4f}'
        })
    
    return {
        'loss': total_loss / len(train_loader),
        'cls_loss': total_cls_loss / len(train_loader),
        'dist_loss': total_dist_loss / len(train_loader),
        'orth_loss': total_orth_loss / len(train_loader),
        'noise_loss': total_noise_loss / len(train_loader),
        'acc': 100. * correct / total
    }


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []  # 用于AUC计算
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            # 解析batch（与训练时相同）
            if isinstance(batch, dict):
                oct_feat = batch['oct_features']
                colpo_feat = batch['colposcopy_features']
                clinical_feat = batch['clinical_features']
                labels = batch['label']
                center_labels = batch.get('center_id', torch.zeros(len(labels), dtype=torch.long))
                clinical_data = None
            elif len(batch) == 5:
                oct_feat, colpo_feat, clinical_feat, labels, center_labels = batch
                clinical_data = None
            elif len(batch) == 6:
                oct_feat, colpo_feat, clinical_feat, labels, center_labels, text_desc = batch
                clinical_data = None
            elif len(batch) == 7:
                # 包含原始图像
                oct_feat, colpo_feat, clinical_feat, labels, center_labels, oct_imgs, col_imgs = batch
                oct_images = oct_imgs
                col_images = col_imgs
                clinical_data = None
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")
            
            # 构建clinical_data用于VLM
            if clinical_data is None and clinical_feat is not None:
                batch_size = clinical_feat.size(0)
                clinical_data = {
                    'hpv': [1 if clinical_feat[i, 1].item() > 0.5 else 0 for i in range(batch_size)],
                    'tct': ['NILM'] * batch_size,
                    'age': [clinical_feat[i, 0].item() * 100 for i in range(batch_size)]
                }
            
            oct_feat = oct_feat.to(device)
            colpo_feat = colpo_feat.to(device)
            clinical_feat = clinical_feat.to(device)
            labels = labels.to(device)
            center_labels = center_labels.to(device)
            
            # 验证时也需要传入原始图像（用于VLM）
            oct_images = None
            col_images = None
            if isinstance(batch, dict):
                oct_images = batch.get('oct_images', None)
                col_images = batch.get('col_images', None)
            elif len(batch) == 7:
                oct_images = oct_imgs
                col_images = col_imgs
            
            if oct_images is not None:
                oct_images = oct_images.to(device)
            if col_images is not None:
                col_images = col_images.to(device)
            
            with autocast():
                outputs = model(
                    oct_features=oct_feat,
                    colpo_features=colpo_feat,
                    clinical_features=clinical_feat,
                    clinical_data=clinical_data,
                    center_labels=center_labels,
                    oct_images=oct_images,  # 传递给VLM
                    colposcopy_images=col_images,  # 传递给VLM
                    return_loss_components=False
                )
                
                logits = outputs['logits']
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 保存预测类别（用于accuracy）
            all_preds.extend(predicted.cpu().numpy())
            # 保存真实标签
            all_labels.extend(labels.cpu().numpy())
            
            # 保存概率（用于AUC计算）
            probs = torch.softmax(logits, dim=1)  # [B, num_classes]
            all_probs.extend(probs[:, 1].cpu().numpy())  # 正类概率
    
    # 计算AUC（使用概率，不是类别）
    from sklearn.metrics import roc_auc_score
    try:
        if len(set(all_labels)) < 2:
            # 如果只有一个类别，无法计算AUC
            auc = 0.0
            print(f"⚠️ 警告：验证集只有一个类别，无法计算AUC")
        else:
            auc = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        print(f"⚠️ AUC计算失败: {e}")
        auc = 0.0
    
    return {
        'loss': total_loss / len(val_loader),
        'acc': 100. * correct / total,
        'auc': auc
    }


def main():
    parser = argparse.ArgumentParser(description='BIDA Training')
    parser.add_argument('--data_path', type=str, default='/data2/hmy/5Center_datas/5centers_multi_leave_centers_out')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lambda_kl', type=float, default=0.1)
    parser.add_argument('--lambda_orth', type=float, default=0.1)
    parser.add_argument('--lambda_adv', type=float, default=0.1)
    args = parser.parse_args()
    
    # 创建Args对象
    train_args = Args()
    train_args.data_path = args.data_path
    train_args.batch_size = args.batch_size
    train_args.num_epochs = args.num_epochs
    train_args.device = 'cuda:1'  # 强制使用cuda:1
    train_args.learning_rate = args.learning_rate
    train_args.lambda_kl = args.lambda_kl
    train_args.lambda_orth = args.lambda_orth
    train_args.lambda_adv = args.lambda_adv
    
    device = torch.device(train_args.device)
    print(f"🚀 Using device: {device}")
    
    # 加载数据集
    print("📂 Loading datasets...")
    train_dataset = build_enhanced_dataset(is_train='train', args=train_args)
    val_dataset = build_enhanced_dataset(is_train='val', args=train_args)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_args.batch_size,
        shuffle=True,
        num_workers=train_args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_args.batch_size,
        shuffle=False,
        num_workers=train_args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    print("🏗️ Creating BIDA model...")
    model = BIDAModel(
        embed_dim=train_args.embed_dim,
        num_classes=train_args.num_classes,
        num_centers=train_args.num_centers,
        vlm_model=train_args.vlm_model
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_args.learning_rate,
        weight_decay=train_args.weight_decay
    )
    scaler = GradScaler()
    
    # 学习率调度器（使用warmup + cosine annealing）
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    
    # Warmup阶段：前5个epoch线性增加学习率
    warmup_epochs = 5
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    # Cosine annealing阶段：剩余epoch使用cosine衰减
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_args.num_epochs - warmup_epochs,
        eta_min=1e-6  # 最小学习率
    )
    
    # 组合调度器
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    # 训练日志
    log_file = train_args.log_dir / f'train_bida_bs{train_args.batch_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    best_auc = 0.0
    best_epoch = 0
    
    print(f"📝 Training log: {log_file}")
    print("=" * 80)
    
    # 训练循环
    for epoch in range(1, train_args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{train_args.num_epochs}")
        print("-" * 80)
        
        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, train_args, device)
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录日志
        log_msg = (
            f"Epoch {epoch}: "
            f"Train Loss={train_metrics['loss']:.4f}, Train Acc={train_metrics['acc']:.2f}%, "
            f"Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['acc']:.2f}%, Val AUC={val_metrics['auc']:.4f}\n"
        )
        print(log_msg)
        with open(log_file, 'a') as f:
            f.write(log_msg)
        
        # 保存最佳模型
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
            }, train_args.output_dir / 'best_model.pth')
            print(f"✅ New best model saved! AUC: {best_auc:.4f}")
    
    print("=" * 80)
    print(f"🎉 Training completed! Best AUC: {best_auc:.4f} at epoch {best_epoch}")


if __name__ == '__main__':
    main()

