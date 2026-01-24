#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练VLM增强的因果贝叶斯CLIP
针对A6000 48GB优化配置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
import argparse
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

try:
    from vlm_enhanced_causal_clip import VLMEnhancedCausalBayesianCLIP
except ImportError:
    print("警告: vlm_enhanced_causal_clip.py未找到，使用基础版本")
    from enhanced_causal_clip import EnhancedCausalBayesianCLIP as VLMEnhancedCausalBayesianCLIP

# 导入数据加载器（需要根据实际情况调整）
# from your_data_loader import YourDataset


class FocalLoss(nn.Module):
    """Focal Loss"""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    scaler,
    device,
    epoch,
    use_amp=True,
    max_grad_norm=1.0
):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # 数据移动到GPU
        oct_images = batch['oct_images'].to(device)
        colposcopy_images = batch['colposcopy_images'].to(device)
        clinical_features = batch['clinical_features'].to(device)
        labels = batch['labels'].to(device)
        
        # 混合精度训练
        with autocast(enabled=use_amp):
            outputs = model(
                oct_images=oct_images,
                colposcopy_images=colposcopy_images,
                clinical_features=clinical_features
            )
            
            logits = outputs['logits']
            loss = criterion(logits, labels)
            
            # 添加KL损失（如果有）
            if 'kl_loss' in outputs:
                loss = loss + outputs['kl_loss']
            
            # 添加因果损失（如果有）
            if 'causal_loss' in outputs:
                loss = loss + outputs['causal_loss']
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度裁剪
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # 统计
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(
    model,
    dataloader,
    criterion,
    device,
    use_amp=True
):
    """验证"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            oct_images = batch['oct_images'].to(device)
            colposcopy_images = batch['colposcopy_images'].to(device)
            clinical_features = batch['clinical_features'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast(enabled=use_amp):
                outputs = model(
                    oct_images=oct_images,
                    colposcopy_images=colposcopy_images,
                    clinical_features=clinical_features
                )
                
                logits = outputs['logits']
                loss = criterion(logits, labels)
            
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }


def main():
    parser = argparse.ArgumentParser(description='VLM增强因果CLIP训练')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='/data2/hmy/5Center_datas/5centers_multi_leave_centers_out', help='数据路径')
    parser.add_argument('--output_dir', type=str, default='vlm_causal_clip_results', help='输出目录')
    
    # 模型参数
    parser.add_argument('--vlm_model', type=str, default='Qwen/Qwen2-VL-2B-Instruct', 
                       choices=['Qwen/Qwen2-VL-2B-Instruct', 'Qwen/Qwen2-VL-7B-Instruct'],
                       help='VLM模型名称')
    parser.add_argument('--use_medical_kb', action='store_true', help='使用医学知识库')
    parser.add_argument('--use_vlm_guidance', action='store_true', default=True, help='使用VLM引导')
    parser.add_argument('--use_report_generation', action='store_true', help='生成诊断报告')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=10, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='学习率（VLM部分）')
    parser.add_argument('--learning_rate_other', type=float, default=3e-4, help='其他部分学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
    
    # 优化参数
    parser.add_argument('--use_amp', action='store_true', default=True, help='使用混合精度')
    parser.add_argument('--freeze_vlm', action='store_true', help='冻结VLM参数')
    parser.add_argument('--vlm_trainable_layers', type=int, default=2, help='VLM可训练层数')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='梯度检查点')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='最大梯度范数')
    
    # 损失参数
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss gamma')
    parser.add_argument('--label_smoothing', type=float, default=0.01, help='标签平滑')
    parser.add_argument('--kl_weight', type=float, default=0.001, help='KL损失权重')
    parser.add_argument('--causal_loss_weight', type=float, default=0.001, help='因果损失权重')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载workers')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据 - 支持Leave-Centers-Out数据集结构
    print("📥 加载数据集...")
    from src.data.enhanced_multimodal_dataset import build_enhanced_dataset
    
    # 创建Args对象用于数据加载
    class DataArgs:
        def __init__(self):
            self.data_path = args.data_path
            self.input_size = 224
            self.oct_num_frames = 48
            self.col_num_frames = 3
            self.oct_cache_dir = 'oct_cache_optimized'
            self.use_text_contrastive = False
            self.oct_points = 12
            self.oct_frames_per_point = 10
            self.use_pretrained_backbones = False
            self.cache_oct_features = False
    
    data_args = DataArgs()
    
    # 加载训练集和内部验证集
    train_dataset = build_enhanced_dataset('train', data_args, use_external_test=False)
    val_dataset = build_enhanced_dataset('val', data_args, use_external_test=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"✅ 训练集: {len(train_dataset)} 样本")
    print(f"✅ 验证集: {len(val_dataset)} 样本")
    
    # 创建模型
    print(f"创建模型: {args.vlm_model}")
    model = VLMEnhancedCausalBayesianCLIP(
        embed_dim=768,
        clinical_dim=7,
        num_classes=2,
        vlm_model_name=args.vlm_model,
        use_medical_kb=args.use_medical_kb,
        use_vlm_guidance=args.use_vlm_guidance,
        use_report_generation=args.use_report_generation
    ).to(device)
    
    # 冻结VLM（如果指定）
    if args.freeze_vlm:
        print(f"冻结VLM参数（除了最后{args.vlm_trainable_layers}层）")
        for param in model.vlm_encoder.vlm.parameters():
            param.requires_grad = False
        
        # 解冻最后N层
        if hasattr(model.vlm_encoder.vlm, 'vision_model'):
            for layer in list(model.vlm_encoder.vlm.vision_model.layers[-args.vlm_trainable_layers:]):
                for param in layer.parameters():
                    param.requires_grad = True
    
    # 梯度检查点
    if args.gradient_checkpointing and hasattr(model.vlm_encoder.vlm, 'gradient_checkpointing_enable'):
        print("启用梯度检查点")
        model.vlm_encoder.vlm.gradient_checkpointing_enable()
    
    # 优化器（不同参数组使用不同学习率）
    vlm_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'vlm_encoder' in name:
                vlm_params.append(param)
            else:
                other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': vlm_params, 'lr': args.learning_rate},
        {'params': other_params, 'lr': args.learning_rate_other}
    ], weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )
    
    # 损失函数
    criterion = FocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    
    # 混合精度
    scaler = GradScaler()
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_f1': []
    }
    
    best_auc = 0.0
    
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)
    print(f"VLM模型: {args.vlm_model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"学习率: VLM={args.learning_rate}, 其他={args.learning_rate_other}")
    print(f"混合精度: {args.use_amp}")
    print("=" * 80 + "\n")
    
    # 训练循环
    for epoch in range(1, args.num_epochs + 1):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, epoch, use_amp=args.use_amp, max_grad_norm=args.max_grad_norm
        )
        
        # 验证
        val_results = validate(model, val_loader, criterion, device, use_amp=args.use_amp)
        
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
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_results['loss']:.4f}, Val Acc: {val_results['accuracy']*100:.2f}%")
        print(f"Val AUC: {val_results['auc']:.4f}, Val F1: {val_results['f1']:.4f}")
        
        # 保存最佳模型
        if val_results['auc'] > best_auc:
            best_auc = val_results['auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'args': vars(args)
            }, output_dir / 'best_model.pth')
            print(f"✅ 保存最佳模型 (AUC: {best_auc:.4f})")
        
        # 保存历史
        with open(output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print(f"最佳AUC: {best_auc:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

