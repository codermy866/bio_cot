#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT模型在襄阳数据集上的训练脚本
只使用OCT图像（单模态）
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from xiangyang_dataset import XiangyangDataset
except ImportError:
    from src.data.xiangyang_dataset import XiangyangDataset

from src.models.bida.bio_cot_model import BioCOTModel
from src.models.bida.prior_net import build_clinical_vector
from src.utils.anti_overfitting import FocalLoss


class BioCOTXiangyangArgs:
    """Bio-COT训练参数"""
    def __init__(self):
        # 数据路径
        self.data_root = '/data2/hmy/5Center_datas/襄阳按点图片分类/襄阳按点图片分类'
        self.output_dir = Path(__file__).parent / 'results'
        self.checkpoint_dir = Path(__file__).parent / 'checkpoints'
        self.log_dir = Path(__file__).parent / 'logs'
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练参数
        self.device = 'cuda:1'
        self.num_epochs = 50
        self.batch_size = 16  # 减小batch size，因为Bio-COT模型较大
        self.num_workers = 4
        self.pin_memory = True
        
        # 优化器参数（平衡设置）
        self.learning_rate = 1.2e-4  # 略微降低学习率，提升稳定性
        self.weight_decay = 1e-4  # 标准正则化
        self.momentum = 0.9
        
        # 学习率调度（使用Warmup + CosineAnnealing）
        self.warmup_epochs = 3  # Warmup轮数
        self.use_warmup = True  # 启用Warmup
        
        # 梯度裁剪
        self.max_grad_norm = 1.0  # 梯度裁剪阈值
        
        # 数据增强
        self.input_size = 224
        self.use_augmentation = True
        
        # Early Stopping
        self.patience = 10
        self.min_delta = 0.001
        
        # Bio-COT模型参数
        self.embed_dim = 768
        self.num_classes = 2
        self.num_centers = 1  # 单中心数据集
        self.input_dim = 512
        self.use_vlm_encoder = False  # 不使用VLM，使用传统MLP编码器
        self.pretrained_student_prior = None  # 可选：预训练的Student Prior路径


def get_data_transforms(args, is_train=True):
    """获取数据增强变换"""
    if is_train and args.use_augmentation:
        return transforms.Compose([
            transforms.Resize((args.input_size + 32, args.input_size + 32)),
            transforms.RandomCrop(args.input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])


def extract_features_with_resnet50(images, device):
    """使用ResNet50提取图像特征"""
    from torchvision import models
    resnet = models.resnet50(pretrained=True).to(device)
    resnet.eval()
    resnet.fc = nn.Identity()  # 移除分类层
    
    with torch.no_grad():
        features = resnet(images)  # [B, 2048]
    
    # 投影到512维（匹配Bio-COT的input_dim）
    proj = nn.Linear(2048, 512).to(device)
    features_512 = proj(features)  # [B, 512]
    
    return features_512


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, args):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # EMA平滑loss（用于显示）
    ema_loss = None
    ema_decay = 0.9
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Train]')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播（模型内部处理特征提取）
        optimizer.zero_grad()
        logits = model(images)  # 直接输入图像
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        # 梯度裁剪（稳定训练）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()
        
        # 统计
        batch_loss = loss.item()
        running_loss += batch_loss
        
        # EMA平滑loss（用于进度条显示）
        if ema_loss is None:
            ema_loss = batch_loss
        else:
            ema_loss = ema_decay * ema_loss + (1 - ema_decay) * batch_loss
        
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 计算每个batch的阳性类预测情况
        pos_preds = (preds == 1).sum().item()
        pos_labels = (labels == 1).sum().item()
        pos_probs = torch.softmax(logits, dim=1)[:, 1].mean().item()
        
        # 更新进度条（显示EMA平滑后的loss）
        acc = accuracy_score(all_labels, all_preds)
        pbar.set_postfix({
            'loss': f'{ema_loss:.4f}',  # 显示平滑后的loss
            'raw_loss': f'{batch_loss:.4f}',  # 原始loss
            'acc': f'{acc:.4f}',
            'pos_pred': f'{pos_preds}/{pos_labels}',
            'pos_prob': f'{pos_probs:.3f}'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return {
        'loss': epoch_loss,
        'acc': epoch_acc
    }


def validate(model, dataloader, criterion, device, epoch, args):
    """验证"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Val]')
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播（模型内部处理特征提取）
            logits = model(images)  # 直接输入图像
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            # 使用调整后的阈值（降低阈值，更倾向于预测阳性）
            # 默认阈值0.5对不平衡数据不够，调整为0.35-0.4
            adjusted_threshold = 0.35
            preds = (probs[:, 1] > adjusted_threshold).long()  # 使用调整后的阈值
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
            # 更新进度条
            acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # 计算AUC
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except:
        epoch_auc = 0.0
    
    # 计算F1-score（用于Early Stopping）
    try:
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')  # 宏平均F1
        epoch_f1_pos = f1_score(all_labels, all_preds, pos_label=1)  # 阳性类F1
    except:
        epoch_f1 = 0.0
        epoch_f1_pos = 0.0
    
    return {
        'loss': epoch_loss,
        'acc': epoch_acc,
        'auc': epoch_auc,
        'f1': epoch_f1,
        'f1_pos': epoch_f1_pos,
        'preds': all_preds,
        'labels': all_labels,
        'probs': all_probs
    }


def plot_training_curves(history, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss曲线
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy曲线
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
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    # 使用英文标签避免中文字体问题
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative (0)', 'Positive (1)'], 
                yticklabels=['Negative (0)', 'Positive (1)'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存: {save_path}")


def main():
    args = BioCOTXiangyangArgs()
    
    # 设置设备
    if not torch.cuda.is_available():
        print("⚠️ CUDA不可用，使用CPU")
        args.device = 'cpu'
    else:
        device_idx = int(args.device.split(':')[1]) if ':' in args.device else 0
        if device_idx >= torch.cuda.device_count():
            print(f"⚠️ GPU {device_idx} 不可用，使用GPU 0")
            args.device = 'cuda:0'
    
    device = torch.device(args.device)
    print(f"✅ 使用设备: {args.device}")
    
    # 创建数据集
    print("📂 加载数据集...")
    train_dataset = XiangyangDataset(
        data_root=args.data_root,
        split='train',
        transform=get_data_transforms(args, is_train=True)
    )
    val_dataset = XiangyangDataset(
        data_root=args.data_root,
        split='val',
        transform=get_data_transforms(args, is_train=False)
    )
    
    # 创建加权采样器（平衡每个batch的类别分布）
    # 计算每个样本的权重（少数类样本权重更高）
    sample_weights = []
    for i in range(len(train_dataset)):
        label = train_dataset[i]['label'].item()
        # 少数类（阳性）权重更高
        weight = pos_weight_ratio if label == 1 else 1.0
        sample_weights.append(weight)
    
    # 创建加权采样器
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True  # 允许重复采样，确保每个batch都有少数类样本
    )
    print(f"📊 使用加权采样器: 阳性样本权重={pos_weight_ratio:.2f}x")
    
    # 创建数据加载器（使用加权采样器）
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,  # 使用加权采样器，不再使用shuffle
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
    
    # 计算类别权重（增强方法，处理严重不平衡）
    train_labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    pos_count = class_counts[1]
    neg_count = class_counts[0]
    
    # 方法1: 标准权重（作为基准）
    class_weights_standard = total_samples / (args.num_classes * class_counts)
    
    # 方法2: 更激进的权重（严重不平衡需要更强措施）
    # 直接使用逆频率比，但限制最大值避免过度
    imbalance_ratio = neg_count / pos_count  # 4.85:1
    pos_weight_ratio = min(imbalance_ratio * 0.8, 5.0)  # 更激进的权重，限制最大为5.0
    class_weights_aggressive = torch.FloatTensor([1.0, pos_weight_ratio]).to(device)
    
    # 使用更激进的权重
    class_weights = class_weights_aggressive
    print(f"📊 类别分布: 阴性={neg_count}, 阳性={pos_count} (比例 {imbalance_ratio:.2f}:1)")
    print(f"📊 标准权重: 阴性={class_weights_standard[0]:.4f}, 阳性={class_weights_standard[1]:.4f}")
    print(f"📊 激进权重: 阴性={class_weights[0].item():.4f}, 阳性={class_weights[1].item():.4f} (权重比 {pos_weight_ratio:.2f}:1)")
    
    # 创建简化模型：只使用BIDA的OCT编码器和分类器
    print("🏗️ 创建简化模型（仅OCT编码器 + 分类器）...")
    from src.models.bida.bida_model import DualHeadImageEncoder
    
    # ResNet50作为backbone提取特征
    from torchvision import models
    backbone = models.resnet50(pretrained=True)
    num_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    backbone = backbone.to(device)
    backbone.eval()  # 冻结backbone
    
    # 特征投影：ResNet50的2048维 -> 512维
    feature_proj = nn.Linear(num_features, 512).to(device)
    
    # BIDA的image_encoder（只使用causal_head）
    image_encoder = DualHeadImageEncoder(input_dim=512, embed_dim=args.embed_dim).to(device)
    
    # BIDA的classifier
    classifier = nn.Sequential(
        nn.Linear(args.embed_dim, args.embed_dim),
        nn.LayerNorm(args.embed_dim),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(args.embed_dim, args.embed_dim // 2),
        nn.LayerNorm(args.embed_dim // 2),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(args.embed_dim // 2, args.num_classes)
    ).to(device)
    
    # 组合成完整模型
    class SimpleOCTModel(nn.Module):
        def __init__(self, backbone, feature_proj, image_encoder, classifier):
            super().__init__()
            self.backbone = backbone
            self.feature_proj = feature_proj
            self.image_encoder = image_encoder
            self.classifier = classifier
        
        def forward(self, x):
            # ResNet50提取特征
            with torch.no_grad():
                features = self.backbone(x)  # [B, 2048]
            # 投影到512维
            features_512 = self.feature_proj(features)  # [B, 512]
            # BIDA image_encoder（只使用causal特征）
            z_causal, _ = self.image_encoder(features_512)  # [B, embed_dim], [B, embed_dim]
            # 分类
            logits = self.classifier(z_causal)  # [B, num_classes]
            return logits
    
    model = SimpleOCTModel(backbone, feature_proj, image_encoder, classifier)
    
    # 加载BIDA预训练权重
    bida_checkpoint_path = Path('/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp2_bida/exp_bida/best_model.pth')
    if bida_checkpoint_path.exists():
        print(f"📂 加载BIDA预训练权重: {bida_checkpoint_path}")
        try:
            import numpy.core.multiarray
            torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
        except:
            pass
        bida_checkpoint = torch.load(str(bida_checkpoint_path), map_location=device, weights_only=False)
        bida_state_dict = bida_checkpoint['model_state_dict']
        
        # 加载image_encoder和classifier的权重
        model_state_dict = model.state_dict()
        loaded_keys = []
        for key in model_state_dict.keys():
            if 'image_encoder' in key or 'classifier' in key:
                bida_key = key
                if bida_key in bida_state_dict:
                    model_state_dict[key] = bida_state_dict[bida_key]
                    loaded_keys.append(key)
        
        model.load_state_dict(model_state_dict, strict=False)
        print(f"✅ 已加载BIDA权重: {len(loaded_keys)} 个参数")
        print(f"   加载的模块: image_encoder, classifier")
    else:
        print("⚠️ 未找到BIDA权重文件，从头开始训练")
    
    # 使用Focal Loss处理类别不平衡（更有效）
    # 计算Focal Loss的alpha权重（更激进地提升少数类）
    focal_alpha = torch.FloatTensor([1.0, pos_weight_ratio]).to(device)  # [Negative, Positive]
    # 增加gamma值，更关注难分类样本（特别是少数类）
    focal_gamma = 3.0  # 从2.0增加到3.0，更关注难样本
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, label_smoothing=0.1)
    print(f"📊 使用Focal Loss: alpha=[Negative={focal_alpha[0].item():.2f}, Positive={focal_alpha[1].item():.2f}], gamma={focal_gamma}, label_smoothing=0.1")
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # 使用Warmup + CosineAnnealing调度器（更平滑的学习率变化）
    if args.use_warmup:
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                # Warmup阶段：线性增长
                return (epoch + 1) / args.warmup_epochs
            else:
                # CosineAnnealing阶段
                progress = (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)
                cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                # 保持最小学习率
                min_lr_ratio = 0.1
                return max(min_lr_ratio, cosine_factor)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"📊 使用Warmup + CosineAnnealing调度器: warmup_epochs={args.warmup_epochs}")
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_f1': [],
        'val_f1_pos': []
    }
    
    # Early Stopping - 使用F1-score（对类别不平衡更公平）
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None
    
    # 训练循环
    print("🚀 开始训练Bio-COT模型...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.log_dir / f'train_bio_cot_xiangyang_{timestamp}.log'
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"训练开始时间: {datetime.now()}\n")
        f.write(f"模型: Bio-COT (单模态OCT)\n")
        f.write(f"训练集: {len(train_dataset)} 样本\n")
        f.write(f"验证集: {len(val_dataset)} 样本\n")
        f.write("-" * 80 + "\n")
    
    for epoch in range(args.num_epochs):
        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args)
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device, epoch, args)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['acc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_f1_pos'].append(val_metrics['f1_pos'])
        
        # 打印结果
        print(f"\nEpoch {epoch+1}/{args.num_epochs}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}, F1_pos: {val_metrics['f1_pos']:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 记录日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Epoch {epoch+1}: Train Loss={train_metrics['loss']:.4f}, Train Acc={train_metrics['acc']:.4f}, "
                   f"Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['acc']:.4f}, Val AUC={val_metrics['auc']:.4f}, "
                   f"Val F1={val_metrics['f1']:.4f}, Val F1_pos={val_metrics['f1_pos']:.4f}\n")
        
        # Early Stopping - 使用F1-score（对类别不平衡更公平）
        # 同时考虑F1-score和AUC，优先F1-score
        current_metric = val_metrics['f1'] * 0.7 + val_metrics['auc'] * 0.3  # 加权组合
        if current_metric > best_val_f1 + args.min_delta:
            best_val_f1 = current_metric
            best_val_acc = val_metrics['acc']
            best_val_auc = val_metrics['auc']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # 保存最佳模型
            checkpoint_path = args.checkpoint_dir / f'best_bio_cot_{timestamp}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_auc': best_val_auc,
                'val_f1': val_metrics['f1'],
                'val_f1_pos': val_metrics['f1_pos'],
            }, checkpoint_path)
            print(f"  ✅ 保存最佳模型 (F1={val_metrics['f1']:.4f}, F1_pos={val_metrics['f1_pos']:.4f}, AUC={val_metrics['auc']:.4f}): {checkpoint_path}")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"\n⏹️ Early Stopping: 验证集F1-score {args.patience} 个epoch未提升")
            break
    
    # 加载最佳模型进行最终评估
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n📊 使用最佳模型进行最终评估 (Val F1: {best_val_f1:.4f}, Val Acc: {best_val_acc:.4f}, Val AUC: {best_val_auc:.4f})")
        final_metrics = validate(model, val_loader, criterion, device, epoch, args)
        
        # 分类报告
        print("\n分类报告:")
        print(classification_report(final_metrics['labels'], final_metrics['preds'], 
                                  target_names=['Negative (0)', 'Positive (1)']))
        
        # 保存结果
        results = {
            'model': 'Bio-COT (单模态OCT)',
            'best_val_acc': float(best_val_acc),
            'best_val_auc': float(final_metrics['auc']),
            'final_metrics': {
                'loss': float(final_metrics['loss']),
                'acc': float(final_metrics['acc']),
                'auc': float(final_metrics['auc'])
            },
            'classification_report': classification_report(final_metrics['labels'], final_metrics['preds'], 
                                                         target_names=['Negative (0)', 'Positive (1)'], output_dict=True)
        }
        
        results_path = args.output_dir / f'results_bio_cot_{timestamp}.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ 结果已保存: {results_path}")
        
        # 绘制图表
        plot_training_curves(history, args.output_dir / f'training_curves_bio_cot_{timestamp}.png')
        plot_confusion_matrix(final_metrics['labels'], final_metrics['preds'], 
                            args.output_dir / f'confusion_matrix_bio_cot_{timestamp}.png')
    
    print("\n✅ 训练完成!")


if __name__ == '__main__':
    main()

