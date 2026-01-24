#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
襄阳数据集训练脚本
单模态图像分类（Colposcopy图像）
"""

import sys
from pathlib import Path
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

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# 优先使用本地的方法文件
sys.path.insert(0, str(Path(__file__).parent))

try:
    from xiangyang_dataset import XiangyangDataset
except ImportError:
    from src.data.xiangyang_dataset import XiangyangDataset


class XiangyangArgs:
    """训练参数"""
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
        self.batch_size = 32
        self.num_workers = 4
        self.pin_memory = True
        
        # 优化器参数
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.momentum = 0.9
        
        # 学习率调度
        self.step_size = 15
        self.gamma = 0.1
        
        # 数据增强
        self.input_size = 224
        self.use_augmentation = True
        
        # Early Stopping
        self.patience = 10
        self.min_delta = 0.001
        
        # 模型配置
        self.model_name = 'resnet50'  # 'resnet50', 'efficientnet_b0', 'efficientnet_b3'
        self.pretrained = True
        self.num_classes = 2


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


def create_model(args, use_bida_weights=False):
    """创建模型
    
    Args:
        args: 训练参数
        use_bida_weights: 是否使用BIDA权重（需要ResNet50 + BIDA的image_encoder和classifier）
    """
    if use_bida_weights:
        # 使用BIDA架构：ResNet50 backbone + BIDA的image_encoder + classifier
        from src.models.bida.bida_model import DualHeadImageEncoder
        
        # ResNet50作为backbone提取特征
        backbone = models.resnet50(pretrained=args.pretrained)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()  # 移除原始分类层，只提取特征
        
        # BIDA的image_encoder（输入512维特征，输出768维）
        # 需要将ResNet50的2048维特征投影到512维
        feature_proj = nn.Linear(num_features, 512)
        image_encoder = DualHeadImageEncoder(input_dim=512, embed_dim=768)
        
        # BIDA的classifier
        classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(768, 768 // 2),
            nn.LayerNorm(768 // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768 // 2, args.num_classes)
        )
        
        # 组合成完整模型
        class HybridModel(nn.Module):
            def __init__(self, backbone, feature_proj, image_encoder, classifier):
                super().__init__()
                self.backbone = backbone
                self.feature_proj = feature_proj
                self.image_encoder = image_encoder
                self.classifier = classifier
            
            def forward(self, x):
                # ResNet50提取特征
                features = self.backbone(x)  # [B, 2048]
                # 投影到512维
                features_512 = self.feature_proj(features)  # [B, 512]
                # BIDA image_encoder
                z_causal, z_noise = self.image_encoder(features_512)  # [B, 768], [B, 768]
                # 分类
                logits = self.classifier(z_causal)  # [B, num_classes]
                return logits
        
        model = HybridModel(backbone, feature_proj, image_encoder, classifier)
        return model
    
    # 原始ResNet模型
    if args.model_name.startswith('resnet'):
        if args.model_name == 'resnet50':
            model = models.resnet50(pretrained=args.pretrained)
        elif args.model_name == 'resnet34':
            model = models.resnet34(pretrained=args.pretrained)
        else:
            model = models.resnet18(pretrained=args.pretrained)
        
        # 替换分类层
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, args.num_classes)
    
    elif args.model_name.startswith('efficientnet'):
        from torchvision.models import efficientnet_b0, efficientnet_b3
        
        if args.model_name == 'efficientnet_b0':
            model = efficientnet_b0(pretrained=args.pretrained)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, args.num_classes)
        elif args.model_name == 'efficientnet_b3':
            model = efficientnet_b3(pretrained=args.pretrained)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, args.num_classes)
        else:
            raise ValueError(f"未知的EfficientNet模型: {args.model_name}")
    
    else:
        raise ValueError(f"未知的模型: {args.model_name}")
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, args):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Train]')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 更新进度条
        acc = accuracy_score(all_labels, all_preds)
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
    
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
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 阳性概率
            
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
    
    return {
        'loss': epoch_loss,
        'acc': epoch_acc,
        'auc': epoch_auc,
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['阴性', '阳性'], 
                yticklabels=['阴性', '阳性'])
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存: {save_path}")


def main():
    args = XiangyangArgs()
    
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
    
    # 创建数据加载器
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
    
    # 计算类别权重（解决类别不平衡问题）
    train_labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    class_weights = total_samples / (args.num_classes * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"📊 类别分布: {dict(enumerate(class_counts))}")
    print(f"📊 类别权重: {dict(enumerate(class_weights.cpu().numpy()))}")
    
    # 创建模型
    print(f"🏗️ 创建模型: {args.model_name}")
    
    # 尝试加载BIDA权重（可以通过环境变量控制）
    import os
    use_bida_weights_env = os.getenv('USE_BIDA_WEIGHTS', 'true').lower() == 'true'
    bida_checkpoint_path = Path('/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp2_bida/exp_bida/best_model.pth')
    use_bida_weights = use_bida_weights_env and bida_checkpoint_path.exists()
    
    if use_bida_weights:
        print(f"📂 使用BIDA权重初始化模型: {bida_checkpoint_path}")
        model = create_model(args, use_bida_weights=True)
        # 确保device已定义
        if 'device' not in locals():
            device = torch.device(args.device)
        model = model.to(device)
        
        # 加载BIDA权重
        try:
            # 尝试添加numpy的safe globals以支持加载旧格式的checkpoint
            import numpy.core.multiarray
            torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
        except:
            pass
        bida_checkpoint = torch.load(str(bida_checkpoint_path), map_location=device, weights_only=False)
        bida_state_dict = bida_checkpoint['model_state_dict']
        
        # 提取image_encoder和classifier的权重
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
        model = create_model(args, use_bida_weights=False)
        model = model.to(device)
    
    # 尝试从本地最佳checkpoint加载模型（用于继续训练）
    checkpoint_path = args.checkpoint_dir / 'best_model_20260105_204538.pth'
    start_epoch = 0
    checkpoint = None
    if checkpoint_path.exists() and not use_bida_weights:
        print(f"📂 从本地checkpoint加载模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"✅ 已加载checkpoint (epoch {start_epoch}, val_acc: {checkpoint.get('val_acc', 0):.4f})")
    elif not use_bida_weights:
        print("⚠️ 未找到checkpoint，从头开始训练")
    
    # 损失函数和优化器（使用加权损失解决类别不平衡）
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # 如果从checkpoint加载，也加载优化器状态
    if checkpoint is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ 已加载优化器状态")
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }
    
    # Early Stopping
    best_val_acc = checkpoint.get('val_acc', 0.0) if checkpoint is not None else 0.0
    patience_counter = 0
    best_model_state = None
    
    # 训练循环
    print("🚀 开始训练...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.log_dir / f'train_xiangyang_{timestamp}.log'
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"训练开始时间: {datetime.now()}\n")
        f.write(f"模型: {args.model_name}\n")
        f.write(f"训练集: {len(train_dataset)} 样本\n")
        f.write(f"验证集: {len(val_dataset)} 样本\n")
        f.write("-" * 80 + "\n")
    
    for epoch in range(start_epoch, args.num_epochs):
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
        
        # 打印结果
        print(f"\nEpoch {epoch+1}/{args.num_epochs}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 记录日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Epoch {epoch+1}: Train Loss={train_metrics['loss']:.4f}, Train Acc={train_metrics['acc']:.4f}, "
                   f"Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['acc']:.4f}, Val AUC={val_metrics['auc']:.4f}\n")
        
        # Early Stopping
        if val_metrics['acc'] > best_val_acc + args.min_delta:
            best_val_acc = val_metrics['acc']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # 保存最佳模型
            checkpoint_path = args.checkpoint_dir / f'best_model_{timestamp}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_auc': val_metrics['auc'],
            }, checkpoint_path)
            print(f"  ✅ 保存最佳模型: {checkpoint_path}")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"\n⏹️ Early Stopping: 验证集准确率 {args.patience} 个epoch未提升")
            break
    
    # 加载最佳模型进行最终评估
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n📊 使用最佳模型进行最终评估 (Val Acc: {best_val_acc:.4f})")
        final_metrics = validate(model, val_loader, criterion, device, epoch, args)
        
        # 分类报告
        print("\n分类报告:")
        print(classification_report(final_metrics['labels'], final_metrics['preds'], 
                                  target_names=['阴性', '阳性']))
        
        # 保存结果
        results = {
            'best_val_acc': float(best_val_acc),
            'best_val_auc': float(final_metrics['auc']),
            'final_metrics': {
                'loss': float(final_metrics['loss']),
                'acc': float(final_metrics['acc']),
                'auc': float(final_metrics['auc'])
            },
            'classification_report': classification_report(final_metrics['labels'], final_metrics['preds'], 
                                                         target_names=['阴性', '阳性'], output_dict=True)
        }
        
        results_path = args.output_dir / f'results_{timestamp}.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ 结果已保存: {results_path}")
        
        # 绘制图表
        plot_training_curves(history, args.output_dir / f'training_curves_{timestamp}.png')
        plot_confusion_matrix(final_metrics['labels'], final_metrics['preds'], 
                            args.output_dir / f'confusion_matrix_{timestamp}.png')
    
    print("\n✅ 训练完成!")


if __name__ == '__main__':
    main()

