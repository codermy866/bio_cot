#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 4.0 (LACT框架): 训练脚本
核心改进：LACT Loss (Language-Anchored Causal Transport Loss)
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# GPU自动选择
import os
import subprocess
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        gpu_usage = {}
        for line in lines:
            parts = line.split(', ')
            if len(parts) == 2:
                gpu_id = int(parts[0])
                usage = int(parts[1])
                gpu_usage[gpu_id] = usage
        if gpu_usage:
            best_gpu = min(gpu_usage.items(), key=lambda x: x[1])[0]
            os.environ['CUDA_VISIBLE_DEVICES'] = str(best_gpu)
            print(f"✅ 自动选择GPU {best_gpu} (使用率: {gpu_usage[best_gpu]}%)")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
except:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import Dict, List

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.bio_cot_v4 import BioCOT_V4, create_bio_cot_v4
from data.dataset_v4 import FiveCentersMultimodalDatasetV4
from config import BioCOT_v4_Config
try:
    from training.extract_vit_patches import extract_patch_features_with_vit
except ImportError:
    from .extract_vit_patches import extract_patch_features_with_vit
from src.utils.anti_overfitting import FocalLoss


def visualize_training(history: Dict, log_dir: Path, timestamp: str, best_auc: float):
    """生成训练过程可视化图表"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    epochs = range(1, len(history['train_loss']) + 1)
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Loss曲线
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy曲线
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. AUC曲线
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs, history['val_auc'], 'g-', label='Val AUC', linewidth=2, marker='o', markersize=4)
    ax3.axhline(y=best_auc, color='r', linestyle='--', linewidth=2, label=f'Best AUC: {best_auc:.4f}')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('AUC', fontsize=12)
    ax3.set_title('Validation AUC', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. F1-Score曲线
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(epochs, history['val_f1'], 'm-', label='Val F1', linewidth=2, marker='s', markersize=4)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('F1-Score', fontsize=12)
    ax4.set_title('Validation F1-Score', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # 5. Loss组件分解
    ax5 = plt.subplot(2, 3, 5)
    if history.get('cls_loss') and len(history['cls_loss']) > 0:
        ax5.plot(epochs, history['cls_loss'], 'b-', label='Classification Loss', linewidth=2)
    if history.get('ot_loss') and len(history['ot_loss']) > 0 and any(v > 0 for v in history['ot_loss']):
        ax5.plot(epochs, history['ot_loss'], 'g-', label='LACT Loss', linewidth=2)
    if history.get('sparse_loss') and len(history['sparse_loss']) > 0 and any(v > 0 for v in history['sparse_loss']):
        ax5.plot(epochs, history['sparse_loss'], 'orange', label='Sparse Loss', linewidth=2)
    if history.get('consist_loss') and len(history['consist_loss']) > 0 and any(v > 0 for v in history['consist_loss']):
        ax5.plot(epochs, history['consist_loss'], 'purple', label='Consistency Loss', linewidth=2)
    if history.get('adv_loss') and len(history['adv_loss']) > 0 and any(v > 0 for v in history['adv_loss']):
        ax5.plot(epochs, history['adv_loss'], 'r-', label='Adversarial Loss', linewidth=2)
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Loss', fontsize=12)
    ax5.set_title('Loss Components', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # 6. 综合性能指标
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(epochs, history['val_auc'], 'g-', label='AUC', linewidth=2, marker='o', markersize=4)
    ax6.plot(epochs, history['val_acc'], 'b-', label='Accuracy', linewidth=2, marker='s', markersize=4)
    ax6.plot(epochs, history['val_f1'], 'm-', label='F1-Score', linewidth=2, marker='^', markersize=4)
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Score', fontsize=12)
    ax6.set_title('Comprehensive Performance Metrics', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1])
    
    fig.suptitle(f'Bio-COT 4.0 (LACT) Training Results (Best AUC: {best_auc:.4f})', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_path = log_dir / f"training_curves_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 可视化图表已保存: {output_path}")


def get_dynamic_beta(epoch: int, max_epochs: int = 100) -> float:
    """动态Beta策略（Warm-up）"""
    if epoch < 5:
        return 1.0
    elif epoch < 20:
        return 1.0 - (0.9 * (epoch - 5) / 15)
    else:
        return 0.1


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, config, log_print=None):
    """训练一个epoch (Bio-COT 4.0版本)"""
    if log_print is None:
        log_print = print
    
    model.train()
    model.set_epoch(epoch)
    current_beta = get_dynamic_beta(epoch, config.num_epochs)
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    cls_losses = []
    lact_losses = []  # LACT Loss (Language-Anchored Causal Transport)
    sparse_losses = []
    consist_losses = []
    adv_losses = []
    
    log_print(f"\n{'='*80}")
    log_print(f"📊 Epoch {epoch}/{config.num_epochs} - 训练阶段 (Bio-COT 4.0 LACT)")
    log_print(f"{'='*80}")
    log_print(f"当前Beta: {current_beta:.3f}")
    log_print(f"总batch数: {len(dataloader)}")
    
    try:
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs} [Train]', 
                   file=sys.stdout if sys.stdout.isatty() else None,
                   disable=not sys.stdout.isatty())
    except:
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs} [Train]', disable=True)
    
    for batch_idx, batch in enumerate(pbar):
        # ⚠️ 关键改动：获取图像文件名（用于VLM检索）
        image_names = batch['image_name']  # List[str] 长度=B
        clinical_info = batch.get('clinical_info_str', None)  # List[str] 或None
        
        # 提取特征
        oct_images = batch['oct_images'].to(device, non_blocking=True)
        colposcopy_images = batch['colposcopy_images'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        center_labels = batch['center_idx'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # 从ViT提取Patch特征
        B_oct = oct_images.shape[0]
        if len(oct_images.shape) == 5:  # [B, F, C, H, W]
            F_oct = oct_images.shape[1]
            oct_images_flat = oct_images.view(B_oct * F_oct, *oct_images.shape[2:])
            oct_features_patch = extract_patch_features_with_vit(oct_images_flat, device)
            oct_features_patch = oct_features_patch.view(B_oct, F_oct, 196, 768).mean(dim=1)
        else:
            oct_features_patch = extract_patch_features_with_vit(oct_images, device)
        
        B_colpo = colposcopy_images.shape[0]
        if len(colposcopy_images.shape) == 5:  # [B, N, C, H, W]
            N_colpo = colposcopy_images.shape[1]
            colpo_images_flat = colposcopy_images.view(B_colpo * N_colpo, *colposcopy_images.shape[2:])
            colpo_features_patch = extract_patch_features_with_vit(colpo_images_flat, device)
            colpo_features_patch = colpo_features_patch.view(B_colpo, N_colpo, 196, 768).mean(dim=1)
        else:
            colpo_features_patch = extract_patch_features_with_vit(colposcopy_images, device)
        
        # 融合多模态特征 (简单平均)
        combined_features = (oct_features_patch + colpo_features_patch) / 2.0  # [B, 196, 768]
        
        # ⚠️ 关键改动：前向传播（传入image_names和clinical_info）
        outputs = model(
            images=combined_features,  # [B, N, D] Patch特征
            clinical_data=clinical_info,  # List[str] 或None
            image_names=image_names,  # List[str] 必需
            center_labels=center_labels,
            return_loss_components=True,
            use_counterfactual=config.use_dual,
            current_beta=current_beta
        )
        
        logits = outputs['logits']
        
        # 计算分类损失
        L_cls = criterion(logits, labels)
        total_loss_batch = config.lambda_cls * L_cls
        
        # ⚠️ 关键改动：LACT Loss (Language-Anchored Causal Transport)
        # 强制视觉因果特征靠近文本锚点
        if config.use_ot and 'z_causal' in outputs and 'z_anchor' in outputs:
            z_causal = outputs['z_causal']
            z_anchor = outputs['z_anchor'].detach()  # detach，teacher不更新
            
            # LACT Loss: Cosine Similarity Loss
            lact_loss = 1 - torch.nn.functional.cosine_similarity(
                z_causal, z_anchor, dim=1
            ).mean()
            
            total_loss_batch = total_loss_batch + config.lambda_ot * lact_loss
            lact_losses.append(lact_loss.item())
        else:
            lact_losses.append(0.0)
        
        # 添加稀疏性损失
        if config.use_visual_notes and 'L_sparse' in outputs.get('loss_components', {}):
            L_sparse = outputs['loss_components']['L_sparse']
            total_loss_batch = total_loss_batch + config.lambda_sparse * L_sparse
            sparse_losses.append(L_sparse.item())
        else:
            sparse_losses.append(0.0)
        
        # 添加一致性损失
        if config.use_dual and 'L_consist' in outputs.get('loss_components', {}):
            L_consist = outputs['loss_components']['L_consist']
            total_loss_batch = total_loss_batch + config.lambda_consist * L_consist
            consist_losses.append(L_consist.item())
        else:
            consist_losses.append(0.0)
        
        # 添加对抗损失
        if config.use_dual and 'L_adv' in outputs.get('loss_components', {}):
            L_adv = outputs['loss_components']['L_adv']
            total_loss_batch = total_loss_batch + config.lambda_adv * L_adv
            adv_losses.append(L_adv.item())
        else:
            adv_losses.append(0.0)
        
        # 反向传播
        total_loss_batch.backward()
        optimizer.step()
        
        # 统计
        total_loss += total_loss_batch.item()
        cls_losses.append(L_cls.item())
        
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())
        
        try:
            pbar.set_postfix({
                'loss': f"{total_loss_batch.item():.4f}",
                'cls': f"{L_cls.item():.4f}",
                'lact': f"{lact_losses[-1]:.4f}" if lact_losses else "0.0000",
                'acc': f"{accuracy_score(all_labels, all_preds):.4f}"
            })
        except:
            if batch_idx % 10 == 0:
                log_print(f"      Batch {batch_idx}/{len(dataloader)}: loss={total_loss_batch.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    
    log_print(f"\n  📊 Epoch {epoch} 训练统计:")
    log_print(f"     - 平均损失: {avg_loss:.6f}")
    log_print(f"     - 准确率: {acc:.4f}")
    log_print(f"     - 分类损失: {np.mean(cls_losses):.6f}")
    if config.use_ot:
        log_print(f"     - LACT损失: {np.mean(lact_losses):.6f}")
    if config.use_visual_notes:
        log_print(f"     - 稀疏性损失: {np.mean(sparse_losses):.6f}")
    if config.use_dual:
        log_print(f"     - 一致性损失: {np.mean(consist_losses):.6f}")
        log_print(f"     - 对抗损失: {np.mean(adv_losses):.6f}")
    
    return {
        'loss': avg_loss,
        'acc': acc,
        'cls_loss': np.mean(cls_losses),
        'ot_loss': np.mean(lact_losses) if lact_losses else 0.0,
        'sparse_loss': np.mean(sparse_losses) if sparse_losses else 0.0,
        'consist_loss': np.mean(consist_losses) if consist_losses else 0.0,
        'adv_loss': np.mean(adv_losses) if adv_losses else 0.0,
    }


def validate(model, dataloader, criterion, device, epoch, config, log_print=None):
    """验证 (Bio-COT 4.0版本)"""
    if log_print is None:
        log_print = print
    
    model.eval()
    model.set_epoch(epoch)
    current_beta = 0.1
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    log_print(f"\n  📊 Epoch {epoch}/{config.num_epochs} - 验证阶段")
    
    with torch.no_grad():
        try:
            pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs} [Val]',
                       file=sys.stdout if sys.stdout.isatty() else None,
                       disable=not sys.stdout.isatty())
        except:
            pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs} [Val]', disable=True)
        
        for batch_idx, batch in enumerate(pbar):
            image_names = batch['image_name']
            clinical_info = batch.get('clinical_info_str', None)
            
            oct_images = batch['oct_images'].to(device, non_blocking=True)
            colposcopy_images = batch['colposcopy_images'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            center_labels = batch['center_idx'].to(device, non_blocking=True)
            
            # 提取特征
            B_oct = oct_images.shape[0]
            if len(oct_images.shape) == 5:
                F_oct = oct_images.shape[1]
                oct_images_flat = oct_images.view(B_oct * F_oct, *oct_images.shape[2:])
                oct_features_patch = extract_patch_features_with_vit(oct_images_flat, device)
                oct_features_patch = oct_features_patch.view(B_oct, F_oct, 196, 768).mean(dim=1)
            else:
                oct_features_patch = extract_patch_features_with_vit(oct_images, device)
            
            B_colpo = colposcopy_images.shape[0]
            if len(colposcopy_images.shape) == 5:
                N_colpo = colposcopy_images.shape[1]
                colpo_images_flat = colposcopy_images.view(B_colpo * N_colpo, *colposcopy_images.shape[2:])
                colpo_features_patch = extract_patch_features_with_vit(colpo_images_flat, device)
                colpo_features_patch = colpo_features_patch.view(B_colpo, N_colpo, 196, 768).mean(dim=1)
            else:
                colpo_features_patch = extract_patch_features_with_vit(colposcopy_images, device)
            
            combined_features = (oct_features_patch + colpo_features_patch) / 2.0
            
            # 前向传播
            outputs = model(
                images=combined_features,
                clinical_data=clinical_info,
                image_names=image_names,
                center_labels=center_labels,
                return_loss_components=False,
                current_beta=current_beta
            )
            
            logits = outputs['logits']
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            
            try:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{accuracy_score(all_labels, all_preds):.4f}"
                })
            except:
                pass
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    try:
        unique_labels = np.unique(all_labels)
        if len(unique_labels) < 2:
            f1 = 0.0
        else:
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    except:
        f1 = 0.0
    
    log_print(f"\n  📊 Epoch {epoch} 验证统计:")
    log_print(f"     - 平均损失: {avg_loss:.6f}")
    log_print(f"     - 准确率: {acc:.4f}")
    log_print(f"     - AUC: {auc:.4f}")
    log_print(f"     - F1-Score: {f1:.4f}")
    
    return {'loss': avg_loss, 'acc': acc, 'auc': auc, 'f1': f1}


def main():
    """主函数"""
    config = BioCOT_v4_Config()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(config.log_dir) / f"train_bio_cot_v4_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    log_f = open(log_file, 'w', encoding='utf-8')
    
    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_f)
        log_f.flush()
    
    log_print("=" * 80)
    log_print("Bio-COT 4.0 (LACT框架): Language-Anchored Causal Transport")
    log_print("=" * 80)
    log_print(f"数据路径: {config.data_root}")
    log_print(f"VLM缓存路径: {config.vlm_json_path}")
    log_print(f"模块配置:")
    log_print(f"  - use_visual_notes: {config.use_visual_notes}")
    log_print(f"  - use_ot (LACT): {config.use_ot}")
    log_print(f"  - use_dual: {config.use_dual}")
    log_print(f"  - use_cross_attn: {config.use_cross_attn}")
    log_print("=" * 80)
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        log_print(f"✅ 使用GPU设备: {device}")
    else:
        raise RuntimeError("❌ CUDA不可用！")
    
    # 加载数据集
    log_print("\n📊 开始加载数据集...")
    train_csv = Path(config.data_root) / 'internal_train' / 'labels.csv'
    val_csv = Path(config.data_root) / 'internal_val' / 'labels.csv'
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = FiveCentersMultimodalDatasetV4(
        csv_path=str(train_csv),
        transform=transform,
        oct_num_frames=config.oct_frames,
        max_col_images=config.colposcopy_images,
        balance_negative_frames=True
    )
    log_print(f"  ✅ 训练集加载完成: {len(train_dataset)} 个样本")
    
    val_dataset = FiveCentersMultimodalDatasetV4(
        csv_path=str(val_csv),
        transform=transform,
        oct_num_frames=config.oct_frames,
        max_col_images=config.colposcopy_images,
        balance_negative_frames=True
    )
    log_print(f"  ✅ 验证集加载完成: {len(val_dataset)} 个样本")
    
    # 创建加权采样器
    train_labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
    class_counts = pd.Series(train_labels).value_counts().sort_index()
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, sampler=sampler,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    # 创建模型
    log_print("\n📊 正在创建Bio-COT 4.0模型...")
    model = create_bio_cot_v4(config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f"✅ 模型创建完成")
    log_print(f"   总参数量: {total_params:,}")
    log_print(f"   可训练参数量: {trainable_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    log_print("\n" + "=" * 80)
    log_print("🚀 开始训练...")
    log_print("=" * 80)
    
    best_auc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_f1': [],
        'cls_loss': [], 'ot_loss': [], 'sparse_loss': [], 'consist_loss': [], 'adv_loss': []
    }
    
    for epoch in range(1, config.num_epochs + 1):
        try:
            train_results = train_epoch(model, train_loader, criterion, optimizer, device, epoch, config, log_print=log_print)
            val_results = validate(model, val_loader, criterion, device, epoch, config, log_print=log_print)
        except Exception as e:
            log_print(f"\n❌ Epoch {epoch} 训练失败: {e}")
            import traceback
            log_print(f"详细错误信息:\n{traceback.format_exc()}")
            break
        
        history['train_loss'].append(train_results['loss'])
        history['train_acc'].append(train_results['acc'])
        history['val_loss'].append(val_results['loss'])
        history['val_acc'].append(val_results['acc'])
        history['val_auc'].append(val_results['auc'])
        history['val_f1'].append(val_results['f1'])
        history['cls_loss'].append(train_results['cls_loss'])
        history['ot_loss'].append(train_results['ot_loss'])
        history['sparse_loss'].append(train_results['sparse_loss'])
        history['consist_loss'].append(train_results['consist_loss'])
        history['adv_loss'].append(train_results['adv_loss'])
        
        if val_results['auc'] > best_auc:
            best_auc = val_results['auc']
            checkpoint_path = Path(config.checkpoint_dir) / f"best_model_v4_{timestamp}.pth"
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_auc': best_auc,
                'config': config.__dict__, 'history': history
            }, checkpoint_path)
            log_print(f"  ✅ 保存最佳模型 (AUC: {best_auc:.4f})")
    
    log_print(f"\n✅ 训练完成！最佳AUC: {best_auc:.4f}")
    
    history_file = Path(config.log_dir) / f"training_history_{timestamp}.json"
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    log_print(f"📁 训练历史已保存到: {history_file}")
    
    log_print(f"\n📊 正在生成可视化图表...")
    visualize_training(history, Path(config.log_dir), timestamp, best_auc)
    log_print(f"✅ 可视化完成！")
    
    log_f.close()


if __name__ == '__main__':
    main()
