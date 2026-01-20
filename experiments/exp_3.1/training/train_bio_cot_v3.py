#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.0: 训练脚本
实现动态Beta策略和稀疏性损失
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# 在导入torch之前设置CUDA_VISIBLE_DEVICES
import os
# 自动选择可用的GPU（优先使用GPU 1，如果被占用则使用GPU 0）
# 检查GPU使用情况，选择使用率较低的GPU
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
        
        # 选择使用率最低的GPU
        if gpu_usage:
            best_gpu = min(gpu_usage.items(), key=lambda x: x[1])[0]
            os.environ['CUDA_VISIBLE_DEVICES'] = str(best_gpu)
            print(f"✅ 自动选择GPU {best_gpu} (使用率: {gpu_usage[best_gpu]}%)")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
except:
    # 如果无法检测，默认使用GPU 0
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
from typing import Dict

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.bio_cot_v3 import BioCOT_v3, create_bio_cot_v3
from data.dataset_v3 import FiveCentersMultimodalDatasetV3
from config import BioCOT_v3_Config
# 修复漏洞1：使用新的函数提取Patch特征（丢弃[CLS]）
try:
    from .extract_vit_patches import extract_patch_features_with_vit
except ImportError:
    # 如果相对导入失败，使用绝对导入
    from training.extract_vit_patches import extract_patch_features_with_vit
from src.utils.anti_overfitting import FocalLoss


def visualize_training(history: Dict, log_dir: Path, timestamp: str, best_auc: float):
    """
    生成训练过程可视化图表
    
    Args:
        history: 训练历史字典
        log_dir: 日志目录
        timestamp: 时间戳
        best_auc: 最佳AUC值
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 创建图表
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
        ax5.plot(epochs, history['ot_loss'], 'g-', label='OT Loss', linewidth=2)
    if history.get('align_loss') and len(history['align_loss']) > 0 and any(v > 0 for v in history['align_loss']):
        ax5.plot(epochs, history['align_loss'], 'cyan', label='🔥 Alignment Loss', linewidth=2)  # 🔥 [NEW]
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
    ax5.set_yscale('log')  # 使用对数刻度
    
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
    
    # 添加总标题
    fig.suptitle(f'Bio-COT 3.1 Logic Loop Training Results (Best AUC: {best_auc:.4f})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # 保存图表
    output_path = log_dir / f"training_curves_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 可视化图表已保存: {output_path}")


def get_dynamic_beta(epoch: int, max_epochs: int = 100) -> float:
    """
    动态Beta策略（Warm-up）
    
    Args:
        epoch: 当前epoch
        max_epochs: 总epoch数
    
    Returns:
        beta: 背景抑制系数
    """
    # 前5个Epoch不进行过滤 (beta=1.0)，让模型先学全局
    if epoch < 5:
        current_beta = 1.0
    elif epoch < 20:
        # 线性衰减: 1.0 -> 0.1
        current_beta = 1.0 - (0.9 * (epoch - 5) / 15)
    else:
        current_beta = 0.1  # 强过滤模式
    
    return current_beta


def train_epoch(
    model, 
    dataloader, 
    criterion, 
    optimizer, 
    device, 
    epoch, 
    config, 
    log_print=None
):
    """训练一个epoch"""
    if log_print is None:
        log_print = print
    
    model.train()
    model.set_epoch(epoch)  # 设置epoch（用于Warm-up）
    
    # 获取动态Beta
    current_beta = get_dynamic_beta(epoch, config.num_epochs)
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    cls_losses = []
    ot_losses = []
    align_losses = []  # 🔥 [NEW] 对齐损失
    align_recalls = []  # 🔥 [NEW] 对齐召回率 Recall@1（越高越好）
    sparse_losses = []
    consist_losses = []
    adv_losses = []
    
    log_print(f"\n{'='*80}")
    log_print(f"📊 Epoch {epoch}/{config.num_epochs} - 训练阶段")
    log_print(f"{'='*80}")
    log_print(f"当前Beta: {current_beta:.3f} (Warm-up策略)")
    log_print(f"总batch数: {len(dataloader)}")
    
    # 修复BrokenPipeError：设置tqdm的file参数，避免管道错误
    import sys
    try:
        # 尝试使用标准输出，如果失败则禁用tqdm
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs} [Train]', 
                   file=sys.stdout if sys.stdout.isatty() else None,
                   disable=not sys.stdout.isatty())
    except (BrokenPipeError, OSError):
        # 如果管道错误，禁用tqdm
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs} [Train]', disable=True)
    
    for batch_idx, batch in enumerate(pbar):
        # 数据传输到GPU
        oct_images = batch['oct_images'].to(device, non_blocking=True)
        colposcopy_images = batch['colposcopy_images'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        center_labels = batch['center_idx'].to(device, non_blocking=True)
        knowledge_embeddings = batch['knowledge_embedding'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # 修复漏洞1：从ViT提取Patch特征（丢弃[CLS] token）
        # 处理多帧/多图的情况
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
        
        # 前向传播
        outputs = model(
            f_oct=oct_features_patch,
            f_colpo=colpo_features_patch,
            note_embeds=knowledge_embeddings,
            center_labels=center_labels,
            return_loss_components=True,
            current_beta=current_beta
        )
        
        logits = outputs['pred']
        
        # 计算分类损失
        L_cls = criterion(logits, labels)
        
        # 计算总损失
        total_loss_batch = L_cls
        
        # 添加OT损失
        if config.use_ot and 'L_ot' in outputs.get('loss_components', {}):
            L_ot = outputs['loss_components']['L_ot']
            total_loss_batch = total_loss_batch + config.lambda_ot * L_ot
            ot_losses.append(L_ot.item())
        
        # 🔥 [NEW] 添加对齐损失 (Alignment Loss - 逻辑闭环关键)
        if 'L_align' in outputs.get('loss_components', {}):
            L_align = outputs['loss_components']['L_align']
            total_loss_batch = total_loss_batch + config.lambda_align * L_align
            align_losses.append(L_align.item())
        else:
            align_losses.append(0.0)

        # 🔥 [NEW] 记录对齐召回率（不参与反传）
        if 'Recall_Align' in outputs.get('loss_components', {}):
            try:
                align_recalls.append(float(outputs['loss_components']['Recall_Align'].detach().cpu().item()))
            except Exception:
                align_recalls.append(0.0)
        else:
            align_recalls.append(0.0)
        
        # 添加稀疏性损失（修复漏洞3：已添加下界保护）
        if config.use_visual_notes and 'L_sparse' in outputs.get('loss_components', {}):
            L_sparse = outputs['loss_components']['L_sparse']
            total_loss_batch = total_loss_batch + config.lambda_sparse * L_sparse
            sparse_losses.append(L_sparse.item())
            
            # 监控注意力值（用于调试）
            if 'attn_mean_oct' in outputs.get('loss_components', {}):
                attn_mean_oct = outputs['loss_components']['attn_mean_oct']
                attn_mean_colpo = outputs['loss_components'].get('attn_mean_colpo', 0.0)
                if batch_idx == 0:  # 只打印第一个batch
                    log_print(f"     注意力均值: OCT={attn_mean_oct:.4f}, Colpo={attn_mean_colpo:.4f}")
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
        
        # 更新进度条（捕获可能的管道错误）
        try:
            pbar.set_postfix({
                'loss': f"{total_loss_batch.item():.4f}",
                'cls': f"{L_cls.item():.4f}",
                'beta': f"{current_beta:.3f}",
                'acc': f"{accuracy_score(all_labels, all_preds):.4f}"
            })
        except (BrokenPipeError, OSError):
            # 如果管道错误，只更新日志，不更新进度条
            if batch_idx % 10 == 0:  # 每10个batch打印一次
                log_print(f"      Batch {batch_idx}/{len(dataloader)}: loss={total_loss_batch.item():.4f}, acc={accuracy_score(all_labels, all_preds):.4f}")
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    
    log_print(f"\n  📊 Epoch {epoch} 训练统计:")
    log_print(f"     - 平均损失: {avg_loss:.6f}")
    log_print(f"     - 准确率: {acc:.4f}")
    log_print(f"     - 分类损失: {np.mean(cls_losses):.6f}")
    if config.use_ot:
        log_print(f"     - OT损失: {np.mean(ot_losses):.6f}")
    if align_losses and any(v > 0 for v in align_losses):
        log_print(f"     - 🔥 对齐损失 (Alignment): {np.mean(align_losses):.6f}")
        log_print(f"     - ✅ 对齐召回率 (Recall@1): {np.mean(align_recalls):.4f}")
    if config.use_visual_notes:
        log_print(f"     - 稀疏性损失: {np.mean(sparse_losses):.6f}")
    if config.use_dual:
        log_print(f"     - 一致性损失: {np.mean(consist_losses):.6f}")
        log_print(f"     - 对抗损失: {np.mean(adv_losses):.6f}")
    
    return {
        'loss': avg_loss,
        'acc': acc,
        'cls_loss': np.mean(cls_losses),
        'ot_loss': np.mean(ot_losses) if ot_losses else 0.0,
        'align_loss': np.mean(align_losses) if align_losses else 0.0,  # 🔥 [NEW]
        'align_recall': float(np.mean(align_recalls)) if align_recalls else 0.0,  # 🔥 [NEW]
        'sparse_loss': np.mean(sparse_losses) if sparse_losses else 0.0,
        'consist_loss': np.mean(consist_losses) if consist_losses else 0.0,
        'adv_loss': np.mean(adv_losses) if adv_losses else 0.0,
    }


def validate(model, dataloader, criterion, device, epoch, config, log_print=None):
    """验证"""
    if log_print is None:
        log_print = print
    
    model.eval()
    model.set_epoch(epoch)
    
    # 使用固定的Beta（强过滤模式）
    current_beta = 0.1
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    align_recalls = []  # 🔥 [NEW] 验证阶段的对齐召回率
    
    log_print(f"\n  📊 Epoch {epoch}/{config.num_epochs} - 验证阶段")
    log_print(f"     当前Beta: {current_beta:.3f}")
    
    with torch.no_grad():
        # 修复BrokenPipeError：设置tqdm的file参数
        import sys
        try:
            pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs} [Val]',
                       file=sys.stdout if sys.stdout.isatty() else None,
                       disable=not sys.stdout.isatty())
        except (BrokenPipeError, OSError):
            pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs} [Val]', disable=True)
        
        for batch_idx, batch in enumerate(pbar):
            oct_images = batch['oct_images'].to(device, non_blocking=True)
            colposcopy_images = batch['colposcopy_images'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            center_labels = batch['center_idx'].to(device, non_blocking=True)
            knowledge_embeddings = batch['knowledge_embedding'].to(device, non_blocking=True)
            
            # 修复漏洞1：从ViT提取Patch特征（丢弃[CLS] token）
            # 注意：oct_images可能是[B, F, C, H, W]，colposcopy_images可能是[B, N, C, H, W]
            # 需要先处理多帧/多图的情况
            
            # OCT: [B, F, C, H, W] -> [B*F, C, H, W] -> [B*F, 196, 768] -> [B, F, 196, 768] -> [B, F*196, 768] -> [B, 196, 768] (平均)
            B_oct = oct_images.shape[0]
            if len(oct_images.shape) == 5:  # [B, F, C, H, W]
                F_oct = oct_images.shape[1]
                oct_images_flat = oct_images.view(B_oct * F_oct, *oct_images.shape[2:])  # [B*F, C, H, W]
                oct_features_patch = extract_patch_features_with_vit(oct_images_flat, device)  # [B*F, 196, 768]
                # 重新reshape并平均
                oct_features_patch = oct_features_patch.view(B_oct, F_oct, 196, 768)  # [B, F, 196, 768]
                oct_features_patch = oct_features_patch.mean(dim=1)  # [B, 196, 768] 平均所有帧
            else:
                oct_features_patch = extract_patch_features_with_vit(oct_images, device)  # [B, 196, 768]
            
            # Colposcopy: [B, N, C, H, W] -> [B*N, C, H, W] -> [B*N, 196, 768] -> [B, N, 196, 768] -> [B, 196, 768] (平均)
            B_colpo = colposcopy_images.shape[0]
            if len(colposcopy_images.shape) == 5:  # [B, N, C, H, W]
                N_colpo = colposcopy_images.shape[1]
                colpo_images_flat = colposcopy_images.view(B_colpo * N_colpo, *colposcopy_images.shape[2:])  # [B*N, C, H, W]
                colpo_features_patch = extract_patch_features_with_vit(colpo_images_flat, device)  # [B*N, 196, 768]
                # 重新reshape并平均
                colpo_features_patch = colpo_features_patch.view(B_colpo, N_colpo, 196, 768)  # [B, N, 196, 768]
                colpo_features_patch = colpo_features_patch.mean(dim=1)  # [B, 196, 768] 平均所有图
            else:
                colpo_features_patch = extract_patch_features_with_vit(colposcopy_images, device)  # [B, 196, 768]
            
            # 前向传播
            # [关键修改] 验证时也设置 return_loss_components=True 以获取 Recall
            outputs = model(
                f_oct=oct_features_patch,
                f_colpo=colpo_features_patch,
                note_embeds=knowledge_embeddings,
                center_labels=center_labels,
                return_loss_components=True,  # 🔥 [FIX] 改为 True 以获取 Recall
                current_beta=current_beta
            )
            
            logits = outputs['pred']
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # 🔥 [NEW] 提取并记录对齐召回率
            if 'loss_components' in outputs and 'Recall' in outputs['loss_components']:
                try:
                    recall_val = float(outputs['loss_components']['Recall'].detach().cpu().item())
                    align_recalls.append(recall_val)
                except Exception:
                    align_recalls.append(0.0)
            elif 'loss_components' in outputs and 'Recall_Align' in outputs['loss_components']:
                try:
                    recall_val = float(outputs['loss_components']['Recall_Align'].detach().cpu().item())
                    align_recalls.append(recall_val)
                except Exception:
                    align_recalls.append(0.0)
            else:
                align_recalls.append(0.0)
            
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            
            # 调试：打印第一个batch的预测分布
            if batch_idx == 0:
                log_print(f"     第一个batch预测分布: {torch.bincount(preds)}")
                log_print(f"     第一个batch标签分布: {torch.bincount(labels)}")
                log_print(f"     第一个batch预测概率范围: [{probs[:, 1].min():.4f}, {probs[:, 1].max():.4f}]")
            
            # 更新进度条（捕获可能的管道错误）
            try:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{accuracy_score(all_labels, all_preds):.4f}"
                })
            except (BrokenPipeError, OSError):
                pass  # 忽略管道错误，继续训练
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        auc = 0.0
        log_print(f"     ⚠️ AUC计算失败: {e}")
    
    # 🔧 修复F1计算：使用weighted F1，即使某个类别没有预测到也能计算
    # 原因：早期训练时，模型可能只预测一个类别，导致F1=0
    # 解决方案：使用weighted F1，按类别样本数加权，更公平
    try:
        # 检查是否有两个类别
        unique_labels = np.unique(all_labels)
        unique_preds = np.unique(all_preds)
        
        if len(unique_labels) < 2:
            log_print(f"     ⚠️ 标签只有{len(unique_labels)}个类别，无法计算F1")
            f1 = 0.0
        elif len(unique_preds) < 2:
            # 🔧 修复：即使预测只有1个类别，也使用weighted F1计算
            # weighted F1会按类别样本数加权，即使某个类别没有预测到，也能给出有意义的值
            f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_weighted  # 使用weighted F1作为主要指标
            
            # 只在第一个epoch或前几个epoch打印警告
            if epoch <= 3:
                log_print(f"     ⚠️ 预测只有{len(unique_preds)}个类别（早期训练正常现象）")
                log_print(f"     使用weighted F1: {f1_weighted:.4f}, macro F1: {f1_macro:.4f}")
        else:
            # 正常情况：使用weighted F1（更公平，考虑类别不平衡）
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            
            # 额外检查：如果F1很低，打印混淆矩阵用于调试
            if f1 < 0.1 and epoch <= 5:
                cm = confusion_matrix(all_labels, all_preds)
                log_print(f"     ⚠️ F1较低（{f1:.4f}），混淆矩阵:\n{cm}")
                log_print(f"     标签分布: {np.bincount(all_labels)}")
                log_print(f"     预测分布: {np.bincount(all_preds)}")
    except Exception as e:
        f1 = 0.0
        log_print(f"     ⚠️ F1计算失败: {e}")
    
    # 🔥 [NEW] 计算平均对齐召回率
    avg_recall = np.mean(align_recalls) if align_recalls else 0.0
    
    log_print(f"\n  📊 Epoch {epoch} 验证统计:")
    log_print(f"     - 平均损失: {avg_loss:.6f}")
    log_print(f"     - 准确率: {acc:.4f}")
    log_print(f"     - AUC: {auc:.4f}")
    log_print(f"     - 🔗 对齐召回率 (Recall@1): {avg_recall:.4f}")  # 🔥 [NEW] 验证阶段的 Recall
    log_print(f"     - F1-Score: {f1:.4f}")
    
    return {
        'loss': avg_loss,
        'acc': acc,
        'auc': auc,
        'f1': f1,
        'align_recall': avg_recall,  # 🔥 [NEW] 返回 Recall 值
    }


def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Bio-COT 3.1 训练脚本')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（可选，默认使用 config.py）')
    parser.add_argument('--gpu', type=int, default=None,
                       help='指定GPU ID（可选，默认自动选择）')
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        # 从指定路径加载配置
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        # 查找配置类（通常是 Config 结尾的类）
        config_classes = [cls for cls in config_module.__dict__.values() 
                        if isinstance(cls, type) and 'Config' in cls.__name__]
        if config_classes:
            config = config_classes[0]()
        else:
            raise ValueError(f"在 {args.config} 中未找到配置类")
    else:
        # 使用默认配置
        from config import BioCOT_v3_Config
        config = BioCOT_v3_Config()
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(config.log_dir) / f"train_bio_cot_v3_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    log_f = open(log_file, 'w', encoding='utf-8')
    
    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_f)
        log_f.flush()
    
    log_print("=" * 80)
    log_print("Bio-COT 3.1 Logic Loop: 以医学先验为核心的自适应推理系统")
    log_print("核心升级:")
    log_print("  1. Adaptive Modality Gating (自适应模态门控)")
    log_print("  2. Enhanced Visual Notes (Cross-Attention机制)")
    log_print("  3. Semantic-Visual Alignment Loop (语义-视觉对齐闭环)")
    log_print("=" * 80)
    log_print(f"数据路径: {config.data_root}")
    log_print(f"知识库路径: {config.knowledge_base_path}")
    log_print(f"Knowledge Note Embeddings: {config.knowledge_embed_path}")
    log_print(f"模块配置:")
    log_print(f"  - use_visual_notes: {config.use_visual_notes}")
    log_print(f"  - use_ot: {config.use_ot}")
    log_print(f"  - use_dual: {config.use_dual}")
    log_print(f"  - use_cross_attn: {config.use_cross_attn}")
    log_print("=" * 80)
    
    # 选择GPU
    if torch.cuda.is_available():
        if args.gpu is not None:
            device = torch.device(f'cuda:{args.gpu}')
        else:
            # 自动选择GPU（使用率最低的）
            import subprocess
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
                gpu_utils = []
                for line in result.stdout.strip().split('\n'):
                    idx, util = line.split(', ')
                    gpu_utils.append((int(idx), int(util)))
                best_gpu = min(gpu_utils, key=lambda x: x[1])[0]
                device = torch.device(f'cuda:{best_gpu}')
                log_print(f"✅ 自动选择GPU {best_gpu} (使用率: {min(gpu_utils, key=lambda x: x[1])[1]}%)")
            except:
                device = torch.device('cuda:0')
                log_print(f"✅ 使用GPU设备: {device}")
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
    
    train_dataset = FiveCentersMultimodalDatasetV3(
        csv_path=str(train_csv),
        knowledge_embed_path=config.knowledge_embed_path,
        transform=transform,
        oct_num_frames=config.oct_frames,
        max_col_images=config.colposcopy_images,
        balance_negative_frames=True
    )
    log_print(f"  ✅ 训练集加载完成: {len(train_dataset)} 个样本")
    
    val_dataset = FiveCentersMultimodalDatasetV3(
        csv_path=str(val_csv),
        knowledge_embed_path=config.knowledge_embed_path,
        transform=transform,
        oct_num_frames=config.oct_frames,
        max_col_images=config.colposcopy_images,
        balance_negative_frames=True
    )
    log_print(f"  ✅ 验证集加载完成: {len(val_dataset)} 个样本")
    
    # 创建加权采样器
    # ⚠️ 性能修复：不要通过 train_dataset[i] 逐个触发图像加载（会非常慢）
    if hasattr(train_dataset, "df") and "label" in getattr(train_dataset, "df").columns:
        train_labels = train_dataset.df["label"].astype(int).values
    else:
        # fallback（不推荐）：只读取label字段，避免加载图片
        train_labels = [int(train_dataset.df.iloc[i]["label"]) for i in range(len(train_dataset))] if hasattr(train_dataset, "df") else [train_dataset[i]["label"].item() for i in range(len(train_dataset))]

    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[int(label)] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    # 创建模型
    log_print("\n📊 正在创建Bio-COT 3.0模型...")
    model = create_bio_cot_v3(config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f"✅ 模型创建完成")
    log_print(f"   总参数量: {total_params:,}")
    log_print(f"   可训练参数量: {trainable_params:,}")
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # 训练循环
    log_print("\n" + "=" * 80)
    log_print("🚀 开始训练...")
    log_print("=" * 80)
    
    best_auc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_f1': [],
        'cls_loss': [], 'ot_loss': [], 'align_loss': [], 'align_recall': [], 'sparse_loss': [], 'consist_loss': [], 'adv_loss': []  # 🔥 添加align_recall
    }
    
    for epoch in range(1, config.num_epochs + 1):
        try:
            # 训练
            train_results = train_epoch(model, train_loader, criterion, optimizer, device, epoch, config, log_print=log_print)
            
            # 验证
            val_results = validate(model, val_loader, criterion, device, epoch, config, log_print=log_print)
        except Exception as e:
            log_print(f"\n❌ Epoch {epoch} 训练失败: {e}")
            import traceback
            log_print(f"详细错误信息:\n{traceback.format_exc()}")
            log_print("训练已停止，请检查错误信息")
            break
        
        # 记录历史
        history['train_loss'].append(train_results['loss'])
        history['train_acc'].append(train_results['acc'])
        history['val_loss'].append(val_results['loss'])
        history['val_acc'].append(val_results['acc'])
        history['val_auc'].append(val_results['auc'])
        history['val_f1'].append(val_results['f1'])
        history['cls_loss'].append(train_results['cls_loss'])
        history['ot_loss'].append(train_results['ot_loss'])
        history['align_loss'].append(train_results.get('align_loss', 0.0))  # 🔥 [NEW]
        history['align_recall'].append(train_results.get('align_recall', 0.0))  # 🔥 [NEW]
        history['sparse_loss'].append(train_results['sparse_loss'])
        history['consist_loss'].append(train_results['consist_loss'])
        history['adv_loss'].append(train_results['adv_loss'])
        
        # 保存最佳模型
        if val_results['auc'] > best_auc:
            best_auc = val_results['auc']
            checkpoint_path = Path(config.checkpoint_dir) / f"best_model_v3_{timestamp}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'config': config.__dict__,
                'history': history
            }, checkpoint_path)
            log_print(f"  ✅ 保存最佳模型 (AUC: {best_auc:.4f})")
    
    log_print(f"\n✅ 训练完成！最佳AUC: {best_auc:.4f}")
    
    # 保存训练历史
    history_file = Path(config.log_dir) / f"training_history_{timestamp}.json"
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    log_print(f"📁 训练历史已保存到: {history_file}")
    
    # 生成可视化图表
    log_print(f"\n📊 正在生成可视化图表...")
    visualize_training(history, Path(config.log_dir), timestamp, best_auc)
    log_print(f"✅ 可视化完成！")
    
    log_f.close()


if __name__ == '__main__':
    main()

