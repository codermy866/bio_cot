#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 2.0: 训练脚本
支持模块化消融实验（use_ot, use_dual, use_llm, use_cross_attn）
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# 在导入torch之前设置CUDA_VISIBLE_DEVICES，确保只使用GPU 1
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
try:
    import seaborn as sns
except ImportError:
    sns = None
from datetime import datetime
import json
from typing import Dict, List, Tuple

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from src.models.bida.bio_cot_v2 import BioCOT_v2
from experiments.exp_5centers.dataset_v2 import FiveCentersMultimodalDatasetV2
from experiments.exp_5centers.train_bio_cot_5centers_multimodal import extract_features_with_vit
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
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
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
    ax5.plot(epochs, history['cls_loss'], 'b-', label='Classification Loss', linewidth=2)
    if history.get('ot_loss') and len(history['ot_loss']) > 0:
        ax5.plot(epochs, history['ot_loss'], 'g-', label='OT Loss', linewidth=2)
    if history.get('consist_loss') and len(history['consist_loss']) > 0:
        ax5.plot(epochs, history['consist_loss'], 'orange', label='Consistency Loss', linewidth=2)
    if history.get('adv_loss') and len(history['adv_loss']) > 0:
        ax5.plot(epochs, history['adv_loss'], 'r-', label='Adversarial Loss', linewidth=2)
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Loss', fontsize=12)
    ax5.set_title('Loss Components', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')  # 使用对数刻度，因为不同loss的尺度可能差异很大
    
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
    fig.suptitle(f'Bio-COT 2.0 Training Progress (Best AUC: {best_auc:.4f})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # 保存图表
    plot_file = log_dir / f"training_curves_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 可视化图表已保存到: {plot_file}")
    
    # 也保存PDF版本（矢量图，更清晰）
    plot_file_pdf = log_dir / f"training_curves_{timestamp}.pdf"
    plt.savefig(plot_file_pdf, bbox_inches='tight')
    print(f"📊 PDF版本已保存到: {plot_file_pdf}")
    
    plt.close()


class BioCOT_v2_Args:
    """Bio-COT 2.0训练参数"""
    
    def __init__(self):
        # 数据路径
        self.data_root = '/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal'
        # 使用从实际训练/验证/测试集CSV文件生成的嵌入
        self.clinical_embed_path = '/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_5centers/data/clinical_embeddings_from_csv.pkl'
        
        # 模型配置（模块化消融实验）
        # 先使用简单配置，逐步升级
        self.use_llm = True  # True=LLM嵌入, False=传统MLP
        self.use_cross_attn = False  # 先关闭Cross-Attention，使用简单拼接（与v1一致）
        self.use_ot = True  # True=Sinkhorn OT, False=无OT损失
        self.use_dual = True  # True=Dual-Head, False=单头
        
        # LLM配置
        self.llm_embed_dim = 768  # 根据实际LLM调整（bert-base-uncased=768, meditron-7b=4096）
        
        # 训练配置
        self.batch_size = 16  # 从64降低到16，提高训练稳定性
        self.num_epochs = 100  # 减少到100个epoch
        self.learning_rate = 0.00012
        self.num_workers = 4
        self.pin_memory = True
        self.oct_frames = 20
        self.colposcopy_images = 3
        
        # 输出目录
        script_dir = Path(__file__).resolve().parent
        self.output_dir = script_dir / 'results_bio_cot_v2'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = script_dir / 'checkpoints_bio_cot_v2'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = script_dir / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, args, log_print=None):
    """训练一个epoch"""
    if log_print is None:
        log_print = print
    
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    cls_losses = []
    ot_losses = []
    consist_losses = []
    adv_losses = []
    
    log_print(f"\n{'='*80}")
    log_print(f"📊 Epoch {epoch}/{args.num_epochs} - 训练阶段")
    log_print(f"{'='*80}")
    log_print(f"总batch数: {len(dataloader)}")
    log_print(f"Batch size: {args.batch_size}")
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.num_epochs} [Train]')
    
    for batch_idx, batch in enumerate(pbar):
        if batch_idx == 0:
            log_print(f"  📦 Batch {batch_idx+1}/{len(dataloader)}: 开始处理...")
            log_print(f"     - OCT图像形状: {batch['oct_images'].shape}")
            log_print(f"     - Colposcopy图像形状: {batch['colposcopy_images'].shape}")
            log_print(f"     - 标签形状: {batch['label'].shape}")
            if args.use_llm:
                log_print(f"     - LLM嵌入形状: {batch['clinical_embedding'].shape}")
        # 数据传输到GPU
        oct_images = batch['oct_images'].to(device, non_blocking=True)
        colposcopy_images = batch['colposcopy_images'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        center_labels = batch['center_idx'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if batch_idx == 0:
            log_print(f"     - 数据传输到GPU完成")
            log_print(f"     - 开始提取图像特征...")
        
        # 提取图像特征
        oct_features = extract_features_with_vit(oct_images, device)
        colpo_features = extract_features_with_vit(colposcopy_images, device)
        
        if batch_idx == 0:
            log_print(f"     - OCT特征形状: {oct_features.shape}")
            log_print(f"     - Colposcopy特征形状: {colpo_features.shape}")
            log_print(f"     - 开始前向传播...")
        
        # 准备输入数据
        if args.use_llm:
            # LLM路径：使用预计算的嵌入
            clinical_embeddings = batch['clinical_embedding'].to(device, non_blocking=True)
            clinical_data = None
        else:
            # 传统路径：使用原始临床数据
            clinical_embeddings = None
            clinical_data = [item for item in batch['clinical_data']]
            batch_clinical_data = {
                'hpv': [cd.get('hpv', 0) for cd in clinical_data],
                'tct': [cd.get('tct', 'NILM') for cd in clinical_data],
                'age': [cd.get('age', 50.0) for cd in clinical_data]
            }
            clinical_data = batch_clinical_data
        
        # 前向传播
        outputs = model(
            oct_features=oct_features,
            colpo_features=colpo_features,
            clinical_embeddings=clinical_embeddings,
            clinical_data=clinical_data,
            center_labels=center_labels,
            return_loss_components=True,
            use_counterfactual=args.use_dual
        )
        
        logits = outputs['logits']
        
        if batch_idx == 0:
            log_print(f"     - 模型输出logits形状: {logits.shape}")
            if 'z_causal' in outputs:
                log_print(f"     - z_causal形状: {outputs['z_causal'].shape}")
            if 'z_sem' in outputs:
                log_print(f"     - z_sem形状: {outputs['z_sem'].shape}")
            if 'z_noise' in outputs:
                log_print(f"     - z_noise形状: {outputs['z_noise'].shape}")
            if 'loss_components' in outputs:
                log_print(f"     - 损失组件: {list(outputs['loss_components'].keys())}")
        
        # 计算分类损失
        L_cls = criterion(logits, labels)
        
        if batch_idx == 0:
            log_print(f"     - 分类损失 (L_cls): {L_cls.item():.6f}")
        
        # 计算总损失
        total_loss_batch = L_cls
        
        # 添加OT损失
        if args.use_ot and 'L_ot' in outputs.get('loss_components', {}):
            L_ot = outputs['loss_components']['L_ot']
            total_loss_batch = total_loss_batch + 1.0 * L_ot
            ot_losses.append(L_ot.item())
            if batch_idx == 0:
                log_print(f"     - OT损失 (L_ot): {L_ot.item():.6f}")
        else:
            ot_losses.append(0.0)
        
        # 添加一致性损失
        if args.use_dual and 'L_consist' in outputs.get('loss_components', {}):
            L_consist = outputs['loss_components']['L_consist']
            total_loss_batch = total_loss_batch + 0.5 * L_consist
            consist_losses.append(L_consist.item())
            if batch_idx == 0:
                log_print(f"     - 一致性损失 (L_consist): {L_consist.item():.6f}")
        else:
            consist_losses.append(0.0)
        
        # 添加对抗损失
        if args.use_dual and 'L_adv' in outputs.get('loss_components', {}):
            L_adv = outputs['loss_components']['L_adv']
            total_loss_batch = total_loss_batch + 1.0 * L_adv
            adv_losses.append(L_adv.item())
            if batch_idx == 0:
                log_print(f"     - 对抗损失 (L_adv): {L_adv.item():.6f}")
        else:
            adv_losses.append(0.0)
        
        if batch_idx == 0:
            log_print(f"     - 总损失: {total_loss_batch.item():.6f}")
            log_print(f"     - 开始反向传播...")
        
        # 反向传播
        total_loss_batch.backward()
        
        if batch_idx == 0:
            # 检查梯度
            total_grad_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.data.norm(2)
                    total_grad_norm += param_grad_norm.item() ** 2
            total_grad_norm = total_grad_norm ** (1. / 2)
            log_print(f"     - 梯度范数: {total_grad_norm:.6f}")
        
        optimizer.step()
        
        if batch_idx == 0:
            log_print(f"     - 优化器更新完成")
        
        # 统计
        total_loss += total_loss_batch.item()
        cls_losses.append(L_cls.item())
        
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())
        
        # 更新进度条
        current_postfix = {
            'loss': f"{total_loss_batch.item():.4f}",
            'cls': f"{L_cls.item():.4f}",
        }
        
        if args.use_ot:
            current_postfix['ot'] = f"{ot_losses[-1]:.4f}"
        if args.use_dual:
            current_postfix['consist'] = f"{consist_losses[-1]:.4f}"
            current_postfix['adv'] = f"{adv_losses[-1]:.4f}"
        
        current_postfix['acc'] = f"{accuracy_score(all_labels, all_preds):.4f}"
        current_postfix['pos'] = f"{sum(all_preds)}/{len(all_preds)}"
        current_postfix['prob'] = f"{np.mean(all_probs):.3f}"
        
        pbar.set_postfix(current_postfix)
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    
    log_print(f"\n  📊 Epoch {epoch} 训练统计:")
    log_print(f"     - 平均损失: {avg_loss:.6f}")
    log_print(f"     - 准确率: {acc:.4f}")
    log_print(f"     - 分类损失: {np.mean(cls_losses):.6f}")
    if args.use_ot:
        log_print(f"     - OT损失: {np.mean(ot_losses):.6f}")
    if args.use_dual:
        log_print(f"     - 一致性损失: {np.mean(consist_losses):.6f}")
        log_print(f"     - 对抗损失: {np.mean(adv_losses):.6f}")
    log_print(f"     - 预测分布: 阴性={sum(np.array(all_preds)==0)}, 阳性={sum(np.array(all_preds)==1)}")
    log_print(f"     - 平均概率: {np.mean(all_probs):.4f}")
    
    return {
        'loss': avg_loss,
        'acc': acc,
        'cls_loss': np.mean(cls_losses),
        'ot_loss': np.mean(ot_losses) if ot_losses else 0.0,
        'consist_loss': np.mean(consist_losses) if consist_losses else 0.0,
        'adv_loss': np.mean(adv_losses) if adv_losses else 0.0,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs
    }


def validate(model, dataloader, criterion, device, epoch, args, log_print=None):
    """验证"""
    if log_print is None:
        log_print = print
    
    model.eval()
    total_loss = 0.0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    log_print(f"\n  📊 Epoch {epoch}/{args.num_epochs} - 验证阶段")
    log_print(f"     总batch数: {len(dataloader)}")
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.num_epochs} [Val]')
        
        for batch_idx, batch in enumerate(pbar):
            oct_images = batch['oct_images'].to(device, non_blocking=True)
            colposcopy_images = batch['colposcopy_images'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            center_labels = batch['center_idx'].to(device, non_blocking=True)
            
            # 提取特征
            oct_features = extract_features_with_vit(oct_images, device)
            colpo_features = extract_features_with_vit(colposcopy_images, device)
            
            # 准备输入数据
            if args.use_llm:
                clinical_embeddings = batch['clinical_embedding'].to(device, non_blocking=True)
                clinical_data = None
            else:
                clinical_embeddings = None
                clinical_data = [item for item in batch['clinical_data']]
                batch_clinical_data = {
                    'hpv': [cd.get('hpv', 0) for cd in clinical_data],
                    'tct': [cd.get('tct', 'NILM') for cd in clinical_data],
                    'age': [cd.get('age', 50.0) for cd in clinical_data]
                }
                clinical_data = batch_clinical_data
            
            # 前向传播
            outputs = model(
                oct_features=oct_features,
                colpo_features=colpo_features,
                clinical_embeddings=clinical_embeddings,
                clinical_data=clinical_data,
                center_labels=center_labels,
                return_loss_components=False,
                use_counterfactual=False
            )
            
            logits = outputs['logits']
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy_score(all_labels, all_preds):.4f}"
            })
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    
    # 计算AUC和F1
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    try:
        f1 = f1_score(all_labels, all_preds)
    except:
        f1 = 0.0
    
    log_print(f"\n  📊 Epoch {epoch} 验证统计:")
    log_print(f"     - 平均损失: {avg_loss:.6f}")
    log_print(f"     - 准确率: {acc:.4f}")
    log_print(f"     - AUC: {auc:.4f}")
    log_print(f"     - F1-Score: {f1:.4f}")
    log_print(f"     - 预测分布: 阴性={sum(np.array(all_preds)==0)}, 阳性={sum(np.array(all_preds)==1)}")
    log_print(f"     - 平均概率: {np.mean(all_probs):.4f}")
    
    return {
        'loss': avg_loss,
        'acc': acc,
        'auc': auc,
        'f1': f1,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs
    }


def main():
    """主函数"""
    # 确保transforms已导入
    from torchvision import transforms
    
    args = BioCOT_v2_Args()
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.log_dir / f"train_bio_cot_v2_{timestamp}.log"
    
    log_f = open(log_file, 'w', encoding='utf-8')
    
    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_f)
        log_f.flush()
    
    log_print("=" * 80)
    log_print("Bio-COT 2.0: 基于LLM语义锚点与因果最优传输的多模态分类框架")
    log_print("=" * 80)
    log_print(f"数据路径: {args.data_root}")
    log_print(f"LLM嵌入路径: {args.clinical_embed_path}")
    log_print(f"模块配置:")
    log_print(f"  - use_llm: {args.use_llm} ({'LLM嵌入' if args.use_llm else '传统MLP'})")
    log_print(f"  - use_cross_attn: {args.use_cross_attn} ({'Cross-Attention' if args.use_cross_attn else 'Concat'})")
    log_print(f"  - use_ot: {args.use_ot} ({'Sinkhorn OT' if args.use_ot else '无OT'})")
    log_print(f"  - use_dual: {args.use_dual} ({'Dual-Head' if args.use_dual else '单头'})")
    log_print("=" * 80)
    
    # 选择GPU（已通过CUDA_VISIBLE_DEVICES=1限制，只使用物理GPU 1）
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # 因为CUDA_VISIBLE_DEVICES=1，所以cuda:0实际对应物理GPU 1
        log_print(f"✅ 使用GPU设备: {device} (物理GPU 1，通过CUDA_VISIBLE_DEVICES=1限制)")
        log_print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
        log_print(f"   总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
        log_print(f"   可见GPU数量: {torch.cuda.device_count()} (应该为1)")
    else:
        raise RuntimeError("❌ CUDA不可用！请检查GPU环境！")
    
    # 加载数据集
    log_print("📊 开始加载数据集...")
    train_csv = Path(args.data_root) / 'internal_train' / 'labels.csv'
    val_csv = Path(args.data_root) / 'internal_val' / 'labels.csv'
    
    log_print(f"  - 训练集CSV: {train_csv}")
    log_print(f"  - 验证集CSV: {val_csv}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    log_print("  - 图像变换配置完成")
    
    log_print("  - 正在初始化训练集...")
    train_dataset = FiveCentersMultimodalDatasetV2(
        csv_path=str(train_csv),
        clinical_embed_path=args.clinical_embed_path if args.use_llm else None,
        transform=transform,
        oct_num_frames=args.oct_frames,
        max_col_images=args.colposcopy_images,
        balance_negative_frames=True,
        use_llm=args.use_llm
    )
    log_print(f"  ✅ 训练集加载完成: {len(train_dataset)} 个样本")
    
    log_print("  - 正在初始化验证集...")
    val_dataset = FiveCentersMultimodalDatasetV2(
        csv_path=str(val_csv),
        clinical_embed_path=args.clinical_embed_path if args.use_llm else None,
        transform=transform,
        oct_num_frames=args.oct_frames,
        max_col_images=args.colposcopy_images,
        balance_negative_frames=True,
        use_llm=args.use_llm
    )
    log_print(f"  ✅ 验证集加载完成: {len(val_dataset)} 个样本")
    
    # 创建加权采样器
    log_print("  - 正在创建加权采样器...")
    train_labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
    class_counts = pd.Series(train_labels).value_counts().sort_index()
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    log_print(f"  ✅ 加权采样器创建完成")
    log_print(f"     - 类别权重: {class_weights.to_dict()}")
    
    # 创建DataLoader
    log_print("  - 正在创建DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else 2,
        drop_last=True
    )
    log_print(f"  ✅ 训练DataLoader创建完成")
    log_print(f"     - Batch size: {args.batch_size}")
    log_print(f"     - Num workers: {args.num_workers}")
    log_print(f"     - Pin memory: {args.pin_memory}")
    log_print(f"     - 总batch数: {len(train_loader)}")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else 2
    )
    log_print(f"  ✅ 验证DataLoader创建完成")
    log_print(f"     - 总batch数: {len(val_loader)}")
    
    # 检测特征维度
    log_print("\n📊 正在检测特征维度...")
    log_print("  - 正在获取样本batch...")
    sample_batch = next(iter(train_loader))
    log_print(f"  - 样本batch获取成功")
    log_print(f"     - OCT图像形状: {sample_batch['oct_images'][:1].shape}")
    log_print("  - 开始提取OCT特征...")
    oct_features = extract_features_with_vit(sample_batch['oct_images'][:1], device)
    input_dim = oct_features.shape[-1]
    log_print(f"✅ 检测到图像特征维度: {input_dim}")
    
    # 检测LLM嵌入维度
    if args.use_llm:
        log_print("  - 正在检测LLM嵌入维度...")
        sample_embed = sample_batch['clinical_embedding'][0]
        llm_embed_dim = sample_embed.shape[0]
        log_print(f"✅ 检测到LLM嵌入维度: {llm_embed_dim}")
        args.llm_embed_dim = llm_embed_dim
    else:
        llm_embed_dim = None
        log_print("  - 使用传统MLP，无需检测LLM嵌入维度")
    
    # 创建模型
    log_print("\n📊 正在创建Bio-COT 2.0模型...")
    log_print(f"  - 模型配置:")
    log_print(f"     - embed_dim: 768")
    log_print(f"     - num_classes: 2")
    log_print(f"     - num_centers: 4")
    log_print(f"     - input_dim: {input_dim}")
    log_print(f"     - llm_embed_dim: {llm_embed_dim if args.use_llm else None}")
    log_print(f"     - use_ot: {args.use_ot}")
    log_print(f"     - use_dual: {args.use_dual}")
    log_print(f"     - use_llm: {args.use_llm}")
    log_print(f"     - use_cross_attn: {args.use_cross_attn}")
    model = BioCOT_v2(
        embed_dim=768,
        num_classes=2,
        num_centers=4,  # 根据实际中心数调整
        input_dim=input_dim,
        llm_embed_dim=llm_embed_dim if args.use_llm else None,
        use_ot=args.use_ot,
        use_dual=args.use_dual,
        use_llm=args.use_llm,
        use_cross_attn=args.use_cross_attn
    )
    
    log_print("  - 正在将模型移动到GPU...")
    model = model.to(device)
    log_print(f"  ✅ 模型已移动到 {device}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f"✅ 模型创建完成")
    log_print(f"   总参数量: {total_params:,}")
    log_print(f"   可训练参数量: {trainable_params:,}")
    
    # 优化器
    log_print("\n📊 配置优化器和损失函数...")
    # 使用平方根缩放而非线性缩放，更保守的学习率策略
    # 线性缩放: lr = base_lr * (batch_size / 8)  # 对于batch_size=16: 0.00024
    # 平方根缩放: lr = base_lr * sqrt(batch_size / 8)  # 对于batch_size=16: 约0.00017
    # 保守策略: 使用2倍基础学习率
    scaled_lr = args.learning_rate * 2  # 0.00024 (更保守，避免训练不稳定)
    optimizer = optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=1e-5)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    log_print(f"  ✅ 优化器: AdamW")
    log_print(f"     - 基础学习率: {args.learning_rate}")
    log_print(f"     - 缩放后学习率: {scaled_lr:.6f} (batch_size={args.batch_size}, 线性缩放)")
    log_print(f"     - 权重衰减: 1e-5")
    log_print(f"  ✅ 损失函数: FocalLoss (alpha=0.25, gamma=2.0)")
    
    # 训练循环
    log_print("\n" + "=" * 80)
    log_print("🚀 开始训练...")
    log_print("=" * 80)
    log_print(f"总epoch数: {args.num_epochs}")
    log_print(f"训练集大小: {len(train_dataset)}")
    log_print(f"验证集大小: {len(val_dataset)}")
    log_print(f"每epoch训练batch数: {len(train_loader)}")
    log_print(f"每epoch验证batch数: {len(val_loader)}")
    log_print("=" * 80)
    
    best_auc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_f1': [],
        'cls_loss': [], 'ot_loss': [], 'consist_loss': [], 'adv_loss': []
    }
    
    for epoch in range(1, args.num_epochs + 1):
        # 训练
        train_results = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args, log_print=log_print)
        
        # 验证
        val_results = validate(model, val_loader, criterion, device, epoch, args, log_print=log_print)
        
        # 记录历史
        history['train_loss'].append(train_results['loss'])
        history['train_acc'].append(train_results['acc'])
        history['val_loss'].append(val_results['loss'])
        history['val_acc'].append(val_results['acc'])
        history['val_auc'].append(val_results['auc'])
        history['val_f1'].append(val_results['f1'])
        history['cls_loss'].append(train_results['cls_loss'])
        history['ot_loss'].append(train_results['ot_loss'])
        history['consist_loss'].append(train_results['consist_loss'])
        history['adv_loss'].append(train_results['adv_loss'])
        
        # 打印结果
        log_print(f"\nEpoch {epoch}/{args.num_epochs}:")
        log_print(f"  Train - Loss: {train_results['loss']:.4f}, Acc: {train_results['acc']:.4f}")
        log_print(f"  Val   - Loss: {val_results['loss']:.4f}, Acc: {val_results['acc']:.4f}, AUC: {val_results['auc']:.4f}, F1: {val_results['f1']:.4f}")
        
        loss_components = []
        if args.use_ot:
            loss_components.append(f"OT: {train_results['ot_loss']:.4f}")
        if args.use_dual:
            loss_components.append(f"Consist: {train_results['consist_loss']:.4f}")
            loss_components.append(f"Adv: {train_results['adv_loss']:.4f}")
        loss_components.append(f"CLS: {train_results['cls_loss']:.4f}")
        log_print(f"  Loss Components - {', '.join(loss_components)}")
        
        # 保存最佳模型
        if val_results['auc'] > best_auc:
            best_auc = val_results['auc']
            checkpoint_path = args.checkpoint_dir / f"best_model_v2_{timestamp}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'args': args.__dict__,
                'history': history
            }, checkpoint_path)
            log_print(f"  ✅ 保存最佳模型 (AUC: {best_auc:.4f})")
        
        log_print("")
    
    # 在关闭文件之前打印最终结果
    log_print(f"✅ 训练完成！最佳AUC: {best_auc:.4f}")
    log_print(f"📊 训练历史已保存，可用于可视化分析")
    
    # 保存训练历史到JSON文件
    history_file = args.log_dir / f"training_history_{timestamp}.json"
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    log_print(f"📁 训练历史已保存到: {history_file}")
    
    # 关闭日志文件
    log_f.close()
    
    # 生成可视化图表
    try:
        visualize_training(history, args.log_dir, timestamp, best_auc)
        print(f"✅ 可视化图表已生成")
    except Exception as e:
        print(f"⚠️ 可视化生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

