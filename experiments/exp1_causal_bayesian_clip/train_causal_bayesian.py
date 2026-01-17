#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练脚本：因果约束的贝叶斯CLIP
集成到现有数据加载器
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import os
from datetime import datetime

# 导入现有模块
import sys
# 将项目根目录加入路径，保证可以使用`src.*`包导入
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 导入模型和数据
from src.models.causal.bayesian_clip_framework import (
    CausalBayesianCLIP,
    CausalBayesianCLIPLoss
)
from src.data.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
from src.data.enhanced_multimodal_dataset import build_enhanced_dataset


class Args:
    """训练参数"""
    def __init__(self):
        self.data_path = '/data2/hmy/5Center_datas/5centers_multi_leave_centers_out'
        self.batch_size = 8  # 较小批次以处理不确定性
        self.num_epochs = 20
        self.num_classes = 2
        self.learning_rate = 5e-5  # 较小学习率
        self.weight_decay = 1e-5
        self.num_workers = 4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_dir = Path('causal_bayesian_clip_results')
        self.save_dir.mkdir(exist_ok=True)
        # 兼容增强数据集的必要参数
        self.input_size = 224
        self.oct_num_frames = 48
        self.oct_points = 12
        self.oct_frames_per_point = 10
        self.use_text_contrastive = False
        # 训练稳定与不平衡
        self.use_focal = True
        self.focal_gamma = 1.5
        self.label_smoothing = 0.02
        # 对比学习
        self.contrastive_weight = 0.02
        self.temperature = 0.07
        # 两阶段微调
        self.two_stage = True
        self.freeze_epochs = 5
        self.backbone_lr = 5e-6
        self.head_lr = 2e-5
        # 预训练视觉主干开关（使用原始oct/col图像）
        self.use_pretrained_backbones = True
        # 域对抗与EMA、TTA
        self.domain_weight = 0.0
        self.use_ema = False
        self.ema_decay = 0.999
        self.use_tta = False
        # 域对抗&EMA&TTA
        self.domain_weight = 0.05
        self.use_ema = True
        self.ema_decay = 0.999
        self.use_tta = True
        # 采样策略
        self.use_weighted_sampler = True


def train_epoch(model, dataloader, criterion, optimizer, device, grad_accum_steps=4, scaler: GradScaler = None, domain_weight: float = 0.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_kl_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, sample in enumerate(pbar):
        # 兼容tuple或dict
        if isinstance(sample, dict):
            oct_feat = sample['oct_features']
            colpo_feat = sample['colposcopy_features']
            clinical_feat = sample['clinical_features']
            labels = sample['label']
            oct_images = sample.get('oct_images', None)
            col_images = sample.get('col_images', None)
        else:
            # tuple: (oct, colpo, clinical, label) 或 (.., .., .., .., text)
            oct_feat, colpo_feat, clinical_feat, labels = sample[:4]
            oct_images = None
            col_images = None
            center_ids = None
        
        oct_feat = oct_feat.to(device)
        colpo_feat = colpo_feat.to(device)
        clinical_feat = clinical_feat.to(device)
        labels = labels.to(device)
        center_ids = sample.get('center_id', None) if isinstance(sample, dict) else None
        if center_ids is not None:
            center_ids = center_ids.to(device)
        
        # 前向传播（AMP）
        # 关闭AMP以避免早期数值不稳定
        if oct_images is not None:
            oct_images = oct_images.to(device)
        if col_images is not None:
            col_images = col_images.to(device)
        output = model(oct_feat, colpo_feat, clinical_feat, return_uncertainty=False, oct_images=oct_images, col_images=col_images)
        loss_dict = criterion(output, labels)
        loss = loss_dict['total_loss'] / grad_accum_steps
        ce_loss = loss_dict['ce_loss']
        kl_loss = loss_dict['kl_loss']
        # 域对抗损失
        if domain_weight > 0.0 and output.get('domain_logits', None) is not None and center_ids is not None:
            domain_logits = output['domain_logits']
            d_loss = nn.functional.cross_entropy(domain_logits, center_ids)
            loss = loss + (domain_weight * d_loss) / grad_accum_steps
        
        # 反向传播
        # NaN/Inf保护
        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # 统计
        total_loss += (loss.item() * grad_accum_steps)
        total_ce_loss += ce_loss.item()
        total_kl_loss += kl_loss.item()
        
        preds = torch.argmax(output['logits'], dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'ce_loss': total_ce_loss / len(dataloader),
        'kl_loss': total_kl_loss / len(dataloader),
        'accuracy': 100 * correct / total
    }


def validate(model, dataloader, criterion, device, num_classes: int = 2, use_tta: bool = False):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []  # binary: [p1], multiclass: [C]
    uncertainties = []
    
    with torch.no_grad():
        for sample in tqdm(dataloader, desc='Validation'):
            if isinstance(sample, dict):
                oct_feat = sample['oct_features']
                colpo_feat = sample['colposcopy_features']
                clinical_feat = sample['clinical_features']
                labels = sample['label']
                oct_images = sample.get('oct_images', None)
                col_images = sample.get('col_images', None)
            else:
                oct_feat, colpo_feat, clinical_feat, labels = sample[:4]
                oct_images = None
                col_images = None
            
            oct_feat = oct_feat.to(device)
            colpo_feat = colpo_feat.to(device)
            clinical_feat = clinical_feat.to(device)
            labels = labels.to(device)
            
            if oct_images is not None:
                oct_images = oct_images.to(device)
            if col_images is not None:
                col_images = col_images.to(device)
            # 简单TTA：原始 + 水平翻转，logits平均
            if use_tta and (oct_images is not None or col_images is not None):
                logits_list = []
                out0 = model(oct_feat, colpo_feat, clinical_feat, return_uncertainty=False, oct_images=oct_images, col_images=col_images)
                logits_list.append(out0['logits'])
                oct_flip = torch.flip(oct_images, dims=[-1]) if oct_images is not None else None
                col_flip = torch.flip(col_images, dims=[-1]) if col_images is not None else None
                out1 = model(oct_feat, colpo_feat, clinical_feat, return_uncertainty=False, oct_images=oct_flip, col_images=col_flip)
                logits_list.append(out1['logits'])
                logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)
                output = {'logits': logits, 'uncertainty': None}
            else:
                output = model(oct_feat, colpo_feat, clinical_feat, return_uncertainty=False, oct_images=oct_images, col_images=col_images)
            
            # 计算验证损失：若输出包含mean/var，使用完整criterion；否则退回到CE
            try:
                loss_dict = criterion(output, labels)
                loss = loss_dict['total_loss']
            except KeyError:
                logits_ce = output['logits']
                loss = nn.functional.cross_entropy(logits_ce, labels)
            
            if torch.isfinite(loss):
                total_loss += loss.item()
            
            logits = output['logits']
            sm = torch.softmax(logits, dim=1)
            if num_classes == 2:
                probs_np = sm[:, 1].detach().cpu().numpy()
            else:
                probs_np = sm.detach().cpu().numpy()  # shape [B, C]
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            if num_classes == 2:
                all_probs.extend(probs_np.tolist())
            else:
                all_probs.extend(probs_np.tolist())
            
            if output['uncertainty'] is not None:
                uncertainties.extend(output['uncertainty'].cpu().numpy())
    
    avg_uncertainty = np.mean(uncertainties) if uncertainties else 0
    # 计算更多指标（兼容二/多分类）
    best_thr = None
    try:
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
        y_true = np.array(all_labels)
        if num_classes == 2:
            y_prob = np.array(all_probs)
            # 阈值扫描（提升F1/Acc），搜索区间0.1~0.9
            best = {'thr': 0.5, 'f1': -1, 'acc': -1, 'prec': 0.0, 'rec': 0.0}
            for thr in np.linspace(0.1, 0.9, 17):
                yp = (y_prob >= thr).astype(int)
                p, r, f, _ = precision_recall_fscore_support(y_true, yp, average='binary', zero_division=0)
                a = accuracy_score(y_true, yp)
                if f > best['f1'] or (f == best['f1'] and a > best['acc']):
                    best = {'thr': float(thr), 'f1': float(f), 'acc': float(a), 'prec': float(p), 'rec': float(r)}
            # 使用最佳阈值统计
            best_thr = best['thr']
            y_pred = (y_prob >= best['thr']).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float('nan')
            acc = accuracy_score(y_true, y_pred)
        else:
            y_prob = np.array(all_probs)  # [N, C]
            y_pred = y_prob.argmax(axis=1)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
            # AUC-ovr
            try:
                auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            except Exception:
                auc = float('nan')
            acc = accuracy_score(y_true, y_pred)
    except Exception:
        precision = recall = f1 = auc = float('nan')
        acc = 100.0 * correct / total
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100 * correct / total,
        'uncertainty': avg_uncertainty,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'acc_bin': acc,
        'best_thr': best_thr
    }

class EMAHelper:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]

    def apply_to(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self, model: nn.Module):
        if hasattr(self, 'backup'):
            for name, param in model.named_parameters():
                if name in self.backup:
                    param.data = self.backup[name].clone()
            self.backup = {}
def set_backbone_trainable(model: CausalBayesianCLIP, requires_grad: bool):
    # 定义主干：模态编码器与临床投影层
    backbone_modules = [
        model.oct_encoder,
        model.colposcopy_encoder,
        model.clinical_encoder,
        model.clinical_proj
    ]
    # 将视觉主干与投影层纳入可控范围（若存在）
    if hasattr(model, 'vision_oct_backbone'):
        backbone_modules.append(model.vision_oct_backbone)
    if hasattr(model, 'vision_col_backbone'):
        backbone_modules.append(model.vision_col_backbone)
    if hasattr(model, 'vision_proj'):
        backbone_modules.append(model.vision_proj)
    for m in backbone_modules:
        for p in m.parameters():
            p.requires_grad = requires_grad


def build_optimizer(args: Args, model: CausalBayesianCLIP, stage: str = 'head'):
    # 参数组：主干与头部不同学习率
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ['oct_encoder', 'colposcopy_encoder', 'clinical_encoder', 'clinical_proj']):
            backbone_params.append(p)
        else:
            head_params.append(p)
    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': args.backbone_lr, 'weight_decay': args.weight_decay})
    if head_params:
        param_groups.append({'params': head_params, 'lr': args.head_lr, 'weight_decay': args.weight_decay})
    optimizer = AdamW(param_groups)
    return optimizer


def main():
    import argparse
    parser = argparse.ArgumentParser(description='因果约束的贝叶斯CLIP训练')
    parser.add_argument('--data_path', type=str, default=None, help='数据路径')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=None, help='学习率')
    parser.add_argument('--device', type=str, default=None, help='设备')
    parser.add_argument('--num_workers', type=int, default=None, help='数据加载workers')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    cmd_args = parser.parse_args()
    
    args = Args()
    
    # 从命令行参数覆盖默认值
    if cmd_args.data_path is not None:
        args.data_path = cmd_args.data_path
    if cmd_args.batch_size is not None:
        args.batch_size = cmd_args.batch_size
    if cmd_args.num_epochs is not None:
        args.num_epochs = cmd_args.num_epochs
    if cmd_args.learning_rate is not None:
        args.learning_rate = cmd_args.learning_rate
    if cmd_args.device is not None:
        args.device = cmd_args.device
    if cmd_args.num_workers is not None:
        args.num_workers = cmd_args.num_workers
    if cmd_args.output_dir is not None:
        args.save_dir = Path(cmd_args.output_dir)
        args.save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🚀 因果约束的贝叶斯CLIP训练")
    print("=" * 80)
    print(f"📁 数据路径: {args.data_path}")
    print(f"💻 使用设备: {args.device}")
    print(f"📦 批次大小: {args.batch_size}")
    print(f"🎯 训练轮数: {args.num_epochs}")
    print(f"📚 学习率: {args.learning_rate}")
    print("=" * 80)
    
    # 创建数据集
    print("\n📥 加载数据集...")
    # 确保OCT缓存目录
    if not hasattr(args, 'oct_cache_dir'):
        args.oct_cache_dir = 'oct_cache_optimized'
    Path(args.oct_cache_dir).mkdir(exist_ok=True)

    # 支持Leave-Centers-Out数据集结构
    train_dataset = build_enhanced_dataset('train', args, use_external_test=False)
    # 使用内部验证集（internal_train/val/）
    val_dataset = build_enhanced_dataset('val', args, use_external_test=False)
    # 临时禁用增强OCT以绕过 int.unsqueeze 错误
    train_dataset.use_enhanced_oct = False
    val_dataset.use_enhanced_oct = False
    
    # 使用类别加权采样，缓解不平衡
    from torch.utils.data import WeightedRandomSampler
    import pandas as pd
    import numpy as np
    df_train_sampler = pd.read_csv(Path(args.data_path) / 'train_labels.csv')
    labels_np_sampler = df_train_sampler['label'].astype(int).values
    pos_s = max((labels_np_sampler == 1).sum(), 1)
    neg_s = max((labels_np_sampler == 0).sum(), 1)
    weight_per_class = {0: 1.0 / neg_s, 1: 1.0 / pos_s}
    sample_weights = np.array([weight_per_class[int(l)] for l in labels_np_sampler], dtype=np.float32)
    sampler = WeightedRandomSampler(sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)

    if args.use_weighted_sampler:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=0
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✅ 训练集: {len(train_dataset)} 样本")
    print(f"✅ 验证集: {len(val_dataset)} 样本")

    # 动态确定类别数（与数据对齐）
    try:
        import pandas as pd
        train_csv = Path(args.data_path) / 'train_labels.csv'
        val_csv = Path(args.data_path) / 'val_labels.csv'
        
        # 从训练集读取标签
        y_train = pd.read_csv(train_csv)['label'].astype(int).unique().tolist()
        
        # 尝试从验证集读取标签（如果存在）
        y_val = []
        if val_csv.exists():
            y_val = pd.read_csv(val_csv)['label'].astype(int).unique().tolist()
        
        # 尝试从测试集读取标签（如果存在，用于Leave-Centers-Out数据集）
        test_csv = Path(args.data_path) / 'test_labels.csv'
        y_test = []
        if test_csv.exists():
            y_test = pd.read_csv(test_csv)['label'].astype(int).unique().tolist()
        
        # 合并所有标签
        uniq = sorted(set(y_train) | set(y_val) | set(y_test))
        args.num_classes = max(len(uniq), 2)
        print(f"🔧 自动检测到类别数: {args.num_classes} (labels={uniq}, 来源: train={y_train}, val={y_val}, test={y_test})")
    except Exception as e:
        print(f"⚠️ 类别自动检测失败，使用默认: {args.num_classes}. 错误: {e}")
    
    # 创建模型
    print("\n🤖 创建模型...")
    # 统计中心数量
    num_centers = len(getattr(train_dataset, 'center_to_idx', {}) or {})

    model = CausalBayesianCLIP(
        embed_dim=768,
        clinical_dim=7,
        num_classes=args.num_classes,
        temperature=0.07,
        kl_weight=0.0,
        use_pretrained_backbones=getattr(args, 'use_pretrained_backbones', False),
        num_centers=num_centers
    ).to(args.device)
    
    print(f"📊 模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 优化器
    # 统计类别权重
    import pandas as pd
    import numpy as np
    train_csv = Path(args.data_path) / 'train_labels.csv'
    df = pd.read_csv(train_csv)
    labels = df['label'].astype(int).values
    pos = max(np.sum(labels == 1), 1)
    neg = max(np.sum(labels == 0), 1)
    w_pos = neg / (pos + neg)
    w_neg = pos / (pos + neg)
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=args.device)

    # 使用Focal/标签平滑，先关闭KL（0）并加入可选对比损失
    criterion = CausalBayesianCLIPLoss(
        temperature=args.temperature,
        kl_weight=0.0,
        class_weights=class_weights,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        contrastive_weight=args.contrastive_weight
    )

    # 两阶段微调：阶段A冻结主干，仅训练头部
    if args.two_stage:
        set_backbone_trainable(model, requires_grad=False)
        optimizer = build_optimizer(args, model, stage='head')
    else:
        set_backbone_trainable(model, requires_grad=True)
        optimizer = build_optimizer(args, model, stage='full')

    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    # 初期禁用AMP，待训练稳定再启用
    scaler = None
    
    # 训练
    print("\n🎓 开始训练...\n")
    best_val_acc = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_uncertainty': []
    }
    
    # EMA
    ema = EMAHelper(model, decay=args.ema_decay) if args.use_ema else None

    for epoch in range(args.num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{args.num_epochs}")
        print("-" * 60)

        # 阶段切换：到达freeze_epochs后解冻主干并重建优化器
        if args.two_stage and epoch == args.freeze_epochs:
            set_backbone_trainable(model, requires_grad=True)
            optimizer = build_optimizer(args, model, stage='full')
            print("🔁 已解冻主干，进入微调阶段（低LR）")
        
        # ramp-up 对比与域对抗权重（前5个epoch线性增长）
        ramp_epochs = max(1, min(5, args.num_epochs))
        ramp = min(1.0, (epoch + 1) / ramp_epochs)
        # 动态更新对比损失权重
        try:
            criterion.contrastive_weight = float(args.contrastive_weight) * float(ramp)
        except Exception:
            pass
        
        # 简单Warmup（首Epoch前10% step线性增长）
        current_domain_weight = float(args.domain_weight) * float(ramp)
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, args.device, grad_accum_steps=4, scaler=scaler, domain_weight=current_domain_weight)
        if ema is not None:
            ema.update(model)
        
        # 训练集（无TTA）评估：每epoch自动估计阈值并统计Acc/F1（便于直观对照）
        model.eval()
        train_eval_metrics = validate(model, train_loader, criterion, args.device, num_classes=args.num_classes, use_tta=False)
        model.train()

        # 验证
        if ema is not None:
            ema.apply_to(model)
        val_metrics = validate(model, val_loader, criterion, args.device, num_classes=args.num_classes, use_tta=args.use_tta)
        if ema is not None:
            ema.restore(model)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_uncertainty'].append(val_metrics['uncertainty'])
        
        # 打印结果
        print(f"\n📊 Train: Loss={train_metrics['loss']:.4f}, Acc(argmax)={train_metrics['accuracy']:.2f}%, Acc(best-thr)={train_eval_metrics.get('acc_bin', float('nan')):.2f}%, F1={train_eval_metrics.get('f1', float('nan')):.3f}, thr={train_eval_metrics.get('best_thr', None)}")
        print(f"📊 Valid: Loss={val_metrics['loss']:.4f}, Acc(best-thr)={val_metrics.get('acc_bin', float('nan')):.2f}%, F1={val_metrics.get('f1', float('nan')):.3f}, AUC={val_metrics.get('auc', float('nan')):.3f}, thr={val_metrics.get('best_thr', None)} | ramp={ramp:.2f}, contra_w={getattr(criterion,'contrastive_weight',0):.3f}, domain_w={current_domain_weight:.3f}")
        
        # 保存最佳模型
        if val_metrics.get('acc_bin', 0) > best_val_acc:
            best_val_acc = val_metrics.get('acc_bin', 0)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'history': history
            }, args.save_dir / 'best_model.pth')
            print(f"\n💾 保存最佳模型 (Val Acc: {best_val_acc:.2f}%)")
    
    # 保存历史
    with open(args.save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ 训练完成！")
    print(f"📁 结果保存在: {args.save_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

