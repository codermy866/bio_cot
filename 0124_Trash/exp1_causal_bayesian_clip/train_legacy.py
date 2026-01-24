#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强因果CLIP训练脚本
核心创新：可学习因果图发现 + 不确定性分解
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from collections import defaultdict
from contextlib import nullcontext
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, 
    recall_score, confusion_matrix, roc_curve
)
import time
from datetime import datetime
import os
import json
import sys
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入模型和数据
try:
    from src.models.enhanced_causal_clip import EnhancedCausalBayesianCLIP
    from src.models.causal_bayesian_clip_framework import CausalBayesianCLIPLoss
except ImportError:
    # 如果导入失败，尝试直接导入
    sys.path.insert(0, str(project_root / 'src'))
    from models.enhanced_causal_clip import EnhancedCausalBayesianCLIP
    from models.causal_bayesian_clip_framework import CausalBayesianCLIPLoss

from utils.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
from utils.advanced_clinical_metrics import calculate_clinical_metrics, calculate_calibration_metrics

# 导入特征提取器
from models.SwinT.swin_image_encoder import SwinTImageEncoder


class FeatureExtractor(nn.Module):
    """
    特征提取器：从原始图像提取特征
    使用Swin-T作为backbone（因为性能最好）
    修复：支持特征维度投影 + 部分微调（解冻最后几层）
    """
    def __init__(self, embed_dim=768, device='cuda', fine_tune_last_n_layers=2):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.fine_tune_last_n_layers = fine_tune_last_n_layers
        
        # OCT编码器
        self.oct_encoder = SwinTImageEncoder(
            embed_dim=embed_dim,
            num_frames=48,
            model_name='swin_tiny_patch4_window7_224',
            pretrained=True,
            input_size=224,
            use_frame_attention=False
        ).to(device)
        
        # Colposcopy编码器
        self.col_encoder = SwinTImageEncoder(
            embed_dim=embed_dim,
            num_frames=3,
            model_name='swin_tiny_patch4_window7_224',
            pretrained=True,
            input_size=224,
            use_frame_attention=False
        ).to(device)
        
        # 特征投影层（预先创建，避免动态创建导致的不一致）
        # 假设Swin-T输出维度为768，如果需要投影，会在forward中动态调整
        self.oct_proj = None
        self.col_proj = None
        
        # 部分微调：解冻最后几层
        self._setup_fine_tuning()
    
    def _setup_fine_tuning(self):
        """设置部分微调：冻结大部分层，只解冻最后几层"""
        # 冻结所有参数
        for param in self.oct_encoder.parameters():
            param.requires_grad = False
        for param in self.col_encoder.parameters():
            param.requires_grad = False
        
        # 解冻最后几层（如果存在）
        # SwinTImageEncoder的结构：self.backbone（timm模型）
        try:
            # 尝试解冻OCT编码器的最后几层
            if hasattr(self.oct_encoder, 'backbone'):
                backbone = self.oct_encoder.backbone
                # timm的Swin模型通常有layers属性
                if hasattr(backbone, 'layers'):
                    layers = backbone.layers
                    for layer in layers[-self.fine_tune_last_n_layers:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                # 或者有stages属性
                elif hasattr(backbone, 'stages'):
                    stages = backbone.stages
                    for stage in stages[-self.fine_tune_last_n_layers:]:
                        for param in stage.parameters():
                            param.requires_grad = True
                # 解冻投影层（如果有）
                if hasattr(self.oct_encoder, 'proj'):
                    for param in self.oct_encoder.proj.parameters():
                        param.requires_grad = True
            
            # 尝试解冻Colposcopy编码器的最后几层
            if hasattr(self.col_encoder, 'backbone'):
                backbone = self.col_encoder.backbone
                if hasattr(backbone, 'layers'):
                    layers = backbone.layers
                    for layer in layers[-self.fine_tune_last_n_layers:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                elif hasattr(backbone, 'stages'):
                    stages = backbone.stages
                    for stage in stages[-self.fine_tune_last_n_layers:]:
                        for param in stage.parameters():
                            param.requires_grad = True
                # 解冻投影层（如果有）
                if hasattr(self.col_encoder, 'proj'):
                    for param in self.col_encoder.proj.parameters():
                        param.requires_grad = True
        except Exception as e:
            # 如果结构不同，保持冻结状态
            print(f"⚠️  部分微调设置失败: {e}，将保持完全冻结状态")
    
    def forward(self, oct_images, col_images):
        """
        提取特征
        
        Args:
            oct_images: [B, T, C, H, W]
            col_images: [B, T, C, H, W]
        
        Returns:
            oct_feat: [B, embed_dim]
            col_feat: [B, embed_dim]
        """
        # 如果部分层需要微调，不使用no_grad
        use_grad = any(p.requires_grad for p in self.oct_encoder.parameters()) or \
                   any(p.requires_grad for p in self.col_encoder.parameters())
        
        if use_grad:
            # 部分微调模式：允许梯度传播
            oct_feat = self.oct_encoder(oct_images)
            col_feat = self.col_encoder(col_images)
        else:
            # 完全冻结模式：使用no_grad
            with torch.no_grad():
                oct_feat = self.oct_encoder(oct_images)
                col_feat = self.col_encoder(col_images)
        
        # 如果维度不匹配，使用投影层
        if oct_feat.size(-1) != self.embed_dim:
            if self.oct_proj is None:
                self.oct_proj = nn.Linear(oct_feat.size(-1), self.embed_dim).to(self.device)
            oct_feat = self.oct_proj(oct_feat)
        
        if col_feat.size(-1) != self.embed_dim:
            if self.col_proj is None:
                self.col_proj = nn.Linear(col_feat.size(-1), self.embed_dim).to(self.device)
            col_feat = self.col_proj(col_feat)
        
        return oct_feat, col_feat


def prepare_data_loaders(data_path='5centers_multi', batch_size=6, num_workers=4):
    """准备数据加载器"""
    print("📥 加载数据集...")
    
    # 创建Args对象
    class Args:
        def __init__(self):
            self.data_path = data_path
            self.input_size = 224
            self.oct_num_frames = 120  # 使用全部帧
            self.col_num_frames = 3
            self.oct_cache_dir = 'oct_cache_optimized'
            self.use_text_contrastive = False
            self.oct_points = 12
            self.oct_frames_per_point = 10
            self.use_pretrained_backbones = False
            self.cache_oct_features = False  # 关闭缓存，强制读原图
    
    args = Args()
    
    # 使用build_enhanced_dataset函数
    try:
        from src.data.enhanced_multimodal_dataset import build_enhanced_dataset
        train_dataset = build_enhanced_dataset('train', args)
        val_dataset = build_enhanced_dataset('test', args)
    except ImportError:
        # 如果导入失败，直接创建
        from src.data.enhanced_multimodal_dataset import create_enhanced_transform
        train_transform = create_enhanced_transform(args, is_training=True)
        val_transform = create_enhanced_transform(args, is_training=False)
        
        train_dataset = EnhancedMultimodalCervicalDataset(
            root=os.path.join(data_path, 'train'),
            is_train='train',
            args=args,
            transform=train_transform,
            use_enhanced_oct=True,
            cache_oct_features=True
        )
        
        val_dataset = EnhancedMultimodalCervicalDataset(
            root=os.path.join(data_path, 'test'),
            is_train='test',
            args=args,
            transform=val_transform,
            use_enhanced_oct=True,
            cache_oct_features=True
        )
    
    # 自定义collate函数，确保tensor不requires_grad
    def collate_fn(batch):
        """自定义collate函数，处理requires_grad问题"""
        from torch.utils.data._utils.collate import default_collate
        
        # 分离数据
        items = []
        for item in batch:
            if isinstance(item, dict):
                # 如果是字典，确保所有tensor不requires_grad
                new_item = {}
                for k, v in item.items():
                    if torch.is_tensor(v):
                        new_item[k] = v.detach()
                    else:
                        new_item[k] = v
                items.append(new_item)
            elif torch.is_tensor(item):
                items.append(item.detach())
            else:
                items.append(item)
        
        return default_collate(items)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 暂时设为0避免多进程问题
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 暂时设为0避免多进程问题
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"✅ 训练集: {len(train_dataset)} 样本")
    print(f"✅ 验证集: {len(val_dataset)} 样本")
    
    return train_loader, val_loader


def train_one_epoch(model, feature_extractor, train_loader, criterion, optimizer, 
                   scaler, device, epoch, num_epochs, use_amp=True, causal_loss_weight=0.005):
    """训练一个epoch，记录分项损失与梯度范数"""
    model.train()
    # 如果特征提取器有可训练参数，设置为train模式；否则为eval模式
    if any(p.requires_grad for p in feature_extractor.parameters()):
        feature_extractor.train()
    else:
        feature_extractor.eval()
    
    total_loss = 0
    loss_comp_sum = defaultdict(float)
    loss_comp_count = 0
    grad_norms = []
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    
    for batch_idx, batch in enumerate(pbar):
        # 获取数据（处理字典或元组格式）
        if isinstance(batch, dict):
            # 字典格式
            if 'oct_images' in batch:
                oct_images = batch['oct_images'].to(device)
                col_images = batch['col_images'].to(device)
            else:
                # 可能是特征格式
                oct_feat = batch.get('oct_features', batch.get('oct_feat'))
                col_feat = batch.get('colposcopy_features', batch.get('col_feat'))
                if oct_feat is None or col_feat is None:
                    print(f"⚠️  警告: 批次 {batch_idx} 格式不正确，跳过")
                    continue
                # 直接使用特征，不需要特征提取器
                oct_feat = oct_feat.to(device)
                col_feat = col_feat.to(device)
                oct_images = None
                col_images = None
            
            clinical_features = batch.get('clinical_features', batch.get('clinical'))
            if clinical_features is None:
                print("⚠️  警告: 未找到临床特征，使用零向量")
                clinical_features = torch.zeros(oct_feat.size(0) if oct_images is None else oct_images.size(0), 7).to(device)
            else:
                clinical_features = clinical_features.to(device)
            
            labels = batch.get('label', batch.get('labels'))
            if labels is None:
                print(f"⚠️  警告: 批次 {batch_idx} 没有标签，跳过")
                continue
            labels = labels.to(device)
        else:
            # 元组格式：(oct_features, col_features, clinical_features, label)
            if len(batch) == 4:
                oct_feat, col_feat, clinical_features, labels = batch
                oct_feat = oct_feat.to(device)
                col_feat = col_feat.to(device)
                clinical_features = clinical_features.to(device)
                labels = labels.to(device)
                oct_images = None
                col_images = None
            else:
                print(f"⚠️  警告: 批次 {batch_idx} 格式不正确（长度={len(batch)}），跳过")
                continue
        
        # 提取特征（如果需要）
        if oct_images is not None:
            # 仅首个batch打印shape以确认帧数是否为120
            if batch_idx == 0:
                print(f"[DEBUG] oct_images shape: {oct_images.shape}, col_images shape: {col_images.shape}")
            oct_feat, col_feat = feature_extractor(oct_images, col_images)
        else:
            # 如果数据集返回的是特征，需要投影到768维
            # 确保特征是2D的 [B, D]
            if oct_feat.dim() == 1:
                oct_feat = oct_feat.unsqueeze(0)
            if col_feat.dim() == 1:
                col_feat = col_feat.unsqueeze(0)
            
            # 确保特征维度正确
            if oct_feat.size(-1) != 768:
                # 使用特征提取器的投影层
                if feature_extractor.oct_proj is None:
                    feature_extractor.oct_proj = nn.Linear(oct_feat.size(-1), 768).to(device)
                oct_feat = feature_extractor.oct_proj(oct_feat)
            
            if col_feat.size(-1) != 768:
                if feature_extractor.col_proj is None:
                    feature_extractor.col_proj = nn.Linear(col_feat.size(-1), 768).to(device)
                col_feat = feature_extractor.col_proj(col_feat)
        
        # 构造伪干预mask（随机选择1个模态置为干预，模拟do操作）
        intervention_mask = None
        if oct_feat is not None:
            B = oct_feat.size(0)
            intervention_mask = torch.zeros(B, 3, device=device)
            # 以一定概率应用干预，避免过强约束
            if np.random.rand() < 0.5:
                rand_idx = torch.randint(low=0, high=3, size=(B,), device=device)
                intervention_mask.scatter_(1, rand_idx.unsqueeze(1), 1.0)
            else:
                intervention_mask = None
        
        # 前向传播
        optimizer.zero_grad()
        
        ctx = autocast(enabled=use_amp) if use_amp else nullcontext()
        with ctx:
            output = model(
                oct_feat, col_feat, clinical_features,
                return_uncertainty=True,
                return_causal_penalty=True,
                intervention_mask=intervention_mask
            )
            logits = output['logits']
            
            # 计算损失
            loss_dict = criterion(output, labels)
            loss = loss_dict['total_loss']
            
            # 添加因果图正则化损失
            # 分项损失累计
            for k, v in loss_dict.items():
                if torch.is_tensor(v):
                    loss_comp_sum[k] += float(v.detach().cpu())
            loss_comp_count += 1
            
            causal_penalty_total = None
            if 'causal_penalties' in output and output['causal_penalties'] is not None:
                causal_penalty_total = output['causal_penalties'].get('total', None)
            elif 'causal_penalty' in output and output['causal_penalty'] is not None:
                causal_penalty_total = output['causal_penalty']
            
            if causal_penalty_total is not None:
                loss = loss + causal_loss_weight * causal_penalty_total
            
            # 检查NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  跳过NaN/Inf loss (batch {batch_idx})")
                continue
        
        # 反向传播
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            all_params = list(model.parameters())
            if any(p.requires_grad for p in feature_extractor.parameters()):
                all_params.extend([p for p in feature_extractor.parameters() if p.requires_grad])
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            if torch.isfinite(grad_norm):
                grad_norms.append(float(grad_norm.detach().cpu()))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            all_params = list(model.parameters())
            if any(p.requires_grad for p in feature_extractor.parameters()):
                all_params.extend([p for p in feature_extractor.parameters() if p.requires_grad])
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            if torch.isfinite(grad_norm):
                grad_norms.append(float(grad_norm.detach().cpu()))
            optimizer.step()
        
        # 统计
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.detach().detach().cpu().numpy())
        all_labels.extend(labels.detach().detach().cpu().numpy())
        all_probs.extend(probs[:, 1].detach().detach().cpu().numpy())
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy_score(all_labels, all_preds):.2%}'
        })
    
    avg_loss = total_loss / len(train_loader)
    acc = accuracy_score(all_labels, all_preds)
    
    # 分项损失平均
    loss_comp_avg = {k: v / max(loss_comp_count, 1) for k, v in loss_comp_sum.items()}
    grad_norm_avg = float(np.mean(grad_norms)) if grad_norms else 0.0
    
    stats = {
        'loss_components': loss_comp_avg,
        'grad_norm_avg': grad_norm_avg
    }
    
    return avg_loss, acc, all_probs, all_labels, stats


def validate(model, feature_extractor, val_loader, criterion, device, use_amp=True):
    """验证"""
    model.eval()
    feature_extractor.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_uncertainties = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='验证中'):
            # 获取数据（处理字典或元组格式）
            if isinstance(batch, dict):
                if 'oct_images' in batch:
                    oct_images = batch['oct_images'].to(device)
                    col_images = batch['col_images'].to(device)
                    oct_feat, col_feat = feature_extractor(oct_images, col_images)
                else:
                    oct_feat = batch.get('oct_features', batch.get('oct_feat')).to(device)
                    col_feat = batch.get('colposcopy_features', batch.get('col_feat')).to(device)
                
                clinical_features = batch.get('clinical_features', batch.get('clinical'))
                if clinical_features is None:
                    clinical_features = torch.zeros(oct_feat.size(0), 7).to(device)
                else:
                    clinical_features = clinical_features.to(device)
                
                labels = batch.get('label', batch.get('labels')).to(device)
            else:
                # 元组格式
                if len(batch) == 4:
                    oct_feat, col_feat, clinical_features, labels = batch
                    oct_feat = oct_feat.to(device)
                    col_feat = col_feat.to(device)
                    clinical_features = clinical_features.to(device)
                    labels = labels.to(device)
                    
                    # 确保特征是2D的 [B, D]
                    if oct_feat.dim() == 1:
                        oct_feat = oct_feat.unsqueeze(0)
                    if col_feat.dim() == 1:
                        col_feat = col_feat.unsqueeze(0)
                    
                    # 如果特征维度不是768，需要投影
                    if oct_feat.size(-1) != 768:
                        if feature_extractor.oct_proj is None:
                            feature_extractor.oct_proj = nn.Linear(oct_feat.size(-1), 768).to(device)
                        oct_feat = feature_extractor.oct_proj(oct_feat)
                    
                    if col_feat.size(-1) != 768:
                        if feature_extractor.col_proj is None:
                            feature_extractor.col_proj = nn.Linear(col_feat.size(-1), 768).to(device)
                        col_feat = feature_extractor.col_proj(col_feat)
                else:
                    continue
            
            # 前向传播
            ctx = autocast(enabled=use_amp) if use_amp else nullcontext()
            with ctx:
                output = model(oct_feat, col_feat, clinical_features, return_uncertainty=True, return_causal_penalty=False)
                logits = output['logits']
                
                loss_dict = criterion(output, labels)
                loss = loss_dict['total_loss']
            
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.detach().detach().cpu().numpy())
            all_labels.extend(labels.detach().detach().cpu().numpy())
            all_probs.extend(probs[:, 1].detach().detach().cpu().numpy())
            
            # 收集不确定性
            if output['uncertainty'] is not None:
                if 'total' in output['uncertainty']:
                    all_uncertainties.extend(output['uncertainty']['total'].detach().detach().cpu().numpy().flatten())
                elif isinstance(output['uncertainty'], dict):
                    # 如果有分解的不确定性，使用总不确定性
                    total_unc = output['uncertainty'].get('total', 
                        output['uncertainty'].get('epistemic', torch.zeros(probs.size(0), 1).to(device)))
                    all_uncertainties.extend(total_unc.detach().detach().cpu().numpy().flatten())
    
    avg_loss = total_loss / len(val_loader)
    acc = accuracy_score(all_labels, all_preds)
    
    # 计算AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    # 计算其他指标
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    
    # 计算临床指标
    try:
        clinical_metrics = calculate_clinical_metrics(all_labels, all_preds, all_probs)
    except:
        clinical_metrics = {}
    
    # 计算最优阈值（Youden指数）
    try:
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
        
        # 使用最优阈值重新计算指标
        optimal_preds = (np.array(all_probs) >= optimal_threshold).astype(int)
        optimal_acc = accuracy_score(all_labels, optimal_preds)
        optimal_f1 = f1_score(all_labels, optimal_preds)
        optimal_sensitivity = recall_score(all_labels, optimal_preds, zero_division=0)
        optimal_specificity = precision_score(all_labels, optimal_preds, zero_division=0)
    except:
        optimal_threshold = 0.5
        optimal_acc = acc
        optimal_f1 = f1
        optimal_sensitivity = recall
        optimal_specificity = precision
    
    return {
        'loss': avg_loss,
        'accuracy': acc,
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'optimal_threshold': float(optimal_threshold),
        'optimal_accuracy': optimal_acc,
        'optimal_f1': optimal_f1,
        'optimal_sensitivity': optimal_sensitivity,
        'optimal_specificity': optimal_specificity,
        'clinical_metrics': clinical_metrics,
        'uncertainties': all_uncertainties if all_uncertainties else None
    }


def main():
    """主训练函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='增强因果CLIP训练')
    parser.add_argument('--data_path', type=str, default='5centers_multi', help='数据路径')
    parser.add_argument('--output_dir', type=str, default='enhanced_causal_clip_results', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=24, help='批次大小（防止120帧OCT下OOM）')
    parser.add_argument('--num_epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--no_amp', action='store_true', help='关闭混合精度')
    parser.add_argument('--label_smoothing', type=float, default=0.05, help='标签平滑')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal gamma')
    parser.add_argument('--contrastive_weight', type=float, default=0.05, help='对比损失权重')
    parser.add_argument('--kl_weight', type=float, default=0.003, help='KL损失权重')
    parser.add_argument('--causal_loss_weight', type=float, default=0.002, help='因果正则总权重（乘在总惩罚上）')
    parser.add_argument('--dag_penalty_weight', type=float, default=0.02, help='DAG惩罚权重')
    parser.add_argument('--sparsity_weight', type=float, default=2e-4, help='稀疏惩罚权重')
    parser.add_argument('--intervention_weight', type=float, default=0.01, help='干预惩罚权重')
    parser.add_argument('--embed_dim', type=int, default=768, help='嵌入维度')
    parser.add_argument('--clinical_dim', type=int, default=7, help='临床特征维度')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()
    
    # 参数设置
    data_path = args.data_path
    output_dir = args.output_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    embed_dim = args.embed_dim
    clinical_dim = args.clinical_dim
    device = args.device if torch.cuda.is_available() else 'cpu'
    use_amp = not args.no_amp
    
    print("=" * 80)
    print("🚀 增强因果CLIP训练（可学习因果图 + 不确定性分解）")
    print("=" * 80)
    print(f"📁 数据路径: {data_path}")
    print(f"💻 使用设备: {device}")
    print(f"📦 批次大小: {batch_size}")
    print(f"🎯 训练轮数: {num_epochs}")
    print(f"📚 学习率: {learning_rate}")
    print(f"🔬 创新点: 可学习因果图 + 不确定性分解")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    train_loader, val_loader = prepare_data_loaders(data_path, batch_size, num_workers=4)
    
    # 创建特征提取器（部分微调模式）
    print("\n🔧 创建特征提取器...")
    feature_extractor = FeatureExtractor(
        embed_dim=embed_dim, 
        device=device,
        fine_tune_last_n_layers=2  # 解冻最后2层进行微调
    )
    # 统计可训练参数
    trainable_feat_params = sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad)
    total_feat_params = sum(p.numel() for p in feature_extractor.parameters())
    print(f"✅ 特征提取器创建成功（使用Swin-T backbone）")
    print(f"   特征提取器可训练参数: {trainable_feat_params / 1e6:.2f}M / {total_feat_params / 1e6:.2f}M")
    
    # 创建模型
    print("\n🔧 创建增强因果CLIP模型...")
    model = EnhancedCausalBayesianCLIP(
        embed_dim=embed_dim,
        clinical_dim=clinical_dim,
        num_classes=2,
        temperature=0.07,
        kl_weight=args.kl_weight,
        use_learnable_causal=True,  # 启用可学习因果图
        use_uncertainty_decomposition=True,  # 启用不确定性分解
        causal_loss_weight=0.01,  # 因果图正则化权重
        num_centers=0
    ).to(device)
    # 更新因果图惩罚权重
    if model.use_learnable_causal and model.learnable_causal_graph is not None:
        model.learnable_causal_graph.dag_penalty_weight = args.dag_penalty_weight
        model.learnable_causal_graph.sparsity_weight = args.sparsity_weight
        model.learnable_causal_graph.intervention_weight = args.intervention_weight
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 模型创建成功")
    print(f"   总参数量: {total_params / 1e6:.2f}M")
    print(f"   可训练参数: {trainable_params / 1e6:.2f}M")
    
    # 创建损失函数（增强正则化）
    criterion = CausalBayesianCLIPLoss(
        temperature=0.07,
        kl_weight=args.kl_weight,
        contrastive_weight=args.contrastive_weight,
        use_focal=True,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing
    )
    
    # 创建优化器（增强正则化 + 降低学习率）
    # 收集所有需要训练的参数（包括特征提取器的可训练参数）
    trainable_params = list(model.parameters())
    if any(p.requires_grad for p in feature_extractor.parameters()):
        trainable_params.extend([p for p in feature_extractor.parameters() if p.requires_grad])
    
    optimizer = optim.AdamW(
        trainable_params,
        lr=learning_rate * 0.3,  # 进一步降低学习率（从0.5到0.3）
        weight_decay=5e-4,  # 增加weight decay（从1e-4到5e-4）
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    # 混合精度训练
    scaler = GradScaler(enabled=use_amp)
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_f1': [],
        'best_auc': 0.0,
        'best_epoch': 0
    }
    
    # 开始训练
    print("\n🎓 开始训练...\n")
    best_auc = 0.0
    patience = 5  # Early Stopping patience
    patience_counter = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc, _, _, train_stats = train_one_epoch(
            model, feature_extractor, train_loader, criterion,
            optimizer, scaler, device, epoch, num_epochs,
            use_amp=use_amp,
            causal_loss_weight=args.causal_loss_weight
        )
        
        # 验证
        val_results = validate(model, feature_extractor, val_loader, criterion, device, use_amp=use_amp)
        
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
        print(f"\n{'='*80}")
        print(f"📊 Epoch [{epoch+1}/{num_epochs}] 结果")
        print(f"{'='*80}")
        print(f"训练集:")
        print(f"  Loss: {train_loss:.4f}  |  Accuracy: {train_acc:.2%}")
        if train_stats:
            comp = train_stats.get('loss_components', {})
            comp_str = ", ".join([f"{k}:{v:.4f}" for k,v in comp.items()])
            print(f"  分项损失: {comp_str}")
            print(f"  平均梯度范数: {train_stats.get('grad_norm_avg', 0.0):.4f}")
        print(f"验证集:")
        print(f"  Loss: {val_results['loss']:.4f}")
        print(f"  Accuracy: {val_results['accuracy']:.2%}")
        print(f"  AUC: {val_results['auc']:.4f}")
        print(f"  F1-Score: {val_results['f1']:.4f}")
        print(f"  最优阈值: {val_results['optimal_threshold']:.4f}")
        print(f"  最优准确率: {val_results['optimal_accuracy']:.2%}")
        print(f"  最优敏感性: {val_results['optimal_sensitivity']:.4f}")
        print(f"  最优特异性: {val_results['optimal_specificity']:.4f}")
        
        # Early Stopping检查（基于验证损失）
        val_loss = val_results['loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 保存最佳模型（基于AUC）
        if val_results['auc'] > best_auc:
            best_auc = val_results['auc']
            history['best_auc'] = best_auc
            history['best_epoch'] = epoch + 1
            
            # 保存模型
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'feature_extractor_state_dict': feature_extractor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_auc': best_auc,
                'val_results': val_results,
                'history': history
            }
            torch.save(checkpoint, os.path.join(output_dir, 'best_model.pth'))
            print(f"\n💾 保存最佳模型 (Epoch {epoch+1}, AUC: {best_auc:.4f})")
        
        # Early Stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early Stopping触发 (patience={patience}, 验证损失未改善)")
            print(f"   最佳AUC: {best_auc:.4f} (Epoch {history['best_epoch']})")
            break
        
        # 保存训练历史
        with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        # 保存指标（修复JSON序列化问题）
        def convert_to_serializable(obj):
            """将numpy/torch类型转换为Python原生类型"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            elif isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            return obj
        
        metrics = {
            'epoch': epoch + 1,
            'train': {
                'loss': float(train_loss),
                'accuracy': float(train_acc)
            },
            'val': convert_to_serializable(val_results),
            'best_auc': float(best_auc),
            'best_epoch': int(history['best_epoch'])
        }
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ 训练完成！")
    print(f"   最佳AUC: {best_auc:.4f} (Epoch {history['best_epoch']})")
    print(f"   结果保存在: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

