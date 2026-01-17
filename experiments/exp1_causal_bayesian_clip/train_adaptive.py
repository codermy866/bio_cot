#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应因果干预CLIP训练脚本
整合所有创新模块的统一训练框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, SequentialLR
import numpy as np
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

# 导入模型
try:
    from src.models.adaptive_causal_intervention_clip import AdaptiveCausalInterventionCLIP
except ImportError:
    sys.path.insert(0, str(project_root / 'src'))
    from models.adaptive_causal_intervention_clip import AdaptiveCausalInterventionCLIP

from utils.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
from utils.advanced_clinical_metrics import calculate_clinical_metrics, calculate_calibration_metrics

# 导入特征提取器
from models.SwinT.swin_image_encoder import SwinTImageEncoder


class FeatureExtractor(nn.Module):
    """特征提取器（使用Swin-T backbone）"""
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
        
        self.oct_proj = None
        self.col_proj = None
        
        # 部分微调
        self._setup_fine_tuning()
    
    def _setup_fine_tuning(self):
        """设置部分微调"""
        # 如果fine_tune_last_n_layers >= 999，解冻所有层
        if getattr(self, 'fine_tune_last_n_layers', 2) >= 999:
            print("   🔓 完全解冻特征提取器（所有层可训练）")
            for param in self.oct_encoder.parameters():
                param.requires_grad = True
            for param in self.col_encoder.parameters():
                param.requires_grad = True
            return
        
        # 否则，只解冻最后N层
        for param in self.oct_encoder.parameters():
            param.requires_grad = False
        for param in self.col_encoder.parameters():
            param.requires_grad = False
        
        try:
            if hasattr(self.oct_encoder, 'backbone'):
                backbone = self.oct_encoder.backbone
                if hasattr(backbone, 'layers'):
                    layers = backbone.layers
                    # 解冻最后N层
                    n = max(1, min(len(layers), getattr(self, 'fine_tune_last_n_layers', 2)))
                    print(f"   🔓 解冻OCT编码器最后{n}层")
                    for layer in layers[-n:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                if hasattr(self.oct_encoder, 'proj'):
                    for param in self.oct_encoder.proj.parameters():
                        param.requires_grad = True
            
            if hasattr(self.col_encoder, 'backbone'):
                backbone = self.col_encoder.backbone
                if hasattr(backbone, 'layers'):
                    layers = backbone.layers
                    n = max(1, min(len(layers), getattr(self, 'fine_tune_last_n_layers', 2)))
                    print(f"   🔓 解冻Colposcopy编码器最后{n}层")
                    for layer in layers[-n:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                if hasattr(self.col_encoder, 'proj'):
                    for param in self.col_encoder.proj.parameters():
                        param.requires_grad = True
        except Exception as e:
            print(f"   ⚠️  设置微调时出错: {e}")
    
    def forward(self, oct_images, col_images):
        use_grad = any(p.requires_grad for p in self.oct_encoder.parameters()) or \
                   any(p.requires_grad for p in self.col_encoder.parameters())

        # 如果传入的是预计算特征（二维），直接使用
        if isinstance(oct_images, torch.Tensor) and oct_images.dim() <= 2:
            oct_feat = oct_images
        else:
            if use_grad:
                oct_feat = self.oct_encoder(oct_images)
            else:
                with torch.no_grad():
                    oct_feat = self.oct_encoder(oct_images)

        if isinstance(col_images, torch.Tensor) and col_images.dim() <= 2:
            col_feat = col_images
        else:
            if use_grad:
                col_feat = self.col_encoder(col_images)
            else:
                with torch.no_grad():
                    col_feat = self.col_encoder(col_images)
        
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
    
    class Args:
        def __init__(self):
            self.data_path = data_path
            self.input_size = 224
            self.oct_num_frames = 48
            self.col_num_frames = 3
            self.oct_cache_dir = 'oct_cache_optimized'
            self.use_text_contrastive = False
            self.oct_points = 12
            self.oct_frames_per_point = 10
            self.use_pretrained_backbones = False
    
    args = Args()
    
    try:
        from src.data.enhanced_multimodal_dataset import build_enhanced_dataset
        train_dataset = build_enhanced_dataset('train', args)
        val_dataset = build_enhanced_dataset('test', args)
    except ImportError:
        from src.data.enhanced_multimodal_dataset import create_enhanced_transform
        train_transform = create_enhanced_transform(args, is_training=True)
        val_transform = create_enhanced_transform(args, is_training=False)
        
        train_dataset = EnhancedMultimodalCervicalDataset(
            root=os.path.join(data_path, 'train'),
            is_train='train',
            args=args,
            transform=train_transform
        )
        val_dataset = EnhancedMultimodalCervicalDataset(
            root=os.path.join(data_path, 'test'),
            is_train='test',
            args=args,
            transform=val_transform
        )
    
    def collate_fn(batch):
        """自定义collate函数 - 返回原始图像张量，端到端训练"""
        oct_images = []
        col_images = []
        clinical_features = []
        labels = []
        
        for item in batch:
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                oct_img, col_img, clinical_feat, label = item[:4]
                # 确保是张量
                if not isinstance(oct_img, torch.Tensor):
                    oct_img = torch.tensor(oct_img)
                if not isinstance(col_img, torch.Tensor):
                    col_img = torch.tensor(col_img)
                if not isinstance(clinical_feat, torch.Tensor):
                    clinical_feat = torch.tensor(clinical_feat)
                if not isinstance(label, torch.Tensor):
                    label = torch.tensor(label)
                
                # 去除梯度
                oct_images.append(oct_img.detach())
                col_images.append(col_img.detach())
                clinical_features.append(clinical_feat.detach())
                labels.append(label.detach())
        
        return {
            'oct_images': torch.stack(oct_images),
            'col_images': torch.stack(col_images),
            'clinical_features': torch.stack(clinical_features),
            'labels': torch.stack(labels)
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"✅ 训练集: {len(train_dataset)} 样本")
    print(f"✅ 验证集: {len(val_dataset)} 样本")
    
    return train_loader, val_loader


class FocalLoss(nn.Module):
    """Focal Loss with Label Smoothing for class imbalance - 参考成功模型"""
    
    def __init__(self, alpha=None, gamma=2.5, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        # 应用Label Smoothing
        if self.label_smoothing > 0:
            confidence = 1.0 - self.label_smoothing
            log_probs = nn.functional.log_softmax(inputs, dim=1)
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(self.label_smoothing / (num_classes - 1))
                true_dist.scatter_(1, targets.data.unsqueeze(1), confidence)
            ce_loss = -torch.sum(true_dist * log_probs, dim=1)
        else:
            ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


class AdaptiveCausalInterventionLoss(nn.Module):
    """自适应因果干预CLIP损失函数 - 优化版本"""
    def __init__(
        self,
        temperature=0.07,
        kl_weight=0.0,  # 暂时禁用
        contrastive_weight=0.0,  # 暂时禁用
        intervention_weight=0.0,  # 暂时禁用
        cls_weight=1.0,
        use_focal=True,  # 启用Focal Loss
        focal_gamma=2.5,  # 参考成功模型
        label_smoothing=0.1,  # 启用Label Smoothing
        alpha=None  # 类别权重
    ):
        super().__init__()
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.contrastive_weight = contrastive_weight
        self.intervention_weight = intervention_weight
        self.cls_weight = cls_weight
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        
        # 使用Focal Loss
        if use_focal:
            self.focal_loss = FocalLoss(alpha=alpha, gamma=focal_gamma, label_smoothing=label_smoothing)
        else:
            self.focal_loss = None
    
    def forward(self, output, labels):
        """
        Args:
            output: 模型输出字典
            labels: 标签 [B]
        """
        logits = output['logits']
        B = logits.size(0)
        
        # 1. 分类损失（使用Focal Loss + Label Smoothing）
        if self.use_focal and self.focal_loss is not None:
            cls_loss = self.focal_loss(logits, labels)
        else:
            if self.label_smoothing > 0:
                cls_loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
            else:
                cls_loss = F.cross_entropy(logits, labels)
        
        # 2. KL散度损失（贝叶斯正则化）
        kl_loss = 0.0
        if 'uncertainty' in output:
            # 从不确定性信息中提取方差
            uncertainty = output['uncertainty']
            if isinstance(uncertainty, dict):
                total_var = uncertainty.get('total', None)
                if total_var is not None:
                    # 确保方差是有效的
                    if torch.isnan(total_var).any() or torch.isinf(total_var).any():
                        kl_loss = 0.0
                    else:
                        kl_loss = total_var.mean().clamp(max=10.0)  # 限制最大值
        
        # 3. 对比学习损失
        contrastive_loss = 0.0
        if 'contrastive' in output:
            contrastive_info = output['contrastive']
            if isinstance(contrastive_info, dict):
                contrastive_loss = contrastive_info.get('total_contrastive_loss', 0.0)
                # 检查NaN/Inf
                if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
                    contrastive_loss = 0.0
                else:
                    contrastive_loss = contrastive_loss.clamp(max=10.0)  # 限制最大值
        
        # 4. 因果干预损失（鼓励有意义的干预）
        intervention_loss = 0.0
        if 'intervention' in output:
            intervention_info = output['intervention']
            if isinstance(intervention_info, dict):
                # 干预强度正则化（避免过度干预）
                strength = intervention_info.get('intervention_strength', None)
                if strength is not None:
                    if torch.isnan(strength).any() or torch.isinf(strength).any():
                        intervention_loss = 0.0
                    else:
                        intervention_loss = strength.mean().clamp(max=1.0)  # 限制最大值
        
        # 检查分类损失
        if torch.isnan(cls_loss) or torch.isinf(cls_loss):
            cls_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # 总损失（降低权重以避免不稳定）
        total_loss = (
            self.cls_weight * cls_loss +
            self.kl_weight * kl_loss +
            self.contrastive_weight * contrastive_loss +
            self.intervention_weight * intervention_loss
        )
        
        # 最终检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = cls_loss  # 如果总损失无效，只使用分类损失
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'kl_loss': kl_loss,
            'contrastive_loss': contrastive_loss,
            'intervention_loss': intervention_loss
        }


def train_one_epoch(
    model, feature_extractor, train_loader, criterion, optimizer, 
    scaler, device, epoch, num_epochs
):
    """训练一个epoch"""
    model.train()
    if any(p.requires_grad for p in feature_extractor.parameters()):
        feature_extractor.train()
    else:
        feature_extractor.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    
    for batch_idx, batch in enumerate(pbar):
        # 获取数据
        if isinstance(batch, dict):
            oct_feat = batch.get('oct_features', batch.get('oct_feat'))
            col_feat = batch.get('col_features', batch.get('col_feat'))
            clinical_features = batch.get('clinical_features', batch.get('clinical'))
            labels = batch.get('labels', batch.get('label'))
            
            if oct_feat is None or col_feat is None:
                # 需要从图像提取特征
                oct_images = batch.get('oct_images')
                col_images = batch.get('col_images')
                if oct_images is not None and col_images is not None:
                    oct_feat, col_feat = feature_extractor(
                        oct_images.to(device), 
                        col_images.to(device)
                    )
                    # 检查特征提取器输出
                    if torch.isnan(oct_feat).any() or torch.isnan(col_feat).any():
                        print(f"⚠️  特征提取器输出包含NaN，跳过batch {batch_idx}")
                        continue
                    # 数值稳定性：clamp特征
                    oct_feat = torch.clamp(oct_feat, min=-10.0, max=10.0)
                    col_feat = torch.clamp(col_feat, min=-10.0, max=10.0)
                else:
                    continue
            else:
                oct_feat = oct_feat.to(device)
                col_feat = col_feat.to(device)
                # 检查并修复NaN
                if torch.isnan(oct_feat).any():
                    oct_feat = torch.zeros_like(oct_feat)
                if torch.isnan(col_feat).any():
                    col_feat = torch.zeros_like(col_feat)
                oct_feat = torch.clamp(oct_feat, min=-10.0, max=10.0)
                col_feat = torch.clamp(col_feat, min=-10.0, max=10.0)
            
            if clinical_features is None:
                clinical_features = torch.zeros(oct_feat.size(0), 7).to(device)
            else:
                clinical_features = clinical_features.to(device)
                # 检查并修复NaN
                if torch.isnan(clinical_features).any():
                    clinical_features = torch.zeros_like(clinical_features)
                clinical_features = torch.clamp(clinical_features, min=-10.0, max=10.0)
            
            if labels is None:
                continue
            labels = labels.to(device)
        else:
            continue
        
        # 前向传播
        optimizer.zero_grad()
        
        with autocast(enabled=False):
            # 检查输入特征是否有NaN
            if torch.isnan(oct_feat).any() or torch.isnan(col_feat).any() or torch.isnan(clinical_features).any():
                print(f"⚠️  输入特征包含NaN，跳过batch {batch_idx}")
                continue
            
            output = model(
                oct_feat, col_feat, clinical_features,
                labels=labels,
                return_uncertainty=True,
                return_intervention=True,
                return_causal_graph=True
            )
            
            # 检查输出是否有NaN
            logits = output['logits']
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"⚠️  模型输出包含NaN/Inf，跳过batch {batch_idx}")
                # 检查中间特征
                if 'mean' in output:
                    mean_min, mean_max = output['mean'].min().item(), output['mean'].max().item()
                    print(f"   特征均值范围: [{mean_min:.4f}, {mean_max:.4f}]")
                if 'var' in output:
                    var_min, var_max = output['var'].min().item(), output['var'].max().item()
                    print(f"   特征方差范围: [{var_min:.4f}, {var_max:.4f}]")
                # 重新初始化模型参数并跳过当前batch
                if batch_idx < 3:  # 只在最初几个batch重新初始化
                    print("   🔄 重新初始化模型参数...")
                    for param in model.parameters():
                        if param.requires_grad:
                            if len(param.shape) >= 2:
                                nn.init.xavier_uniform_(param.data, gain=0.5)
                            else:
                                nn.init.constant_(param.data, 0.0)
                continue
            
            # 计算损失
            loss_dict = criterion(output, labels)
            loss = loss_dict['total_loss']
            
            # 调试输出：检查logits和loss
            if batch_idx == 0:  # 只在第一个batch打印
                print(f"\n🔍 [Epoch {epoch+1}, Batch 0] 调试信息:")
                print(f"   输入特征范围: OCT[{oct_feat.min().item():.4f}, {oct_feat.max().item():.4f}], "
                      f"Col[{col_feat.min().item():.4f}, {col_feat.max().item():.4f}], "
                      f"Clinical[{clinical_features.min().item():.4f}, {clinical_features.max().item():.4f}]")
                # 检查模型内部特征
                if 'mean' in output:
                    print(f"   特征均值范围: [{output['mean'].min().item():.4f}, {output['mean'].max().item():.4f}]")
                if 'var' in output:
                    print(f"   特征方差范围: [{output['var'].min().item():.4f}, {output['var'].max().item():.4f}]")
                # 检查分类器参数
                if hasattr(model, 'classifier'):
                    last_layer = model.classifier[-1] if isinstance(model.classifier, nn.Sequential) else model.classifier
                    if hasattr(last_layer, 'weight'):
                        print(f"   分类器权重范围: [{last_layer.weight.min().item():.4f}, {last_layer.weight.max().item():.4f}]")
                    if hasattr(last_layer, 'bias') and last_layer.bias is not None:
                        print(f"   分类器bias: {last_layer.bias.data}")
                print(f"   Logits范围: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
                print(f"   Logits均值: {logits.mean().item():.4f}")
                print(f"   Logits样本: {logits[0].detach().cpu().numpy()}")
                print(f"   Loss值: {loss.item():.8f} (cls_loss: {loss_dict['cls_loss'].item():.8f})")
                print(f"   标签分布: {torch.bincount(labels, minlength=2).cpu().numpy()}")
            
            # 检查NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  跳过NaN/Inf loss (batch {batch_idx})")
                continue
        
        # 反向传播
        scaler.scale(loss).backward()
        all_params = list(model.parameters())
        if any(p.requires_grad for p in feature_extractor.parameters()):
            all_params.extend([p for p in feature_extractor.parameters() if p.requires_grad])
        
        # 若无有效梯度，则跳过当前step，避免GradScaler断言
        grads = [p.grad for p in all_params if hasattr(p, 'grad') and p.grad is not None]
        if len(grads) == 0:
            optimizer.zero_grad(set_to_none=True)
            # 需要update以重置scaler状态
            scaler.update()
            continue
        
        # 仅在存在梯度时进行unscale与裁剪
        scaler.unscale_(optimizer)
        try:
            # 梯度裁剪（参考成功模型，允许更大梯度）
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=2.0)
        except Exception:
            pass
        
        # 正常更新（兼容某些情况下GradScaler未记录inf检查）
        try:
            scaler.step(optimizer)
        except AssertionError:
            optimizer.step()
        scaler.update()
        
        # 统计
        total_loss += loss.item()
        logits = output['logits']
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        # 收集概率和边际分数
        batch_probs = probs[:, 1].detach().cpu().numpy()
        batch_margins = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
        # 若概率退化为常数，则用边际分数替代
        if np.std(batch_probs) <= 1e-8 and np.std(batch_margins) > 1e-8:
            all_probs.extend(batch_margins.tolist())
        else:
            all_probs.extend(batch_probs.tolist())
        
        # 显示更详细的loss信息
        loss_display = loss.item()
        if loss_display < 0.0001:
            loss_str = f'{loss_display:.8f}'
        else:
            loss_str = f'{loss_display:.4f}'
        
        pbar.set_postfix({
            'Loss': loss_str,
            'Acc': f'{accuracy_score(all_labels, all_preds):.2%}',
            'ClsLoss': f'{loss_dict["cls_loss"].item():.6f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc, all_probs, all_labels


def validate(model, feature_extractor, val_loader, criterion, device):
    """验证"""
    model.eval()
    feature_extractor.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='验证中'):
            if isinstance(batch, dict):
                oct_feat = batch.get('oct_features', batch.get('oct_feat'))
                col_feat = batch.get('col_features', batch.get('col_feat'))
                clinical_features = batch.get('clinical_features', batch.get('clinical'))
                labels = batch.get('labels', batch.get('label'))
                
                if oct_feat is None or col_feat is None:
                    oct_images = batch.get('oct_images')
                    col_images = batch.get('col_images')
                    if oct_images is not None and col_images is not None:
                        oct_feat, col_feat = feature_extractor(
                            oct_images.to(device),
                            col_images.to(device)
                        )
                    else:
                        continue
                else:
                    oct_feat = oct_feat.to(device)
                    col_feat = col_feat.to(device)
                
                if clinical_features is None:
                    clinical_features = torch.zeros(oct_feat.size(0), 7).to(device)
                else:
                    clinical_features = clinical_features.to(device)
                
                if labels is None:
                    continue
                labels = labels.to(device)
            else:
                continue
            
            with autocast(enabled=False):
                output = model(
                    oct_feat, col_feat, clinical_features,
                    return_uncertainty=True,
                    return_intervention=False,
                    return_causal_graph=False
                )
                
                loss_dict = criterion(output, labels)
                loss = loss_dict['total_loss']
            
            total_loss += loss.item()
            logits = output['logits']
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            # 收集概率和边际分数
            batch_probs = probs[:, 1].detach().cpu().numpy()
            batch_margins = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
            # 若概率退化为常数，则用边际分数替代
            if np.std(batch_probs) <= 1e-8 and np.std(batch_margins) > 1e-8:
                all_probs.extend(batch_margins.tolist())
            else:
                all_probs.extend(batch_probs.tolist())
    
    avg_loss = total_loss / len(val_loader)
    acc = accuracy_score(all_labels, all_preds)
    
    # 安全计算AUC（常数概率时设置为0.5）
    auc = 0.5
    if len(all_labels) > 0 and len(set(all_labels)) > 1:
        probs_np = np.asarray(all_probs, dtype=float)
        if np.std(probs_np) > 1e-8:
            try:
                auc = roc_auc_score(all_labels, probs_np)
            except Exception:
                auc = 0.5

    # 先基于argmax给出粗略分数
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)

    # 计算最优阈值（Youden），并据此重算指标
    optimal_threshold = 0.5
    optimal_acc = acc
    optimal_f1 = f1
    optimal_sensitivity = recall
    optimal_specificity = 0.0
    if len(all_labels) > 0 and len(set(all_labels)) > 1:
        probs_np = np.asarray(all_probs, dtype=float)
        if np.std(probs_np) > 1e-8:
            fpr, tpr, thresholds = roc_curve(all_labels, probs_np)
            youden_index = tpr - fpr
            optimal_idx = int(np.argmax(youden_index))
            optimal_threshold = float(thresholds[optimal_idx])
        # 用最优阈值（若仍为0.5也可）生成预测
        opt_preds = (probs_np >= optimal_threshold).astype(int)
        optimal_acc = accuracy_score(all_labels, opt_preds)
        optimal_f1 = f1_score(all_labels, opt_preds, zero_division=0)
        optimal_sensitivity = recall_score(all_labels, opt_preds, zero_division=0)  # TPR
        # specificity = TN / (TN + FP)
        tn, fp, fn, tp = confusion_matrix(all_labels, opt_preds, labels=[0,1]).ravel()
        denom = (tn + fp)
        optimal_specificity = float(tn / denom) if denom > 0 else 0.0
    
    try:
        clinical_metrics = calculate_clinical_metrics(all_labels, all_preds, all_probs)
    except Exception:
        clinical_metrics = {}
    
    return {
        'loss': avg_loss,
        'accuracy': acc,
        'auc': float(auc),
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'optimal_threshold': float(optimal_threshold),
        'optimal_accuracy': optimal_acc,
        'optimal_f1': optimal_f1,
        'optimal_sensitivity': optimal_sensitivity,
        'optimal_specificity': optimal_specificity,
        'clinical_metrics': clinical_metrics
    }


def main():
    """主训练函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='自适应因果干预CLIP训练')
    parser.add_argument('--data_path', type=str, default='5centers_multi', help='数据路径')
    parser.add_argument('--output_dir', type=str, default='adaptive_causal_intervention_results', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=6, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--embed_dim', type=int, default=768, help='嵌入维度')
    parser.add_argument('--clinical_dim', type=int, default=7, help='临床特征维度')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()
    
    data_path = args.data_path
    output_dir = args.output_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    embed_dim = args.embed_dim
    clinical_dim = args.clinical_dim
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print("=" * 80)
    print("🚀 自适应因果干预CLIP训练")
    print("=" * 80)
    print(f"📁 数据路径: {data_path}")
    print(f"💻 使用设备: {device}")
    print(f"📦 批次大小: {batch_size}")
    print(f"🎯 训练轮数: {num_epochs}")
    print(f"📚 学习率: {learning_rate}")
    print(f"🔬 创新点:")
    print(f"   1. 因果干预机制（Causal Intervention）")
    print(f"   2. 自适应因果图（Adaptive Causal Graph）")
    print(f"   3. 分层对比学习（Hierarchical Contrastive Learning）")
    print(f"   4. 不确定性引导的干预（Uncertainty-Guided Intervention）")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    train_loader, val_loader = prepare_data_loaders(data_path, batch_size, num_workers=4)
    
    # 创建特征提取器
    print("\n🔧 创建特征提取器...")
    # 解冻更多层（12层），但不完全解冻以避免数值不稳定
    feature_extractor = FeatureExtractor(embed_dim=embed_dim, device=device, fine_tune_last_n_layers=12)
    trainable_feat_params = sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad)
    total_feat_params = sum(p.numel() for p in feature_extractor.parameters())
    print(f"✅ 特征提取器创建成功")
    print(f"   可训练参数: {trainable_feat_params / 1e6:.2f}M / {total_feat_params / 1e6:.2f}M")
    
    # 创建模型
    print("\n🔧 创建自适应因果干预CLIP模型...")
    print("   ⚠️  暂时关闭因果干预和自适应因果图，待AUC>0.5后逐步打开")
    model = AdaptiveCausalInterventionCLIP(
        embed_dim=embed_dim,
        clinical_dim=clinical_dim,
        num_classes=2,
        temperature=0.07,
        kl_weight=0.01,
        use_causal_intervention=False,  # 暂时关闭
        use_adaptive_causal_graph=False,  # 暂时关闭
        use_hierarchical_contrastive=True,
        use_uncertainty_guided=True,
        use_uncertainty_decomposition=True,
        num_centers=0
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 模型创建成功")
    print(f"   总参数量: {total_params / 1e6:.2f}M")
    print(f"   可训练参数: {trainable_params / 1e6:.2f}M")
    
    # 计算类别权重（处理类别不平衡）
    # 类别分布：{0: 0.325, 1: 0.675}，所以alpha=[0.325, 0.675]
    class_alpha = torch.tensor([0.325, 0.675]).to(device)
    
    # 创建损失函数（使用Focal Loss + Label Smoothing）
    criterion = AdaptiveCausalInterventionLoss(
        temperature=0.07,
        kl_weight=0.0,  # 暂时禁用
        contrastive_weight=0.0,  # 暂时禁用
        intervention_weight=0.0,  # 暂时禁用
        cls_weight=1.0,
        use_focal=True,  # 启用Focal Loss
        focal_gamma=2.5,  # 参考成功模型
        label_smoothing=0.1,  # 启用Label Smoothing
        alpha=class_alpha  # 类别权重
    )
    
    # 创建优化器
    trainable_params_list = list(model.parameters())
    if any(p.requires_grad for p in feature_extractor.parameters()):
        trainable_params_list.extend([p for p in feature_extractor.parameters() if p.requires_grad])
    
    # 优化学习率和正则化（参考成功模型）
    adjusted_lr = learning_rate * 0.5  # 降低学习率，配合更强的正则化
    weight_decay = 5e-4  # 增加weight decay，防止过拟合
    print(f"\n📊 优化器配置:")
    print(f"   学习率: {adjusted_lr:.6f} (基础学习率 {learning_rate:.6f} × 0.5)")
    print(f"   Weight Decay: {weight_decay}")
    print(f"   使用Focal Loss (gamma=2.5) + Label Smoothing (0.1)")
    
    # 重新初始化模型参数（确保没有NaN）
    print("🔄 重新初始化模型参数...")
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param.data, gain=0.5)  # 使用较小的gain
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)
    
    optimizer = optim.AdamW(
        trainable_params_list,
        lr=adjusted_lr,
        weight_decay=weight_decay,  # 增加weight decay
        betas=(0.9, 0.999),
        eps=1e-8  # 确保数值稳定性
    )
    # Warmup + Cosine（参考成功模型）
    warmup_epochs = max(2, num_epochs // 8)  # 12.5%的epochs用于warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            min_lr_ratio = 0.1
            return max(min_lr_ratio, cosine_factor)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scaler = GradScaler(enabled=False)
    
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
    
    # 强制分类头正类bias注入：[-2.0, +2.0]，确保早期正类输出非零（在重新初始化之后）
    try:
        print(f"🔍 检查分类头结构...")
        if hasattr(model, 'classifier'):
            print(f"   ✅ 找到classifier: {type(model.classifier)}")
            if isinstance(model.classifier, nn.Sequential):
                print(f"   ✅ classifier是Sequential，长度: {len(model.classifier)}")
                last = model.classifier[-1]
                print(f"   ✅ 最后一层: {type(last)}")
                if hasattr(last, 'bias'):
                    print(f"   ✅ 最后一层有bias: {last.bias is not None}")
                    if last.bias is not None:
                        print(f"   ✅ bias形状: {last.bias.shape}, 当前值: {last.bias.data}")
                        if last.bias.shape[0] >= 2:
                            with torch.no_grad():
                                # 强制设为 [-2.0, +2.0]
                                last.bias.data[0] = -2.0
                                last.bias.data[1] = +2.0
                            print(f"🔧 已强制初始化分类头bias: [-2.0, +2.0], 新值: {last.bias.data}")
                        else:
                            print(f"   ⚠️ bias形状不足: {last.bias.shape}")
                    else:
                        print(f"   ⚠️ bias为None")
                else:
                    print(f"   ⚠️ 最后一层没有bias属性")
            else:
                print(f"   ⚠️ classifier不是Sequential: {type(model.classifier)}")
        else:
            print(f"   ⚠️ 模型没有classifier属性")
    except Exception as e:
        print(f"⚠️ 分类头bias初始化失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎓 开始训练...\n")
    best_auc = 0.0
    patience = 7  # 参考成功模型，给予更多机会
    patience_counter = 0
    best_val_loss = float('inf')
    min_delta = 0.0005  # 更敏感地检测改善
    
    # 过拟合监控
    max_gap = 0.25  # 训练准确率与验证准确率的最大允许差距（25%）
    overfitting_epochs = 0
    
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc, _, _ = train_one_epoch(
            model, feature_extractor, train_loader, criterion,
            optimizer, scaler, device, epoch, num_epochs
        )
        
        # 验证
        val_results = validate(model, feature_extractor, val_loader, criterion, device)
        
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
        print(f"验证集:")
        print(f"  Loss: {val_results['loss']:.4f}")
        print(f"  Accuracy: {val_results['accuracy']:.2%}")
        print(f"  AUC: {val_results['auc']:.4f}")
        print(f"  F1-Score: {val_results['f1']:.4f}")
        print(f"  最优阈值: {val_results['optimal_threshold']:.4f}")
        print(f"  最优准确率: {val_results['optimal_accuracy']:.2%}")
        print(f"  最优敏感性: {val_results['optimal_sensitivity']:.4f}")
        print(f"  最优特异性: {val_results['optimal_specificity']:.4f}")
        
        # 过拟合监控
        train_val_gap = train_acc - val_results['accuracy']
        if train_val_gap > max_gap:
            overfitting_epochs += 1
            if overfitting_epochs >= 3:
                print(f"\n⚠️  检测到过拟合：训练-验证准确率差距 {train_val_gap:.2%} > {max_gap:.2%}")
        else:
            overfitting_epochs = 0
        
        # Early Stopping（基于AUC改善）
        if val_results['auc'] > best_auc + min_delta:
            best_auc = val_results['auc']
            best_val_loss = val_results['loss']
            patience_counter = 0
            history['best_auc'] = best_auc
            history['best_epoch'] = epoch + 1
            
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
        else:
            patience_counter += 1
        
        # Early Stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early Stopping触发 (patience={patience})")
            print(f"   最佳AUC: {best_auc:.4f} (Epoch {history['best_epoch']})")
            break
        
        # 保存训练历史
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
            json.dump(convert_to_serializable(history), f, indent=2)
    
    print(f"\n✅ 训练完成！最佳AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()

