#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BIDA优化训练脚本
- 使用预计算的VLM特征（避免训练时重复计算）
- 优化的损失权重
- 学习率调度
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast, GradScaler

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
from src.models.bida.bida_model import BIDAModel
from src.models.bida.orthogonal_loss import DistributionMatchingLoss, OrthogonalLoss, NoiseSupervisionLoss

class OptimizedArgs:
    """优化的训练参数"""
    def __init__(self):
        # 基础参数
        self.data_root = '/data2/hmy/5Center_datas/5centers_multi_leave_centers_out'
        self.device = 'cuda:1'
        self.num_epochs = 50
        self.batch_size = 32  # 减小batch size，因为VLM特征已预计算
        self.num_workers = 8
        self.learning_rate = 2e-4
        self.weight_decay = 5e-4  # 增加权重衰减
        
        # 优化的损失权重（根据分析调整）
        self.lambda_kl = 0.1      # 从0.005增加到0.1（分布锚定更重要）
        self.lambda_orth = 0.5    # 从0.01增加到0.5（正交约束更重要）
        self.lambda_adv = 0.3     # 从0.05增加到0.3（噪声监督更重要）
        
        # 学习率调度
        self.warmup_epochs = 5
        self.warmup_lr = 1e-5
        self.min_lr = 1e-6
        
        # VLM特征缓存
        self.use_vlm_cache = True  # 使用预计算的VLM特征
        self.vlm_cache_dir = Path(self.data_root) / 'vlm_features_cache'
        
        # 输出目录
        self.output_dir = Path(__file__).parent / 'exp_bida_optimized'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)


class CachedVLMDistributionalAnchor(nn.Module):
    """使用缓存的VLM特征的DistributionalAnchor"""
    def __init__(self, embed_dim=768, vlm_cache_dir=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.vlm_cache_dir = Path(vlm_cache_dir) if vlm_cache_dir else None
        
        # Distribution head
        self.distribution_head = nn.Sequential(
            nn.Linear(1536, embed_dim * 2),  # VLM特征维度是1536
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim * 2)  # 输出μ和σ
        )
        
        # Fallback MLP（如果VLM特征不可用）
        self.fallback_encoder = nn.Sequential(
            nn.Linear(3, 128),  # age, hpv, tct
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim * 2)
        )
    
    def load_vlm_feature(self, patient_id):
        """加载预计算的VLM特征"""
        if self.vlm_cache_dir is None:
            return None
        
        feature_file = self.vlm_cache_dir / f"{patient_id}_vlm_feat.npy"
        if feature_file.exists():
            return np.load(feature_file)
        return None
    
    def forward(self, clinical_features, patient_ids=None):
        """
        使用缓存的VLM特征
        
        Args:
            clinical_features: [B, 3] 临床特征（fallback用）
            patient_ids: List[str] 患者ID列表
        """
        B = clinical_features.size(0)
        
        # 尝试加载VLM特征
        vlm_features = []
        for i, pid in enumerate(patient_ids if patient_ids else [None] * B):
            if pid:
                feat = self.load_vlm_feature(pid)
                if feat is not None:
                    vlm_features.append(torch.from_numpy(feat).float())
                else:
                    # 使用fallback
                    vlm_features.append(None)
            else:
                vlm_features.append(None)
        
        # 如果有VLM特征，使用它们；否则使用fallback
        if any(f is not None for f in vlm_features):
            # 混合使用VLM特征和fallback
            text_features_list = []
            for i, feat in enumerate(vlm_features):
                if feat is not None:
                    text_features_list.append(feat.to(clinical_features.device))
                else:
                    # 使用fallback
                    fallback_feat = self.fallback_encoder(clinical_features[i:i+1])
                    # 投影到1536维（模拟VLM特征）
                    if not hasattr(self, 'fallback_proj'):
                        self.fallback_proj = nn.Linear(embed_dim * 2, 1536).to(clinical_features.device)
                    proj_feat = self.fallback_proj(fallback_feat)
                    text_features_list.append(proj_feat.squeeze(0))
            
            text_features = torch.stack(text_features_list, dim=0)  # [B, 1536]
        else:
            # 全部使用fallback
            fallback_feat = self.fallback_encoder(clinical_features)
            if not hasattr(self, 'fallback_proj'):
                self.fallback_proj = nn.Linear(self.embed_dim * 2, 1536).to(clinical_features.device)
            text_features = self.fallback_proj(fallback_feat)  # [B, 1536]
        
        # 生成分布参数
        dist_params = self.distribution_head(text_features)  # [B, embed_dim * 2]
        μ_bio = dist_params[:, :self.embed_dim]  # [B, embed_dim]
        σ_bio = torch.nn.functional.softplus(dist_params[:, self.embed_dim:]) + 1e-6  # [B, embed_dim]
        
        return μ_bio, σ_bio


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
        # 解析batch
        if isinstance(batch, dict):
            oct_feat = batch['oct_features'].to(device)
            colpo_feat = batch['colposcopy_features'].to(device)
            clinical_feat = batch['clinical_features'].to(device)
            labels = batch['label'].to(device)
            center_labels = batch['center_id'].to(device)
            patient_ids = batch.get('patient_id', [None] * len(labels))
        else:
            continue
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(
                oct_features=oct_feat,
                colpo_features=colpo_feat,
                clinical_features=clinical_feat,
                center_labels=center_labels,
                return_loss_components=True
            )
            
            logits = outputs['logits']
            cls_loss = criterion(logits, labels)
            
            loss_components = outputs['loss_components']
            dist_loss = loss_components['L_dist']
            orth_loss = loss_components['L_orth']
            noise_loss = loss_components['L_noise']
            
            # 优化的总损失
            total_loss_batch = (
                cls_loss +
                args.lambda_kl * dist_loss +
                args.lambda_orth * orth_loss +
                args.lambda_adv * noise_loss
            )
        
        scaler.scale(total_loss_batch).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # 统计
        total_loss += total_loss_batch.item()
        total_cls_loss += cls_loss.item()
        total_dist_loss += dist_loss.item()
        total_orth_loss += orth_loss.item()
        total_noise_loss += noise_loss.item()
        
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
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
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            if isinstance(batch, dict):
                oct_feat = batch['oct_features'].to(device)
                colpo_feat = batch['colposcopy_features'].to(device)
                clinical_feat = batch['clinical_features'].to(device)
                labels = batch['label'].to(device)
            else:
                continue
            
            with autocast():
                outputs = model(
                    oct_features=oct_feat,
                    colpo_features=colpo_feat,
                    clinical_features=clinical_feat,
                    return_loss_components=False
                )
                logits = outputs['logits']
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    from sklearn.metrics import roc_auc_score
    try:
        if len(set(all_labels)) < 2:
            auc = 0.0
        else:
            auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return {
        'loss': total_loss / len(val_loader),
        'acc': 100. * correct / total,
        'auc': auc
    }


def main():
    """主函数"""
    args = OptimizedArgs()
    device = torch.device(args.device)
    
    print("📂 加载数据集...")
    train_dataset = EnhancedMultimodalCervicalDataset(
        root=Path(args.data_root) / 'internal_train' / 'train',
        labels_file=Path(args.data_root) / 'train_labels.csv',
        use_pretrained_backbones=True
    )
    val_dataset = EnhancedMultimodalCervicalDataset(
        root=Path(args.data_root) / 'internal_train' / 'val',
        labels_file=Path(args.data_root) / 'val_labels.csv',
        use_pretrained_backbones=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print("🏗️ 创建模型...")
    # 注意：这里需要修改BIDAModel以支持缓存的VLM特征
    # 暂时使用原始模型，但可以后续修改
    model = BIDAModel(
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
        num_centers=args.num_centers
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler()
    
    # 学习率调度器
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_scheduler = LinearLR(optimizer, start_factor=args.warmup_lr/args.learning_rate, 
                                end_factor=1.0, total_iters=len(train_loader) * args.warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=(args.num_epochs - args.warmup_epochs) * len(train_loader),
                                       eta_min=args.min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler],
                             milestones=[args.warmup_epochs * len(train_loader)])
    
    best_auc = 0.0
    
    print("🚀 开始训练...")
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 80)
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, args, device)
        scheduler.step()
        
        val_metrics = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, Train Acc={train_metrics['acc']:.2f}%, "
              f"Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['acc']:.2f}%, Val AUC={val_metrics['auc']:.4f}")
        
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(model.state_dict(), args.output_dir / 'best_model.pth')
            print(f"✅ New best model saved! AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()

