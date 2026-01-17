#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard CLIP Baseline - 标准CLIP方法（无因果约束，无不确定性量化）
用于对比实验，证明因果约束和不确定性量化的价值

方法: 标准跨模态对比学习 + 注意力融合
特点: 有对比学习，无因果约束，无不确定性量化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import json
from datetime import datetime
from typing import Optional

# 添加项目路径
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.enhanced_multimodal_dataset import build_enhanced_dataset


class StandardCLIPEncoder(nn.Module):
    """标准CLIP编码器（无不确定性量化）"""
    def __init__(self, embed_dim=768, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


class StandardCLIPBaseline(nn.Module):
    """
    标准CLIP Baseline
    特点: 有对比学习，无因果约束，无不确定性量化
    """
    def __init__(self, embed_dim=768, clinical_dim=7, num_classes=2, temperature=0.07):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        
        # 临床特征投影
        self.clinical_proj = nn.Linear(clinical_dim, embed_dim)
        
        # 标准编码器（无不确定性）
        self.oct_encoder = StandardCLIPEncoder(embed_dim)
        self.col_encoder = StandardCLIPEncoder(embed_dim)
        self.clinical_encoder = StandardCLIPEncoder(embed_dim)
        
        # 多头注意力融合
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=False
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 分类器
        self.classifier = nn.Linear(embed_dim * 2, num_classes)
    
    def forward(self, oct_feat, col_feat, clinical_feat):
        """
        Args:
            oct_feat: [B, 512]
            col_feat: [B, 512]
            clinical_feat: [B, 7]
        """
        B = oct_feat.size(0)
        
        # 投影临床特征
        clinical_proj = self.clinical_proj(clinical_feat)  # [B, embed_dim]
        
        # 投影到统一维度（如果需要）
        if oct_feat.size(-1) != self.embed_dim:
            oct_proj = nn.Linear(oct_feat.size(-1), self.embed_dim).to(oct_feat.device)(oct_feat)
        else:
            oct_proj = oct_feat
        if col_feat.size(-1) != self.embed_dim:
            col_proj = nn.Linear(col_feat.size(-1), self.embed_dim).to(col_feat.device)(col_feat)
        else:
            col_proj = col_feat
        
        # 标准编码（无不确定性）
        oct_encoded = self.oct_encoder(oct_proj)  # [B, embed_dim]
        col_encoded = self.col_encoder(col_proj)  # [B, embed_dim]
        clinical_encoded = self.clinical_encoder(clinical_proj)  # [B, embed_dim]
        
        # 构建序列
        multimodal_seq = torch.stack([
            oct_encoded,
            col_encoded,
            clinical_encoded
        ], dim=1)  # [B, 3, embed_dim]
        
        # 扩展到embed_dim*2
        multimodal_seq_expanded = torch.cat([
            F.pad(multimodal_seq[:, 0:1, :], (0, self.embed_dim)),  # OCT
            F.pad(multimodal_seq[:, 1:2, :], (0, self.embed_dim)),  # Colposcopy
            F.pad(multimodal_seq[:, 2:3, :], (0, self.embed_dim * 2 // 3))  # Clinical
        ], dim=-1)
        multimodal_seq_expanded = multimodal_seq_expanded[:, :, :self.embed_dim * 2]
        
        # 转置为 [seq_len, B, embed_dim*2] 用于MultiheadAttention
        multimodal_seq_t = multimodal_seq_expanded.transpose(0, 1)  # [3, B, embed_dim*2]
        
        # 多头注意力融合（无因果约束）
        attn_output, attn_weights = self.multihead_attn(
            multimodal_seq_t, multimodal_seq_t, multimodal_seq_t
        )  # [3, B, embed_dim*2]
        
        # 转回并融合
        attn_output = attn_output.transpose(0, 1)  # [B, 3, embed_dim*2]
        fused = attn_output.mean(dim=1)  # [B, embed_dim*2]
        fused = self.fusion(fused)  # [B, embed_dim]
        
        # 分类
        fused_expanded = torch.cat([fused, fused], dim=-1)[:, :self.embed_dim * 2]
        logits = self.classifier(fused_expanded)  # [B, num_classes]
        
        # 用于对比学习的嵌入（L2归一化）
        emb_oct = F.normalize(oct_encoded, p=2, dim=-1)
        emb_col = F.normalize(col_encoded, p=2, dim=-1)
        emb_clin = F.normalize(clinical_encoded, p=2, dim=-1)
        
        return {
            'logits': logits,
            'emb_oct': emb_oct,
            'emb_col': emb_col,
            'emb_clin': emb_clin
        }


class StandardCLIPLoss(nn.Module):
    """标准CLIP损失（对比学习 + 分类损失）"""
    def __init__(self, temperature=0.07, contrastive_weight=0.1):
        super().__init__()
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
    
    def _info_nce(self, zi, zj, temperature):
        """InfoNCE损失"""
        zi = F.normalize(zi, p=2, dim=-1)
        zj = F.normalize(zj, p=2, dim=-1)
        logits = torch.matmul(zi, zj.t()) / temperature
        labels = torch.arange(zi.size(0), device=zi.device)
        return F.cross_entropy(logits, labels)
    
    def forward(self, predictions, targets):
        logits = predictions['logits']
        emb_oct = predictions.get('emb_oct')
        emb_col = predictions.get('emb_col')
        emb_clin = predictions.get('emb_clin')
        
        # 分类损失
        ce_loss = F.cross_entropy(logits, targets)
        
        # 对比学习损失
        contrastive_loss = 0.0
        if emb_oct is not None and emb_col is not None and emb_clin is not None:
            # OCT-Colposcopy对比
            contrastive_loss += self._info_nce(emb_oct, emb_col, self.temperature)
            # OCT-Clinical对比
            contrastive_loss += self._info_nce(emb_oct, emb_clin, self.temperature)
            # Colposcopy-Clinical对比
            contrastive_loss += self._info_nce(emb_col, emb_clin, self.temperature)
            contrastive_loss = contrastive_loss / 3.0
        
        total_loss = ce_loss + self.contrastive_weight * contrastive_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'contrastive_loss': contrastive_loss
        }


class Args:
    def __init__(self):
        self.data_path = '/data2/hmy/5Center_datas/5centers_multi_leave_centers_out'
        self.batch_size = 24
        self.num_epochs = 100
        self.learning_rate = 3e-4
        self.weight_decay = 1e-5
        self.num_workers = 4
        self.device = 'cuda:0'
        self.input_size = 224
        self.oct_num_frames = 120
        self.col_num_frames = 3
        self.oct_cache_dir = None
        self.use_text_contrastive = False
        self.oct_points = 12
        self.oct_frames_per_point = 10
        self.use_pretrained_backbones = False
        self.cache_oct_features = False
        self.num_classes = 2


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for sample in pbar:
        if isinstance(sample, dict):
            oct_feat = sample['oct_features'].to(device)
            col_feat = sample['colposcopy_features'].to(device)
            clinical_feat = sample['clinical_features'].to(device)
            labels = sample['label'].to(device)
        else:
            oct_feat, col_feat, clinical_feat, labels = sample[:4]
            oct_feat = oct_feat.to(device)
            col_feat = col_feat.to(device)
            clinical_feat = clinical_feat.to(device)
            labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=scaler is not None):
            output = model(oct_feat, col_feat, clinical_feat)
            loss_dict = criterion(output, labels)
            loss = loss_dict['total_loss']
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(output['logits'], dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100 * correct / total


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for sample in tqdm(dataloader, desc='Validation'):
            if isinstance(sample, dict):
                oct_feat = sample['oct_features'].to(device)
                col_feat = sample['colposcopy_features'].to(device)
                clinical_feat = sample['clinical_features'].to(device)
                labels = sample['label'].to(device)
            else:
                oct_feat, col_feat, clinical_feat, labels = sample[:4]
                oct_feat = oct_feat.to(device)
                col_feat = col_feat.to(device)
                clinical_feat = clinical_feat.to(device)
                labels = labels.to(device)
            
            output = model(oct_feat, col_feat, clinical_feat)
            loss_dict = criterion(output, labels)
            loss = loss_dict['total_loss']
            
            total_loss += loss.item()
            probs = torch.softmax(output['logits'], dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # 计算指标
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    
    # 最优阈值
    best_thr = 0.5
    best_acc = 0
    for thr in np.linspace(0.1, 0.9, 17):
        y_pred = (y_prob >= thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    
    y_pred = (y_prob >= best_thr).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return {
        'loss': total_loss / len(dataloader),
        'auc': auc,
        'acc': acc,
        'f1': f1,
        'best_thr': best_thr
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Standard CLIP Baseline训练')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./baseline_standard_clip/checkpoints')
    args_cmd = parser.parse_args()
    
    args = Args()
    if args_cmd.data_path:
        args.data_path = args_cmd.data_path
    if args_cmd.batch_size:
        args.batch_size = args_cmd.batch_size
    if args_cmd.num_epochs:
        args.num_epochs = args_cmd.num_epochs
    if args_cmd.learning_rate:
        args.learning_rate = args_cmd.learning_rate
    if args_cmd.device:
        args.device = args_cmd.device
    if args_cmd.num_workers:
        args.num_workers = args_cmd.num_workers
    if args_cmd.output_dir:
        output_dir = Path(args_cmd.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = output_dir.parent / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🚀 Standard CLIP Baseline训练")
    print("=" * 80)
    print(f"📁 数据路径: {args.data_path}")
    print(f"💻 使用设备: {args.device}")
    print(f"📦 批次大小: {args.batch_size}")
    print("=" * 80)
    
    # 加载数据
    print("\n📥 加载数据集...")
    train_dataset = build_enhanced_dataset('train', args, use_external_test=False)
    val_dataset = build_enhanced_dataset('val', args, use_external_test=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"✅ 训练集: {len(train_dataset)} 样本")
    print(f"✅ 验证集: {len(val_dataset)} 样本")
    
    # 创建模型
    print("\n🤖 创建模型...")
    model = StandardCLIPBaseline(
        embed_dim=768,
        clinical_dim=7,
        num_classes=2,
        temperature=0.07
    ).to(args.device)
    
    print(f"📊 模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 损失函数和优化器
    criterion = StandardCLIPLoss(temperature=0.07, contrastive_weight=0.1)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler()
    
    # 训练
    print("\n🎓 开始训练...")
    best_auc = 0
    history = []
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n📅 Epoch {epoch}/{args.num_epochs}")
        print("-" * 60)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device, scaler)
        val_metrics = validate(model, val_loader, criterion, args.device)
        
        print(f"📊 Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"📊 Valid: Loss={val_metrics['loss']:.4f}, Acc={val_metrics['acc']:.2%}, F1={val_metrics['f1']:.4f}, AUC={val_metrics['auc']:.4f}")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['acc'],
            'val_auc': val_metrics['auc'],
            'val_f1': val_metrics['f1']
        })
        
        # 保存最佳模型
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'val_metrics': val_metrics
            }, output_dir / 'best_model.pth')
            print(f"💾 保存最佳模型 (Val AUC: {best_auc:.4f})")
    
    # 保存训练历史
    with open(output_dir.parent / 'logs' / f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n✅ 训练完成！")
    print(f"最佳AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()

