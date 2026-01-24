#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 3: TTPA (Test-Time Prior Adaptation) 测试时适配
在测试时进行在线适配，提升AUC的关键trick
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
from src.models.bida.bio_cot_model import BioCOTModel
from src.models.bida.losses import SinkhornDistance


def test_time_adaptation(model, student_prior, test_loader, device, adaptation_steps=1, lr=1e-3):
    """
    TTPA: Test-Time Prior Adaptation
    
    Args:
        model: BioCOTModel
        student_prior: StudentPriorNet (实际上在model内部)
        test_loader: 测试数据加载器
        device: 设备
        adaptation_steps: 每个batch的适配步数
        lr: 适配学习率
    """
    model.eval()
    
    # 策略1: 仅打开BN/LN层的参数更新 (Tent策略)
    # 策略2: 仅打开Feature Encoder的参数
    # 这里使用策略2：仅打开image_encoder的参数
    for name, param in model.named_parameters():
        if "image_encoder" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # 创建优化器（仅优化image_encoder）
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        momentum=0.9
    )
    
    # Sinkhorn损失（用于适配）
    sinkhorn_loss = SinkhornDistance(eps=0.1, max_iter=100)
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    print("🔄 开始TTPA测试...")
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="TTPA Testing")):
        if isinstance(batch, dict):
            oct_feat = batch['oct_features'].to(device)
            colpo_feat = batch['colposcopy_features'].to(device)
            clinical_feat = batch['clinical_features'].to(device)
            labels = batch['label'].to(device)
            clinical_data = batch.get('clinical_data', None)
        else:
            continue
        
        # 1. 适配步骤 (Adaptation Step)
        for _ in range(adaptation_steps):
            optimizer.zero_grad()
            
            with autocast():
                # 获取语义锚点
                if clinical_data is not None:
                    from src.models.bida.prior_net import build_clinical_vector
                    clinical_vec = build_clinical_vector(clinical_data, device=device)
                else:
                    if clinical_feat.size(-1) == 7:
                        clinical_vec = clinical_feat
                    else:
                        clinical_vec = torch.zeros(len(labels), 7, device=device)
                
                z_sem = model.student_prior(clinical_vec)  # [B, 768]
                
                # 获取图像特征
                if oct_feat.size(-1) != colpo_feat.size(-1):
                    min_dim = min(oct_feat.size(-1), colpo_feat.size(-1))
                    image_feat = (oct_feat[..., :min_dim] + colpo_feat[..., :min_dim]) / 2
                else:
                    image_feat = (oct_feat + colpo_feat) / 2
                
                z_causal, _ = model.image_encoder(image_feat)
                
                # 仅优化分布对齐损失（无监督）
                loss_tta = sinkhorn_loss(z_causal, z_sem)
            
            loss_tta.backward()
            optimizer.step()
        
        # 2. 预测步骤 (Prediction Step)
        with torch.no_grad():
            outputs = model(
                oct_features=oct_feat,
                colpo_features=colpo_feat,
                clinical_features=clinical_feat,
                clinical_data=clinical_data,
                return_loss_components=False,
                use_counterfactual=False
            )
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1)
            
            _, preds = logits.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    try:
        if len(set(all_labels)) < 2:
            auc = 0.0
        else:
            auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'acc': acc,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }


def standard_test(model, test_loader, device):
    """标准测试（无TTPA）"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Standard Testing"):
            if isinstance(batch, dict):
                oct_feat = batch['oct_features'].to(device)
                colpo_feat = batch['colposcopy_features'].to(device)
                clinical_feat = batch['clinical_features'].to(device)
                labels = batch['label'].to(device)
                clinical_data = batch.get('clinical_data', None)
            else:
                continue
            
            with autocast():
                outputs = model(
                    oct_features=oct_feat,
                    colpo_features=colpo_feat,
                    clinical_features=clinical_feat,
                    clinical_data=clinical_data,
                    return_loss_components=False,
                    use_counterfactual=False
                )
                logits = outputs['logits']
                probs = torch.softmax(logits, dim=1)
                
                _, preds = logits.max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    try:
        if len(set(all_labels)) < 2:
            auc = 0.0
        else:
            auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'acc': acc,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='TTPA测试')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--data_root', type=str, 
                       default='/data2/hmy/5Center_datas/5centers_multi_leave_centers_out',
                       help='数据根目录')
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='val', help='测试集')
    parser.add_argument('--device', type=str, default='cuda:1', help='设备')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--use_ttpa', action='store_true', help='使用TTPA')
    parser.add_argument('--adaptation_steps', type=int, default=1, help='每个batch的适配步数')
    parser.add_argument('--ttpa_lr', type=float, default=1e-3, help='TTPA学习率')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 加载模型
    print(f"📂 加载模型: {args.model_path}")
    model = BioCOTModel(embed_dim=768, num_classes=2, num_centers=5, input_dim=512).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("✅ 模型加载完成")
    
    # 加载测试集
    print(f"📂 加载测试集: {args.split}")
    if args.split == 'val':
        test_dataset = EnhancedMultimodalCervicalDataset(
            root=Path(args.data_root) / 'internal_train' / 'val',
            labels_file=Path(args.data_root) / 'val_labels.csv',
            use_pretrained_backbones=True
        )
    else:
        test_dataset = EnhancedMultimodalCervicalDataset(
            root=Path(args.data_root) / 'external_test',
            labels_file=Path(args.data_root) / 'external_test_labels.csv',
            use_pretrained_backbones=True
        )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"✅ 测试集加载完成: {len(test_dataset)} 个样本")
    
    # 测试
    if args.use_ttpa:
        print("🔄 使用TTPA进行测试...")
        results = test_time_adaptation(
            model, None, test_loader, device,
            adaptation_steps=args.adaptation_steps,
            lr=args.ttpa_lr
        )
        print(f"\n📊 TTPA测试结果:")
    else:
        print("🔄 标准测试（无TTPA）...")
        results = standard_test(model, test_loader, device)
        print(f"\n📊 标准测试结果:")
    
    print(f"  Accuracy: {results['acc']:.4f}")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    {results['confusion_matrix']}")


if __name__ == '__main__':
    main()

