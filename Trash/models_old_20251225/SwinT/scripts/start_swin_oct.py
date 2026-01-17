#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# 添加项目根目录和utils目录到路径
UTILS_ROOT = PROJECT_ROOT / 'utils'
for p in [PROJECT_ROOT, UTILS_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# 注意：这里需要从项目根目录导入，因为utils在项目根目录下
from models.SwinT.swin_multimodal_model import SwinTOctClassifier
try:
    from utils.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
except ImportError:
    # 如果utils下没有，尝试从src.data导入
    from src.data.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


def build_loaders(data_path: str, batch_size: int):
    class Args:
        def __init__(self, data_path):
            self.data_path = data_path
            self.input_size = 224
            self.oct_num_frames = 120
            self.oct_cache_dir = 'oct_cache_optimized'
            self.use_text_contrastive = False
            self.use_pretrained_backbones = True
            self.oct_points = 12
            self.oct_frames_per_point = 10

    args = Args(data_path)
    train_set = EnhancedMultimodalCervicalDataset(
        root=os.path.join(data_path, 'train'),
        is_train='train',
        args=args,
        transform=None,
        use_enhanced_oct=True,
        cache_oct_features=True,
    )
    test_set = EnhancedMultimodalCervicalDataset(
        root=os.path.join(data_path, 'test'),
        is_train='test',
        args=args,
        transform=None,
        use_enhanced_oct=True,
        cache_oct_features=True,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='5centers_multi')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--output_root', type=str, default='models/SwinT/_results')
    parser.add_argument('--no_pretrained', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    output_dir = os.path.join(args.output_root, 'oct')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = build_loaders(args.data_path, args.batch_size)

    model = SwinTOctClassifier(
        num_classes=2,
        embed_dim=768,
        dropout=0.2,
        oct_num_frames=120,
        swin_name='swin_tiny_patch4_window7_224',
        pretrained=(not args.no_pretrained),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate * 0.6, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - max(1, args.epochs // 10)))
    criterion = FocalLoss(alpha=torch.tensor([0.325, 0.675]).to(device), gamma=2.5)
    scaler = GradScaler()

    best_auc = 0.0
    history = {k: [] for k in ['train_loss','train_acc','val_loss','val_acc','val_auc','val_f1','val_precision','val_recall','learning_rate']}

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            if isinstance(batch, dict):
                oct_images = batch.get('oct_images')
                labels = batch['label'] if 'label' in batch else batch.get('labels')
            else:
                if len(batch) >= 4:
                    oct_images, _, _, labels = batch[:4]
                else:
                    oct_images, labels = batch[0], batch[-1]
            oct_images = oct_images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(oct_images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        train_loss = epoch_loss / max(1, len(train_loader))
        train_acc = 100.0 * correct / max(1, total)

        # eval
        model.eval()
        val_loss = 0.0
        all_labels, all_probs, all_preds = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    oct_images = batch.get('oct_images')
                    labels = batch['label'] if 'label' in batch else batch.get('labels')
                else:
                    if len(batch) >= 4:
                        oct_images, _, _, labels = batch[:4]
                    else:
                        oct_images, labels = batch[0], batch[-1]
                oct_images = oct_images.to(device)
                labels = labels.to(device)

                with autocast():
                    outputs = model(oct_images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = outputs.argmax(dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_loss = val_loss / max(1, len(test_loader))
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            val_auc = 0.5
        val_acc = accuracy_score(all_labels, all_preds) * 100.0
        val_f1 = f1_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds, zero_division=0)
        val_recall = recall_score(all_labels, all_preds, zero_division=0)

        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        history['val_auc'].append(float(val_auc))
        history['val_f1'].append(float(val_f1))
        history['val_precision'].append(float(val_precision))
        history['val_recall'].append(float(val_recall))
        history['learning_rate'].append(float(scheduler.get_last_lr()[0]))

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(), 'val_auc': float(val_auc)}, os.path.join(output_dir, 'best_model_oct.pth'))

        # save metrics per-epoch
        with open(os.path.join(output_dir, 'metrics_oct.json'), 'w') as f:
            json.dump({k: v for k, v in history.items()}, f, indent=2)

        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | Train {train_loss:.4f}/{train_acc:.2f}% | Val {val_loss:.4f}/{val_acc:.2f}% AUC {val_auc:.4f}")

    with open(os.path.join(output_dir, 'training_summary_oct.json'), 'w') as f:
        json.dump({'best_auc': float(best_auc), 'epochs': args.epochs}, f, indent=2)


if __name__ == '__main__':
    main()


