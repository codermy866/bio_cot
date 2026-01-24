#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级5分类训练脚本
使用轻量级模型快速验证5分类功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

# 导入轻量级模型
from lightweight_5class_model import create_lightweight_5class_model

class Lightweight5ClassDataset(Dataset):
    """轻量级5分类数据集"""
    
    def __init__(self, csv_path, num_samples=100):
        self.data_df = pd.read_csv(csv_path)
        self.num_samples = min(num_samples, len(self.data_df))
        
        # 使用tct_5class列作为标签
        if 'tct_5class' in self.data_df.columns:
            self.labels = self.data_df['tct_5class'].values[:self.num_samples]
        else:
            # 如果没有tct_5class列，使用原始label并映射到5分类
            self.labels = self.data_df['label'].values[:self.num_samples]
            # 简单映射：0->0, 1->4 (正常->癌变)
            self.labels = np.where(self.labels == 0, 0, 4)
        
        print(f"数据集标签分布: {np.bincount(self.labels)}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 创建轻量级虚拟数据
        oct_images = torch.randn(5, 3, 224, 224)   # 减少OCT帧数
        col_images = torch.randn(3, 3, 224, 224)    # Colposcopy图像
        clinical_features = torch.randn(8)          # 临床特征
        label = self.labels[idx]
        
        return oct_images, col_images, clinical_features, label, {}

def train_lightweight_5class():
    """轻量级5分类训练"""
    print("🚀 开始轻量级5分类训练")
    print("=" * 50)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 创建数据集
    train_dataset = Lightweight5ClassDataset('5centers_multi_5class/train/train_labels.csv', num_samples=50)
    test_dataset = Lightweight5ClassDataset('5centers_multi_5class/test/test_labels.csv', num_samples=20)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)  # 减少batch size
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    # 创建轻量级模型
    model = create_lightweight_5class_model(num_classes=5, clinical_dim=8)
    model = model.to(device)
    
    print(f"🏗️ 轻量级模型创建完成")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练循环
    num_epochs = 5
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (oct_data, col_data, clinical_data, labels, metadata) in enumerate(train_loader):
            oct_data = oct_data.to(device)
            col_data = col_data.to(device)
            clinical_data = clinical_data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(oct_data, col_data, clinical_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for oct_data, col_data, clinical_data, labels, metadata in test_loader:
                oct_data = oct_data.to(device)
                col_data = col_data.to(device)
                clinical_data = clinical_data.to(device)
                labels = labels.to(device)
                
                outputs = model(oct_data, col_data, clinical_data)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(test_loader)
        val_acc = val_correct / val_total
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # 计算F1分数
        if len(all_labels) > 0:
            f1_macro = f1_score(all_labels, all_preds, average='macro')
            print(f"验证 - F1(macro): {f1_macro:.4f}")
    
    # 保存结果
    os.makedirs('lightweight_5class_results', exist_ok=True)
    
    # 保存模型
    torch.save({
        'state_dict': model.state_dict(),
        'num_classes': 5,
        'class_names': ['NILM(正常)', 'ASC-US(非典型)', 'LSIL(低度病变)', 'HSIL(高度病变)', 'Cancer(癌变)']
    }, 'lightweight_5class_results/lightweight_model_5class.pth')
    
    # 保存训练历史
    with open('lightweight_5class_results/training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('lightweight_5class_results/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 最终评估
    print(f"\n📊 最终评估结果:")
    print(f"  最终训练准确率: {train_acc:.4f}")
    print(f"  最终验证准确率: {val_acc:.4f}")
    
    if len(all_labels) > 0:
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        print(f"  最终验证F1分数: {f1_macro:.4f}")
        
        # 分类报告
        print(f"\n📋 分类报告:")
        class_names = ['NILM(正常)', 'ASC-US(非典型)', 'LSIL(低度病变)', 'HSIL(高度病变)', 'Cancer(癌变)']
        report = classification_report(all_labels, all_preds, target_names=class_names)
        print(report)
    
    print(f"\n🎉 轻量级5分类训练完成！")
    print(f"📁 结果保存在: lightweight_5class_results/")
    
    return model, history

if __name__ == "__main__":
    train_lightweight_5class()
