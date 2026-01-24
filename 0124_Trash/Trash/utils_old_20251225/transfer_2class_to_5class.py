#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从2分类迁移到5分类
利用已有的78%准确率的2分类模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import json
import os

# 导入模型
from cnn_multimodal_model import CNNMultimodalTransformer

class Transfer5ClassModel(nn.Module):
    """基于2分类模型扩展的5分类模型"""
    
    def __init__(self, base_model_path='cnn_training_latest/best_model.pth', num_classes=5):
        super().__init__()
        
        # 加载2分类模型的检查点
        checkpoint = torch.load(base_model_path, map_location='cpu')
        
        # 创建5分类模型（保持编码器架构）
        self.model = CNNMultimodalTransformer(
            num_classes=num_classes,
            embed_dim=512,
            clinical_dim=8,
            dropout=0.5
        )
        
        # 使用预训练权重的策略
        pretrained_dict = checkpoint['state_dict']
        model_dict = self.model.state_dict()
        
        # 只加载兼容的层（跳过分类器）
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and 'classifier' not in k}
        
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=False)
        
        print(f"✅ 已加载2分类模型的预训练权重")
        print(f"   加载的层: {len(pretrained_dict)}")
        print(f"   跳过的层: 分类器层（将重新初始化）")
    
    def forward(self, oct_images, col_images, clinical_features):
        return self.model(oct_images, col_images, clinical_features)
    
    def freeze_encoders(self):
        """冻结编码器，只训练分类器"""
        for param in self.model.oct_encoder.parameters():
            param.requires_grad = False
        for param in self.model.col_encoder.parameters():
            param.requires_grad = False
        for param in self.model.clinical_encoder.parameters():
            param.requires_grad = False
        print("🔒 编码器已冻结，只训练分类器")

def create_transfer_dataset():
    """创建迁移学习用的数据集"""
    print("📂 创建迁移学习数据集...")
    
    # 加载原始数据
    train_df = pd.read_csv('5centers_multi/train_labels.csv')
    test_df = pd.read_csv('5centers_multi/test_labels.csv')
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # 创建5分类标签
    labels_5class = []
    for _, row in all_df.iterrows():
        tct_val = str(row['TCT清洗']) if pd.notna(row.get('TCT清洗')) else 'NILM'
        label_val = row.get('label', 0)
        
        if tct_val == 'NILM':
            labels_5class.append(0)
        elif tct_val == 'ASC-US':
            labels_5class.append(1)
        elif tct_val == 'LSIL':
            labels_5class.append(2)
        elif tct_val == 'HSIL':
            labels_5class.append(3)
        elif tct_val == '1' or label_val == 1:
            labels_5class.append(4)
        else:
            labels_5class.append(0)
    
    all_df['tct_5class'] = labels_5class
    
    # 重新分割（基于5分类标签）
    train_len = len(train_df)
    train_df_5class = all_df.iloc[:train_len].copy()
    test_df_5class = all_df.iloc[train_len:].copy()
    
    # 保存
    os.makedirs('5centers_multi_5class_transfer', exist_ok=True)
    os.makedirs('5centers_multi_5class_transfer/train', exist_ok=True)
    os.makedirs('5centers_multi_5class_transfer/test', exist_ok=True)
    
    train_df_5class.to_csv('5centers_multi_5class_transfer/train/train_labels.csv', index=False)
    test_df_5class.to_csv('5centers_multi_5class_transfer/test/test_labels.csv', index=False)
    
    print(f"✅ 数据集已创建")
    print(f"   训练集: {len(train_df_5class)} 样本")
    print(f"   测试集: {len(test_df_5class)} 样本")
    
    # 检查类别分布
    print("\n📊 类别分布:")
    class_names = ['NILM', 'ASC-US', 'LSIL', 'HSIL', 'Cancer']
    train_dist = train_df_5class['tct_5class'].value_counts().sort_index()
    for idx, count in train_dist.items():
        print(f"   类别{idx} ({class_names[idx]}): {count} 样本 ({count/len(train_df_5class)*100:.1f}%)")
    
    return train_df_5class, test_df_5class

def train_transfer_model():
    """训练迁移学习的5分类模型"""
    print("\n🚀 开始5分类迁移学习训练")
    print("=" * 60)
    
    # 创建数据集
    train_df, test_df = create_transfer_dataset()
    
    # 创建模型
    print("\n🏗️ 创建5分类模型...")
    model = Transfer5Model(
        base_model_path='cnn_training_latest/best_model.pth',
        num_classes=5
    )
    
    # 第一阶段: 冻结编码器，只训练分类器
    print("\n🔒 第一阶段: 冻结编码器训练分类器")
    model.freeze_encoders()
    
    # 这里应该设置训练循环
    # 由于数据加载问题，这里只是演示框架
    
    print("\n✅ 迁移学习框架已准备完成")
    print("   需要: 修复数据加载后再运行完整训练")
    
    return model

if __name__ == "__main__":
    train_transfer_model()



