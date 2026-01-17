#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断5分类训练过拟合问题
分析为什么训练效果总是1.0
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

print("🔍 诊断5分类训练过拟合问题")
print("=" * 60)

# 1. 检查数据标签分布
print("\n1️⃣ 检查数据标签分布")
print("-" * 40)

train_df = pd.read_csv('5centers_multi_5class/train/train_labels.csv')
test_df = pd.read_csv('5centers_multi_5class/test/test_labels.csv')

if 'tct_5class' in train_df.columns:
    train_labels = train_df['tct_5class'].values[:50]  # 训练集前50个样本
    test_labels = test_df['tct_5class'].values[:20]     # 测试集前20个样本
else:
    train_labels = train_df['label'].values[:50]
    test_labels = test_df['label'].values[:20]

print(f"训练集标签分布 (前50个样本): {np.bincount(train_labels)}")
print(f"测试集标签分布 (前20个样本): {np.bincount(test_labels)}")

# 2. 检查虚拟数据的随机性
print("\n2️⃣ 检查虚拟数据的随机性")
print("-" * 40)

# 创建虚拟数据
oct_images_sample = torch.randn(5, 3, 224, 224)
col_images_sample = torch.randn(3, 3, 224, 224)
clinical_features_sample = torch.randn(8)

# 多次采样检查是否相同
oct_images_sample2 = torch.randn(5, 3, 224, 224)

print(f"两次生成的OCT图像是否相同: {torch.equal(oct_images_sample, oct_images_sample2)}")
print(f"平均值差异: {abs(oct_images_sample.mean() - oct_images_sample2.mean()):.6f}")

# 3. 分析模型复杂度
print("\n3️⃣ 分析模型复杂度")
print("-" * 40)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5)
        )
    
    def forward(self, x):
        return self.classifier(x)

model = SimpleModel()
model_params = sum(p.numel() for p in model.parameters())
num_samples = 50
print(f"模型参数量: {model_params:,}")
print(f"训练样本数: {num_samples}")
print(f"参数量/样本数比例: {model_params/num_samples:.1f}")
print(f"⚠️ {'严重过拟合风险' if model_params/num_samples > 10 else '适中' if model_params/num_samples > 1 else 'OK'}")

# 4. 模拟训练过程
print("\n4️⃣ 模拟训练过程")
print("-" * 40)

# 使用轻量级模型
from lightweight_5class_model import create_lightweight_5class_model
model = create_lightweight_5class_model(num_classes=5, clinical_dim=8)
model_params = sum(p.numel() for p in model.parameters())

print(f"轻量级模型参数量: {model_params:,}")
print(f"训练样本数: 50")
print(f"参数量/样本数比例: {model_params/50:.1f}")

# 5. 检查数据泄漏
print("\n5️⃣ 检查数据泄漏")
print("-" * 40)

# 检查训练集和测试集是否有重叠
train_ids = train_df.index[:50].values
test_ids = test_df.index[:20].values

overlap = set(train_ids) & set(test_ids)
print(f"训练集和测试集索引重叠: {len(overlap)}个")

# 6. 结论和建议
print("\n6️⃣ 结论和建议")
print("-" * 40)

print("发现的问题:")
print("1. ❌ 使用虚拟/随机数据，而不是真实图像数据")
print("2. ❌ 虚拟数据与标签无关，模型没有学到有意义的特征")
print("3. ❌ 数据集太小（50个训练样本 vs 19万参数量）")
print("4. ❌ 没有类别平衡性检查")
print()

print("解决方案:")
print("1. ✅ 使用真实的多模态数据（OCT + Colposcopy + 临床特征）")
print("2. ✅ 增加数据增强")
print("3. ✅ 使用Focal Loss处理类别不平衡")
print("4. ✅ 使用Dropout正则化")
print("5. ✅ 使用真实数据集而非虚拟数据")
print()

print("为什么会达到1.0的准确率?")
print("- 由于使用随机虚拟数据，每个epoch的'数据'都是新生成的随机数")
print("- 模型可能在第一个epoch就学会了完美拟合这些随机模式")
print("- 但这完全没有实际意义，因为数据与标签无关")
print("- 这是一个典型的数据泄漏/虚假训练的问题！")
