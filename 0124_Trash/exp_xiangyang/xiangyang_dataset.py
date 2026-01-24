#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
襄阳数据集加载器
从文件夹结构加载图像数据：
- 低级别/ 和 炎/ → label=0 (阴性)
- 高级别/ 和 癌/ → label=1 (阳性)
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split


class XiangyangDataset(Dataset):
    """襄阳按点图片分类数据集"""
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',  # 'train', 'val', or 'test'
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        transform=None,
        input_size: int = 224,
        seed: int = 42
    ):
        """
        Args:
            data_root: 数据集根目录，包含子文件夹：低级别、炎、高级别、癌
            split: 数据集分割类型
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            transform: 数据增强变换
            input_size: 输入图像尺寸
            seed: 随机种子
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.input_size = input_size
        
        # 定义类别映射
        # 阴性 (label=0): 低级别、炎
        # 阳性 (label=1): 高级别、癌
        self.class_folders = {
            0: ['低级别', '炎'],  # 阴性
            1: ['高级别', '癌']   # 阳性
        }
        
        # 加载所有图像路径和标签
        self.samples = []
        self._load_samples()
        
        # 数据集分割
        self._split_dataset(train_ratio, val_ratio, test_ratio, seed)
        
        print(f"✅ {split}数据集加载完成: {len(self.samples)} 个样本")
        if len(self.samples) > 0:
            labels = [s[1] for s in self.samples]
            label_counts = np.bincount(labels)
            print(f"   标签分布: 阴性={label_counts[0] if len(label_counts) > 0 else 0}, "
                  f"阳性={label_counts[1] if len(label_counts) > 1 else 0}")
    
    def _load_samples(self):
        """加载所有图像路径和标签"""
        for label, folder_names in self.class_folders.items():
            for folder_name in folder_names:
                folder_path = self.data_root / folder_name
                if not folder_path.exists():
                    print(f"⚠️ 警告: 文件夹不存在: {folder_path}")
                    continue
                
                # 查找所有图像文件（包括TIFF格式）
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
                for ext in image_extensions:
                    # 小写扩展名
                    for img_path in folder_path.glob(f'*{ext}'):
                        # 跳过隐藏文件（以._开头的文件）
                        if not img_path.name.startswith('._'):
                            self.samples.append((str(img_path), label))
                    # 大写扩展名
                    for img_path in folder_path.glob(f'*{ext.upper()}'):
                        if not img_path.name.startswith('._'):
                            self.samples.append((str(img_path), label))
        
        print(f"📂 从 {self.data_root} 加载了 {len(self.samples)} 个图像文件")
    
    def _split_dataset(self, train_ratio, val_ratio, test_ratio, seed):
        """按比例分割数据集"""
        if len(self.samples) == 0:
            return
        
        # 按标签分层分割
        labels = [s[1] for s in self.samples]
        paths = [s[0] for s in self.samples]
        
        # 先分出训练集
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            paths, labels, test_size=(1 - train_ratio), stratify=labels, random_state=seed
        )
        
        # 再从剩余数据中分出验证集和测试集
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=(1 - val_test_ratio), 
            stratify=temp_labels, random_state=seed
        )
        
        # 根据split选择对应的数据
        if self.split == 'train':
            self.samples = [(p, l) for p, l in zip(train_paths, train_labels)]
        elif self.split == 'val':
            self.samples = [(p, l) for p, l in zip(val_paths, val_labels)]
        elif self.split == 'test':
            self.samples = [(p, l) for p, l in zip(test_paths, test_labels)]
        else:
            raise ValueError(f"未知的split类型: {self.split}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ 警告: 无法加载图像 {img_path}: {e}")
            # 创建黑色占位图像
            image = Image.new('RGB', (self.input_size, self.input_size), color='black')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        else:
            # 默认变换
            transform = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)
        
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            'image': image,
            'label': label,
            'path': img_path
        }

