#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强数据处理模块
解决类别不平衡问题，实施高级数据增强策略
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedDataAugmentation:
    """高级数据增强策略，专门针对医学图像"""
    
    def __init__(self, input_size: int = 224, is_training: bool = True):
        self.input_size = input_size
        self.is_training = is_training
        
        if is_training:
            # 训练时的强增强
            self.transform = A.Compose([
                A.Resize(input_size, input_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=15, p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=0.3
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.RandomShadow(p=0.2),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.3
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            # 验证/测试时的标准化
            self.transform = A.Compose([
                A.Resize(input_size, input_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        return self.transform(image=image)['image']

class ClassBalancingStrategy:
    """类别平衡策略"""
    
    def __init__(self, strategy: str = 'smote', random_state: int = 42):
        self.strategy = strategy
        self.random_state = random_state
        self.sampler = None
        
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用类别平衡策略"""
        
        logger.info(f"原始数据分布: {np.bincount(y)}")
        
        if self.strategy == 'smote':
            sampler = SMOTE(random_state=self.random_state, k_neighbors=3)
        elif self.strategy == 'adasyn':
            sampler = ADASYN(random_state=self.random_state)
        elif self.strategy == 'borderline_smote':
            sampler = BorderlineSMOTE(random_state=self.random_state)
        elif self.strategy == 'smote_tomek':
            sampler = SMOTETomek(random_state=self.random_state)
        elif self.strategy == 'smote_enn':
            sampler = SMOTEENN(random_state=self.random_state)
        elif self.strategy == 'undersample':
            sampler = RandomUnderSampler(random_state=self.random_state)
        else:
            logger.warning(f"未知的平衡策略: {self.strategy}")
            return X, y
        
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            logger.info(f"平衡后数据分布: {np.bincount(y_resampled)}")
            return X_resampled, y_resampled
        except Exception as e:
            logger.error(f"类别平衡失败: {e}")
            return X, y

class EnhancedMultimodalDataset(Dataset):
    """增强的多模态数据集"""
    
    def __init__(
        self,
        data_path: str,
        clinical_data: pd.DataFrame,
        labels: np.ndarray,
        transform=None,
        is_training: bool = True
    ):
        self.data_path = Path(data_path)
        self.clinical_data = clinical_data
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        
        # 确保数据一致性
        assert len(clinical_data) == len(labels), "临床数据和标签长度不匹配"
        
        logger.info(f"数据集大小: {len(self.labels)}")
        logger.info(f"标签分布: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # 获取临床特征
        clinical_features = self.clinical_data.iloc[idx].values.astype(np.float32)
        
        # 获取图像路径（假设有OCT和COL图像）
        sample_id = self.clinical_data.index[idx]
        
        # 加载OCT图像
        oct_path = self.data_path / "oct" / f"{sample_id}.jpg"
        if oct_path.exists():
            oct_image = Image.open(oct_path).convert('RGB')
        else:
            # 创建占位图像
            oct_image = Image.new('RGB', (224, 224), color='black')
        
        # 加载COL图像
        col_path = self.data_path / "col" / f"{sample_id}.jpg"
        if col_path.exists():
            col_image = Image.open(col_path).convert('RGB')
        else:
            # 创建占位图像
            col_image = Image.new('RGB', (224, 224), color='black')
        
        # 应用数据增强
        if self.transform:
            oct_image = self.transform(oct_image)
            col_image = self.transform(col_image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'oct_image': oct_image,
            'col_image': col_image,
            'clinical_features': torch.tensor(clinical_features, dtype=torch.float32),
            'label': label,
            'sample_id': sample_id
        }

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, data_path: str, config: Dict):
        self.data_path = Path(data_path)
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
        """加载和预处理数据"""
        
        logger.info("开始加载和预处理数据...")
        
        # 加载临床数据
        clinical_data = self._load_clinical_data()
        
        # 加载标签
        labels = self._load_labels()
        
        # 数据预处理
        clinical_data = self._preprocess_clinical_data(clinical_data)
        
        # 类别平衡
        if self.config.get('use_class_balancing', True):
            clinical_data, labels = self._apply_class_balancing(clinical_data, labels)
        
        # 划分数据集
        train_data, val_data, test_data = self._split_dataset(clinical_data, labels)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = self._create_data_loaders(
            train_data, val_data, test_data
        )
        
        # 计算类别权重
        class_weights = self._calculate_class_weights(labels)
        
        logger.info("数据加载和预处理完成")
        
        return train_loader, val_loader, test_loader, class_weights
    
    def _load_clinical_data(self) -> pd.DataFrame:
        """加载临床数据"""
        # 这里需要根据实际数据格式调整
        # 假设有一个CSV文件包含临床特征
        clinical_file = self.data_path / "clinical_features.csv"
        
        if clinical_file.exists():
            clinical_data = pd.read_csv(clinical_file, index_col=0)
        else:
            # 创建示例数据
            logger.warning("未找到临床数据文件，创建示例数据")
            n_samples = 200
            clinical_data = pd.DataFrame({
                'age': np.random.normal(45, 15, n_samples),
                'bmi': np.random.normal(25, 5, n_samples),
                'systolic_bp': np.random.normal(120, 20, n_samples),
                'diastolic_bp': np.random.normal(80, 15, n_samples),
                'glucose': np.random.normal(100, 30, n_samples),
                'cholesterol': np.random.normal(200, 50, n_samples),
                'hdl': np.random.normal(50, 15, n_samples),
                'ldl': np.random.normal(120, 40, n_samples)
            })
        
        return clinical_data
    
    def _load_labels(self) -> np.ndarray:
        """加载标签"""
        # 这里需要根据实际数据格式调整
        labels_file = self.data_path / "labels.csv"
        
        if labels_file.exists():
            labels_df = pd.read_csv(labels_file, index_col=0)
            labels = labels_df['label'].values
        else:
            # 创建示例标签（2:1的不平衡比例）
            logger.warning("未找到标签文件，创建示例标签")
            n_samples = 200
            labels = np.concatenate([
                np.zeros(133, dtype=int),  # 正常样本
                np.ones(67, dtype=int)     # 异常样本
            ])
            np.random.shuffle(labels)
        
        return labels
    
    def _preprocess_clinical_data(self, clinical_data: pd.DataFrame) -> pd.DataFrame:
        """预处理临床数据"""
        # 处理缺失值
        clinical_data = clinical_data.fillna(clinical_data.median())
        
        # 异常值处理
        for col in clinical_data.columns:
            Q1 = clinical_data[col].quantile(0.25)
            Q3 = clinical_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            clinical_data[col] = clinical_data[col].clip(lower_bound, upper_bound)
        
        # 标准化
        clinical_data = pd.DataFrame(
            self.scaler.fit_transform(clinical_data),
            columns=clinical_data.columns,
            index=clinical_data.index
        )
        
        return clinical_data
    
    def _apply_class_balancing(self, clinical_data: pd.DataFrame, labels: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """应用类别平衡策略"""
        balancing_strategy = self.config.get('balancing_strategy', 'smote')
        
        balancer = ClassBalancingStrategy(strategy=balancing_strategy)
        clinical_resampled, labels_resampled = balancer.fit_resample(
            clinical_data.values, labels
        )
        
        # 重新创建DataFrame
        clinical_data_resampled = pd.DataFrame(
            clinical_resampled,
            columns=clinical_data.columns
        )
        
        return clinical_data_resampled, labels_resampled
    
    def _split_dataset(self, clinical_data: pd.DataFrame, labels: np.ndarray) -> Tuple[Dict, Dict, Dict]:
        """划分数据集"""
        # 首先划分训练集和临时集
        X_temp, X_test, y_temp, y_test = train_test_split(
            clinical_data, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # 再划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )
        
        train_data = {'clinical': X_train, 'labels': y_train}
        val_data = {'clinical': X_val, 'labels': y_val}
        test_data = {'clinical': X_test, 'labels': y_test}
        
        logger.info(f"训练集大小: {len(y_train)}")
        logger.info(f"验证集大小: {len(y_val)}")
        logger.info(f"测试集大小: {len(y_test)}")
        
        return train_data, val_data, test_data
    
    def _create_data_loaders(self, train_data: Dict, val_data: Dict, test_data: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建数据加载器"""
        
        # 数据增强
        train_transform = AdvancedDataAugmentation(is_training=True)
        val_transform = AdvancedDataAugmentation(is_training=False)
        
        # 创建数据集
        train_dataset = EnhancedMultimodalDataset(
            self.data_path, train_data['clinical'], train_data['labels'],
            transform=train_transform, is_training=True
        )
        
        val_dataset = EnhancedMultimodalDataset(
            self.data_path, val_data['clinical'], val_data['labels'],
            transform=val_transform, is_training=False
        )
        
        test_dataset = EnhancedMultimodalDataset(
            self.data_path, test_data['clinical'], test_data['labels'],
            transform=val_transform, is_training=False
        )
        
        # 创建数据加载器
        batch_size = self.config.get('batch_size', 32)
        num_workers = self.config.get('num_workers', 8)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def _calculate_class_weights(self, labels: np.ndarray) -> Dict:
        """计算类别权重"""
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        num_classes = len(class_counts)
        
        # 计算权重（逆频率）
        class_weights = total_samples / (num_classes * class_counts)
        
        # 归一化权重
        class_weights = class_weights / class_weights.sum() * num_classes
        
        weights_dict = {
            'class_weights': torch.tensor(class_weights, dtype=torch.float32),
            'class_counts': class_counts,
            'class_weights_dict': {i: class_weights[i] for i in range(num_classes)}
        }
        
        logger.info(f"类别权重: {weights_dict['class_weights_dict']}")
        
        return weights_dict

def create_enhanced_data_loaders(data_path: str, config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """创建增强的数据加载器"""
    
    processor = DataProcessor(data_path, config)
    return processor.load_and_preprocess_data()

if __name__ == "__main__":
    # 测试数据处理器
    config = {
        'batch_size': 32,
        'num_workers': 8,
        'use_class_balancing': True,
        'balancing_strategy': 'smote'
    }
    
    train_loader, val_loader, test_loader, class_weights = create_enhanced_data_loaders(
        "5centers_multi", config
    )
    
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    print(f"类别权重: {class_weights['class_weights_dict']}")