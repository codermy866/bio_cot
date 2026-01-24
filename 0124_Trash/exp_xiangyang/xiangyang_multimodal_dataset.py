#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
襄阳多模态数据集加载器
支持OCT、Colposcopy和临床特征
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from typing import Tuple, Dict, List
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import glob


class XiangyangMultimodalDataset(Dataset):
    def __init__(
        self, 
        data_root: str, 
        split: str = 'train', 
        transform=None, 
        test_size: float = 0.2, 
        random_state: int = 42,
        oct_num_frames: int = 60,
        oct_frames_per_point: int = 5
    ):
        self.data_root = Path(data_root)
        self.transform = transform
        self.split = split
        self.oct_num_frames = oct_num_frames
        self.oct_frames_per_point = oct_frames_per_point
        
        # 定义类别映射
        negative_folders = ['低级别', '炎']
        positive_folders = ['癌', '高级别']
        
        # 加载临床信息
        clinical_csv = self.data_root / 'clinical_info' / 'clinical_info.csv'
        clinical_df = None
        if clinical_csv.exists():
            try:
                clinical_df = pd.read_csv(clinical_csv, encoding='utf-8')
            except:
                try:
                    clinical_df = pd.read_csv(clinical_csv, encoding='gbk')
                except:
                    print("⚠️ 无法读取临床信息CSV文件")
        
        # 收集所有样本
        all_samples = []
        
        # 处理阴性类别
        for folder_name in negative_folders:
            folder_path = self.data_root / folder_name
            if folder_path.exists():
                oct_dir = folder_path / 'oct'
                col_dir = folder_path / 'col'
                
                # 获取所有OCT ID（从文件名提取）
                if oct_dir.exists():
                    oct_files = list(oct_dir.glob('*.tiff')) + list(oct_dir.glob('*.tif'))
                    for oct_file in oct_files:
                        # 从文件名提取ID：M22102_2023_P0000011_circle_3.0x3.1_C2_S2.tiff -> M22102_2023_P0000011
                        oct_id = oct_file.stem.split('_circle')[0]
                        
                        # 查找对应的colposcopy图像
                        col_files = list((col_dir).glob(f'{oct_id}_col_*.jpg'))
                        if len(col_files) == 0:
                            # 尝试其他匹配方式
                            col_files = list((col_dir).glob(f'{oct_id.split("_")[-1]}*col*.jpg'))
                        
                        # 获取临床信息
                        clinical_info = self._get_clinical_info(clinical_df, oct_id)
                        
                        all_samples.append({
                            'oct_id': oct_id,
                            'oct_path': oct_dir,
                            'oct_file': oct_file,
                            'col_path': col_dir,
                            'col_files': sorted(col_files)[:3],  # 最多3张
                            'label': 0,  # 阴性
                            'category': folder_name,
                            'clinical': clinical_info
                        })
        
        # 处理阳性类别
        for folder_name in positive_folders:
            folder_path = self.data_root / folder_name
            if folder_path.exists():
                oct_dir = folder_path / 'oct'
                col_dir = folder_path / 'col'
                
                if oct_dir.exists():
                    oct_files = list(oct_dir.glob('*.tiff')) + list(oct_dir.glob('*.tif'))
                    for oct_file in oct_files:
                        oct_id = oct_file.stem.split('_circle')[0]
                        
                        col_files = list((col_dir).glob(f'{oct_id}_col_*.jpg'))
                        if len(col_files) == 0:
                            col_files = list((col_dir).glob(f'{oct_id.split("_")[-1]}*col*.jpg'))
                        
                        clinical_info = self._get_clinical_info(clinical_df, oct_id)
                        
                        all_samples.append({
                            'oct_id': oct_id,
                            'oct_path': oct_dir,
                            'oct_file': oct_file,
                            'col_path': col_dir,
                            'col_files': sorted(col_files)[:3],
                            'label': 1,  # 阳性
                            'category': folder_name,
                            'clinical': clinical_info
                        })
        
        if not all_samples:
            print(f"⚠️ 警告: 从 {self.data_root} 加载了 0 个样本")
            self.samples = []
            return
        
        # 划分训练集和验证集
        labels = [s['label'] for s in all_samples]
        train_samples, val_samples = train_test_split(
            all_samples, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=labels
        )
        
        if self.split == 'train':
            self.samples = train_samples
        elif self.split == 'val':
            self.samples = val_samples
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")
        
        print(f"✅ {self.split}数据集加载完成: {len(self.samples)} 个样本")
        labels = [s['label'] for s in self.samples]
        neg_count = labels.count(0)
        pos_count = labels.count(1)
        print(f"   标签分布: 阴性={neg_count}, 阳性={pos_count}")
    
    def _get_clinical_info(self, clinical_df: pd.DataFrame, oct_id: str) -> Dict:
        """获取临床信息"""
        default_info = {
            'age': 50.0,
            'hpv': 0.0,
            'tct': np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # 5维TCT one-hot
        }
        
        if clinical_df is None:
            return default_info
        
        # 尝试匹配oct_id
        matches = clinical_df[clinical_df['oct_id'].str.contains(oct_id.split('_')[-1], na=False)]
        if len(matches) == 0:
            return default_info
        
        row = matches.iloc[0]
        
        # 解析年龄
        age = float(row['age']) if pd.notna(row['age']) else 50.0
        
        # 解析HPV
        hpv = 0.0
        if pd.notna(row['hpv']):
            hpv_str = str(row['hpv']).lower()
            if any(k in hpv_str for k in ['16', '18', 'positive', '阳性', '高危']):
                hpv = 1.0
        
        # 解析TCT（简化为5维one-hot）
        tct = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        if pd.notna(row['tct']):
            tct_str = str(row['tct']).upper()
            if 'ASC-US' in tct_str:
                tct[0] = 1.0
            elif 'ASC-H' in tct_str:
                tct[1] = 1.0
            elif 'LSIL' in tct_str:
                tct[2] = 1.0
            elif 'HSIL' in tct_str:
                tct[3] = 1.0
            elif 'SCC' in tct_str or '癌' in tct_str:
                tct[4] = 1.0
        
        return {
            'age': age,
            'hpv': hpv,
            'tct': tct
        }
    
    def _load_oct_frames(self, oct_path: Path, oct_id: str) -> torch.Tensor:
        """加载OCT帧序列"""
        # 查找所有匹配的OCT文件
        oct_files = sorted(list(oct_path.glob(f'{oct_id}*.tiff')) + 
                          list(oct_path.glob(f'{oct_id}*.tif')))
        
        if not oct_files:
            # 返回零张量
            return torch.zeros(self.oct_num_frames, 3, 224, 224)
        
        # 均匀采样到指定帧数
        if len(oct_files) >= self.oct_num_frames:
            indices = np.linspace(0, len(oct_files) - 1, self.oct_num_frames, dtype=int)
            oct_files = [oct_files[i] for i in indices]
        else:
            # 循环填充
            while len(oct_files) < self.oct_num_frames:
                oct_files.extend(oct_files)
            oct_files = oct_files[:self.oct_num_frames]
        
        # 加载图像
        frames = []
        for oct_file in oct_files:
            try:
                img = Image.open(oct_file).convert('RGB')
                img = img.resize((224, 224))
                if self.transform:
                    img = self.transform(img)
                else:
                    from torchvision import transforms
                    img = transforms.ToTensor()(img)
                frames.append(img)
            except Exception as e:
                # 使用零张量代替
                frames.append(torch.zeros(3, 224, 224))
        
        # 堆叠为 [F, C, H, W]
        return torch.stack(frames)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 加载OCT帧序列
        oct_frames = self._load_oct_frames(sample['oct_path'], sample['oct_id'])
        
        # 加载Colposcopy图像
        col_images = []
        for col_file in sample['col_files']:
            try:
                img = Image.open(sample['col_path'] / col_file).convert('RGB')
                img = img.resize((224, 224))
                if self.transform:
                    img = self.transform(img)
                else:
                    from torchvision import transforms
                    img = transforms.ToTensor()(img)
                col_images.append(img)
            except:
                col_images.append(torch.zeros(3, 224, 224))
        
        # 如果少于3张，用零张量填充
        while len(col_images) < 3:
            col_images.append(torch.zeros(3, 224, 224))
        
        col_images = torch.stack(col_images[:3])  # [3, C, H, W]
        
        # 构建临床特征向量 [age, hpv, tct_5dim] = 7维
        clinical = sample['clinical']
        clinical_features = torch.tensor([
            clinical['age'] / 100.0,  # 归一化年龄
            clinical['hpv'],
            *clinical['tct'].tolist()
        ], dtype=torch.float32)
        
        return {
            'oct_images': oct_frames,  # [F, C, H, W]
            'colposcopy_images': col_images,  # [3, C, H, W]
            'clinical_features': clinical_features,  # [7]
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'oct_id': sample['oct_id']
        }


