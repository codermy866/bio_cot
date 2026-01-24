#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 2.0: 数据集加载器（支持LLM嵌入）
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import pickle
from typing import Optional, Dict

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.exp_5centers.train_bio_cot_5centers_multimodal import FiveCentersMultimodalDataset


class FiveCentersMultimodalDatasetV2(FiveCentersMultimodalDataset):
    """
    5中心多模态数据集加载器（Bio-COT 2.0版本）
    支持加载预计算的LLM嵌入
    """
    
    def __init__(
        self,
        csv_path: str,
        clinical_embed_path: Optional[str] = None,  # LLM嵌入文件路径
        transform=None,
        oct_num_frames: int = 20,
        max_col_images: int = 3,
        balance_negative_frames: bool = True,
        use_llm: bool = True,  # 是否使用LLM嵌入
    ):
        """
        Args:
            csv_path: CSV文件路径（labels.csv）
            clinical_embed_path: LLM嵌入文件路径（.pkl或.npy）
            transform: 图像变换
            oct_num_frames: 每个样本使用的OCT帧数
            max_col_images: 每个样本使用的Colposcopy图像数
            balance_negative_frames: 如果True，阴性病人使用与阳性病人相同的帧数
            use_llm: 是否使用LLM嵌入（True=使用预计算嵌入，False=使用原始临床数据）
        """
        # 调用父类初始化（加载图像和基础数据）
        super().__init__(
            csv_path=csv_path,
            transform=transform,
            oct_num_frames=oct_num_frames,
            max_col_images=max_col_images,
            balance_negative_frames=balance_negative_frames
        )
        
        self.use_llm = use_llm
        self.clinical_embed_path = clinical_embed_path
        
        # 加载LLM嵌入（如果使用）
        if use_llm:
            if clinical_embed_path is None:
                raise ValueError("use_llm=True时，必须提供clinical_embed_path")
            
            print(f"📥 正在加载LLM嵌入: {clinical_embed_path}")
            embed_path = Path(clinical_embed_path)
            
            if not embed_path.exists():
                raise FileNotFoundError(f"LLM嵌入文件不存在: {clinical_embed_path}")
            
            # 加载嵌入字典
            if embed_path.suffix == '.pkl':
                with open(embed_path, 'rb') as f:
                    self.clinical_embeddings = pickle.load(f)
            elif embed_path.suffix == '.npy':
                self.clinical_embeddings = np.load(embed_path, allow_pickle=True).item()
            else:
                raise ValueError(f"不支持的嵌入文件格式: {embed_path.suffix}")
            
            print(f"✅ 加载了 {len(self.clinical_embeddings)} 个LLM嵌入")
            
            # 检查嵌入维度
            sample_embed = next(iter(self.clinical_embeddings.values()))
            if isinstance(sample_embed, np.ndarray):
                self.llm_embed_dim = sample_embed.shape[0]
            else:
                self.llm_embed_dim = len(sample_embed)
            print(f"   LLM嵌入维度: {self.llm_embed_dim}")
        else:
            self.clinical_embeddings = None
            self.llm_embed_dim = None
            print(f"✅ 使用传统临床数据编码（MLP）")
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        Returns:
            dict: 包含图像、临床数据、标签等
        """
        # 调用父类方法获取基础数据
        sample = super().__getitem__(idx)
        
        # 获取oct_id用于匹配LLM嵌入
        oct_id = sample['oct_id']
        
        # 根据use_llm标志选择临床数据格式
        if self.use_llm:
            # LLM路径：返回预计算的嵌入
            if oct_id in self.clinical_embeddings:
                clinical_embedding = self.clinical_embeddings[oct_id]
                # 转换为tensor
                if isinstance(clinical_embedding, np.ndarray):
                    clinical_embedding = torch.from_numpy(clinical_embedding).float()
                else:
                    clinical_embedding = torch.tensor(clinical_embedding, dtype=torch.float32)
            else:
                # 如果找不到对应的嵌入，使用零向量（应该不会发生，但做防御性处理）
                print(f"⚠️ 警告: 样本 {oct_id} 没有找到LLM嵌入，使用零向量")
                clinical_embedding = torch.zeros(self.llm_embed_dim, dtype=torch.float32)
            
            # 添加LLM嵌入到返回字典
            sample['clinical_embedding'] = clinical_embedding
            # 保留原始clinical_data用于兼容性（如果需要）
            # sample['clinical_data'] = sample.get('clinical_data', None)
        else:
            # 传统路径：使用原始clinical_data（父类已处理）
            # 确保clinical_data存在
            if 'clinical_data' not in sample:
                # 从clinical_features重建
                clinical_features = sample['clinical_features']
                sample['clinical_data'] = {
                    'hpv': int(clinical_features[1].item()),
                    'tct': 'NILM',  # 默认值
                    'age': float(clinical_features[0].item() * 100)
                }
        
        return sample

