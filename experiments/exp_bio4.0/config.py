#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 4.0 (LACT框架) 配置文件
引入 Frozen VLM + Trainable Adapter 机制
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class BioCOT_v4_Config:
    """Bio-COT 4.0 (LACT框架) 配置类"""
    
    # 数据路径
    data_root: str = '/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal'
    
    # ⚠️ VLM缓存路径（必需）
    vlm_json_path: str = 'data/vlm_profiles_v1.json'
    
    # 模型配置
    embed_dim: int = 768
    num_classes: int = 2
    num_centers: int = 5
    input_dim: int = 768
    
    # Text Encoder配置（冻结的）
    text_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    
    # Visual Notes配置
    use_visual_notes: bool = True
    visual_threshold: float = 0.6
    background_suppress: float = 0.3
    warmup_epochs: int = 10
    
    # Bio-COT核心配置
    use_ot: bool = True  # LACT Loss (Language-Anchored Causal Transport)
    use_dual: bool = True
    use_cross_attn: bool = True
    
    # 损失权重（LACT框架）
    lambda_cls: float = 2.0      # 分类损失权重
    lambda_ot: float = 0.5       # LACT损失权重（视觉-文本对齐）
    lambda_consist: float = 0.2   # 一致性损失权重
    lambda_adv: float = 0.5      # 对抗损失权重
    lambda_sparse: float = 0.01   # 稀疏性损失权重
    sparse_lower_bound: float = 0.01
    
    # 类别权重（处理类别不平衡）
    focal_alpha: list = None     # 将在训练脚本中根据数据分布自动计算
    focal_gamma: float = 2.0     # Focal Loss的gamma参数
    
    # 决策阈值
    classification_threshold: float = 0.580
    
    # 训练配置
    batch_size: int = 48
    num_epochs: int = 100
    learning_rate: float = 0.0002
    weight_decay: float = 1e-5
    num_workers: int = 4
    pin_memory: bool = True
    
    # 图像配置
    oct_frames: int = 20
    colposcopy_images: int = 3
    
    # 输出目录
    output_dir: str = 'results'
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    
    def __post_init__(self):
        """后处理：创建输出目录"""
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

