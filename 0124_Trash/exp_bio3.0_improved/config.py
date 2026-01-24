#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.0 Improved 配置文件
应用性能改进：增加分类损失权重、调整类别权重、使用最优阈值
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class BioCOT_v3_Config:
    """Bio-COT 3.0 Improved 配置类"""
    
    # 数据路径
    data_root: str = '/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal'
    knowledge_base_path: str = '../exp_bio3.0/knowledge_base/medical_guidelines.json'
    knowledge_embed_path: Optional[str] = '../exp_bio3.0/data/knowledge_embeddings.pt'
    
    # 模型配置
    embed_dim: int = 768
    num_classes: int = 2
    num_centers: int = 5
    input_dim: int = 768
    llm_embed_dim: int = 768
    
    # Knowledge Notes配置
    use_knowledge_notes: bool = True
    knowledge_top_k: int = 5
    llm_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    
    # Visual Notes配置
    use_visual_notes: bool = True
    visual_threshold: float = 0.6
    background_suppress: float = 0.3
    warmup_epochs: int = 10
    
    # Bio-COT核心配置
    use_ot: bool = True
    use_dual: bool = True
    use_cross_attn: bool = True
    
    # 损失权重（🎯 性能改进：增加分类损失权重）
    lambda_cls: float = 2.0      # 从1.0增加到2.0，让模型更关注分类任务
    lambda_ot: float = 0.5       # 从0.8降低到0.5，减少OT损失的影响
    lambda_consist: float = 0.2   # 从0.3降低到0.2，减少一致性损失的约束
    lambda_adv: float = 0.5      # 从0.8降低到0.5，减少对抗损失的约束
    lambda_sparse: float = 0.01   # 保持0.01
    sparse_lower_bound: float = 0.01
    
    # 类别权重（🎯 性能改进：处理类别不平衡）
    focal_alpha: list = None     # 将在训练脚本中根据数据分布自动计算
    focal_gamma: float = 2.0     # Focal Loss的gamma参数
    
    # 决策阈值（🎯 性能改进：使用最优阈值）
    classification_threshold: float = 0.580  # 从0.5调整到0.580（最优阈值）
    
    # 训练配置
    batch_size: int = 48
    num_epochs: int = 100
    learning_rate: float = 0.0002  # 稍微降低学习率，更稳定
    weight_decay: float = 1e-5
    num_workers: int = 4
    pin_memory: bool = True
    
    # 图像配置
    oct_frames: int = 20
    colposcopy_images: int = 3
    
    # 输出目录（指向新文件夹）
    output_dir: str = 'results'
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    
    def __post_init__(self):
        """后处理：创建输出目录"""
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
