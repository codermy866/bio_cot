#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 5.0 (Manifold Edition) 配置文件
核心创新：mHC融合（基于Sinkhorn的Birkhoff流形投影）
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class BioCOT_v5_Config:
    """Bio-COT 5.0 (mHC框架) 配置类"""
    
    # ==========================================================
    # 1. 基础实验配置
    # ==========================================================
    project_name: str = "Bio-COT_v5_Manifold"
    version: str = "v5.0"
    seed: int = 42
    
    # ==========================================================
    # 2. 数据路径配置
    # ==========================================================
    data_root: str = '/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal'
    
    # ⚠️ VLM缓存路径（必需）
    vlm_json_path: str = 'data/vlm_profiles_v1.json'
    
    # ==========================================================
    # 3. 模型配置
    # ==========================================================
    embed_dim: int = 768
    num_classes: int = 2
    num_centers: int = 5
    input_dim: int = 768  # ViT输出维度
    
    # Text Encoder配置（冻结的）
    text_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    
    # ==========================================================
    # 4. Bio-COT 5.0 核心模块配置 (mHC & Causal)
    # ==========================================================
    
    # 🔥 Stage 1: mHC Fusion (Manifold-Constrained Hyper-Connections)
    use_mhc: bool = True
    clinical_input_dim: int = 7  # ⚠️ 临床特征维度 (age/100, hpv, tct_onehot[5])
    sinkhorn_iters: int = 3       # mHC Sinkhorn迭代次数 (3-5次通常足够，支持梯度反向传播)
    mhc_epsilon: float = 0.05     # 熵正则化系数
    mhc_hidden_dim: int = 512     # mHC投影到的流形空间维度
    
    # Stage 2: Visual Notes & Causal Disentanglement
    use_visual_notes: bool = True
    visual_threshold: float = 0.6
    background_suppress: float = 0.3
    warmup_epochs: int = 10
    
    # Stage 3: Sinkhorn Optimal Transport (与Stage 1形成理论闭环)
    use_ot: bool = True  # LACT Loss (Language-Anchored Causal Transport)
    use_dual: bool = True
    use_cross_attn: bool = True
    
    # ==========================================================
    # 5. 损失权重（LACT框架 + mHC）
    # ==========================================================
    lambda_cls: float = 2.0      # 分类损失权重
    lambda_ot: float = 0.5       # LACT损失权重（视觉-文本对齐）
    lambda_consist: float = 0.2   # 一致性损失权重
    lambda_adv: float = 0.5      # 对抗损失权重
    lambda_sparse: float = 0.01   # 稀疏性损失权重
    lambda_ortho: float = 0.1    # 正交损失权重（强迫因果/噪声解耦）
    sparse_lower_bound: float = 0.01
    
    # 类别权重（处理类别不平衡）
    focal_alpha: list = None     # 将在训练脚本中根据数据分布自动计算
    focal_gamma: float = 2.0     # Focal Loss的gamma参数
    
    # 决策阈值
    classification_threshold: float = 0.580
    
    # ==========================================================
    # 6. 训练配置
    # ==========================================================
    batch_size: int = 48  # 建议适当减小，因为 mHC 会增加显存
    num_epochs: int = 100
    learning_rate: float = 0.0002
    weight_decay: float = 1e-5
    num_workers: int = 4
    pin_memory: bool = True
    
    # 图像配置
    oct_frames: int = 20
    colposcopy_images: int = 3
    
    # ==========================================================
    # 7. 输出目录
    # ==========================================================
    output_dir: str = 'results'
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    
    def __post_init__(self):
        """后处理：创建输出目录"""
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

