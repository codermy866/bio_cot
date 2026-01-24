#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.0 配置文件
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class BioCOT_v3_Config:
    """Bio-COT 3.0 配置类"""
    
    # 数据路径
    data_root: str = '/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal'
    knowledge_base_path: str = 'knowledge_base/medical_guidelines.json'
    knowledge_embed_path: Optional[str] = 'data/knowledge_embeddings.pt'  # 预计算的Knowledge Note Embeddings（修复：使用.pt字典格式确保对齐）
    
    # 模型配置
    embed_dim: int = 768
    num_classes: int = 2
    num_centers: int = 5
    input_dim: int = 768  # ViT输出维度
    llm_embed_dim: int = 768  # LLM嵌入维度
    
    # Knowledge Notes配置
    use_knowledge_notes: bool = True
    knowledge_top_k: int = 5  # 检索top-k条指南
    llm_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    
    # Visual Notes配置
    use_visual_notes: bool = True
    visual_threshold: float = 0.6  # 注意力阈值λ
    background_suppress: float = 0.3  # 背景抑制系数β（从0.1提高到0.3，减少过度抑制）
    warmup_epochs: int = 10  # Warm-up轮数（从5增加到10，更充分的预热）
    
    # Bio-COT核心配置
    use_ot: bool = True  # Sinkhorn OT损失
    use_dual: bool = True  # Dual-Head结构
    use_cross_attn: bool = True  # Cross-Attention融合
    
    # 损失权重（🎯 AUC提升改进：优化损失函数权重平衡）
    lambda_cls: float = 1.0      # 保持分类损失权重
    lambda_ot: float = 0.8       # 从1.0降低到0.8，减少OT损失的影响
    lambda_consist: float = 0.3  # 从0.5降低到0.3，减少一致性损失的约束
    lambda_adv: float = 0.8      # 从1.0降低到0.8，减少对抗损失的约束
    lambda_sparse: float = 0.01  # 从0.02降低到0.01，进一步减少稀疏损失的抑制
    sparse_lower_bound: float = 0.01  # 稀疏损失下界（修复漏洞3：防止注意力坍塌）
    
    # 训练配置
    batch_size: int = 48  # 从32增加到48，进一步充分利用显存（显存使用率约20%，可以继续增加）
    num_epochs: int = 100  # 设置为100个epoch（用户要求）
    learning_rate: float = 0.00024  # batch_size增大后，可以保持学习率不变（或按比例调整）
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

