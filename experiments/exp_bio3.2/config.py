#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2 (Enhanced Logic Loop Version) 配置文件
融合3.1和4.0的优势：
1. 保留3.1的所有优点（显式对齐、自适应模态融合、增强Visual Notes）
2. 引入4.0的优势（Frozen VLM + Trainable Adapter、动态知识生成）
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class BioCOT_v3_2_Config:
    """Bio-COT 3.2 Enhanced 配置类"""
    
    # 数据路径
    data_root: str = '/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal'
    
    # ⚠️ VLM缓存路径（必需，从4.0引入）
    vlm_json_path: str = '../exp_bio4.0/data/vlm_profiles_v1.json'
    
    # 模型配置
    embed_dim: int = 768
    num_classes: int = 2
    num_centers: int = 5
    input_dim: int = 768
    llm_embed_dim: int = 768
    hidden_dim: int = 768  # Visual Notes隐藏层维度（3.1的优势）
    
    # Text Encoder配置（从4.0引入：Frozen Text Encoder）
    text_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    
    # Visual Notes配置（3.1的优势：增强版Cross-Attention）
    use_visual_notes: bool = True
    visual_threshold: float = 0.6
    background_suppress: float = 0.3
    warmup_epochs: int = 10
    
    # Bio-COT核心配置
    use_ot: bool = True
    use_dual: bool = True
    use_cross_attn: bool = True
    use_adaptive_gating: bool = True  # 3.1的优势：自适应模态门控
    
    # 损失权重（3.1的优势：显式对齐）
    lambda_cls: float = 2.0      # 分类损失权重
    lambda_ot: float = 0.5       # Optimal Transport损失权重
    lambda_align: float = 0.5    # 🔥 对齐损失权重（3.1的优势：显式对齐）
    lambda_consist: float = 0.2  # 一致性损失权重
    lambda_adv: float = 0.5      # 对抗损失权重
    lambda_sparse: float = 0.05  # 注意力稀疏损失权重
    sparse_lower_bound: float = 0.01
    
    # 类别权重（处理类别不平衡）
    focal_alpha: list = None     # 将在训练脚本中根据数据分布自动计算
    focal_gamma: float = 2.0     # Focal Loss的gamma参数
    
    # 决策阈值
    classification_threshold: float = 0.580  # 最优阈值
    
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
        """后处理：创建输出目录并验证VLM缓存路径"""
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
        
        # 验证VLM缓存路径
        vlm_path = Path(self.vlm_json_path)
        if not vlm_path.is_absolute():
            # 尝试相对路径
            possible_paths = [
                Path(__file__).parent / vlm_path,
                Path(__file__).parent.parent / vlm_path,
                vlm_path
            ]
            for p in possible_paths:
                if p.exists():
                    self.vlm_json_path = str(p.resolve())
                    print(f"✅ 找到VLM缓存文件: {self.vlm_json_path}")
                    return
            
            print(f"⚠️ 警告：VLM缓存文件未找到，请检查路径: {self.vlm_json_path}")

