#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study: w/o Alignment Loss
移除语义-视觉对齐损失（L_align）
"""

from pathlib import Path
from dataclasses import dataclass
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BioCOT_v3_Config


@dataclass
class NoAlignmentLossConfig(BioCOT_v3_Config):
    """移除 Alignment Loss 的配置"""
    
    experiment_name: str = "w/o_alignment_loss"
    experiment_description: str = "Ablation: 移除语义-视觉对齐损失（L_align）"
    
    # 禁用 Alignment Loss
    lambda_align: float = 0.0
    
    # 训练配置（消融实验使用20个epoch）
    num_epochs: int = 20
    
    # 输出目录
    output_dir: str = 'ablation_studies/w/o_alignment_loss/results'
    checkpoint_dir: str = 'ablation_studies/w/o_alignment_loss/checkpoints'
    log_dir: str = 'ablation_studies/w/o_alignment_loss/logs'
    
    def __post_init__(self):
        # 强制设置epoch数为20（确保覆盖父类默认值）
        self.num_epochs = 20
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

