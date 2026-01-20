#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study: w/o OT Loss
移除 Optimal Transport 损失
"""

from pathlib import Path
from dataclasses import dataclass
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BioCOT_v3_Config


@dataclass
class NoOTLossConfig(BioCOT_v3_Config):
    """移除 OT Loss 的配置"""
    
    experiment_name: str = "w/o_ot_loss"
    experiment_description: str = "Ablation: 移除 Optimal Transport 损失"
    
    # 禁用 OT Loss
    use_ot: bool = False
    lambda_ot: float = 0.0
    
    # 输出目录
    output_dir: str = 'ablation_studies/w/o_ot_loss/results'
    checkpoint_dir: str = 'ablation_studies/w/o_ot_loss/checkpoints'
    log_dir: str = 'ablation_studies/w/o_ot_loss/logs'
    
    def __post_init__(self):
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

