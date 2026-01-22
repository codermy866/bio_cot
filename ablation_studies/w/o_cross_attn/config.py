#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study: w/o Cross-Attention (Bio-COT 3.2)
"""

from pathlib import Path
from dataclasses import dataclass
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BioCOT_v3_2_Config


@dataclass
class NoCrossAttnConfig(BioCOT_v3_2_Config):
    experiment_name: str = "w/o_cross_attn"
    experiment_description: str = "Ablation: 移除跨模态融合（Cross-Attention）"
    
    num_epochs: int = 20
    use_cross_attn: bool = False
    
    output_dir: str = 'ablation_studies/w/o_cross_attn/results'
    checkpoint_dir: str = 'ablation_studies/w/o_cross_attn/checkpoints'
    log_dir: str = 'ablation_studies/w/o_cross_attn/logs'
    
    def __post_init__(self):
        self.num_epochs = 20
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

