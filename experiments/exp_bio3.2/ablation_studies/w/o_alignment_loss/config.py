#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study: w/o Alignment Loss (Bio-COT 3.2)
"""

from pathlib import Path
from dataclasses import dataclass
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BioCOT_v3_2_Config


@dataclass
class NoAlignmentLossConfig(BioCOT_v3_2_Config):
    experiment_name: str = "w/o_alignment_loss"
    experiment_description: str = "Ablation: 移除深度对齐损失"
    
    num_epochs: int = 20
    lambda_align: float = 0.0
    
    output_dir: str = 'ablation_studies/w/o_alignment_loss/results'
    checkpoint_dir: str = 'ablation_studies/w/o_alignment_loss/checkpoints'
    log_dir: str = 'ablation_studies/w/o_alignment_loss/logs'
    
    def __post_init__(self):
        self.num_epochs = 20
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

