#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study: Baseline (Bio-COT 3.2)
移除所有高级模块，仅保留基础分类功能
"""

from pathlib import Path
from dataclasses import dataclass
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BioCOT_v3_2_Config


@dataclass
class BaselineConfig(BioCOT_v3_2_Config):
    """Baseline 配置：移除所有高级模块"""
    
    experiment_name: str = "baseline"
    experiment_description: str = "Baseline: 移除所有高级模块"
    
    num_epochs: int = 20
    
    use_visual_notes: bool = False
    use_ot: bool = False
    use_dual: bool = False
    use_cross_attn: bool = False
    use_adaptive_gating: bool = False
    
    lambda_align: float = 0.0
    lambda_ot: float = 0.0
    lambda_consist: float = 0.0
    lambda_adv: float = 0.0
    lambda_sparse: float = 0.0
    
    output_dir: str = 'ablation_studies/baseline/results'
    checkpoint_dir: str = 'ablation_studies/baseline/checkpoints'
    log_dir: str = 'ablation_studies/baseline/logs'
    
    def __post_init__(self):
        self.num_epochs = 20
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

