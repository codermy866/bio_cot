#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study: w/o Adaptive Modality Gating
移除自适应模态门控（使用固定权重融合）
注意：这需要在模型代码中禁用 AdaptiveModalityGating
"""

from pathlib import Path
from dataclasses import dataclass
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BioCOT_v3_Config


@dataclass
class NoAdaptiveGatingConfig(BioCOT_v3_Config):
    """移除 Adaptive Modality Gating 的配置"""
    
    experiment_name: str = "w/o_adaptive_gating"
    experiment_description: str = "Ablation: 移除自适应模态门控（使用固定权重 0.6/0.4）"
    
    # 标记：需要在模型代码中处理
    use_adaptive_gating: bool = False
    
    # 输出目录
    output_dir: str = 'ablation_studies/w/o_adaptive_gating/results'
    checkpoint_dir: str = 'ablation_studies/w/o_adaptive_gating/checkpoints'
    log_dir: str = 'ablation_studies/w/o_adaptive_gating/logs'
    
    def __post_init__(self):
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

