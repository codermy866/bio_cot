#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study: w/o Dual Head
移除双头因果解耦模块
"""

from pathlib import Path
from dataclasses import dataclass
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BioCOT_v3_Config


@dataclass
class NoDualHeadConfig(BioCOT_v3_Config):
    """移除 Dual Head 的配置"""
    
    experiment_name: str = "w/o_dual_head"
    experiment_description: str = "Ablation: 移除双头因果解耦模块"
    
    # 禁用 Dual Head
    use_dual: bool = False
    lambda_consist: float = 0.0
    lambda_adv: float = 0.0
    
    # 训练配置（消融实验使用20个epoch）
    num_epochs: int = 20
    
    # 输出目录
    output_dir: str = 'ablation_studies/w/o_dual_head/results'
    checkpoint_dir: str = 'ablation_studies/w/o_dual_head/checkpoints'
    log_dir: str = 'ablation_studies/w/o_dual_head/logs'
    
    def __post_init__(self):
        # 强制设置epoch数为20（确保覆盖父类默认值）
        self.num_epochs = 20
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

