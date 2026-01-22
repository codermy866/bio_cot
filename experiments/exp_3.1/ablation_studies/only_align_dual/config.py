#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
组合消融实验: only_align_dual
只保留 Alignment Loss + Dual Head，其他模块全部关闭
测试对齐损失与因果解耦的协同效应
"""

from pathlib import Path
from dataclasses import dataclass
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BioCOT_v3_Config


@dataclass
class OnlyAlignDualConfig(BioCOT_v3_Config):
    """只保留 Alignment Loss + Dual Head 的配置"""
    
    experiment_name: str = "only_align_dual"
    experiment_description: str = "组合消融: 只保留 Alignment Loss + Dual Head"
    
    # 训练配置（消融实验使用20个epoch）
    num_epochs: int = 20
    
    # 只保留这两个模块
    use_dual: bool = True
    lambda_align: float = 0.5
    lambda_consist: float = 0.2
    lambda_adv: float = 0.5
    
    # 关闭其他模块
    use_visual_notes: bool = False
    use_ot: bool = False
    use_cross_attn: bool = False
    
    # 关闭相关损失
    lambda_ot: float = 0.0
    lambda_sparse: float = 0.0
    
    # 输出目录
    output_dir: str = 'ablation_studies/only_align_dual/results'
    checkpoint_dir: str = 'ablation_studies/only_align_dual/checkpoints'
    log_dir: str = 'ablation_studies/only_align_dual/logs'
    
    def __post_init__(self):
        # 强制设置epoch数为20（确保覆盖父类默认值）
        self.num_epochs = 20
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

