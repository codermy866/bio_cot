#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
组合消融实验: only_ot_crossattn
只保留 OT Loss + Cross-Attention，其他模块全部关闭
测试分布匹配与跨模态融合的协同效应
"""

from pathlib import Path
from dataclasses import dataclass
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BioCOT_v3_Config


@dataclass
class OnlyOTCrossAttnConfig(BioCOT_v3_Config):
    """只保留 OT Loss + Cross-Attention 的配置"""
    
    experiment_name: str = "only_ot_crossattn"
    experiment_description: str = "组合消融: 只保留 OT Loss + Cross-Attention"
    
    # 训练配置（消融实验使用20个epoch）
    num_epochs: int = 20
    
    # 只保留这两个模块
    use_ot: bool = True
    use_cross_attn: bool = True
    lambda_ot: float = 0.5
    
    # 关闭其他模块
    use_visual_notes: bool = False
    use_dual: bool = False
    
    # 关闭相关损失
    lambda_align: float = 0.0
    lambda_consist: float = 0.0
    lambda_adv: float = 0.0
    lambda_sparse: float = 0.0
    
    # 输出目录
    output_dir: str = 'ablation_studies/only_ot_crossattn/results'
    checkpoint_dir: str = 'ablation_studies/only_ot_crossattn/checkpoints'
    log_dir: str = 'ablation_studies/only_ot_crossattn/logs'
    
    def __post_init__(self):
        # 强制设置epoch数为20（确保覆盖父类默认值）
        self.num_epochs = 20
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

