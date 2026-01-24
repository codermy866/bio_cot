#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study: Baseline
移除所有高级模块，仅保留基础分类功能
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BioCOT_v3_Config


@dataclass
class BaselineConfig(BioCOT_v3_Config):
    """Baseline 配置：移除所有高级模块"""
    
    # 实验标识
    experiment_name: str = "baseline"
    experiment_description: str = "Baseline: 移除所有高级模块"
    
    # 训练配置（消融实验使用20个epoch） - 必须在最前面，确保覆盖父类默认值
    num_epochs: int = 20
    
    # 禁用所有高级模块
    use_visual_notes: bool = False
    use_ot: bool = False
    use_dual: bool = False
    use_cross_attn: bool = False
    
    # 损失权重（禁用相关损失）
    lambda_align: float = 0.0
    lambda_ot: float = 0.0
    lambda_consist: float = 0.0
    lambda_adv: float = 0.0
    lambda_sparse: float = 0.0
    
    # 输出目录（独立）
    output_dir: str = 'ablation_studies/baseline/results'
    checkpoint_dir: str = 'ablation_studies/baseline/checkpoints'
    log_dir: str = 'ablation_studies/baseline/logs'
    
    def __post_init__(self):
        """后处理：创建输出目录并确保epoch数为20"""
        # 强制设置epoch数为20（确保覆盖父类默认值）
        self.num_epochs = 20
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

