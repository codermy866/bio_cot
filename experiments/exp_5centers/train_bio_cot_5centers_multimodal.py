#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
兼容性数据集定义：FiveCentersMultimodalDataset

目的：
- 旧版 Bio-COT 3.0 的 `data/dataset_v3.py` 依赖：
    `from experiments.exp_5centers.train_bio_cot_5centers_multimodal import FiveCentersMultimodalDataset`
- 新版中，真正的实现已经迁移到：
    `experiments.exp_bio3.2.data.parent_dataset.train_bio_cot_5centers_multimodal`

为避免大规模修改旧代码，这里简单地转发类定义：
- 直接复用 exp_bio3.2 中已经验证过的
  `FiveCentersMultimodalDataset` 实现（基于 5centers_multi 数据）
"""

from pathlib import Path
import sys
import importlib.util

# 当前文件路径: .../experiments/exp_5centers/train_bio_cot_5centers_multimodal.py
# 工程根目录: .../VLM_Caus_Rm_Mics
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 将 exp_bio3.2 的 parent_dataset 文件路径
PARENT_DATASET_FILE = PROJECT_ROOT / "experiments" / "exp_bio3.2" / "data" / "parent_dataset" / "train_bio_cot_5centers_multimodal.py"

if not PARENT_DATASET_FILE.exists():
    raise ImportError(
        f"无法找到父数据集文件: {PARENT_DATASET_FILE}\n"
        f"请确保 exp_bio3.2/data/parent_dataset/train_bio_cot_5centers_multimodal.py 存在"
    )

# 使用 importlib 直接从文件路径导入，避免循环导入
try:
    spec = importlib.util.spec_from_file_location(
        "parent_train_bio_cot_5centers_multimodal",  # 使用不同的模块名避免冲突
        str(PARENT_DATASET_FILE)
    )
    parent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent_module)
    
    _FiveCentersMultimodalDatasetImpl = parent_module.FiveCentersMultimodalDataset
except Exception as e:
    raise ImportError(
        f"无法从 {PARENT_DATASET_FILE} 导入 FiveCentersMultimodalDataset: {e}\n"
        f"请检查文件是否存在且可读"
    ) from e


class FiveCentersMultimodalDataset(_FiveCentersMultimodalDatasetImpl):
    """
    5中心多模态数据集加载器（兼容旧接口）

    说明：
    - 完全继承自 exp_bio3.2 的实现
    - 主要用于兼容旧版 import 路径
    - 行为与新版 Bio-COT 3.2 使用的父类一致
    """

    pass


