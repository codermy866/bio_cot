#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
兼容性模块：为旧代码提供 experiments.exp_5centers 命名空间

说明：
- 早期版本的代码（如 Bio-COT 3.0 的 dataset_v3）依赖：
    `from experiments.exp_5centers.train_bio_cot_5centers_multimodal import FiveCentersMultimodalDataset`
- 新版本中，5中心多模态数据集的实现已经迁移到了：
    `experiments.exp_bio3.2.data.parent_dataset.train_bio_cot_5centers_multimodal`

本模块通过简单转发类定义，保证旧代码可以继续工作，
而无需修改大量 import 语句。
"""

from .train_bio_cot_5centers_multimodal import FiveCentersMultimodalDataset  # noqa: F401


