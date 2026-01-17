# VLM-Enhanced Causal Bayesian Framework

## 项目概述

本项目实现了基于Vision-Language Model (VLM)增强的因果贝叶斯CLIP框架，用于医学多模态诊断。

## 目录结构

```
VLM_Caus_Rm_Mics/
├── src/                            # 核心源代码
│   ├── models/                     # 模型定义（因果CLIP、贝叶斯框架等）
│   ├── data/                       # 数据处理
│   ├── evaluation/                 # 评估工具
│   └── training/                   # 训练工具
├── experiments/                    # 实验脚本
│   ├── exp1_causal_bayesian_clip/  # 主实验训练脚本
│   └── baseline/                   # 基线实验
├── exp1_Causal_Bayesian_clip/      # 实验开发目录（代码、文档、可视化）
├── models/                         # 模型Backbone（SwinT, ViT, MedicalViT）
├── utils/                          # 工具函数
├── util/                           # 微调API
├── configs/                        # 配置文件
├── scripts/                        # 启动脚本
├── training/                       # 高级训练脚本
├── analysis/                       # 分析脚本
├── visualization/                 # 可视化脚本
├── docs/                           # 项目文档
├── figures/                        # 生成的图表
├── data/                           # 数据集软链接
│   ├── 5centers_multi -> [原数据集路径]
│   └── 5centers_multi_internal_external_final -> [原数据集路径]
├── cursor_file/                    # Cursor辅助文件（配置、文档等）
│   ├── visualization_config.py    # MICCAI可视化配置（字体、配色）
│   ├── project_config.py          # 项目配置文件（路径、虚拟环境）
│   └── README.md                  # Cursor文件说明
├── Trash/                          # 废弃文件（已重组）
└── README.md                       # 本文件
```

**详细结构说明**: 请参考 `PROJECT_STRUCTURE_REORGANIZED.md`

## 快速开始

### 1. 环境配置

```bash
# 激活虚拟环境（项目自带）
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

# 或使用项目配置
python -c "from cursor_file.project_config import VENV_PATH; print(f'虚拟环境路径: {VENV_PATH}')"
```

### 2. 数据集

数据集保留在原位置，通过软链接访问：
- `data/5centers_multi` -> 原始数据集
- `data/5centers_multi_internal_external` -> 划分后的数据集

### 3. 运行VLM增强实验

**方法1: 使用标准训练脚本**
```bash
cd experiments/exp1_causal_bayesian_clip
python train.py \
    --data_path ../../data/5centers_multi \
    --output_dir ../../results/exp1 \
    --batch_size 24 \
    --num_epochs 100
```

**方法2: 使用VLM增强训练**
```bash
python train_vlm.py \
    --data_path ../../data/5centers_multi \
    --vlm_model Qwen/Qwen2-VL-2B-Instruct \
    --batch_size 10 \
    --use_amp
```

**方法3: 使用实验开发目录**
详见: `exp1_Causal_Bayesian_clip/docs/VLM_ENHANCED_EXPERIMENT_PLAN.md`

## 相关文档

- **实验方案**: `exp1_Causal_Bayesian_clip/docs/VLM_ENHANCED_EXPERIMENT_PLAN.md`
- **快速开始**: `exp1_Causal_Bayesian_clip/docs/VLM_QUICK_START.md`
- **A6000配置**: `exp1_Causal_Bayesian_clip/docs/VLM_A6000_CONFIGURATION.md`
- **实施路线图**: `exp1_Causal_Bayesian_clip/docs/VLM_IMPLEMENTATION_ROADMAP.md`

## 迁移信息

- **迁移日期**: 1766548910.0148673
- **源目录**: /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713
- **目标目录**: /data2/hmy/VLM_Caus_Rm

## 项目重组说明

项目已完成结构化重组，运行逻辑清晰：

- ✅ **核心代码**: `src/` - 模型、数据处理、评估
- ✅ **实验脚本**: `experiments/` - 标准化训练脚本
- ✅ **工具函数**: `utils/` 和 `util/` - 工具和API
- ✅ **废弃文件**: `Trash/` - 已移动到垃圾文件夹

**详细说明**: 请参考 `REORGANIZATION_SUMMARY.md` 和 `PROJECT_STRUCTURE_REORGANIZED.md`

## MICCAI论文准备

### 可视化配置

所有可视化应使用统一的配色方案和字体配置：

```python
from cursor_file.visualization_config import setup_miccai_style, get_color_palette

# 设置MICCAI样式（统一字体、300 DPI等）
setup_miccai_style()

# 获取配色方案（8种颜色）
colors = get_color_palette(4)  # 获取4种颜色
```

**配色方案**: `#c7522a, #e5c185, #f0daa5, #fbf2c4, #b8cdab, #74a892, #008585, #004343`

### 项目配置

使用统一的项目配置管理路径：

```python
from cursor_file.project_config import DATA_DIR, VENV_PATH, get_config

# 使用配置路径
data_path = DATA_DIR / '5centers_multi'
config = get_config()  # 获取所有配置
```

## 注意事项

1. **数据集**: 使用软链接访问原数据集
2. **虚拟环境**: 位于 `/data2/hmy/VLM_Caus_Rm_Mics/my_retfound`
3. **Cursor文件**: 与核心逻辑无关的文件统一放在 `cursor_file/` 目录
4. **可视化**: 所有图表使用 `cursor_file/visualization_config.py` 中的配置
5. **Trash文件夹**: 可以随时删除，不影响项目核心功能

