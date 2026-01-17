# 项目结构重组说明

## 📋 项目运行逻辑线

### 核心运行流程

```
1. 数据准备
   data/ (软链接到原始数据集)
   ↓
   src/data/ (数据处理和预处理)
   ↓
   utils/enhanced_multimodal_dataset.py (数据集类)

2. 模型定义
   models/ (Backbone: SwinT, ViT, MedicalViT)
   ↓
   src/models/backbones/ (编码器封装)
   ↓
   src/models/causal/ (因果推理模块)
   ↓
   src/models/fusion/ (多模态融合)
   ↓
   src/models/enhanced_causal_clip.py (完整模型)

3. 训练流程
   experiments/exp1_causal_bayesian_clip/train.py (主训练入口)
   ↓
   加载模型和数据
   ↓
   训练循环 (使用 src/training/)
   ↓
   评估 (src/evaluation/)
   ↓
   保存结果

4. 分析和可视化
   analysis/ (结果分析)
   ↓
   visualization/ (可视化生成)
   ↓
   figures/ (生成的图表)
```

## 📁 重组后的目录结构

```
VLM_Caus_Rm_Mics/
├── README.md                    # 项目主README
├── README_GITHUB.md            # GitHub README
├── LICENSE                     # 许可证
├── setup.py                    # 安装配置
├── requirements.txt            # 依赖列表
├── requirements_enhanced.txt   # 增强依赖
│
├── src/                        # 核心源代码
│   ├── models/                # 模型定义
│   │   ├── backbones/        # Backbone编码器
│   │   ├── causal/           # 因果推理模块
│   │   ├── fusion/           # 多模态融合
│   │   ├── uncertainty/      # 不确定性量化
│   │   └── enhanced_causal_clip.py  # 主模型
│   ├── data/                  # 数据处理
│   ├── training/              # 训练工具
│   ├── evaluation/           # 评估工具
│   ├── eval/                 # 评估工具（备用）
│   └── utils/                # 工具函数
│
├── experiments/               # 实验脚本
│   ├── exp1_causal_bayesian_clip/  # 主实验
│   │   ├── train.py          # 标准训练脚本
│   │   ├── train_vlm.py      # VLM增强训练
│   │   └── ...               # 其他训练变体
│   └── baseline/             # 基线实验
│       ├── train_cnn_baseline.py
│       ├── train_swin_baseline.py
│       └── train_vmamba_baseline.py
│
├── exp1_Causal_Bayesian_clip/  # 实验开发目录
│   ├── code/                 # 实验代码
│   ├── docs/                 # 实验文档
│   └── visualization/        # 可视化工具
│
├── models/                    # 模型Backbone
│   ├── SwinT/               # Swin Transformer
│   ├── ViT/                 # Vision Transformer
│   ├── MedicalViT/          # Medical ViT
│   └── ensemble_model.py    # 集成模型
│
├── utils/                     # 工具函数
│   ├── enhanced_multimodal_dataset.py  # 数据集
│   ├── advanced_clinical_metrics.py     # 临床指标
│   └── ...                   # 其他工具
│
├── util/                      # 微调API（保留）
│   └── finetune_api.py
│
├── configs/                   # 配置文件
│   ├── binary_clinical_mapping.json
│   └── model_configs/
│
├── scripts/                   # 启动脚本
│   ├── run_*.sh              # Shell启动脚本
│   ├── start_*.py            # Python启动脚本
│   └── ...                   # 其他脚本
│
├── training/                  # 训练脚本（高级）
│   ├── advanced_training_pipeline.py
│   └── knowledge_distillation_training.py
│
├── analysis/                  # 分析脚本
│   ├── run_lancet_analysis.py
│   └── ...                   # 其他分析脚本
│
├── visualization/             # 可视化脚本
│   ├── draw_detailed_innovation_flowcharts.py
│   └── advanced_visualizations.py
│
├── docs/                      # 项目文档
│   ├── paper/                # 论文相关文档
│   └── ...                   # 其他文档
│
├── figures/                   # 生成的图表
│   ├── architecture/         # 架构图
│   ├── causal_graphs/        # 因果图
│   └── results/              # 结果图
│
├── data/                      # 数据目录（软链接）
│   ├── 5centers_multi
│   └── 5centers_multi_internal_external_final
│
├── results/                   # 训练结果（.gitignore）
├── tests/                     # 测试文件
│
├── Trash/                     # 废弃文件（已移动）
│   ├── notes_file/           # 开发笔记
│   ├── paper1_hierarchical_multimodal/  # 独立实验
│   ├── lancet_primary_care/   # 独立实验
│   └── scripts/legacy/       # 旧脚本
│
└── my_retfound/              # 虚拟环境（.gitignore）
```

## 🔗 核心文件依赖关系

### 训练脚本依赖链

```
experiments/exp1_causal_bayesian_clip/train.py
  ├── src/models/enhanced_causal_clip.py
  │   ├── src/models/causal/bayesian_clip_framework.py
  │   ├── src/models/causal/adaptive_causal_graph.py
  │   ├── src/models/backbones/swin_encoder.py
  │   └── src/models/fusion/hierarchical_fusion.py
  ├── utils/enhanced_multimodal_dataset.py
  │   └── src/data/preprocessing.py
  ├── models/SwinT/swin_image_encoder.py
  └── utils/advanced_clinical_metrics.py
```

### 数据流

```
data/5centers_multi (原始数据)
  ↓
src/data/preprocessing.py (预处理)
  ↓
utils/enhanced_multimodal_dataset.py (数据集类)
  ↓
DataLoader (PyTorch)
  ↓
训练脚本
```

## ✅ 重组完成项

1. ✅ 移动废弃文件到 `Trash/`
   - notes_file/ (开发笔记)
   - paper1_hierarchical_multimodal/ (独立实验)
   - lancet_primary_care/ (独立实验)
   - scripts/legacy/ (旧脚本)
   - 重复的重组脚本

2. ✅ 保留核心结构
   - src/ (核心源代码)
   - experiments/ (实验脚本)
   - models/ (模型Backbone)
   - utils/ (工具函数)
   - configs/ (配置文件)

3. ✅ 清理说明
   - Python缓存文件 (__pycache__, *.pyc) - 已在.gitignore中
   - 日志文件 (*.log) - 已在.gitignore中

## 📝 使用说明

### 运行主实验

```bash
# 激活虚拟环境
source my_retfound/bin/activate

# 运行标准训练
python experiments/exp1_causal_bayesian_clip/train.py \
    --data_path data/5centers_multi \
    --output_dir results/exp1

# 运行VLM增强训练
python experiments/exp1_causal_bayesian_clip/train_vlm.py \
    --data_path data/5centers_multi \
    --vlm_model Qwen/Qwen2-VL-2B-Instruct
```

### 运行基线实验

```bash
# CNN基线
python experiments/baseline/train_cnn_baseline.py

# Swin基线
python experiments/baseline/train_swin_baseline.py

# VMamba基线
python experiments/baseline/train_vmamba_baseline.py
```

## 🎯 项目核心创新点

1. **可学习因果图发现** - `src/models/causal/adaptive_causal_graph.py`
2. **不确定性分解** - `src/models/uncertainty/`
3. **贝叶斯CLIP框架** - `src/models/causal/bayesian_clip_framework.py`
4. **VLM增强** - `exp1_Causal_Bayesian_clip/code/vlm_enhanced_causal_clip.py`

## ⚠️ 注意事项

1. **数据路径**: 数据通过软链接访问，确保原始数据路径正确
2. **虚拟环境**: 使用 `my_retfound` 虚拟环境
3. **依赖关系**: 确保所有核心模块在Python路径中
4. **Trash文件夹**: 包含废弃文件，可以随时删除

---
**重组日期**: 2025-01-XX
**重组说明**: 已将所有无运行逻辑关系的文件移动到Trash/文件夹，项目结构已清晰化

