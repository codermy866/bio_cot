# 项目重组计划 - MICCAI论文发表准备

## 📋 一、分析依据说明

### 我分析时使用的关键文件：

1. **核心文档**（理解项目整体）：
   - `DEEP_TECHNICAL_ANALYSIS.md` - 技术深度分析（1327行）
   - `TECHNICAL_WORK_SUMMARY.md` - 技术工作总结
   - `README.md` - 项目概述
   - `exp1_Causal_Bayesian_clip/README_exp1.md` - 实验详细说明

2. **模型架构文件**（理解技术实现）：
   - `exp1_Causal_Bayesian_clip/code/enhanced_causal_clip.py` - 核心模型
   - `src/models/causal_bayesian_clip_framework.py` - 因果贝叶斯CLIP框架
   - `models/cnn_multimodal_model.py` - CNN多模态模型
   - `models/vmamba_multimodal_model.py` - VMamba模型
   - `models/SwinT/` - Swin-T模型

3. **训练脚本**（理解工作流程）：
   - `exp1_Causal_Bayesian_clip/code/train_enhanced_causal_clip.py` - 主训练脚本
   - `training/train_enhanced_causal_clip.py` - 训练脚本
   - `src/train/train_causal_bayesian_clip.py` - 训练入口

4. **项目结构文档**：
   - `PROJECT_STRUCTURE.md` - 项目结构说明
   - `docs/` - 各种技术文档

---

## 🎯 二、当前项目结构问题

### 问题1：目录混乱
- 根目录文件过多（20+个Python脚本）
- 重复目录：`training/`, `exp1_Causal_Bayesian_clip/code/`, `src/train/`
- 文档分散：根目录、`docs/`、`exp1_Causal_Bayesian_clip/docs/`

### 问题2：命名不规范
- 脚本命名不一致（`train_*.py`, `*_training.py`）
- 结果目录散落（`vmamba_result/`, `cnn_result/`, `exp1_Causal_Bayesian_clip/results/`）

### 问题3：缺少标准结构
- 没有`setup.py`（无法pip安装）
- 没有统一的`requirements.txt`
- 缺少`tests/`目录
- 缺少清晰的`experiments/`组织

---

## 📁 三、推荐的项目结构（MICCAI/GitHub标准）

```
VLM_Caus_Rm/
├── README.md                    # 项目主README（英文，GitHub展示）
├── README_CN.md                 # 中文README
├── LICENSE                      # 开源许可证
├── requirements.txt             # 依赖包列表
├── setup.py                     # 安装脚本
├── .gitignore                   # Git忽略文件
│
├── docs/                        # 📚 文档目录
│   ├── paper/                   # 论文相关文档
│   │   ├── MICCAI2025/          # MICCAI论文
│   │   ├── technical_notes/     # 技术笔记
│   │   └── figures/             # 论文图表
│   ├── api/                     # API文档
│   └── tutorials/               # 教程文档
│
├── src/                         # 🔧 源代码（核心代码库）
│   ├── __init__.py
│   ├── models/                  # 模型定义
│   │   ├── __init__.py
│   │   ├── backbones/           # 基础编码器
│   │   │   ├── cnn_encoder.py
│   │   │   ├── vmamba_encoder.py
│   │   │   ├── swin_encoder.py
│   │   │   └── vit_encoder.py
│   │   ├── fusion/              # 融合模块
│   │   │   ├── cross_modal_fusion.py
│   │   │   └── causal_fusion.py
│   │   ├── causal/              # 因果推理模块
│   │   │   ├── causal_graph.py
│   │   │   ├── causal_intervention.py
│   │   │   └── causal_gnn.py
│   │   ├── uncertainty/        # 不确定性量化
│   │   │   ├── bayesian_encoder.py
│   │   │   └── uncertainty_decomposition.py
│   │   └── enhanced_causal_clip.py  # 主模型
│   │
│   ├── data/                    # 数据处理
│   │   ├── __init__.py
│   │   ├── dataset.py           # 数据集类
│   │   ├── transforms.py        # 数据增强
│   │   └── preprocessing.py    # 数据预处理
│   │
│   ├── training/                # 训练相关
│   │   ├── __init__.py
│   │   ├── trainer.py           # 训练器基类
│   │   ├── losses.py            # 损失函数
│   │   └── optimizers.py        # 优化器配置
│   │
│   ├── evaluation/              # 评估相关
│   │   ├── __init__.py
│   │   ├── metrics.py           # 评估指标
│   │   ├── dca.py               # 决策曲线分析
│   │   └── uncertainty_analysis.py
│   │
│   └── utils/                   # 工具函数
│       ├── __init__.py
│       ├── visualization.py    # 可视化工具
│       └── misc.py              # 其他工具
│
├── experiments/                # 🧪 实验代码（按论文组织）
│   ├── exp1_causal_bayesian_clip/  # 实验1：因果贝叶斯CLIP
│   │   ├── train.py             # 训练脚本
│   │   ├── config.yaml          # 配置文件
│   │   └── README.md            # 实验说明
│   │
│   ├── exp2_vlm_enhanced/      # 实验2：VLM增强（如果适用）
│   │   └── ...
│   │
│   └── baseline/                # 基线方法
│       ├── cnn_baseline.py
│       ├── vmamba_baseline.py
│       └── swin_baseline.py
│
├── scripts/                     # 📜 启动脚本
│   ├── train.sh                 # 训练启动脚本
│   ├── evaluate.sh             # 评估脚本
│   └── visualize.sh            # 可视化脚本
│
├── configs/                     # ⚙️ 配置文件
│   ├── default.yaml             # 默认配置
│   ├── exp1_config.yaml         # 实验1配置
│   └── model_configs/           # 模型配置
│
├── results/                     # 📊 实验结果（统一管理）
│   ├── exp1_causal_bayesian_clip/
│   │   ├── checkpoints/         # 模型检查点
│   │   ├── logs/                # 训练日志
│   │   └── metrics/             # 评估指标
│   └── baseline/
│
├── figures/                     # 📈 论文图表
│   ├── architecture/           # 架构图
│   ├── results/                 # 结果图
│   └── causal_graphs/           # 因果图
│
├── tests/                       # 🧪 测试代码
│   ├── test_models.py
│   ├── test_data.py
│   └── test_training.py
│
└── data/                        # 💾 数据（软链接或说明）
    ├── README.md                # 数据说明
    └── [软链接到实际数据]
```

---

## 🔄 四、重组步骤

### 阶段1：创建新结构
1. 创建新目录结构
2. 移动核心代码到`src/`
3. 整理实验代码到`experiments/`
4. 统一结果目录到`results/`

### 阶段2：清理和整合
1. 删除重复文件
2. 统一命名规范
3. 更新导入路径
4. 创建统一的配置文件

### 阶段3：文档和工具
1. 更新README
2. 创建setup.py
3. 整理requirements.txt
4. 创建.gitignore

### 阶段4：测试和验证
1. 测试导入路径
2. 验证训练脚本
3. 检查文档完整性

---

## 📝 五、文件映射表

### 当前 → 新结构

| 当前路径 | 新路径 | 说明 |
|---------|--------|------|
| `exp1_Causal_Bayesian_clip/code/enhanced_causal_clip.py` | `src/models/enhanced_causal_clip.py` | 核心模型 |
| `exp1_Causal_Bayesian_clip/code/train_enhanced_causal_clip.py` | `experiments/exp1_causal_bayesian_clip/train.py` | 训练脚本 |
| `src/models/causal_bayesian_clip_framework.py` | `src/models/causal/bayesian_clip.py` | 因果框架 |
| `models/cnn_multimodal_model.py` | `src/models/backbones/cnn_encoder.py` | CNN编码器 |
| `models/vmamba_multimodal_model.py` | `src/models/backbones/vmamba_encoder.py` | VMamba编码器 |
| `models/SwinT/` | `src/models/backbones/swin_encoder.py` | Swin-T编码器 |
| `utils/enhanced_multimodal_dataset.py` | `src/data/dataset.py` | 数据集 |
| `training/train_*.py` | `experiments/*/train.py` | 训练脚本 |
| `analysis/` | `src/evaluation/` | 评估代码 |
| `visualization/` | `src/utils/visualization.py` | 可视化 |
| `docs/` | `docs/` | 文档（整理） |
| `exp1_Causal_Bayesian_clip/results/` | `results/exp1_causal_bayesian_clip/` | 结果目录 |

---

## ✅ 六、执行计划

1. **立即执行**：创建重组脚本
2. **备份数据**：确保所有代码已备份
3. **逐步迁移**：按模块逐步迁移
4. **测试验证**：每个模块迁移后测试
5. **文档更新**：同步更新所有文档

---

**下一步**：我将创建自动重组脚本，帮您完成项目整理！

