# VLM-Enhanced Causal Bayesian Framework - 项目结构

## 📁 目录结构

```
VLM_Caus_Rm/
├── README.md                          # 项目主README
├── PROJECT_STRUCTURE_VLM.md          # 本文件 - 项目结构说明
├── MIGRATION_REPORT.json              # 迁移报告
│
├── exp1_Causal_Bayesian_clip/        # ⭐ 核心实验目录
│   ├── README_exp1.md                # 实验说明
│   ├── requirements_enhanced.txt     # 依赖包
│   │
│   ├── code/                         # 核心代码
│   │   ├── vlm_enhanced_causal_clip.py      # VLM增强主模型
│   │   ├── enhanced_causal_clip.py          # 基础因果CLIP模型
│   │   ├── train_vlm_causal_clip.py         # VLM训练脚本 ⭐
│   │   ├── train_enhanced_causal_clip.py     # 基础训练脚本
│   │   ├── test_vlm_memory.py                # VLM显存测试 ⭐
│   │   ├── split_dataset_by_centers.py      # 数据集划分
│   │   ├── visualize_dataset_split.py        # 数据可视化
│   │   └── ... (其他工具脚本)
│   │
│   ├── docs/                         # 实验文档
│   │   ├── VLM_ENHANCED_EXPERIMENT_PLAN.md   # ⭐ VLM实验方案
│   │   ├── VLM_QUICK_START.md                # ⭐ 快速开始指南
│   │   ├── VLM_A6000_CONFIGURATION.md         # ⭐ A6000配置
│   │   ├── VLM_IMPLEMENTATION_ROADMAP.md     # 实施路线图
│   │   ├── VLM_GPU_MEMORY_ANALYSIS.md         # 显存分析
│   │   └── ... (其他文档)
│   │
│   └── visualization/                # 可视化结果
│
├── data/                             # 数据集（软链接）
│   ├── 5centers_multi -> [原数据集路径]
│   └── 5centers_multi_internal_external -> [原数据集路径]
│
├── training/                         # 训练相关脚本
├── models/                           # 模型定义
├── utils/                            # 工具函数
├── src/                              # 源代码
├── configs/                          # 配置文件
├── scripts/                          # 脚本集合
├── docs/                             # 通用文档
├── analysis/                         # 分析脚本
├── visualization/                    # 可视化工具
│
└── lancet_primary_care/              # Lancet论文相关
```

---

## 🎯 核心文件说明

### VLM实验相关（重点）

#### 1. 实验方案
- **`exp1_Causal_Bayesian_clip/docs/VLM_ENHANCED_EXPERIMENT_PLAN.md`**
  - 完整的VLM增强实验方案
  - 5个实验设计
  - 技术实现细节
  - 顶刊发表策略

#### 2. 快速开始
- **`exp1_Causal_Bayesian_clip/docs/VLM_QUICK_START.md`**
  - 三步快速开始指南
  - 推荐配置
  - 常见问题

#### 3. A6000配置
- **`exp1_Causal_Bayesian_clip/docs/VLM_A6000_CONFIGURATION.md`**
  - A6000 48GB优化配置
  - 显存使用分析
  - 推荐方案

#### 4. 训练脚本
- **`exp1_Causal_Bayesian_clip/code/train_vlm_causal_clip.py`**
  - VLM增强模型训练脚本
  - 支持Qwen-VL等VLM模型
  - A6000优化配置

#### 5. 显存测试
- **`exp1_Causal_Bayesian_clip/code/test_vlm_memory.py`**
  - 测试VLM模型显存使用
  - 不同batch size测试
  - 量化测试

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 进入新项目目录
cd /data2/hmy/VLM_Caus_Rm

# 使用原虚拟环境（或创建新的）
source /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/my_retfound/bin/activate

# 安装VLM相关依赖
pip install transformers>=4.40.0
pip install qwen-vl-utils
# 可选: pip install bitsandbytes  # 如果需要4bit量化
```

### 2. 测试显存

```bash
cd exp1_Causal_Bayesian_clip/code
python test_vlm_memory.py
```

### 3. 开始训练

```bash
python train_vlm_causal_clip.py \
    --data_path ../../data/5centers_multi \
    --output_dir ../results/vlm_causal_clip_results \
    --vlm_model Qwen/Qwen2-VL-2B-Instruct \
    --batch_size 10 \
    --num_epochs 100 \
    --learning_rate 1e-5 \
    --use_amp \
    --use_vlm_guidance \
    --vlm_trainable_layers 2
```

---

## 📊 数据集说明

数据集通过软链接访问，保留在原位置：

- **`data/5centers_multi`** → 原始数据集
- **`data/5centers_multi_internal_external`** → 划分后的数据集
  - 内部开发集：Enshi, Xiangyang, Shiyan
  - 外部测试集：Jingzhou, Wuda

**注意**: 数据集未迁移，通过软链接访问，确保原数据集路径可访问。

---

## 🔧 关键配置

### 推荐配置（A6000 48GB）

```python
VLM_CONFIG = {
    'model': 'Qwen/Qwen2-VL-2B-Instruct',
    'batch_size': 10,
    'precision': 'fp16',
    'freeze_vlm': False,
    'trainable_layers': 2,
    'learning_rate_vlm': 1e-5,
    'learning_rate_other': 3e-4,
}
```

### 路径配置

- **数据集路径**: `../../data/5centers_multi` (相对路径)
- **输出路径**: `../results/vlm_causal_clip_results` (相对路径)
- **虚拟环境**: 使用原路径或创建新的

---

## 📝 重要提示

1. **数据集路径**: 已更新为相对路径 `../../data/5centers_multi`
2. **虚拟环境**: 可以使用原虚拟环境，或创建新的
3. **结果目录**: 需要重新训练生成，旧结果未迁移
4. **路径引用**: 已自动更新部分路径，但请检查所有脚本

---

## 🔗 相关文档

- **实验方案**: `exp1_Causal_Bayesian_clip/docs/VLM_ENHANCED_EXPERIMENT_PLAN.md`
- **快速开始**: `exp1_Causal_Bayesian_clip/docs/VLM_QUICK_START.md`
- **A6000配置**: `exp1_Causal_Bayesian_clip/docs/VLM_A6000_CONFIGURATION.md`
- **实施路线图**: `exp1_Causal_Bayesian_clip/docs/VLM_IMPLEMENTATION_ROADMAP.md`
- **迁移报告**: `MIGRATION_REPORT.json`

---

**最后更新**: 2025-12-17

