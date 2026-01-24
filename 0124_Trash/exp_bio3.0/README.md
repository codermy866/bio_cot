# Bio-COT 3.0: 知识笔记引导的因果最优传输架构

## 📋 项目概述

Bio-COT 3.0 是 Bio-COT 2.0 的重大升级版本，结合 NoteMR (CVPR 2025) 的创新点，引入了 **"Note-Guided" (笔记引导)** 机制，通过知识笔记和视觉笔记实现更精准的医学图像诊断。

## 🎯 核心创新点

### v2.0 vs v3.0 对比

| 特性 | Bio-COT v2.0 | Bio-COT v3.0 |
|------|-------------|-------------|
| **语义增强** | 仅使用临床数据的隐式Embedding | ✅ 引入Knowledge Notes（医学知识笔记） |
| **视觉处理** | 对全图进行处理 | ✅ 引入Visual Notes（视觉笔记）机制 |
| **知识来源** | 仅原始临床数据 | ✅ 结合外部医学指南（RAG思想） |
| **特征聚焦** | 全局特征 | ✅ 显式掩码聚焦病灶区域 |

## 🏗️ 整体架构

```
输入层
├── OCT/Colposcopy图像
├── 临床数据 (HPV, TCT, Age)
└── 医学知识库 (Medical Knowledge Base)

    ↓
    
【Note生成模块 (NoteMR)】
├── 知识检索：从知识库检索相关指南
├── 笔记生成：使用冻结LLM生成诊断摘要
└── 语义锚点：z_sem = TextProjector(LLM(Note))

    ↓
    
【视觉笔记模块】
├── 图像特征提取：F_img = ViT(Images)
├── 跨模态注意力：A = Attention(F_img, z_sem)
├── 掩码生成：M_visual = Threshold(A, λ)
└── 特征过滤：F_note = F_img ⊙ M_visual

    ↓
    
【因果解耦模块 (Bio-COT核心)】
├── 双头编码器：z_causal, z_noise = DualHead(F_note)
├── Sinkhorn OT：L_ot = OT(z_causal, z_sem)
└── 分类预测：y = Classifier(z_causal)
```

## 📁 目录结构

```
exp_bio3.0/
├── README.md                          # 本文件
├── config.py                          # 配置文件
├── knowledge_base/
│   ├── medical_guidelines.json        # 医学知识库
│   └── build_knowledge_base.py        # 知识库构建脚本
├── models/
│   ├── bio_cot_v3.py                  # Bio-COT 3.0主模型
│   ├── knowledge_notes.py             # 知识笔记生成模块
│   ├── visual_notes.py                # 视觉笔记生成模块
│   └── __init__.py
├── data/
│   ├── dataset_v3.py                  # 数据集（支持知识库）
│   └── __init__.py
├── training/
│   ├── train_bio_cot_v3.py            # 训练脚本
│   └── train_utils.py                 # 训练工具函数
├── evaluation/
│   ├── evaluate_v3.py                 # 评估脚本
│   └── visualize_notes.py             # 可视化脚本
├── checkpoints/                       # 模型检查点
├── logs/                              # 训练日志
└── results/                           # 实验结果
    ├── knowledge_notes/                # 知识笔记可视化
    ├── visual_notes/                  # 视觉笔记可视化
    └── metrics/                       # 性能指标
```

## 🚀 快速开始

### 1. 环境准备

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics
source my_retfound/bin/activate
```

### 2. 构建医学知识库

```bash
cd experiments/exp_bio3.0
python knowledge_base/build_knowledge_base.py \
    --output knowledge_base/medical_guidelines.json
```

### 3. 训练模型

```bash
python training/train_bio_cot_v3.py \
    --data_root /data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal \
    --knowledge_base knowledge_base/medical_guidelines.json \
    --output_dir results \
    --batch_size 16 \
    --num_epochs 100
```

### 4. 评估模型

```bash
python evaluation/evaluate_v3.py \
    --checkpoint checkpoints/best_model.pth \
    --data_root /data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal \
    --output_dir results/metrics
```

## 📊 核心模块

### 1. 知识笔记生成 (Knowledge Notes)

- **输入**：临床数据 + 医学知识库
- **输出**：诊断摘要文本 → 语义锚点 z_sem
- **实现**：`models/knowledge_notes.py`

### 2. 视觉笔记生成 (Visual Notes)

- **输入**：图像特征 + 语义锚点
- **输出**：掩码 + 过滤后的特征 F_note
- **实现**：`models/visual_notes.py`

### 3. 因果解耦 (Causal Decoupling)

- **输入**：过滤后的特征 F_note
- **输出**：因果特征 z_causal + 噪声特征 z_noise
- **实现**：`models/bio_cot_v3.py`

## 📈 预期效果

| 指标 | v2.0基线 | v3.0预期 | 提升 |
|------|---------|---------|------|
| **AUC** | 0.84 | 0.87-0.89 | +3-5% |
| **准确率** | 78% | 81-83% | +3-5% |
| **F1-Score** | 0.66 | 0.69-0.71 | +3-5% |

## 🔬 消融实验

### 实验1: Knowledge Notes vs Raw Clinical Data
- **Baseline**: 仅使用原始临床数据
- **v3.0**: 使用Knowledge Notes
- **预期**: Knowledge Notes提升 +1-2% AUC

### 实验2: Visual Note Masking vs Global Attention
- **Baseline**: 全局Cross-Attention
- **v3.0**: Visual Note Masking
- **预期**: Visual Notes提升 +2-3% AUC

## 📝 参考文献

- NoteMR: Notes-guided MLLM Reasoning (CVPR 2025)
- Bio-COT 2.0: 基于LLM语义锚点与因果最优传输的多模态分类框架

## 📧 联系方式

如有问题，请查看日志文件：`logs/train_bio_cot_v3_*.log`

---

**版本**：v3.0  
**最后更新**：2025-01-08

