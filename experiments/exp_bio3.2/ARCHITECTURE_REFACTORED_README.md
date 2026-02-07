# BioLCoT 重构架构说明

## 概述

根据您的要求，我们将原始的BioLCoT架构重新组织为三个主要模块：

1. **Module 1: Multimodal Feature Encoding & Evidence Accumulation** (A + B1合并)
2. **Module 2: Adaptive Reliability-Weighted Fusion & Hypothesis Verification** (B2 + B3合并)
3. **Module 3: Joint Diagnostic Inference & Manifold Regularization** (C1 + C2合并)

## 模块详细说明

### Module 1: Multimodal Feature Encoding & Evidence Accumulation

**合并内容：**
- **A部分（原始特征编码）：**
  - 视觉流：Vision Backbone (ViT/CNN) [Frozen]
  - 文本流：Text Encoder (BioBERT) [Frozen] + Text Adapter [Trainable]
  - 知识注入：Knowledge Retrieval (VLM Retriever) [Trainable]

- **B1部分（循环证据累积）：**
  - Visual Adapter [Trainable]
  - Memory Bank (M) [Trainable] - 可学习的槽矩阵，包含Read/Write Heads
  - Gate机制控制Memory Bank的更新

**输出：**
- Refined Visual Features (F_v)
- Text Prototypes (P_t)
- Knowledge Embeddings (K_e)

**功能：** 将多模态输入（图像、文本、知识）编码为统一特征表示，并通过循环记忆机制累积证据。

---

### Module 2: Adaptive Reliability-Weighted Fusion & Hypothesis Verification

**合并内容：**
- **B2部分（自适应可靠性加权融合）：**
  - Adaptive Gating [Trainable] - 基于F_v和Z_noise生成权重α
  - Weighted Sum - 融合P_t和F_v：F_cm = α·P_t + (1-α)·F_v
  - OT Loss (Sinkhorn Distance) - 最优传输损失

- **B3部分（风险引导假设验证）：**
  - Reasoning Unit [Trainable] - 生成初始隐藏状态h_0
  - Iterative Reasoning Loop [Trainable] - 迭代细化步骤 (T=1...N)

**输出：**
- Final Reasoning Feature (h_N)

**功能：** 根据可靠性自适应融合多模态特征，并通过迭代推理循环验证和细化假设。

---

### Module 3: Joint Diagnostic Inference & Manifold Regularization

**合并内容：**
- **C1部分（联合诊断推理）：**
  - Global Pool - 全局池化
  - Classifier [Trainable] - 生成诊断概率

- **C2部分（流形正则化分支）：**
  - Linear Projection [Trainable] - 将h_N投影到共享空间
  - Align Loss (Contrastive) - 对比学习损失，对齐h_N和K_e

**输出：**
- Diagnosis Probability - 诊断概率

**功能：** 基于推理特征进行诊断预测，同时通过流形正则化确保特征与知识嵌入对齐。

---

## 架构图文件

- **PNG格式：** `Bio_COT_3.2_Refactored_Architecture.png` (高分辨率，适合展示)
- **PDF格式：** `Bio_COT_3.2_Refactored_Architecture.pdf` (矢量格式，适合论文)

## 生成脚本

运行以下命令重新生成架构图：

```bash
python draw_architecture_refactored.py
```

## 关键设计特点

1. **模块化设计：** 三个清晰的模块，每个模块负责特定的功能阶段
2. **数据流清晰：** 从输入到输出的完整数据流路径
3. **损失函数标注：** OT Loss和Align Loss明确标注在架构图中
4. **冻结/可训练区分：** 使用不同颜色和标注区分冻结和可训练组件
5. **循环机制可视化：** Module 2中的迭代推理循环通过循环箭头清晰展示

## 与原始架构的对应关系

| 原始模块 | 重构后位置 |
|---------|-----------|
| A: Heterogeneous Multimodal Feature Encoding | Module 1 |
| B1: Recurrent Evidence Accumulation | Module 1 |
| B2: Adaptive Reliability-Weighted Fusion | Module 2 |
| B3: Risk-Guided Hypothesis Verification | Module 2 |
| C1: Joint Diagnostic Inference | Module 3 |
| C2: Manifold Regularization Branch | Module 3 |

