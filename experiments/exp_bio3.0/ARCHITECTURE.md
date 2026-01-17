# Bio-COT 3.0: 完整架构文档

## 📋 目录

1. [整体架构](#整体架构)
2. [核心模块](#核心模块)
3. [数学公式](#数学公式)
4. [实施细节](#实施细节)

---

## 一、整体架构

### 1.1 流程概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    Bio-COT 3.0 架构图                            │
└─────────────────────────────────────────────────────────────────┘

输入层
├── OCT图像: X_oct [B, F, C, H, W]
├── Colposcopy图像: X_colpo [B, K, C, H, W]
├── 临床数据: C [B, 7] (HPV, TCT, Age)
└── 医学知识库: KB (JSON格式)

    ↓
    
【模块1: 知识笔记生成 (Knowledge Notes)】
├── 知识检索: Retrieve(C, KB) → Top-K指南
├── 笔记生成: Note_text = LLM(Prompt(C, Top-K))
└── 语义锚点: z_sem = TextProjector(LLM(Note_text)) [B, 768]

    ↓
    
【模块2: 视觉笔记生成 (Visual Notes)】
├── 图像特征提取: F_img = ViT(X) [B, N, 768] (N=196 patches)
├── 跨模态注意力: A = Attention(F_img, z_sem) [B, N]
├── 掩码生成: M = Threshold(A, λ=0.6) [B, N]
└── 特征过滤: F_note = F_img ⊙ M + F_img ⊙ (1-M) ⊙ β [B, N, 768]

    ↓
    
【模块3: 因果解耦 (Causal Decoupling)】
├── 全局池化: F_global = Pool(F_note) [B, 768]
├── 双头编码: z_causal, z_noise = DualHead(F_global)
└── 输出: z_causal [B, 768], z_noise [B, 768]

    ↓
    
【模块4: 跨模态融合】
├── Cross-Attention: fused_feat = Attention(z_causal, z_sem)
└── 输出: fused_feat [B, 768]

    ↓
    
【模块5: 分类预测】
└── logits = Classifier(fused_feat) [B, 2]
```

---

## 二、核心模块

### 2.1 知识笔记生成模块

**文件**：`models/knowledge_notes.py`

**关键类**：
- `KnowledgeRetriever`：知识检索器
- `KnowledgeNotesGenerator`：笔记生成器（使用冻结LLM）
- `KnowledgeNotesModule`：整合模块

**流程**：
1. 根据临床数据检索Top-K条医学指南
2. 使用冻结的医学LLM生成诊断摘要
3. 将摘要编码为语义锚点 z_sem

### 2.2 视觉笔记生成模块

**文件**：`models/visual_notes.py`

**关键类**：
- `CrossModalAttention`：跨模态注意力计算
- `VisualNotesGenerator`：视觉笔记生成器
- `VisualNotesModule`：整合模块（支持Warm-up）

**流程**：
1. 计算图像Patch特征与语义锚点的注意力
2. 生成掩码（阈值或Top-K选择）
3. 应用掩码过滤特征（保留高响应，抑制背景）

### 2.3 Bio-COT v3主模型

**文件**：`models/bio_cot_v3.py`

**关键特性**：
- 整合Knowledge Notes和Visual Notes
- 支持Warm-up策略（训练初期不进行过滤）
- 新增稀疏性损失（鼓励注意力聚焦）

---

## 三、数学公式

### 3.1 知识笔记生成

**知识检索**：
$$P = \text{Retrieve}(\mathbf{C}, \mathcal{KB}, k=5)$$

**笔记生成**：
$$\text{Note}_{text} = \text{LLM}(\text{Prompt}(\mathbf{C}, P))$$

**语义锚点**：
$$\mathbf{z}_{sem} = \text{TextProjector}(\text{TextEncoder}(\text{Note}_{text})) \in \mathbb{R}^{B \times 768}$$

### 3.2 视觉笔记生成

**跨模态注意力**：
$$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{F}_{img} (\mathbf{z}_{sem} \mathbf{W}_Q)^T}{\sqrt{D}}\right) \in \mathbb{R}^{B \times N \times 1}$$

**掩码生成**：
$$\mathbf{M}_{visual} = \mathbb{I}(\mathbf{A} > \lambda) \quad \text{或} \quad \mathbf{M}_{visual} = \text{TopK}(\mathbf{A}, k)$$

**特征过滤**：
$$\mathbf{F}_{note} = \mathbf{F}_{img} \odot \mathbf{M}_{visual} + \mathbf{F}_{img} \odot (1 - \mathbf{M}_{visual}) \cdot \beta$$

其中 $\beta$ 是背景抑制系数（Warm-up策略动态调整）。

### 3.3 损失函数

**总损失**：
$$\mathcal{L}_{total} = \mathcal{L}_{cls} + \lambda_{ot} \mathcal{L}_{ot} + \lambda_{consist} \mathcal{L}_{consist} + \lambda_{adv} \mathcal{L}_{adv} + \lambda_{sparse} \mathcal{L}_{sparse}$$

**新增稀疏性损失**：
$$\mathcal{L}_{sparse} = \frac{1}{B} \sum_{i=1}^{B} |\mathbf{A}_i|_1$$

---

## 四、实施细节

### 4.1 Warm-up策略

**动态背景抑制系数**：
$$\beta(epoch) = \begin{cases}
1.0 & \text{if } epoch < 5 \\
1.0 - \frac{epoch - 5}{15} \cdot 0.9 & \text{if } 5 \leq epoch < 20 \\
0.1 & \text{if } epoch \geq 20
\end{cases}$$

**实现**：
```python
# 每个epoch开始时调用
model.set_epoch(epoch)
```

### 4.2 知识库格式

**JSON结构**：
```json
{
  "HPV16_POS_Age>30": "High-risk HPV type 16...",
  "TCT_HSIL": "HSIL cytology: Immediate colposcopy...",
  ...
}
```

**检索键构建**：
- 根据HPV状态、TCT结果、年龄组合构建查询键
- 支持多个键的组合查询

### 4.3 视觉掩码策略

**两种模式**：
1. **阈值模式**：`M = (A > 0.6)`
2. **Top-K模式**：`M = TopK(A, k=50)`（保留Top-50个Patch）

**推荐**：训练初期使用阈值模式，稳定后可以尝试Top-K模式。

---

## 五、与v2.0的对比

| 特性 | v2.0 | v3.0 |
|------|------|------|
| **语义增强** | 仅LLM嵌入 | ✅ Knowledge Notes（RAG） |
| **视觉处理** | 全图处理 | ✅ Visual Notes（掩码聚焦） |
| **知识来源** | 仅原始数据 | ✅ 外部知识库 |
| **特征聚焦** | 全局特征 | ✅ 病灶区域聚焦 |
| **损失函数** | 4项损失 | ✅ 5项损失（+稀疏性） |

---

**文档版本**：v1.0  
**最后更新**：2025-01-08

