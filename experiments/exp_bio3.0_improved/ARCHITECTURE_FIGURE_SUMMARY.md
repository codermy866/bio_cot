# Bio-COT 3.0 架构图总结

## ✅ 已生成的架构图

### 1. Figure 1: 简洁版架构图（适合SCI论文Overview）
**文件**: 
- `logs/architecture_figure1_20260114_085736.pdf` (48KB, PDF矢量图)
- `logs/architecture_figure1_20260114_085736.png` (711KB, PNG位图)

**特点**:
- ✅ 清晰的模块划分和颜色编码
- ✅ 完整的数据流和箭头标注
- ✅ 关键维度信息（[B, F, C, H, W]等）
- ✅ 损失函数说明
- ✅ 图例说明
- ✅ 适合作为SCI论文的Figure 1

**包含模块**:
1. **输入层** (浅蓝色)
   - OCT Images [B, F, C, H, W]
   - Colposcopy Images [B, N, C, H, W]
   - Clinical Data (HPV, TCT, Age)

2. **特征提取** (浅绿色)
   - ViT Encoder (OCT)
   - ViT Encoder (Colposcopy)

3. **Knowledge Notes模块** (浅橙色)
   - Medical Knowledge Base
   - RAG Retrieval
   - Knowledge Notes Embedding [768-dim]

4. **Visual Notes模块** (浅蓝色)
   - Cross-Modal Attention
   - Visual Notes (OCT) Attention Map
   - Visual Notes (Colposcopy) Attention Map

5. **Dual-Head编码器** (浅红色/浅灰色)
   - Causal Head Encoder → z_causal
   - Noise Head Encoder → z_noise

6. **特征融合** (浅紫色)
   - Cross-Attention Fusion
   - z_causal + z_sem

7. **Optimal Transport** (浅橙色)
   - Sinkhorn OT
   - Domain Alignment
   - Memory Bank (Counterfactual)

8. **分类器** (浅绿色)
   - Classifier (MLP)
   - Prediction P(y|x)

### 2. Figure 2: 详细版架构图（含数学公式）
**文件**:
- `logs/architecture_detailed_20260114_085736.pdf` (43KB, PDF矢量图)
- `logs/architecture_detailed_20260114_085736.png` (770KB, PNG位图)

**特点**:
- ✅ 包含数学公式
- ✅ 更详细的模块说明
- ✅ 数据维度完整标注
- ✅ 损失函数公式
- ✅ 适合作为补充材料或详细说明

---

## 📊 架构图设计细节

### 数据流路径

```
1. 输入层
   ├─ OCT Images → ViT Encoder → Visual Notes (OCT)
   ├─ Colposcopy Images → ViT Encoder → Visual Notes (Colposcopy)
   └─ Clinical Data → Medical KB → RAG → Knowledge Notes

2. Visual Notes生成
   ├─ Cross-Modal Attention (Knowledge Notes + Image Features)
   ├─ Attention Map生成
   └─ 特征过滤 (Masked Features)

3. Dual-Head编码
   ├─ Causal Head → z_causal (疾病相关特征)
   └─ Noise Head → z_noise (中心相关噪声)

4. 特征融合
   ├─ Cross-Attention Fusion (z_causal + z_sem)
   └─ z_fused

5. Optimal Transport
   ├─ Sinkhorn OT (域对齐)
   └─ Memory Bank (反事实干预)

6. 分类
   └─ Classifier → P(y|x)
```

### 关键公式标注

在详细版架构图中包含：
- **Cross-Modal Attention**: A = softmax(QK^T/√d)
- **Visual Notes Filtering**: F_note = F_img ⊙ (A + (1-A)·β)
- **Sinkhorn OT**: L_OT = ⟨P, C⟩ + λH(P)
- **Total Loss**: L = λ_cls·L_cls + λ_ot·L_ot + λ_sparse·L_sparse + λ_consist·L_consist + λ_adv·L_adv
- **Classification**: P(y|x) = softmax(MLP(z_fused))

---

## 🎯 论文使用建议

### Figure 1 (Overview) - 推荐使用
**文件**: `architecture_figure1_*.pdf`

**优点**:
- 清晰简洁，适合Overview
- 颜色编码清晰，易于理解
- 包含所有关键模块
- 数据流清晰

**在论文中的位置**:
- **Section 3.1 Overview** 或 **Section 2 Related Work** 之后
- 作为整体架构的概览图

### 详细架构图 - 补充材料
**文件**: `architecture_detailed_*.pdf`

**优点**:
- 包含数学公式
- 更详细的模块说明
- 适合深入理解

**在论文中的位置**:
- **补充材料** (Supplementary Material)
- 或 **Section 3.2-3.4** 的详细说明部分

---

## 📐 架构图关键信息

### 核心创新点（在图中标注）

1. **Knowledge Notes** (浅橙色)
   - RAG-based语义增强
   - 外部医学知识整合

2. **Visual Notes** (浅蓝色)
   - 知识引导的病灶定位
   - 跨模态注意力机制

3. **Dual-Head Encoder** (浅红色/浅灰色)
   - 因果/噪声特征解耦
   - 域不变性学习

4. **Optimal Transport** (浅橙色)
   - Sinkhorn OT域对齐
   - Memory Bank反事实干预

### 数据维度标注

所有关键模块都标注了数据维度：
- 输入: [B, F, C, H, W] 或 [B, N, C, H, W]
- 特征: [B, 196, 768] (Patch features)
- 嵌入: [B, 768] (Embeddings)
- 输出: [B, 2] (Classification logits)

---

## 🔧 生成脚本

### 使用方法
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0_improved
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate
python generate_architecture_figure.py
```

### 输出文件
- `logs/architecture_figure1_*.pdf` - 简洁版（Figure 1）
- `logs/architecture_detailed_*.pdf` - 详细版（含公式）

---

## 📝 架构图说明文字（可用于论文Caption）

### Figure 1 Caption (简洁版)

**Caption**: "Bio-COT 3.0 Architecture Overview. The framework consists of five main components: (1) Input Layer: multi-modal inputs including OCT images, Colposcopy images, and clinical data; (2) Knowledge Notes Module: RAG-based retrieval and generation of diagnostic summaries from medical knowledge base; (3) Visual Notes Module: cross-modal attention for lesion localization; (4) Dual-Head Encoder: causal and noise feature decoupling; (5) Feature Fusion and Classification: cross-attention fusion, optimal transport for domain alignment, and final classification. The color coding indicates different module types (see legend)."

### Figure 2 Caption (详细版)

**Caption**: "Detailed Bio-COT 3.0 Architecture with Mathematical Formulations. This figure provides a comprehensive view of the architecture including: (a) Knowledge Notes generation with RAG retrieval and LLM-based summary generation; (b) Visual Notes generation with cross-modal attention and soft masking; (c) Dual-Head encoder for causal/noise decoupling; (d) Cross-attention fusion and optimal transport; (e) Classification with loss functions. Key mathematical formulations are annotated on the corresponding modules."

---

## ✅ 质量检查

### 已完成的检查
- ✅ 所有模块都已包含
- ✅ 数据流清晰
- ✅ 颜色编码合理
- ✅ 维度标注完整
- ✅ 公式标注（详细版）
- ✅ PDF格式（矢量图，适合论文）
- ✅ 高分辨率（300 DPI）

### 建议改进（可选）
- 可以添加更多数学公式标注
- 可以添加模块间的交互细节
- 可以添加训练/推理流程的区别

---

**最后更新**: 2025-01-14  
**生成状态**: ✅ 完成  
**文件位置**: `logs/architecture_*.pdf`

