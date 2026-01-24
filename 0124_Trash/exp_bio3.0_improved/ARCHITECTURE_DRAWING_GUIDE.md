# Bio-COT 3.0 架构图绘制指南

## 📋 提供给您的绘图资源

我已经为您准备了以下资源，方便您在其他工具中绘制：

### 1. 详细规范文档
**文件**: `ARCHITECTURE_DRAWING_SPEC.md`
- 包含所有模块的详细规范
- 位置、尺寸、颜色、内容
- 连接关系说明

### 2. Draw.io XML格式
**文件**: `ARCHITECTURE_DRAWIO_XML.txt`
- 可直接导入Draw.io
- 包含所有模块和连接
- 使用方法：
  1. 打开 https://app.diagrams.net/
  2. File -> Open -> 粘贴XML内容
  3. 调整样式
  4. 导出PDF

### 3. JSON数据结构
**文件**: `ARCHITECTURE_JSON_STRUCTURE.json`
- 完整的模块和连接数据
- 可用于程序化生成
- 包含所有样式信息

---

## 🎨 推荐绘图工具

### 1. Draw.io (diagrams.net) ⭐⭐⭐⭐⭐
**优点**:
- 免费、在线
- 支持导入XML
- 导出PDF/SVG/PNG
- 专业流程图工具

**使用方法**:
1. 打开 https://app.diagrams.net/
2. File -> Open -> 粘贴 `ARCHITECTURE_DRAWIO_XML.txt` 内容
3. 调整布局和样式
4. File -> Export as -> PDF/PNG

### 2. TikZ (LaTeX) ⭐⭐⭐⭐⭐
**优点**:
- 高质量输出
- 适合学术论文
- 可嵌入LaTeX文档

**示例代码**: 见下方

### 3. PowerPoint / Keynote ⭐⭐⭐⭐
**优点**:
- 简单易用
- 快速绘制
- 导出PDF

### 4. Figma / Adobe Illustrator ⭐⭐⭐⭐⭐
**优点**:
- 专业设计工具
- 精细调整
- 高质量输出

---

## 📐 模块布局建议

### 水平布局（从左到右）

```
[输入层] → [特征提取] → [Knowledge Notes] → [Visual Notes] → [Dual-Head] → [融合] → [OT] → [分类器] → [输出]
```

### 垂直分组

**左侧列（输入）**:
- OCT Images (上)
- Colposcopy Images (中)
- Clinical Data (下)

**中间列（处理）**:
- ViT Encoders (上中)
- Knowledge Notes (下)
- Visual Notes (右上)
- Cross-Attention (中)

**右侧列（编码和分类）**:
- Causal Head (上)
- Noise Head (中上)
- Fusion (中)
- OT (中下)
- Memory Bank (下)
- Classifier (右下)
- Output (最下)

---

## 🎨 颜色方案（RGB/HEX）

| 模块类型 | HEX | RGB | 用途 |
|---------|-----|-----|------|
| 输入层 | #E8F4F8 | (232, 244, 248) | 原始数据 |
| 编码器 | #F0F8E8 | (240, 248, 232) | 特征提取 |
| Knowledge Notes | #FFF4E6 | (255, 244, 230) | 知识增强 |
| Visual Notes | #E6F3FF | (230, 243, 255) | 视觉注意力 |
| 因果特征 | #FFE6E6 | (255, 230, 230) | 疾病相关 |
| 噪声特征 | #E6E6E6 | (230, 230, 230) | 中心相关 |
| 融合 | #F5E6FF | (245, 230, 255) | 特征融合 |
| OT | #FFF0E6 | (255, 240, 230) | 最优传输 |
| 输出 | #E8F8E8 | (232, 248, 232) | 分类预测 |

---

## 📝 模块文本内容

### 输入层
1. **OCT Images**
   - 标题: "OCT Images"
   - 维度: "[B, F, C, H, W]"

2. **Colposcopy Images**
   - 标题: "Colposcopy Images"
   - 维度: "[B, N, C, H, W]"

3. **Clinical Data**
   - 标题: "Clinical Data"
   - 内容: "(HPV, TCT, Age)"

### 特征提取
4. **ViT Encoder (OCT)**
   - 标题: "ViT Encoder"
   - 子标题: "(OCT)"
   - 输出: "f_oct [B, 196, 768]"

5. **ViT Encoder (Colposcopy)**
   - 标题: "ViT Encoder"
   - 子标题: "(Colposcopy)"
   - 输出: "f_colpo [B, 196, 768]"

### Knowledge Notes
6. **Medical Knowledge Base**
   - 标题: "Medical Knowledge Base"
   - 或分三行: "Medical\nKnowledge\nBase"

7. **RAG Retrieval**
   - 标题: "RAG Retrieval"

8. **Knowledge Notes Embedding**
   - 标题: "Knowledge Notes Embedding"
   - 输出: "z_sem [B, 768]"

### Visual Notes
9. **Cross-Modal Attention**
   - 标题: "Cross-Modal Attention"
   - 公式: "A = softmax(QK^T/√d)"

10. **Visual Notes (OCT)**
    - 标题: "Visual Notes"
    - 子标题: "(OCT)"
    - 输出: "A_oct [B, 196, 1]"

11. **Visual Notes (Colposcopy)**
    - 标题: "Visual Notes"
    - 子标题: "(Colposcopy)"
    - 输出: "A_colpo [B, 196, 1]"

### Dual-Head
12. **Causal Head**
    - 标题: "Causal Head Encoder"
    - 输出: "z_causal [B, 768]"
    - 说明: "Domain-invariant"

13. **Noise Head**
    - 标题: "Noise Head Encoder"
    - 输出: "z_noise [B, 768]"
    - 说明: "Domain-variant"

### 融合和OT
14. **Cross-Attention Fusion**
    - 标题: "Cross-Attention Fusion"
    - 输入: "z_causal + z_sem"
    - 输出: "z_fused [B, 768]"

15. **Sinkhorn OT**
    - 标题: "Sinkhorn OT"
    - 公式: "L_OT = ⟨P, C⟩ + λH(P)"

16. **Memory Bank**
    - 标题: "Memory Bank"
    - 说明: "(Counterfactual)"

### 分类器
17. **Classifier**
    - 标题: "Classifier"
    - 结构: "MLP"

18. **Output**
    - 标题: "Prediction"
    - 输出: "P(y|x) [B, 2]"

---

## 🔗 连接关系（箭头）

### 主要数据流（从左到右）

1. **输入 → 特征提取**
   - OCT Images → ViT Encoder (OCT)
   - Colposcopy Images → ViT Encoder (Colposcopy)
   - Clinical Data → Medical KB

2. **Knowledge Notes流程**
   - Medical KB → RAG Retrieval
   - RAG Retrieval → Knowledge Notes Embedding

3. **特征提取 → Visual Notes**
   - ViT (OCT) → Cross-Modal Attention
   - ViT (Colposcopy) → Cross-Modal Attention
   - Knowledge Notes Embedding → Cross-Modal Attention

4. **Visual Notes → Dual-Head**
   - Visual Notes (OCT) → Causal Head
   - Visual Notes (Colposcopy) → Noise Head

5. **Dual-Head → 融合**
   - Causal Head → Cross-Attention Fusion
   - Knowledge Notes Embedding → Cross-Attention Fusion

6. **融合 → OT → 分类**
   - Cross-Attention Fusion → Sinkhorn OT
   - Sinkhorn OT → Classifier
   - Classifier → Output

7. **Memory Bank ↔ OT** (双向)
   - Memory Bank ↔ Sinkhorn OT

---

## 📐 坐标参考（像素）

假设画布大小: 2000 × 1400 像素

### 输入层 (x: 50-200)
- OCT Images: (125, 240)
- Colposcopy Images: (125, 360)
- Clinical Data: (125, 480)

### 特征提取 (x: 250-400)
- ViT (OCT): (325, 240)
- ViT (Colposcopy): (325, 360)

### Knowledge Notes (x: 250-600, y: 440-520)
- Medical KB: (290, 480)
- RAG: (380, 480)
- Embedding: (525, 480)

### Visual Notes (x: 450-800, y: 200-400)
- Cross-Attention: (525, 360)
- Visual Notes (OCT): (725, 240)
- Visual Notes (Colposcopy): (725, 360)

### Dual-Head (x: 850-1030, y: 180-400)
- Causal Head: (940, 230)
- Noise Head: (940, 350)

### 融合和OT (x: 450-1050, y: 600-830)
- Fusion: (550, 650)
- OT: (775, 650)
- Memory Bank: (775, 790)
- Classifier: (975, 650)
- Output: (975, 790)

---

## 🎯 绘图建议

### 1. 使用专业工具
- **Draw.io**: 推荐，免费且专业
- **TikZ**: 适合LaTeX论文
- **Figma/Illustrator**: 适合精细设计

### 2. 保持简洁
- 避免过多装饰
- 重点突出数据流
- 颜色编码清晰

### 3. 标注完整
- 所有模块标注维度
- 关键公式标注
- 图例说明

### 4. 导出格式
- **PDF**: 矢量图，适合论文
- **PNG**: 位图，300 DPI
- **SVG**: 矢量图，可编辑

---

## 📝 论文Caption建议

**Figure 1 Caption**:
"Bio-COT 3.0 Architecture Overview. The framework processes multi-modal medical data through five key modules: (1) Knowledge Notes Generation: RAG-based retrieval and LLM-based generation of diagnostic summaries from medical knowledge base; (2) Visual Notes Generation: cross-modal attention for knowledge-guided lesion localization; (3) Dual-Head Encoder: causal and noise feature decoupling for domain-invariant learning; (4) Feature Fusion and Optimal Transport: cross-attention fusion and Sinkhorn OT for cross-center domain alignment; (5) Classification: final prediction with counterfactual consistency. Color coding indicates different module types (see legend)."

---

**最后更新**: 2025-01-14  
**所有资源文件已准备完成，可直接使用！**

