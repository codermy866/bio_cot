# Bio-COT 3.0 架构图绘制规范

## 📐 架构图详细规范

### 整体布局建议
- **宽度**: 建议 20-24cm (适合A4纸单栏或双栏)
- **高度**: 建议 14-16cm
- **方向**: 从左到右的数据流
- **风格**: 简洁、专业、符合SCI论文标准

---

## 🏗️ 模块详细规范

### 1. 输入层（左侧，垂直排列）

#### 1.1 OCT Images
- **位置**: 左上
- **尺寸**: 宽3cm × 高1.5cm
- **内容**: 
  - 标题: "OCT Images"
  - 维度: "[B, F, C, H, W]"
  - 图标: 可添加小图标（可选）
- **颜色**: 浅蓝色 (#E8F4F8)
- **边框**: 深灰色 (#333333), 2px

#### 1.2 Colposcopy Images
- **位置**: 左中
- **尺寸**: 宽3cm × 高1.5cm
- **内容**:
  - 标题: "Colposcopy Images"
  - 维度: "[B, N, C, H, W]"
- **颜色**: 浅蓝色 (#E8F4F8)
- **边框**: 深灰色 (#333333), 2px

#### 1.3 Clinical Data
- **位置**: 左下
- **尺寸**: 宽3cm × 高1.5cm
- **内容**:
  - 标题: "Clinical Data"
  - 内容: "HPV, TCT, Age"
- **颜色**: 浅蓝色 (#E8F4F8)
- **边框**: 深灰色 (#333333), 2px

---

### 2. 特征提取层（左侧偏中）

#### 2.1 ViT Encoder (OCT)
- **位置**: 左上偏右（对应OCT Images）
- **尺寸**: 宽3cm × 高1.5cm
- **内容**:
  - 标题: "ViT Encoder"
  - 子标题: "(OCT)"
  - 输出: "f_oct [B, 196, 768]"
- **颜色**: 浅绿色 (#F0F8E8)
- **边框**: 深灰色 (#333333), 2px

#### 2.2 ViT Encoder (Colposcopy)
- **位置**: 左中偏右（对应Colposcopy Images）
- **尺寸**: 宽3cm × 高1.5cm
- **内容**:
  - 标题: "ViT Encoder"
  - 子标题: "(Colposcopy)"
  - 输出: "f_colpo [B, 196, 768]"
- **颜色**: 浅绿色 (#F0F8E8)
- **边框**: 深灰色 (#333333), 2px

---

### 3. Knowledge Notes模块（中上）

#### 3.1 Medical Knowledge Base
- **位置**: 中上偏左
- **尺寸**: 宽2cm × 高1.5cm
- **内容**:
  - 标题: "Medical KB"
  - 或 "Knowledge Base"
- **颜色**: 浅橙色 (#FFF4E6)
- **边框**: 深灰色 (#333333), 2px

#### 3.2 RAG Retrieval
- **位置**: 中上中间
- **尺寸**: 宽2cm × 高1.5cm
- **内容**:
  - 标题: "RAG"
  - 子标题: "Retrieval"
- **颜色**: 浅橙色 (#FFF4E6)
- **边框**: 深灰色 (#333333), 2px

#### 3.3 Knowledge Notes Embedding
- **位置**: 中上偏右
- **尺寸**: 宽2.5cm × 高1.5cm
- **内容**:
  - 标题: "Knowledge Notes"
  - 输出: "z_sem [B, 768]"
- **颜色**: 浅橙色 (#FFF4E6)
- **边框**: 深灰色 (#333333), 2px

**连接关系**:
- Clinical Data → Medical KB
- Medical KB → RAG Retrieval
- RAG Retrieval → Knowledge Notes Embedding

---

### 4. Visual Notes模块（中上偏右）

#### 4.1 Cross-Modal Attention
- **位置**: 中上（Knowledge Notes下方）
- **尺寸**: 宽3cm × 高1.5cm
- **内容**:
  - 标题: "Cross-Modal"
  - 子标题: "Attention"
  - 公式: "A = softmax(QK^T/√d)"
- **颜色**: 浅蓝色 (#E6F3FF)
- **边框**: 深灰色 (#333333), 2px

#### 4.2 Visual Notes (OCT)
- **位置**: 右上（对应OCT）
- **尺寸**: 宽2.5cm × 高1.5cm
- **内容**:
  - 标题: "Visual Notes"
  - 子标题: "(OCT)"
  - 输出: "A_oct [B, 196, 1]"
- **颜色**: 浅蓝色 (#E6F3FF)
- **边框**: 深灰色 (#333333), 2px

#### 4.3 Visual Notes (Colposcopy)
- **位置**: 右中（对应Colposcopy）
- **尺寸**: 宽2.5cm × 高1.5cm
- **内容**:
  - 标题: "Visual Notes"
  - 子标题: "(Colposcopy)"
  - 输出: "A_colpo [B, 196, 1]"
- **颜色**: 浅蓝色 (#E6F3FF)
- **边框**: 深灰色 (#333333), 2px

**连接关系**:
- ViT (OCT) → Cross-Modal Attention
- ViT (Colposcopy) → Cross-Modal Attention
- Knowledge Notes → Cross-Modal Attention
- Cross-Modal Attention → Visual Notes (OCT)
- Cross-Modal Attention → Visual Notes (Colposcopy)

---

### 5. Dual-Head编码器（右侧中上）

#### 5.1 Causal Head
- **位置**: 右侧中上
- **尺寸**: 宽3cm × 高2cm
- **内容**:
  - 标题: "Causal Head"
  - 子标题: "Encoder"
  - 输出: "z_causal [B, 768]"
  - 说明: "Disease-related, Domain-invariant"
- **颜色**: 浅红色 (#FFE6E6)
- **边框**: 深灰色 (#333333), 2px

#### 5.2 Noise Head
- **位置**: 右侧中下（Causal Head下方）
- **尺寸**: 宽3cm × 高2cm
- **内容**:
  - 标题: "Noise Head"
  - 子标题: "Encoder"
  - 输出: "z_noise [B, 768]"
  - 说明: "Center-specific, Domain-variant"
- **颜色**: 浅灰色 (#E6E6E6)
- **边框**: 深灰色 (#333333), 2px

**连接关系**:
- Visual Notes (OCT) → Causal Head
- Visual Notes (Colposcopy) → Causal Head
- Visual Notes (OCT) → Noise Head
- Visual Notes (Colposcopy) → Noise Head

---

### 6. 特征融合（右侧中）

#### 6.1 Cross-Attention Fusion
- **位置**: 右侧中（Dual-Head下方）
- **尺寸**: 宽3.5cm × 高2cm
- **内容**:
  - 标题: "Cross-Attention"
  - 子标题: "Fusion"
  - 输入: "z_causal + z_sem"
  - 输出: "z_fused [B, 768]"
- **颜色**: 浅紫色 (#F5E6FF)
- **边框**: 深灰色 (#333333), 2px

**连接关系**:
- Causal Head → Cross-Attention Fusion
- Knowledge Notes → Cross-Attention Fusion

---

### 7. Optimal Transport（右侧中下）

#### 7.1 Sinkhorn OT
- **位置**: 右侧中下（Fusion下方）
- **尺寸**: 宽2.5cm × 高2cm
- **内容**:
  - 标题: "Sinkhorn"
  - 子标题: "OT"
  - 公式: "L_OT = ⟨P, C⟩ + λH(P)"
- **颜色**: 浅橙色 (#FFF0E6)
- **边框**: 深灰色 (#333333), 2px

#### 7.2 Memory Bank
- **位置**: 右侧下（OT下方）
- **尺寸**: 宽2.5cm × 高1.8cm
- **内容**:
  - 标题: "Memory Bank"
  - 子标题: "Counterfactual"
- **颜色**: 浅橙色 (#FFF0E6)
- **边框**: 深灰色 (#333333), 2px

**连接关系**:
- Cross-Attention Fusion → Sinkhorn OT
- Memory Bank ↔ Sinkhorn OT (双向)

---

### 8. 分类器（最右侧）

#### 8.1 Classifier
- **位置**: 最右侧中
- **尺寸**: 宽2.5cm × 高2cm
- **内容**:
  - 标题: "Classifier"
  - 结构: "MLP"
  - 输入: "z_fused"
- **颜色**: 浅绿色 (#E8F8E8)
- **边框**: 深灰色 (#333333), 2px

#### 8.2 Output
- **位置**: 最右侧下
- **尺寸**: 宽2.5cm × 高1.8cm
- **内容**:
  - 标题: "Prediction"
  - 输出: "P(y|x) [B, 2]"
- **颜色**: 浅绿色 (#E8F8E8)
- **边框**: 深灰色 (#333333), 2px

**连接关系**:
- Sinkhorn OT → Classifier
- Classifier → Output

---

## 🔗 连接关系（箭头）

### 主要数据流（从左到右）

1. **输入 → 特征提取**
   - OCT Images → ViT Encoder (OCT)
   - Colposcopy Images → ViT Encoder (Colposcopy)
   - Clinical Data → Medical KB

2. **特征提取 → Visual Notes**
   - ViT Encoder (OCT) → Cross-Modal Attention
   - ViT Encoder (Colposcopy) → Cross-Modal Attention

3. **Knowledge Notes → Visual Notes**
   - Knowledge Notes Embedding → Cross-Modal Attention

4. **Visual Notes → Dual-Head**
   - Visual Notes (OCT) → Causal Head
   - Visual Notes (OCT) → Noise Head
   - Visual Notes (Colposcopy) → Causal Head
   - Visual Notes (Colposcopy) → Noise Head

5. **Dual-Head → 融合**
   - Causal Head → Cross-Attention Fusion
   - Knowledge Notes → Cross-Attention Fusion

6. **融合 → OT → 分类**
   - Cross-Attention Fusion → Sinkhorn OT
   - Sinkhorn OT → Classifier
   - Classifier → Output

7. **Memory Bank ↔ OT** (双向)
   - Memory Bank ↔ Sinkhorn OT

---

## 🎨 颜色方案

| 模块类型 | 颜色代码 | RGB值 | 用途 |
|---------|---------|-------|------|
| 输入层 | #E8F4F8 | (232, 244, 248) | 原始数据输入 |
| 编码器 | #F0F8E8 | (240, 248, 232) | 特征提取 |
| Knowledge Notes | #FFF4E6 | (255, 244, 230) | 知识增强 |
| Visual Notes | #E6F3FF | (230, 243, 255) | 视觉注意力 |
| 因果特征 | #FFE6E6 | (255, 230, 230) | 疾病相关特征 |
| 噪声特征 | #E6E6E6 | (230, 230, 230) | 中心相关特征 |
| 融合 | #F5E6FF | (245, 230, 255) | 特征融合 |
| OT | #FFF0E6 | (255, 240, 230) | 最优传输 |
| 输出 | #E8F8E8 | (232, 248, 232) | 分类预测 |
| 边框 | #333333 | (51, 51, 51) | 所有模块边框 |
| 箭头 | #666666 | (102, 102, 102) | 数据流箭头 |

---

## 📏 布局坐标（参考）

### 使用坐标系 (0, 0) 到 (20, 14)

```
输入层 (x: 0.5-3.5)
├── OCT Images: (1.75, 12.25)
├── Colposcopy Images: (1.75, 9.75)
└── Clinical Data: (1.75, 7.25)

特征提取 (x: 4-7)
├── ViT (OCT): (5.25, 12.25)
└── ViT (Colposcopy): (5.25, 9.75)

Knowledge Notes (x: 4-9.5, y: 6.5-8)
├── Medical KB: (4.6, 7.25)
├── RAG: (5.9, 7.25)
└── Embedding: (8.5, 7.25)

Visual Notes (x: 7.5-13, y: 9-12.5)
├── Cross-Attention: (8.5, 9.75)
├── Visual Notes (OCT): (11.5, 12.25)
└── Visual Notes (Colposcopy): (11.5, 9.75)

Dual-Head (x: 13.5-16, y: 8.5-13)
├── Causal Head: (14.75, 12.5)
└── Noise Head: (14.75, 9.75)

融合 (x: 7.5-10.5, y: 3.5-5.5)
└── Cross-Attention Fusion: (9, 4.5)

OT (x: 11.5-14, y: 1-5.5)
├── Sinkhorn OT: (12.75, 4.5)
└── Memory Bank: (12.75, 1.9)

分类器 (x: 15-17.5, y: 0.5-5.5)
├── Classifier: (16.25, 4.5)
└── Output: (16.25, 1.9)
```

---

## 📝 文本内容

### 模块标题和说明

#### 输入层
- **OCT Images**: "OCT Images\n[B, F, C, H, W]"
- **Colposcopy Images**: "Colposcopy Images\n[B, N, C, H, W]"
- **Clinical Data**: "Clinical Data\n(HPV, TCT, Age)"

#### 特征提取
- **ViT Encoder (OCT)**: "ViT Encoder\n(OCT)\nf_oct [B, 196, 768]"
- **ViT Encoder (Colposcopy)**: "ViT Encoder\n(Colposcopy)\nf_colpo [B, 196, 768]"

#### Knowledge Notes
- **Medical KB**: "Medical\nKnowledge\nBase"
- **RAG Retrieval**: "RAG\nRetrieval"
- **Knowledge Notes Embedding**: "Knowledge Notes\nEmbedding\nz_sem [B, 768]"

#### Visual Notes
- **Cross-Modal Attention**: "Cross-Modal\nAttention\nA = softmax(QK^T/√d)"
- **Visual Notes (OCT)**: "Visual Notes\n(OCT)\nA_oct [B, 196, 1]"
- **Visual Notes (Colposcopy)**: "Visual Notes\n(Colposcopy)\nA_colpo [B, 196, 1]"

#### Dual-Head
- **Causal Head**: "Causal Head\nEncoder\nz_causal [B, 768]\n(Domain-invariant)"
- **Noise Head**: "Noise Head\nEncoder\nz_noise [B, 768]\n(Domain-variant)"

#### 融合
- **Cross-Attention Fusion**: "Cross-Attention\nFusion\nz_causal + z_sem\nz_fused [B, 768]"

#### OT
- **Sinkhorn OT**: "Sinkhorn\nOT\nL_OT = ⟨P, C⟩ + λH(P)"
- **Memory Bank**: "Memory Bank\n(Counterfactual)"

#### 分类器
- **Classifier**: "Classifier\nMLP"
- **Output**: "Prediction\nP(y|x) [B, 2]"

---

## 🔧 绘图工具建议

### 推荐工具

1. **Draw.io (diagrams.net)**
   - 免费、在线
   - 支持导出PDF/SVG
   - 适合绘制流程图

2. **TikZ (LaTeX)**
   - 专业、高质量
   - 适合学术论文
   - 需要LaTeX环境

3. **PowerPoint / Keynote**
   - 简单易用
   - 适合快速绘制
   - 导出PDF

4. **Figma / Adobe Illustrator**
   - 专业设计工具
   - 高质量输出
   - 适合精细调整

5. **Python + Graphviz / NetworkX**
   - 程序化生成
   - 可复现
   - 适合自动化

---

## 📋 绘图检查清单

### 必须包含的元素
- [ ] 所有输入模块（OCT, Colposcopy, Clinical Data）
- [ ] 所有处理模块（ViT, Knowledge Notes, Visual Notes, Dual-Head, Fusion, OT, Classifier）
- [ ] 所有数据流箭头
- [ ] 维度信息标注
- [ ] 颜色编码
- [ ] 图例说明
- [ ] 标题

### 可选但推荐的元素
- [ ] 数学公式标注
- [ ] 模块分组标签
- [ ] 损失函数说明
- [ ] 创新点标注

---

## 🎯 绘图优先级

### 高优先级（必须）
1. 模块位置和大小
2. 数据流箭头
3. 维度信息
4. 颜色编码

### 中优先级（推荐）
1. 数学公式
2. 模块说明
3. 图例

### 低优先级（可选）
1. 图标装饰
2. 背景网格
3. 阴影效果

---

**最后更新**: 2025-01-14

