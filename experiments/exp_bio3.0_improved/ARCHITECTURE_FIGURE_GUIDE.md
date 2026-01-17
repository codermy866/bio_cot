# Bio-COT 3.0 架构图生成指南

## 📊 生成的架构图

### Figure 1: 简洁版架构图（适合SCI论文Overview）
**文件**: `architecture_figure1_*.pdf`

**特点**:
- 清晰的模块划分
- 完整的数据流
- 标注了关键维度
- 包含损失函数说明
- 适合作为论文的Figure 1

**包含内容**:
1. **输入层**: OCT图像、Colposcopy图像、临床数据
2. **特征提取**: ViT编码器
3. **Knowledge Notes**: 医学知识库、RAG检索、知识笔记嵌入
4. **Visual Notes**: 跨模态注意力、视觉笔记生成
5. **Dual-Head编码器**: 因果特征和噪声特征
6. **特征融合**: Cross-Attention融合
7. **Optimal Transport**: Sinkhorn OT、Memory Bank
8. **分类器**: 最终预测

### Figure 2: 详细版架构图（含数学公式）
**文件**: `architecture_detailed_*.pdf`

**特点**:
- 包含数学公式
- 更详细的模块说明
- 数据维度标注
- 适合作为补充材料

---

## 🎨 架构图设计说明

### 颜色方案
- **输入层**: 浅蓝色 (#E8F4F8)
- **Knowledge Notes**: 浅橙色 (#FFF4E6)
- **Visual Notes**: 浅蓝色 (#E6F3FF)
- **编码器**: 浅绿色 (#F0F8E8)
- **因果特征**: 浅红色 (#FFE6E6)
- **噪声特征**: 浅灰色 (#E6E6E6)
- **融合**: 浅紫色 (#F5E6FF)
- **OT**: 浅橙色 (#FFF0E6)
- **输出**: 浅绿色 (#E8F8E8)

### 数据流标注
- 所有模块都标注了输入/输出维度
- 箭头表示数据流向
- 关键公式标注在相应模块

---

## 📐 架构图关键元素

### 1. 输入层
```
OCT Images: [B, F, C, H, W]
Colposcopy Images: [B, N, C, H, W]
Clinical Data: (HPV, TCT, Age)
```

### 2. Knowledge Notes模块
```
Medical KB → RAG Retrieval → Knowledge Notes Embedding [B, 768]
```

### 3. Visual Notes模块
```
ViT Features → Cross-Modal Attention → Visual Notes (Attention Map)
```

### 4. Dual-Head编码器
```
Filtered Features → Causal Head → z_causal [B, 768]
                 → Noise Head → z_noise [B, 768]
```

### 5. 特征融合
```
z_causal + z_sem → Cross-Attention Fusion → z_fused [B, 768]
```

### 6. Optimal Transport
```
Sinkhorn OT: L_OT = ⟨P, C⟩ + λH(P)
Memory Bank: Counterfactual Intervention
```

### 7. 分类器
```
z_fused → MLP → P(y|x) [B, 2]
```

---

## 🔧 使用方法

### 生成架构图
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0_improved
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate
python generate_architecture_figure.py
```

### 输出文件
- `logs/architecture_figure1_*.pdf` - 简洁版（适合Figure 1）
- `logs/architecture_figure1_*.png` - PNG版本
- `logs/architecture_detailed_*.pdf` - 详细版（含公式）
- `logs/architecture_detailed_*.png` - PNG版本

---

## 📝 论文使用建议

### Figure 1 (Overview)
- **使用**: `architecture_figure1_*.pdf`
- **位置**: 论文开头，Method部分之前
- **说明**: 提供整体架构概览

### 详细架构图
- **使用**: `architecture_detailed_*.pdf`
- **位置**: Method部分，或补充材料
- **说明**: 提供详细的数学公式和模块说明

---

## 🎯 架构图关键信息

### 核心创新点
1. **Knowledge Notes**: RAG-based语义增强
2. **Visual Notes**: 知识引导的病灶定位
3. **Dual-Head**: 因果/噪声特征解耦
4. **Optimal Transport**: 跨中心域对齐
5. **Memory Bank**: 反事实干预

### 数据流
```
Input → ViT → Visual Notes → Dual-Head → Fusion → OT → Classifier → Output
         ↓
    Knowledge Notes ──────────────┘
```

### 损失函数
```
L_total = λ_cls·L_cls + λ_ot·L_ot + λ_sparse·L_sparse + λ_consist·L_consist + λ_adv·L_adv
```

---

**最后更新**: 2025-01-14

