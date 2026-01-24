# Bio-COT 3.0 Improved 架构概览与详细说明

> **本文档提供Bio-COT 3.0 Improved的整体架构概览图说明和每个模块的详细内容提示，用于SCI论文撰写**

---

## 📊 架构图概览

### 整体架构流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Bio-COT 3.0 Improved Architecture                    │
│              Knowledge Notes Guided Causal Optimal Transport                 │
└─────────────────────────────────────────────────────────────────────────────┘

【输入层 Input Layer】
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ OCT Images   │  │Colposcopy    │  │ Clinical     │  │ Medical     │
│ [B,F,C,H,W]  │  │Images        │  │ Data         │  │ Knowledge   │
│ F=20 frames  │  │[B,N,C,H,W]   │  │(HPV,TCT,Age) │  │ Base (JSON) │
│              │  │N=3 images    │  │              │  │             │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │                 │
       └─────────────────┴─────────────────┴─────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────────┐
        │   Step 1: Feature Extraction (ViT)   │
        └──────────────────────────────────────┘
        │
        ├─ ViT(OCT) → F_oct [B,197,768] → Drop [CLS] → F_oct_patch [B,196,768]
        └─ ViT(Colpo) → F_colpo [B,197,768] → Drop [CLS] → F_colpo_patch [B,196,768]
                         │
                         ▼
        ┌──────────────────────────────────────┐
        │   Step 2: Knowledge Notes Generation │
        │   (Offline Preprocessing)             │
        └──────────────────────────────────────┘
        │
        ├─ Knowledge Retrieval: RAG({HPV,TCT,Age}) → Retrieved Guidelines
        ├─ Note Generation: LLM(Clinical Data + Guidelines) → Note Text
        └─ Semantic Anchor: z_sem = TextProjector(LLM_Embedding) [B,768]
                         │
                         ▼
        ┌──────────────────────────────────────┐
        │   Step 3: Visual Notes Filtering     │
        └──────────────────────────────────────┘
        │
        ├─ Cross-Modal Attention: A = Attention(F_patch, z_sem) [B,196,1]
        ├─ Soft Masking: M = Sigmoid(A), clamped to [0.05, 1.0]
        ├─ Background Suppression: F_note = F_patch ⊙ (M + (1-M)·β)
        └─ Global Pooling: F_pooled = GAP(F_note) [B,768]
                         │
                         ▼
        ┌──────────────────────────────────────┐
        │   Step 4: Multimodal Fusion         │
        └──────────────────────────────────────┘
        │
        └─ Weighted Fusion: F_fused = 0.6·F_oct_pooled + 0.4·F_colpo_pooled [B,768]
                         │
                         ▼
        ┌──────────────────────────────────────┐
        │   Step 5: Causal Decoupling          │
        │   (Dual-Head Encoder)                │
        └──────────────────────────────────────┘
        │
        ├─ Causal Head: z_causal = CausalHead(F_fused) [B,768]
        └─ Noise Head: z_noise = NoiseHead(F_fused) [B,768]
                         │
                         ▼
        ┌──────────────────────────────────────┐
        │   Step 6: Cross-Modal Fusion        │
        └──────────────────────────────────────┘
        │
        └─ Cross-Attention: F_final = CrossAttn(z_causal, z_sem) [B,768]
                         │
                         ▼
        ┌──────────────────────────────────────┐
        │   Step 7: Classification             │
        └──────────────────────────────────────┘
        │
        └─ Prediction: y = Classifier(F_final) [B,2]
                         │
                         ▼
                    ┌─────────┐
                    │ Output  │
                    │ P(y|x)  │
                    └─────────┘

【损失函数 Loss Functions】
┌─────────────────────────────────────────────────────────────────────────┐
│ L_total = λ_cls·L_cls + λ_ot·L_ot + λ_sparse·L_sparse +                │
│           λ_consist·L_consist + λ_adv·L_adv                            │
│                                                                         │
│ • L_cls: Focal Loss (α=0.25, γ=2.0)                                    │
│ • L_ot: Sinkhorn Optimal Transport Distance                             │
│ • L_sparse: Entropy Loss + L1 Loss (for attention sparsity)            │
│ • L_consist: Counterfactual Consistency Loss                           │
│ • L_adv: Adversarial Loss (for domain invariance)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 模块详细说明

### 模块1: 输入层 (Input Layer)

**功能**: 接收多模态输入数据

**输入**:
- **OCT图像**: `[B, F, C, H, W]`
  - `B`: Batch size
  - `F = 20`: 每个样本的OCT帧数
  - `C = 3`: RGB通道
  - `H = W = 224`: 图像尺寸
- **Colposcopy图像**: `[B, N, C, H, W]`
  - `N = 3`: 每个样本的Colposcopy图像数
- **临床数据**: `{HPV, TCT, Age}`
  - `HPV ∈ {0, 1}`: HPV状态
  - `TCT ∈ {NILM, ASCUS, LSIL, HSIL}`: 细胞学结果
  - `Age ∈ [0, 100]`: 年龄
- **医学知识库**: JSON格式的医学指南

**代码位置**: `data/dataset_v3.py`

---

### 模块2: 特征提取 (Feature Extraction)

**功能**: 使用Vision Transformer (ViT)提取图像Patch特征

**处理流程**:
1. **ViT编码**: 
   ```python
   F_oct = ViT(I_oct)  # [B*F, 197, 768]
   F_colpo = ViT(I_colpo)  # [B*N, 197, 768]
   ```
   - ViT输出包含197个tokens: 1个[CLS] token + 196个Patch tokens
   - 每个token维度: 768 (ViT-Base)

2. **丢弃[CLS] token** (关键修复):
   ```python
   F_oct_patch = F_oct[:, 1:, :]  # [B*F, 196, 768]
   F_colpo_patch = F_colpo[:, 1:, :]  # [B*N, 196, 768]
   ```
   - 原因: Visual Notes需要Patch特征，不需要[CLS] token

3. **多帧/多图平均**:
   ```python
   F_oct_patch = F_oct_patch.view(B, F, 196, 768).mean(dim=1)  # [B, 196, 768]
   F_colpo_patch = F_colpo_patch.view(B, N, 196, 768).mean(dim=1)  # [B, 196, 768]
   ```

**数学公式**:
$$\mathbf{F}_{oct} = \text{ViT}(\mathbf{I}_{oct}) \in \mathbb{R}^{B \times 197 \times 768}$$
$$\mathbf{F}_{oct}^{patch} = \mathbf{F}_{oct}[:, 1:, :] \in \mathbb{R}^{B \times 196 \times 768}$$

**代码位置**: `training/extract_vit_patches.py`

---

### 模块3: Knowledge Notes生成 (Knowledge Notes Generation)

**功能**: 结合外部医学知识库，生成语义锚点

**处理流程**:

#### 3.1 知识检索 (Knowledge Retrieval)
```python
retrieved_knowledge = KnowledgeRetriever.retrieve(
    clinical_data={HPV, TCT, Age},
    top_k=5
)
```

**检索策略**:
- HPV相关: 如果HPV=1，检索"HPV_HR_POS_Age>30"
- TCT相关: 根据TCT结果检索对应指南
- 年龄相关: 根据年龄范围检索
- 组合状态: 检索高风险组合

#### 3.2 诊断摘要生成 (Note Generation)
```python
prompt = f"""
Patient Profile: A {age}-year-old female patient.
HPV Status: {hpv_desc}.
Cytology Result: {tct_desc}.

Retrieved Medical Guidelines:
{knowledge_text}

Based on the patient's clinical information...
"""
```

#### 3.3 语义锚点生成 (Semantic Anchor)
```python
# 使用冻结的医学LLM（PubMedBERT）
with torch.no_grad():
    llm_embedding = LLM(prompt)  # [1, hidden_size]

# 投影到目标维度
z_sem = TextProjector(llm_embedding)  # [B, 768]
```

**数学公式**:
$$\mathbf{z}_{sem} = \text{TextProjector}(\text{LLM}(\text{Prompt}(\mathbf{d}_{clinical}, \mathcal{K}_{retrieved})))$$

其中:
- $\mathbf{d}_{clinical} = \{HPV, TCT, Age\}$: 临床数据
- $\mathcal{K}_{retrieved}$: 检索到的医学指南
- $\text{LLM}(\cdot)$: 冻结的医学LLM（PubMedBERT）

**离线预处理**: Knowledge Note Embeddings在训练前离线生成，保存为`.pt`字典格式

**代码位置**: `models/knowledge_notes.py`

---

### 模块4: Visual Notes过滤 (Visual Notes Filtering)

**功能**: 使用Knowledge Notes作为Query，对图像特征进行显式掩码，聚焦病灶区域

**处理流程**:

#### 4.1 跨模态注意力计算
```python
# 投影到同一空间
q = TextProj(z_sem).unsqueeze(1)  # [B, 1, hidden_dim] (Query)
k = ImgProj(F_patch)  # [B, N, hidden_dim] (Key)

# Dot Product Attention
attn_logits = matmul(k, q.transpose(1, 2)) / sqrt(hidden_dim)  # [B, N, 1]
attn_map = Sigmoid(attn_logits)  # [B, N, 1]
```

#### 4.2 下界保护（防止注意力坍塌）
```python
min_attn = 0.05  # 最小注意力值（5%）
attn_map = clamp(attn_map, min=min_attn, max=1.0)
```

#### 4.3 Soft Masking（背景抑制）
```python
# 核心公式: F_note = F * Mask + F * (1 - Mask) * beta
mask_weight = attn_map + (1 - attn_map) * beta  # [B, N, 1]
F_note = F_patch * mask_weight  # [B, N, D]
```

#### 4.4 全局池化
```python
F_pooled = F_note.mean(dim=1)  # [B, D] (Global Average Pooling)
```

**数学公式**:
$$\mathbf{A}_{ij} = \text{Sigmoid}\left(\frac{\mathbf{K}_i^T \mathbf{Q}}{\sqrt{d_h}}\right)$$
$$\mathbf{M}_{visual} = \text{Clamp}(\mathbf{A}, \min=0.05, \max=1.0)$$
$$\mathbf{F}_{note} = \mathbf{F}_{patch} \odot \left(\mathbf{M}_{visual} + (1 - \mathbf{M}_{visual}) \cdot \beta\right)$$

**动态Beta策略** (Warm-up):
$$\beta(t) = \begin{cases}
1.0 & \text{if } t < 10 \\
1.0 - 0.7 \cdot \frac{t - 10}{20} & \text{if } 10 \leq t < 30 \\
0.3 & \text{if } t \geq 30
\end{cases}$$

**代码位置**: `models/visual_notes.py`

---

### 模块5: 多模态特征融合 (Multimodal Fusion)

**功能**: 融合OCT和Colposcopy的过滤后特征

**处理流程**:
```python
F_fused = 0.6 * F_oct_pooled + 0.4 * F_colpo_pooled  # [B, 768]
```

**权重设计**: 
- OCT权重: 0.6 (更重要的模态)
- Colposcopy权重: 0.4

**代码位置**: `models/bio_cot_v3.py` (第279行)

---

### 模块6: 因果解耦 (Causal Decoupling)

**功能**: 将图像特征分解为因果特征和噪声特征

**处理流程**:

#### 6.1 Dual-Head编码器
```python
# 特征投影
proj_feat = FeatureProj(F_fused)  # [B, 768]

# 双头编码
z_causal = CausalHead(proj_feat)  # [B, 768] (用于分类)
z_noise = NoiseHead(proj_feat)    # [B, 768] (包含域信息)
```

**网络结构**:
- **Feature Projection**: Linear(768 → 1536) → LayerNorm → GELU → Dropout → Linear(1536 → 768)
- **Causal Head**: Linear(768 → 768) → LayerNorm → GELU → Dropout → Linear(768 → 768)
- **Noise Head**: Linear(768 → 768) → LayerNorm → GELU → Dropout → Linear(768 → 768)

**数学公式**:
$$\mathbf{z}_{causal} = \text{CausalHead}(\text{Proj}(\mathbf{F}_{fused}))$$
$$\mathbf{z}_{noise} = \text{NoiseHead}(\text{Proj}(\mathbf{F}_{fused}))$$

**设计思想**:
- **z_causal**: 包含与疾病相关的因果特征，应该与语义锚点对齐
- **z_noise**: 包含域特定信息（中心差异、设备差异等），应该被对抗训练抑制

**代码位置**: `src/models/bida/bio_cot_v2.py` (DualHeadImageEncoder类)

---

### 模块7: 跨模态融合 (Cross-Modal Fusion)

**功能**: 使用Cross-Attention融合因果特征和语义锚点

**处理流程**:
```python
# Cross-Attention: Query=Image, Key/Value=Text
attn_out, _ = CrossAttn(
    query=z_causal.unsqueeze(1),  # [B, 1, D]
    key=z_sem.unsqueeze(1),       # [B, 1, D]
    value=z_sem.unsqueeze(1)       # [B, 1, D]
)

# Residual + LayerNorm
x = LayerNorm1(z_causal.unsqueeze(1) + attn_out)

# Feed-Forward
ffn_out = FFN(x)
F_final = LayerNorm2(x + ffn_out)  # [B, 1, D] → [B, D]
```

**数学公式**:
$$\text{CrossAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}}\right)\mathbf{V}$$
$$\mathbf{F}_{final} = \text{LayerNorm}_2(\mathbf{x} + \text{FFN}(\mathbf{x}))$$

其中:
- $\mathbf{Q} = \mathbf{z}_{causal}$: Query（图像特征）
- $\mathbf{K} = \mathbf{V} = \mathbf{z}_{sem}$: Key/Value（语义锚点）

**代码位置**: `src/models/bida/bio_cot_v2.py` (CrossModalFusion类)

---

### 模块8: 分类预测 (Classification)

**功能**: 最终分类预测

**处理流程**:
```python
# 分类器网络
F_final → Linear(768 → 384) → LayerNorm → GELU → Dropout → Linear(384 → 2)

# 预测
y = Classifier(F_final)  # [B, 2]
P(y|x) = Softmax(y)  # [B, 2]
```

**决策阈值**: 0.580 (最优阈值，而非默认0.5)

**代码位置**: `models/bio_cot_v3.py` (第122-128行)

---

## 🔧 损失函数详细说明

### 总损失函数

$$\mathcal{L}_{total} = \lambda_{cls} \mathcal{L}_{cls} + \lambda_{ot} \mathcal{L}_{ot} + \lambda_{sparse} \mathcal{L}_{sparse} + \lambda_{consist} \mathcal{L}_{consist} + \lambda_{adv} \mathcal{L}_{adv}$$

**权重配置** (改进后):
- $\lambda_{cls} = 2.0$ ⬆️ (增加，让模型更关注分类任务)
- $\lambda_{ot} = 0.5$ ⬇️ (降低，减少约束)
- $\lambda_{sparse} = 0.01$ (稀疏性损失)
- $\lambda_{consist} = 0.2$ ⬇️ (降低)
- $\lambda_{adv} = 0.5$ ⬇️ (降低)

### 1. 分类损失 (Focal Loss)

$$\mathcal{L}_{cls} = -\frac{1}{B}\sum_{i=1}^{B} \alpha_{y_i} (1-p_{i,y_i})^{\gamma} \log(p_{i,y_i})$$

- $\alpha = 0.25$: 类别权重
- $\gamma = 2.0$: 聚焦参数

### 2. Sinkhorn最优传输损失

**代价矩阵**:
$$\mathbf{C}_{ij} = \|\mathbf{z}_{causal}^{(i)} - \mathbf{z}_{sem}^{(j)}\|_2^2$$

**熵正则化最优传输**:
$$\min_{\mathbf{P} \in \mathcal{U}(\mathbf{a}, \mathbf{b})} \langle \mathbf{P}, \mathbf{C} \rangle - \epsilon H(\mathbf{P})$$

**Sinkhorn迭代**:
$$\mathbf{u}^{(t+1)} = \frac{1}{B} \oslash (\mathbf{K} \mathbf{v}^{(t)} + \delta)$$
$$\mathbf{v}^{(t+1)} = \frac{1}{B} \oslash (\mathbf{K}^T \mathbf{u}^{(t+1)} + \delta)$$

**最优传输距离**:
$$\mathcal{L}_{ot} = \frac{1}{B} \sum_{i,j} P_{ij}^{*} C_{ij}$$

### 3. 稀疏性损失

$$\mathcal{L}_{sparse} = \begin{cases}
0.3 \cdot H(\mathbf{M}_{norm}) + 0.1 \cdot \|\mathbf{M}\|_1 & \text{if } \bar{M} < 0.01 \\
0.5 \cdot H(\mathbf{M}_{norm}) + 0.2 \cdot \|\mathbf{M}\|_1 & \text{otherwise}
\end{cases}$$

### 4. 反事实一致性损失

$$\mathcal{L}_{consist} = \frac{1}{B}\sum_{i=1}^{B} \|\mathbf{y}_{orig}^{(i)} - \mathbf{y}_{cf}^{(i)}\|_2^2$$

其中:
- $\mathbf{y}_{orig} = \text{Classifier}(\text{Fusion}(\mathbf{z}_{causal}, \mathbf{z}_{sem}))$
- $\mathbf{y}_{cf} = \text{Classifier}(\text{Fusion}(\mathbf{z}_{causal} + \mathbf{z}_{noise}^{cf}, \mathbf{z}_{sem}))$

### 5. 对抗损失

$$\mathcal{L}_{adv} = \max(0, H_{max} - H(\mathbf{p}_{center}))$$

其中:
- $H(\mathbf{p}_{center}) = -\sum_c p_c \log p_c$: 预测分布的熵
- $H_{max} = \log(\text{num\_centers})$: 最大熵

---

## 🎯 关键技术细节

### 1. 漏洞修复

#### 漏洞1: [CLS] Token处理
- **问题**: ViT输出包含[CLS] token，但Visual Notes需要Patch特征
- **修复**: 丢弃[CLS] token，只保留Patch tokens
- **位置**: `models/bio_cot_v3.py` (第152-193行)

#### 漏洞2: Knowledge Note对齐
- **问题**: Knowledge Note Embeddings需要与样本ID精确对齐
- **修复**: 使用字典格式（`.pt`文件）存储，键为`patient_id`
- **位置**: `data/dataset_v3.py` (第88-117行)

#### 漏洞3: 注意力坍塌
- **问题**: 稀疏性损失可能导致注意力完全为0
- **修复**: 
  1. 添加下界保护: `attn_map = clamp(attn_map, min=0.05, max=1.0)`
  2. 使用熵损失替代纯L1损失
  3. 如果注意力已经过度稀疏，降低损失权重
- **位置**: `models/visual_notes.py` (第72-76行)

### 2. 性能改进策略

#### 改进1: 损失权重调整
- **效果**: 准确率从53.57%提升到73.81% (+20.24%)

#### 改进2: 最优阈值发现
- **问题**: 默认阈值0.5导致假阳性率极高
- **解决方案**: 通过ROC曲线找到最优阈值0.580
- **效果**: 特异性从34.51%提升到76.11% (+41.59%)

#### 改进3: Beta策略优化
- **改进前**: 最终Beta=0.1 (强抑制)
- **改进后**: 最终Beta=0.3 (温和抑制)
- **效果**: 减少过度抑制，保留更多有用信息

---

## 📊 数据流维度变化

| 模块 | 输入维度 | 输出维度 | 说明 |
|------|---------|---------|------|
| ViT编码器 | [B, F, C, H, W] | [B, 196, 768] | 丢弃[CLS] token |
| Knowledge Notes | {HPV, TCT, Age} | [B, 768] | 离线预处理 |
| Visual Notes | [B, 196, 768] | [B, 768] | GAP池化 |
| 多模态融合 | [B, 768] × 2 | [B, 768] | 加权融合 |
| Dual-Head | [B, 768] | [B, 768] × 2 | 因果+噪声 |
| Cross-Attention | [B, 768] × 2 | [B, 768] | 融合 |
| 分类器 | [B, 768] | [B, 2] | 最终预测 |

---

## 📚 参考文献

1. **NoteMR**: Notes-guided MLLM Reasoning (CVPR 2025)
2. **Sinkhorn算法**: Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport. NIPS.
3. **Focal Loss**: Lin, T. Y., et al. (2017). Focal loss for dense object detection. ICCV.

---

**文档版本**: v1.0  
**最后更新**: 2025-01-13  
**作者**: Bio-COT 3.0 Improved Team

