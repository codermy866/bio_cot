# Bio-COT 2.0: 数学公式与方法架构（SCI论文版）

## 📋 目录
1. [整体架构](#整体架构)
2. [数学公式](#数学公式)
3. [方法流程图](#方法流程图)
4. [模块详解](#模块详解)

---

## 一、整体架构

### 1.1 问题定义

**输入数据**：
- OCT图像序列：$\mathbf{X}_{oct} \in \mathbb{R}^{B \times F \times C \times H \times W}$，其中 $F$ 为帧数（48-120帧）
- Colposcopy图像：$\mathbf{X}_{colpo} \in \mathbb{R}^{B \times K \times C \times H \times W}$，其中 $K \leq 3$
- 临床数据：$\mathbf{C} \in \mathbb{R}^{B \times 7}$（HPV状态、TCT结果、年龄等）

**目标**：学习映射函数 $f: (\mathbf{X}_{oct}, \mathbf{X}_{colpo}, \mathbf{C}) \rightarrow \hat{y} \in \{0, 1\}$，实现二分类（正常 vs 异常）

### 1.2 核心思想

Bio-COT 2.0通过以下机制实现跨中心域不变性：
1. **LLM语义锚点**：使用医学LLM提取临床数据的语义嵌入，作为域不变的生物流形
2. **因果特征解耦**：将图像特征分解为因果特征（疾病相关）和噪声特征（中心相关）
3. **最优传输对齐**：使用Sinkhorn算法对齐因果特征和语义锚点
4. **反事实一致性**：通过Memory Bank机制实现反事实干预，确保模型对噪声不敏感

---

## 二、数学公式

### 2.1 图像特征提取

**ViT编码器**：
$$\mathbf{F}_{oct} = \text{ViT}(\mathbf{X}_{oct}) \in \mathbb{R}^{B \times 768}$$
$$\mathbf{F}_{colpo} = \text{ViT}(\mathbf{X}_{colpo}) \in \mathbb{R}^{B \times 768}$$

**多模态图像融合**（加权平均）：
$$\mathbf{F}_{img} = \alpha \cdot \mathbf{F}_{oct} + (1-\alpha) \cdot \mathbf{F}_{colpo}, \quad \alpha = 0.6$$

### 2.2 双头图像编码器（Dual-Head Image Encoder）

**特征投影**：
$$\mathbf{F}_{proj} = \text{MLP}_{proj}(\mathbf{F}_{img}) \in \mathbb{R}^{B \times 768}$$

**因果特征提取**：
$$\mathbf{z}_{causal} = \text{MLP}_{causal}(\mathbf{F}_{proj}) \in \mathbb{R}^{B \times 768}$$

**噪声特征提取**：
$$\mathbf{z}_{noise} = \text{MLP}_{noise}(\mathbf{F}_{proj}) \in \mathbb{R}^{B \times 768}$$

其中：
- $\mathbf{z}_{causal}$：与疾病相关的因果特征，跨中心不变
- $\mathbf{z}_{noise}$：与中心相关的噪声特征，用于对抗训练

### 2.3 语义锚点生成

#### 方案A：LLM嵌入路径（Bio-COT 2.0）

**离线LLM嵌入提取**：
$$\mathbf{E}_{llm} = \text{LLM}(\text{Prompt}(\mathbf{C})) \in \mathbb{R}^{B \times d_{llm}}$$

其中 $d_{llm} = 768$（bert-base-uncased）或 $4096$（meditron-7b）

**语义投影**：
$$\mathbf{z}_{sem} = \text{TextProjector}(\mathbf{E}_{llm}) \in \mathbb{R}^{B \times 768}$$

**TextProjector结构**：
$$\mathbf{z}_{sem} = \text{Linear}_{2}(\text{Dropout}(\text{GELU}(\text{LayerNorm}(\text{Linear}_{1}(\mathbf{E}_{llm}))))$$

其中：
- $\text{Linear}_{1}: \mathbb{R}^{d_{llm}} \rightarrow \mathbb{R}^{2048}$
- $\text{Linear}_{2}: \mathbb{R}^{2048} \rightarrow \mathbb{R}^{768}$

#### 方案B：传统MLP路径（Bio-COT v1）

**临床向量构建**：
$$\mathbf{c}_{vec} = [\text{HPV}, \text{TCT}_{one-hot}, \text{Age}] \in \mathbb{R}^{B \times 7}$$

**Student Prior网络**：
$$\mathbf{z}_{sem} = \text{StudentPriorNet}(\mathbf{c}_{vec}) \in \mathbb{R}^{B \times 768}$$

**StudentPriorNet结构**：
$$\mathbf{z}_{sem} = \text{MLP}(\mathbf{c}_{vec}) = \text{Linear}_{3}(\text{Dropout}(\text{LeakyReLU}(\text{LayerNorm}(\text{Linear}_{2}(\text{LeakyReLU}(\text{LayerNorm}(\text{Linear}_{1}(\mathbf{c}_{vec})))))))$$

其中：
- $\text{Linear}_{1}: \mathbb{R}^{7} \rightarrow \mathbb{R}^{256}$
- $\text{Linear}_{2}: \mathbb{R}^{256} \rightarrow \mathbb{R}^{512}$
- $\text{Linear}_{3}: \mathbb{R}^{512} \rightarrow \mathbb{R}^{768}$

### 2.4 跨模态融合

#### 方案A：Cross-Attention融合（Bio-COT 2.0）

**多头交叉注意力**：
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

其中：
- $\mathbf{Q} = \mathbf{z}_{causal} \in \mathbb{R}^{B \times 1 \times 768}$（Query来自图像）
- $\mathbf{K} = \mathbf{V} = \mathbf{z}_{sem} \in \mathbb{R}^{B \times 1 \times 768}$（Key/Value来自语义锚点）

**残差连接与层归一化**：
$$\mathbf{h}_{1} = \text{LayerNorm}(\mathbf{z}_{causal} + \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}))$$

**前馈网络（FFN）**：
$$\mathbf{h}_{2} = \text{Linear}_{2}(\text{Dropout}(\text{GELU}(\text{Linear}_{1}(\mathbf{h}_{1}))))$$

其中：
- $\text{Linear}_{1}: \mathbb{R}^{768} \rightarrow \mathbb{R}^{3072}$
- $\text{Linear}_{2}: \mathbb{R}^{3072} \rightarrow \mathbb{R}^{768}$

**最终融合特征**：
$$\mathbf{f}_{fused} = \text{LayerNorm}(\mathbf{h}_{1} + \mathbf{h}_{2}) \in \mathbb{R}^{B \times 768}$$

#### 方案B：简单拼接融合（Bio-COT v1）

$$\mathbf{f}_{fused} = \text{MLP}(\text{Concat}([\mathbf{z}_{causal}, \mathbf{z}_{sem}])) \in \mathbb{R}^{B \times 768}$$

其中：
$$\text{MLP}: \mathbb{R}^{1536} \rightarrow \mathbb{R}^{768}$$

### 2.5 分类预测

$$\hat{\mathbf{y}} = \text{Classifier}(\mathbf{f}_{fused}) \in \mathbb{R}^{B \times 2}$$

$$\text{Classifier}(\mathbf{x}) = \text{Linear}_{2}(\text{Dropout}(\text{GELU}(\text{LayerNorm}(\text{Linear}_{1}(\mathbf{x})))))$$

其中：
- $\text{Linear}_{1}: \mathbb{R}^{768} \rightarrow \mathbb{R}^{384}$
- $\text{Linear}_{2}: \mathbb{R}^{384} \rightarrow \mathbb{R}^{2}$

**预测概率**：
$$\mathbf{p} = \text{softmax}(\hat{\mathbf{y}}) \in \mathbb{R}^{B \times 2}$$

---

## 三、损失函数

### 3.1 总损失函数

$$\mathcal{L}_{total} = \mathcal{L}_{cls} + \lambda_{ot} \mathcal{L}_{ot} + \lambda_{consist} \mathcal{L}_{consist} + \lambda_{adv} \mathcal{L}_{adv}$$

其中：
- $\lambda_{ot} = 1.0$（Sinkhorn OT损失权重）
- $\lambda_{consist} = 0.5$（反事实一致性损失权重）
- $\lambda_{adv} = 1.0$（对抗损失权重）

### 3.2 分类损失（Focal Loss）

$$\mathcal{L}_{cls} = -\frac{1}{B}\sum_{i=1}^{B} \alpha_{y_i} (1-p_{i,y_i})^{\gamma} \log(p_{i,y_i})$$

其中：
- $p_{i,y_i}$：样本 $i$ 在真实类别 $y_i$ 上的预测概率
- $\alpha = 0.25$：类别权重
- $\gamma = 2.0$：聚焦参数

### 3.3 Sinkhorn最优传输损失

**代价矩阵**：
$$\mathbf{C}_{ij} = \|\mathbf{z}_{causal}^{(i)} - \mathbf{z}_{sem}^{(j)}\|_2^2, \quad i,j \in \{1, \ldots, B\}$$

**熵正则化最优传输**：
$$\min_{\mathbf{P} \in \mathcal{U}(\mathbf{a}, \mathbf{b})} \langle \mathbf{P}, \mathbf{C} \rangle - \epsilon H(\mathbf{P})$$

其中：
- $\mathcal{U}(\mathbf{a}, \mathbf{b}) = \{\mathbf{P} \in \mathbb{R}_{+}^{B \times B} : \mathbf{P}\mathbf{1} = \mathbf{a}, \mathbf{P}^T\mathbf{1} = \mathbf{b}\}$：传输计划约束
- $\mathbf{a} = \mathbf{b} = \frac{1}{B}\mathbf{1}$：均匀分布
- $H(\mathbf{P}) = -\sum_{ij} P_{ij} \log P_{ij}$：熵项
- $\epsilon = 0.1$：熵正则化系数

**Sinkhorn迭代**（对数域，数值稳定）：
$$\mathbf{u}^{(t+1)} = \frac{1}{B} \oslash (\mathbf{K} \mathbf{v}^{(t)} + \delta)$$
$$\mathbf{v}^{(t+1)} = \frac{1}{B} \oslash (\mathbf{K}^T \mathbf{u}^{(t+1)} + \delta)$$

其中：
- $\mathbf{K} = \exp(-\mathbf{C}/\epsilon)$：核矩阵
- $\delta = 10^{-8}$：数值稳定性常数
- $\oslash$：逐元素除法

**最优传输距离**：
$$\mathcal{L}_{ot} = \frac{1}{B} \sum_{i,j} P_{ij}^{*} C_{ij}$$

其中 $\mathbf{P}^{*} = \text{diag}(\mathbf{u}^{*}) \mathbf{K} \text{diag}(\mathbf{v}^{*})$ 是收敛后的传输计划。

### 3.4 反事实一致性损失

**Memory Bank更新**：
$$\mathcal{M}_c \leftarrow \text{FIFO}(\mathcal{M}_c, \{\mathbf{z}_{noise}^{(i)} : c_i = c\})$$

其中 $\mathcal{M}_c$ 是中心 $c$ 的噪声特征库（容量 $K=100$）。

**反事实噪声采样**：
$$\mathbf{z}_{noise}^{cf} = \text{Sample}(\mathcal{M}_{c'})$$

其中 $c' \neq c$ 是随机选择的其他中心。

**反事实特征构建**：
$$\mathbf{z}_{mix} = \mathbf{z}_{causal} + \alpha \cdot \mathbf{z}_{noise}^{cf}, \quad \alpha = 0.3$$

**一致性损失**：
$$\mathcal{L}_{consist} = \frac{1}{B}\sum_{i=1}^{B} \|\hat{\mathbf{y}}^{(i)} - \hat{\mathbf{y}}_{cf}^{(i)}\|_2^2$$

其中：
- $\hat{\mathbf{y}}^{(i)} = \text{Classifier}(\mathbf{f}_{fused}^{(i)})$：原始预测
- $\hat{\mathbf{y}}_{cf}^{(i)} = \text{Classifier}(\mathbf{z}_{mix}^{(i)})$：反事实预测

### 3.5 对抗损失

**中心判别器**：
$$\hat{c} = \text{CenterDiscriminator}(\mathbf{z}_{noise}) \in \mathbb{R}^{B \times |\mathcal{C}|}$$

其中 $|\mathcal{C}|$ 是中心数量。

**对抗损失**（鼓励 $\mathbf{z}_{noise}$ 不包含中心信息）：
$$\mathcal{L}_{adv} = \begin{cases}
-\frac{1}{B}\sum_{i=1}^{B} H(\text{softmax}(\hat{c}^{(i)})) & \text{if } |\mathcal{C}| \leq 2 \\
-\frac{1}{B}\sum_{i=1}^{B} \log \text{softmax}(\hat{c}^{(i)})_{c_i} & \text{otherwise}
\end{cases}$$

其中 $H(\cdot)$ 是熵函数，鼓励预测不确定性（高熵）。

---

## 四、方法流程图

### 4.1 整体架构图（文字描述）

```
┌─────────────────────────────────────────────────────────────────┐
│                     Bio-COT 2.0 架构图                           │
└─────────────────────────────────────────────────────────────────┘

输入层
├── OCT图像序列: X_oct [B, F, C, H, W]
├── Colposcopy图像: X_colpo [B, K, C, H, W]  
└── 临床数据: C [B, 7] (HPV, TCT, Age)

    ↓
    
【模块1: 图像特征提取】
├── ViT编码器: F_oct = ViT(X_oct) [B, 768]
├── ViT编码器: F_colpo = ViT(X_colpo) [B, 768]
└── 加权融合: F_img = 0.6·F_oct + 0.4·F_colpo [B, 768]

    ↓
    
【模块2: 双头图像编码器】
├── 特征投影: F_proj = MLP_proj(F_img) [B, 768]
├── 因果头: z_causal = MLP_causal(F_proj) [B, 768]
└── 噪声头: z_noise = MLP_noise(F_proj) [B, 768]

    ↓
    
【模块3: 语义锚点生成】
├── 路径A (LLM): E_llm = LLM(Prompt(C)) → z_sem = TextProjector(E_llm)
└── 路径B (MLP): c_vec = [HPV, TCT, Age] → z_sem = StudentPriorNet(c_vec)

    ↓
    
【模块4: 跨模态融合】
├── 方案A (Cross-Attn): 
│   └── Q=z_causal, K=V=z_sem → Attention → f_fused
└── 方案B (Concat):
    └── f_fused = MLP([z_causal, z_sem])

    ↓
    
【模块5: 分类预测】
└── y_hat = Classifier(f_fused) [B, 2]

    ↓
    
【模块6: 损失计算（训练时）】
├── L_cls: Focal Loss (分类损失)
├── L_ot: Sinkhorn OT (z_causal ↔ z_sem)
├── L_consist: Counterfactual Consistency (y_hat vs y_hat_cf)
└── L_adv: Adversarial Loss (z_noise → Center Discriminator)
```

### 4.2 关键模块详细图

#### 图1: Cross-Attention融合模块

```
输入: z_causal [B, 768], z_sem [B, 768]

    z_causal ──┐
               │
               ├─→ [Unsqueeze] → [B, 1, 768] (Query)
               │
    z_sem ─────┼─→ [Unsqueeze] → [B, 1, 768] (Key/Value)
               │
               ↓
    ┌──────────────────────────────┐
    │  Multi-Head Cross-Attention   │
    │  (8 heads, dim=768)           │
    │  Q = z_causal, K=V = z_sem    │
    └──────────────────────────────┘
               │
               ↓
    ┌──────────────────────────────┐
    │  Residual + LayerNorm        │
    │  h1 = LN(z_causal + Attn)    │
    └──────────────────────────────┘
               │
               ↓
    ┌──────────────────────────────┐
    │  Feed-Forward Network        │
    │  FFN: 768 → 3072 → 768       │
    └──────────────────────────────┘
               │
               ↓
    ┌──────────────────────────────┐
    │  Residual + LayerNorm        │
    │  f_fused = LN(h1 + FFN(h1))  │
    └──────────────────────────────┘
               │
               ↓
    输出: f_fused [B, 768]
```

#### 图2: Sinkhorn最优传输流程

```
输入: z_causal [B, 768], z_sem [B, 768]

    ┌──────────────────────────────┐
    │  1. 归一化特征                │
    │  z_causal = L2_norm(z_causal) │
    │  z_sem = L2_norm(z_sem)       │
    └──────────────────────────────┘
               │
               ↓
    ┌──────────────────────────────┐
    │  2. 计算代价矩阵              │
    │  C_ij = ||z_causal^(i) -      │
    │         z_sem^(j)||²         │
    │  C ∈ [B, B]                   │
    └──────────────────────────────┘
               │
               ↓
    ┌──────────────────────────────┐
    │  3. 计算核矩阵                │
    │  K = exp(-C / ε)              │
    │  ε = 0.1 (熵正则化系数)       │
    └──────────────────────────────┘
               │
               ↓
    ┌──────────────────────────────┐
    │  4. Sinkhorn迭代 (t=1...T)    │
    │  u^(t+1) = 1/B ⊘ (K·v^(t))   │
    │  v^(t+1) = 1/B ⊘ (K^T·u^(t+1))│
    │  T = 100 (最大迭代次数)       │
    └──────────────────────────────┘
               │
               ↓
    ┌──────────────────────────────┐
    │  5. 计算传输计划               │
    │  P* = diag(u*) · K · diag(v*) │
    └──────────────────────────────┘
               │
               ↓
    ┌──────────────────────────────┐
    │  6. 计算OT距离                │
    │  L_ot = (1/B) · sum(P*_ij · C_ij)│
    └──────────────────────────────┘
               │
               ↓
    输出: L_ot (标量)
```

#### 图3: 反事实一致性机制

```
训练时:

    z_causal [B, 768] ──┐
                        │
    z_noise [B, 768] ───┼─→ Memory Bank更新
                        │   M_c ← FIFO(M_c, z_noise)
                        │
                        ↓
    ┌─────────────────────────────────────┐
    │  反事实噪声采样                      │
    │  z_noise^cf = Sample(M_c')          │
    │  c' ≠ c (随机选择其他中心)           │
    └─────────────────────────────────────┘
                        │
                        ↓
    ┌─────────────────────────────────────┐
    │  反事实特征构建                      │
    │  z_mix = z_causal + α·z_noise^cf    │
    │  α = 0.3 (混合系数)                 │
    └─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ↓                               ↓
    ┌──────────────┐            ┌──────────────┐
    │  原始预测     │            │  反事实预测  │
    │  y_hat =      │            │  y_hat_cf =  │
    │  Classifier(  │            │  Classifier( │
    │   f_fused)    │            │   z_mix)     │
    └──────────────┘            └──────────────┘
        │                               │
        └───────────────┬───────────────┘
                        ↓
            ┌───────────────────────┐
            │  一致性损失             │
            │  L_consist =            │
            │  ||y_hat - y_hat_cf||²  │
            └───────────────────────┘
```

---

## 五、模块详解

### 5.1 TextProjector（LLM语义投影器）

**功能**：将离线提取的LLM嵌入映射到与图像特征对齐的维度

**数学表达**：
$$\mathbf{z}_{sem} = \text{TextProjector}(\mathbf{E}_{llm})$$

$$\text{TextProjector}(\mathbf{x}) = \mathbf{W}_2 \cdot \text{Dropout}(\text{GELU}(\text{LayerNorm}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1))) + \mathbf{b}_2$$

其中：
- $\mathbf{W}_1 \in \mathbb{R}^{2048 \times d_{llm}}, \mathbf{b}_1 \in \mathbb{R}^{2048}$
- $\mathbf{W}_2 \in \mathbb{R}^{768 \times 2048}, \mathbf{b}_2 \in \mathbb{R}^{768}$
- Dropout率：0.2

### 5.2 CrossModalFusion（跨模态融合）

**功能**：使用Cross-Attention机制融合图像因果特征和语义锚点

**数学表达**：
$$\mathbf{Q} = \mathbf{z}_{causal} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{z}_{sem} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{z}_{sem} \mathbf{W}_V$$

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

$$\mathbf{h}_1 = \text{LayerNorm}(\mathbf{z}_{causal} + \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}))$$

$$\mathbf{f}_{fused} = \text{LayerNorm}(\mathbf{h}_1 + \text{FFN}(\mathbf{h}_1))$$

其中：
- $d_k = 768 / 8 = 96$（每个注意力头的维度）
- 注意力头数：8
- FFN：$\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \cdot \text{Dropout}(\text{GELU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)) + \mathbf{b}_2$
  - $\mathbf{W}_1 \in \mathbb{R}^{3072 \times 768}, \mathbf{W}_2 \in \mathbb{R}^{768 \times 3072}$

### 5.3 DualHeadImageEncoder（双头图像编码器）

**功能**：从图像特征中解耦出因果特征和噪声特征

**数学表达**：
$$\mathbf{F}_{proj} = \text{MLP}_{proj}(\mathbf{F}_{img})$$

$$\mathbf{z}_{causal} = \text{MLP}_{causal}(\mathbf{F}_{proj})$$
$$\mathbf{z}_{noise} = \text{MLP}_{noise}(\mathbf{F}_{proj})$$

其中：
- $\text{MLP}_{proj}: \mathbb{R}^{768} \rightarrow \mathbb{R}^{1536} \rightarrow \mathbb{R}^{768}$
- $\text{MLP}_{causal}: \mathbb{R}^{768} \rightarrow \mathbb{R}^{768} \rightarrow \mathbb{R}^{768}$
- $\text{MLP}_{noise}: \mathbb{R}^{768} \rightarrow \mathbb{R}^{768} \rightarrow \mathbb{R}^{768}$

### 5.4 Memory Bank（噪声特征库）

**功能**：为每个中心维护一个噪声特征库，用于反事实干预

**数学表达**：
$$\mathcal{M}_c = \{\mathbf{z}_{noise}^{(1)}, \ldots, \mathbf{z}_{noise}^{(K)}\}$$

其中 $K=100$ 是每个中心的容量。

**更新策略（FIFO）**：
$$\mathcal{M}_c \leftarrow \text{FIFO}(\mathcal{M}_c, \{\mathbf{z}_{noise}^{(i)} : c_i = c\})$$

**采样策略**：
$$\mathbf{z}_{noise}^{cf} = \begin{cases}
\text{Random}(\mathcal{M}_{c'}) & \text{if strategy='random'} \\
\text{Mean}(\mathcal{M}_{c'}) & \text{if strategy='mean'} \\
\text{Nearest}(\mathcal{M}_{c'}, \mathbf{z}_{causal}) & \text{if strategy='nearest'}
\end{cases}$$

---

## 六、消融实验配置

### 6.1 模块化设计

Bio-COT 2.0支持以下开关：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `use_llm` | 是否使用LLM嵌入（True=LLM, False=MLP） | True |
| `use_cross_attn` | 是否使用Cross-Attention（True=Cross-Attn, False=Concat） | True |
| `use_ot` | 是否使用Sinkhorn OT损失 | True |
| `use_dual` | 是否使用Dual-Head结构 | True |

### 6.2 消融实验配置表

| 配置 | use_llm | use_cross_attn | use_ot | use_dual | 说明 |
|------|---------|----------------|--------|----------|------|
| Baseline | False | False | False | False | 简单拼接融合 |
| + Cross-Attn | False | True | False | False | 添加Cross-Attention |
| + LLM | True | True | False | False | 添加LLM嵌入 |
| + OT | True | True | True | False | 添加Sinkhorn OT |
| Full v2.0 | True | True | True | True | 完整Bio-COT 2.0 |

---

## 七、训练配置

### 7.1 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Batch Size | 16 | 批次大小 |
| Learning Rate | 0.00024 | 学习率（线性缩放：base_lr × 2） |
| Weight Decay | 1e-5 | 权重衰减 |
| Epochs | 100 | 训练轮数 |
| Optimizer | AdamW | 优化器 |
| Loss Function | Focal Loss | $\alpha=0.25, \gamma=2.0$ |

### 7.2 损失权重

| 损失项 | 权重 | 说明 |
|--------|------|------|
| $\mathcal{L}_{cls}$ | 1.0 | 分类损失 |
| $\mathcal{L}_{ot}$ | 1.0 | Sinkhorn OT损失 |
| $\mathcal{L}_{consist}$ | 0.5 | 反事实一致性损失 |
| $\mathcal{L}_{adv}$ | 1.0 | 对抗损失 |

---

## 八、方法图绘制建议

### 8.1 主架构图（Figure 1）

**建议布局**：
1. **左侧**：输入数据（OCT、Colposcopy、Clinical）
2. **中间上方**：图像特征提取路径（ViT → Dual-Head）
3. **中间下方**：语义锚点生成路径（LLM/MLP）
4. **右侧**：跨模态融合（Cross-Attention）→ 分类器

**颜色建议**：
- 图像路径：蓝色系
- 文本路径：绿色系
- 融合模块：橙色系
- 损失函数：红色系

### 8.2 Sinkhorn OT流程图（Figure 2）

**建议布局**：
1. 左侧：输入特征（z_causal, z_sem）
2. 中间：Sinkhorn迭代过程（可视化迭代步骤）
3. 右侧：输出OT距离

**可视化元素**：
- 代价矩阵热力图
- 传输计划矩阵可视化
- 迭代收敛曲线

### 8.3 反事实一致性机制图（Figure 3）

**建议布局**：
1. 上方：原始预测路径
2. 中间：Memory Bank和反事实采样
3. 下方：反事实预测路径
4. 右侧：一致性损失计算

**可视化元素**：
- Memory Bank结构示意图
- 反事实特征混合过程
- 预测一致性对比

---

## 九、关键创新点总结

1. **LLM语义锚点**：使用医学LLM提取临床数据的语义嵌入，替代传统MLP，提升语义理解能力
2. **Cross-Attention融合**：使用多头交叉注意力机制实现更灵活的跨模态交互
3. **Sinkhorn最优传输**：替代KL散度，实现更灵活的分布对齐，具有几何直观性
4. **反事实一致性**：通过Memory Bank机制实现真正的反事实干预，确保模型对噪声不敏感

---

## 十、参考文献格式建议

在论文中引用相关方法时，建议格式：

1. **Sinkhorn算法**：Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport. NIPS.
2. **Cross-Attention**：Vaswani, A., et al. (2017). Attention is all you need. NIPS.
3. **Focal Loss**：Lin, T. Y., et al. (2017). Focal loss for dense object detection. ICCV.
4. **Vision Transformer**：Dosovitskiy, A., et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. ICLR.

---

**文档版本**：v1.0  
**最后更新**：2025-01-08  
**作者**：Bio-COT 2.0 Research Team

