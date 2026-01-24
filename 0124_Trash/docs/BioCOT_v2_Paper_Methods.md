# Bio-COT 2.0: SCI论文方法部分（精简版）

## 一、方法概述

Bio-COT 2.0 (Biological Causal Optimal Transport 2.0) 是一个多模态医学图像分类框架，通过LLM语义锚点、因果特征解耦和最优传输理论实现跨中心域不变性。

---

## 二、整体架构

### 2.1 输入数据

给定多模态医学数据：
- **OCT图像序列**：$\mathbf{X}_{oct} \in \mathbb{R}^{B \times F \times C \times H \times W}$
- **Colposcopy图像**：$\mathbf{X}_{colpo} \in \mathbb{R}^{B \times K \times C \times H \times W}$
- **临床数据**：$\mathbf{C} = [\text{HPV}, \text{TCT}, \text{Age}] \in \mathbb{R}^{B \times 7}$

### 2.2 特征提取

**图像特征提取**：
$$\mathbf{F}_{oct} = \text{ViT}(\mathbf{X}_{oct}), \quad \mathbf{F}_{colpo} = \text{ViT}(\mathbf{X}_{colpo})$$

**多模态图像融合**：
$$\mathbf{F}_{img} = \alpha \mathbf{F}_{oct} + (1-\alpha) \mathbf{F}_{colpo}, \quad \alpha = 0.6$$

**双头图像编码**：
$$\mathbf{z}_{causal} = \text{MLP}_{causal}(\mathbf{F}_{img}), \quad \mathbf{z}_{noise} = \text{MLP}_{noise}(\mathbf{F}_{img})$$

其中 $\mathbf{z}_{causal} \in \mathbb{R}^{B \times 768}$ 是因果特征（疾病相关），$\mathbf{z}_{noise} \in \mathbb{R}^{B \times 768}$ 是噪声特征（中心相关）。

### 2.3 语义锚点生成

**LLM嵌入提取**（离线预处理）：
$$\mathbf{E}_{llm} = \text{LLM}(\text{Prompt}(\mathbf{C})) \in \mathbb{R}^{B \times d_{llm}}$$

**语义投影**：
$$\mathbf{z}_{sem} = \text{TextProjector}(\mathbf{E}_{llm}) \in \mathbb{R}^{B \times 768}$$

其中 $\text{TextProjector}$ 是一个两层MLP：$\mathbb{R}^{d_{llm}} \rightarrow \mathbb{R}^{2048} \rightarrow \mathbb{R}^{768}$。

### 2.4 跨模态融合

**Cross-Attention机制**：
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

其中 $\mathbf{Q} = \mathbf{z}_{causal}$（Query），$\mathbf{K} = \mathbf{V} = \mathbf{z}_{sem}$（Key/Value）。

**残差连接与层归一化**：
$$\mathbf{f}_{fused} = \text{LayerNorm}(\mathbf{z}_{causal} + \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) + \text{FFN}(\cdot))$$

### 2.5 分类预测

$$\hat{\mathbf{y}} = \text{Classifier}(\mathbf{f}_{fused}) \in \mathbb{R}^{B \times 2}$$

---

## 三、损失函数

### 3.1 总损失

$$\mathcal{L}_{total} = \mathcal{L}_{cls} + \lambda_{ot} \mathcal{L}_{ot} + \lambda_{consist} \mathcal{L}_{consist} + \lambda_{adv} \mathcal{L}_{adv}$$

其中 $\lambda_{ot} = 1.0$，$\lambda_{consist} = 0.5$，$\lambda_{adv} = 1.0$。

### 3.2 分类损失（Focal Loss）

$$\mathcal{L}_{cls} = -\frac{1}{B}\sum_{i=1}^{B} \alpha_{y_i} (1-p_{i,y_i})^{\gamma} \log(p_{i,y_i})$$

其中 $\alpha = 0.25$，$\gamma = 2.0$。

### 3.3 Sinkhorn最优传输损失

**代价矩阵**：
$$\mathbf{C}_{ij} = \|\mathbf{z}_{causal}^{(i)} - \mathbf{z}_{sem}^{(j)}\|_2^2$$

**熵正则化最优传输**：
$$\min_{\mathbf{P} \in \mathcal{U}(\mathbf{a}, \mathbf{b})} \langle \mathbf{P}, \mathbf{C} \rangle - \epsilon H(\mathbf{P})$$

其中 $\mathcal{U}(\mathbf{a}, \mathbf{b})$ 是传输计划约束，$\epsilon = 0.1$ 是熵正则化系数。

**Sinkhorn迭代**：
$$\mathbf{u}^{(t+1)} = \frac{1}{B} \oslash (\mathbf{K} \mathbf{v}^{(t)}), \quad \mathbf{v}^{(t+1)} = \frac{1}{B} \oslash (\mathbf{K}^T \mathbf{u}^{(t+1)})$$

其中 $\mathbf{K} = \exp(-\mathbf{C}/\epsilon)$。

**OT距离**：
$$\mathcal{L}_{ot} = \frac{1}{B} \sum_{i,j} P_{ij}^{*} C_{ij}$$

其中 $\mathbf{P}^{*} = \text{diag}(\mathbf{u}^{*}) \mathbf{K} \text{diag}(\mathbf{v}^{*})$ 是收敛后的传输计划。

### 3.4 反事实一致性损失

**Memory Bank更新**：
$$\mathcal{M}_c \leftarrow \text{FIFO}(\mathcal{M}_c, \{\mathbf{z}_{noise}^{(i)} : c_i = c\})$$

**反事实特征构建**：
$$\mathbf{z}_{mix} = \mathbf{z}_{causal} + \alpha \cdot \mathbf{z}_{noise}^{cf}, \quad \alpha = 0.3$$

其中 $\mathbf{z}_{noise}^{cf}$ 是从其他中心 $\mathcal{M}_{c'}$ 采样的噪声特征。

**一致性损失**：
$$\mathcal{L}_{consist} = \frac{1}{B}\sum_{i=1}^{B} \|\hat{\mathbf{y}}^{(i)} - \hat{\mathbf{y}}_{cf}^{(i)}\|_2^2$$

其中 $\hat{\mathbf{y}}_{cf}^{(i)} = \text{Classifier}(\mathbf{z}_{mix}^{(i)})$ 是反事实预测。

### 3.5 对抗损失

$$\mathcal{L}_{adv} = -\frac{1}{B}\sum_{i=1}^{B} H(\text{softmax}(\text{CenterDiscriminator}(\mathbf{z}_{noise}^{(i)})))$$

其中 $H(\cdot)$ 是熵函数，鼓励 $\mathbf{z}_{noise}$ 不包含中心信息。

---

## 四、方法图描述

### Figure 1: Bio-COT 2.0整体架构

**布局建议**：
- **左侧列**：输入数据（OCT图像、Colposcopy图像、临床数据）
- **中间上方**：图像特征提取路径（ViT编码器 → 双头图像编码器）
- **中间下方**：语义锚点生成路径（LLM嵌入 → TextProjector）
- **右侧**：跨模态融合（Cross-Attention）→ 分类器 → 预测结果

**关键连接**：
- 图像路径：$\mathbf{X}_{oct}, \mathbf{X}_{colpo} \rightarrow \mathbf{F}_{img} \rightarrow \mathbf{z}_{causal}, \mathbf{z}_{noise}$
- 文本路径：$\mathbf{C} \rightarrow \mathbf{E}_{llm} \rightarrow \mathbf{z}_{sem}$
- 融合路径：$\mathbf{z}_{causal}, \mathbf{z}_{sem} \rightarrow \mathbf{f}_{fused} \rightarrow \hat{\mathbf{y}}$

**颜色编码**：
- 图像模块：蓝色
- 文本模块：绿色
- 融合模块：橙色
- 损失函数：红色

### Figure 2: Sinkhorn最优传输流程

**布局建议**：
- **左侧**：输入特征 $\mathbf{z}_{causal}$ 和 $\mathbf{z}_{sem}$
- **中间上方**：代价矩阵 $\mathbf{C}$ 的可视化（热力图）
- **中间下方**：Sinkhorn迭代过程（可视化迭代步骤 $t=1,2,\ldots,T$）
- **右侧**：传输计划 $\mathbf{P}^{*}$ 和OT距离 $\mathcal{L}_{ot}$

**可视化元素**：
- 代价矩阵热力图（颜色表示距离）
- 传输计划矩阵（箭头表示传输方向）
- 迭代收敛曲线（横轴：迭代次数，纵轴：OT距离）

### Figure 3: 反事实一致性机制

**布局建议**：
- **上方**：原始预测路径（$\mathbf{z}_{causal} \rightarrow \hat{\mathbf{y}}$）
- **中间**：Memory Bank结构（每个中心一个特征库）
- **下方**：反事实预测路径（$\mathbf{z}_{causal} + \alpha \mathbf{z}_{noise}^{cf} \rightarrow \hat{\mathbf{y}}_{cf}$）
- **右侧**：一致性损失计算（$\|\hat{\mathbf{y}} - \hat{\mathbf{y}}_{cf}\|_2^2$）

**可视化元素**：
- Memory Bank结构图（每个中心的特征库）
- 反事实特征混合过程（$\mathbf{z}_{mix} = \mathbf{z}_{causal} + \alpha \mathbf{z}_{noise}^{cf}$）
- 预测一致性对比（散点图：$\hat{\mathbf{y}}$ vs $\hat{\mathbf{y}}_{cf}$）

---

## 五、关键创新点

1. **LLM语义锚点**：使用医学LLM提取临床数据的语义嵌入，替代传统MLP，提升语义理解能力
2. **Cross-Attention融合**：使用多头交叉注意力机制实现更灵活的跨模态交互
3. **Sinkhorn最优传输**：替代KL散度，实现更灵活的分布对齐，具有几何直观性
4. **反事实一致性**：通过Memory Bank机制实现真正的反事实干预，确保模型对噪声不敏感

---

## 六、实验配置

### 6.1 超参数

| 参数 | 值 |
|------|-----|
| Batch Size | 16 |
| Learning Rate | 0.00024 |
| Weight Decay | 1e-5 |
| Epochs | 100 |
| Optimizer | AdamW |
| Loss Function | Focal Loss ($\alpha=0.25, \gamma=2.0$) |

### 6.2 损失权重

| 损失项 | 权重 |
|--------|------|
| $\mathcal{L}_{cls}$ | 1.0 |
| $\mathcal{L}_{ot}$ | 1.0 |
| $\mathcal{L}_{consist}$ | 0.5 |
| $\mathcal{L}_{adv}$ | 1.0 |

---

## 七、消融实验配置

| 配置 | LLM | Cross-Attn | OT | Dual-Head | 说明 |
|------|-----|-----------|----|-----------|------|
| Baseline | ✗ | ✗ | ✗ | ✗ | 简单拼接融合 |
| + Cross-Attn | ✗ | ✓ | ✗ | ✗ | 添加Cross-Attention |
| + LLM | ✓ | ✓ | ✗ | ✗ | 添加LLM嵌入 |
| + OT | ✓ | ✓ | ✓ | ✗ | 添加Sinkhorn OT |
| Full v2.0 | ✓ | ✓ | ✓ | ✓ | 完整Bio-COT 2.0 |

---

**文档版本**：v1.0  
**最后更新**：2025-01-08

