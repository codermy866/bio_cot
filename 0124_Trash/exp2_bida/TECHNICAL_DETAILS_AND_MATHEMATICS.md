# Bio-COT方法详细技术方案与数学公式

**文档版本**: v1.0  
**创建时间**: 2026-01-04  
**目的**: 为CCF A类期刊发表提供完整的技术细节和数学推导

---

## 目录

1. [整体架构与符号定义](#1-整体架构与符号定义)
2. [Student Prior网络](#2-student-prior网络)
3. [多模态图像特征融合](#3-多模态图像特征融合)
4. [双头图像编码器](#4-双头图像编码器)
5. [三模态融合](#5-三模态融合)
6. [Memory Bank反事实干预](#6-memory-bank反事实干预)
7. [损失函数详细推导](#7-损失函数详细推导)
8. [训练策略](#8-训练策略)
9. [实验设置](#9-实验设置)

---

## 1. 整体架构与符号定义

### 1.1 输入符号定义

设批次大小为 $B$，定义输入：

- **OCT特征**: $\mathbf{X}_{oct} \in \mathbb{R}^{B \times 512}$ - OCT图像提取的特征
- **Colposcopy特征**: $\mathbf{X}_{colpo} \in \mathbb{R}^{B \times 512}$ - 阴道镜图像提取的特征
- **临床数据**: $\mathbf{c} \in \mathbb{R}^{B \times 7}$ - 临床数据向量
  - HPV: $c_{hpv} \in \{0, 1\}$ (1维)
  - TCT: $\mathbf{c}_{tct} \in \{0,1\}^5$ (5维one-hot: NILM, ASC-US, LSIL, HSIL, 其他)
  - Age: $c_{age} \in [0, 1]$ (1维，归一化)
- **中心标签**: $\mathbf{y}_{center} \in \{0,1,\ldots,4\}^B$ - 医疗中心ID (5个中心)
- **分类标签**: $\mathbf{y} \in \{0,1\}^B$ - 二分类标签 (0=正常, 1=异常)

### 1.2 中间特征符号

- **语义锚点**: $\mathbf{z}_{sem} \in \mathbb{R}^{B \times 768}$ - Student Prior输出
- **因果特征**: $\mathbf{z}_{causal} \in \mathbb{R}^{B \times 768}$ - 图像编码器因果头输出
- **噪声特征**: $\mathbf{z}_{noise} \in \mathbb{R}^{B \times 768}$ - 图像编码器噪声头输出
- **融合特征**: $\mathbf{z}_{fused} \in \mathbb{R}^{B \times 768}$ - 三模态融合后特征
- **分类logits**: $\mathbf{l} \in \mathbb{R}^{B \times 2}$ - 分类器输出

### 1.3 模型参数

- **嵌入维度**: $d = 768$
- **输入维度**: $d_{in} = 512$
- **中心数量**: $C = 5$
- **类别数量**: $K = 2$

---

## 2. Student Prior网络

### 2.1 网络架构

Student Prior网络 $f_{prior}: \mathbb{R}^7 \to \mathbb{R}^{768}$ 是一个多层感知机：

$$
\mathbf{z}_{sem} = f_{prior}(\mathbf{c}) = \text{MLP}(\mathbf{c})
$$

**具体架构**:

$$
\begin{align}
\mathbf{h}_1 &= \text{BN}(\text{Linear}_1(\mathbf{c})) \in \mathbb{R}^{B \times 256} \\
\mathbf{h}_1 &= \text{LeakyReLU}(\mathbf{h}_1, \alpha=0.2) \\
\mathbf{h}_1 &= \text{Dropout}(\mathbf{h}_1, p=0.2) \\
\\
\mathbf{h}_2 &= \text{BN}(\text{Linear}_2(\mathbf{h}_1)) \in \mathbb{R}^{B \times 512} \\
\mathbf{h}_2 &= \text{LeakyReLU}(\mathbf{h}_2, \alpha=0.2) \\
\mathbf{h}_2 &= \text{Dropout}(\mathbf{h}_2, p=0.2) \\
\\
\mathbf{z}_{sem} &= \text{Linear}_3(\mathbf{h}_2) \in \mathbb{R}^{B \times 768}
\end{align}
$$

**参数**:
- $\text{Linear}_1: \mathbb{R}^7 \to \mathbb{R}^{256}$ (权重矩阵 $\mathbf{W}_1 \in \mathbb{R}^{256 \times 7}$, 偏置 $\mathbf{b}_1 \in \mathbb{R}^{256}$)
- $\text{Linear}_2: \mathbb{R}^{256} \to \mathbb{R}^{512}$ (权重矩阵 $\mathbf{W}_2 \in \mathbb{R}^{512 \times 256}$, 偏置 $\mathbf{b}_2 \in \mathbb{R}^{512}$)
- $\text{Linear}_3: \mathbb{R}^{512} \to \mathbb{R}^{768}$ (权重矩阵 $\mathbf{W}_3 \in \mathbb{R}^{768 \times 512}$, 偏置 $\mathbf{b}_3 \in \mathbb{R}^{768}$)

**总参数量**: $7 \times 256 + 256 + 256 \times 512 + 512 + 512 \times 768 + 768 = 590,080$

### 2.2 预训练目标

Student Prior通过预训练学习从临床数据到VLM语义空间的映射：

**预训练损失**:

$$
\mathcal{L}_{pretrain} = \frac{1}{N} \sum_{i=1}^{N} \|\mathbf{z}_{sem}^{(i)} - \mathbf{z}_{vlm}^{(i)}\|_2^2
$$

其中：
- $\mathbf{z}_{sem}^{(i)} = f_{prior}(\mathbf{c}^{(i)})$ - Student Prior输出
- $\mathbf{z}_{vlm}^{(i)} = \text{Proj}(f_{vlm}(\mathbf{x}_{img}^{(i)}, \mathbf{t}^{(i)}))$ - VLM特征（投影到768维）
- $N$ - 训练样本数

**优化**:

$$
\theta_{prior}^* = \arg\min_{\theta_{prior}} \mathcal{L}_{pretrain}
$$

使用Adam优化器，学习率 $\eta_{pretrain} = 2 \times 10^{-3}$，训练20个epoch。

---

## 3. 多模态图像特征融合

### 3.1 传统融合方法（不使用VLM）

#### 3.1.1 可学习加权融合

定义可学习模态权重 $\boldsymbol{\omega} \in \mathbb{R}^2$（初始化为 $[0.5, 0.5]$）：

$$
\boldsymbol{\omega}_{norm} = \text{softmax}(\boldsymbol{\omega}) = \left[\frac{e^{\omega_1}}{e^{\omega_1} + e^{\omega_2}}, \frac{e^{\omega_2}}{e^{\omega_1} + e^{\omega_2}}\right]
$$

加权融合：

$$
\mathbf{x}_{weighted} = \omega_{norm,1} \cdot \mathbf{X}_{oct} + \omega_{norm,2} \cdot \mathbf{X}_{colpo} \in \mathbb{R}^{B \times 512}
$$

#### 3.1.2 拼接+融合

拼接特征：

$$
\mathbf{x}_{concat} = [\mathbf{X}_{oct}; \mathbf{X}_{colpo}] \in \mathbb{R}^{B \times 1024}
$$

通过融合层：

$$
\begin{align}
\mathbf{h}_{fuse} &= \text{LayerNorm}(\text{Linear}_1(\mathbf{x}_{concat})) \in \mathbb{R}^{B \times 768} \\
\mathbf{h}_{fuse} &= \text{GELU}(\mathbf{h}_{fuse}) \\
\mathbf{h}_{fuse} &= \text{Dropout}(\mathbf{h}_{fuse}, p=0.1) \\
\mathbf{x}_{fused} &= \text{Linear}_2(\mathbf{h}_{fuse}) \in \mathbb{R}^{B \times 768}
\end{align}
$$

#### 3.1.3 最终融合

$$
\mathbf{x}_{image} = 0.7 \cdot \mathbf{x}_{fused} + 0.3 \cdot \text{Proj}(\mathbf{x}_{weighted}) \in \mathbb{R}^{B \times 768}
$$

其中 $\text{Proj}: \mathbb{R}^{512} \to \mathbb{R}^{768}$ 是投影层（如果维度不匹配）。

### 3.2 VLM增强融合（可选）

如果使用VLM编码器，融合VLM特征和传统特征：

$$
\begin{align}
\mathbf{z}_{oct}^{vlm}, \mathbf{z}_{noise,oct}^{vlm} &= f_{vlm}(\mathbf{I}_{oct}, \mathbf{t}) \\
\mathbf{z}_{colpo}^{vlm}, \mathbf{z}_{noise,colpo}^{vlm} &= f_{vlm}(\mathbf{I}_{colpo}, \mathbf{t}) \\
\\
\mathbf{x}_{oct}^{fused} &= 0.6 \cdot \mathbf{z}_{oct}^{vlm} + 0.4 \cdot \text{Proj}(\mathbf{X}_{oct}) \\
\mathbf{x}_{colpo}^{fused} &= 0.6 \cdot \mathbf{z}_{colpo}^{vlm} + 0.4 \cdot \text{Proj}(\mathbf{X}_{colpo}) \\
\\
\mathbf{x}_{image} &= \omega_{norm,1} \cdot \mathbf{x}_{oct}^{fused} + \omega_{norm,2} \cdot \mathbf{x}_{colpo}^{fused}
\end{align}
$$

其中 $\mathbf{t}$ 是文本提示，由临床数据生成。

---

## 4. 双头图像编码器

### 4.1 特征投影

$$
\begin{align}
\mathbf{h}_{proj} &= \text{LayerNorm}(\text{Linear}_1(\mathbf{x}_{image})) \in \mathbb{R}^{B \times 1536} \\
\mathbf{h}_{proj} &= \text{GELU}(\mathbf{h}_{proj}) \\
\mathbf{h}_{proj} &= \text{Dropout}(\mathbf{h}_{proj}, p=0.1) \\
\mathbf{x}_{proj} &= \text{Linear}_2(\mathbf{h}_{proj}) \in \mathbb{R}^{B \times 768}
\end{align}
$$

### 4.2 因果头（Causal Head）

$$
\begin{align}
\mathbf{h}_{causal} &= \text{LayerNorm}(\text{Linear}_3(\mathbf{x}_{proj})) \in \mathbb{R}^{B \times 768} \\
\mathbf{h}_{causal} &= \text{GELU}(\mathbf{h}_{causal}) \\
\mathbf{h}_{causal} &= \text{Dropout}(\mathbf{h}_{causal}, p=0.1) \\
\mathbf{z}_{causal} &= \text{Linear}_4(\mathbf{h}_{causal}) \in \mathbb{R}^{B \times 768}
\end{align}
$$

### 4.3 噪声头（Noise Head）

$$
\begin{align}
\mathbf{h}_{noise} &= \text{LayerNorm}(\text{Linear}_5(\mathbf{x}_{proj})) \in \mathbb{R}^{B \times 768} \\
\mathbf{h}_{noise} &= \text{GELU}(\mathbf{h}_{noise}) \\
\mathbf{h}_{noise} &= \text{Dropout}(\mathbf{h}_{noise}, p=0.1) \\
\mathbf{z}_{noise} &= \text{Linear}_6(\mathbf{h}_{noise}) \in \mathbb{R}^{B \times 768}
\end{align}
$$

**设计理念**: 
- $\mathbf{z}_{causal}$ 包含与疾病相关的因果特征（中心不变）
- $\mathbf{z}_{noise}$ 包含中心特异性特征（用于对抗训练）

---

## 5. 三模态融合

将图像因果特征和临床语义特征融合：

$$
\begin{align}
\mathbf{z}_{multimodal} &= [\mathbf{z}_{causal}; \mathbf{z}_{sem}] \in \mathbb{R}^{B \times 1536} \\
\\
\mathbf{h}_{multi} &= \text{LayerNorm}(\text{Linear}_7(\mathbf{z}_{multimodal})) \in \mathbb{R}^{B \times 768} \\
\mathbf{h}_{multi} &= \text{GELU}(\mathbf{h}_{multi}) \\
\mathbf{h}_{multi} &= \text{Dropout}(\mathbf{h}_{multi}, p=0.2) \\
\mathbf{z}_{fused} &= \mathbf{h}_{multi} \in \mathbb{R}^{B \times 768}
\end{align}
$$

---

## 6. Memory Bank反事实干预

### 6.1 Memory Bank结构

为每个中心 $c \in \{0,1,\ldots,4\}$ 维护一个噪声特征库：

$$
\mathcal{B}_c = \{\mathbf{z}_{noise}^{(i)} : i \in \mathcal{I}_c\}
$$

其中 $\mathcal{I}_c$ 是中心 $c$ 的样本索引集合。

**存储结构**: $\mathbf{B} \in \mathbb{R}^{C \times M \times d}$，其中 $M=100$ 是每个中心的容量。

### 6.2 更新策略（FIFO）

对于批次中的每个样本 $i$，其噪声特征 $\mathbf{z}_{noise}^{(i)}$ 和中心标签 $y_{center}^{(i)}$：

$$
\begin{align}
c &= y_{center}^{(i)} \\
\text{ptr}_c &= (\text{ptr}_c + 1) \bmod M \\
\mathbf{B}[c, \text{ptr}_c] &= \mathbf{z}_{noise}^{(i)} \\
\text{count}_c &= \min(\text{count}_c + 1, M)
\end{align}
$$

### 6.3 反事实噪声采样

对于目标中心 $c_{target}$，采样策略：

**随机采样**:
$$
\mathbf{z}_{noise}^{cf} = \mathbf{B}[c_{target}, \text{rand}(0, \text{count}_{c_{target}})]
$$

**平均采样**:
$$
\mathbf{z}_{noise}^{cf} = \frac{1}{\text{count}_{c_{target}}} \sum_{i=0}^{\text{count}_{c_{target}}-1} \mathbf{B}[c_{target}, i]
$$

### 6.4 反事实特征生成

$$
\mathbf{z}_{mix} = \mathbf{z}_{causal} + \alpha \cdot \mathbf{z}_{noise}^{cf}
$$

其中 $\alpha = 0.3$ 是混合系数。

**反事实预测**:
$$
\mathbf{l}_{cf} = f_{classifier}(\mathbf{z}_{mix})
$$

---

## 7. 损失函数详细推导

### 7.1 分类损失（Classification Loss）

使用交叉熵损失，带标签平滑：

$$
\mathcal{L}_{cls} = -\frac{1}{B} \sum_{i=1}^{B} \sum_{k=1}^{K} \tilde{y}_{i,k} \log p_{i,k}
$$

其中：
- $p_{i,k} = \text{softmax}(\mathbf{l}_i)_k = \frac{\exp(l_{i,k})}{\sum_{j=1}^{K} \exp(l_{i,j})}$ - 预测概率
- $\tilde{y}_{i,k}$ - 平滑后的标签：
  $$
  \tilde{y}_{i,k} = \begin{cases}
  1 - \epsilon + \frac{\epsilon}{K} & \text{if } k = y_i \\
  \frac{\epsilon}{K} & \text{otherwise}
  \end{cases}
  $$
  其中 $\epsilon = 0.15$ 是标签平滑系数。

### 7.2 Sinkhorn最优传输损失（OT Loss）

#### 7.2.1 代价矩阵

计算 $\mathbf{z}_{causal}$ 和 $\mathbf{z}_{sem}$ 之间的代价矩阵：

$$
\mathbf{C}_{ij} = \|\mathbf{z}_{causal}^{(i)} - \mathbf{z}_{sem}^{(j)}\|_2^2
$$

其中 $\mathbf{C} \in \mathbb{R}^{B \times B}$，$C_{ij}$ 表示将 $\mathbf{z}_{causal}^{(i)}$ 传输到 $\mathbf{z}_{sem}^{(j)}$ 的代价。

**归一化**（提高数值稳定性）:
$$
\begin{align}
\tilde{\mathbf{z}}_{causal} &= \frac{\mathbf{z}_{causal}}{\|\mathbf{z}_{causal}\|_2} \\
\tilde{\mathbf{z}}_{sem} &= \frac{\mathbf{z}_{sem}}{\|\mathbf{z}_{sem}\|_2}
\end{align}
$$

**代价矩阵限制**:
$$
\mathbf{C} = \text{clamp}(\mathbf{C}, 0, 10.0)
$$

#### 7.2.2 Sinkhorn迭代

初始化：
$$
\mathbf{u}^{(0)} = \mathbf{0} \in \mathbb{R}^B, \quad \mathbf{v}^{(0)} = \mathbf{0} \in \mathbb{R}^B
$$

计算核矩阵：
$$
\mathbf{K} = \exp\left(-\frac{\mathbf{C}}{\epsilon}\right) \in \mathbb{R}^{B \times B}
$$

其中 $\epsilon = 0.1$ 是熵正则化系数。

**迭代更新**（$t = 1, 2, \ldots, T$，$T=100$）:
$$
\begin{align}
\mathbf{u}^{(t)} &= \frac{1}{\mathbf{K} \mathbf{v}^{(t-1)} + \delta} \\
\mathbf{v}^{(t)} &= \frac{1}{\mathbf{K}^T \mathbf{u}^{(t)} + \delta}
\end{align}
$$

其中 $\delta = 10^{-8}$ 是数值稳定性常数。

**限制范围**:
$$
\begin{align}
\mathbf{u}^{(t)} &= \text{clamp}(\mathbf{u}^{(t)}, 10^{-8}, 10^8) \\
\mathbf{v}^{(t)} &= \text{clamp}(\mathbf{v}^{(t)}, 10^{-8}, 10^8)
\end{align}
$$

#### 7.2.3 传输计划与距离

传输计划：
$$
\boldsymbol{\Gamma} = \text{diag}(\mathbf{u}) \cdot \mathbf{K} \cdot \text{diag}(\mathbf{v}) \in \mathbb{R}^{B \times B}
$$

最优传输距离：
$$
\mathcal{L}_{ot} = \frac{1}{B} \sum_{i=1}^{B} \sum_{j=1}^{B} \Gamma_{ij} C_{ij}
$$

**数值稳定性检查**: 如果 $\boldsymbol{\Gamma}$ 包含 NaN 或 Inf，回退到MSE：
$$
\mathcal{L}_{ot} = \frac{1}{B} \sum_{i=1}^{B} \|\mathbf{z}_{causal}^{(i)} - \mathbf{z}_{sem}^{(i)}\|_2^2
$$

### 7.3 反事实一致性损失（Consistency Loss）

确保添加反事实噪声后，预测结果保持不变：

$$
\mathcal{L}_{consist} = \frac{1}{B} \sum_{i=1}^{B} \|\mathbf{l}_i - \mathbf{l}_{cf,i}\|_2^2
$$

其中：
- $\mathbf{l}_i = f_{classifier}(\mathbf{z}_{fused}^{(i)})$ - 原始预测
- $\mathbf{l}_{cf,i} = f_{classifier}(\mathbf{z}_{mix}^{(i)})$ - 反事实预测
- $\mathbf{z}_{mix}^{(i)} = \mathbf{z}_{causal}^{(i)} + \alpha \cdot \mathbf{z}_{noise}^{cf,(i)}$

**限制范围**:
$$
\mathcal{L}_{consist} = \text{clamp}(\mathcal{L}_{consist}, 0, 10.0)
$$

### 7.4 对抗损失（Adversarial Loss）

使用中心判别器 $D: \mathbb{R}^{768} \to \mathbb{R}^C$ 预测中心ID：

$$
\mathcal{L}_{adv} = -\frac{1}{B} \sum_{i=1}^{B} \sum_{c=0}^{C-1} \mathbb{1}[y_{center}^{(i)} = c] \log p_{center}^{(i)}(c)
$$

其中：
- $p_{center}^{(i)}(c) = \text{softmax}(D(\mathbf{z}_{noise}^{(i)}))_c$ - 预测中心 $c$ 的概率

**限制范围**:
$$
\mathcal{L}_{adv} = \text{clamp}(\mathcal{L}_{adv}, 0, 10.0)
$$

### 7.5 总损失函数

$$
\mathcal{L}_{total} = \lambda_{cls} \mathcal{L}_{cls} + \lambda_{ot} \mathcal{L}_{ot} + \lambda_{consist} \mathcal{L}_{consist} + \lambda_{adv} \mathcal{L}_{adv}
$$

**权重配置**（当前优化后）:
- $\lambda_{cls} = 3.0$ - 分类损失权重（主要任务）
- $\lambda_{ot} = 0.05$ - OT损失权重
- $\lambda_{consist} = 0.1$ - 一致性损失权重
- $\lambda_{adv} = 0.01$ - 对抗损失权重

**动态权重调整**（前5个epoch）:
$$
\lambda_{aux}(e) = \begin{cases}
0.3 + 0.7 \cdot \frac{e}{5} & \text{if } e < 5 \\
1.0 & \text{if } e \geq 5
\end{cases}
$$

调整后的总损失：
$$
\mathcal{L}_{total} = \lambda_{cls} \mathcal{L}_{cls} + \lambda_{aux}(e) \cdot (\lambda_{ot} \mathcal{L}_{ot} + \lambda_{consist} \mathcal{L}_{consist} + \lambda_{adv} \mathcal{L}_{adv})
$$

---

## 8. 训练策略

### 8.1 优化器

使用AdamW优化器：

$$
\begin{align}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \mathbf{g}_t \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) \mathbf{g}_t^2 \\
\hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1-\beta_1^t} \\
\hat{\mathbf{v}}_t &= \frac{\mathbf{v}_t}{1-\beta_2^t} \\
\boldsymbol{\theta}_{t+1} &= \boldsymbol{\theta}_t - \eta_t \left(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda_{wd} \boldsymbol{\theta}_t\right)
\end{align}
$$

**超参数**:
- $\beta_1 = 0.9$, $\beta_2 = 0.999$
- $\epsilon = 10^{-8}$
- $\lambda_{wd} = 2 \times 10^{-3}$ - Weight Decay

### 8.2 学习率调度

**Warmup阶段**（前5个epoch）:
$$
\eta_t = \eta_{warmup} + (\eta_{max} - \eta_{warmup}) \cdot \frac{t}{T_{warmup}}
$$

其中：
- $\eta_{warmup} = 10^{-5}$
- $\eta_{max} = 5 \times 10^{-5}$
- $T_{warmup} = 5$ epochs

**Cosine退火阶段**（5-50 epochs）:
$$
\eta_t = \eta_{min} + (\eta_{max} - \eta_{min}) \cdot \frac{1 + \cos(\pi \cdot \frac{t - T_{warmup}}{T_{total} - T_{warmup}})}{2}
$$

其中：
- $\eta_{min} = 10^{-6}$
- $T_{total} = 50$ epochs

### 8.3 数据增强

#### 8.3.1 MixUp（从Epoch 3开始）

对于样本 $(\mathbf{x}_i, y_i)$ 和 $(\mathbf{x}_j, y_j)$：

$$
\begin{align}
\lambda &\sim \text{Beta}(\alpha_{mixup}, \alpha_{mixup}), \quad \alpha_{mixup} = 0.2 \\
\lambda' &= \max(\lambda, 1-\lambda) \\
\tilde{\mathbf{x}} &= \lambda' \mathbf{x}_i + (1-\lambda') \mathbf{x}_j \\
\tilde{y}_a &= y_i, \quad \tilde{y}_b = y_j
\end{align}
$$

**损失计算**:
$$
\mathcal{L}_{mixup} = \lambda' \mathcal{L}(\tilde{\mathbf{x}}, \tilde{y}_a) + (1-\lambda') \mathcal{L}(\tilde{\mathbf{x}}, \tilde{y}_b)
$$

#### 8.3.2 CutMix（从Epoch 5开始）

对于样本 $(\mathbf{x}_i, y_i)$ 和 $(\mathbf{x}_j, y_j)$：

$$
\begin{align}
\lambda &\sim \text{Beta}(\alpha_{cutmix}, \alpha_{cutmix}), \quad \alpha_{cutmix} = 1.0 \\
W, H &= \text{随机选择区域大小} \\
\tilde{\mathbf{x}} &= \mathbf{M} \odot \mathbf{x}_i + (1-\mathbf{M}) \odot \mathbf{x}_j \\
\tilde{y}_a &= y_i, \quad \tilde{y}_b = y_j
\end{align}
$$

其中 $\mathbf{M}$ 是二进制掩码。

### 8.4 梯度裁剪

$$
\mathbf{g}_t = \begin{cases}
\mathbf{g}_t & \text{if } \|\mathbf{g}_t\|_2 \leq \tau \\
\tau \cdot \frac{\mathbf{g}_t}{\|\mathbf{g}_t\|_2} & \text{otherwise}
\end{cases}
$$

其中 $\tau = 1.0$ 是最大梯度范数。

### 8.5 Early Stopping

监控验证集准确率，如果连续 $P=5$ 个epoch没有提升（提升阈值 $\delta = 0.001$），则停止训练。

---

## 9. 实验设置

### 9.1 数据集

- **训练集**: 669样本
- **验证集**: 168样本
- **中心数**: 5个医疗中心
- **数据划分**: Leave-Centers-Out策略

### 9.2 模型参数

- **嵌入维度**: $d = 768$
- **输入维度**: $d_{in} = 512$
- **Batch Size**: $B = 32$
- **总参数量**: 约600万

### 9.3 训练超参数

| 超参数 | 值 | 说明 |
|--------|-----|------|
| 初始学习率 | $5 \times 10^{-5}$ | 主训练阶段 |
| 预训练学习率 | $2 \times 10^{-3}$ | Student Prior预训练 |
| Weight Decay | $2 \times 10^{-3}$ | L2正则化 |
| Label Smoothing | $0.15$ | 标签平滑系数 |
| Dropout (分类器) | $0.2, 0.15$ | 两层dropout |
| Max Epochs | $50$ | 最大训练轮数 |
| Warmup Epochs | $5$ | 学习率warmup |
| Early Stopping Patience | $5$ | 早停耐心值 |
| Gradient Clipping | $1.0$ | 梯度裁剪阈值 |

### 9.4 损失权重

| 损失项 | 权重 | 动态调整 |
|--------|------|---------|
| $\lambda_{cls}$ | $3.0$ | 固定 |
| $\lambda_{ot}$ | $0.05$ | 前5个epoch从30%逐渐增加到100% |
| $\lambda_{consist}$ | $0.1$ | 前5个epoch从30%逐渐增加到100% |
| $\lambda_{adv}$ | $0.01$ | 前5个epoch从30%逐渐增加到100% |

### 9.5 评估指标

- **准确率 (Accuracy)**: $\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}$
- **AUC**: ROC曲线下面积
- **F1 Score**: $F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- **敏感性 (Sensitivity)**: $\text{Sen} = \frac{TP}{TP + FN}$
- **特异性 (Specificity)**: $\text{Spec} = \frac{TN}{TN + FP}$

---

## 10. 计算复杂度分析

### 10.1 前向传播复杂度

- **Student Prior**: $O(B \cdot (7 \cdot 256 + 256 \cdot 512 + 512 \cdot 768)) = O(B \cdot 590K)$
- **图像编码器**: $O(B \cdot (512 \cdot 1536 + 1536 \cdot 768 + 2 \cdot 768^2)) = O(B \cdot 2.4M)$
- **三模态融合**: $O(B \cdot (1536 \cdot 768)) = O(B \cdot 1.2M)$
- **分类器**: $O(B \cdot (768^2 + 768 \cdot 384 + 384 \cdot 2)) = O(B \cdot 0.9M)$
- **Sinkhorn OT**: $O(B^2 \cdot d \cdot T) = O(B^2 \cdot 768 \cdot 100)$

**总复杂度**: $O(B \cdot 5M + B^2 \cdot 76.8K)$

### 10.2 内存占用

- **模型参数**: 约600万 × 4 bytes = 24 MB
- **激活值**: $B \times d \times \text{层数} \times 4$ bytes ≈ 5-10 MB (batch_size=32)
- **优化器状态**: 约3倍参数量 = 72 MB
- **总计**: 约100-120 MB

---

## 11. 理论分析（待补充）

### 11.1 Student Prior的理论保证

**待证明**: Student Prior能够以误差 $\epsilon$ 近似VLM的语义表示：

$$
\|\mathbf{z}_{sem} - \mathbf{z}_{vlm}\|_2 \leq \epsilon
$$

### 11.2 Sinkhorn OT的收敛性

**已知**: Sinkhorn算法在熵正则化下以 $O(1/T)$ 的速率收敛到最优传输计划。

### 11.3 Memory Bank的因果解释

**待建立**: Memory Bank实现的反事实干预满足因果推理的do-operator：

$$
P(Y | do(\text{center} = c')) = P(Y | \mathbf{z}_{causal} + \alpha \cdot \mathbf{z}_{noise}^{cf}(c'))
$$

---

## 12. 实现细节

### 12.1 数值稳定性

- **特征归一化**: 所有特征在计算前进行L2归一化
- **梯度裁剪**: 限制梯度范数 ≤ 1.0
- **损失裁剪**: OT损失、一致性损失、对抗损失限制在 [0, 10.0]
- **NaN/Inf检查**: 所有中间结果检查NaN和Inf，出现时使用fallback

### 12.2 混合精度训练

使用 `torch.cuda.amp` 进行混合精度训练：

```python
with autocast():
    outputs = model(...)
    loss = compute_loss(outputs)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 附录：关键公式总结

### A.1 前向传播流程

$$
\begin{align}
\mathbf{c} &\to \mathbf{z}_{sem} = f_{prior}(\mathbf{c}) \\
\mathbf{X}_{oct}, \mathbf{X}_{colpo} &\to \mathbf{x}_{image} = \text{Fusion}(\mathbf{X}_{oct}, \mathbf{X}_{colpo}) \\
\mathbf{x}_{image} &\to \mathbf{z}_{causal}, \mathbf{z}_{noise} = f_{encoder}(\mathbf{x}_{image}) \\
\mathbf{z}_{causal}, \mathbf{z}_{sem} &\to \mathbf{z}_{fused} = \text{MultimodalFusion}(\mathbf{z}_{causal}, \mathbf{z}_{sem}) \\
\mathbf{z}_{fused} &\to \mathbf{l} = f_{classifier}(\mathbf{z}_{fused})
\end{align}
$$

### A.2 损失函数总结

$$
\mathcal{L}_{total} = 3.0 \cdot \mathcal{L}_{cls} + \lambda_{aux}(e) \cdot (0.05 \cdot \mathcal{L}_{ot} + 0.1 \cdot \mathcal{L}_{consist} + 0.01 \cdot \mathcal{L}_{adv})
$$

---

**文档结束**

