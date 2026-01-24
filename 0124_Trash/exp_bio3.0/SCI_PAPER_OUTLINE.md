# Bio-COT 3.0 SCI论文大纲

## 📋 论文结构

### 1. Abstract
- 背景：多中心医学图像诊断中的域偏移和因果混淆问题
- 方法：Knowledge Notes + Visual Notes + Causal Optimal Transport
- 结果：AUC 0.85，相比基线提升显著
- 贡献：首次将RAG思想引入医学图像诊断，结合知识引导的视觉注意力

### 2. Introduction
- 2.1 研究背景
- 2.2 问题陈述
- 2.3 主要贡献
- 2.4 论文组织

### 3. Related Work
- 3.1 多模态医学图像诊断
- 3.2 域适应与因果学习
- 3.3 知识增强的视觉模型
- 3.4 最优传输在医学中的应用

### 4. Method ✅ (已完成)
- 4.1 Overview
- 4.2 Knowledge Notes Generation
  - 4.2.1 Medical Knowledge Retrieval
  - 4.2.2 Diagnostic Summary Generation
  - 4.2.3 Semantic Anchor Extraction
- 4.3 Visual Notes Generation
  - 4.3.1 Cross-Modal Attention Computation
  - 4.3.2 Soft Attention Masking
  - 4.3.3 Visual Note Filtering
- 4.4 Causal Feature Decoupling
  - 4.4.1 Dual-Head Image Encoder
  - 4.4.2 Cross-Modal Fusion
  - 4.4.3 Classification
- 4.5 Loss Functions
  - 4.5.1 Total Loss
  - 4.5.2 Classification Loss
  - 4.5.3 Sinkhorn Optimal Transport Loss
  - 4.5.4 Counterfactual Consistency Loss
  - 4.5.5 Adversarial Loss
  - 4.5.6 Sparse Attention Loss

### 5. Experiments ✅ (已完成)
- 5.1 Dataset
  - 5.1.1 Data Description
  - 5.1.2 Data Preprocessing
  - 5.1.3 Evaluation Metrics
- 5.2 Implementation Details
  - 5.2.1 Model Architecture
  - 5.2.2 Training Configuration
  - 5.2.3 Baseline Methods
- 5.3 Results
  - 5.3.1 Main Results
  - 5.3.2 Ablation Studies
  - 5.3.3 Analysis and Discussion

### 6. Conclusion
- 6.1 总结
- 6.2 贡献
- 6.3 局限性
- 6.4 未来工作

---

## 📊 关键公式索引

### Knowledge Notes
- 知识检索: $\mathcal{K}_{retrieved} = \text{Retrieve}(\mathbf{c}, \mathcal{K}, k)$
- 语义锚点: $\mathbf{z}_{sem} = \text{TextProjector}(\text{LLM}(\mathbf{s}))$

### Visual Notes
- 注意力计算: $\mathbf{A}_{logits} = \frac{\mathbf{K} \mathbf{Q}^T}{\sqrt{h}}$
- 软掩码: $\mathbf{A} = \text{clamp}(\sigma(\mathbf{A}_{logits}), \min=0.05, \max=1.0)$
- 特征过滤: $\mathbf{F}_{note} = \mathbf{F}_{img} \odot (\mathbf{A} + (1 - \mathbf{A}) \cdot \beta)$
- 动态Beta: $\beta(t) = \begin{cases} 1.0 & t < 10 \\ 1.0 - 0.7 \cdot \frac{t-10}{20} & 10 \leq t < 30 \\ 0.3 & t \geq 30 \end{cases}$

### Causal Decoupling
- 双头编码: $[\mathbf{z}_{causal}, \mathbf{z}_{noise}] = \text{DualHead}(\mathbf{F}_{note})$
- 跨模态融合: $\mathbf{f}_{fused} = \text{CrossAttention}(\mathbf{z}_{causal}, \mathbf{z}_{sem})$

### Loss Functions
- 总损失: $\mathcal{L}_{total} = \lambda_{cls} \mathcal{L}_{cls} + \lambda_{ot} \mathcal{L}_{ot} + \lambda_{consist} \mathcal{L}_{consist} + \lambda_{adv} \mathcal{L}_{adv} + \lambda_{sparse} \mathcal{L}_{sparse}$
- Focal Loss: $\mathcal{L}_{cls} = -\frac{1}{B}\sum_{i=1}^{B} \alpha_{y_i} (1-p_{i,y_i})^{\gamma} \log(p_{i,y_i})$
- Sinkhorn OT: $\mathcal{L}_{ot} = \min_{\mathbf{P} \in \mathcal{U}(\mathbf{a}, \mathbf{b})} \langle \mathbf{P}, \mathbf{C} \rangle - \epsilon H(\mathbf{P})$
- 一致性损失: $\mathcal{L}_{consist} = \frac{1}{B}\sum_{i=1}^{B} \|\hat{\mathbf{y}}^{(i)} - \hat{\mathbf{y}}_{cf}^{(i)}\|_2^2$
- 稀疏损失: $\mathcal{L}_{sparse} = \sum_{m} \left[\alpha_{ent} \cdot \mathcal{H}(\bar{\mathbf{A}}_m) + \alpha_{L1} \cdot \|\mathbf{A}_m\|_1\right]$

---

## 📈 实验结果总结

### 主要结果
- **AUC**: 0.85 (vs. 0.84 for Bio-COT 2.0)
- **Accuracy**: 0.79 (vs. 0.78 for Bio-COT 2.0)
- **F1-Score**: 0.69 (vs. 0.68 for Bio-COT 2.0)

### 消融实验
1. Knowledge Notes: +0.02 AUC
2. Visual Notes: +0.02 AUC
3. OT Loss: +0.03 AUC
4. Consistency Loss: +0.01 AUC
5. Sparse Loss: +0.01 AUC

---

## 🎯 创新点总结

1. **Knowledge Notes**: 首次将RAG思想引入医学图像诊断，结合外部医学知识库
2. **Visual Notes**: 知识引导的视觉注意力，显式聚焦病灶区域
3. **动态Beta策略**: Warm-up机制，逐步引入视觉过滤
4. **稀疏注意力损失**: 熵损失 + L1损失，鼓励局部化聚焦

---

**文档状态**: ✅ Method和Experiments部分已完成  
**下一步**: 补充Introduction、Related Work、Conclusion部分

