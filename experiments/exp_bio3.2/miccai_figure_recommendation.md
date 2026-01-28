# MICCAI论文图表推荐方案

## 📊 图表冲突分析

### ⚠️ 存在信息重复的图表对：

1. **Scatterplot Matrix** vs **Linear Regression Marginal**
   - **冲突点**：都展示特征相关性
   - **建议**：二选一，推荐Scatterplot Matrix（信息更全面）

2. **Large Distributions** vs **Scatterplot Matrix的对角线**
   - **冲突点**：都展示特征分布
   - **建议**：如果选Scatterplot Matrix，Large Distributions可省略

3. **Conditional Means** vs **t-SNE/UMAP**
   - **冲突点**：都展示多中心差异
   - **建议**：两者互补，Conditional Means展示统计差异，t-SNE/UMAP展示空间分布

---

## 🎯 MICCAI论文推荐方案

### **方案A：核心性能展示（推荐）⭐⭐⭐**

**适用场景**：强调模型性能和多中心验证

| 图表 | 位置 | 作用 | 优先级 |
|------|------|------|--------|
| **Confusion Matrix** | Results主图 | 核心分类性能指标 | ⭐⭐⭐ 必须 |
| **ROC Curve** | Results主图 | 分类性能对比 | ⭐⭐⭐ 必须 |
| **Conditional Means** | Results/Supplementary | 多中心验证，展示泛化性 | ⭐⭐ 强烈推荐 |
| **t-SNE/UMAP** | Supplementary | 特征空间可视化 | ⭐ 可选 |

**优势**：
- ✅ 核心性能指标完整（混淆矩阵+ROC）
- ✅ 多中心验证充分（Conditional Means）
- ✅ 避免信息重复
- ✅ 符合MICCAI审稿人关注点

**Caption建议**：
- Figure 1: Confusion Matrix + ROC Curve（主图）
- Figure 2: Conditional Means across Centers（多中心验证）
- Supplementary Figure 1: t-SNE/UMAP visualization（特征空间）

---

### **方案B：完整特征分析（适合方法创新强的论文）⭐⭐**

**适用场景**：强调特征工程和可解释性

| 图表 | 位置 | 作用 | 优先级 |
|------|------|------|--------|
| **Confusion Matrix** | Results主图 | 核心分类性能指标 | ⭐⭐⭐ 必须 |
| **ROC Curve** | Results主图 | 分类性能对比 | ⭐⭐⭐ 必须 |
| **Scatterplot Matrix** | Results | 特征关系分析 | ⭐⭐ 推荐 |
| **Conditional Means** | Results | 多中心验证 | ⭐⭐ 推荐 |

**优势**：
- ✅ 特征分析深入（Scatterplot Matrix）
- ✅ 多中心验证完整
- ✅ 展示方法学严谨性

**注意**：
- ⚠️ 如果选择Scatterplot Matrix，**不要**同时使用Linear Regression Marginal（避免重复）

---

### **方案C：精简版（适合页数限制）⭐**

**适用场景**：论文页数紧张，只展示核心结果

| 图表 | 位置 | 作用 | 优先级 |
|------|------|------|--------|
| **Confusion Matrix** | Results主图 | 核心分类性能指标 | ⭐⭐⭐ 必须 |
| **ROC Curve** | Results主图 | 分类性能对比 | ⭐⭐⭐ 必须 |
| **Conditional Means** | Results | 多中心验证 | ⭐⭐ 推荐 |

**优势**：
- ✅ 最精简，信息密度高
- ✅ 满足MICCAI基本要求

---

## 🚫 不推荐的组合（避免冲突）

### ❌ 组合1：Scatterplot Matrix + Linear Regression Marginal
**问题**：两者都展示特征相关性，信息重复
**建议**：只选Scatterplot Matrix（信息更全面）

### ❌ 组合2：Large Distributions + Scatterplot Matrix
**问题**：Large Distributions的分布信息在Scatterplot Matrix对角线已有
**建议**：如果选Scatterplot Matrix，Large Distributions可省略

### ❌ 组合3：只有特征分析图，没有性能指标
**问题**：MICCAI审稿人最关心性能，必须有混淆矩阵和ROC
**建议**：至少包含Confusion Matrix + ROC Curve

---

## 📝 具体推荐（基于你的图表）

### **最推荐方案：方案A（核心性能展示）**

**主图（Results部分）**：
1. **Figure 1: Model Performance**
   - (a) Confusion Matrix
   - (b) ROC Curve
   - **Caption**: "Performance evaluation of Bio-COT 3.2. (a) Confusion matrix showing classification results on the test set. (b) ROC curve demonstrating the model's discriminative ability (AUC = 0.8827)."

2. **Figure 2: Multi-center Validation**
   - Conditional Means with Observations
   - **Caption**: "Feature distributions across five medical centers. The visualization demonstrates center-specific variations and validates model generalizability across diverse clinical settings."

**Supplementary Materials**：
3. **Supplementary Figure 1: Feature Space Visualization**
   - t-SNE或UMAP（二选一）
   - **Caption**: "3D visualization of the learned feature space using [t-SNE/UMAP]. Data points are color-coded by class (Negative: blue-gray, Positive: red-brown) and medical center."

---

## 🎯 MICCAI审稿人关注点

根据MICCAI审稿标准，审稿人最关注：

1. **性能指标**（必须）
   - ✅ Confusion Matrix
   - ✅ ROC Curve
   - ✅ 数值指标（Accuracy, Precision, Recall, F1, AUC）

2. **多中心验证**（强烈推荐）
   - ✅ Conditional Means（展示中心间差异）
   - ✅ 外部验证集结果

3. **可解释性**（加分项）
   - ✅ 特征分析（Scatterplot Matrix或Linear Regression）
   - ✅ 特征空间可视化（t-SNE/UMAP）

4. **方法学严谨性**（加分项）
   - ✅ 统计检验（ANOVA/Kruskal-Wallis）
   - ✅ 分布特征分析

---

## 💡 最终建议

**推荐使用方案A**，包含：
1. **Confusion Matrix**（必须）
2. **ROC Curve**（必须）
3. **Conditional Means**（强烈推荐，展示多中心验证）
4. **t-SNE或UMAP**（可选，Supplementary Materials）

**避免使用**：
- ❌ Linear Regression Marginal（与Scatterplot Matrix重复）
- ❌ Large Distributions（与Scatterplot Matrix对角线重复）
- ❌ 同时使用Scatterplot Matrix和Linear Regression Marginal

**理由**：
- ✅ 满足MICCAI审稿人核心关注点（性能+多中心验证）
- ✅ 避免信息重复，图表含义清晰
- ✅ 符合MICCAI论文常见图表配置
- ✅ 页数控制合理（2-3个主图 + 1个Supplementary）

---

## 📋 图表Caption模板（可直接使用）

### Figure 1: Model Performance
**Figure 1. Performance evaluation of Bio-COT 3.2.** (a) Confusion matrix showing classification results on the independent test set (n=168). The model achieved an accuracy of 79.76% with high specificity (91.15%) and moderate sensitivity (56.36%). (b) Receiver operating characteristic (ROC) curve demonstrating the model's discriminative ability. The area under the curve (AUC) was 0.8827, indicating excellent classification performance.

### Figure 2: Multi-center Validation
**Figure 2. Feature distributions across medical centers.** Distribution of four key features stratified by medical center is displayed. Individual observations are shown as scatter points with jittering to prevent overplotting. Center-specific means ± standard deviations are represented by error bars, and dashed lines connect mean values across centers to facilitate visual comparison. The analysis includes data from five medical centers: Enshi, Wuda, Xiangyang, Enshi Wuda, and Shiyan Jingzhou (n=120 per center). Statistical comparisons between centers were performed using analysis of variance (ANOVA) or Kruskal-Wallis test, as appropriate. This visualization demonstrates center-specific variations and validates model generalizability across diverse clinical settings.

### Supplementary Figure 1: Feature Space Visualization
**Supplementary Figure 1. Feature space visualization using [t-SNE/UMAP].** The learned feature representations are projected into a 3D space for visualization. Data points are color-coded by class (Negative: blue-gray, Positive: red-brown) and medical center (gradient palette). The visualization reveals the separability of different classes in the learned feature space and demonstrates the model's ability to learn discriminative representations.

---

**生成时间**: 2025-01-26  
**适用会议**: MICCAI 2025

