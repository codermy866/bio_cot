# SCI论文图表Caption - Conditional_Means & Scatterplot_Matrix

## Figure X: Conditional Means with Observations

**Figure X. Conditional means of key features across medical centers.** The panels display the distribution of four selected features stratified by medical center, with individual observations shown as scatter points (jittered to prevent overplotting) and center-specific means ± standard deviations represented by error bars. The dashed lines connect mean values across centers to facilitate visual comparison. Each center is color-coded using a gradient palette (deep red-brown to deep blue-gray) for visual distinction. The analysis includes data from five medical centers: Enshi, Wuda, Xiangyang, Jingzhou, and Shiyan (n=120 per center). Statistical comparisons between centers were performed using analysis of variance (ANOVA) for normally distributed data or Kruskal-Wallis test for non-normal distributions, as appropriate based on distributional assumptions assessed by Shapiro-Wilk or D'Agostino's K-squared tests. This visualization enables assessment of center-specific variations and potential data heterogeneity across different medical institutions, which is crucial for evaluating model generalizability across diverse clinical settings and identifying potential sources of bias in multi-center studies. The observed variations reflect both biological differences and potential center-specific imaging protocols or patient populations, highlighting the importance of robust multi-center validation in medical AI applications.

---

## Figure Y: Scatterplot Matrix

**Figure Y. Scatterplot matrix of key features with highest discriminative power.** Pairwise relationships among four selected features that demonstrated the highest discriminative power for class separation are displayed. Features were selected based on independent t-test statistics comparing negative and positive cases (p < 0.05). Diagonal panels show kernel density estimation (KDE) curves representing the univariate probability density distribution of each feature, with filled areas indicating the probability density. Off-diagonal panels illustrate bivariate scatter plots with individual data points color-coded by class (Negative: deep blue-gray, Positive: deep red-brown). Pearson correlation coefficients (r) are annotated in the upper right corner of each scatter plot, with only statistically significant correlations (|r| > 0.2 or p < 0.05) prominently displayed. This visualization enables comprehensive assessment of multivariate relationships and distribution characteristics within the learned feature space, facilitating identification of feature interactions, potential redundancy, and class-separability patterns. The analysis demonstrates that the learned features exhibit strong discriminative power and reveal meaningful relationships between different feature modalities (OCT texture and intensity, Colposcopy color and vascular patterns, and Clinical parameters), supporting the model's ability to capture clinically relevant patterns for cervical lesion classification.

---

## 学术写作要点

### Conditional_Means Caption要点：
1. **多中心验证**：强调5个医疗中心的数据
2. **统计方法**：明确说明使用的统计检验（ANOVA/Kruskal-Wallis）
3. **临床意义**：强调可泛化性和数据异质性评估
4. **样本量**：明确标注每个中心的样本数（n=120）

### Scatterplot_Matrix Caption要点：
1. **特征选择**：说明基于t-test的特征选择方法
2. **可视化方法**：详细描述KDE和散点图
3. **统计信息**：说明相关系数的标注标准
4. **临床相关性**：强调特征与临床模态的对应关系

---

## 在论文中的使用建议

### Conditional_Means：
- **位置**：Results部分，多中心验证章节
- **作用**：展示模型在不同医疗中心的表现一致性
- **配合**：可与混淆矩阵、ROC曲线一起展示，证明模型的可泛化性

### Scatterplot_Matrix：
- **位置**：Results部分，特征分析章节
- **作用**：展示学习到的特征的质量和判别能力
- **配合**：可与t-SNE/UMAP一起展示，从不同角度分析特征空间

---

**生成时间**: 2025-01-27  
**适用期刊**: Nature, Science, Cell, Nature Medicine, JAMA, NEJM, MICCAI等顶级期刊和会议

