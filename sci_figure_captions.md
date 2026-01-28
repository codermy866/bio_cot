# SCI论文图表Caption - 学术级别解释

## 1. Scatterplot Matrix (特征散点图矩阵)

### 图表内容
- **布局**: 4×4矩阵，展示4个关键特征之间的成对关系
- **对角线**: KDE密度分布图，展示每个特征的分布特征
- **非对角线**: 散点图，展示特征对之间的相关性
- **标签区分**: 使用Negative/Positive颜色区分不同类别
- **相关系数**: 每个散点图标注Pearson相关系数

### SCI学术Caption (Figure Legend)

**Figure X. Scatterplot matrix of key features.** Pairwise relationships among four selected features with the highest discriminative power for class separation are displayed. Features were selected based on t-test statistics (p < 0.05). Diagonal panels show kernel density estimation (KDE) curves representing the univariate probability density distribution of each feature. Off-diagonal panels illustrate bivariate scatter plots with individual data points color-coded by class (Negative: blue-gray, Positive: red-brown). Pearson correlation coefficients (r) are annotated in the upper right corner of each scatter plot, with significance levels indicated by asterisks (* p < 0.05, ** p < 0.01, *** p < 0.001). This visualization enables comprehensive assessment of multivariate relationships and distribution characteristics within the feature space, facilitating identification of feature interactions, redundancy, and class-separability patterns.

### 学术解释要点
- **特征选择**: 基于标签分离度（t检验）选择最重要的4个特征
- **相关性分析**: 展示特征间的线性关系
- **分布特征**: KDE曲线展示每个特征的分布形状
- **类别分离**: 通过颜色区分，直观展示不同类别在特征空间中的分布

---

## 2. Conditional Means with Observations (条件均值与观测值)

### 图表内容
- **布局**: 2×2子图，展示4个特征的条件均值
- **数据点**: 原始观测值散点图（按中心分组，不同颜色）
- **条件均值**: 每个中心的均值±标准差（误差棒）
- **连接线**: 虚线连接各中心的均值点
- **中心名称**: 使用具体医院名称（Enshi, Wuda, Xiangyang, Enshi Wuda, Shiyan Jingzhou）

### SCI学术Caption (Figure Legend)

**Figure X. Conditional means of features across medical centers.** Distribution of four key features stratified by medical center is displayed. Individual observations are shown as scatter points with jittering to prevent overplotting. Center-specific means ± standard deviations are represented by error bars, and dashed lines connect mean values across centers to facilitate visual comparison. Each center is color-coded using a gradient palette (red-brown to blue-gray) for visual distinction. The analysis includes data from five medical centers: Enshi, Wuda, Xiangyang, Enshi Wuda, and Shiyan Jingzhou (n=120 per center). Statistical comparisons between centers were performed using analysis of variance (ANOVA) or Kruskal-Wallis test, as appropriate based on distributional assumptions. This visualization enables assessment of center-specific variations and potential data heterogeneity across different medical institutions, which is crucial for evaluating model generalizability across diverse clinical settings and identifying potential sources of bias in multi-center studies.

### 学术解释要点
- **多中心比较**: 展示不同医疗中心间的特征差异
- **数据异质性**: 评估中心间数据分布的差异
- **可泛化性**: 支持评估模型在不同临床环境中的表现
- **统计信息**: 显示均值和标准差，提供变异性信息

---

## 3. Large Distributions (大规模分布可视化)

### 图表内容
- **布局**: 2×2子图，展示4个特征的分布
- **直方图**: 频率分布（灰色柱状图）
- **KDE曲线**: 平滑的概率密度估计（红色曲线）
- **统计检验**: 正态性检验结果（如果按标签分组，包含组间比较）

### SCI学术Caption (Figure Legend)

**Figure X. Distribution characteristics of key features.** Probability density distributions of four selected features are illustrated using histograms (gray bars, 50 bins) and kernel density estimation (KDE) curves (red lines). Histograms represent frequency distributions normalized to density, while KDE curves provide smooth, non-parametric estimates of the underlying probability density functions using Gaussian kernels. The visualization facilitates assessment of distributional properties, including normality, skewness, kurtosis, and potential outliers. Distribution characteristics were evaluated using the Shapiro-Wilk test for normality (n ≤ 5000) or D'Agostino's K-squared test for larger samples. For features stratified by class, group comparisons were performed using independent t-tests (for normally distributed data) or Mann-Whitney U tests (for non-normal distributions). Features exhibiting non-normal distributions (p < 0.05) were subsequently analyzed using non-parametric statistical methods. This exploratory analysis is essential for selecting appropriate statistical tests, understanding data structure, and validating distributional assumptions before model development.

### 学术解释要点
- **分布特征**: 展示数据的分布形状（正态/非正态、偏态方向）
- **统计方法选择**: 为选择参数或非参数检验提供依据
- **数据质量**: 识别异常值和分布异常
- **方法学验证**: 支持统计方法选择的合理性

---

## 4. Linear Regression with Marginal Distributions (带边缘分布的线性回归)

### 图表内容
- **主图**: 两个特征之间的散点图，包含线性回归拟合线和95%置信区间
- **上边缘**: X轴特征的分布（直方图 + KDE曲线）
- **右边缘**: Y轴特征的分布（直方图 + KDE曲线）
- **标签区分**: 使用Negative/Positive颜色区分
- **统计信息**: R²、p值显示在标题中

### SCI学术Caption (Figure Legend)

**Figure X. Linear regression analysis with marginal distributions.** The central panel displays a bivariate scatter plot of two features exhibiting the highest absolute correlation coefficient, overlaid with a linear regression fit (dashed red line) and 95% confidence interval (shaded region). Data points are color-coded by class (Negative: blue-gray, Positive: red-brown) to visualize class-specific patterns within the bivariate relationship. The top panel shows the marginal distribution of the x-axis feature, while the right panel displays the marginal distribution of the y-axis feature. Both marginal distributions are represented by histograms (gray bars, 40 bins) normalized to density and KDE curves (red lines) using Gaussian kernels. The regression model's goodness-of-fit is quantified by the coefficient of determination (R²) and statistical significance (p-value), both displayed in the title. The 95% confidence interval for the regression line was calculated using standard errors of the predicted values. This comprehensive visualization enables simultaneous assessment of the linear relationship between features, their individual distributional properties, and potential class-specific differences, providing a holistic view of the bivariate data structure and facilitating identification of feature interactions relevant to classification.

### 学术解释要点
- **线性关系**: 展示两个特征间的线性相关性
- **回归拟合**: 提供线性回归模型和置信区间
- **边缘分布**: 展示每个特征的单独分布特征
- **统计显著性**: R²和p值评估关系的强度和显著性

---

## 通用学术写作建议

### Caption结构
1. **标题**: 简洁描述图表内容
2. **方法描述**: 说明使用的可视化方法
3. **数据说明**: 样本量、分组信息
4. **统计信息**: 关键统计检验结果
5. **结果解释**: 图表揭示的主要发现

### 语言风格
- 使用被动语态（"Data are displayed..."）
- 使用专业术语（"kernel density estimation", "marginal distributions"）
- 避免口语化表达
- 保持客观、准确的描述

### 在论文中的位置
- **Scatterplot Matrix**: 结果部分（特征关系分析）
- **Conditional Means**: 结果部分（多中心比较）
- **Large Distributions**: 方法学部分（数据描述）或结果部分（数据特征）
- **Linear Regression**: 结果部分（特征相关性分析）

---

**生成时间**: 2025-01-26  
**适用期刊**: Nature, Science, Cell, Nature Medicine, JAMA, NEJM等顶级期刊

