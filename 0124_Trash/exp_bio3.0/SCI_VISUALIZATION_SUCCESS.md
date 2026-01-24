# ✅ SCI顶刊级可视化生成成功！

## 📊 已生成的可视化图表

所有图表已成功生成，保存在 `logs/` 目录下（时间戳：20260113_144429）

### 1. Knowledge Notes分布图 ✅
**文件**: `knowledge_notes_distribution_20260113_144429.png`

**内容**:
- (a) PCA降维后的Knowledge Notes分布（按标签着色）
- (b) 小提琴图：按标签分组的Knowledge Notes分布
- (c) 箱线图：按标签分组的Knowledge Notes分布
- (d) 统计显著性检验（t-test）

**解读**: 展示Knowledge Notes在不同类别间的分布差异，验证Knowledge Notes的有效性。

---

### 2. Visual Notes注意力分布图 ✅
**文件**: `visual_notes_attention_20260113_144429.png`

**内容**:
- (a) OCT注意力分布（小提琴图）
- (b) Colposcopy注意力分布（小提琴图）
- (c) 注意力对比箱线图（按模态和标签）
- (d) 平均注意力热图（OCT，14×14 patch grid）
- (e) 平均注意力热图（Colposcopy，14×14 patch grid）
- (f) 注意力统计（均值和标准差）

**解读**: 展示模型关注的图像区域，验证Visual Notes的病灶定位效果。

---

### 3. 火山图（Volcano Plot）✅
**文件**: `volcano_plot_20260113_144429.png`

**内容**: 
- 因果特征的差异分析
- Fold Change vs -Log10(p-value)
- 标记显著特征（p<0.05, |FC|>0.5）

**解读**: 识别显著差异的因果特征，理解哪些特征对分类最重要。

---

### 4. t-SNE和UMAP可视化 ✅
**文件**: `tsne_umap_20260113_144429.png`

**内容**:
- (a) t-SNE by Label
- (b) t-SNE by Center
- (c) t-SNE by Prediction Probability
- (d) UMAP by Label
- (e) UMAP by Center
- (f) UMAP by Prediction Probability

**解读**: 可视化高维特征空间，展示同类样本聚集、域偏移、模型置信度等信息。

---

### 5. 综合小提琴图 ✅
**文件**: `violin_plots_comprehensive_20260113_144429.png`

**内容**:
- (a) 预测概率按真实标签分组
- (b) 预测概率按中心分组
- (c) 预测概率按中心和标签组合
- (d) 预测概率分布直方图

**解读**: 展示预测概率的分布，分析模型在不同条件下的表现。

---

### 6. CAM图（类激活映射）✅
**文件**: `cam_samples_20260113_144429.png`

**内容**: 
- 12个样本的Grad-CAM可视化
- 显示模型关注的图像区域
- 叠加在原始图像上

**解读**: 可视化模型关注的图像区域，红色区域表示高激活（模型关注的重点），用于解释模型的决策过程。

---

## 🎨 图表特点

1. **SCI顶刊级质量**:
   - 300 DPI高分辨率
   - Times New Roman字体
   - 专业配色方案
   - 清晰的图例和标签

2. **信息完整**:
   - 包含统计检验
   - 显著性标记
   - 置信区间
   - 子图标题和说明

3. **可直接用于论文**:
   - 符合Nature/Science等顶刊要求
   - 图表清晰、专业
   - 信息密度适中

---

## 📁 文件位置

所有文件位于：
```
/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/logs/
```

---

## 🔄 重新生成

如果需要重新生成可视化，运行：

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate
python generate_sci_visualizations.py
```

---

**生成时间**: 2025-01-13 14:44:29  
**状态**: ✅ 所有可视化成功生成

