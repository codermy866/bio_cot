# Bio-COT 3.0 SCI顶刊级可视化生成指南

## 📊 可视化类型

本脚本生成以下SCI顶刊级可视化图表：

### 1. Knowledge Notes分布可视化 ✅
- **文件**: `knowledge_notes_distribution_*.png`
- **内容**:
  - (a) PCA降维后的Knowledge Notes分布（按标签着色）
  - (b) 小提琴图：按标签分组的Knowledge Notes分布
  - (c) 箱线图：按标签分组的Knowledge Notes分布
  - (d) 统计显著性检验（t-test）

### 2. Visual Notes注意力分布可视化 ✅
- **文件**: `visual_notes_attention_*.png`
- **内容**:
  - (a) OCT注意力分布（小提琴图）
  - (b) Colposcopy注意力分布（小提琴图）
  - (c) 注意力对比箱线图（按模态和标签）
  - (d) 平均注意力热图（OCT）
  - (e) 平均注意力热图（Colposcopy）
  - (f) 注意力统计（均值和标准差）

### 3. 火山图（Volcano Plot）✅
- **文件**: `volcano_plot_*.png`
- **内容**: 
  - 因果特征的差异分析
  - Fold Change vs -Log10(p-value)
  - 标记显著特征（p<0.05, |FC|>0.5）

### 4. t-SNE和UMAP可视化 ✅
- **文件**: `tsne_umap_*.png`
- **内容**:
  - (a) t-SNE by Label
  - (b) t-SNE by Center
  - (c) t-SNE by Prediction Probability
  - (d) UMAP by Label（如果可用）
  - (e) UMAP by Center（如果可用）
  - (f) UMAP by Prediction Probability（如果可用）

### 5. 综合小提琴图 ✅
- **文件**: `violin_plots_comprehensive_*.png`
- **内容**:
  - (a) 预测概率按真实标签分组
  - (b) 预测概率按中心分组
  - (c) 预测概率按中心和标签组合
  - (d) 预测概率分布直方图

### 6. CAM图（类激活映射）✅
- **文件**: `cam_samples_*.png`
- **内容**: 
  - 12个样本的Grad-CAM可视化
  - 显示模型关注的图像区域
  - 叠加在原始图像上

---

## 🚀 使用方法

### 方法1：直接运行脚本

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate
python generate_sci_visualizations.py
```

### 方法2：后台运行（推荐，因为可能需要较长时间）

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate
nohup python generate_sci_visualizations.py > vis_log.txt 2>&1 &
```

---

## 📁 输出文件

所有可视化文件保存在 `logs/` 目录下：

```
logs/
├── knowledge_notes_distribution_YYYYMMDD_HHMMSS.png
├── visual_notes_attention_YYYYMMDD_HHMMSS.png
├── volcano_plot_YYYYMMDD_HHMMSS.png
├── tsne_umap_YYYYMMDD_HHMMSS.png
├── violin_plots_comprehensive_YYYYMMDD_HHMMSS.png
└── cam_samples_YYYYMMDD_HHMMSS.png
```

---

## ⚙️ 配置说明

### 依赖包

```bash
pip install matplotlib seaborn scikit-learn pandas pillow opencv-python scipy
pip install umap-learn  # 可选，用于UMAP可视化
```

### 注意事项

1. **需要训练完成**: 脚本需要加载训练好的模型检查点
2. **GPU内存**: 特征提取需要GPU，确保有足够显存
3. **数据量**: t-SNE和UMAP在大数据集上可能较慢，建议使用验证集
4. **UMAP**: 如果未安装UMAP，将跳过UMAP可视化，只生成t-SNE

---

## 🔍 图表说明

### Knowledge Notes分布
- **用途**: 展示Knowledge Notes在不同类别间的分布差异
- **解读**: 
  - 如果正负样本的Knowledge Notes分布有明显分离，说明Knowledge Notes有效
  - PCA降维后的可视化帮助理解高维特征空间

### Visual Notes注意力
- **用途**: 展示模型关注的图像区域
- **解读**:
  - 高注意力值表示模型认为该区域重要
  - 热图显示病灶区域的定位效果

### 火山图
- **用途**: 识别显著差异的因果特征
- **解读**:
  - 右上角和左上角的点表示显著差异特征
  - 用于理解哪些特征对分类最重要

### t-SNE/UMAP
- **用途**: 可视化高维特征空间
- **解读**:
  - 同类样本聚集 → 模型学习到好的特征表示
  - 不同中心的数据分布不同 → 存在域偏移
  - 预测概率高的区域 → 模型置信度高

### 小提琴图
- **用途**: 展示预测概率的分布
- **解读**:
  - 宽度表示该值出现的频率
  - 用于分析模型在不同条件下的表现

### CAM图
- **用途**: 可视化模型关注的图像区域
- **解读**:
  - 红色区域：模型高度关注（高激活）
  - 蓝色区域：模型较少关注（低激活）
  - 用于解释模型的决策过程

---

## 📝 故障排查

### 问题1: 找不到CSV文件
**解决**: 确保数据路径正确，CSV文件位于 `data_root/internal_val/labels.csv`

### 问题2: 内存不足
**解决**: 减少batch size或使用更少的样本

### 问题3: UMAP未安装
**解决**: 安装 `pip install umap-learn` 或跳过UMAP可视化

### 问题4: 注意力数据为空
**解决**: 确保模型在训练时保存了注意力图（`return_loss_components=True`）

---

**创建时间**: 2025-01-13  
**状态**: ✅ 所有可视化功能已实现

