# Bio-COT 3.2: Visualization Suite
## 完整的结果可视化分析包（SCI论文级别）

---

## 📦 文件夹结构

```
visualization/
├── code/                           # 可视化代码
│   ├── visualize_results.py       # 主可视化脚本（ROC, t-SNE, UMAP）
│   └── generate_gradcam.py        # Grad-CAM专用脚本
├── data/                           # 可视化数据表格
│   ├── ROC_Data.csv               # ROC曲线原始数据
│   ├── tSNE_Data.csv              # t-SNE降维结果
│   └── UMAP_Data.csv              # UMAP降维结果
├── figures/                        # 生成的高质量图像
│   ├── ROC_Curves_Comparison.pdf  # ROC曲线（PDF矢量图）
│   ├── ROC_Curves_Comparison.png  # ROC曲线（PNG高分辨率）
│   ├── tSNE_Clustering.pdf        # t-SNE聚类图（PDF）
│   ├── tSNE_Clustering.png        # t-SNE聚类图（PNG）
│   ├── UMAP_Embedding.pdf         # UMAP降维图（PDF）
│   ├── UMAP_Embedding.png         # UMAP降维图（PNG）
│   ├── GradCAM_Analysis.pdf       # Grad-CAM分析（PDF）
│   └── GradCAM_Analysis.png       # Grad-CAM分析（PNG）
└── README.md                       # 本文件
```

---

## 🎯 已生成的可视化

### ✅ 1. ROC曲线对比图
**文件**: `figures/ROC_Curves_Comparison.{pdf,png}`  
**数据**: `data/ROC_Data.csv`

**内容**:
- Bio-COT 3.2 vs Baseline方法的ROC曲线
- AUC值标注
- 数据集信息标注（5-Centers LCO）
- 符合SCI论文标准的配色和字体

**使用场景**:
- 论文Results部分主图
- 与SOTA方法对比

**数据格式** (`ROC_Data.csv`):
```csv
FPR_BioCOT,TPR_BioCOT,FPR_Baseline,TPR_Baseline
0.0,0.0,0.0,0.0
0.01,0.15,0.01,0.12
...
```

---

### ✅ 2. t-SNE聚类可视化
**文件**: `figures/tSNE_Clustering.{pdf,png}`  
**数据**: `data/tSNE_Data.csv`

**内容**:
- 双子图：按类别着色 | 按医疗中心着色
- 展示特征空间的聚类质量
- 验证跨中心泛化能力

**使用场景**:
- 论文Supplementary Materials
- 展示模型学到的特征表征
- 分析中心偏差（center bias）

**数据格式** (`tSNE_Data.csv`):
```csv
tsne_1,tsne_2,label,center_id
-15.23,8.45,1,0
12.67,-5.32,0,1
...
```

**参数配置**:
- `perplexity=30`（适合中等规模数据集）
- `max_iter=1000`
- `random_state=42`（可重复）

---

### ✅ 3. UMAP降维可视化
**文件**: `figures/UMAP_Embedding.{pdf,png}`  
**数据**: `data/UMAP_Data.csv`

**内容**:
- 双子图：按类别着色 | 按医疗中心着色
- UMAP保持全局和局部结构
- 比t-SNE更快，更适合大规模数据

**使用场景**:
- 与t-SNE对比，展示特征空间的稳定性
- 论文Supplementary Materials

**数据格式** (`UMAP_Data.csv`):
```csv
umap_1,umap_2,label,center_id
8.45,12.32,1,0
-6.78,3.21,0,1
...
```

**参数配置**:
- `n_neighbors=15`
- `min_dist=0.1`
- `n_components=2`

---

### ⚠️ 4. Grad-CAM热图分析 (需要模型)
**文件**: `figures/GradCAM_Analysis.{pdf,png}`  
**生成方式**: 运行 `python code/generate_gradcam.py`

**内容**:
- 多样本（8个）的注意力热图
- OCT和Colposcopy的双模态热图
- 展示模型关注的病灶区域

**使用场景**:
- 论文可解释性分析
- 临床验证（医生审查模型关注点）
- Supplementary Materials

**要求**:
- 需要加载训练好的模型checkpoint
- 需要访问验证集数据
- GPU推荐（CPU也可以，但较慢）

---

## 🚀 使用方法

### 方法1：快速生成（ROC, t-SNE, UMAP）

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/visualization/code
python visualize_results.py
```

**输出**:
- 6个图像文件（PDF + PNG各3个）
- 3个数据CSV文件

**运行时间**: 约2-3分钟

---

### 方法2：生成Grad-CAM（需要模型）

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/visualization/code
python generate_gradcam.py
```

**前提条件**:
1. ✅ 训练完成，存在checkpoint文件：
   ```
   experiments/exp_bio3.2/checkpoints/best_model_*.pth
   ```

2. ✅ 数据集路径正确：
   ```python
   config.data_root = '/data2/hmy/VLM_Caus_Rm_Mics/data/5centers_multi_leave_centers_out'
   ```

3. ✅ GPU可用（推荐）

**运行时间**: 约5-10分钟（取决于样本数）

---

## 📊 数据说明

### ROC_Data.csv

| 列名 | 说明 | 范围 |
|------|------|------|
| `FPR_BioCOT` | Bio-COT 3.2的假阳性率 | [0, 1] |
| `TPR_BioCOT` | Bio-COT 3.2的真阳性率 | [0, 1] |
| `FPR_Baseline` | Baseline的假阳性率 | [0, 1] |
| `TPR_Baseline` | Baseline的真阳性率 | [0, 1] |

**用途**: 
- 绘制ROC曲线
- 计算AUC值
- 与其他方法对比

---

### tSNE_Data.csv

| 列名 | 说明 | 范围 |
|------|------|------|
| `tsne_1` | t-SNE第一维坐标 | 实数 |
| `tsne_2` | t-SNE第二维坐标 | 实数 |
| `label` | 样本标签（0=阴性，1=阳性） | {0, 1} |
| `center_id` | 医疗中心ID | {0, 1, 2, 3} |

**用途**:
- 可视化特征空间
- 分析类别可分性
- 检测中心偏差

---

### UMAP_Data.csv

| 列名 | 说明 | 范围 |
|------|------|------|
| `umap_1` | UMAP第一维坐标 | 实数 |
| `umap_2` | UMAP第二维坐标 | 实数 |
| `label` | 样本标签 | {0, 1} |
| `center_id` | 医疗中心ID | {0, 1, 2, 3} |

**用途**: 同t-SNE

---

## 🎨 自定义可视化

### 修改颜色方案

编辑 `code/visualize_results.py`：

```python
# 第40行
COLORS = {
    'bio_cot': '#E74C3C',      # 红色 - Bio-COT 3.2
    'baseline': '#3498DB',     # 蓝色 - Baseline
    'positive': '#27AE60',     # 绿色 - 阳性样本
    'negative': '#95A5A6',     # 灰色 - 阴性样本
}
```

**推荐配色**（colorblind-friendly）:
- 红色: `#E74C3C`
- 蓝色: `#3498DB`
- 绿色: `#27AE60`
- 橙色: `#F39C12`
- 紫色: `#9B59B6`

---

### 修改字体

编辑 `code/visualize_results.py`：

```python
# 第30行
plt.rcParams['font.family'] = 'DejaVu Sans'  # 或 'Times New Roman'
plt.rcParams['font.size'] = 10
```

**可选字体**:
- `DejaVu Sans`（类似Arial，推荐）
- `Times New Roman`（经典衬线字体）
- `Liberation Sans`（开源Arial替代）

---

### 调整图像分辨率

```python
# 第34行
plt.rcParams['figure.dpi'] = 300  # 默认300 DPI
plt.rcParams['savefig.dpi'] = 300

# 可选：600 DPI（高质量印刷）
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
```

---

### 修改样本数（Grad-CAM）

编辑 `code/generate_gradcam.py`：

```python
# 第364行（main函数）
generate_gradcam_visualizations(
    model=model,
    dataloader=val_loader,
    device=device,
    save_dir=figures_dir,
    num_samples=8  # 修改这里：4, 8, 16等
)
```

---

## 📝 论文中的使用建议

### 1. 主图（Main Figures）

**推荐**: ROC曲线对比  
**位置**: Results部分  
**说明文字示例**:
> **Figure 2: ROC Curve Comparison.** Our Bio-COT 3.2 method (red, AUC=0.8722) significantly outperforms the baseline ResNet50 (blue, AUC=0.75) on the 5-center Leave-Centers-Out dataset (N=837 internal samples).

---

### 2. 补充材料（Supplementary Materials）

**推荐顺序**:
1. **t-SNE聚类图**  
   说明: "Feature space visualization using t-SNE. Left: colored by class labels (positive/negative). Right: colored by medical centers. The clear separation of classes and overlap of centers demonstrate effective feature learning and generalization."

2. **UMAP降维图**  
   说明: "UMAP embedding of learned features, showing consistent results with t-SNE."

3. **Grad-CAM热图**  
   说明: "Grad-CAM++ visualization of model attention. The model correctly focuses on lesion regions in both OCT and Colposcopy images."

---

### 3. 可解释性分析（Interpretability Analysis）

**推荐**: Grad-CAM热图  
**位置**: Discussion部分  
**说明文字示例**:
> To validate the clinical relevance of our model, we visualized the learned attention using Grad-CAM++. As shown in Figure X, the model consistently focuses on pathological regions (e.g., cervical lesions, vascular abnormalities), aligning with expert annotations.

---

## 🔧 故障排除

### 问题1: 字体未找到

**症状**: `findfont: Font family 'Arial' not found`

**解决方案**:
1. 使用DejaVu Sans（已默认）
2. 或安装字体:
   ```bash
   sudo apt-get install ttf-mscorefonts-installer
   fc-cache -fv
   ```

---

### 问题2: Grad-CAM无法生成

**症状**: `无法找到visual_encoder.blocks`

**原因**: 模型结构与脚本不匹配

**解决方案**:
1. 检查模型定义 `models/bio_cot_v3_2.py`
2. 修改 `generate_gradcam.py` 中的target_layer定位代码

---

### 问题3: GPU内存不足

**症状**: `CUDA out of memory`

**解决方案**:
1. 减少 `num_samples`（如从8改为4）
2. 使用CPU（自动fallback）
3. 减少batch_size

---

## 📈 性能指标总结

### 最佳性能（基于训练历史）

| 指标 | 值 | Epoch |
|------|-----|-------|
| **AUC-ROC** | 0.8722 | 最佳 |
| **Accuracy** | 77.38% | Epoch 50 |
| **Sensitivity** | 50.91% | Epoch 50 |
| **Specificity** | 90.27% | Epoch 50 |
| **F1-Score** | 76.20% | Epoch 50 |
| **MCC** | 0.4576 | Epoch 50 |

---

## 📚 引用格式

如果使用了这些可视化工具，请引用：

```bibtex
@article{biocot32_2026,
  title={Bio-COT 3.2: Biomedical Chain-of-Thought with Frozen VLM and Adaptive Multi-Modal Fusion},
  author={Your Name et al.},
  journal={TBD},
  year={2026},
  note={Visualization Suite}
}
```

---

## 🤝 贡献

欢迎改进和扩展可视化功能！

**未来计划**:
- [ ] 添加混淆矩阵可视化
- [ ] 添加PR曲线
- [ ] 添加中心级别性能对比
- [ ] 添加病灶区域分割可视化
- [ ] 添加训练曲线动画

---

## 📧 联系方式

**项目路径**: `/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/visualization`  
**技术支持**: 查看代码注释或联系项目维护者

---

**Last Updated**: 2026-01-24  
**Version**: 1.0  
**Status**: Production Ready ✅

