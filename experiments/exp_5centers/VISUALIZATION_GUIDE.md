# Bio-COT 2.0 可视化工具使用指南

## 📊 可视化工具概览

本项目提供了三种专业的可视化工具，适合论文发表：

1. **训练曲线可视化** (`visualize_training_paper.py`)
2. **t-SNE特征空间可视化** (`visualize_tsne.py`)
3. **通用训练曲线** (`visualize_training.py`)

---

## 1. 训练曲线可视化（论文级）

### 功能
生成适合论文发表的专业训练曲线，包括：
- 主图：Loss、Accuracy、AUC、F1-Score
- 补充图：Loss组件分解（线性和对数刻度）
- 综合图：所有性能指标对比

### 使用方法

```bash
# 从日志文件生成
python experiments/exp_5centers/visualize_training_paper.py \
    --log experiments/exp_5centers/logs/train_bio_cot_v2_fixed_20260108_162453.log

# 从JSON文件生成（如果训练时保存了历史）
python experiments/exp_5centers/visualize_training_paper.py \
    --json experiments/exp_5centers/logs/training_history_20260108_162453.json

# 自动查找最新文件
python experiments/exp_5centers/visualize_training_paper.py
```

### 输出文件
- `training_curves_paper_main_*.png/pdf` - 主图（2x2布局）
- `training_curves_paper_supp_*.png/pdf` - 补充图（Loss组件）
- `training_curves_paper_comprehensive_*.png/pdf` - 综合性能图

### 特点
- ✅ 学术论文风格（Times New Roman字体）
- ✅ 高分辨率（300 DPI）
- ✅ PDF矢量图（文字可编辑）
- ✅ 清晰的子图标注（(a), (b), (c), (d)）

---

## 2. t-SNE特征空间可视化

### 功能
可视化模型学习到的特征空间分布：
- 按标签着色（阴性/阳性）
- 按中心着色（不同医院）
- 对比不同特征（z_causal vs z_sem）

### 使用方法

```bash
# 基本使用
python experiments/exp_5centers/visualize_tsne.py \
    --checkpoint experiments/exp_5centers/checkpoints_bio_cot_v2/best_model_v2_*.pth \
    --split val \
    --feature fused \
    --batch_size 16

# 可视化不同特征
python experiments/exp_5centers/visualize_tsne.py \
    --checkpoint <checkpoint_path> \
    --split val \
    --feature z_causal  # 或 z_sem, z_noise, fused

# 使用训练集
python experiments/exp_5centers/visualize_tsne.py \
    --checkpoint <checkpoint_path> \
    --split train \
    --feature fused
```

### 参数说明
- `--checkpoint`: 模型检查点路径（必需）
- `--split`: 数据集分割（train/val/test，默认val）
- `--feature`: 要可视化的特征（z_causal/z_sem/z_noise/fused，默认fused）
- `--batch_size`: 批次大小（默认32）
- `--data_root`: 数据根目录
- `--clinical_embed_path`: LLM嵌入路径
- `--output`: 输出目录（默认logs目录）

### 输出文件
- `tsne_fused_*.png/pdf` - 融合特征的t-SNE图（按标签和中心着色）
- `tsne_comparison_*.png/pdf` - z_causal vs z_sem对比图

### 特点
- ✅ 自动PCA预降维（如果特征维度>50）
- ✅ 高质量t-SNE可视化
- ✅ 多视角对比（标签 vs 中心）
- ✅ 特征对比（因果特征 vs 语义特征）

### 技术细节
- **PCA预降维**：如果特征维度>50，先用PCA降到50维（保留99%+方差）
- **t-SNE参数**：perplexity=30, n_iter=1000
- **降维时间**：约1-3分钟（取决于样本数）

---

## 3. 通用训练曲线可视化

### 功能
生成详细的训练曲线（包含所有指标）

### 使用方法

```bash
# 从日志文件生成
python experiments/exp_5centers/visualize_training.py \
    --log experiments/exp_5centers/logs/train_bio_cot_v2_*.log

# 自动查找最新文件
python experiments/exp_5centers/visualize_training.py
```

---

## 📝 论文使用建议

### 1. 训练曲线图
**推荐使用**：`training_curves_paper_main_*.pdf`
- 适合放在论文主图
- 包含4个子图：Loss、Accuracy、AUC、F1-Score
- 清晰的标注和说明

**补充材料**：`training_curves_paper_supp_*.pdf`
- 展示Loss组件分解
- 说明模型各部分的作用

### 2. t-SNE图
**推荐使用**：`tsne_fused_*.pdf`
- 展示模型学习到的特征空间
- 证明模型能够区分不同类别
- 可以观察是否存在中心偏差

**特征对比**：`tsne_comparison_*.pdf`
- 对比因果特征和语义特征
- 说明多模态融合的有效性

### 3. 图表质量
- ✅ 所有图表都是300 DPI高分辨率
- ✅ PDF格式为矢量图，可无损缩放
- ✅ 字体为Times New Roman（符合期刊要求）
- ✅ 图表尺寸适合单栏或双栏布局

---

## 🔧 故障排除

### 问题1：t-SNE运行很慢
**解决方案**：
- 减少样本数（使用`--split val`而不是`train`）
- 降低batch_size（如果内存允许）
- 特征维度会自动用PCA预降维

### 问题2：找不到检查点文件
**解决方案**：
```bash
# 查找所有检查点文件
find experiments/exp_5centers/checkpoints_bio_cot_v2 -name "*.pth"

# 使用最新的文件
ls -t experiments/exp_5centers/checkpoints_bio_cot_v2/*.pth | head -1
```

### 问题3：内存不足
**解决方案**：
- 减小batch_size（如`--batch_size 8`）
- 使用验证集而不是训练集（`--split val`）
- 只可视化部分特征（如`--feature z_causal`）

### 问题4：字体显示问题
**解决方案**：
- PDF格式中的文字是可编辑的
- 如果PNG中字体显示异常，可以编辑PDF后导出

---

## 📊 示例输出

### 训练曲线主图
- 4个子图展示核心性能指标
- 清晰的图例和标注
- 适合论文主图使用

### t-SNE图
- 左图：按标签着色（展示分类能力）
- 右图：按中心着色（展示中心偏差）
- 特征对比图：展示不同特征空间的差异

---

## 🎯 最佳实践

1. **论文主图**：使用`training_curves_paper_main_*.pdf`
2. **补充材料**：使用`training_curves_paper_supp_*.pdf`和`tsne_*.pdf`
3. **特征分析**：使用`tsne_comparison_*.pdf`展示多模态融合效果
4. **所有图表**：优先使用PDF格式（矢量图，可编辑）

---

## 📚 相关文件

- `train_bio_cot_v2.py` - 训练脚本（已集成可视化）
- `visualize_training_paper.py` - 论文级训练曲线
- `visualize_tsne.py` - t-SNE特征可视化
- `visualize_training.py` - 通用训练曲线

---

## ✅ 已生成的图表

根据最新训练结果（AUC=0.8210），已生成以下图表：

1. **训练曲线主图**：`training_curves_paper_main_fixed_20260108_162453.pdf`
2. **训练曲线补充图**：`training_curves_paper_supp_fixed_20260108_162453.pdf`
3. **综合性能图**：`training_curves_paper_comprehensive_fixed_20260108_162453.pdf`
4. **t-SNE融合特征图**：`tsne_fused_*.pdf`
5. **t-SNE特征对比图**：`tsne_comparison_*.pdf`

所有图表都保存在 `experiments/exp_5centers/logs/` 目录下。



