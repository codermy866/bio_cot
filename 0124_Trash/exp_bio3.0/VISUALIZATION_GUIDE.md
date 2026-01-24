# Bio-COT 3.0 可视化指南

## 📊 可视化类型

### 1. 训练曲线图 ✅
- **文件**: `visualize_results.py`
- **内容**: Loss、Accuracy、AUC、F1-Score等训练曲线
- **输出**: `logs/comprehensive_analysis_*.png`

### 2. t-SNE可视化 ✅
- **文件**: `comprehensive_visualization.py`
- **内容**: 特征空间分布，按标签/中心/预测概率着色
- **输出**: `logs/tsne_visualization_*.png`

### 3. 箱线图 ✅
- **文件**: `comprehensive_visualization.py`
- **内容**: AUC、准确率、F1-Score、损失组件的分布
- **输出**: `logs/boxplots_*.png`

### 4. 小提琴图 ✅
- **文件**: `comprehensive_visualization.py`
- **内容**: 更详细的分布可视化，显示密度
- **输出**: `logs/violinplots_*.png`

### 5. CAM图 ✅
- **文件**: `comprehensive_visualization.py`
- **内容**: 类激活映射，显示模型关注的区域
- **输出**: `logs/cam_samples_*.png`

---

## 🚀 使用方法

### 方法1：运行综合可视化脚本

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate
python comprehensive_visualization.py
```

### 方法2：只生成训练曲线

```bash
python visualize_results.py
```

---

## 📁 输出文件

所有可视化文件保存在 `logs/` 目录下：

- `comprehensive_analysis_*.png` - 训练曲线综合图
- `tsne_visualization_*.png` - t-SNE可视化
- `boxplots_*.png` - 箱线图
- `violinplots_*.png` - 小提琴图
- `cam_samples_*.png` - CAM图样本

---

## 🔍 可视化说明

### t-SNE可视化
- **按标签着色**: 显示不同类别在特征空间中的分布
- **按中心着色**: 显示不同中心的数据分布
- **按预测概率着色**: 显示模型预测的置信度分布

### 箱线图
- 显示指标的中位数、四分位数、异常值
- 用于快速了解性能分布

### 小提琴图
- 结合箱线图和密度图
- 显示分布的完整形状

### CAM图
- 显示模型关注的图像区域
- 红色区域表示高激活（模型关注的重点）

---

## 📝 注意事项

1. **需要训练完成**: t-SNE和CAM需要加载训练好的模型
2. **GPU内存**: CAM生成需要GPU，确保有足够显存
3. **数据量**: t-SNE在大数据集上可能较慢，建议使用验证集

---

**最后更新**: 2025-01-13

