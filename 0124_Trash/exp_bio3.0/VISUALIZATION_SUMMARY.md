# Bio-COT 3.0 可视化功能总结

## ✅ 已实现的可视化类型

### 1. 训练曲线图 ✅
- **脚本**: `visualize_results.py`
- **内容**: 
  - Loss曲线（训练/验证）
  - Accuracy曲线
  - AUC曲线
  - F1-Score曲线
  - 损失组件分解
  - 综合性能指标
- **输出**: `logs/comprehensive_analysis_*.png`

### 2. t-SNE可视化 ✅
- **脚本**: `comprehensive_visualization.py`
- **内容**:
  - 按标签着色的特征分布
  - 按中心着色的特征分布
  - 按预测概率着色的特征分布
  - 组合视图（标签+中心）
- **输出**: `logs/tsne_visualization_*.png`

### 3. 箱线图 ✅
- **脚本**: `comprehensive_visualization.py`
- **内容**:
  - AUC分布
  - 准确率分布
  - F1-Score分布
  - 损失组件分布
- **输出**: `logs/boxplots_*.png`

### 4. 小提琴图 ✅
- **脚本**: `comprehensive_visualization.py`
- **内容**:
  - AUC分布（显示密度）
  - 准确率分布（显示密度）
  - F1-Score分布（显示密度）
  - 训练/验证对比
- **输出**: `logs/violinplots_*.png`

### 5. CAM图（类激活映射） ✅
- **脚本**: `comprehensive_visualization.py`
- **内容**:
  - 显示模型关注的图像区域
  - 8个样本的CAM可视化
  - 叠加在原始图像上
- **输出**: `logs/cam_samples_*.png`

---

## 🚀 使用方法

### 方法1：一键生成所有可视化

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0
bash GENERATE_ALL_VISUALIZATIONS.sh
```

### 方法2：分别生成

```bash
# 1. 生成训练曲线
python visualize_results.py

# 2. 生成综合可视化（t-SNE、箱线图、小提琴图、CAM）
python comprehensive_visualization.py
```

---

## 📁 输出文件位置

所有可视化文件保存在 `logs/` 目录下：

```
logs/
├── comprehensive_analysis_YYYYMMDD_HHMMSS.png  # 训练曲线
├── tsne_visualization_YYYYMMDD_HHMMSS.png      # t-SNE
├── boxplots_YYYYMMDD_HHMMSS.png                # 箱线图
├── violinplots_YYYYMMDD_HHMMSS.png              # 小提琴图
└── cam_samples_YYYYMMDD_HHMMSS.png             # CAM图
```

---

## 📊 可视化说明

### t-SNE可视化
- **用途**: 展示高维特征在2D空间的分布
- **解读**: 
  - 同类样本聚集在一起 → 模型学习到好的特征表示
  - 不同中心的数据分布不同 → 存在域偏移
  - 预测概率高的区域 → 模型置信度高

### 箱线图
- **用途**: 快速了解性能指标的分布
- **解读**:
  - 中位数线：性能的中等水平
  - 箱体：50%的数据分布范围
  - 须线：数据的整体范围
  - 异常值：性能的极端情况

### 小提琴图
- **用途**: 更详细的分布可视化
- **解读**:
  - 宽度表示该值出现的频率
  - 结合箱线图，显示分布的完整形状
  - 双峰分布可能表示模型不稳定

### CAM图
- **用途**: 可视化模型关注的图像区域
- **解读**:
  - 红色区域：模型高度关注（高激活）
  - 蓝色区域：模型较少关注（低激活）
  - 用于解释模型的决策过程

---

## ⚙️ 配置说明

### 依赖包
```bash
pip install matplotlib seaborn scikit-learn pandas pillow opencv-python
```

### 注意事项
1. **需要训练完成**: t-SNE和CAM需要加载训练好的模型
2. **GPU内存**: CAM生成需要GPU，确保有足够显存
3. **数据量**: t-SNE在大数据集上可能较慢，建议使用验证集
4. **模型检查点**: 确保有最佳模型检查点文件

---

## 🔍 故障排查

### 问题1: 找不到模型检查点
**解决**: 确保训练已完成，检查 `checkpoints/` 目录

### 问题2: t-SNE运行缓慢
**解决**: 减少数据量或使用PCA预降维

### 问题3: CAM图全蓝
**解决**: 检查模型结构和hook注册是否正确

---

**创建时间**: 2025-01-13  
**状态**: ✅ 所有可视化功能已实现

