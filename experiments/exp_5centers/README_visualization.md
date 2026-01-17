# 可视化脚本使用说明

## 概述

本目录包含两个可视化相关文件：

1. **`visualization_standalone.py`**: 独立的可视化脚本，用于重新绘制训练结果图表
2. **`train_bio_cot_5centers_multimodal.py`**: 训练脚本（会自动保存完整结果到JSON文件）

## 使用方法

### 1. 训练完成后自动保存的数据

训练脚本会自动保存完整结果到：
```
results_multimodal/results_bio_cot_multimodal_balanced_<timestamp>.json
```

该JSON文件包含：
- 训练历史（history）：所有epoch的损失、准确率、AUC、F1等
- 最终验证结果（final_val_metrics）：标签、预测概率、预测结果等
- 训练配置（training_config）：epoch数、batch size、学习率等

### 2. 重新绘制图表（自定义色调）

#### 基本用法

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_5centers
python visualization_standalone.py --json_path results_multimodal/results_bio_cot_multimodal_balanced_<timestamp>.json
```

#### 指定输出目录和时间戳

```bash
python visualization_standalone.py \
    --json_path results_multimodal/results_bio_cot_multimodal_balanced_<timestamp>.json \
    --output_dir results_multimodal/custom_plots \
    --timestamp custom_20240108
```

### 3. 自定义色调和样式

#### 方法1：修改脚本中的全局配置

编辑 `visualization_standalone.py`，修改以下配置：

```python
# 颜色配置（第30-40行左右）
COLOR_PALETTE = {
    'primary': '#2E86AB',      # 主色：蓝色（可改为任何颜色）
    'secondary': '#A23B72',   # 次色：紫红色
    'positive': '#28A745',     # 阳性：绿色
    'negative': '#DC3545',     # 阴性：红色
    # ... 更多颜色
}

# 图表样式配置（第42-52行左右）
PLOT_STYLE = {
    'figsize': (12, 8),        # 图表大小
    'dpi': 300,                # 分辨率
    'fontsize': 12,             # 字体大小
    'linewidth': 2.5,          # 线条宽度
    'alpha': 0.7,              # 透明度
    # ... 更多样式
}

# 热图颜色映射（第54行左右）
HEATMAP_CMAP = 'viridis'  # 可选：'viridis', 'plasma', 'coolwarm', 'RdYlBu', 'Blues'等
```

#### 方法2：在代码中调用函数时传入参数

```python
from visualization_standalone import plot_training_curves, plot_roc_curve_custom
from pathlib import Path

# 自定义颜色
custom_colors = {
    'train': '#FF6B6B',    # 红色
    'val': '#4ECDC4',      # 青色
    'positive': '#95E1D3', # 浅绿色
    'negative': '#F38181'  # 浅红色
}

# 自定义样式
custom_style = {
    'linewidth': 3,
    'alpha': 0.8,
    'fontsize': 14
}

# 绘制图表
plot_training_curves(history, output_dir, timestamp, 
                    colors=custom_colors, style=custom_style)
plot_roc_curve_custom(y_true, y_probs, output_dir, timestamp,
                     color='#FF6B6B', style=custom_style)
```

## 可用的绘图函数

### 1. `plot_training_curves`
绘制训练曲线（损失、准确率、AUC、F1）

### 2. `plot_confusion_matrix_custom`
绘制混淆矩阵（可自定义颜色映射）

### 3. `plot_roc_curve_custom`
绘制ROC曲线（可自定义颜色）

### 4. `plot_prediction_distribution_custom`
绘制预测概率分布（9种图表：直方图、箱线图、小提琴图、热图、CDF、Q-Q图、统计摘要、密度对比、直方图+KDE）

### 5. `plot_loss_heatmap_custom`
绘制损失组件热图（可自定义颜色映射）

## 颜色参考

### 常用颜色代码

- **蓝色系**: `#2E86AB`, `#1E88E5`, `#2196F3`, `#0D47A1`
- **绿色系**: `#28A745`, `#4CAF50`, `#06A77D`, `#2E7D32`
- **红色系**: `#DC3545`, `#F44336`, `#C73E1D`, `#B71C1C`
- **橙色系**: `#F18F01`, `#FF9800`, `#FF6F00`
- **紫色系**: `#A23B72`, `#9C27B0`, `#7B1FA2`
- **灰色系**: `#6C757D`, `#757575`, `#424242`

### 热图颜色映射选项

- `'viridis'`: 紫-绿-黄（默认，适合大多数情况）
- `'plasma'`: 紫-粉-黄
- `'coolwarm'`: 蓝-白-红（适合有正负值的数据）
- `'RdYlBu'`: 红-黄-蓝
- `'Blues'`: 蓝色渐变（适合混淆矩阵）
- `'Reds'`: 红色渐变
- `'Greens'`: 绿色渐变

## 论文发表建议

### 期刊要求

不同期刊对图表有不同要求：

1. **Nature/Science**: 通常要求高分辨率（600 DPI），简洁的配色
2. **IEEE**: 通常要求灰度图或特定配色方案
3. **医学期刊**: 通常要求清晰的对比度和可读性

### 推荐配置

#### 高质量论文（彩色）
```python
COLOR_PALETTE = {
    'primary': '#1E88E5',      # 明亮蓝色
    'secondary': '#D32F2F',   # 明亮红色
    'positive': '#388E3C',    # 深绿色
    'negative': '#E53935',    # 深红色
}
PLOT_STYLE = {'dpi': 600, 'fontsize': 14}
```

#### 灰度图（适合黑白打印）
```python
COLOR_PALETTE = {
    'primary': '#000000',     # 黑色
    'secondary': '#666666',   # 深灰色
    'positive': '#333333',    # 中灰色
    'negative': '#999999',   # 浅灰色
}
HEATMAP_CMAP = 'gray'
```

## 示例脚本

创建一个自定义绘制脚本 `custom_plot.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""自定义绘制脚本示例"""

from pathlib import Path
from visualization_standalone import (
    load_training_results,
    plot_training_curves,
    plot_roc_curve_custom,
    plot_prediction_distribution_custom,
    ensure_numpy_array
)

# 加载数据
json_path = Path('results_multimodal/results_bio_cot_multimodal_balanced_20260108_090241.json')
data = load_training_results(str(json_path))

# 提取数据
history = data['history']
final_val_metrics = data['final_val_metrics']

# 自定义配色（例如：使用期刊要求的配色）
custom_colors = {
    'train': '#1E88E5',      # 蓝色
    'val': '#D32F2F',        # 红色
    'positive': '#388E3C',   # 绿色
    'negative': '#E53935'    # 红色
}

# 自定义样式
custom_style = {
    'figsize': (14, 10),
    'dpi': 600,              # 高分辨率
    'fontsize': 14,
    'title_fontsize': 16,
    'linewidth': 3,
    'alpha': 0.8
}

output_dir = Path('results_multimodal/custom_plots')
output_dir.mkdir(exist_ok=True)
timestamp = 'journal_submission'

# 绘制图表
plot_training_curves(history, output_dir, timestamp, 
                    colors=custom_colors, style=custom_style)

y_true = ensure_numpy_array(final_val_metrics['labels'])
y_probs = ensure_numpy_array(final_val_metrics['probs'])

plot_roc_curve_custom(y_true, y_probs, output_dir, timestamp,
                     color=custom_colors['primary'], style=custom_style)

plot_prediction_distribution_custom(y_true, y_probs, output_dir, timestamp,
                                   colors=custom_colors, style=custom_style)

print("✅ 自定义图表已生成！")
```

## 常见问题

### Q1: 如何生成灰度图？

修改 `HEATMAP_CMAP = 'gray'` 并将颜色改为灰度值。

### Q2: 如何调整字体大小？

修改 `PLOT_STYLE['fontsize']` 或调用函数时传入 `style={'fontsize': 14}`。

### Q3: 如何保存为PDF格式？

在 `plt.savefig()` 中将 `.png` 改为 `.pdf`。

### Q4: 如何调整图表大小？

修改 `PLOT_STYLE['figsize']` 或调用函数时传入 `style={'figsize': (16, 12)}`。

## 联系

如有问题，请检查：
1. JSON文件路径是否正确
2. JSON文件是否包含必要的数据字段
3. 依赖库是否已安装（matplotlib, seaborn, scipy等）

