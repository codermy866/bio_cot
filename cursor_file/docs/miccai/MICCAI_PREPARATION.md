<!--
文件生成信息:
- 生成时间: 2025-01-XX 15:50:00
- 生成需求: 用户要求创建MICCAI论文准备指南，指导后续一个月的论文准备工作
- 生成原因: 用户需要在接下来一个月内完成MICCAI论文，需要详细的准备计划和检查清单
- 相关任务: MICCAI论文准备和规划

文件功能: MICCAI论文准备指南，包含准备工作清单、开发工作流和检查清单
-->

# MICCAI论文准备指南

## 🎯 目标

在一个月内完成MICCAI论文的准备工作，确保代码、实验和可视化都符合MICCAI标准。

## ✅ 已完成的准备工作

### 1. 项目结构重组
- ✅ 核心代码结构清晰
- ✅ 废弃文件已移动到Trash
- ✅ 空文件夹已删除
- ✅ 辅助文件统一放在 `cursor_file/` 目录

### 2. 配置管理
- ✅ 项目配置 (`project_config.py`) - 统一管理路径和虚拟环境
- ✅ 可视化配置 (`visualization_config.py`) - 统一字体和配色方案

### 3. 虚拟环境
- ✅ 虚拟环境路径: `/data2/hmy/VLM_Caus_Rm_Mics/my_retfound`
- ✅ 已配置并验证

## 🎨 可视化标准

### 配色方案（8种颜色）

```python
MICCAI_COLORS = [
    '#c7522a',  # 深红棕色
    '#e5c185',  # 浅金色
    '#f0daa5',  # 米黄色
    '#fbf2c4',  # 浅米色
    '#b8cdab',  # 浅绿色
    '#74a892',  # 青绿色
    '#008585',  # 深青色
    '#004343',  # 深墨绿色
]
```

### 字体要求
- ✅ 统一使用英文字体（DejaVu Sans或Arial）
- ✅ 标题字体大小: 14-16pt
- ✅ 正文字体大小: 10-12pt
- ✅ 图表分辨率: 300 DPI

### 使用示例

```python
from cursor_file.visualization_config import setup_miccai_style, get_color_palette
import matplotlib.pyplot as plt

# 1. 设置样式
setup_miccai_style()

# 2. 创建图表
fig, ax = plt.subplots(figsize=(8, 6))

# 3. 使用配色方案
colors = get_color_palette(4)
for i, color in enumerate(colors):
    ax.plot(x, y, color=color, label=f'Series {i+1}')

# 4. 保存（自动300 DPI）
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
```

## 📁 项目结构

### 核心目录
- `src/` - 核心源代码（模型、数据处理、评估）
- `experiments/` - 实验脚本
- `models/` - 模型Backbone
- `utils/` - 工具函数

### 辅助目录
- `cursor_file/` - 配置文件、文档、工具脚本
- `Trash/` - 废弃文件（可删除）

## 🔧 开发工作流

### 1. 激活环境
```bash
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate
```

### 2. 运行实验
```bash
cd experiments/exp1_causal_bayesian_clip
python train.py --data_path ../../data/5centers_multi
```

### 3. 生成可视化
```python
# 使用统一配置
from cursor_file.visualization_config import setup_miccai_style
setup_miccai_style()
# ... 生成图表
```

### 4. 文件管理
- 核心代码 → 放在对应目录（src/, experiments/等）
- 辅助文件 → 放在 `cursor_file/` 目录
- 废弃文件 → 移动到 `Trash/` 目录

## 📝 检查清单

### 代码质量
- [ ] 所有训练脚本可以正常运行
- [ ] 模型定义完整且正确
- [ ] 数据处理流程清晰
- [ ] 评估指标计算正确

### 可视化质量
- [ ] 所有图表使用统一配色方案
- [ ] 字体统一（英文）
- [ ] 分辨率300 DPI
- [ ] 图表清晰、美观

### 文档完整性
- [ ] README更新完整
- [ ] 代码注释清晰
- [ ] 实验记录完整

### 实验完整性
- [ ] 基线实验完成
- [ ] 主实验完成
- [ ] 消融实验完成
- [ ] 结果可复现

## 🚀 下一步工作

1. **代码优化**
   - 检查并修复导入路径问题
   - 优化训练脚本
   - 完善错误处理

2. **实验运行**
   - 运行基线实验
   - 运行主实验
   - 记录实验结果

3. **可视化生成**
   - 生成架构图
   - 生成结果对比图
   - 生成消融实验图

4. **论文撰写**
   - 整理实验结果
   - 撰写方法部分
   - 撰写实验部分

---
**准备日期**: 2025-01-XX
**目标**: MICCAI 2025投稿

