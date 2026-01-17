# Bio-COT 模型架构图说明

## 文件位置

所有架构图文件保存在：`docs/figures/`

## 生成的文件

1. **BioCOT_Architecture_Paper.pdf**
   - 格式：PDF（矢量格式）
   - 用途：论文提交（期刊/会议）
   - 特点：无损缩放，适合印刷

2. **BioCOT_Architecture_Paper.png**
   - 格式：PNG（位图）
   - 分辨率：300 DPI
   - 用途：预览、演示文稿
   - 特点：高分辨率，适合屏幕显示

3. **BioCOT_Architecture_Paper.svg**
   - 格式：SVG（矢量格式）
   - 用途：可编辑版本
   - 特点：可用Inkscape/Illustrator编辑

## 架构图特点

### 1. 完整的模块展示
- ✅ 7个核心模块（输入层、融合、编码器、分类器等）
- ✅ 所有数据维度标注（[B, 512], [B, 768]等）
- ✅ 网络结构细节（MLP层数、Dropout率等）

### 2. 清晰的数据流
- ✅ 主数据流（实线箭头）
- ✅ 损失函数连接（虚线箭头）
- ✅ 辅助连接（点线箭头）

### 3. 创新点突出
- ✅ ① Student Prior网络（替代在线VLM）
- ✅ ② Memory Bank + 反事实干预（因果解耦）
- ✅ ③ Sinkhorn最优传输（灵活分布对齐）

### 4. 损失函数详细说明
- ✅ 4个损失函数的数学公式（LaTeX格式）
- ✅ 损失权重参数（λ值）
- ✅ 总损失公式

### 5. 专业设计
- ✅ 配色方案（色盲友好）
- ✅ 清晰的模块边界
- ✅ 统一的字体和标注风格
- ✅ 适合2栏论文布局（16:10比例）

## 使用方法

### 在论文中使用

1. **LaTeX文档**：
```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/BioCOT_Architecture_Paper.pdf}
    \caption{Bio-COT模型架构图。展示了从多模态输入到最终分类输出的完整流程，包括三个核心创新点：①Student Prior网络、②Memory Bank反事实干预、③Sinkhorn最优传输。}
    \label{fig:architecture}
\end{figure}
```

2. **Word文档**：
   - 直接插入PDF或PNG文件
   - 建议使用PDF格式以保持清晰度

### 重新生成架构图

如果需要修改架构图，编辑脚本：
```bash
python scripts/draw_bio_cot_architecture_paper.py
```

## 技术规格

- **画布尺寸**：20×12 英寸
- **分辨率**：300 DPI
- **颜色模式**：RGB（适合屏幕和打印）
- **字体**：Arial/DejaVu Sans（无衬线字体，易读）

## 建议

1. **论文提交前**：
   - 检查所有标注是否正确
   - 确认颜色在黑白打印时仍可区分
   - 验证数学公式是否正确

2. **演示文稿**：
   - 使用PNG格式（高分辨率）
   - 可以放大关键模块进行讲解

3. **海报展示**：
   - 可以进一步放大尺寸
   - 建议使用PDF格式保持清晰度

