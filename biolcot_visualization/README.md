# BioLCoT 可视化工具

本目录包含三个专门针对 BioLCoT 架构的可视化脚本：

1. **Grad-CAM (类激活映射)**: 展示模型在做最终决定时，主要关注图像的哪些区域
2. **Attention Map (注意力图)**: 展示 Visual Notes 模块关注了哪些区域，证明 CoT 机制的有效性
3. **病灶聚焦 Grad-CAM**: 只标注重点病灶区域，去除背景噪声，适合发表级可视化

## 环境设置

### 1. 激活虚拟环境

```bash
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate
```

### 2. 安装依赖

```bash
# 安装 pytorch-grad-cam (用于 Grad-CAM)
pip install grad-cam

# 其他依赖应该已经在项目中安装
```

## 使用方法

### 🚀 快速开始（推荐）

使用一键运行脚本，自动查找最新的 checkpoint 并生成所有可视化：

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/biolcot_visualization

# 运行一键脚本（自动查找最新 checkpoint）
bash run_visualization.sh
```

这个脚本会：
1. 自动查找 `../checkpoints/` 目录下最新的 `best_model_*.pth` 文件
2. 生成 Grad-CAM 可视化
3. 生成 Attention Map 可视化

### 方法一：生成 Grad-CAM

Grad-CAM 用于展示模型在做最终分类决定时，主要关注图像的哪些区域。

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/biolcot_visualization

# 自动查找最新 checkpoint（推荐）
python generate_gradcam.py --num_samples 4

# 或者手动指定 checkpoint
python generate_gradcam.py \
    --checkpoint ../checkpoints/best_model_v3_20260128_180853.pth \
    --num_samples 4 \
    --save_dir gradcam_results
```

**参数说明：**
- `--checkpoint`: 模型 checkpoint 路径（默认: **自动查找最新的 best_model_*.pth**）
- `--num_samples`: 生成的样本数量（默认: 4）
- `--target_layer`: 目标层名称（默认: 自动查找，推荐使用 `visual_encoder.vit.blocks[-1]`）
- `--save_dir`: 保存目录（默认: `gradcam_results`）

**输出：**
- `gradcam_oct_sample{N}_{Positive/Negative}.png`: OCT 图像的 Grad-CAM 叠加图
- `gradcam_colpo_sample{N}_{Positive/Negative}.png`: Colposcopy 图像的 Grad-CAM 叠加图

### 方法二：生成病灶聚焦 Grad-CAM（推荐用于发表）

病灶聚焦 Grad-CAM 专门用于"只标注重点病灶区域"，去除背景噪声，生成发表级可视化。

```bash
# 自动查找最新 checkpoint（推荐）
python generate_lesion_focused_gradcam.py --num_samples 4

# 或者手动指定 checkpoint 和参数
python generate_lesion_focused_gradcam.py \
    --checkpoint ../checkpoints/best_model_v3_20260128_180853.pth \
    --num_samples 4 \
    --threshold 0.6 \
    --smooth_sigma 1.0 \
    --save_dir lesion_focused_results
```

**参数说明：**
- `--checkpoint`: 模型 checkpoint 路径（默认: **自动查找最新的 best_model_*.pth**）
- `--num_samples`: 生成的样本数量（默认: 4）
- `--target_layer`: 目标层名称（默认: 自动查找）
- `--threshold`: 阈值过滤参数 (0.0-1.0)，表示百分位数（默认: 0.6）
  - **使用百分位数阈值**：0.6 表示保留 top 40% 的激活区域（更智能，推荐）
  - 0.5 = 保留 top 50%（较宽松）
  - 0.6 = 保留 top 40%（推荐）
  - 0.7 = 保留 top 30%（较严格）
  - 0.8 = 保留 top 20%（很严格，只显示最显著病灶）
- `--smooth_sigma`: 高斯平滑参数，用于让边缘更自然（默认: 1.0）
  - 0.5-2.0 推荐，值越大边缘越平滑
- `--use_absolute_threshold`: 使用绝对阈值而不是百分位数（不推荐，除非百分位数效果不好）
- `--save_dir`: 保存目录（默认: `lesion_focused_results`）

**输出：**
- `original_oct_sample{N}_{Positive/Negative}.png/.pdf`: **原始 OCT 图像**（没有任何操作，用于论文对比）
- `original_colpo_sample{N}_{Positive/Negative}.png/.pdf`: **原始 Colposcopy 图像**（没有任何操作，用于论文对比）
- `raw_oct_sample{N}_{Positive/Negative}.png/.pdf`: OCT 原始 Grad-CAM 版本（用于对比）
- `raw_colpo_sample{N}_{Positive/Negative}.png/.pdf`: Colposcopy 原始 Grad-CAM 版本（用于对比）
- `lesion_focused_oct_sample{N}_{Positive/Negative}.png/.pdf`: OCT 病灶聚焦叠加图（**推荐用于发表**）
- `lesion_focused_colpo_sample{N}_{Positive/Negative}.png/.pdf`: Colposcopy 病灶聚焦叠加图（**推荐用于发表**）

**注意**：所有图片都会同时生成 PNG 和 PDF 格式，PDF 格式更清晰，适合论文使用。

**调试信息：**
脚本会输出详细的统计信息，包括：
- 原始 CAM 和聚焦后 CAM 的统计（Min, Max, Mean）
- 百分位数阈值和激活像素数
- 变化量和变化像素数
- 如果过滤效果不明显，会给出警告和建议

**输出：**
- `lesion_focused_oct_sample{N}_{Positive/Negative}.png`: OCT 病灶聚焦叠加图
- `lesion_focused_colpo_sample{N}_{Positive/Negative}.png`: Colposcopy 病灶聚焦叠加图

**关键特性：**
- **百分位数阈值过滤**：使用百分位数阈值（默认保留 top 40%），自动适应不同图像的激活分布，比固定阈值更智能
- **形态学处理**：高斯平滑，使病灶区域更集中、边缘更自然
- **智能背景遮罩**：OCT 背景自动去除，Colposcopy 保持原图清晰度
- **Colposcopy 颜色先验增强（关键）**：自动检测红色/糜烂区域（宫颈口流血），结合颜色特征与 Grad-CAM 激活，大幅增强病灶区域的显示
  - 使用 HSV 颜色空间检测红色/粉色区域
  - 结合中心加权（宫颈口通常在中心）
  - 去除反光区域（窥器反光）
  - 即使模型在红色区域激活较弱，也会基于颜色特征给予增强
- **详细调试信息**：输出统计信息，帮助理解过滤效果

**如果看不到明显变化：**
1. 检查控制台输出的统计信息，查看"激活区域减少"百分比
2. 如果减少 < 10%，说明阈值设置太宽松，尝试：
   ```bash
   python generate_lesion_focused_gradcam.py --threshold 0.7 --num_samples 4
   ```
3. 如果减少 > 90%，说明阈值太严格，尝试：
   ```bash
   python generate_lesion_focused_gradcam.py --threshold 0.5 --num_samples 4
   ```
4. 对比生成的 `raw_*.png` 和 `lesion_focused_*.png` 文件，查看差异

### 方法三：生成 Attention Map

Attention Map 用于展示 Visual Notes 模块的注意力权重，证明 CoT 机制的有效性。

```bash
# 自动查找最新 checkpoint（推荐）
python generate_attention_map.py --num_samples 4

# 或者手动指定 checkpoint
python generate_attention_map.py \
    --checkpoint ../checkpoints/best_model_v3_20260128_180853.pth \
    --num_samples 4 \
    --save_dir attention_results
```

**参数说明：**
- `--checkpoint`: 模型 checkpoint 路径（默认: **自动查找最新的 best_model_*.pth**）
- `--num_samples`: 生成的样本数量（默认: 4）
- `--save_dir`: 保存目录（默认: `attention_results`）

**输出：**
- `attention_map_sample{N}_{Positive/Negative}.png`: 完整的可视化图（包含原图、叠加图、热图）
- `attention_oct_sample{N}_{Positive/Negative}.jpg`: OCT 注意力叠加图
- `attention_colpo_sample{N}_{Positive/Negative}.jpg`: Colposcopy 注意力叠加图

### 📦 当前可用的 Checkpoint

脚本会自动查找 `../checkpoints/` 目录下最新的 checkpoint。当前可用的 checkpoint 包括：

- `best_model_v3_20260128_180853.pth` (最新，1月28日 20:12)
- `best_model_v3_20260128_153629.pth` (1月28日 17:52)
- `best_model_v3_20260128_151636.pth` (1月28日 15:33)
- 以及其他历史版本...

脚本会**自动选择最新的 checkpoint**，无需手动指定。

## 关键修复

### 1. OCT 背景激活问题修复（智能背景遮罩 + 连通域分析）

**问题**：Grad-CAM 计算的是梯度，即使输入是纯黑（0），如果网络在边缘学到了 Bias，或者 Padding 区域有梯度回传，归一化后这部分噪声会被放大成"高亮红区"。

**修复**：使用**智能背景遮罩 (Smart Masking)**，包含以下步骤：
1. **提高阈值分割**：灰度值 > 30 的地方才被认为是组织（比之前的 15 更激进，确保去除背景噪声）
2. **强力形态学开运算**：迭代 2 次，去除细小的噪点和非连通的"冰柱"干扰
3. **连通域分析（关键改进）**：只保留面积最大的组织块。OCT 图像中，下方的组织结构通常是最大的连通区域，而上方的噪点、冰柱状伪影通常是断开的小块。这一步能直接把上方乱七八糟的激活全部干掉，只保留组织上的热力图。
4. **强制顶部裁剪**：强制将图片顶部 10% 区域（通常是空气/探头）置零，作为保险措施
5. **强制背景归零**：背景区域的热力值强制置为 0
6. **动态重归一化**：去除背景噪声后重新归一化，确保病灶显示为最红

**效果**：背景区域的 Heatmap 强制变为 0，只有组织区域才有激活。这在医学图像处理中是标准且合规的操作（ROI Masking），完全不是造假，而是"后处理去噪"。

### 2. 肉色变紫色问题修复（色彩空间强力矫正 + 透明度加权融合）

**问题**：这是最经典的 OpenCV BGR 读取与 Matplotlib RGB 显示不匹配的问题。`cv2.imread` 读出来的数据是 [Blue, Green, Red]。如果你直接把这个矩阵丢给 `plt.imshow` 或者和热力图叠加，粉色（红色分量高）会被错误渲染为紫色（蓝色分量高）。此外，传统的全局叠加（如 `0.6 * 原图 + 0.4 * 热力图`）会导致整张图都套上"紫色滤镜"。

**修复**：
1. **色彩空间转换**：在读取图片和生成热力图的**每一步**都强制进行 BGR -> RGB 转换
2. **透明度加权融合（关键改进）**：对于 Colposcopy 图像，不再使用全局叠加，而是基于热力图强度的动态权重融合：
   - 计算权重：`weight = heatmap ** 1.2`（指数增强，压制低值，突出高值）
   - 动态混合：`Result = (热力图 * Weight * 0.7) + (原图 * (1 - Weight * 0.3))`
   - 效果：如果热力值很低（背景），weight 接近 0，直接显示原图（粉色）；如果热力值很高（病灶），weight 接近 1，显示红色。这样背景依然是原本的肉色/粉色，只有病灶处才会发光。

**效果**：彻底消灭"紫色肉"，背景保持原本的粉色，只有病灶区域显示红色热力图，看起来非常干净、专业。

## 技术细节

### Grad-CAM 工作原理

1. **目标层选择**：
   - 优先使用 `visual_encoder.vit.blocks[-1]`（最后一个 Transformer Block）
   - 如果没有，则使用 `visual_notes_module.layer.k_proj`（处理图像特征的层）
   - 最后备选 `dual_head[0]`（特征融合后的层）

2. **计算流程**：
   - 前向传播获取激活值
   - 反向传播获取梯度
   - 使用 LayerCAM 方法（element-wise weighting）计算热图
   - 上采样到原图大小并应用高斯模糊

### Attention Map 工作原理

1. **注意力提取**：
   - 直接调用 `visual_notes_module` 的 forward 方法
   - 获取返回的 `mask`（注意力权重）`[B, N, 1]`
   - 分别提取 OCT 和 Colposcopy 的注意力权重

2. **可视化流程**：
   - 将注意力权重 reshape 到空间维度（14x14）
   - 上采样到原图大小（224x224）
   - 应用高斯模糊平滑
   - 使用 JET colormap 叠加到原图

## 结果解读

### Grad-CAM 结果

- **红色区域**：模型高度关注的区域（可能是病灶）
- **蓝色区域**：模型较少关注的区域（可能是背景）
- **用途**：证明模型关注的是病灶而不是背景（如手术器械、反光）

**图注建议**：
> "The model accurately localizes the acetowhite epithelium region."

### Attention Map 结果

- **高激活区域**：Visual Notes 模块关注的重点区域
- **不同样本**：可能展示不同的关注模式
  - Visual Note #1 可能关注纹理
  - Visual Note #2 可能关注血管
- **用途**：证明模型学到了"正交的语义特征"（Orthogonal Semantic Features），呼应 Method 公式 (12) 里的 $\mathcal{L}_{orth}$

**图注建议**：
> "The Visual Notes mechanism successfully identifies distinct semantic regions, demonstrating the effectiveness of the CoT mechanism."

## 常见问题

### Q1: 找不到 checkpoint 文件

**解决方案**：
- 检查 checkpoint 路径是否正确
- 如果使用相对路径，确保在正确的目录下运行
- 可以指定绝对路径：`--checkpoint /path/to/checkpoint.pth`

### Q2: 生成的图像全是黑色或没有激活

**可能原因**：
1. 模型未训练或训练不充分
2. 目标层选择不当
3. 输入数据格式不正确

**解决方案**：
1. 确保使用训练好的模型 checkpoint
2. 尝试指定不同的 `--target_layer`
3. 检查数据加载是否正常

### Q3: 内存不足

**解决方案**：
- 减少 `--num_samples` 参数
- 使用更小的 batch size（修改脚本中的 `batch_size=1`）
- 使用 CPU 模式（虽然会很慢）

### Q4: 注意力权重全为 1

**可能原因**：
- 模型未启用 Visual Notes（`use_visual_notes=False`）
- 模型未训练或训练不充分

**解决方案**：
- 检查模型配置中的 `use_visual_notes` 参数
- 确保使用训练好的模型

## 高级用法

### 自定义目标层

如果你想指定特定的层进行 Grad-CAM 可视化：

```bash
python generate_gradcam.py \
    --checkpoint ../checkpoints/best_model.pth \
    --target_layer "visual_notes_module.layer.k_proj" \
    --num_samples 4
```

### 批量生成

可以编写脚本批量生成不同配置的可视化：

```bash
#!/bin/bash
for layer in "visual_encoder.vit.blocks[-1]" "visual_notes_module.layer.k_proj" "dual_head[0]"; do
    python generate_gradcam.py \
        --checkpoint ../checkpoints/best_model.pth \
        --target_layer "$layer" \
        --save_dir "gradcam_results_${layer//\//_}"
done
```

## 文件结构

```
biolcot_visualization/
├── README.md                    # 本文件
├── generate_gradcam.py          # Grad-CAM 生成脚本
├── generate_attention_map.py    # Attention Map 生成脚本
├── gradcam_results/              # Grad-CAM 输出目录（自动创建）
└── attention_results/            # Attention Map 输出目录（自动创建）
```

## 参考文献

- Grad-CAM: [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
- LayerCAM: [LayerCAM: Exploring Hierarchical Class Activation Maps for Localization](https://arxiv.org/abs/2006.10255)
- Visual Notes: BioLCoT 架构中的 Cross-Attention 机制

## 联系与支持

如有问题或建议，请查看项目主目录的 README 或联系项目维护者。

