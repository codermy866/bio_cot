# 可视化图片对比说明

## 两种 OCT 图片的区别

脚本会生成两种 OCT 可视化图片，用于对比分析：

### 1. `raw_oct_sample{N}_{Positive/Negative}.png` - 原始 Grad-CAM

**含义**：标准的 Grad-CAM 可视化，未经过病灶聚焦处理

**特点**：
- ✅ 显示模型的所有激活区域（包括背景噪声）
- ✅ 保留完整的激活信息，不做过滤
- ✅ 适合用于：
  - 了解模型的原始关注点
  - 分析哪些区域被模型激活（包括可能的误激活）
  - 作为对比基准，查看病灶聚焦处理的效果

**处理流程**：
1. 使用 `generate_gradcam()` 生成原始 CAM
2. 应用 OCT 背景遮罩（去除顶部保护套、黑色背景）
3. 直接叠加到原图上

**示例**：
- `raw_oct_sample1_Negative.png` - 样本1的原始 Grad-CAM（阴性）
- `raw_oct_sample2_Positive.png` - 样本2的原始 Grad-CAM（阳性）

---

### 2. `lesion_focused_oct_sample{N}_{Positive/Negative}.png` - 病灶聚焦 Grad-CAM

**含义**：经过病灶聚焦处理的 Grad-CAM，只显示高置信度的病灶区域

**特点**：
- ✅ **只标注重点病灶区域**，去除背景噪声
- ✅ 使用百分位数阈值过滤（默认保留 top 40% 的激活）
- ✅ 高斯平滑处理，使病灶区域更集中、边缘更自然
- ✅ 适合用于：
  - **发表级可视化**（推荐）
  - 向审稿人展示模型关注的病灶区域
  - 证明模型能够准确定位病灶

**处理流程**：
1. 使用 `generate_gradcam()` 生成原始 CAM
2. 应用 OCT 背景遮罩（去除顶部保护套、黑色背景）
3. **病灶聚焦处理**（`apply_lesion_focus`）：
   - 去除顶部 20% 区域（保护套）
   - 百分位数阈值过滤（默认保留 top 40%）
   - 高斯平滑（sigma=1.0）
   - 重新归一化，使病灶更明显
4. 叠加到原图上

**示例**：
- `lesion_focused_oct_sample1_Negative.png` - 样本1的病灶聚焦 Grad-CAM（阴性）
- `lesion_focused_oct_sample2_Positive.png` - 样本2的病灶聚焦 Grad-CAM（阳性）

---

## 关键区别总结

| 特性 | `raw_oct_*.png` | `lesion_focused_oct_*.png` |
|------|----------------|---------------------------|
| **激活区域** | 所有激活（包括噪声） | 只保留高置信度病灶（top 40%） |
| **背景噪声** | 可能包含背景激活 | 已过滤背景噪声 |
| **阈值过滤** | ❌ 无 | ✅ 百分位数阈值（默认 0.6） |
| **高斯平滑** | ❌ 无 | ✅ 平滑处理（sigma=1.0） |
| **适用场景** | 调试、分析模型行为 | **发表级可视化**（推荐） |
| **视觉效果** | 可能包含较多蓝色/低激活区域 | 只显示红色/高激活区域 |

---

## 两种 Colposcopy 图片的区别

同样的逻辑也适用于 Colposcopy 图像：

### 1. `raw_colpo_sample{N}_{Positive/Negative}.png` - 原始 Grad-CAM

- 标准的 Grad-CAM，未经过病灶聚焦和颜色增强
- 显示所有激活区域

### 2. `lesion_focused_colpo_sample{N}_{Positive/Negative}.png` - 病灶聚焦 Grad-CAM

- 经过病灶聚焦处理
- **额外特性**：
  - ✅ 颜色先验增强：自动检测红色/糜烂区域（宫颈口流血）
  - ✅ 中心加权：宫颈口通常在中心，给予中心区域更高权重
  - ✅ 渐变边缘遮罩：去除边缘无关信息（窥器边缘、图像边界）
  - ✅ 反光去除：去除窥器反光区域

---

## 使用建议

### 用于论文发表
**推荐使用**：`lesion_focused_oct_*.png` 和 `lesion_focused_colpo_*.png`
- 只显示高置信度的病灶区域
- 视觉效果更清晰、专业
- 符合医学图像可视化的标准

### 用于调试和分析
**推荐使用**：`raw_oct_*.png` 和 `raw_colpo_*.png`
- 查看模型的完整激活信息
- 分析可能的误激活区域
- 理解模型的行为

### 对比分析
**同时查看两种图片**：
- 对比处理前后的差异
- 验证病灶聚焦处理的效果
- 确认过滤是否合理

---

## 参数调整

如果需要调整病灶聚焦的严格程度，可以修改 `generate_lesion_focused_gradcam.py` 的参数：

```bash
# 更严格（只保留 top 30%）
python generate_lesion_focused_gradcam.py --threshold 0.7 --num_samples 4

# 更宽松（保留 top 50%）
python generate_lesion_focused_gradcam.py --threshold 0.5 --num_samples 4
```

---

## 技术细节

### 病灶聚焦处理的核心步骤

1. **百分位数阈值过滤**：
   - 计算热力图的百分位数阈值（默认 60th percentile）
   - 只保留高于此阈值的区域
   - 自动适应不同图像的激活分布

2. **高斯平滑**：
   - 使用高斯模糊（sigma=1.0）平滑热力图
   - 使病灶区域更集中、边缘更自然

3. **重新归一化**：
   - 过滤后重新归一化到 [0, 1]
   - 确保病灶区域显示为最红

4. **OCT 特殊处理**：
   - 去除顶部 20% 区域（保护套）
   - 应用背景遮罩（去除黑色背景）

5. **Colposcopy 特殊处理**：
   - 颜色先验增强（红色/糜烂区域）
   - 中心加权
   - 渐变边缘遮罩
   - 反光去除

---

## 文件命名规则

- `raw_oct_sample{N}_{Positive/Negative}.png` - 原始版本
- `lesion_focused_oct_sample{N}_{Positive/Negative}.png` - 病灶聚焦版本
- `raw_colpo_sample{N}_{Positive/Negative}.png` - 原始版本
- `lesion_focused_colpo_sample{N}_{Positive/Negative}.png` - 病灶聚焦版本

其中：
- `{N}` = 样本编号（1, 2, 3, ...）
- `{Positive/Negative}` = 标签（Positive 或 Negative）

