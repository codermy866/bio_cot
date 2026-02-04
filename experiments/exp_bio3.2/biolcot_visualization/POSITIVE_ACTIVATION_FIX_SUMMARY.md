# 阳性病例激活修复总结

## 修改日期
2025-01-26

## 修改文件
`biolcot_visualization/generate_lesion_focused_gradcam.py`

## 实现的三个方案

### ✅ 方案1：使用预测类别而非真实标签

**位置**：`visualize_lesion_focused_gradcam` 函数（第532-563行）

**实现逻辑**：
- 在生成Grad-CAM之前，先获取模型的预测结果
- 对于阳性病例，如果预测概率很低（<0.3），使用**预测类别**而非真实标签
- 这样可以显示模型实际关注的区域，即使预测错误

**代码片段**：
```python
# 🔥 方案3：添加诊断信息
with torch.no_grad():
    output = model(...)
    probs = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()

# 🔥 方案1：使用预测类别而非真实标签
if label == 1 and probs[0, 1] < 0.3:
    target_class_for_cam = predicted_class  # 使用预测类别
    print(f"     ✅ 使用预测类别 {target_class_for_cam} 生成Grad-CAM")
else:
    target_class_for_cam = label  # 使用真实标签
    print(f"     ✅ 使用真实标签 {target_class_for_cam} 生成Grad-CAM")
```

### ✅ 方案2：对阳性病例降低阈值

**位置**：`apply_lesion_focus` 函数（第237-261行）

**实现逻辑**：
- 添加 `label` 参数到 `apply_lesion_focus` 函数
- 对于阳性病例（label=1），将阈值降低30%（threshold * 0.7）
- 这样可以保留更多的激活区域，避免过度过滤

**代码片段**：
```python
def apply_lesion_focus(..., label=None):
    # 🔥 方案2：对阳性病例降低阈值
    original_threshold = threshold
    if label == 1:  # 阳性病例
        threshold = threshold * 0.7  # 降低30%
        print(f"  📊 阳性病例：使用降低的阈值 {threshold:.3f} (原始: {original_threshold:.3f})")
```

### ✅ 方案3：添加诊断信息

**位置**：`visualize_lesion_focused_gradcam` 函数（第532-563行）

**实现逻辑**：
- 在生成CAM之前，显示模型预测诊断信息
- 包括：真实标签、预测类别、阴性概率、阳性概率
- 如果预测概率很低，给出警告

**代码片段**：
```python
# 🔥 方案3：添加诊断信息
print(f"\n  📊 模型预测诊断 (样本 {samples_collected + 1}):")
print(f"     真实标签: {label} ({label_str})")
print(f"     预测类别: {predicted_class} ({'Positive' if predicted_class == 1 else 'Negative'})")
print(f"     阴性概率: {probs[0, 0]:.4f}")
print(f"     阳性概率: {probs[0, 1]:.4f}")

# 如果预测概率很低，给出警告
if label == 1 and probs[0, 1] < 0.3:
    print(f"     ⚠️  警告：阳性概率很低 ({probs[0, 1]:.4f})，Grad-CAM可能不明显")
```

## 修改的函数签名

### 1. `apply_lesion_focus`
```python
# 修改前：
def apply_lesion_focus(heatmap, threshold=0.6, smooth_sigma=1.0, use_percentile=True, image_rgb=None, is_colposcopy=False):

# 修改后：
def apply_lesion_focus(heatmap, threshold=0.6, smooth_sigma=1.0, use_percentile=True, image_rgb=None, is_colposcopy=False, label=None):
```

### 2. `generate_lesion_focused_gradcam`
```python
# 修改前：
def generate_lesion_focused_gradcam(..., image_rgb: Optional[np.ndarray] = None) -> np.ndarray:

# 修改后：
def generate_lesion_focused_gradcam(..., image_rgb: Optional[np.ndarray] = None, label: Optional[int] = None) -> np.ndarray:
```

## 修改的调用位置

### 1. OCT CAM生成（第580-590行）
```python
# 修改前：
raw_oct_cam = generate_raw_gradcam(..., target_class=label, ...)
oct_cam = generate_lesion_focused_gradcam(..., target_class=label, ...)

# 修改后：
raw_oct_cam = generate_raw_gradcam(..., target_class=target_class_for_cam, ...)
oct_cam = generate_lesion_focused_gradcam(..., target_class=target_class_for_cam, ..., label=label)
```

### 2. Colposcopy CAM生成（第602-616行）
```python
# 修改前：
raw_colpo_cam = generate_raw_gradcam(..., target_class=label, ...)
colpo_cam = generate_lesion_focused_gradcam(..., target_class=label, ...)

# 修改后：
raw_colpo_cam = generate_raw_gradcam(..., target_class=target_class_for_cam, ...)
colpo_cam = generate_lesion_focused_gradcam(..., target_class=target_class_for_cam, ..., label=label)
```

## 预期效果

1. **更好的激活显示**：
   - 对于预测概率低的阳性病例，使用预测类别生成CAM，显示模型实际关注的区域
   - 降低阈值，保留更多激活区域

2. **诊断信息**：
   - 每次生成CAM前，显示模型预测信息
   - 帮助理解为什么某些病例没有激活

3. **自适应处理**：
   - 根据模型预测自动选择最佳策略
   - 对于预测准确的病例，使用真实标签
   - 对于预测不准确的病例，使用预测类别

## 使用说明

运行脚本时，会自动应用这些修复：

```bash
python generate_lesion_focused_gradcam.py --num_samples 4
```

**输出示例**：
```
📊 模型预测诊断 (样本 1):
    真实标签: 1 (Positive)
    预测类别: 0 (Negative)
    阴性概率: 0.8234
    阳性概率: 0.1766
    ⚠️  警告：阳性概率很低 (0.1766)，Grad-CAM可能不明显
    建议：将使用预测类别而非真实标签来生成Grad-CAM
    ✅ 使用预测类别 0 生成Grad-CAM（预测概率更高）
📊 阳性病例：使用降低的阈值 0.420 (原始: 0.600)
```

## 注意事项

1. **阈值调整**：如果仍然没有激活，可以进一步降低阈值（修改 `threshold * 0.7` 为 `threshold * 0.5`）

2. **预测概率阈值**：当前使用0.3作为阈值，如果模型预测更不准确，可以降低到0.2

3. **诊断信息**：每次运行都会显示诊断信息，帮助理解模型行为

---

**修改完成** ✅  
**测试建议**：运行脚本，检查阳性病例的激活效果是否改善

