# CAM生成问题分析与修复步骤

## 🔍 问题诊断

### 问题1: 模型forward调用参数缺失
**当前代码**（第269行）：
```python
input_dict = {
    'oct_images': sample['oct'],
    'colposcopy_images': sample['colpo'],
    'clinical_features': sample['clinical']
}
oct_cam = grad_cam.generate_cam(input_dict, target_class=sample['label'])
```

**问题**：
- Bio-COT 3.2的forward方法需要**位置参数**，不是字典
- **缺少必需的`image_names`参数**（forward方法会报错）
- 参数名称不匹配（应该是`f_oct`, `f_colpo`，不是`oct_images`）

### 问题2: 目标层选择可能不正确
**当前代码**（第198-211行）：
```python
if hasattr(model, 'visual_encoder'):
    if hasattr(model.visual_encoder, 'blocks'):
        target_layer = model.visual_encoder.blocks[-1]
```

**问题**：
- Bio-COT 3.2使用`HierarchicalViT`，结构可能不同
- 需要找到正确的特征提取层

### 问题3: 梯度传播路径
**问题**：
- 如果目标层选择不对，梯度可能无法正确传播
- Transformer的梯度传播路径与CNN不同

## 📋 修复步骤（逐步进行）

### Step 1: 修复模型forward调用
**需要修改**：
1. 准备正确的参数格式
2. 提供`image_names`（必需）
3. 使用位置参数调用

### Step 2: 定位正确的目标层
**需要检查**：
1. 模型的实际结构
2. 找到特征提取的关键层
3. 验证梯度能否传播

### Step 3: 修复GradCAM类
**需要修改**：
1. 支持Bio-COT 3.2的特殊forward签名
2. 正确处理多模态输入
3. 分别处理OCT和Colposcopy

### Step 4: 测试单个样本
**需要验证**：
1. 能否成功调用模型forward
2. 能否获取梯度
3. 能否生成CAM

### Step 5: 批量生成
**最后一步**：
1. 批量处理多个样本
2. 保存可视化结果

