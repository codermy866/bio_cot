# 变量名冲突修复

## ❌ 错误原因

在 `encode_image` 方法中：
```python
B, F, C, H, W = images.shape
```

这里 `F` 被赋值为帧数（一个整数），覆盖了文件顶部导入的：
```python
import torch.nn.functional as F
```

导致后续调用 `F.normalize()` 时出现：
```
AttributeError: 'int' object has no attribute 'normalize'
```

## ✅ 修复方案

将解包变量名从 `F` 改为 `num_frames`，避免覆盖 `torch.nn.functional`：

```python
# 修复前
B, F, C, H, W = images.shape  # F被赋值为整数

# 修复后
B, num_frames, C, H, W = images.shape  # 使用num_frames，不覆盖F
```

## 📝 修复的文件

- ✅ `medclip/train_medclip.py` - 已修复变量名冲突

## ✅ 验证

修复后，`F.normalize()` 应该能正常工作，因为 `F` 仍然是 `torch.nn.functional` 模块。

---

**修复完成！变量名冲突已解决！** 🎉

