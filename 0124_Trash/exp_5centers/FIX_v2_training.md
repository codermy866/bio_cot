# Bio-COT v2 训练修复方案

## 🔧 已应用的修复

### 1. 降低Batch Size
- **修改前**: `batch_size = 64`
- **修改后**: `batch_size = 16`
- **原因**: 64太大，可能导致梯度估计不准确

### 2. 降低学习率
- **修改前**: `scaled_lr = base_lr * (batch_size / 8)` = 0.000960
- **修改后**: `scaled_lr = base_lr * 2` = 0.00024
- **原因**: 0.000960太高，导致训练不稳定

### 3. 暂时关闭Cross-Attention
- **修改前**: `use_cross_attn = True`
- **修改后**: `use_cross_attn = False` (使用简单拼接，与v1一致)
- **原因**: 先验证LLM嵌入是否有效，再逐步添加Cross-Attention

## 📊 预期改进

修复后预期性能：
- **AUC**: 从0.5993提升到0.75-0.80
- **训练稳定性**: 损失值更稳定，梯度范数更合理
- **收敛速度**: 可能更快收敛

## 🚀 下一步

1. 重新启动训练，观察效果
2. 如果AUC达到0.75+，再逐步开启Cross-Attention
3. 如果仍然不理想，考虑使用传统MLP（use_llm=False）作为对比
