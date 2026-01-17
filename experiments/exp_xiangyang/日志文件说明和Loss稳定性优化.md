# 日志文件说明和Loss稳定性优化

## 📝 两个日志文件的区别

### 1. `train_bio_cot_xiangyang_20260106_103350.log`
- **来源**: 脚本内部写入的日志文件
- **位置**: 代码第465行 `log_file = args.log_dir / f'train_bio_cot_xiangyang_{timestamp}.log'`
- **内容**: 只记录每个epoch的摘要信息（Loss, Acc, AUC, F1等）
- **格式**: 简洁的摘要格式，便于快速查看训练进度

### 2. `train_bio_cot_xiangyang_balanced_20260106_103328.log`
- **来源**: 通过`tee`命令捕获的完整终端输出
- **位置**: 命令行 `python train_bio_cot_xiangyang.py 2>&1 | tee logs/train_bio_cot_xiangyang_balanced_$(date +%Y%m%d_%H%M%S).log`
- **内容**: 包含所有实时输出，包括：
  - tqdm进度条的每个batch更新
  - 所有警告信息
  - 完整的训练过程细节
- **格式**: 详细的实时输出，便于调试和分析

### 为什么需要两个日志？
- **摘要日志**: 快速查看训练进度和最终结果
- **完整日志**: 详细分析训练过程，调试问题，查看batch级别的loss变化

**注意**: 两个日志记录的是**同一次训练**，只是记录方式不同。

## 🔍 Loss不稳定的原因分析

从训练日志可以看出，loss曲线不够平滑，主要原因：

### 1. **没有梯度裁剪** ❌
- 当前代码没有使用`torch.nn.utils.clip_grad_norm_`
- 梯度爆炸会导致loss剧烈波动

### 2. **Focal Loss的特性** ⚠️
- Focal Loss关注难样本，batch之间的难样本分布不同
- 导致batch间loss差异较大

### 3. **Batch Size较小** ⚠️
- 当前batch_size=16，batch间方差较大
- 小batch导致loss波动更明显

### 4. **学习率可能偏大** ⚠️
- 当前lr=1.5e-4，可能导致训练不稳定
- 没有warmup机制

### 5. **没有使用EMA平滑** ❌
- 没有使用指数移动平均来平滑loss显示

## 🔧 优化方案

### 1. 添加梯度裁剪
```python
# 在loss.backward()之后添加
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 2. 降低学习率 + Warmup
```python
# 使用Warmup + CosineAnnealing
warmup_epochs = max(2, args.num_epochs // 8)
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (args.num_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### 3. 使用EMA平滑Loss显示
```python
# 使用指数移动平均平滑loss
ema_loss = 0.99 * ema_loss + 0.01 * loss.item()
```

### 4. 增加Batch Size（如果内存允许）
- 从16增加到32，减少batch间方差

### 5. 使用梯度累积
- 如果内存不足，使用梯度累积模拟更大的batch size





