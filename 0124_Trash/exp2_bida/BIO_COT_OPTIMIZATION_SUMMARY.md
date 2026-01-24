# Bio-COT 训练参数优化总结

## 优化时间
**开始时间:** 2025-12-30 09:09:51
**日志文件:** `train_bio_cot_optimized_bs64_20251230_090951.log`

## 优化内容

### 1. 加速训练参数

| 参数 | 原值 | 优化值 | 说明 |
|------|------|--------|------|
| **Batch Size** | 32 | **64** | 增加batch size，加速训练，提高GPU利用率 |
| **Student Prior Epochs** | 50 | **20** | 减少预训练轮数，快速进入主训练 |
| **Num Workers** | 8 | **2** | 减少数据加载进程，降低内存占用 |
| **Warmup Epochs** | 5 | **3** | 更快进入正常训练阶段 |

### 2. 提高效果参数

| 参数 | 原值 | 优化值 | 说明 |
|------|------|--------|------|
| **Learning Rate** | 2e-4 | **3e-4** | 提高学习率，加快收敛速度 |
| **Weight Decay** | 5e-4 | **1e-3** | 增强正则化，防止过拟合 |
| **Student Prior LR** | 1e-3 | **2e-3** | 加快Student Prior预训练收敛 |

### 3. 损失权重优化

| 损失项 | 原值 | 优化值 | 说明 |
|--------|------|--------|------|
| **lambda_cls** | 1.0 | **2.0** | 加强分类损失，提高分类准确性 |
| **lambda_ot** | 1.0 | **1.5** | 加强Sinkhorn OT损失，提高语义对齐 |
| **lambda_consist** | 2.0 | **3.0** | 加强反事实一致性，强化因果解耦 |
| **lambda_adv** | 0.5 | **1.0** | 加强对抗损失，提高域不变性 |

### 4. 数据加载优化

- **pin_memory=True**: 加速GPU数据传输
- **prefetch_factor=2**: 预取2个batch，减少等待时间
- **persistent_workers=True**: 保持worker进程，避免重复创建

## 预期效果

1. **训练速度提升**: 
   - Batch size增加2倍，每个epoch时间减少约50%
   - Student Prior预训练从50 epochs减少到20 epochs，节省60%时间

2. **模型效果提升**:
   - 更高的学习率和损失权重，加快收敛
   - 更强的正则化，提高泛化能力
   - 更平衡的损失权重，各模块协同工作

## 监控命令

```bash
# 查看训练日志
tail -f experiments/exp2_bida/exp_bio_cot/logs/train_bio_cot_optimized_bs64_20251230_090951.log

# 检查进程状态
ps aux | grep train_bio_cot

# 检查GPU使用
nvidia-smi
```

