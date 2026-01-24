# Bio-COT 3.0 Improved 训练指南

## 🚀 快速开始

### 1. 启动训练
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0_improved
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate
bash start_training.sh
```

### 2. 监控训练进度
```bash
# 方法1: 使用监控脚本（推荐）
bash monitor_and_visualize.sh

# 方法2: 直接查看日志
tail -f logs/train_improved_*.log
```

### 3. 训练完成后生成可视化
```bash
# 如果使用monitor_and_visualize.sh，会自动生成
# 或者手动运行：
bash generate_all_visualizations.sh
```

---

## 📊 改进配置

### 损失函数权重调整
- **lambda_cls**: 1.0 → **2.0** (增加分类损失权重)
- **lambda_ot**: 0.8 → **0.5** (减少OT损失)
- **lambda_consist**: 0.3 → **0.2** (减少一致性损失)
- **lambda_adv**: 0.8 → **0.5** (减少对抗损失)

### 决策阈值
- **classification_threshold**: 0.5 → **0.580** (使用最优阈值)

### 训练参数
- **batch_size**: 48
- **num_epochs**: 100
- **learning_rate**: 0.0002

---

## 📁 文件说明

### 训练脚本
- `start_training.sh`: 启动训练
- `monitor_and_visualize.sh`: 监控训练并自动生成可视化
- `generate_all_visualizations.sh`: 生成所有可视化结果

### 配置文件
- `config.py`: 改进的配置文件

### 输出目录
- `logs/`: 训练日志和中间结果
- `checkpoints/`: 模型检查点
- `visualizations/`: 最终的可视化图片（PDF和PNG）

---

## 🔍 训练状态检查

### 检查训练是否在运行
```bash
# 查看训练PID
cat logs/train_pid.txt

# 检查进程
ps aux | grep train_bio_cot_v3.py
```

### 查看最新日志
```bash
tail -f logs/train_improved_*.log
```

### 查看训练历史
```bash
ls -lh logs/training_history_*.json
```

---

## 📊 预期结果

### 性能目标
- **准确率**: 75-80% (从73.81%进一步提升)
- **精确率**: 65-70%
- **召回率**: 70-75%
- **特异性**: 75-80%
- **F1分数**: 70-75%

### 可视化结果
训练完成后会自动生成：
1. 基础可视化（SCI论文级别）
2. 3D可视化
3. 补充可视化（ROC/PR曲线、混淆矩阵、错误分析等）

---

## ⚠️ 注意事项

1. **训练时间**: 100个epoch可能需要数小时，请耐心等待
2. **GPU使用**: 训练会自动选择使用率最低的GPU
3. **日志文件**: 所有日志保存在 `logs/` 目录
4. **检查点**: 最佳模型保存在 `checkpoints/best_model_v3_*.pth`

---

## 🆘 故障排除

### 训练中断
```bash
# 检查日志文件
tail -50 logs/train_improved_*.log

# 重新启动训练
bash start_training.sh
```

### 可视化生成失败
```bash
# 手动生成可视化
bash generate_all_visualizations.sh
```

### 内存不足
- 减少 `batch_size` 在 `config.py` 中
- 减少 `num_workers` 在 `config.py` 中

---

**最后更新**: 2025-01-13

