# Bio-COT 3.0 训练日志文件路径

## 📁 主要日志文件

### 1. 最新训练日志（实时更新）
```
/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/logs/train_bio_cot_v3_20260112_194840.log
```
**说明**：这是训练脚本自动生成的日志文件，包含完整的训练输出

### 2. 实时输出日志（如果使用nohup）
```
/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/training_live.log
```
**说明**：使用nohup启动时的实时输出

### 3. 训练历史JSON（训练完成后生成）
```
/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/logs/training_history_20260112_194840.json
```
**说明**：包含所有epoch的训练指标（loss, acc, auc, f1等）

### 4. 可视化图表（训练完成后生成）
```
/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/logs/training_curves_20260112_194840.png
```
**说明**：训练曲线可视化图表

### 5. 最佳模型检查点
```
/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/checkpoints/best_model_v3_20260112_194840.pth
```
**说明**：保存的最佳模型权重

---

## 🔍 实时查看日志命令

### 方式1：实时跟踪最新日志
```bash
tail -f /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/logs/train_bio_cot_v3_20260112_194840.log
```

### 方式2：查看最后N行
```bash
tail -100 /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/logs/train_bio_cot_v3_20260112_194840.log
```

### 方式3：使用监控脚本
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0
python monitor_training.py
```

### 方式4：查看实时输出
```bash
tail -f /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/training_live.log
```

---

## 📊 当前训练状态

**训练进程PID**: 1736633  
**状态**: ✅ 正在运行  
**GPU使用**: cuda:0  
**当前进度**: 已完成2个epoch，正在进行第3个epoch

**已修复问题**:
- ✅ F1计算问题（添加zero_division参数和详细调试信息）
- ✅ 训练异常处理（添加try-catch捕获错误）
- ✅ 多帧/多图维度问题（已修复）

---

## ⚠️ 注意事项

1. **日志文件会自动创建**：每次运行训练会生成新的时间戳日志文件
2. **最佳模型会自动保存**：当验证AUC提升时会自动保存到checkpoints目录
3. **训练历史会自动保存**：训练完成后会生成JSON格式的历史记录

---

**最后更新**: 2025-01-12 20:32

