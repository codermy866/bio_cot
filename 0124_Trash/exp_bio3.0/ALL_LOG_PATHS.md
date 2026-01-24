# Bio-COT 3.0 所有日志文件路径

## 📁 最新训练日志（推荐查看）

```
/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/logs/train_bio_cot_v3_20260112_221231.log
```

**说明**: 这是最新的训练日志文件，实时更新训练进度。

---

## 📁 其他相关日志文件

### 1. 实时输出日志
```
/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/training_new_session.log
```

### 2. 历史训练日志
```
/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/logs/train_bio_cot_v3_20260112_211316.log
/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/logs/train_bio_cot_v3_20260112_205544.log
/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/logs/train_bio_cot_v3_20260112_211051.log
```

### 3. 训练历史JSON（训练完成后）
```
/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/logs/training_history_<timestamp>.json
```

### 4. 可视化图表（训练完成后）
```
/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/logs/training_curves_<timestamp>.png
```

### 5. 最佳模型检查点
```
/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/checkpoints/best_model_v3_<timestamp>.pth
```

---

## 🔍 查看日志命令

### 实时查看（推荐）
```bash
tail -f /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/logs/train_bio_cot_v3_20260112_221231.log
```

### 查看最后N行
```bash
tail -100 /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/logs/train_bio_cot_v3_20260112_221231.log
```

### 搜索特定内容
```bash
grep -E "(稀疏性损失|一致性损失|AUC)" /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0/logs/train_bio_cot_v3_20260112_221231.log
```

---

## 📊 当前训练状态

- **训练进程**: ✅ 正在运行（PID: 1752896）
- **GPU**: cuda:0（显存使用: 1197 MB）
- **最新日志**: `logs/train_bio_cot_v3_20260112_221231.log`

---

**最后更新**: 2025-01-12 22:15

