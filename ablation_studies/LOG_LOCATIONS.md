# 监控日志位置说明

## 📁 日志目录结构

### 1. 自动运行脚本日志（主监控日志）
**位置**：`/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/ablation_studies/`

**文件命名**：`ablation_runner_*.log`

**示例**：
- `ablation_runner_complete_v5_20260123_100722.log`
- `ablation_runner_final_v5_20260123_100419.log`
- `ablation_runner_monitored_20260123_095713.log`

**内容**：包含所有实验的启动、完成状态、错误信息等

**查看方式**：
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/ablation_studies
tail -f ablation_runner_*.log  # 查看最新日志
```

---

### 2. 各实验的详细日志

每个实验都有独立的日志目录：

#### Baseline实验
**位置**：`ablation_studies/baseline/logs/`

**日志文件**：
- `nohup_baseline_*.log` - 完整的训练输出（推荐查看）
- `train_bio_cot_v3_*.log` - 训练过程详细日志
- `training_history_*.json` - 训练历史数据（JSON格式）

#### 其他消融实验
**位置**：`ablation_studies/w/o_*/logs/`

**日志文件**：
- `nohup_w_o_*.log` - 完整的训练输出
- `train_bio_cot_v3_*.log` - 训练过程详细日志
- `training_history_*.json` - 训练历史数据

---

## 🔍 快速查看日志命令

### 查看自动运行脚本日志（推荐）
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/ablation_studies
tail -f ablation_runner_*.log
```

### 查看baseline实验日志
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/ablation_studies
tail -f baseline/logs/nohup_baseline_*.log
```

### 查看所有实验的最新日志
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/ablation_studies
for exp in baseline w/o_*; do
    echo "=== $exp ==="
    ls -t $exp/logs/nohup_*.log 2>/dev/null | head -1 | xargs tail -20
done
```

---

## 📊 实时监控

### 使用监控脚本
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/ablation_studies
python monitor_experiments.py
```

### 持续监控（每60秒刷新）
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/ablation_studies
bash continuous_monitor.sh
```

---

## 📝 日志文件说明

| 日志文件 | 用途 | 位置 |
|---------|------|------|
| `ablation_runner_*.log` | 自动运行脚本的主日志 | `ablation_studies/` |
| `nohup_*.log` | 单个实验的完整输出 | `各实验/logs/` |
| `train_bio_cot_v3_*.log` | 训练过程详细日志 | `各实验/logs/` |
| `training_history_*.json` | 训练历史数据 | `各实验/logs/` |

