# 对比实验结果和日志说明文档

## 📊 结果保存位置

所有对比实验的结果都会自动保存到以下目录结构：

```
experiments/exp_bio3.2/comparison_experiments/results/
├── MedCLIP/
│   └── baseline_medclip_v3_2/
│       └── results/
│           ├── all_results.csv          # 所有运行结果的CSV格式
│           ├── all_results.json          # 所有运行结果的JSON格式
│           └── statistics.json           # 统计信息（均值、标准差等）
├── ConVIRT/
│   └── baseline_convirt_v3_2/
│       └── results/
│           ├── all_results.csv
│           ├── all_results.json
│           └── statistics.json
├── mmFormer/
│   └── baseline_mmformer_v3_2/
│       └── results/
│           ├── all_results.csv
│           ├── all_results.json
│           └── statistics.json
└── Swin-T_Fusion/
    └── baseline_swin_t_v3_2/
        └── results/
            ├── all_results.csv
            ├── all_results.json
            └── statistics.json
```

### 结果文件说明

1. **`all_results.csv`** - CSV格式，包含所有5次运行（5个不同seed）的详细结果
   - 列包括：`run_id`, `seed`, `auc`, `accuracy`, `precision`, `recall`, `specificity`, `f1_score`, `training_time` 等

2. **`all_results.json`** - JSON格式，与CSV内容相同，便于程序读取

3. **`statistics.json`** - 统计信息，包含每个指标的：
   - `mean`: 平均值
   - `std`: 标准差
   - `min`: 最小值
   - `max`: 最大值
   - `median`: 中位数

### 每个运行的结果

每个方法会运行5次（5个不同的seed），每次运行的结果保存在：
```
results/{方法名}/baseline_{方法名}_v3_2/run_{运行编号}_seed_{seed值}/
├── best_model.pth              # 最佳模型检查点
├── training_history_{seed}.json # 训练历史（每个epoch的指标）
└── config.json                  # 运行配置
```

## 📝 日志输出位置

所有对比实验的日志都会自动保存到：

```
experiments/exp_bio3.2/comparison_experiments/logs/
├── MedCLIP_sequential_{时间戳}.log
├── ConVIRT_sequential_{时间戳}.log
├── mmFormer_sequential_{时间戳}.log
└── Swin-T_Fusion_sequential_{时间戳}.log
```

### 日志文件说明

- **命名格式**: `{方法名}_sequential_{时间戳}.log`
- **时间戳格式**: `YYYYMMDD_HHMMSS` (例如: `20260128_214038`)
- **内容**: 包含完整的训练过程输出，包括：
  - 实验配置信息
  - 每个epoch的训练和验证指标
  - 进度条显示
  - 错误信息（如果有）

### 查看日志的方法

1. **查看最新日志**:
   ```bash
   ls -lt /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/comparison_experiments/logs/*.log | head -5
   ```

2. **实时查看日志** (类似 `tail -f`):
   ```bash
   tail -f /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/comparison_experiments/logs/MedCLIP_sequential_*.log
   ```

3. **查看日志最后几行**:
   ```bash
   tail -50 /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/comparison_experiments/logs/MedCLIP_sequential_*.log
   ```

4. **搜索日志中的特定内容**:
   ```bash
   grep "Epoch" /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/comparison_experiments/logs/MedCLIP_sequential_*.log
   ```

## 🔍 快速检查实验状态

### 检查哪些实验已完成

```bash
# 查看所有已保存的结果文件
find /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/comparison_experiments/results -name "all_results.csv" -type f
```

### 查看实验结果摘要

```bash
# 查看MedCLIP的结果
cat /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/comparison_experiments/results/MedCLIP/baseline_medclip_v3_2/results/statistics.json

# 查看所有方法的CSV结果
for dir in MedCLIP ConVIRT mmFormer Swin-T_Fusion; do
    echo "=== $dir ==="
    cat /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/comparison_experiments/results/$dir/baseline_*/results/all_results.csv
    echo ""
done
```

### 使用监控脚本

运行监控脚本查看实时状态：
```bash
/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/comparison_experiments/monitor_comparison_progress.sh
```

## 📈 结果文件示例

### all_results.csv 示例
```csv
run_id,seed,auc,accuracy,precision,recall,specificity,f1_score,training_time
1,42,0.7234,0.7143,0.6892,0.6545,0.7742,0.6712,1234.56
2,123,0.7456,0.7381,0.7123,0.6909,0.7852,0.7015,1256.78
...
```

### statistics.json 示例
```json
{
  "auc": {
    "mean": 0.7345,
    "std": 0.0123,
    "min": 0.7234,
    "max": 0.7567,
    "median": 0.7345
  },
  "accuracy": {
    "mean": 0.7262,
    "std": 0.0112,
    ...
  }
}
```

## ⚠️ 注意事项

1. **结果自动保存**: 每个实验完成后，结果会自动保存，无需手动操作
2. **日志实时写入**: 日志文件在训练过程中实时写入，可以随时查看
3. **多个日志文件**: 如果多次运行同一个实验，会生成多个日志文件（带不同时间戳）
4. **结果完整性**: 只有当所有5次运行都完成后，`statistics.json` 才会包含完整的统计信息

## 🚀 下一步

实验完成后，可以使用以下脚本汇总所有结果：
- 结果汇总脚本（如果存在）
- 统计分析脚本
- 可视化脚本

