# 对比实验执行说明

## ✅ 实验已启动

完整的对比实验已经在后台运行，包括：

1. **Simple Fusion Baseline** - 特征拼接 + MLP
2. **Standard CLIP Baseline** - 标准CLIP方法  
3. **ViT + Clinical Fusion Baseline** - ViT特征 + 临床特征融合

## 📊 实验配置

- **运行次数**: 5次（使用不同随机种子：42, 123, 456, 789, 2024）
- **训练轮数**: 100 epochs
- **Batch Size**: 32
- **学习率**: 0.0002
- **优化器**: AdamW

## 🔍 检查实验状态

### 方法1: 查看GPU使用情况
```bash
nvidia-smi
```

### 方法2: 查看运行进程
```bash
ps aux | grep -E "train_simple_fusion|train_standard_clip|train_vit_clinical|run_and_visualize" | grep -v grep
```

### 方法3: 使用状态检查脚本
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0_improved
python comparison_experiments/check_status.py
```

### 方法4: 查看日志
```bash
# 查看最新日志
tail -f comparison_experiments/logs/run_full_*.log

# 查看所有日志
ls -lht comparison_experiments/logs/ | head -10
```

## 📁 结果文件位置

所有结果将保存在以下位置：

```
comparison_experiments/
├── results/
│   ├── baseline_simple_fusion/
│   │   ├── results/
│   │   │   ├── all_results.json      # 所有运行结果（JSON）
│   │   │   ├── all_results.csv       # 所有运行结果（CSV）
│   │   │   └── statistics.json        # 统计信息
│   │   └── logs/                      # 训练日志
│   ├── baseline_standard_clip/
│   │   └── ...
│   └── baseline_vit_clinical_fusion/
│       └── ...
├── visualizations/                    # 可视化图表（实验完成后生成）
│   ├── performance_comparison.png
│   ├── box_plots.png
│   ├── statistical_significance.png
│   ├── summary_table.png
│   └── summary_table.csv
└── logs/                              # 执行日志
    └── run_full_*.log
```

## ⏱️ 预计完成时间

- 每个baseline实验: 约2-3小时（取决于GPU性能）
- 总共3个baseline: 约6-9小时
- 可视化生成: 约1-2分钟

## 📈 实验完成后

实验完成后会自动：
1. ✅ 保存所有结果（JSON、CSV格式）
2. ✅ 计算统计信息（均值、标准差、置信区间）
3. ✅ 生成可视化图表
4. ✅ 保存执行日志

## 🛠️ 如果实验中断

如果实验意外中断，可以：

1. **检查结果**: 查看`comparison_experiments/results/`目录，看哪些实验已完成
2. **继续运行**: 使用`--skip_existing`参数跳过已完成的实验
   ```bash
   python comparison_experiments/run_and_visualize.py \
       --num_runs 5 \
       --num_epochs 100 \
       --batch_size 32 \
       --skip_existing
   ```
3. **单独运行**: 如果某个baseline失败，可以单独运行
   ```bash
   python comparison_experiments/baselines/simple_fusion/train_simple_fusion.py \
       --experiment_name baseline_simple_fusion \
       --num_runs 5 \
       --num_epochs 100 \
       --batch_size 32
   ```

## 📝 注意事项

1. **GPU内存**: 如果遇到OOM错误，可以减小batch_size
2. **数据路径**: 确保数据路径正确
3. **磁盘空间**: 确保有足够的磁盘空间保存结果
4. **日志文件**: 日志文件会持续增长，注意磁盘空间

## 🎯 下一步

实验完成后，可以：
1. 查看可视化图表
2. 分析统计显著性
3. 准备论文中的对比表格
4. 与Bio-COT 3.0 Improved方法进行对比

