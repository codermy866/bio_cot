# 对比实验运行状态

## ✅ 当前状态

### 运行中的进程
- **并行调度脚本**: `run_all_comparisons_parallel.py` (PID: 1528968)
- **最大并行数**: 3个实验同时运行
- **运行模式**: 智能调度，根据GPU显存自动分配

### GPU使用情况
- **GPU 0**: 6177MB / 42499MB 使用 (利用率: 98%) - 正在运行Bio-COT 3.2训练
- **GPU 1**: 19MB / 48657MB 使用 (利用率: 0%) - **完全空闲，可用于并行运行**

### 实验进度

#### Bio-COT 3.2 (Full) - 我们的方法
- ✅ Run 1 (Seed 42): 已完成/运行中
- ✅ Run 2 (Seed 123): 已完成/运行中  
- ✅ Run 3 (Seed 456): 已完成/运行中
- ✅ Run 4 (Seed 789): 已完成/运行中
- ⏳ Run 5 (Seed 2024): 运行中

#### 其他SOTA方法（自动调度中）
- ⏳ MedCLIP: 等待调度
- ⏳ ConVIRT: 等待调度
- ⏳ mmFormer: 等待调度
- ⏳ Swin-T_Fusion: 等待调度

---

## 📁 文件位置

### 日志文件
- **主日志**: `run_parallel_*.log` - 并行调度脚本的日志
- **实验日志**: `logs/*.log` - 每个实验运行的详细日志

### 结果文件
- **结果目录**: `results/` - 每个实验的结果
- **配置汇总**: `results/experiment_config.json` - 实验配置
- **结果汇总**: `results/results_summary.json` - 最终结果汇总

---

## 🔍 监控命令

### 快速查看状态
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/comparison_experiments
./check_status.sh
```

### 详细监控
```bash
python monitor_comparisons.py
```

### 持续监控（每60秒刷新）
```bash
python monitor_comparisons.py --watch
```

### 查看最新日志
```bash
tail -f run_parallel_*.log
tail -f logs/*.log
```

---

## ⚙️ 并行运行说明

### 智能调度机制
1. **GPU显存检查**: 自动检查每个GPU的可用显存
2. **任务队列**: 所有任务放入队列，按顺序调度
3. **并行执行**: 根据可用显存，最多同时运行3个实验
4. **自动重试**: 如果显存不足，等待后重试

### 预估显存需求
- Bio-COT 3.2: ~4000MB
- MedCLIP: ~3000MB
- ConVIRT: ~3500MB
- mmFormer: ~4000MB
- Swin-T_Fusion: ~2500MB

### 当前GPU容量
- GPU 0: 49140MB 总容量
- GPU 1: 49140MB 总容量

**结论**: GPU 1完全空闲，可以同时运行多个实验！

---

## 📊 预计完成时间

- **总任务数**: 5个方法 × 5次运行 = 25个任务
- **当前进度**: Bio-COT 3.2的5次运行正在进行中
- **剩余任务**: 20个任务（4个方法 × 5次）

**预计时间**: 
- 如果并行运行3个: ~80小时（约3-4天）
- 如果并行运行更多: 可进一步缩短

---

## ✅ 注意事项

1. **不要手动终止**: 并行脚本会自动管理所有任务
2. **GPU 0正在使用**: Bio-COT训练在GPU 0上，不要干扰
3. **GPU 1完全空闲**: 可以充分利用进行并行运行
4. **自动调度**: 脚本会智能分配GPU，无需手动干预

---

**最后更新**: 2026-01-28 15:36

