<!--
文件生成信息:
- 生成时间: 2025-12-25 22:30:00 (使用nvidia-smi获取CUDA时间)
- 生成需求: 用户要求自动开始执行训练
- 生成原因: 记录训练启动状态和关键信息
- 相关任务: 训练执行和监控

文件功能: 记录训练启动状态、关键参数、监控命令等
-->

# 训练启动状态报告

**启动时间**: 2025-12-25 22:30:00  
**实验名称**: 多中心对齐实验 (Multicenter Alignment)  
**状态**: 🚀 **训练已启动**

---

## 📊 训练配置

### 数据集
- **路径**: `/data2/hmy/5Center_datas/5centers_multi_leave_centers_out`
- **训练集**: 669样本 (train_labels.csv)
- **验证集**: 168样本 (val_labels.csv)
- **外部测试**: 148样本 (external_test_labels.csv)

### 模型配置
- **架构**: Enhanced Causal Bayesian CLIP
- **训练脚本**: `train.py`
- **Batch Size**: 8
- **Epochs**: 100
- **Learning Rate**: 3e-4
- **Device**: cuda:0
- **混合精度**: 启用

### 实验目录
- **检查点**: `./exp_multicenter_alignment/checkpoints`
- **日志**: `./exp_multicenter_alignment/logs`
- **结果**: `./exp_multicenter_alignment/results`

---

## 🔍 监控命令

### 查看训练日志
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp1_causal_bayesian_clip
tail -f exp_multicenter_alignment/logs/train_*.log
```

### 查看GPU使用情况
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

### 查看训练进程
```bash
ps aux | grep train.py | grep -v grep
```

### 查看最新检查点
```bash
ls -lht exp_multicenter_alignment/checkpoints/
```

---

## ⏱️ 预计时间

- **每个epoch**: ~5-10分钟
- **100个epoch**: ~8-16小时
- **预计完成**: 明天早上 (2025-12-26 06:00-14:00)

---

## 📝 关键信息

### 训练命令
```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp1_causal_bayesian_clip
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate
python train.py \
    --data_path /data2/hmy/5Center_datas/5centers_multi_leave_centers_out \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 3e-4 \
    --device cuda:0 \
    --num_workers 4 \
    --save_dir ./exp_multicenter_alignment/checkpoints
```

### 日志文件
- **位置**: `./exp_multicenter_alignment/logs/train_YYYYMMDD_HHMMSS.log`
- **实时查看**: `tail -f exp_multicenter_alignment/logs/train_*.log`

---

## ✅ 检查清单

- [x] GPU可用 (NVIDIA RTX A6000 x2)
- [x] 数据集路径正确
- [x] 实验目录创建
- [x] 训练脚本启动
- [ ] 训练正常进行 (需检查日志)
- [ ] 模型保存正常 (需检查检查点)

---

**状态**: 🚀 **训练已启动，请监控日志确认正常运行**

