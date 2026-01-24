# 消融实验执行状态

## ✅ 实验已启动

**启动时间**: 2026-01-20 21:31  
**监控脚本**: 运行中（PID: 3955365）  
**当前实验**: Baseline（PID: 3954853）

## 📊 实验执行计划

所有实验将**自动顺序执行**，监控脚本会在前一个实验完成后自动启动下一个。

### 实验列表（按顺序）

1. ✅ **Baseline** - 运行中
   - 描述: 移除所有高级模块，仅保留基础分类
   - 配置: `ablation_studies/baseline/config.py`
   - 状态: 🟢 运行中

2. ⏳ **w/o Visual Notes** - 等待中
   - 描述: 移除增强型视觉笔记模块
   - 配置: `ablation_studies/w/o_visual_notes/config.py`
   - 状态: ⚪ 未开始

3. ⏳ **w/o Alignment Loss** - 等待中
   - 描述: 移除语义-视觉对齐损失
   - 配置: `ablation_studies/w/o_alignment_loss/config.py`
   - 状态: ⚪ 未开始

4. ⏳ **w/o OT Loss** - 等待中
   - 描述: 移除 Optimal Transport 损失
   - 配置: `ablation_studies/w/o_ot_loss/config.py`
   - 状态: ⚪ 未开始

5. ⏳ **w/o Dual Head** - 等待中
   - 描述: 移除双头因果解耦模块
   - 配置: `ablation_studies/w/o_dual_head/config.py`
   - 状态: ⚪ 未开始

6. ⏳ **w/o Adaptive Gating** - 等待中
   - 描述: 移除自适应模态门控
   - 配置: `ablation_studies/w/o_adaptive_gating/config.py`
   - 状态: ⚪ 未开始

7. ⏳ **w/o Cross-Attention** - 等待中
   - 描述: 移除 Cross-Attention 机制
   - 配置: `ablation_studies/w/o_cross_attn/config.py`
   - 状态: ⚪ 未开始

## 🔍 查看实验状态

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_3.1
python ablation_studies/monitor_and_run.py --check-only
```

## 📈 预计完成时间

- **每个实验**: 约 30-60 分钟（20个epoch，取决于GPU性能）
- **全部完成**: 约 3.5-7 小时

## 📝 实验日志位置

- Baseline: `ablation_studies/baseline/logs/`
- w/o Visual Notes: `ablation_studies/w/o_visual_notes/logs/`
- w/o Alignment Loss: `ablation_studies/w/o_alignment_loss/logs/`
- w/o OT Loss: `ablation_studies/w/o_ot_loss/logs/`
- w/o Dual Head: `ablation_studies/w/o_dual_head/logs/`
- w/o Adaptive Gating: `ablation_studies/w/o_adaptive_gating/logs/`
- w/o Cross-Attention: `ablation_studies/w/o_cross_attn/logs/`

## 🔄 监控脚本

监控脚本每 5 分钟检查一次实验状态，自动启动下一个实验。

**监控日志**: `ablation_studies/monitor.log`

## 📊 结果对比

所有实验完成后，运行：

```bash
python ablation_studies/compare_results.py
```

这将生成：
- 控制台对比表格
- `comparison_results.csv` 文件

## 🛑 停止实验

如果需要停止所有实验：

```bash
# 停止所有训练进程
ps aux | grep "train_bio_cot_v3.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9

# 停止监控脚本
ps aux | grep "monitor_and_run.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

## 💡 提示

- 所有实验使用相同的随机种子（42）确保可复现性
- 所有实验使用相同的训练/验证集划分
- 训练轮数统一为 **20 epochs**（已更新）
- Batch Size 统一为 48

