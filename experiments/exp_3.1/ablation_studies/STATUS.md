# 消融实验执行状态

## 🚀 自动执行中

所有消融实验已设置为**自动顺序执行**。监控脚本会自动检测每个实验的完成状态，并在前一个实验完成后自动启动下一个。

## 📊 当前状态

使用以下命令查看实时状态：

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_3.1
python ablation_studies/monitor_and_run.py --check-only
```

## 🔄 实验执行顺序

1. ✅ **Baseline** - 基础模型（移除所有高级模块）
2. ⏳ **w/o Visual Notes** - 移除增强型视觉笔记模块
3. ⏳ **w/o Alignment Loss** - 移除语义-视觉对齐损失
4. ⏳ **w/o OT Loss** - 移除 Optimal Transport 损失
5. ⏳ **w/o Dual Head** - 移除双头因果解耦模块
6. ⏳ **w/o Adaptive Gating** - 移除自适应模态门控
7. ⏳ **w/o Cross-Attention** - 移除 Cross-Attention 机制

## 📈 查看结果

所有实验完成后，运行以下命令对比结果：

```bash
python ablation_studies/compare_results.py
```

这将生成：
- 控制台对比表格
- `comparison_results.csv` 文件

## 💡 提示

- 每个实验大约需要 **2-4 小时**（取决于GPU性能）
- 所有实验使用相同的随机种子（42）确保可复现性
- 日志文件保存在各自的 `logs/` 目录中
- 检查点保存在各自的 `checkpoints/` 目录中

## 🛑 停止实验

如果需要停止所有实验：

```bash
# 停止所有训练进程
ps aux | grep "train_bio_cot_v3.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9

# 停止监控脚本
ps aux | grep "monitor_and_run.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

## 📝 实验日志位置

- Baseline: `ablation_studies/baseline/logs/`
- w/o Visual Notes: `ablation_studies/w/o_visual_notes/logs/`
- w/o Alignment Loss: `ablation_studies/w/o_alignment_loss/logs/`
- w/o OT Loss: `ablation_studies/w/o_ot_loss/logs/`
- w/o Dual Head: `ablation_studies/w/o_dual_head/logs/`
- w/o Adaptive Gating: `ablation_studies/w/o_adaptive_gating/logs/`
- w/o Cross-Attention: `ablation_studies/w/o_cross_attn/logs/`

