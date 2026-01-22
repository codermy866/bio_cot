#!/bin/bash
# 组合消融实验自动执行脚本
# 用法: ./run_combo_ablation.sh [start_from]

cd "$(dirname "$0")/.."  # 进入 experiments/exp_3.1 目录

START_FROM="${1:-baseline}"  # 默认从 baseline 开始

echo "=================================================================================="
echo "🚀 启动组合消融实验自动执行"
echo "=================================================================================="
echo "起始实验: $START_FROM"
echo "GPU策略: 自动选择空闲GPU (0,1)"
echo "显存阈值: >= 12000MB"
echo "利用率阈值: <= 60%"
echo "检查间隔: 120秒"
echo "=================================================================================="
echo ""

# 使用 nohup 后台运行，日志保存到 combo_ablation_runner.log
nohup python -u ablation_studies/auto_sequential_ablation.py \
    --auto_gpu \
    --gpus "0,1" \
    --gpu_strategy random \
    --min_free_mem_mb 12000 \
    --max_util 60 \
    --interval 120 \
    --start_from "$START_FROM" \
    > ablation_studies/combo_ablation_runner_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo "✅ 调度器已启动 (PID: $PID)"
echo "📝 日志文件: ablation_studies/combo_ablation_runner_*.log"
echo ""
echo "💡 查看实时日志:"
echo "   tail -f ablation_studies/combo_ablation_runner_*.log"
echo ""
echo "💡 查看调度器进程:"
echo "   ps aux | grep auto_sequential_ablation.py | grep -v grep"
echo ""

