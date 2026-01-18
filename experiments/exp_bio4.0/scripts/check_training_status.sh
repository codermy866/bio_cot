#!/bin/bash
# 快速检查训练状态

EXP_DIR="/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio4.0"
cd "$EXP_DIR"

echo "=========================================="
echo "🔍 Bio-COT 4.0 训练状态检查"
echo "=========================================="

# 检查训练进程
echo "📊 训练进程状态:"
if pgrep -f "train_bio_cot_v4.py" > /dev/null; then
    ps aux | grep -E "train_bio_cot_v4" | grep -v grep | head -2
    echo "✅ 训练进程正在运行"
else
    echo "❌ 未找到训练进程"
fi

echo ""
echo "📁 最新日志文件:"
LATEST_LOG=$(ls -t logs/*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "   $LATEST_LOG"
    echo ""
    echo "📝 最新训练信息 (最后30行):"
    tail -30 "$LATEST_LOG" | grep -E "Epoch|AUC|准确率|损失|✅|❌" | tail -10
else
    echo "   ❌ 未找到日志文件"
fi

echo ""
echo "💾 GPU使用情况:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "   N/A"

echo ""
echo "📂 检查点文件:"
ls -lt checkpoints/*/best_model*.pth 2>/dev/null | head -3 || echo "   （暂无检查点）"

