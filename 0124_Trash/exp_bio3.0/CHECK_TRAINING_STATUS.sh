#!/bin/bash
# 快速检查训练状态

cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0

echo "=========================================="
echo "Bio-COT 3.0 训练状态检查"
echo "=========================================="

# 训练进程
echo "🔄 训练进程:"
ps aux | grep "train_bio_cot_v3" | grep -v grep | head -2

# GPU状态
echo ""
echo "📊 GPU状态:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader

# 最新日志
echo ""
echo "📝 最新日志（最后20行）:"
LATEST_LOG=$(find logs -name "train_bio_cot_v3_*.log" -type f -exec ls -t {} \; 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    tail -20 "$LATEST_LOG"
else
    echo "   ⚠️ 未找到日志文件"
fi

echo ""
echo "=========================================="

