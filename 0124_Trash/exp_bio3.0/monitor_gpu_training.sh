#!/bin/bash
# GPU训练实时监控脚本

echo "=========================================="
echo "Bio-COT 3.0 GPU训练监控"
echo "=========================================="

while true; do
    clear
    echo "=========================================="
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    
    # GPU状态
    echo ""
    echo "📊 GPU状态:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
    
    # 训练进程
    echo ""
    echo "🔄 训练进程:"
    ps aux | grep "train_bio_cot_v3" | grep -v grep | head -1
    
    # 最新日志（最后5行）
    echo ""
    echo "📝 最新日志（最后5行）:"
    if [ -f "training_live.log" ]; then
        tail -5 training_live.log
    elif [ -f "training_output.log" ]; then
        tail -5 training_output.log
    else
        LATEST_LOG=$(ls -t logs/train_bio_cot_v3_*.log 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            tail -5 "$LATEST_LOG"
        else
            echo "  未找到日志文件"
        fi
    fi
    
    echo ""
    echo "按 Ctrl+C 退出监控"
    sleep 3
done

