#!/bin/bash
# 监控训练进度，训练完成后自动生成可视化

cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0_improved
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

LOG_FILE=$(ls -t logs/train_improved_*.log 2>/dev/null | head -1)
PID_FILE="logs/train_pid.txt"

if [ -z "$LOG_FILE" ]; then
    echo "❌ 未找到训练日志文件"
    exit 1
fi

if [ ! -f "$PID_FILE" ]; then
    echo "❌ 未找到训练PID文件"
    exit 1
fi

TRAIN_PID=$(cat "$PID_FILE")

echo "=========================================="
echo "监控训练进度"
echo "=========================================="
echo "日志文件: $LOG_FILE"
echo "训练PID: $TRAIN_PID"
echo ""
echo "按 Ctrl+C 停止监控（训练会继续在后台运行）"
echo ""

# 检查进程是否还在运行
if ! kill -0 $TRAIN_PID 2>/dev/null; then
    echo "⚠️ 训练进程已结束，检查训练状态..."
    # 检查最后几行日志，判断是否成功完成
    LAST_LINES=$(tail -5 "$LOG_FILE" 2>/dev/null)
    if echo "$LAST_LINES" | grep -q "训练完成\|Training completed\|Best model saved"; then
        echo "✅ 训练成功完成！"
        echo ""
        echo "📊 开始生成可视化结果..."
        bash generate_all_visualizations.sh
    else
        echo "❌ 训练可能异常结束，请检查日志: $LOG_FILE"
    fi
    exit 0
fi

# 监控训练进度
echo "开始监控训练进度..."
echo ""

while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 30  # 每30秒检查一次
    
    # 显示最新的训练进度
    if [ -f "$LOG_FILE" ]; then
        LAST_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null)
        if [[ "$LAST_LINE" == *"Epoch"* ]] || [[ "$LAST_LINE" == *"Validation"* ]] || [[ "$LAST_LINE" == *"loss"* ]]; then
            echo "[$(date '+%H:%M:%S')] $LAST_LINE"
        fi
    fi
done

# 训练完成
echo ""
echo "=========================================="
echo "训练已完成！"
echo "=========================================="
echo ""

# 检查训练是否成功
LAST_LINES=$(tail -10 "$LOG_FILE" 2>/dev/null)
if echo "$LAST_LINES" | grep -q "训练完成\|Training completed\|Best model saved"; then
    echo "✅ 训练成功完成！"
    echo ""
    echo "📊 开始生成可视化结果..."
    bash generate_all_visualizations.sh
    echo ""
    echo "=========================================="
    echo "✅ 所有任务完成！"
    echo "=========================================="
    echo ""
    echo "📁 结果位置:"
    echo "  - 训练日志: $LOG_FILE"
    echo "  - 模型检查点: checkpoints/"
    echo "  - 可视化图片: visualizations/"
else
    echo "⚠️ 训练可能异常结束，请检查日志: $LOG_FILE"
fi

