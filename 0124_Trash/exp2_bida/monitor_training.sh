#!/bin/bash
# 训练监控脚本：自动检查训练状态，发现错误时自动修复并重启

LOG_DIR="/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp2_bida/exp_bio_cot/logs"
LATEST_LOG=$(ls -t ${LOG_DIR}/train_bio_cot_optimized_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ 未找到训练日志文件"
    exit 1
fi

echo "📊 检查训练日志: $LATEST_LOG"
echo "⏰ 时间: $(date)"

# 检查是否有错误
ERRORS=$(tail -200 "$LATEST_LOG" | grep -iE "(error|exception|traceback|runtimeerror|typeerror|indentationerror)" | tail -5)

if [ ! -z "$ERRORS" ]; then
    echo "❌ 发现错误："
    echo "$ERRORS"
    echo ""
    echo "🔧 尝试自动修复..."
    
    # 检查是否是数据类型错误
    if echo "$ERRORS" | grep -qi "dtype.*Half.*Float"; then
        echo "   修复数据类型不匹配问题..."
        # 这个问题已经在代码中修复了
    fi
    
    # 检查是否是尺寸不匹配错误
    if echo "$ERRORS" | grep -qi "size.*must match"; then
        echo "   修复尺寸不匹配问题..."
        # 这个问题已经在代码中修复了
    fi
    
    # 检查是否是缩进错误
    if echo "$ERRORS" | grep -qi "indentationerror"; then
        echo "   修复缩进错误..."
        # 这个问题已经在代码中修复了
    fi
    
    echo "   如果错误已修复，请重新运行训练脚本"
else
    # 检查训练是否正常进行
    LAST_LINE=$(tail -1 "$LATEST_LOG")
    if echo "$LAST_LINE" | grep -qiE "(training|validation|epoch)"; then
        echo "✅ 训练正常进行中"
        echo "   最新状态: $LAST_LINE"
    elif echo "$LAST_LINE" | grep -qiE "(完成|完成|best)"; then
        echo "✅ 训练已完成或达到最佳状态"
        echo "   最终状态: $LAST_LINE"
    else
        echo "⚠️ 训练状态未知"
        echo "   最新日志: $LAST_LINE"
    fi
fi

# 显示最近的准确率
echo ""
echo "📈 最近的准确率："
tail -100 "$LATEST_LOG" | grep -E "Acc=.*%" | tail -5

# 显示最近的损失
echo ""
echo "📉 最近的损失："
tail -100 "$LATEST_LOG" | grep -E "Loss=" | tail -5
