#!/bin/bash
# 自动训练监控脚本 - 持续监控训练状态，自动检测并修复问题

cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0

echo "=========================================="
echo "Bio-COT 3.0 自动训练监控"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 检查训练进程
TRAIN_PROCS=$(ps aux | grep "train_bio_cot_v3" | grep -v grep)
if [ -z "$TRAIN_PROCS" ]; then
    echo "❌ 没有训练进程在运行"
    echo "   尝试启动新训练..."
    source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate
    nohup python training/train_bio_cot_v3.py > training_auto_run.log 2>&1 &
    echo "✅ 已启动新训练进程"
    sleep 10
else
    echo "✅ 训练进程正在运行:"
    echo "$TRAIN_PROCS" | head -2
fi

echo ""
echo "📊 GPU状态:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo ""
echo "📝 最新训练日志（最后30行）:"
LATEST_LOG=$(find logs -name "train_bio_cot_v3_*.log" -type f -exec ls -t {} \; 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "   文件: $LATEST_LOG"
    tail -30 "$LATEST_LOG" | sed 's/^/   /'
    
    # 检查是否有错误
    if grep -q "Error\|Exception\|Traceback" "$LATEST_LOG"; then
        echo ""
        echo "⚠️  检测到错误，最后10行错误信息:"
        grep -A 5 "Error\|Exception\|Traceback" "$LATEST_LOG" | tail -10 | sed 's/^/   /'
    fi
    
    # 检查注意力是否坍塌
    if tail -50 "$LATEST_LOG" | grep -q "注意力均值: OCT=0\.0000"; then
        echo ""
        echo "⚠️  检测到注意力坍塌（OCT=0.0000）"
    fi
else
    echo "   ⚠️ 未找到日志文件"
fi

echo ""
echo "📈 训练进度:"
LATEST_HISTORY=$(find logs -name "training_history_*.json" -type f -exec ls -t {} \; 2>/dev/null | head -1)
if [ -n "$LATEST_HISTORY" ]; then
    if command -v python3 &> /dev/null; then
        python3 -c "
import json
import sys
try:
    with open('$LATEST_HISTORY', 'r') as f:
        data = json.load(f)
    epochs = len(data.get('val_auc', []))
    if epochs > 0:
        best_auc = max(data.get('val_auc', [0]))
        latest_auc = data.get('val_auc', [0])[-1] if data.get('val_auc') else 0
        latest_acc = data.get('val_acc', [0])[-1] if data.get('val_acc') else 0
        latest_sparse = data.get('sparse_loss', [0])[-1] if data.get('sparse_loss') else 0
        latest_attn_oct = data.get('attn_mean_oct', [0])[-1] if data.get('attn_mean_oct') else 0
        latest_attn_colpo = data.get('attn_mean_colpo', [0])[-1] if data.get('attn_mean_colpo') else 0
        print(f'   已完成Epoch: {epochs}/30')
        print(f'   最佳AUC: {best_auc:.4f}')
        print(f'   最新AUC: {latest_auc:.4f}')
        print(f'   最新准确率: {latest_acc:.4f}')
        print(f'   最新稀疏性损失: {latest_sparse:.6f}')
        print(f'   最新注意力均值: OCT={latest_attn_oct:.4f}, Colpo={latest_attn_colpo:.4f}')
        if latest_attn_oct < 0.01 or latest_attn_colpo < 0.01:
            print('   ⚠️  注意力值过低，可能存在问题')
except Exception as e:
    print(f'   无法读取: {e}')
" 2>/dev/null || echo "   无法解析JSON"
    fi
else
    echo "   ⚠️ 未找到训练历史文件"
fi

echo ""
echo "=========================================="

