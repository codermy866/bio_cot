#!/bin/bash
# Bio-COT 3.0 训练监控脚本

cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0

echo "=========================================="
echo "Bio-COT 3.0 训练监控"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 检查训练进程
echo "🔄 训练进程:"
TRAIN_PROCS=$(ps aux | grep "train_bio_cot_v3" | grep -v grep)
if [ -z "$TRAIN_PROCS" ]; then
    echo "   ❌ 没有训练进程在运行"
else
    echo "$TRAIN_PROCS" | head -3
    PROC_COUNT=$(echo "$TRAIN_PROCS" | wc -l)
    echo "   共 $PROC_COUNT 个训练进程"
fi

echo ""
echo "📊 GPU状态:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo ""
echo "📝 最新训练日志:"
LATEST_LOG=$(find logs -name "train_bio_cot_v3_*.log" -type f -exec ls -t {} \; 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "   文件: $LATEST_LOG"
    echo "   最后10行:"
    tail -10 "$LATEST_LOG" | sed 's/^/   /'
else
    echo "   ⚠️ 未找到日志文件"
fi

echo ""
echo "📈 训练历史:"
LATEST_HISTORY=$(find logs -name "training_history_*.json" -type f -exec ls -t {} \; 2>/dev/null | head -1)
if [ -n "$LATEST_HISTORY" ]; then
    echo "   文件: $LATEST_HISTORY"
    # 提取关键信息
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
        latest_sparse = data.get('sparse_loss', [0])[-1] if data.get('sparse_loss') else 0
        latest_consist = data.get('consist_loss', [0])[-1] if data.get('consist_loss') else 0
        print(f'   已完成Epoch: {epochs}/30')
        print(f'   最佳AUC: {best_auc:.4f}')
        print(f'   最新AUC: {latest_auc:.4f}')
        print(f'   最新稀疏性损失: {latest_sparse:.6f}')
        print(f'   最新一致性损失: {latest_consist:.6f}')
except Exception as e:
    print(f'   无法读取: {e}')
" 2>/dev/null || echo "   无法解析JSON"
    fi
else
    echo "   ⚠️ 未找到训练历史文件"
fi

echo ""
echo "=========================================="

