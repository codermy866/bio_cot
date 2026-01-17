#!/bin/bash
# 自动化脚本：等待embeddings生成完成后自动启动训练

cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

EMBED_FILE="data/knowledge_embeddings.pt"
MAX_WAIT=1800  # 最大等待30分钟
CHECK_INTERVAL=10  # 每10秒检查一次
WAITED=0

echo "=========================================="
echo "Bio-COT 3.0 自动启动训练脚本"
echo "=========================================="
echo "等待Knowledge Note Embeddings生成完成..."
echo ""

# 等待embeddings文件生成
while [ ! -f "$EMBED_FILE" ] && [ $WAITED -lt $MAX_WAIT ]; do
    sleep $CHECK_INTERVAL
    WAITED=$((WAITED + CHECK_INTERVAL))
    
    # 显示进度
    if [ -f "generate_embeddings.log" ]; then
        PROGRESS=$(tail -1 generate_embeddings.log | grep -oP '\d+%' | head -1 || echo "0%")
        echo "[$(date +%H:%M:%S)] 等待中... ($PROGRESS) 已等待: ${WAITED}秒"
    else
        echo "[$(date +%H:%M:%S)] 等待中... 已等待: ${WAITED}秒"
    fi
done

# 检查文件是否存在
if [ -f "$EMBED_FILE" ]; then
    FILE_SIZE=$(du -h "$EMBED_FILE" | cut -f1)
    echo ""
    echo "✅ Knowledge Note Embeddings已生成！"
    echo "   文件: $EMBED_FILE"
    echo "   大小: $FILE_SIZE"
    echo ""
    echo "🚀 启动训练..."
    echo "=========================================="
    
    # 启动训练
    python training/train_bio_cot_v3.py 2>&1 | tee training_output.log
else
    echo ""
    echo "❌ 超时：Knowledge Note Embeddings未在${MAX_WAIT}秒内生成"
    echo "   请检查 generate_embeddings.log 查看错误信息"
    exit 1
fi

