#!/bin/bash
# 检查Knowledge Note Embeddings是否生成完成，然后启动训练

cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

EMBED_FILE="data/knowledge_embeddings.pt"
LOG_FILE="generate_embeddings.log"

echo "=========================================="
echo "检查Knowledge Note Embeddings生成状态"
echo "=========================================="

# 检查文件是否存在
if [ -f "$EMBED_FILE" ]; then
    echo "✅ Knowledge Note Embeddings文件已存在: $EMBED_FILE"
    echo "   文件大小: $(du -h $EMBED_FILE | cut -f1)"
    echo ""
    echo "🚀 启动训练..."
    python training/train_bio_cot_v3.py 2>&1 | tee training_output.log
else
    echo "⏳ Knowledge Note Embeddings文件尚未生成"
    echo "   正在检查生成进度..."
    
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "📄 最新日志（最后10行）:"
        tail -10 $LOG_FILE
        echo ""
        echo "💡 提示: 使用 'tail -f $LOG_FILE' 实时查看生成进度"
    else
        echo "⚠️ 未找到生成日志文件"
    fi
    
    echo ""
    echo "请等待embeddings生成完成后再运行训练"
fi

