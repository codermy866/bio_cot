#!/bin/bash
# 启动改进的训练，并自动生成可视化结果

cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0_improved
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

echo "=========================================="
echo "Bio-COT 3.0 Improved 自动训练"
echo "=========================================="
echo ""

# 确保必要的代码文件存在
if [ ! -d "models" ] || [ ! -d "data" ] || [ ! -d "training" ]; then
    echo "📋 复制必要的代码文件..."
    cp -r ../exp_bio3.0/models . 2>/dev/null
    cp -r ../exp_bio3.0/data . 2>/dev/null
    cp -r ../exp_bio3.0/training . 2>/dev/null
    cp -r ../exp_bio3.0/knowledge_base . 2>/dev/null
    echo "✅ 代码文件已复制"
    echo ""
fi

# 检查Knowledge Embeddings
if [ ! -f "../exp_bio3.0/data/knowledge_embeddings.pt" ]; then
    echo "❌ 错误: Knowledge Embeddings不存在"
    echo "请先运行: cd ../exp_bio3.0 && python knowledge_base/generate_knowledge_notes.py"
    exit 1
fi

# 创建符号链接到Knowledge Embeddings（如果不存在）
if [ ! -f "data/knowledge_embeddings.pt" ]; then
    mkdir -p data
    ln -s ../../exp_bio3.0/data/knowledge_embeddings.pt data/knowledge_embeddings.pt
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_improved_${TIMESTAMP}.log"

echo "🚀 开始训练..."
echo "  日志文件: $LOG_FILE"
echo "  配置: 使用改进的损失权重和类别权重"
echo "  最优阈值: 0.580"
echo ""

# 启动训练（后台运行）
nohup python training/train_bio_cot_v3.py > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo "✅ 训练进程已启动 (PID: $TRAIN_PID)"
echo ""
echo "📊 监控训练进度:"
echo "  tail -f $LOG_FILE"
echo ""
echo "⏳ 训练进行中，请等待完成..."
echo "  (训练完成后会自动生成可视化结果)"
echo ""

# 保存PID以便后续使用
echo $TRAIN_PID > logs/train_pid.txt
echo "训练PID已保存到: logs/train_pid.txt"

echo ""
echo "=========================================="
echo "训练已启动，请使用以下命令监控:"
echo "  tail -f $LOG_FILE"
echo "=========================================="

