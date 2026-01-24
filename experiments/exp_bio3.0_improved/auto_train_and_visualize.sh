#!/bin/bash
# 自动训练和生成可视化结果的脚本

cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0_improved
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

echo "=========================================="
echo "Bio-COT 3.0 Improved 自动训练和可视化"
echo "=========================================="
echo ""

# 1. 检查必要的文件
echo "📋 检查必要的文件..."
if [ ! -d "models" ]; then
    echo "  ⚠️ models目录不存在，从exp_bio3.0复制..."
    cp -r ../exp_bio3.0/models .
fi
if [ ! -d "data" ]; then
    echo "  ⚠️ data目录不存在，从exp_bio3.0复制..."
    cp -r ../exp_bio3.0/data .
fi
if [ ! -d "training" ]; then
    echo "  ⚠️ training目录不存在，从exp_bio3.0复制..."
    cp -r ../exp_bio3.0/training .
fi
if [ ! -d "knowledge_base" ]; then
    echo "  ⚠️ knowledge_base目录不存在，从exp_bio3.0复制..."
    cp -r ../exp_bio3.0/knowledge_base .
fi
echo "  ✅ 文件检查完成"
echo ""

# 2. 检查Knowledge Embeddings
echo "📊 检查Knowledge Embeddings..."
if [ ! -f "../exp_bio3.0/data/knowledge_embeddings.pt" ]; then
    echo "  ⚠️ Knowledge Embeddings不存在，需要生成..."
    echo "  请先运行: cd ../exp_bio3.0 && python knowledge_base/generate_knowledge_notes.py"
    exit 1
fi
echo "  ✅ Knowledge Embeddings存在"
echo ""

# 3. 修改训练脚本以使用新配置和最优阈值
echo "🔧 准备训练脚本..."
# 训练脚本会从当前目录的config.py读取配置
echo "  ✅ 配置已就绪"
echo ""

# 4. 开始训练
echo "🚀 开始训练..."
echo "  训练日志将保存到: logs/train_bio_cot_v3_improved_\$(date +%Y%m%d_%H%M%S).log"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_bio_cot_v3_improved_${TIMESTAMP}.log"

# 在后台运行训练
nohup python training/train_bio_cot_v3.py > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo "  ✅ 训练进程已启动 (PID: $TRAIN_PID)"
echo "  日志文件: $LOG_FILE"
echo "  使用以下命令监控训练:"
echo "    tail -f $LOG_FILE"
echo ""

# 5. 等待训练完成
echo "⏳ 等待训练完成..."
echo "  (训练可能需要数小时，请耐心等待...)"
echo ""

# 监控训练进程
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 60
    # 每60秒检查一次训练进度
    if [ -f "$LOG_FILE" ]; then
        LAST_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null)
        if [[ "$LAST_LINE" == *"Epoch"* ]] || [[ "$LAST_LINE" == *"Validation"* ]]; then
            echo "  $(date '+%Y-%m-%d %H:%M:%S'): $LAST_LINE"
        fi
    fi
done

# 检查训练是否成功完成
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 训练完成！"
    echo ""
    
    # 6. 生成可视化结果
    echo "📊 生成可视化结果..."
    echo ""
    
    # 生成基础可视化
    echo "  1. 生成基础可视化..."
    python ../exp_bio3.0/generate_sci_visualizations.py 2>&1 | tee logs/generate_sci_vis_${TIMESTAMP}.log
    
    # 生成3D可视化
    echo "  2. 生成3D可视化..."
    python ../exp_bio3.0/generate_3d_visualizations.py 2>&1 | tee logs/generate_3d_vis_${TIMESTAMP}.log
    
    # 生成补充可视化
    echo "  3. 生成补充可视化..."
    python ../exp_bio3.0/generate_additional_visualizations.py 2>&1 | tee logs/generate_add_vis_${TIMESTAMP}.log
    
    echo ""
    echo "✅ 所有可视化已完成！"
    echo ""
    echo "📁 结果文件位置:"
    echo "  - 训练日志: $LOG_FILE"
    echo "  - 模型检查点: checkpoints/"
    echo "  - 可视化图片: visualizations/ 和 logs/"
    echo ""
else
    echo ""
    echo "❌ 训练失败，请检查日志: $LOG_FILE"
    exit 1
fi

echo "=========================================="
echo "✅ 所有任务完成！"
echo "=========================================="

