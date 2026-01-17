#!/bin/bash
# CNN 多模态训练启动脚本
# 使用项目虚拟环境

cd "$(dirname "$0")/.." || exit 1

VENV_PYTHON="./my_retfound/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "错误: 虚拟环境 Python 不存在: $VENV_PYTHON"
    exit 1
fi

# 设置默认参数（与SwinT和VMamba统一）
DATA_PATH="${1:-5centers_multi}"
EPOCHS="${2:-30}"  # 统一到30 epochs
BATCH_SIZE="${3:-6}"  # CNN可以用稍大的batch size
LR="${4:-3e-5}"

# 创建输出目录
mkdir -p cnn_result_unified

echo "🚀 启动 CNN 多模态训练（统一设置）"
echo "============================================================"
echo "📋 训练参数:"
echo "  数据路径: $DATA_PATH"
echo "  训练轮数: $EPOCHS"
echo "  批次大小: $BATCH_SIZE"
echo "  学习率: $LR"
echo "  OCT帧数: 120"
echo "  输入尺寸: 224"
echo "  使用最优阈值: Youden指数方法"
echo "============================================================"

# 使用GPU 1（与VMamba共享，如果显存足够）
# 如果显存不足，可以改为GPU 0，但需要确保Swin-T训练完成
CUDA_VISIBLE_DEVICES=1 nohup "$VENV_PYTHON" training/optimized_2class_training.py \
    --data_path "$DATA_PATH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --oct_frames 120 \
    --input_size 224 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --persistent_workers \
    --oct_cache_dir oct_cache_optimized \
    --output_dir cnn_result_unified \
    > cnn_result_unified/train.log 2>&1 &

PID=$!
echo "✅ CNN训练已启动 (PID: $PID)"
echo "📁 日志文件: cnn_result_unified/train.log"
echo "📊 监控命令: tail -f cnn_result_unified/train.log"
echo ""

