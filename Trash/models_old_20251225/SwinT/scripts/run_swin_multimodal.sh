#!/bin/bash
# Swin-T 多模态训练启动脚本
# 使用项目虚拟环境

cd "$(dirname "$0")/../../.." || exit 1

VENV_PYTHON="./my_retfound/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "错误: 虚拟环境 Python 不存在: $VENV_PYTHON"
    exit 1
fi

# 设置默认参数（优化版本）
DATA_PATH="${1:-5centers_multi}"
EPOCHS="${2:-30}"  # 从20增加到30，给予更多训练机会
BATCH_SIZE="${3:-5}"  # 从6降到5，适应48帧的更大内存需求
LR="${4:-3e-5}"

# 创建输出目录
mkdir -p models/SwinT/_results/multimodal

echo "🚀 启动 Swin-T 多模态训练"
echo "  数据路径: $DATA_PATH"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  学习率: $LR"
echo "  输出目录: models/SwinT/_results/multimodal"
echo ""

CUDA_VISIBLE_DEVICES=0 nohup "$VENV_PYTHON" models/SwinT/scripts/start_swin_multimodal.py \
    --data_path "$DATA_PATH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --oct_frames 48 \
    --input_size 192 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --persistent_workers \
    --oct_cache_dir oct_cache_optimized \
    > models/SwinT/_results/multimodal/train.log 2>&1 &

PID=$!
echo "✅ 训练已启动 (PID: $PID)"
echo "📊 查看日志: tail -f models/SwinT/_results/multimodal/train.log"
echo ""

