#!/bin/bash
# Swin-T OCT 单模态训练启动脚本
# 使用项目虚拟环境

cd "$(dirname "$0")/../../.." || exit 1

VENV_PYTHON="./my_retfound/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "错误: 虚拟环境 Python 不存在: $VENV_PYTHON"
    exit 1
fi

# 设置默认参数
DATA_PATH="${1:-5centers_multi}"
EPOCHS="${2:-20}"
BATCH_SIZE="${3:-8}"
LR="${4:-3e-5}"

# 创建输出目录
mkdir -p models/SwinT/_results/oct

echo "🚀 启动 Swin-T OCT 单模态训练"
echo "  数据路径: $DATA_PATH"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  学习率: $LR"
echo "  输出目录: models/SwinT/_results/oct"
echo ""

CUDA_VISIBLE_DEVICES=0 nohup "$VENV_PYTHON" models/SwinT/scripts/start_swin_oct.py \
    --data_path "$DATA_PATH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    > models/SwinT/_results/oct/train.log 2>&1 &

PID=$!
echo "✅ 训练已启动 (PID: $PID)"
echo "📊 查看日志: tail -f models/SwinT/_results/oct/train.log"
echo ""

