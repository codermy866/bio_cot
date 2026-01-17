#!/bin/bash
# Vmamba统一训练设置启动脚本（与SwinT配置统一）

VENV_PYTHON="/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/my_retfound/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "错误: 虚拟环境 Python 不存在: $VENV_PYTHON"
    exit 1
fi

# 设置默认参数（与SwinT统一）
DATA_PATH="${1:-5centers_multi}"
EPOCHS="${2:-30}"  # 统一到30 epochs
BATCH_SIZE="${3:-5}"  # 统一到5，适应48帧
LR="${4:-3e-5}"

# 创建输出目录
mkdir -p vmamba_result_unified

echo "🚀 启动 Vmamba 多模态训练（统一设置）"
echo "============================================================"
echo "📋 训练参数:"
echo "  数据路径: $DATA_PATH"
echo "  训练轮数: $EPOCHS"
echo "  批次大小: $BATCH_SIZE"
echo "  学习率: $LR"
echo "  OCT帧数: 48"
echo "  输入尺寸: 192"
echo "============================================================"

cd /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713

CUDA_VISIBLE_DEVICES=1 nohup "$VENV_PYTHON" training/optimized_vmamba_training.py \
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
    --output_dir vmamba_result_unified \
    > vmamba_result_unified/train.log 2>&1 &

echo "✅ Vmamba训练已启动（后台运行）"
echo "📁 日志文件: vmamba_result_unified/train.log"
echo "📊 监控命令: tail -f vmamba_result_unified/train.log"

