#!/bin/bash
# Swin-S/B 大模型训练启动脚本
# 使用超强数据增强和更大的Swin模型

cd "$(dirname "$0")/.." || exit 1

VENV_PYTHON="./my_retfound/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "错误: 虚拟环境 Python 不存在: $VENV_PYTHON"
    exit 1
fi

# 设置参数
SWIN_MODEL="${1:-swin_small}"  # swin_tiny, swin_small, swin_base
DATA_PATH="${2:-5centers_multi}"
EPOCHS="${3:-30}"
BATCH_SIZE="${4:-4}"
LR="${5:-2e-5}"
INPUT_SIZE="${6:-224}"
OCT_FRAMES="${7:-48}"
GPU_ID="${8:-0}"  # 默认使用GPU 0
GRAD_ACCUM="${9:-1}"  # 梯度累积步数

# 创建输出目录
OUTPUT_DIR="swin_large_results/${SWIN_MODEL}_strong_aug"
mkdir -p "$OUTPUT_DIR"

echo "🚀 启动 Swin-${SWIN_MODEL} 大模型训练（超强数据增强）"
echo "  模型: ${SWIN_MODEL}"
echo "  GPU: ${GPU_ID}"
echo "  数据路径: ${DATA_PATH}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  梯度累积: ${GRAD_ACCUM}"
echo "  学习率: ${LR}"
echo "  输入尺寸: ${INPUT_SIZE}"
echo "  OCT帧数: ${OCT_FRAMES}"
echo "  输出目录: ${OUTPUT_DIR}"
echo ""

CUDA_VISIBLE_DEVICES=${GPU_ID} nohup "$VENV_PYTHON" training/train_swin_large.py \
    --swin_model "$SWIN_MODEL" \
    --data_path "$DATA_PATH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --output_dir "$OUTPUT_DIR" \
    --input_size "$INPUT_SIZE" \
    --oct_frames "$OCT_FRAMES" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    > "${OUTPUT_DIR}/train.log" 2>&1 &

PID=$!
echo "✅ 训练已启动 (PID: $PID)"
echo "📊 查看日志: tail -f ${OUTPUT_DIR}/train.log"
echo ""

