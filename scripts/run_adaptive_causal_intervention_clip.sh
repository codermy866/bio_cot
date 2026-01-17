#!/bin/bash
# 自适应因果干预CLIP训练启动脚本

cd /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713

# 激活虚拟环境
source my_retfound/bin/activate

# 设置参数
DATA_PATH="5centers_multi"
OUTPUT_DIR="adaptive_causal_intervention_results"
BATCH_SIZE=6
NUM_EPOCHS=30
LEARNING_RATE=1e-4
GPU_ID=0

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 启动训练
echo "🚀 启动自适应因果干预CLIP训练..."
echo "📁 数据路径: ${DATA_PATH}"
echo "💾 输出目录: ${OUTPUT_DIR}"
echo "📦 批次大小: ${BATCH_SIZE}"
echo "🎯 训练轮数: ${NUM_EPOCHS}"
echo "📚 学习率: ${LEARNING_RATE}"
echo "🖥️  GPU: ${GPU_ID}"

CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python training/train_adaptive_causal_intervention_clip.py \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    > ${OUTPUT_DIR}/train.log 2>&1 &

echo "✅ 训练已在后台启动，日志文件: ${OUTPUT_DIR}/train.log"
echo "📊 查看训练进度: tail -f ${OUTPUT_DIR}/train.log"

