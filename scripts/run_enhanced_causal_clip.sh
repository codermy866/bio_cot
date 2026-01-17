#!/bin/bash
# 增强因果CLIP训练启动脚本
# 核心创新：可学习因果图发现 + 不确定性分解

cd /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713

# 激活虚拟环境
source my_retfound/bin/activate

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 设置参数
BATCH_SIZE=4
EPOCHS=30
LR=1e-4
OUTPUT_DIR="enhanced_causal_clip_results"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 启动训练
echo "🚀 开始训练增强因果CLIP（可学习因果图 + 不确定性分解）..."
echo "📁 输出目录: ${OUTPUT_DIR}"
echo "📦 批次大小: ${BATCH_SIZE}"
echo "🎯 训练轮数: ${EPOCHS}"
echo "📚 学习率: ${LR}"
echo ""

python training/train_enhanced_causal_clip.py \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --output_dir ${OUTPUT_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/train.log

echo ""
echo "✅ 训练完成！结果保存在: ${OUTPUT_DIR}"

