#!/bin/bash

# 增强版OCT训练脚本
# 最大化利用OCT图像信息，提升训练准确性

set -e

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

# 项目路径
PROJECT_DIR="/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713"
cd "$PROJECT_DIR"

# 激活虚拟环境
source my_retfound/bin/activate

# 创建输出目录
OUTPUT_DIR="experiments_staged_fusion/outputs_enhanced_oct_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# 训练参数
EPOCHS_STAGE1=12
EPOCHS_STAGE2=8
BATCH_SIZE=4  # 减小batch size以适应增强特征
EMBED_DIM=512
NUM_WORKERS=0  # 避免多进程问题

echo "🚀 开始增强版OCT训练..."
echo "📁 输出目录: $OUTPUT_DIR"
echo "🔧 训练参数: Stage1=${EPOCHS_STAGE1}epochs, Stage2=${EPOCHS_STAGE2}epochs, batch_size=${BATCH_SIZE}, embed_dim=${EMBED_DIM}"
echo "🎯 重点优化: 最大化OCT图像信息利用"

# 运行训练
python experiments_staged_fusion/staged_fusion_enhanced_oct.py \
    --data_path 5centers_multi \
    --output_dir "$OUTPUT_DIR" \
    --epochs_stage1 $EPOCHS_STAGE1 \
    --epochs_stage2 $EPOCHS_STAGE2 \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --embed_dim $EMBED_DIM \
    --num_classes 2 \
    --metric_threshold 0.7 \
    --gpu 0 \
    --freeze_backbone_epochs 3 \
    --grad_accum_steps 2 \
    --input_size 224 \
    --oct_num_frames 48 \
    --oct_points 12 \
    --oct_frames_per_point 10 \
    --oct_cache_dir oct_cache_enhanced

echo "✅ 增强版OCT训练完成！"
echo "📊 结果保存在: $OUTPUT_DIR"
echo "🎯 重点优化了OCT图像处理，包括："
echo "   - 按点位智能分组和聚合"
echo "   - 多尺度特征提取"
echo "   - 时序一致性分析"
echo "   - 图像质量评估和加权融合"
echo "   - 增强的跨模态注意力机制"









