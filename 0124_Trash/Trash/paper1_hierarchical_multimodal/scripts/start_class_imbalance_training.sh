#!/bin/bash
# 类别不平衡问题解决方案 - 训练脚本
# 使用方法: bash start_class_imbalance_training.sh [method] [gpu_id]
# method: weighted_ce (类别加权交叉熵) 或 focal (Focal Loss)
# gpu_id: 0 或 1

METHOD=${1:-focal}  # 默认使用Focal Loss
GPU_ID=${2:-0}

cd /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713
source my_retfound/bin/activate

OUTPUT_DIR="paper1_hierarchical_multimodal/results/cuda${GPU_ID}_class_imbalance_${METHOD}"
LOG_FILE="paper1_hierarchical_multimodal/logs/train_cuda${GPU_ID}_class_imbalance_${METHOD}.log"

echo "=========================================="
echo "类别不平衡问题解决方案训练"
echo "方法: ${METHOD}"
echo "GPU: cuda:${GPU_ID}"
echo "输出目录: ${OUTPUT_DIR}"
echo "日志文件: ${LOG_FILE}"
echo "=========================================="

if [ "$METHOD" == "weighted_ce" ]; then
    # 方法1: 类别加权交叉熵（简单有效）
    echo "使用类别加权交叉熵..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} PYTHONUNBUFFERED=1 nohup python -u \
        paper1_hierarchical_multimodal/training/train_hierarchical_multimodal.py \
        --data_path 5centers_multi \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 8 \
        --num_epochs 30 \
        --num_workers 4 \
        --device cuda \
        --log_interval 10 \
        --contrastive_weight 0.3 \
        --learning_rate 5e-5 \
        --weight_decay 1e-4 \
        --warmup_epochs 3 \
        --use_amp \
        --use_weighted_ce \
        --use_class_weights \
        --label_smoothing 0.1 \
        --input_size 224 \
        --oct_num_frames 32 \
        --oct_frames_per_point 5 \
        --max_grad_norm 0.5 \
        > ${LOG_FILE} 2>&1 &

elif [ "$METHOD" == "focal" ]; then
    # 方法2: 优化的Focal Loss
    echo "使用优化的Focal Loss..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} PYTHONUNBUFFERED=1 nohup python -u \
        paper1_hierarchical_multimodal/training/train_hierarchical_multimodal.py \
        --data_path 5centers_multi \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 4 \
        --num_epochs 30 \
        --num_workers 4 \
        --device cuda \
        --log_interval 10 \
        --contrastive_weight 0.3 \
        --learning_rate 5e-5 \
        --weight_decay 1e-4 \
        --warmup_epochs 3 \
        --use_amp \
        --use_focal_loss \
        --focal_gamma 4.0 \
        --use_class_weights \
        --label_smoothing 0.1 \
        --input_size 224 \
        --oct_num_frames 32 \
        --oct_frames_per_point 5 \
        --max_grad_norm 0.5 \
        --use_vit_backbone \
        --backbone_type vit \
        --vit_model_name vit_base_patch16_224 \
        > ${LOG_FILE} 2>&1 &

else
    echo "错误: 未知的方法 '${METHOD}'"
    echo "使用方法: bash start_class_imbalance_training.sh [weighted_ce|focal] [gpu_id]"
    exit 1
fi

PID=$!
echo "训练已启动，PID: ${PID}"
echo "查看日志: tail -f ${LOG_FILE}"
echo "检查进程: ps aux | grep ${PID}"

