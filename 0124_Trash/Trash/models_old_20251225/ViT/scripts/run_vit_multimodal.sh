#!/bin/bash
# ViT多模态训练启动脚本

cd /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713
source my_retfound/bin/activate

# 默认参数（减小batch size以节省显存，降低学习率以稳定训练）
EPOCHS=30
BATCH_SIZE=2
LR=1e-5
DATA_PATH="5centers_multi"
OUTPUT_ROOT="models/ViT/_results"
VIT_MODEL="vit_base_patch16_224"
INPUT_SIZE=224
OCT_FRAMES=48
GPU_ID=0

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --vit_model)
            VIT_MODEL="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "🚀 开始ViT多模态训练"
echo "============================================================"
echo "模型: $VIT_MODEL"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "GPU: $GPU_ID"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU_ID nohup python models/ViT/scripts/start_vit_multimodal.py \
    --data_path $DATA_PATH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --output_root $OUTPUT_ROOT \
    --oct_frames $OCT_FRAMES \
    --input_size $INPUT_SIZE \
    --num_workers 8 \
    --prefetch_factor 4 \
    --persistent_workers \
    --oct_cache_dir oct_cache_optimized \
    --vit_model $VIT_MODEL \
    > $OUTPUT_ROOT/multimodal/train.log 2>&1 &

echo "✅ 训练已在后台启动"
echo "📁 日志文件: $OUTPUT_ROOT/multimodal/train.log"
echo "📊 查看训练进度: tail -f $OUTPUT_ROOT/multimodal/train.log"

