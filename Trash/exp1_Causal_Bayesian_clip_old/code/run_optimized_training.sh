#!/bin/bash
set -e  # 遇到错误立即退出

cd /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713

# 激活环境
source my_retfound/bin/activate

# 停止旧进程
pkill -f train_clip_with_innovations.py || true
sleep 2

echo "=========================================="
echo "🚀 启动优化后的CLIP训练"
echo "=========================================="
echo ""
echo "📋 优化策略："
echo "  1. 暂时关闭所有辅助损失（KL=0, CLIP=0, Interv=0）"
echo "  2. 只训练分类任务，让模型先学会基础分类"
echo "  3. 当AUC>0.5后，自动启用辅助损失"
echo "  4. 使用Focal Loss + Label Smoothing（参考成功配置）"
echo "  5. 改进KL损失计算，防止过大"
echo ""
echo "📊 训练配置："
echo "  - GPU: CUDA设备1"
echo "  - Batch Size: 5"
echo "  - Learning Rate: 2.1e-5"
echo "  - Epochs: 30"
echo "  - Input Size: 192"
echo "  - OCT Frames: 48"
echo ""

# 启动训练
CUDA_VISIBLE_DEVICES=1 nohup python training/train_clip_with_innovations.py \
    --batch_size 5 \
    --num_epochs 30 \
    --data_path 5centers_multi \
    --output_dir adaptive_causal_intervention_results \
    --learning_rate 2.1e-5 \
    --input_size 192 \
    --oct_frames 48 \
    --stage 4 \
    --kl_weight 0.001 \
    --contrastive_weight 0.05 \
    --intervention_weight 0.01 \
    --enable_auxiliary_after_auc 0.5 \
    > adaptive_causal_intervention_results/train_optimized_clip.log 2>&1 &

TRAIN_PID=$!

echo "✅ 训练已启动"
echo "   PID: $TRAIN_PID"
echo "   Log: adaptive_causal_intervention_results/train_optimized_clip.log"
echo ""
echo "🔍 监控命令："
echo "   tail -f adaptive_causal_intervention_results/train_optimized_clip.log"
echo "   nvidia-smi"
echo ""
echo "等待10秒后显示初始日志..."
sleep 10

if [ -f adaptive_causal_intervention_results/train_optimized_clip.log ]; then
    echo ""
    echo "📄 初始日志（最后30行）："
    echo "=========================================="
    tail -n 30 adaptive_causal_intervention_results/train_optimized_clip.log
    echo "=========================================="
else
    echo "⚠️  日志文件尚未创建，请稍后检查"
fi


