#!/bin/bash
cd /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713
source my_retfound/bin/activate

echo "🚀 启动优化后的CLIP训练"
echo "📋 优化策略："
echo "  1. 暂时关闭所有辅助损失（KL=0, CLIP=0, Interv=0）"
echo "  2. 只训练分类任务，让模型先学会基础分类"
echo "  3. 当AUC>0.5后，自动启用辅助损失"
echo "  4. 使用Focal Loss + Label Smoothing（参考成功配置）"
echo "  5. 改进KL损失计算，防止过大"
echo ""

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

echo "✅ 训练已启动，PID: $!"
echo "📁 日志文件: adaptive_causal_intervention_results/train_optimized_clip.log"
echo ""
echo "🔍 监控命令："
echo "  tail -f adaptive_causal_intervention_results/train_optimized_clip.log"

