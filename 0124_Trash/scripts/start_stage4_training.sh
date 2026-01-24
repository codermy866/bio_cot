#!/bin/bash
cd /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713
source my_retfound/bin/activate
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
    > adaptive_causal_intervention_results/train_stage4_all_innovations_fixed.log 2>&1 &

echo "训练已启动，PID: $!"
echo "日志文件: adaptive_causal_intervention_results/train_stage4_all_innovations_fixed.log"

