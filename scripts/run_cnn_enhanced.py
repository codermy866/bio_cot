#!/usr/bin/env python3
# CNN增强模型训练启动脚本
import subprocess
import os
import sys

os.chdir('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/cnn_result_enhanced')
cmd = [
    '../my_retfound/bin/python',
    '../training/optimized_2class_training.py',
    '--data_path', '../5centers_multi',
    '--epochs', '30',  # 统一到30 epochs
    '--batch_size', '5',  # 统一到5，适应48帧
    '--learning_rate', '3e-5',  # 训练脚本内部会调整为2.10e-05
    '--output_dir', '.',
    '--oct_frames', '48',  # 统一到48帧
    '--input_size', '192',
    '--num_workers', '8',
    '--prefetch_factor', '4',
    '--persistent_workers',
    '--oct_cache_dir', '../oct_cache_optimized'
]

with open('train.log', 'a') as f:
    subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '1'})

print("CNN增强模型训练已启动")

