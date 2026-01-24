#!/usr/bin/env python3
# CNN统一训练启动脚本（与SwinT配置统一）
import subprocess
import os
import sys

# 切换到项目根目录
project_root = '/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713'
os.chdir(project_root)

# 创建输出目录
output_dir = os.path.join(project_root, 'cnn_result_unified')
os.makedirs(output_dir, exist_ok=True)

cmd = [
    os.path.join(project_root, 'my_retfound/bin/python'),
    os.path.join(project_root, 'training/optimized_2class_training.py'),
    '--data_path', os.path.join(project_root, '5centers_multi'),
    '--epochs', '30',  # 统一到30 epochs
    '--batch_size', '5',  # 统一到5，适应48帧
    '--learning_rate', '3e-5',  # 训练脚本内部会调整为2.10e-05
    '--output_dir', output_dir,
    '--backbone', 'cnn',  # 明确指定CNN backbone
    '--oct_frames', '48',  # 统一到48帧
    '--input_size', '192',
    '--num_workers', '8',
    '--prefetch_factor', '4',
    '--persistent_workers',
    '--oct_cache_dir', os.path.join(project_root, 'oct_cache_optimized')
]

log_file = os.path.join(output_dir, 'train.log')
with open(log_file, 'w') as f:
    subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'})

print("✅ CNN统一训练已启动（GPU 0）")
print(f"📁 输出目录: {output_dir}")
print(f"📊 监控命令: tail -f {log_file}")

