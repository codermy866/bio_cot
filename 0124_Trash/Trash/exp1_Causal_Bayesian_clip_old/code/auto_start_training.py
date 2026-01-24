#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动启动训练脚本"""

import subprocess
import sys
import os
from pathlib import Path

# 切换到项目目录
project_root = Path(__file__).resolve().parent
os.chdir(project_root)

print("=" * 80)
print("🚀 自动启动优化后的CLIP训练")
print("=" * 80)
print()

# 停止旧进程
print("1. 停止旧训练进程...")
try:
    subprocess.run(['pkill', '-f', 'train_clip_with_innovations.py'], 
                   check=False, capture_output=True)
    print("   ✅ 已停止旧进程")
except:
    print("   ℹ️  没有运行中的进程")

# 激活环境并启动训练
print("\n2. 启动新训练...")
cmd = [
    'bash', '-c',
    'source my_retfound/bin/activate && '
    'CUDA_VISIBLE_DEVICES=1 nohup python training/train_clip_with_innovations.py '
    '--batch_size 5 '
    '--num_epochs 30 '
    '--data_path 5centers_multi '
    '--output_dir adaptive_causal_intervention_results '
    '--learning_rate 2.1e-5 '
    '--input_size 192 '
    '--oct_frames 48 '
    '--stage 4 '
    '--kl_weight 0.001 '
    '--contrastive_weight 0.05 '
    '--intervention_weight 0.01 '
    '--enable_auxiliary_after_auc 0.5 '
    '> adaptive_causal_intervention_results/train_optimized_clip.log 2>&1 &'
]

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(f"   ✅ 训练已启动")
print(f"   📁 日志: adaptive_causal_intervention_results/train_optimized_clip.log")
print()

# 等待并显示初始日志
print("3. 等待10秒后显示初始日志...")
import time
time.sleep(10)

log_file = project_root / 'adaptive_causal_intervention_results' / 'train_optimized_clip.log'
if log_file.exists():
    print("\n" + "=" * 80)
    print("📄 初始日志（最后40行）：")
    print("=" * 80)
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[-40:]:
            print(line.rstrip())
    print("=" * 80)
else:
    print("   ⚠️  日志文件尚未创建，请稍后检查")

print("\n✅ 训练启动完成！")
print("\n🔍 监控命令：")
print("   tail -f adaptive_causal_intervention_results/train_optimized_clip.log")
print("   nvidia-smi")

