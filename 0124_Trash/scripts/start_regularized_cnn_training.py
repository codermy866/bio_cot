#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动正则化优化的CNN训练
解决过拟合问题
"""

import os
import sys
import subprocess

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from training.optimized_2class_training import train_optimized_2class

if __name__ == '__main__':
    print("=" * 80)
    print("🚀 启动正则化优化的CNN训练")
    print("=" * 80)
    print("\n📋 优化措施:")
    print("  ✅ Dropout: 0.1 → 0.3 (提升3倍)")
    print("  ✅ Weight Decay: 2e-4 → 5e-4 (提升2.5倍)")
    print("  ✅ 学习率: 降低40% (3e-5 → 1.8e-5)")
    print("  ✅ 移除加权采样器 (避免过拟合)")
    print("  ✅ Early Stopping: patience=5")
    print("  ✅ 过拟合监控: 最大差距25%")
    print("=" * 80)
    print()
    
    # 训练参数
    epochs = 20
    batch_size = 6  # 稍微减小batch size以增加随机性
    learning_rate = 3e-5
    data_path = os.path.join(project_root, '5centers_multi')
    output_dir = os.path.join(project_root, 'cnn_result_regularized')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 切换到项目根目录
    os.chdir(project_root)
    
    # 启动训练
    train_optimized_2class(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        data_path='5centers_multi',
        output_dir='cnn_result_regularized'
    )

