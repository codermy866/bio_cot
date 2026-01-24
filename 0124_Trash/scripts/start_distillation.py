#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识蒸馏训练启动脚本
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from training.knowledge_distillation_training import train_distillation

if __name__ == '__main__':
    print("=" * 80)
    print("🎓 知识蒸馏训练启动")
    print("=" * 80)
    
    # ========== 配置参数 ==========
    
    # 教师模型路径（需要先训练好大模型）
    # 选项1: CNN Enhanced模型（如果已训练完成）
    # teacher_model_path = 'cnn_result_enhanced/best_model.pth'
    
    # 选项2: CNN Large模型（如果已训练完成）
    # teacher_model_path = 'cnn_result_large/best_model.pth'
    
    # 选项3: VMamba Large模型（如果已训练完成）
    teacher_model_path = 'vmamba_result_large/best_model.pth'
    
    # 检查教师模型是否存在
    if not os.path.exists(teacher_model_path):
        print(f"❌ 教师模型不存在: {teacher_model_path}")
        print("\n💡 请先训练教师模型，或修改teacher_model_path指向已训练好的大模型")
        print("\n可选方案:")
        print("  1. 等待CNN Enhanced模型训练完成")
        print("  2. 等待VMamba Large模型训练完成")
        print("  3. 使用已有的大模型作为教师")
        sys.exit(1)
    
    # 蒸馏参数
    epochs = 20
    batch_size = 6
    learning_rate = 2e-5  # 蒸馏通常使用稍低的学习率
    temperature = 3.0      # 温度参数（建议3-5）
    alpha = 0.7           # 蒸馏权重（建议0.5-0.7）
    
    # 数据路径
    data_path = '5centers_multi'
    output_dir = 'cnn_result_distilled'
    
    print("\n📋 蒸馏配置:")
    print(f"  教师模型: {teacher_model_path}")
    print(f"  温度参数: {temperature}")
    print(f"  蒸馏权重: {alpha}")
    print(f"  学习率: {learning_rate}")
    print(f"  批次大小: {batch_size}")
    print(f"  训练轮数: {epochs}")
    print(f"  输出目录: {output_dir}")
    print("=" * 80)
    print()
    
    # 切换到项目根目录
    os.chdir(project_root)
    
    # 启动蒸馏训练
    train_distillation(
        teacher_model_path=teacher_model_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        temperature=temperature,
        alpha=alpha,
        data_path=data_path,
        output_dir=output_dir
    )

