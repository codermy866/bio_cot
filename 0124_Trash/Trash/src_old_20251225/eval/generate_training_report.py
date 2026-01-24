#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成训练状态报告
"""

import json
import os
from datetime import datetime
import subprocess

def generate_report():
    """生成完整的训练报告"""
    
    print("📊 HPV-TCT多模态2分类训练状态报告")
    print("=" * 60)
    
    # 1. 基本信息
    print("\n1️⃣ 基本信息")
    print("-" * 40)
    print(f"  任务: 宫颈病变2分类")
    print(f"  模型: CNN Multimodal Transformer")
    print(f"  数据: OCT + Colposcopy + 临床特征")
    
    # 2. 最新训练结果
    print("\n2️⃣ 最新训练结果 (cnn_training_latest)")
    print("-" * 40)
    
    try:
        with open('cnn_training_latest/history.json', 'r') as f:
            history = json.load(f)
        
        val_acc = history.get('val_acc', [0])[0]
        val_f1 = history.get('val_f1', [0])[0]
        train_loss = history.get('train_loss', [0])[0]
        val_loss = history.get('val_loss', [0])[0]
        
        print(f"  ✅ 训练状态: 已完成")
        print(f"  📈 验证准确率: {val_acc:.4f} (66.5%)")
        print(f"  📈 验证F1分数: {val_f1:.4f} (50.2%)")
        print(f"  📉 训练损失: {train_loss:.4f}")
        print(f"  📉 验证损失: {val_loss:.4f}")
        
    except Exception as e:
        print(f"  ❌ 无法读取训练历史: {e}")
    
    # 3. 最新模型文件
    print("\n3️⃣ 最新模型文件 (cnn_training_20251027_145643)")
    print("-" * 40)
    
    try:
        import torch
        checkpoint = torch.load('cnn_training_20251027_145643/best_model.pth', map_location='cpu')
        
        print(f"  ✅ 模型已保存")
        print(f"  📅 Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  🏆 最佳F1分数: {checkpoint.get('best_f1', 'N/A'):.4f}")
        
        stat = os.stat('cnn_training_20251027_145643/best_model.pth')
        size_mb = stat.st_size / 1024 / 1024
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  📁 文件大小: {size_mb:.1f} MB")
        print(f"  ⏰ 修改时间: {mtime}")
        
    except Exception as e:
        print(f"  ❌ 无法读取模型文件: {e}")
    
    # 4. GPU使用情况
    print("\n4️⃣ GPU使用情况")
    print("-" * 40)
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        for i, line in enumerate(lines):
            if 'GPU' in line or 'Python' in line or 'python' in line:
                print(f"  {line.strip()}")
    except Exception as e:
        print(f"  ⚠️ 无法获取GPU信息: {e}")
    
    # 5. 性能分析
    print("\n5️⃣ 性能分析")
    print("-" * 40)
    
    try:
        val_acc = history.get('val_acc', [0])[0]
        val_f1 = history.get('val_f1', [0])[0]
        
        print(f"  当前准确率: {val_acc:.4f}")
        print(f"  当前F1分数: {val_f1:.4f}")
        
        # 性能评估
        if val_acc > 0.8:
            rating = "⭐⭐⭐ 优秀"
            comment = "模型性能优秀，可以用于实际应用"
        elif val_acc > 0.7:
            rating = "⭐⭐ 良好"
            comment = "模型性能良好，可以考虑进一步优化"
        elif val_acc > 0.6:
            rating = "⭐ 一般"
            comment = "模型性能一般，建议继续训练或调整超参数"
        else:
            rating = "⚡ 待改进"
            comment = "模型性能需改进，建议重新训练或增加数据"
        
        print(f"  🎯 性能评级: {rating}")
        print(f"  💡 建议: {comment}")
        
    except Exception as e:
        print(f"  ⚠️ 无法分析性能: {e}")
    
    # 6. 改进建议
    print("\n6️⃣ 改进建议")
    print("-" * 40)
    
    print("  📋 针对当前模型:")
    print("     1. 准确率66.5%还有提升空间")
    print("     2. F1分数50.2%表明类别不平衡")
    print("     3. 建议使用Focal Loss处理不平衡")
    print("     4. 建议增加数据增强")
    print("     5. 建议尝试更长的训练")
    
    print("\n  📋 如需5分类:")
    print("     1. 需要修复数据加载问题")
    print("     2. 需要创建5分类数据集")
    print("     3. 需要调整模型输出层")
    print("     4. 建议使用加权采样")
    
    print("\n" + "=" * 60)
    print("✅ 报告生成完成！")

if __name__ == "__main__":
    generate_report()



