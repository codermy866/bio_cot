#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面诊断报告
分析当前训练状态、数据完整性、模型架构和下一步建议
"""

import json
import os
import pandas as pd
import numpy as np
import torch
import subprocess
from datetime import datetime

print("🔍 HPV-TCT多模态诊断报告")
print("=" * 80)

# ============================================================================
# 第一部分: 当前2分类模型状态
# ============================================================================

print("\n【第一部分】当前2分类模型状态")
print("-" * 80)

try:
    with open('cnn_training_latest/history.json', 'r') as f:
        history = json.load(f)
    
    with open('cnn_training_latest/results_calibrated.json', 'r') as f:
        calibrated = json.load(f)
    
    checkpoint = torch.load('cnn_training_latest/best_model.pth', map_location='cpu')
    
    print("✅ 模型基本信息:")
    print(f"   架构: CNN Multimodal Transformer")
    print(f"   参数量: {sum(p.numel() for p in checkpoint['state_dict'].values()):,}")
    print(f"   训练轮数: {checkpoint.get('epoch', 'N/A')}")
    print(f"   验证准确率: {history['val_acc'][0]:.4f}")
    print(f"   验证F1分数: {history['val_f1'][0]:.4f}")
    
    print("\n✅ 校准后性能:")
    print(f"   准确率: {calibrated.get('calibrated_accuracy', 'N/A'):.4f}")
    print(f"   F1分数: {calibrated.get('calibrated_best_f1', 'N/A'):.4f}")
    print(f"   最佳阈值: {calibrated.get('calibrated_best_th', 'N/A'):.4f}")
    print(f"   精确率: {calibrated.get('calibrated_precision', 'N/A'):.4f}")
    print(f"   召回率: {calibrated.get('calibrated_recall', 'N/A'):.4f}")
    
    # 性能评估
    val_acc = calibrated.get('calibrated_accuracy', 0)
    val_f1 = calibrated.get('calibrated_best_f1', 0)
    
    if val_acc > 0.8 and val_f1 > 0.7:
        grade = "⭐⭐⭐ 优秀"
    elif val_acc > 0.75 and val_f1 > 0.65:
        grade = "⭐⭐ 良好"
    elif val_acc > 0.65 and val_f1 > 0.5:
        grade = "⭐ 一般"
    else:
        grade = "⚡ 需改进"
    
    print(f"\n🎯 性能评级: {grade}")
    
except Exception as e:
    print(f"❌ 无法读取2分类模型信息: {e}")

# ============================================================================
# 第二部分: 数据完整性检查
# ============================================================================

print("\n【第二部分】数据完整性检查")
print("-" * 80)

# 检查2分类数据
print("📊 2分类数据:")
try:
    train_df = pd.read_csv('5centers_multi/train_labels.csv')
    test_df = pd.read_csv('5centers_multi/test_labels.csv')
    
    print(f"   训练集: {len(train_df)} 样本")
    print(f"   测试集: {len(test_df)} 样本")
    print(f"   标签分布 - 训练集: {train_df['label'].value_counts().to_dict()}")
    print(f"   标签分布 - 测试集: {test_df['label'].value_counts().to_dict()}")
    
    # 检查数据完整性
    train_oct = train_df['OCT'].notna().sum()
    train_col = train_df['ID'].notna().sum()
    train_clinical = train_df['AGE'].notna().sum()
    
    print(f"   完整样本 - 训练集: {train_oct}/{len(train_df)} ({train_oct/len(train_df)*100:.1f}%)")
    
except Exception as e:
    print(f"❌ 无法读取2分类数据: {e}")

# 检查5分类数据
print("\n📊 5分类数据:")
try:
    train_5class = pd.read_csv('5centers_multi_5class/train/train_labels.csv')
    test_5class = pd.read_csv('5centers_multi_5class/test/test_labels.csv')
    
    print(f"   训练集: {len(train_5class)} 样本")
    print(f"   测试集: {len(test_5class)} 样本")
    
    if 'tct_5class' in train_5class.columns:
        print("   标签分布:")
        dist = train_5class['tct_5class'].value_counts().sort_index()
        class_names = ['NILM', 'ASC-US', 'LSIL', 'HSIL', 'Cancer']
        for idx, count in dist.items():
            print(f"     类别{idx} ({class_names[idx]}): {count} 样本")
        
        # 计算类别平衡性
        min_count = dist.min()
        max_count = dist.max()
        balance = min_count / max_count if max_count > 0 else 0
        print(f"   类别平衡性: {balance:.3f} {'(较差)' if balance < 0.1 else '(一般)' if balance < 0.3 else '(较好)'}")
    
except Exception as e:
    print(f"⚠️ 5分类数据不完整: {e}")

# 检查图像数据
print("\n📁 图像数据检查:")
try:
    # 检查OCT目录
    oct_dirs = []
    for root, dirs, files in os.walk('5centers_multi'):
        if 'oct' in root.lower() and files:
            oct_dirs.append(root)
    
    print(f"   OCT目录数: {len(oct_dirs)}")
    if oct_dirs:
        sample_dir = oct_dirs[0]
        file_count = len(os.listdir(sample_dir))
        print(f"   示例目录 {os.path.basename(sample_dir)}: {file_count} 文件")
    
    # 检查Colposcopy目录  
    col_dirs = []
    for root, dirs, files in os.walk('5centers_multi'):
        if 'col' in root.lower() and files:
            col_dirs.append(root)
    
    print(f"   Colposcopy目录数: {len(col_dirs)}")
    
except Exception as e:
    print(f"⚠️ 无法检查图像数据: {e}")

# ============================================================================
# 第三部分: GPU资源检查
# ============================================================================

print("\n【第三部分】GPU资源检查")
print("-" * 80)

try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv'], 
                           capture_output=True, text=True)
    print(result.stdout)
    
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,gpu_uuid,used_memory', '--format=csv'], 
                           capture_output=True, text=True)
    if 'PID' in result.stdout:
        print("\n当前GPU进程:")
        print(result.stdout)
    
except Exception as e:
    print(f"⚠️ 无法获取GPU信息: {e}")

# ============================================================================
# 第四部分: 模型文件检查
# ============================================================================

print("\n【第四部分】模型文件检查")
print("-" * 80)

model_files = []
for root, dirs, files in os.walk('.'):
    if 'cnn_training' in root and 'best_model.pth' in files:
        model_path = os.path.join(root, 'best_model.pth')
        stat = os.stat(model_path)
        size_mb = stat.st_size / 1024 / 1024
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        model_files.append({
            'path': root,
            'size_mb': size_mb,
            'mtime': mtime
        })

print(f"找到 {len(model_files)} 个已保存的模型:")
for i, m in enumerate(sorted(model_files, key=lambda x: x['mtime'], reverse=True)[:5], 1):
    print(f"  {i}. {os.path.basename(m['path'])}")
    print(f"     大小: {m['size_mb']:.1f} MB")
    print(f"     时间: {m['mtime']}")

# ============================================================================
# 第五部分: 5分类训练准备度评估
# ============================================================================

print("\n【第五部分】5分类训练准备度评估")
print("-" * 80)

issues = []
readiness_score = 0

print("📋 检查项目:")
print()

# 1. 数据标签
try:
    if os.path.exists('5centers_multi_5class/train/train_labels.csv'):
        print("  ✅ 5分类标签文件已创建")
        readiness_score += 20
    else:
        issues.append("5分类标签文件不存在")
        print("  ❌ 5分类标签文件不存在")
except:
    pass

# 2. 模型架构
try:
    if os.path.exists('cnn_multimodal_model_5class.py'):
        print("  ✅ 5分类模型架构已创建")
        readiness_score += 20
    else:
        issues.append("5分类模型架构不存在")
        print("  ❌ 5分类模型架构不存在")
except:
    pass

# 3. 训练脚本
try:
    if os.path.exists('proper_5class_training.py'):
        print("  ✅ 5分类训练脚本已创建")
        readiness_score += 20
    else:
        issues.append("5分类训练脚本不存在")
        print("  ❌ 5分类训练脚本不存在")
except:
    pass

# 4. 数据加载
try:
    # 检查是否能正确加载
    if os.path.exists('5centers_multi/train/train_labels.csv'):
        print("  ✅ 基础数据集可用")
        readiness_score += 20
    else:
        issues.append("基础数据集不存在")
        print("  ❌ 基础数据集不存在")
except:
    pass

# 5. GPU资源
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'], 
                           capture_output=True, text=True)
    gpu_utils = [int(x.strip().rstrip('%')) for x in result.stdout.strip().split('\n')]
    
    free_gpu = [i for i, util in enumerate(gpu_utils) if util < 10]
    if free_gpu:
        print(f"  ✅ GPU资源充足 (空闲GPU: {free_gpu})")
        readiness_score += 20
    else:
        issues.append("无空闲GPU资源")
        print("  ⚠️ 无空闲GPU资源")
except:
    print("  ⚠️ 无法检查GPU资源")

print()
print(f"📊 准备度评分: {readiness_score}/100")

if readiness_score >= 80:
    print("🎯 准备度: 优秀 - 可以立即开始5分类训练")
elif readiness_score >= 60:
    print("🎯 准备度: 良好 - 需要少量准备工作")
elif readiness_score >= 40:
    print("🎯 准备度: 一般 - 需要中等程度准备")
else:
    print("🎯 准备度: 较差 - 需要大量准备工作")

if issues:
    print("\n⚠️ 需要解决的问题:")
    for issue in issues:
        print(f"   - {issue}")

# ============================================================================
# 第六部分: 建议和下一步行动
# ============================================================================

print("\n【第六部分】建议和下一步行动")
print("-" * 80)

print("💡 基于当前状态的建议:")
print()

if readiness_score >= 80:
    print("✅ 推荐方案: 立即开始5分类训练")
    print("   1. 使用迁移学习从2分类扩展到5分类")
    print("   2. 预计训练时间: 2-3小时")
    print("   3. 预期准确率: 60-70%")
    
elif readiness_score >= 60:
    print("✅ 推荐方案: 先完成数据准备，然后开始训练")
    print("   1. 修复数据加载问题")
    print("   2. 验证数据完整性")
    print("   3. 使用轻量级模型快速验证")
    print("   4. 预计准备时间: 30-60分钟")
    
else:
    print("✅ 推荐方案: 先解决基础问题")
    print("   1. 完善5分类数据标签")
    print("   2. 检查数据完整性")
    print("   3. 准备GPU资源")
    print("   4. 预计准备时间: 1-2小时")

print()
print("📋 立即可执行的命令:")
print()
print("   选项A: 继续优化2分类模型")
print("   python improved_training.py")
print()
print("   选项B: 尝试5分类训练（需要修复数据）")
print("   python proper_5class_training.py")
print()
print("   选项C: 使用轻量级模型快速验证")
print("   python lightweight_5class_training.py")
print()
print("   选项D: 查看详细计划")
print("   cat 5CLASS_TRAINING_PLAN.md")

print("\n" + "=" * 80)
print("✅ 诊断报告生成完成！")



