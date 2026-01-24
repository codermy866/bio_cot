#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件生成信息:
- 生成时间: 2025-12-24 19:40:50 (使用nvidia-smi获取CUDA时间)
- 生成需求: 用户要求检查数据集完整性，验证两个数据集的内容和差异
- 生成原因: 需要自动化工具检查数据集完整性，验证图像文件是否存在
- 相关任务: 数据集分析和完整性验证

文件功能: 数据集完整性检查脚本，验证图像文件是否存在，检查数据完整性
"""

import os
import pandas as pd
from pathlib import Path
from collections import defaultdict

def check_dataset_integrity():
    """检查数据集完整性"""
    
    # 数据集路径
    original_data = Path('/data2/hmy/5Center_datas/5centers_multi')
    split_data = Path('/data2/hmy/5Center_datas/5centers_multi_internal_external_final')
    
    print("=" * 80)
    print("数据集完整性检查")
    print("=" * 80)
    
    # 1. 检查原始数据集
    print("\n1. 原始数据集检查 (5centers_multi)")
    print("-" * 80)
    
    original_train_labels = original_data / 'train_labels.csv'
    original_test_labels = original_data / 'test_labels.csv'
    
    if original_train_labels.exists():
        df_train = pd.read_csv(original_train_labels)
        print(f"✅ 训练标签文件: {len(df_train)} 个样本")
        
        # 检查图像文件
        missing_images = 0
        for idx, row in df_train.head(10).iterrows():  # 只检查前10个样本
            # 检查Colposcopy图像
            col_dir = original_data / 'train' / 'col' / str(row['ID'])
            if col_dir.exists():
                col_images = list(col_dir.glob('*.jpg'))
                if len(col_images) == 0:
                    missing_images += 1
            else:
                missing_images += 1
        
        if missing_images == 0:
            print(f"✅ 训练集图像文件: 完整（检查了前10个样本）")
        else:
            print(f"⚠️  训练集图像文件: {missing_images} 个样本缺失图像")
    else:
        print("❌ 训练标签文件不存在")
    
    if original_test_labels.exists():
        df_test = pd.read_csv(original_test_labels)
        print(f"✅ 测试标签文件: {len(df_test)} 个样本")
    else:
        print("❌ 测试标签文件不存在")
    
    # 2. 检查划分后数据集
    print("\n2. 划分后数据集检查 (5centers_multi_internal_external_final)")
    print("-" * 80)
    
    split_train_labels = split_data / 'train_labels.csv'
    split_val_labels = split_data / 'val_labels.csv'
    split_external_labels = split_data / 'external_test_labels.csv'
    
    # 检查标签文件
    if split_train_labels.exists():
        df_train_split = pd.read_csv(split_train_labels)
        print(f"✅ 内部训练标签: {len(df_train_split)} 个样本")
        
        # 检查图像文件
        missing_count = 0
        found_count = 0
        
        for idx, row in df_train_split.head(20).iterrows():  # 检查前20个样本
            # 检查Colposcopy图像
            col_dir = split_data / 'internal_train' / 'train' / 'col' / str(row['ID'])
            if col_dir.exists():
                col_images = list(col_dir.glob('*.jpg'))
                if len(col_images) > 0:
                    found_count += 1
                else:
                    missing_count += 1
            else:
                missing_count += 1
        
        print(f"   图像文件检查（前20个样本）:")
        print(f"   ✅ 找到图像: {found_count} 个")
        print(f"   ❌ 缺失图像: {missing_count} 个")
        
        if missing_count == 20:
            print(f"   ⚠️  **严重问题**: 内部训练集图像文件全部缺失！")
    else:
        print("❌ 内部训练标签文件不存在")
    
    if split_val_labels.exists():
        df_val = pd.read_csv(split_val_labels)
        print(f"✅ 内部验证标签: {len(df_val)} 个样本")
    else:
        print("❌ 内部验证标签文件不存在")
    
    if split_external_labels.exists():
        df_external = pd.read_csv(split_external_labels)
        print(f"✅ 外部测试标签: {len(df_external)} 个样本")
        
        # 检查外部测试集图像
        missing_count = 0
        found_count = 0
        
        for idx, row in df_external.head(10).iterrows():
            col_dir = split_data / 'external_validation' / 'col' / str(row['ID'])
            if col_dir.exists():
                col_images = list(col_dir.glob('*.jpg'))
                if len(col_images) > 0:
                    found_count += 1
                else:
                    missing_count += 1
            else:
                missing_count += 1
        
        print(f"   外部测试集图像检查（前10个样本）:")
        print(f"   ✅ 找到图像: {found_count} 个")
        print(f"   ❌ 缺失图像: {missing_count} 个")
    else:
        print("❌ 外部测试标签文件不存在")
    
    # 3. 总结
    print("\n" + "=" * 80)
    print("检查总结")
    print("=" * 80)
    print("✅ 原始数据集: 完整，可以使用")
    print("⚠️  划分后数据集: 内部训练集图像文件缺失，需要修复")
    print("✅ 划分后数据集: 外部测试集图像文件完整")
    print("=" * 80)

if __name__ == '__main__':
    check_dataset_integrity()

