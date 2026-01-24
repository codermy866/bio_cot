#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据兼容性验证脚本
验证实验代码与实际数据的兼容性
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))


def validate_data_format():
    """验证数据格式"""
    print("="*60)
    print("数据格式验证")
    print("="*60)
    
    data_path = '5centers_multi'
    train_df = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test_labels.csv'))
    
    # 1. 检查必需列
    required_cols = ['ID', 'OCT', 'AGE', 'HPV清洗', 'TCT清洗', 'label']
    print("\n[1] 检查必需列...")
    for col in required_cols:
        if col in train_df.columns and col in test_df.columns:
            print(f"  ✅ {col}: 存在")
        else:
            print(f"  ❌ {col}: 缺失")
    
    # 2. 检查HPV数据格式
    print("\n[2] 检查HPV数据格式...")
    hpv_train = train_df['HPV清洗'].astype(str)
    hpv_test = test_df['HPV清洗'].astype(str)
    
    hpv_values_train = hpv_train.unique()
    hpv_values_test = hpv_test.unique()
    
    print(f"  训练集HPV唯一值: {hpv_values_train[:10]}...")  # 只显示前10个
    print(f"  测试集HPV唯一值: {hpv_values_test[:10]}...")
    
    # 检查HPV+筛选
    hpv_pos_train = (hpv_train == '1')
    hpv_pos_test = (hpv_test == '1')
    print(f"  训练集HPV+患者数: {np.sum(hpv_pos_train)}")
    print(f"  测试集HPV+患者数: {np.sum(hpv_pos_test)}")
    
    # 3. 检查TCT数据格式
    print("\n[3] 检查TCT数据格式...")
    tct_train = train_df['TCT清洗'].astype(str).str.upper()
    tct_test = test_df['TCT清洗'].astype(str).str.upper()
    
    tct_values_train = tct_train.unique()
    tct_values_test = tct_test.unique()
    
    print(f"  训练集TCT唯一值: {tct_values_train[:15]}...")  # 只显示前15个
    print(f"  测试集TCT唯一值: {tct_values_test[:15]}...")
    
    # 检查TCT异常判断
    tct_abnormal_train = (
        tct_train.str.contains('ASC-US|LSIL|HSIL', case=False, na=False) |
        (train_df['TCT清洗'].astype(str) == '1')
    )
    tct_abnormal_test = (
        tct_test.str.contains('ASC-US|LSIL|HSIL', case=False, na=False) |
        (test_df['TCT清洗'].astype(str) == '1')
    )
    print(f"  训练集TCT异常数: {np.sum(tct_abnormal_train)}")
    print(f"  测试集TCT异常数: {np.sum(tct_abnormal_test)}")
    
    # 4. 检查标签
    print("\n[4] 检查标签...")
    print(f"  训练集标签分布: {train_df['label'].value_counts().to_dict()}")
    print(f"  测试集标签分布: {test_df['label'].value_counts().to_dict()}")
    
    # 5. 检查图像文件
    print("\n[5] 检查图像文件...")
    train_oct_dir = os.path.join(data_path, 'train', 'oct')
    train_col_dir = os.path.join(data_path, 'train', 'col')
    test_oct_dir = os.path.join(data_path, 'test', 'oct')
    test_col_dir = os.path.join(data_path, 'test', 'col')
    
    print(f"  训练集OCT目录存在: {os.path.exists(train_oct_dir)}")
    print(f"  训练集Col目录存在: {os.path.exists(train_col_dir)}")
    print(f"  测试集OCT目录存在: {os.path.exists(test_oct_dir)}")
    print(f"  测试集Col目录存在: {os.path.exists(test_col_dir)}")
    
    # 6. 验证实验1的数据要求
    print("\n[6] 验证实验1（活检率降低）数据要求...")
    hpv_pos_test = (test_df['HPV清洗'].astype(str) == '1')
    n_hpv_pos = np.sum(hpv_pos_test)
    print(f"  测试集HPV+患者数: {n_hpv_pos}")
    if n_hpv_pos >= 100:
        print(f"  ✅ 满足实验要求（≥100例）")
    else:
        print(f"  ⚠️  样本量较小，建议合并训练+测试集")
    
    # 7. 验证实验2的数据要求
    print("\n[7] 验证实验2（OCT灵敏度）数据要求...")
    n_positive = np.sum(test_df['label'] == 1)
    print(f"  测试集病变患者数: {n_positive}")
    if n_positive >= 50:
        print(f"  ✅ 满足实验要求（≥50例）")
    else:
        print(f"  ⚠️  样本量较小，建议合并训练+测试集")
    
    print("\n" + "="*60)
    print("验证完成！")
    print("="*60)


def test_experiment_functions():
    """测试实验函数"""
    print("\n" + "="*60)
    print("测试实验函数")
    print("="*60)
    
    # 导入实验类（使用相对导入）
    sys.path.insert(0, str(ROOT_DIR))
    from experiments.experiment1_biopsy_reduction import BiopsyReductionExperiment
    
    try:
        # 初始化实验
        experiment = BiopsyReductionExperiment(
            data_path='5centers_multi',
            output_dir='lancet_primary_care/results/test_validation'
        )
        
        print("\n[1] BiopsyReductionExperiment初始化成功 ✅")
        
        # 测试HPV+筛选
        test_df = pd.read_csv('5centers_multi/test_labels.csv')
        hpv_positive = experiment._filter_hpv_positive(test_df)
        print(f"[2] HPV+筛选: {len(hpv_positive)}例 ✅")
        
        # 测试传统筛查规则
        pred, labels = experiment._traditional_screening_rule(test_df)
        print(f"[3] 传统筛查规则: {len(pred)}例预测 ✅")
        print(f"    活检建议数: {np.sum(pred)}")
        print(f"    真实病变数: {np.sum(labels)}")
        
        print("\n✅ 所有测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    validate_data_format()
    test_experiment_functions()

