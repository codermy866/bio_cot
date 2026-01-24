#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
比较两个数据集的划分方式，评估哪个更符合科学标准
"""

import pandas as pd
import json
from pathlib import Path
import re

# 两个数据路径
DATA_PATH_1 = Path('/data2/hmy/VLM_Caus_Rm_Mics/data/5centers_multi_leave_centers_out')
DATA_PATH_2 = Path('/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal')

def extract_center_id(oct_id):
    """从oct_id提取中心ID"""
    if pd.isna(oct_id):
        return 'unknown'
    oct_id_str = str(oct_id)
    if oct_id_str.startswith('M'):
        match = re.match(r'M(\d+)', oct_id_str)
        if match:
            return match.group(1)
        return oct_id_str[1:6] if len(oct_id_str) > 6 else oct_id_str[1:]
    return 'unknown'

def map_center_id_to_name(center_id):
    """将中心ID映射到中心名称"""
    center_mapping = {
        '22105': 'Enshi（恩施）',
        '22102': 'Xiangyang（襄阳）',
        '20203': 'Wuda（武大）',
        '20105': 'Shiyan（十堰）',
        '22101': 'Jingzhou（荆州）',
        '22104': 'Jingzhou（荆州）',
        '0008': 'Jingzhou（荆州）',
    }
    return center_mapping.get(str(center_id), f'Unknown({center_id})')

def analyze_dataset_1():
    """分析数据集1：Leave-Centers-Out"""
    print("="*80)
    print("数据集1分析: Leave-Centers-Out 划分方式")
    print("="*80)
    
    # 读取划分统计信息
    split_stats = json.load(open(DATA_PATH_1 / 'split_statistics.json', 'r', encoding='utf-8'))
    
    print(f"\n📊 划分策略: {split_stats['split_strategy']}")
    print(f"总样本数: {split_stats['total_samples']}")
    print(f"总体阳性率: {split_stats['overall_positive_rate']*100:.2f}%")
    
    print(f"\n🏥 中心划分:")
    print(f"  内部开发集中心: {', '.join(split_stats['leave_centers_out']['internal_centers'])}")
    print(f"  外部测试集中心: {', '.join(split_stats['leave_centers_out']['external_centers'])}")
    print(f"  内部样本数: {split_stats['leave_centers_out']['internal_sample_size']}")
    print(f"  外部样本数: {split_stats['leave_centers_out']['external_sample_size']}")
    print(f"  内部阳性率: {split_stats['leave_centers_out']['internal_positive_rate']*100:.2f}%")
    print(f"  外部阳性率: {split_stats['leave_centers_out']['external_positive_rate']*100:.2f}%")
    
    print(f"\n📋 数据集划分:")
    print(f"  训练集: {split_stats['train_set']['samples']} 样本, 阳性率: {split_stats['train_set']['positive_rate']*100:.2f}%")
    print(f"  验证集: {split_stats['validation_set']['samples']} 样本, 阳性率: {split_stats['validation_set']['positive_rate']*100:.2f}%")
    print(f"  外部测试集: {split_stats['external_test_set']['samples']} 样本, 阳性率: {split_stats['external_test_set']['positive_rate']*100:.2f}%")
    
    print(f"\n✅ 科学合规性检查:")
    compliance = split_stats['sci_paper_compliance']
    for key, value in compliance.items():
        status = "✅" if value else "❌"
        print(f"  {status} {key}: {value}")
    
    # 分析实际数据中的中心分布
    print(f"\n🔍 实际数据中的中心分布分析:")
    
    # 读取训练集
    try:
        train_df = pd.read_csv(DATA_PATH_1 / 'train_labels.csv', encoding='utf-8')
    except:
        train_df = pd.read_csv(DATA_PATH_1 / 'train_labels.csv', encoding='gbk')
    
    # 读取验证集
    try:
        val_df = pd.read_csv(DATA_PATH_1 / 'val_labels.csv', encoding='utf-8')
    except:
        val_df = pd.read_csv(DATA_PATH_1 / 'val_labels.csv', encoding='gbk')
    
    # 读取外部测试集
    try:
        test_df = pd.read_csv(DATA_PATH_1 / 'external_test_labels.csv', encoding='utf-8')
    except:
        test_df = pd.read_csv(DATA_PATH_1 / 'external_test_labels.csv', encoding='gbk')
    
    # 提取中心ID
    train_df['center_id'] = train_df['OCT'].apply(extract_center_id)
    val_df['center_id'] = val_df['OCT'].apply(extract_center_id)
    test_df['center_id'] = test_df['OCT'].apply(extract_center_id)
    
    print(f"\n  训练集中心分布:")
    train_centers = train_df['center_id'].value_counts()
    for center_id, count in train_centers.items():
        center_name = map_center_id_to_name(center_id)
        print(f"    {center_name} (ID: {center_id}): {count} 样本")
    
    print(f"\n  验证集中心分布:")
    val_centers = val_df['center_id'].value_counts()
    for center_id, count in val_centers.items():
        center_name = map_center_id_to_name(center_id)
        print(f"    {center_name} (ID: {center_id}): {count} 样本")
    
    print(f"\n  外部测试集中心分布:")
    test_centers = test_df['center_id'].value_counts()
    for center_id, count in test_centers.items():
        center_name = map_center_id_to_name(center_id)
        print(f"    {center_name} (ID: {center_id}): {count} 样本")
    
    return {
        'split_strategy': split_stats['split_strategy'],
        'compliance': compliance,
        'train_centers': train_centers.to_dict(),
        'val_centers': val_centers.to_dict(),
        'test_centers': test_centers.to_dict(),
        'stats': split_stats
    }

def analyze_dataset_2():
    """分析数据集2：当前使用的数据集"""
    print("\n" + "="*80)
    print("数据集2分析: 5centers_multi_positive_sites_multimodal")
    print("="*80)
    
    # 读取训练集
    train_csv = DATA_PATH_2 / 'internal_train' / 'labels.csv'
    try:
        train_df = pd.read_csv(train_csv, encoding='utf-8')
    except:
        train_df = pd.read_csv(train_csv, encoding='gbk')
    
    # 读取验证集
    val_csv = DATA_PATH_2 / 'internal_val' / 'labels.csv'
    try:
        val_df = pd.read_csv(val_csv, encoding='utf-8')
    except:
        val_df = pd.read_csv(val_csv, encoding='gbk')
    
    # 读取外部测试集
    test_csv = DATA_PATH_2 / 'external_test' / 'labels.csv'
    try:
        test_df = pd.read_csv(test_csv, encoding='utf-8')
    except:
        test_df = pd.read_csv(test_csv, encoding='gbk')
    
    print(f"\n📋 数据集划分:")
    print(f"  训练集: {len(train_df)} 样本")
    print(f"  验证集: {len(val_df)} 样本")
    print(f"  外部测试集: {len(test_df)} 样本")
    
    # 计算阳性率
    train_pos_rate = (train_df['label'] == 1).sum() / len(train_df) if 'label' in train_df.columns else 0
    val_pos_rate = (val_df['label'] == 1).sum() / len(val_df) if 'label' in val_df.columns else 0
    test_pos_rate = (test_df['label'] == 1).sum() / len(test_df) if 'label' in test_df.columns else 0
    
    print(f"  训练集阳性率: {train_pos_rate*100:.2f}%")
    print(f"  验证集阳性率: {val_pos_rate*100:.2f}%")
    print(f"  外部测试集阳性率: {test_pos_rate*100:.2f}%")
    
    # 提取中心ID
    train_df['center_id'] = train_df['oct_id'].apply(extract_center_id)
    val_df['center_id'] = val_df['oct_id'].apply(extract_center_id)
    test_df['center_id'] = test_df['oct_id'].apply(extract_center_id)
    
    print(f"\n🔍 中心分布分析:")
    
    print(f"\n  训练集中心分布:")
    train_centers = train_df['center_id'].value_counts()
    for center_id, count in train_centers.items():
        center_name = map_center_id_to_name(center_id)
        print(f"    {center_name} (ID: {center_id}): {count} 样本")
    
    print(f"\n  验证集中心分布:")
    val_centers = val_df['center_id'].value_counts()
    for center_id, count in val_centers.items():
        center_name = map_center_id_to_name(center_id)
        print(f"    {center_name} (ID: {center_id}): {count} 样本")
    
    print(f"\n  外部测试集中心分布:")
    test_centers = test_df['center_id'].value_counts()
    for center_id, count in test_centers.items():
        center_name = map_center_id_to_name(center_id)
        print(f"    {center_name} (ID: {center_id}): {count} 样本")
    
    # 检查是否有中心泄露（训练集和测试集是否有重叠的中心）
    train_center_set = set(train_centers.index)
    val_center_set = set(val_centers.index)
    test_center_set = set(test_centers.index)
    
    print(f"\n🔒 中心独立性检查:")
    train_val_overlap = train_center_set & val_center_set
    train_test_overlap = train_center_set & test_center_set
    val_test_overlap = val_center_set & test_center_set
    
    print(f"  训练集-验证集重叠中心: {train_val_overlap if train_val_overlap else '无（正常）'}")
    print(f"  训练集-测试集重叠中心: {train_test_overlap if train_test_overlap else '无（符合Leave-Centers-Out）'}")
    print(f"  验证集-测试集重叠中心: {val_test_overlap if val_test_overlap else '无（符合Leave-Centers-Out）'}")
    
    # 判断划分策略
    if not train_test_overlap and not val_test_overlap:
        split_strategy = "Leave-Centers-Out (严格)"
        compliance = {
            'center_independence': True,
            'leave_centers_out': True,
            'class_distribution_consistent': abs(train_pos_rate - test_pos_rate) < 0.1
        }
    elif len(train_test_overlap) < len(test_center_set):
        split_strategy = "Partial Leave-Centers-Out (部分中心独立)"
        compliance = {
            'center_independence': False,
            'leave_centers_out': False,
            'class_distribution_consistent': abs(train_pos_rate - test_pos_rate) < 0.1
        }
    else:
        split_strategy = "Random Split (随机划分)"
        compliance = {
            'center_independence': False,
            'leave_centers_out': False,
            'class_distribution_consistent': abs(train_pos_rate - test_pos_rate) < 0.1
        }
    
    print(f"\n📊 推断的划分策略: {split_strategy}")
    
    return {
        'split_strategy': split_strategy,
        'compliance': compliance,
        'train_centers': train_centers.to_dict(),
        'val_centers': val_centers.to_dict(),
        'test_centers': test_centers.to_dict(),
        'train_pos_rate': train_pos_rate,
        'val_pos_rate': val_pos_rate,
        'test_pos_rate': test_pos_rate
    }

def compare_and_recommend(d1_info, d2_info):
    """比较两个数据集并给出推荐"""
    print("\n" + "="*80)
    print("📊 数据集比较与推荐")
    print("="*80)
    
    print(f"\n【划分策略对比】")
    print(f"数据集1: {d1_info['split_strategy']}")
    print(f"数据集2: {d2_info['split_strategy']}")
    
    print(f"\n【科学合规性对比】")
    print(f"\n数据集1合规性:")
    for key, value in d1_info['compliance'].items():
        status = "✅" if value else "❌"
        print(f"  {status} {key}: {value}")
    
    print(f"\n数据集2合规性:")
    for key, value in d2_info['compliance'].items():
        status = "✅" if value else "❌"
        print(f"  {status} {key}: {value}")
    
    print(f"\n【中心独立性对比】")
    d1_center_indep = d1_info['compliance'].get('center_independence', False)
    d2_center_indep = d2_info['compliance'].get('center_independence', False)
    
    print(f"数据集1中心独立性: {'✅ 完全独立' if d1_center_indep else '❌ 有重叠'}")
    print(f"数据集2中心独立性: {'✅ 完全独立' if d2_center_indep else '❌ 有重叠'}")
    
    print(f"\n【推荐结论】")
    print("="*80)
    
    if d1_center_indep and not d2_center_indep:
        print("✅ **推荐使用数据集1 (Leave-Centers-Out)**")
        print("\n理由:")
        print("1. ✅ 严格遵循Leave-Centers-Out策略，训练/验证集和测试集完全独立")
        print("2. ✅ 符合多中心研究的科学标准，避免中心间数据泄露")
        print("3. ✅ 外部测试集来自完全不同的中心，更能验证模型的泛化能力")
        print("4. ✅ 有明确的划分统计信息和合规性检查")
        print("5. ✅ 符合SCI论文发表的数据划分要求")
    elif d2_center_indep and not d1_center_indep:
        print("✅ **推荐使用数据集2 (5centers_multi_positive_sites_multimodal)**")
        print("\n理由:")
        print("1. ✅ 严格遵循Leave-Centers-Out策略")
        print("2. ✅ 中心独立性更好")
    elif d1_center_indep and d2_center_indep:
        print("✅ **两个数据集都符合Leave-Centers-Out标准**")
        print("\n建议:")
        print("1. 数据集1有更详细的统计信息和合规性检查")
        print("2. 数据集2可能是数据集1的另一个版本或处理方式")
        print("3. 建议使用数据集1，因为其划分信息更透明、可追溯")
    else:
        print("⚠️ **两个数据集都存在中心重叠问题**")
        print("\n建议:")
        print("1. 优先使用数据集1，因为其划分策略更明确")
        print("2. 如果必须使用数据集2，需要说明划分方式并评估影响")
    
    print("\n" + "="*80)
    print("📝 论文撰写建议")
    print("="*80)
    print("""
在论文中描述数据划分时，应明确说明：

1. **划分策略**: Leave-One-Center-Out 或 Leave-Centers-Out
2. **内部开发集中心**: 列出用于训练和验证的中心
3. **外部测试集中心**: 列出用于测试的中心（必须与训练集完全独立）
4. **样本数量**: 训练集、验证集、测试集的样本数
5. **类别分布**: 各数据集的阳性/阴性比例
6. **中心分布**: 各中心在不同数据集中的样本数

示例描述：
"We employed a leave-centers-out cross-validation strategy, where data from 
Enshi, Xiangyang, and Wuda centers were used for internal development (training 
and validation), while data from Shiyan and Jingzhou centers were reserved as 
an external test set to evaluate model generalizability."
    """)

def main():
    """主函数"""
    print("="*80)
    print("数据集划分方式比较分析")
    print("="*80)
    
    # 分析数据集1
    d1_info = analyze_dataset_1()
    
    # 分析数据集2
    d2_info = analyze_dataset_2()
    
    # 比较并推荐
    compare_and_recommend(d1_info, d2_info)

if __name__ == '__main__':
    main()

