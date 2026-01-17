#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复外部验证数据集
从原始数据集正确复制所有数据文件
"""

import os
import sys
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def identify_center(oct_id):
    """从OCT ID识别中心"""
    if pd.isna(oct_id):
        return 'Unknown'
    oct_str = str(oct_id)
    if 'M22105' in oct_str:
        return '恩施'
    elif 'M0008' in oct_str:
        return '荆州'
    elif 'M22104' in oct_str or 'M22101' in oct_str:
        return '十堰'
    elif 'M20203' in oct_str or 'M20105' in oct_str:
        return '武大'
    elif 'M22102' in oct_str:
        return '襄阳'
    return 'Unknown'


def fix_external_validation_data(external_dir='5centers_multi_internal_external_recommended',
                                 original_data_dir='5centers_multi',
                                 external_centers=['十堰', '荆州']):
    """
    修复外部验证数据集，从原始数据集复制所有数据
    
    Args:
        external_dir: 外部验证数据集目录
        original_data_dir: 原始数据集目录
        external_centers: 外部验证中心列表
    """
    print("=" * 80)
    print("🔧 修复外部验证数据集")
    print("=" * 80)
    print()
    
    # 读取原始数据
    print("📥 步骤1: 读取原始数据...")
    train_df = pd.read_csv(os.path.join(original_data_dir, 'train_labels.csv'))
    test_df = pd.read_csv(os.path.join(original_data_dir, 'test_labels.csv'))
    
    # 识别中心
    train_df['center'] = train_df['OCT'].apply(identify_center)
    test_df['center'] = test_df['OCT'].apply(identify_center)
    
    # 合并数据
    all_data = pd.concat([train_df, test_df], ignore_index=True)
    
    # 筛选外部验证集数据（十堰和荆州）
    external_data = all_data[all_data['center'].isin(external_centers)].copy()
    
    print(f"✅ 找到 {len(external_data)} 个外部验证样本")
    print(f"   中心: {external_centers}")
    print()
    
    # 创建外部验证目录结构
    external_validation_dir = Path(external_dir) / 'external_validation'
    external_validation_dir.mkdir(parents=True, exist_ok=True)
    
    oct_dir = external_validation_dir / 'oct'
    col_dir = external_validation_dir / 'col'
    oct_dir.mkdir(parents=True, exist_ok=True)
    col_dir.mkdir(parents=True, exist_ok=True)
    
    print("📁 步骤2: 复制OCT图像...")
    oct_copied = 0
    oct_missing = 0
    
    for idx, row in tqdm(external_data.iterrows(), total=len(external_data), desc="复制OCT"):
        oct_id = str(row['OCT'])
        
        # 检查原始数据集中OCT图像的位置
        source_oct_train = Path(original_data_dir) / 'train' / 'oct' / oct_id
        source_oct_test = Path(original_data_dir) / 'test' / 'oct' / oct_id
        
        target_oct = oct_dir / oct_id
        
        # 如果目标不存在，从原始数据集复制
        if not target_oct.exists():
            if source_oct_train.exists():
                shutil.copytree(source_oct_train, target_oct, dirs_exist_ok=True)
                oct_copied += 1
            elif source_oct_test.exists():
                shutil.copytree(source_oct_test, target_oct, dirs_exist_ok=True)
                oct_copied += 1
            else:
                oct_missing += 1
                print(f"⚠️  OCT图像不存在: {oct_id}")
        else:
            # 检查是否为空
            if not any(target_oct.iterdir()):
                if source_oct_train.exists():
                    shutil.rmtree(target_oct)
                    shutil.copytree(source_oct_train, target_oct, dirs_exist_ok=True)
                    oct_copied += 1
                elif source_oct_test.exists():
                    shutil.rmtree(target_oct)
                    shutil.copytree(source_oct_test, target_oct, dirs_exist_ok=True)
                    oct_copied += 1
    
    print(f"✅ OCT图像: 已复制 {oct_copied} 个，缺失 {oct_missing} 个")
    print()
    
    print("📁 步骤3: 复制阴道镜图像...")
    col_copied = 0
    col_missing = 0
    
    for idx, row in tqdm(external_data.iterrows(), total=len(external_data), desc="复制阴道镜"):
        patient_id = str(row['ID'])
        oct_id = str(row['OCT'])
        
        # 检查原始数据集中阴道镜图像的位置
        source_col_train = Path(original_data_dir) / 'train' / 'col' / patient_id
        source_col_test = Path(original_data_dir) / 'test' / 'col' / patient_id
        
        target_col = col_dir / patient_id
        
        # 如果目标不存在，从原始数据集复制
        if not target_col.exists():
            if source_col_train.exists():
                shutil.copytree(source_col_train, target_col, dirs_exist_ok=True)
                col_copied += 1
            elif source_col_test.exists():
                shutil.copytree(source_col_test, target_col, dirs_exist_ok=True)
                col_copied += 1
            else:
                col_missing += 1
                print(f"⚠️  阴道镜图像不存在: {patient_id} (OCT: {oct_id})")
        else:
            # 检查是否为空
            if not any(target_col.iterdir()):
                if source_col_train.exists():
                    shutil.rmtree(target_col)
                    shutil.copytree(source_col_train, target_col, dirs_exist_ok=True)
                    col_copied += 1
                elif source_col_test.exists():
                    shutil.rmtree(target_col)
                    shutil.copytree(source_col_test, target_col, dirs_exist_ok=True)
                    col_copied += 1
    
    print(f"✅ 阴道镜图像: 已复制 {col_copied} 个，缺失 {col_missing} 个")
    print()
    
    # 保存标签文件
    print("📝 步骤4: 保存标签文件...")
    labels_file = external_validation_dir / 'external_labels.csv'
    external_data.to_csv(labels_file, index=False)
    print(f"✅ 标签文件已保存: {labels_file}")
    print(f"   样本数: {len(external_data)}")
    print()
    
    # 创建test目录结构（用于数据加载器）
    print("📁 步骤5: 创建test目录结构...")
    test_dir = external_validation_dir / 'test'
    test_dir.mkdir(exist_ok=True)
    
    # 创建符号链接或复制
    test_oct_dir = test_dir / 'oct'
    test_col_dir = test_dir / 'col'
    
    if test_oct_dir.exists():
        if test_oct_dir.is_symlink():
            test_oct_dir.unlink()
        elif test_oct_dir.is_dir():
            shutil.rmtree(test_oct_dir)
    
    if test_col_dir.exists():
        if test_col_dir.is_symlink():
            test_col_dir.unlink()
        elif test_col_dir.is_dir():
            shutil.rmtree(test_col_dir)
    
    # 创建符号链接
    test_oct_dir.symlink_to(oct_dir.resolve())
    test_col_dir.symlink_to(col_dir.resolve())
    
    # 复制标签文件
    test_labels_file = test_dir / 'test_labels.csv'
    shutil.copy(labels_file, test_labels_file)
    
    print(f"✅ test目录结构已创建")
    print()
    
    # 验证数据完整性
    print("🔍 步骤6: 验证数据完整性...")
    oct_folders = [d for d in oct_dir.iterdir() if d.is_dir()]
    col_folders = [d for d in col_dir.iterdir() if d.is_dir()]
    
    print(f"   OCT目录: {len(oct_folders)} 个文件夹")
    print(f"   阴道镜目录: {len(col_folders)} 个文件夹")
    print(f"   标签文件: {len(external_data)} 个样本")
    
    # 检查匹配
    oct_ids = set([str(row['OCT']) for _, row in external_data.iterrows()])
    patient_ids = set([str(row['ID']) for _, row in external_data.iterrows()])
    
    oct_found = set([d.name for d in oct_folders])
    col_found = set([d.name for d in col_folders])
    
    oct_missing_set = oct_ids - oct_found
    col_missing_set = patient_ids - col_found
    
    if oct_missing_set:
        print(f"⚠️  缺失OCT图像: {len(oct_missing_set)} 个")
        print(f"   前5个: {list(oct_missing_set)[:5]}")
    
    if col_missing_set:
        print(f"⚠️  缺失阴道镜图像: {len(col_missing_set)} 个")
        print(f"   前5个: {list(col_missing_set)[:5]}")
    
    if not oct_missing_set and not col_missing_set:
        print("✅ 所有数据完整！")
    
    print()
    print("=" * 80)
    print("✅ 外部验证数据集修复完成！")
    print("=" * 80)
    print()
    print(f"📁 数据位置: {external_validation_dir}")
    print(f"   - OCT图像: {oct_dir}")
    print(f"   - 阴道镜图像: {col_dir}")
    print(f"   - 标签文件: {labels_file}")
    print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='修复外部验证数据集')
    parser.add_argument('--external_dir', type=str,
                       default='5centers_multi_internal_external_recommended',
                       help='外部验证数据集目录')
    parser.add_argument('--original_data_dir', type=str,
                       default='5centers_multi',
                       help='原始数据集目录')
    parser.add_argument('--external_centers', type=str, nargs='+',
                       default=['十堰', '荆州'],
                       help='外部验证中心列表')
    
    args = parser.parse_args()
    
    fix_external_validation_data(
        args.external_dir,
        args.original_data_dir,
        args.external_centers
    )

