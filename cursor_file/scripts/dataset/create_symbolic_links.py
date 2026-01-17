#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件生成信息:
- 生成时间: 2025-12-24 19:55:30 (使用nvidia-smi获取CUDA时间)
- 生成需求: 用户要求根据划分好的内外部数据集表格，在原始数据集中寻找对应的数据并创建软链接
- 生成原因: 服务器内存不够，无法复制182GB数据，通过软链接方式节省空间并组织数据
- 相关任务: 数据集组织，MICCAI论文准备

文件功能: 根据划分后的标签文件，在原始数据集中找到对应的图像文件，并创建软链接到划分后的数据集目录
"""

import pandas as pd
from pathlib import Path
import os
import sys
import subprocess
import datetime

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cursor_file.project_config import DATA_5CENTERS_MULTI, DATA_5CENTERS_INTERNAL_EXTERNAL

def get_timestamp():
    """获取当前时间戳"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=timestamp', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            timestamp_str = result.stdout.strip().split('\n')[0]
            try:
                if '/' in timestamp_str:
                    timestamp_clean = timestamp_str.split('.')[0]
                    dt = datetime.datetime.strptime(timestamp_clean, '%Y/%m/%d %H:%M:%S')
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
            except (ValueError, AttributeError):
                pass
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def find_data_in_original_dataset(oct_id: str, patient_id: str, original_data_root: Path, 
                                   search_in_train: bool = True, search_in_test: bool = True):
    """
    在原始数据集中查找对应的图像文件
    
    Args:
        oct_id: OCT图像目录名（如 M22102_2023_P0000147）
        patient_id: 患者ID（如 20230316_黄贵云）
        original_data_root: 原始数据集根目录
        search_in_train: 是否在train目录中搜索
        search_in_test: 是否在test目录中搜索
    
    Returns:
        dict: 包含找到的OCT和Colposcopy图像路径，格式为 {'oct': Path, 'col': Path, 'source': 'train' or 'test'}
    """
    result = {'oct': None, 'col': None, 'source': None}
    
    # 搜索路径
    search_paths = []
    if search_in_train:
        search_paths.append(('train', original_data_root / 'train'))
    if search_in_test:
        search_paths.append(('test', original_data_root / 'test'))
    
    # 分别查找OCT和Colposcopy图像（它们可能在不同的目录中）
    for source_name, base_path in search_paths:
        # 查找OCT图像
        if result['oct'] is None:
            oct_path = base_path / 'oct' / oct_id
            if oct_path.exists() and oct_path.is_dir():
                # 检查目录下是否有图像文件
                image_files = list(oct_path.glob('*.png')) + list(oct_path.glob('*.jpg')) + list(oct_path.glob('*.jpeg'))
                if image_files:
                    result['oct'] = oct_path
                    if result['source'] is None:
                        result['source'] = source_name
        
        # 查找Colposcopy图像
        if result['col'] is None:
            col_path = base_path / 'col' / patient_id
            if col_path.exists() and col_path.is_dir():
                image_files = list(col_path.glob('*.png')) + list(col_path.glob('*.jpg')) + list(col_path.glob('*.jpeg'))
                if image_files:
                    result['col'] = col_path
                    if result['source'] is None:
                        result['source'] = source_name
        
        # 如果两个都找到了，可以提前退出
        if result['oct'] is not None and result['col'] is not None:
            break
    
    return result

def create_symbolic_links_for_split_dataset(split_labels_file: Path, 
                                            original_data_root: Path,
                                            target_base_dir: Path,
                                            split_type: str = 'train'):
    """
    为划分后的数据集创建软链接
    
    Args:
        split_labels_file: 划分后的标签文件路径（如 train_labels.csv）
        original_data_root: 原始数据集根目录
        target_base_dir: 目标目录（划分后数据集的根目录）
        split_type: 划分类型（'train', 'val', 'external_test'）
    """
    print(f"\n{'='*80}")
    print(f"处理 {split_type} 数据集: {split_labels_file.name}")
    print(f"{'='*80}")
    
    if not split_labels_file.exists():
        print(f"❌ 标签文件不存在: {split_labels_file}")
        return False
    
    # 读取标签文件
    df = pd.read_csv(split_labels_file)
    print(f"✅ 读取标签文件: {len(df)} 个样本")
    
    # 确定目标目录结构
    if split_type == 'external_test':
        # 外部测试集: external_validation/oct/ 和 external_validation/col/
        target_oct_dir = target_base_dir / 'external_validation' / 'oct'
        target_col_dir = target_base_dir / 'external_validation' / 'col'
    else:
        # 内部训练集/验证集: internal_train/train/oct/ 和 internal_train/train/col/
        target_oct_dir = target_base_dir / 'internal_train' / split_type / 'oct'
        target_col_dir = target_base_dir / 'internal_train' / split_type / 'col'
    
    # 创建目标目录
    target_oct_dir.mkdir(parents=True, exist_ok=True)
    target_col_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    found_oct_count = 0
    found_col_count = 0
    missing_oct_count = 0
    missing_col_count = 0
    
    # 遍历每个样本
    for idx, row in df.iterrows():
        oct_id = str(row['OCT'])
        patient_id = str(row['ID'])
        
        # 在原始数据集中查找
        data_info = find_data_in_original_dataset(
            oct_id, patient_id, original_data_root,
            search_in_train=True, search_in_test=True
        )
        
        # 创建OCT软链接
        if data_info['oct'] is not None:
            target_oct_link = target_oct_dir / oct_id
            if not target_oct_link.exists():
                try:
                    os.symlink(data_info['oct'], target_oct_link)
                    found_oct_count += 1
                except OSError as e:
                    print(f"⚠️  创建OCT软链接失败: {oct_id} -> {e}")
                    missing_oct_count += 1
            else:
                found_oct_count += 1
        else:
            missing_oct_count += 1
            if idx < 5:  # 只打印前5个缺失的样本
                print(f"❌ 未找到OCT图像: {oct_id}")
        
        # 创建Colposcopy软链接
        if data_info['col'] is not None:
            target_col_link = target_col_dir / patient_id
            if not target_col_link.exists():
                try:
                    os.symlink(data_info['col'], target_col_link)
                    found_col_count += 1
                except OSError as e:
                    print(f"⚠️  创建Colposcopy软链接失败: {patient_id} -> {e}")
                    missing_col_count += 1
            else:
                found_col_count += 1
        else:
            missing_col_count += 1
            if idx < 5:  # 只打印前5个缺失的样本
                print(f"❌ 未找到Colposcopy图像: {patient_id}")
        
        # 进度显示
        if (idx + 1) % 100 == 0:
            print(f"  进度: {idx + 1}/{len(df)} (OCT: {found_oct_count}, Col: {found_col_count})")
    
    # 打印统计信息
    print(f"\n✅ {split_type} 数据集处理完成:")
    print(f"   OCT图像: 找到 {found_oct_count}/{len(df)} ({found_oct_count/len(df)*100:.1f}%)")
    print(f"   Colposcopy图像: 找到 {found_col_count}/{len(df)} ({found_col_count/len(df)*100:.1f}%)")
    print(f"   缺失OCT: {missing_oct_count} 个")
    print(f"   缺失Colposcopy: {missing_col_count} 个")
    
    return found_oct_count > 0 or found_col_count > 0

def main():
    """主函数"""
    print("="*80)
    print("根据划分后的标签文件创建软链接到原始数据集")
    print("="*80)
    
    # 获取配置路径
    # 注意：原始数据集实际路径是软链接指向 /data2/hmy/5Center_datas/5centers_multi
    original_dataset_path = Path('/data2/hmy/5Center_datas/5centers_multi')
    split_dataset_path = DATA_5CENTERS_INTERNAL_EXTERNAL
    
    # 检查路径
    if not original_dataset_path.exists():
        print(f"❌ 原始数据集路径不存在: {original_dataset_path}")
        return
    
    if not split_dataset_path.exists():
        print(f"❌ 划分后数据集路径不存在: {split_dataset_path}")
        return
    
    print(f"\n原始数据集: {original_dataset_path}")
    print(f"划分后数据集: {split_dataset_path}")
    
    # 处理内部训练集
    train_labels = split_dataset_path / 'train_labels.csv'
    if train_labels.exists():
        create_symbolic_links_for_split_dataset(
            train_labels, original_dataset_path, split_dataset_path, 'train'
        )
    else:
        print(f"⚠️  内部训练集标签文件不存在: {train_labels}")
    
    # 处理内部验证集
    val_labels = split_dataset_path / 'val_labels.csv'
    if val_labels.exists():
        create_symbolic_links_for_split_dataset(
            val_labels, original_dataset_path, split_dataset_path, 'val'
        )
    else:
        print(f"⚠️  内部验证集标签文件不存在: {val_labels}")
    
    # 处理外部测试集
    external_test_labels = split_dataset_path / 'external_test_labels.csv'
    if external_test_labels.exists():
        create_symbolic_links_for_split_dataset(
            external_test_labels, original_dataset_path, split_dataset_path, 'external_test'
        )
    else:
        print(f"⚠️  外部测试集标签文件不存在: {external_test_labels}")
    
    print("\n" + "="*80)
    print("✅ 所有数据集处理完成！")
    print("="*80)
    print("\n💡 提示:")
    print("   - 软链接已创建，现在可以在划分后的数据集目录中使用这些数据")
    print("   - 数据实际存储在原始数据集中，节省了磁盘空间")
    print("   - 如果软链接失效，可以重新运行此脚本")

if __name__ == '__main__':
    main()

