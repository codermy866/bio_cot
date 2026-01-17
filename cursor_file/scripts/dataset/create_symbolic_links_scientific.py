#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件生成信息:
- 生成时间: 2025-12-25 10:35:08 (使用nvidia-smi获取CUDA时间)
- 生成需求: 为新划分的科学数据集创建软链接
- 生成原因: 新数据集需要创建软链接指向原始数据集的图像文件
- 相关任务: 数据集软链接创建，SCI论文准备

文件功能: 为新划分的科学数据集创建软链接到原始数据集
"""

import pandas as pd
from pathlib import Path
import os
import sys

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def identify_center(oct_id: str) -> tuple:
    """Identify medical center"""
    oct_str = str(oct_id)
    if 'M22105' in oct_str:
        return 'Enshi', 'internal'
    elif 'M22102' in oct_str:
        return 'Xiangyang', 'internal'
    elif 'M22104' in oct_str or 'M22101' in oct_str:
        return 'Shiyan', 'internal'
    elif 'M0008' in oct_str:
        return 'Jingzhou', 'external'
    elif 'M20203' in oct_str or 'M20105' in oct_str:
        return 'Wuda', 'external'
    return 'Unknown', 'unknown'

def find_data_in_original_dataset(oct_id: str, patient_id: str, original_data_root: Path, 
                                   search_in_train: bool = True, search_in_test: bool = True):
    """Find image files in original dataset"""
    result = {'oct': None, 'col': None, 'source': None}
    
    search_paths = []
    if search_in_train:
        search_paths.append(('train', original_data_root / 'train'))
    if search_in_test:
        search_paths.append(('test', original_data_root / 'test'))
    
    for source_name, base_path in search_paths:
        if result['oct'] is None:
            oct_path = base_path / 'oct' / oct_id
            if oct_path.exists() and oct_path.is_dir():
                image_files = list(oct_path.glob('*.png')) + list(oct_path.glob('*.jpg')) + list(oct_path.glob('*.jpeg'))
                if image_files:
                    result['oct'] = oct_path
                    if result['source'] is None:
                        result['source'] = source_name
        
        if result['col'] is None:
            col_path = base_path / 'col' / patient_id
            if col_path.exists() and col_path.is_dir():
                image_files = list(col_path.glob('*.png')) + list(col_path.glob('*.jpg')) + list(col_path.glob('*.jpeg'))
                if image_files:
                    result['col'] = col_path
                    if result['source'] is None:
                        result['source'] = source_name
        
        if result['oct'] is not None and result['col'] is not None:
            break
    
    return result

def create_symbolic_links_for_split_dataset(split_labels_file: Path, 
                                            original_data_root: Path,
                                            target_base_dir: Path,
                                            split_type: str = 'train'):
    """Create symbolic links for split dataset"""
    print(f"\n{'='*80}")
    print(f"Processing {split_type} dataset: {split_labels_file.name}")
    print(f"{'='*80}")
    
    if not split_labels_file.exists():
        print(f"❌ Label file not found: {split_labels_file}")
        return False
    
    df = pd.read_csv(split_labels_file)
    print(f"✅ Loaded label file: {len(df)} samples")
    
    # Determine target directories
    if split_type == 'external_test':
        target_oct_dir = target_base_dir / 'external_validation' / 'oct'
        target_col_dir = target_base_dir / 'external_validation' / 'col'
    else:
        target_oct_dir = target_base_dir / 'internal_train' / split_type / 'oct'
        target_col_dir = target_base_dir / 'internal_train' / split_type / 'col'
    
    target_oct_dir.mkdir(parents=True, exist_ok=True)
    target_col_dir.mkdir(parents=True, exist_ok=True)
    
    found_oct_count = 0
    found_col_count = 0
    missing_oct_count = 0
    missing_col_count = 0
    
    for idx, row in df.iterrows():
        oct_id = str(row['OCT'])
        patient_id = str(row['ID'])
        
        data_info = find_data_in_original_dataset(
            oct_id, patient_id, original_data_root,
            search_in_train=True, search_in_test=True
        )
        
        # Create OCT symbolic link
        if data_info['oct'] is not None:
            target_oct_link = target_oct_dir / oct_id
            if not target_oct_link.exists():
                try:
                    os.symlink(data_info['oct'], target_oct_link)
                    found_oct_count += 1
                except OSError as e:
                    print(f"⚠️  Failed to create OCT link: {oct_id} -> {e}")
                    missing_oct_count += 1
            else:
                found_oct_count += 1
        else:
            missing_oct_count += 1
        
        # Create Colposcopy symbolic link
        if data_info['col'] is not None:
            target_col_link = target_col_dir / patient_id
            if not target_col_link.exists():
                try:
                    os.symlink(data_info['col'], target_col_link)
                    found_col_count += 1
                except OSError as e:
                    print(f"⚠️  Failed to create Colposcopy link: {patient_id} -> {e}")
                    missing_col_count += 1
            else:
                found_col_count += 1
        else:
            missing_col_count += 1
        
        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx + 1}/{len(df)} (OCT: {found_oct_count}, Col: {found_col_count})")
    
    print(f"\n✅ {split_type} dataset processing completed:")
    print(f"   OCT images: {found_oct_count}/{len(df)} ({found_oct_count/len(df)*100:.1f}%)")
    print(f"   Colposcopy images: {found_col_count}/{len(df)} ({found_col_count/len(df)*100:.1f}%)")
    print(f"   Missing OCT: {missing_oct_count}")
    print(f"   Missing Colposcopy: {missing_col_count}")
    
    return found_oct_count > 0 or found_col_count > 0

def main():
    """Main function"""
    print("="*80)
    print("Create Symbolic Links for Scientific Dataset Split")
    print("="*80)
    
    original_dataset_path = Path('/data2/hmy/5Center_datas/5centers_multi')
    split_dataset_path = Path('/data2/hmy/5Center_datas/5centers_multi_internal_external_final_scientific')
    
    if not original_dataset_path.exists():
        print(f"❌ Original dataset path not found: {original_dataset_path}")
        return
    
    if not split_dataset_path.exists():
        print(f"❌ Split dataset path not found: {split_dataset_path}")
        return
    
    print(f"\nOriginal dataset: {original_dataset_path}")
    print(f"Split dataset: {split_dataset_path}")
    
    # Process train set
    train_labels = split_dataset_path / 'train_labels.csv'
    if train_labels.exists():
        create_symbolic_links_for_split_dataset(
            train_labels, original_dataset_path, split_dataset_path, 'train'
        )
    
    # Process validation set
    val_labels = split_dataset_path / 'val_labels.csv'
    if val_labels.exists():
        create_symbolic_links_for_split_dataset(
            val_labels, original_dataset_path, split_dataset_path, 'val'
        )
    
    # Process external test set
    external_test_labels = split_dataset_path / 'external_test_labels.csv'
    if external_test_labels.exists():
        create_symbolic_links_for_split_dataset(
            external_test_labels, original_dataset_path, split_dataset_path, 'external_test'
        )
    
    print("\n" + "="*80)
    print("✅ All datasets processed!")
    print("="*80)
    print(f"\n💡 Symbolic links created. Dataset ready for use.")

if __name__ == '__main__':
    main()

