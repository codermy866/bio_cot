#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预计算OCT特征并缓存（根本解决训练慢的问题）
类似VLM特征预提取，提前处理OCT图像，训练时直接加载特征
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset


def precompute_oct_features(data_root, split='train', device='cuda:1'):
    """预计算OCT特征并保存"""
    print(f"📂 开始预计算{split}集的OCT特征...")
    
    # 创建dummy args
    class DummyArgs:
        def __init__(self):
            self.data_path = data_root
            self.input_size = 224
            self.oct_num_frames = 48
            self.oct_points = 12
            self.oct_frames_per_point = 10
            self.oct_cache_dir = None
            self.use_pretrained_backbones = True
    
    dummy_args = DummyArgs()
    
    # 创建数据集（启用缓存）
    cache_dir = Path(data_root) / 'oct_features_cache' / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    dummy_args.oct_cache_dir = str(cache_dir)
    
    dataset = EnhancedMultimodalCervicalDataset(
        root=Path(data_root) / 'internal_train' / split,
        is_train=split,
        args=dummy_args,
        use_enhanced_oct=True,
        cache_oct_features=True
    )
    
    print(f"✅ 数据集加载完成: {len(dataset)} 个样本")
    print(f"💾 缓存目录: {cache_dir}")
    
    # 加载CSV获取patient_id（与数据集使用的ID一致）
    csv_path = Path(data_root) / f'{split}_labels.csv'
    patient_ids = []
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # 尝试多种可能的ID列名
        if 'ID' in df.columns:
            patient_ids = df['ID'].astype(str).str.strip().tolist()
        elif 'patient_id' in df.columns:
            patient_ids = df['patient_id'].astype(str).str.strip().tolist()
        elif 'sample_id' in df.columns:
            patient_ids = df['sample_id'].astype(str).str.strip().tolist()
        else:
            # 如果没有ID列，从数据集获取
            print("⚠️ CSV中没有找到ID列，将从数据集获取patient_id")
            patient_ids = None
    else:
        patient_ids = None
    
    # 预计算特征
    print(f"🔄 开始预计算OCT特征...")
    failed_samples = []
    
    for idx in tqdm(range(len(dataset)), desc=f'Processing {split}'):
        try:
            # 从数据集获取真实的oct_id（与缓存文件名一致）
            # 数据集使用image_files字典，key是oct_id（来自CSV的OCT列，如M22105_2023_P0000023）
            if hasattr(dataset, 'image_files') and len(dataset.image_files) > idx:
                oct_id = list(dataset.image_files.keys())[idx]  # image_files的key就是oct_id
                cache_id = oct_id
            elif csv_path.exists():
                # 如果无法从image_files获取，从CSV获取OCT列
                df_check = pd.read_csv(csv_path)
                if 'OCT' in df_check.columns and idx < len(df_check):
                    cache_id = str(df_check.iloc[idx]['OCT']).strip()
                elif patient_ids and idx < len(patient_ids):
                    cache_id = patient_ids[idx]
                else:
                    cache_id = f'sample_{idx}'
            else:
                cache_id = f'sample_{idx}'
            
            cache_path = cache_dir / f"{cache_id}_enhanced.pt"
            
            # 如果已缓存，跳过
            if cache_path.exists():
                continue
            
            # 加载数据（这会触发特征提取和缓存）
            # _get_enhanced_oct_features接收的patient_id参数实际上是oct_id
            _ = dataset[idx]
            
            # 验证缓存是否创建（使用oct_id检查）
            if not cache_path.exists():
                failed_samples.append(cache_id)
                print(f"⚠️ 警告: {cache_id} 特征未成功缓存")
        
        except Exception as e:
            failed_samples.append(sample_id if 'sample_id' in locals() else f'sample_{idx}')
            print(f"❌ 错误处理 {sample_id}: {e}")
            continue
    
    print(f"\n✅ 预计算完成！")
    print(f"   成功: {len(dataset) - len(failed_samples)} 个样本")
    print(f"   失败: {len(failed_samples)} 个样本")
    if failed_samples:
        print(f"   失败样本: {failed_samples[:10]}...")
    
    # 保存特征索引（用于快速加载）
    feature_index = {}
    for cache_file in cache_dir.glob("*_enhanced.pt"):
        sample_id = cache_file.stem.replace("_enhanced", "")
        feature_index[sample_id] = str(cache_file)
    
    index_path = cache_dir / 'feature_index.npy'
    np.save(str(index_path), feature_index, allow_pickle=True)
    print(f"💾 特征索引已保存: {index_path}")
    
    return cache_dir


def main():
    data_root = '/data2/hmy/5Center_datas/5centers_multi_leave_centers_out'
    device = 'cuda:1'
    
    print("=" * 80)
    print("🚀 OCT特征预计算（根本解决训练慢的问题）")
    print("=" * 80)
    
    # 预计算训练集
    train_cache = precompute_oct_features(data_root, split='train', device=device)
    
    # 预计算验证集
    val_cache = precompute_oct_features(data_root, split='val', device=device)
    
    print("\n" + "=" * 80)
    print("✅ 所有OCT特征预计算完成！")
    print(f"   训练集缓存: {train_cache}")
    print(f"   验证集缓存: {val_cache}")
    print("=" * 80)


if __name__ == '__main__':
    main()


