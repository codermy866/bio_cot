#!/usr/bin/env python3
"""
诊断数据加载问题
检查OCT和colposcopy图像是否正确加载
"""

import os
import sys
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
import numpy as np

# 添加项目路径
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset

class Args:
    def __init__(self):
        self.data_path = '/data2/hmy/5Center_datas/5centers_multi_leave_centers_out'
        self.input_size = 224
        self.oct_num_frames = 120
        self.col_num_frames = 3
        self.oct_cache_dir = None  # 不使用缓存
        self.use_text_contrastive = False
        self.oct_points = 12
        self.oct_frames_per_point = 10
        self.use_pretrained_backbones = False
        self.cache_oct_features = False
        self.num_classes = 2

def test_data_loading():
    """测试数据加载"""
    print("=" * 80)
    print("🔍 数据加载诊断")
    print("=" * 80)
    
    args = Args()
    
    # 创建数据集
    print("\n📥 创建训练数据集...")
    try:
        train_dataset = EnhancedMultimodalCervicalDataset(
            root=os.path.join(args.data_path, 'internal_train/train'),
            is_train='train',
            args=args,
            transform=None,
            use_enhanced_oct=True,
            cache_oct_features=False
        )
        print(f"✅ 数据集创建成功，样本数: {len(train_dataset)}")
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 检查前5个样本
    print("\n📊 检查前5个样本的数据加载...")
    for i in range(min(5, len(train_dataset))):
        print(f"\n--- 样本 {i} ---")
        try:
            data = train_dataset[i]
            
            # 检查数据类型
            if isinstance(data, dict):
                oct_feat = data.get('oct_features')
                col_feat = data.get('colposcopy_features')
                clinical_feat = data.get('clinical_features')
                label = data.get('label')
                
                print(f"  OCT特征: {oct_feat.shape if oct_feat is not None else 'None'}")
                print(f"  OCT特征范围: [{oct_feat.min().item():.4f}, {oct_feat.max().item():.4f}]" if oct_feat is not None else "  OCT特征: None")
                print(f"  OCT特征是否全零: {(oct_feat == 0).all().item() if oct_feat is not None else 'N/A'}")
                
                print(f"  Colposcopy特征: {col_feat.shape if col_feat is not None else 'None'}")
                print(f"  Colposcopy特征范围: [{col_feat.min().item():.4f}, {col_feat.max().item():.4f}]" if col_feat is not None else "  Colposcopy特征: None")
                print(f"  Colposcopy特征是否全零: {(col_feat == 0).all().item() if col_feat is not None else 'N/A'}")
                
                print(f"  临床特征: {clinical_feat.shape if clinical_feat is not None else 'None'}")
                print(f"  临床特征值: {clinical_feat.tolist() if clinical_feat is not None else 'None'}")
                
                print(f"  标签: {label.item() if label is not None else 'None'}")
                
                # 检查是否有原始图像
                if 'oct_images' in data:
                    oct_imgs = data['oct_images']
                    print(f"  OCT图像: {oct_imgs.shape if oct_imgs is not None else 'None'}")
                if 'col_images' in data:
                    col_imgs = data['col_images']
                    print(f"  Colposcopy图像: {col_imgs.shape if col_imgs is not None else 'None'}")
                    
            else:
                print(f"  数据格式: {type(data)}")
                if len(data) >= 4:
                    oct_feat, col_feat, clinical_feat, label = data[:4]
                    print(f"  OCT特征: {oct_feat.shape}")
                    print(f"  Colposcopy特征: {col_feat.shape}")
                    print(f"  临床特征: {clinical_feat.shape}")
                    print(f"  标签: {label.item()}")
                    
        except Exception as e:
            print(f"  ❌ 加载样本 {i} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 统计特征统计信息
    print("\n📈 统计特征统计信息...")
    all_oct_feats = []
    all_col_feats = []
    all_clinical_feats = []
    all_labels = []
    
    for i in range(min(100, len(train_dataset))):
        try:
            data = train_dataset[i]
            if isinstance(data, dict):
                oct_feat = data.get('oct_features')
                col_feat = data.get('colposcopy_features')
                clinical_feat = data.get('clinical_features')
                label = data.get('label')
            else:
                oct_feat, col_feat, clinical_feat, label = data[:4]
            
            if oct_feat is not None:
                all_oct_feats.append(oct_feat.numpy())
            if col_feat is not None:
                all_col_feats.append(col_feat.numpy())
            if clinical_feat is not None:
                all_clinical_feats.append(clinical_feat.numpy())
            if label is not None:
                all_labels.append(label.item())
        except Exception as e:
            continue
    
    if all_oct_feats:
        oct_feats_array = np.array(all_oct_feats)
        print(f"  OCT特征统计 (100个样本):")
        print(f"    均值: {oct_feats_array.mean():.4f}")
        print(f"    标准差: {oct_feats_array.std():.4f}")
        print(f"    最小值: {oct_feats_array.min():.4f}")
        print(f"    最大值: {oct_feats_array.max():.4f}")
        print(f"    全零样本数: {(oct_feats_array == 0).all(axis=1).sum()}")
    else:
        print("  ❌ 没有OCT特征")
    
    if all_col_feats:
        col_feats_array = np.array(all_col_feats)
        print(f"  Colposcopy特征统计 (100个样本):")
        print(f"    均值: {col_feats_array.mean():.4f}")
        print(f"    标准差: {col_feats_array.std():.4f}")
        print(f"    最小值: {col_feats_array.min():.4f}")
        print(f"    最大值: {col_feats_array.max():.4f}")
        print(f"    全零样本数: {(col_feats_array == 0).all(axis=1).sum()}")
    else:
        print("  ❌ 没有Colposcopy特征")
    
    if all_clinical_feats:
        clinical_feats_array = np.array(all_clinical_feats)
        print(f"  临床特征统计 (100个样本):")
        print(f"    均值: {clinical_feats_array.mean(axis=0)}")
        print(f"    标准差: {clinical_feats_array.std(axis=0)}")
    
    if all_labels:
        labels_array = np.array(all_labels)
        print(f"  标签分布: {np.bincount(labels_array)}")
    
    print("\n" + "=" * 80)
    print("✅ 诊断完成")
    print("=" * 80)

if __name__ == '__main__':
    test_data_loading()



