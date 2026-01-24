#!/usr/bin/env python3
"""
快速测试修复后的数据加载
"""

import os
import sys
from pathlib import Path
import torch

# 添加项目路径
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.enhanced_multimodal_dataset import build_enhanced_dataset

class Args:
    def __init__(self):
        self.data_path = '/data2/hmy/5Center_datas/5centers_multi_leave_centers_out'
        self.input_size = 224
        self.oct_num_frames = 120
        self.col_num_frames = 3
        self.oct_cache_dir = None
        self.use_text_contrastive = False
        self.oct_points = 12
        self.oct_frames_per_point = 10
        self.use_pretrained_backbones = False
        self.cache_oct_features = False
        self.num_classes = 2

def test_fixed_data_loading():
    """测试修复后的数据加载"""
    print("=" * 80)
    print("🔍 测试修复后的数据加载")
    print("=" * 80)
    
    args = Args()
    
    # 创建数据集
    print("\n📥 创建训练数据集...")
    try:
        train_dataset = build_enhanced_dataset('train', args, use_external_test=False)
        print(f"✅ 数据集创建成功，样本数: {len(train_dataset)}")
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 检查前3个样本
    print("\n📊 检查前3个样本的数据...")
    for i in range(min(3, len(train_dataset))):
        print(f"\n--- 样本 {i} ---")
        try:
            data = train_dataset[i]
            
            if isinstance(data, dict):
                oct_feat = data.get('oct_features')
                col_feat = data.get('colposcopy_features')
                clinical_feat = data.get('clinical_features')
                label = data.get('label')
                
                print(f"  OCT特征: {oct_feat.shape if oct_feat is not None else 'None'}")
                if oct_feat is not None:
                    print(f"    - 范围: [{oct_feat.min().item():.4f}, {oct_feat.max().item():.4f}]")
                    print(f"    - 是否全零: {(oct_feat == 0).all().item()}")
                    print(f"    - 均值: {oct_feat.mean().item():.4f}")
                
                print(f"  Colposcopy特征: {col_feat.shape if col_feat is not None else 'None'}")
                if col_feat is not None:
                    print(f"    - 范围: [{col_feat.min().item():.4f}, {col_feat.max().item():.4f}]")
                    print(f"    - 是否全零: {(col_feat == 0).all().item()}")
                    print(f"    - 均值: {col_feat.mean().item():.4f}")
                
                print(f"  临床特征: {clinical_feat.shape if clinical_feat is not None else 'None'}")
                if clinical_feat is not None:
                    print(f"    - 值: {clinical_feat.tolist()}")
                
                print(f"  标签: {label.item() if label is not None else 'None'}")
                
        except Exception as e:
            print(f"  ❌ 加载样本 {i} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✅ 测试完成")
    print("=" * 80)

if __name__ == '__main__':
    test_fixed_data_loading()

