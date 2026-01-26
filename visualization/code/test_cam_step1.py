#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CAM生成 - Step 1: 测试模型forward调用
验证模型能否正确调用，并找到正确的目标层
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BioCOT_v3_2_Config
from models.bio_cot_v3_2 import create_bio_cot_v3_2
from data.dataset_v3_2 import FiveCentersMultimodalDatasetV3_2
from torch.utils.data import DataLoader
import sys
from pathlib import Path
# 导入特征提取函数
local_training_path = Path(__file__).resolve().parent.parent.parent / 'training'
if local_training_path.exists():
    sys.path.insert(0, str(local_training_path))
    from extract_vit_patches import extract_patch_features_with_vit
else:
    raise ImportError(f"无法找到training目录: {local_training_path}")

def test_model_forward():
    """测试模型forward调用"""
    print("=" * 80)
    print("Step 1: 测试模型forward调用")
    print("=" * 80)
    
    # 1. 加载配置
    config = BioCOT_v3_2_Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备: {device}")
    
    # 2. 创建模型
    print("\n📦 加载模型...")
    model = create_bio_cot_v3_2(config)
    model.to(device)
    model.eval()
    print("✅ 模型加载成功")
    
    # 3. 检查模型结构
    print("\n🔍 检查模型结构...")
    print(f"   模型类型: {type(model)}")
    
    # 查找visual_encoder
    if hasattr(model, 'visual_encoder'):
        print(f"   ✅ 找到 visual_encoder: {type(model.visual_encoder)}")
        
        # 检查visual_encoder的结构
        if hasattr(model.visual_encoder, 'blocks'):
            print(f"   ✅ visual_encoder.blocks 存在，长度: {len(model.visual_encoder.blocks)}")
            print(f"   ✅ 最后一层: {type(model.visual_encoder.blocks[-1])}")
        elif hasattr(model.visual_encoder, 'layers'):
            print(f"   ✅ visual_encoder.layers 存在，长度: {len(model.visual_encoder.layers)}")
        else:
            print(f"   ⚠️ visual_encoder 结构: {dir(model.visual_encoder)}")
    else:
        print("   ⚠️ 未找到 visual_encoder")
        # 列出所有属性
        print(f"   模型属性: {[attr for attr in dir(model) if not attr.startswith('_')]}")
    
    # 4. 加载数据集（只取一个样本）
    print("\n📂 加载数据集...")
    dataset = FiveCentersMultimodalDatasetV3_2(
        csv_path=Path(config.data_root) / 'temp_val_labels.csv',
        data_root=config.data_root
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 5. 获取一个样本
    print("\n🔬 获取测试样本...")
    batch = next(iter(dataloader))
    
    # 提取数据
    oct_images = batch['oct_images'].to(device)  # [B, N_frames, C, H, W]
    colposcopy_images = batch['colposcopy_images'].to(device)  # [B, N_images, C, H, W]
    clinical_features = batch['clinical_features'].to(device)  # [B, 7]
    image_names = batch.get('image_names', ['test_image'] * oct_images.size(0))  # 必需！
    
    print(f"   ✅ oct_images shape: {oct_images.shape}")
    print(f"   ✅ colposcopy_images shape: {colposcopy_images.shape}")
    print(f"   ✅ clinical_features shape: {clinical_features.shape}")
    print(f"   ✅ image_names: {image_names}")
    
    # 6. 提取特征（和训练脚本一样）
    print("\n🔧 提取ViT特征...")
    B_oct = oct_images.shape[0]
    if len(oct_images.shape) == 5:  # [B, F, C, H, W]
        F_oct = oct_images.shape[1]
        oct_images_flat = oct_images.view(B_oct * F_oct, *oct_images.shape[2:])
        oct_features_patch = extract_patch_features_with_vit(oct_images_flat, device, batch_size=4)
        oct_features_patch = oct_features_patch.view(B_oct, F_oct, 196, 768).mean(dim=1)  # [B, 196, 768]
    else:
        oct_features_patch = extract_patch_features_with_vit(oct_images, device, batch_size=4)
    
    B_colpo = colposcopy_images.shape[0]
    if len(colposcopy_images.shape) == 5:  # [B, N, C, H, W]
        N_colpo = colposcopy_images.shape[1]
        colpo_images_flat = colposcopy_images.view(B_colpo * N_colpo, *colposcopy_images.shape[2:])
        colpo_features_patch = extract_patch_features_with_vit(colpo_images_flat, device, batch_size=4)
        colpo_features_patch = colpo_features_patch.view(B_colpo, N_colpo, 196, 768).mean(dim=1)  # [B, 196, 768]
    else:
        colpo_features_patch = extract_patch_features_with_vit(colposcopy_images, device, batch_size=4)
    
    print(f"   ✅ oct_features_patch shape: {oct_features_patch.shape}")
    print(f"   ✅ colpo_features_patch shape: {colpo_features_patch.shape}")
    
    # 7. 测试forward调用
    print("\n🚀 测试模型forward调用...")
    try:
        with torch.no_grad():
            output = model(
                f_oct=oct_features_patch,
                f_colpo=colpo_features_patch,
                image_names=image_names,
                clinical_features=clinical_features
            )
        
        print("   ✅ Forward调用成功！")
        print(f"   ✅ 输出keys: {output.keys()}")
        if 'logits' in output:
            print(f"   ✅ Logits shape: {output['logits'].shape}")
            print(f"   ✅ 预测类别: {output['logits'].argmax(dim=1).item()}")
        
        return True, model, batch
        
    except Exception as e:
        print(f"   ❌ Forward调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

if __name__ == '__main__':
    success, model, batch = test_model_forward()
    if success:
        print("\n" + "=" * 80)
        print("✅ Step 1 完成：模型forward调用成功！")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ Step 1 失败：需要修复模型调用")
        print("=" * 80)

