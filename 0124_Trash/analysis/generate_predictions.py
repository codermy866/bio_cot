#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从训练好的模型重新生成预测结果
用于后续的Bootstrap CI、亚组分析和DCA分析
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset


def load_model_and_predict(result_dir: str, data_path: str = '5centers_multi', 
                           model_type: str = 'swint', device: str = 'cuda'):
    """
    加载模型并生成预测结果
    
    Args:
        result_dir: 模型结果目录（包含best_model.pth）
        data_path: 数据路径
        model_type: 模型类型 ('swint', 'cnn', 'vmamba')
        device: 设备 ('cuda' or 'cpu')
    
    Returns:
        (y_true, y_pred, y_probs, metadata)
    """
    print(f"🔄 加载模型: {result_dir}")
    print(f"   模型类型: {model_type}")
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model_path = os.path.join(result_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"❌ 未找到模型文件: {model_path}")
        return None, None, None, None
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 根据模型类型加载模型
    if model_type.lower() in ['swint', 'swin']:
        try:
            from models.SwinT.swin_multimodal_model import SwinTMultimodalTransformer
            # 从checkpoint或配置文件获取模型参数
            embed_dim = 768  # 默认值，可以从checkpoint中获取
            num_heads = 12
            if 'swin_small' in result_dir or 'swin_base' in result_dir:
                if 'swin_base' in result_dir:
                    embed_dim = 1024
                    num_heads = 16
                    swin_name = 'swin_base_patch4_window7_224'
                else:
                    swin_name = 'swin_small_patch4_window7_224'
            else:
                swin_name = 'swin_tiny_patch4_window7_224'
            
            model = SwinTMultimodalTransformer(
                num_classes=2,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=0.2,
                clinical_dim=7,
                oct_num_frames=48,
                col_num_frames=3,
                swin_name=swin_name,
                pretrained=False,
                input_size=224,
                use_frame_attention=False,
            ).to(device)
        except Exception as e:
            print(f"❌ 加载Swin模型失败: {e}")
            return None, None, None, None
    elif model_type.lower() == 'cnn':
        from models.cnn_multimodal_model import CNNMultimodalTransformer
        model = CNNMultimodalTransformer(
            num_classes=2,
            embed_dim=1280,
            num_heads=20,
            dropout=0.3,
            clinical_dim=7,
            oct_num_frames=120,
            col_num_frames=3
        ).to(device)
    elif model_type.lower() == 'vmamba':
        from models.vmamba_multimodal_model import VMambaMultimodalTransformer
        model = VMambaMultimodalTransformer(
            num_classes=2,
            embed_dim=1024,
            num_heads=16,
            dropout=0.1,
            clinical_dim=7,
            oct_num_frames=120,
            col_num_frames=3,
            img_size=224,
            patch_size=16,
            depth=10,
            d_state=16
        ).to(device)
    else:
        print(f"❌ 未知的模型类型: {model_type}")
        return None, None, None, None
    
    # 加载模型权重（使用strict=False以跳过不匹配的层）
    try:
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 尝试严格加载
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✅ 模型权重加载成功（严格模式）")
        except Exception as e:
            print(f"⚠️  严格加载失败，尝试非严格模式: {e}")
            # 非严格加载：跳过不匹配的层
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() 
                             if k in model_dict and model_dict[k].shape == v.shape}
            
            # 更新模型字典
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            
            skipped = len(state_dict) - len(pretrained_dict)
            loaded = len(pretrained_dict)
            print(f"✅ 模型权重加载成功（非严格模式）")
            print(f"   已加载: {loaded} 层")
            if skipped > 0:
                print(f"   跳过: {skipped} 层（形状不匹配）")
    except Exception as e:
        print(f"❌ 无法加载模型权重: {e}")
        return None, None, None, None
    
    model.eval()
    print(f"✅ 模型加载成功")
    
    # 准备数据
    print(f"📥 加载测试数据...")
    
    class Args:
        def __init__(self, data_path):
            self.data_path = data_path
            self.input_size = 224
            self.oct_num_frames = 48 if 'swin' in model_type.lower() else 120
            self.oct_cache_dir = 'oct_cache_optimized'
            self.use_text_contrastive = False
            self.use_pretrained_backbones = True
            self.oct_points = 12
            self.oct_frames_per_point = 10
    
    args = Args(data_path)
    
    test_dataset = EnhancedMultimodalCervicalDataset(
        root=os.path.join(data_path, 'test'),
        is_train='test',
        args=args,
        transform=None,
        use_enhanced_oct=True,
        cache_oct_features=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✅ 测试数据加载完成: {len(test_dataset)} 个样本")
    
    # 生成预测
    print(f"🔮 生成预测结果...")
    all_labels = []
    all_probs = []
    all_preds = []
    all_metadata = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="预测中"):
            if isinstance(batch, dict):
                oct_images = batch.get('oct_images')
                col_images = batch.get('col_images')
                clinical_features = batch['clinical_features'] if 'clinical_features' in batch else batch.get('clinical')
                labels = batch['label'] if 'label' in batch else batch.get('labels')
                sample_ids = batch.get('sample_id', [f'sample_{i}' for i in range(len(labels))])
            else:
                if len(batch) == 5:
                    oct_images, col_images, clinical_features, labels, sample_ids = batch
                else:
                    oct_images, col_images, clinical_features, labels = batch
                    sample_ids = [f'sample_{i}' for i in range(len(labels))]
            
            oct_images = oct_images.to(device)
            col_images = col_images.to(device)
            clinical_features = clinical_features.to(device)
            labels = labels.to(device)
            
            outputs = model(oct_images, col_images, clinical_features)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 阳性概率
            all_preds.extend(preds.cpu().numpy())
            
            # 收集元数据（如果有）
            if isinstance(sample_ids, (list, tuple)):
                all_metadata.extend(sample_ids)
            else:
                all_metadata.extend([f'sample_{i}' for i in range(len(labels))])
    
    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)
    y_pred = np.array(all_preds)
    
    print(f"✅ 预测完成: {len(y_true)} 个样本")
    
    # 尝试加载元数据（年龄、HPV类型、中心等）
    metadata = None
    try:
        test_labels_file = os.path.join(data_path, 'test_labels.csv')
        if os.path.exists(test_labels_file):
            metadata_df = pd.read_csv(test_labels_file)
            if len(metadata_df) == len(y_true):
                metadata = metadata_df
                print(f"✅ 元数据加载成功")
    except Exception as e:
        print(f"⚠️  加载元数据失败: {e}")
    
    # 保存预测结果
    output_dir = result_dir
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'val_labels.npy'), y_true)
    np.save(os.path.join(output_dir, 'val_probs.npy'), y_probs)
    np.save(os.path.join(output_dir, 'val_preds.npy'), y_pred)
    
    if metadata is not None:
        metadata.to_csv(os.path.join(output_dir, 'val_metadata.csv'), index=False)
        print(f"✅ 预测结果已保存到: {output_dir}")
        print(f"   - val_labels.npy")
        print(f"   - val_probs.npy")
        print(f"   - val_preds.npy")
        print(f"   - val_metadata.csv")
    else:
        print(f"✅ 预测结果已保存到: {output_dir}")
        print(f"   - val_labels.npy")
        print(f"   - val_probs.npy")
        print(f"   - val_preds.npy")
    
    return y_true, y_pred, y_probs, metadata


def main():
    import argparse
    parser = argparse.ArgumentParser(description='从训练好的模型生成预测结果')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='模型结果目录（包含best_model.pth）')
    parser.add_argument('--data_path', type=str, default='5centers_multi',
                       help='数据路径')
    parser.add_argument('--model_type', type=str, default='swint',
                       choices=['swint', 'swin', 'cnn', 'vmamba'],
                       help='模型类型')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda or cpu)')
    
    args = parser.parse_args()
    
    y_true, y_pred, y_probs, metadata = load_model_and_predict(
        args.result_dir,
        args.data_path,
        args.model_type,
        args.device
    )
    
    if y_true is None:
        print("❌ 生成预测结果失败")
        sys.exit(1)
    
    print("\n✅ 预测结果生成完成！")
    print(f"   可以使用以下命令运行分析:")
    print(f"   python analysis/run_lancet_analysis.py --result_dir {args.result_dir}")


if __name__ == '__main__':
    main()

