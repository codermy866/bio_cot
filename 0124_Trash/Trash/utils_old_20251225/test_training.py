import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
sys.path.append('.')

def get_args_parser():
    parser = argparse.ArgumentParser('Test Training Script', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=2, type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_path', default='./5centers_multi/', type=str, help='Dataset path')
    parser.add_argument('--input_size', default=224, type=int, help='Input image size')
    parser.add_argument('--task', default='cervical_cancer', type=str, help='Task name')
    return parser

def test_dataset_loading(args):
    """测试数据集加载"""
    print("[INFO] 开始测试数据集加载...")
    
    try:
        from util.datasets import build_dataset
        dataset_train, class_weights = build_dataset(is_train='train', args=args)
        print(f"[SUCCESS] 数据集加载成功，样本数: {len(dataset_train)}")
        
        # 测试第一个样本
        sample = dataset_train[0]
        print(f"[INFO] 第一个样本类型: {type(sample)}")
        print(f"[INFO] 第一个样本内容: {[type(x) for x in sample]}")
        
        if isinstance(sample, tuple):
            print(f"[INFO] 元组长度: {len(sample)}")
            print(f"[INFO] 各元素形状:")
            for i, item in enumerate(sample):
                if hasattr(item, 'shape'):
                    print(f"  [{i}]: {item.shape}, {item.dtype}")
                else:
                    print(f"  [{i}]: {type(item)}")
        
        return True
    except Exception as e:
        print(f"[ERROR] 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation(args):
    """测试模型创建"""
    print("[INFO] 开始测试模型创建...")
    
    try:
        import models_causal_gnn
        from torchvision.models import vit_b_16
        
        # 创建简单的ViT模型
        oct_model = vit_b_16(pretrained=True)
        col_model = vit_b_16(pretrained=True)
        
        # 修改分类头
        oct_model.heads = nn.Linear(768, args.embed_dim if hasattr(args, 'embed_dim') else 768)
        col_model.heads = nn.Linear(768, args.embed_dim if hasattr(args, 'embed_dim') else 768)
        
        # 创建因果多模态模型
        model = models_causal_gnn.CausalMultimodalTransformer(
            oct_model=oct_model,
            col_model=col_model,
            num_classes=2,
            embed_dim=768,
            causal_dim=64,
            dropout_rate=0.2
        )
        
        print(f"[SUCCESS] 模型创建成功")
        print(f"[INFO] 模型参数数量: {sum(p.numel() for p in model.parameters())}")
        
        return True
    except Exception as e:
        print(f"[ERROR] 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_batch(args):
    """测试单个batch的前向传播"""
    print("[INFO] 开始测试单个batch前向传播...")
    
    try:
        from util.datasets import build_dataset
        import models_causal_gnn
        from torchvision.models import vit_b_16
        
        # 加载数据集
        dataset_train, _ = build_dataset(is_train='train', args=args)
        data_loader = DataLoader(dataset_train, batch_size=2, shuffle=False)
        
        # 创建模型
        oct_model = vit_b_16(pretrained=True)
        col_model = vit_b_16(pretrained=True)
        oct_model.heads = nn.Linear(768, 768)
        col_model.heads = nn.Linear(768, 768)
        
        model = models_causal_gnn.CausalMultimodalTransformer(
            oct_model=oct_model,
            col_model=col_model,
            num_classes=2,
            embed_dim=768,
            causal_dim=64,
            dropout_rate=0.2
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        # 获取一个batch
        for batch in data_loader:
            print(f"[INFO] Batch类型: {type(batch)}")
            print(f"[INFO] Batch长度: {len(batch)}")
            
            # 解包数据
            oct_img = batch[0].to(device)
            col_img = batch[1].to(device)
            meta = batch[2].to(device).float()
            targets = batch[3].to(device)
            temporal = batch[4].to(device).float()
            
            print(f"[INFO] 数据形状:")
            print(f"  oct_img: {oct_img.shape}")
            print(f"  col_img: {col_img.shape}")
            print(f"  meta: {meta.shape}")
            print(f"  targets: {targets.shape}")
            print(f"  temporal: {temporal.shape}")
            
            # 构建临床数据
            clinical_data = torch.cat((meta, temporal), dim=1).to(device).float()
            print(f"  clinical_data: {clinical_data.shape}")
            
            # 前向传播
            with torch.no_grad():
                outputs = model(oct_img, col_img, clinical_data)
            
            print(f"[SUCCESS] 前向传播成功")
            print(f"[INFO] 输出键: {list(outputs.keys())}")
            for k, v in outputs.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: {v.shape}")
            
            break
        
        return True
    except Exception as e:
        print(f"[ERROR] 单个batch测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main(args):
    print("[INFO] 开始测试训练流程...")
    
    # 测试数据集加载
    if not test_dataset_loading(args):
        print("[ERROR] 数据集加载测试失败，退出")
        return
    
    # 测试模型创建
    if not test_model_creation(args):
        print("[ERROR] 模型创建测试失败，退出")
        return
    
    # 测试单个batch
    if not test_single_batch(args):
        print("[ERROR] 单个batch测试失败，退出")
        return
    
    print("[SUCCESS] 所有测试通过！训练流程应该可以正常工作。")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args) 