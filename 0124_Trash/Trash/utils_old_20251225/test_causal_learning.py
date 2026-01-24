#!/usr/bin/env python3
"""
因果学习组件测试脚本
验证因果学习组件是否正常工作
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models_causal_gnn import CausalGNN, CausalMultimodalTransformer
from util.datasets import MultimodalCervicalDataset
from torch.utils.data import DataLoader
import argparse

def test_causal_gnn():
    """测试因果GNN组件"""
    print("=== 测试因果GNN组件 ===")
    
    # 创建测试数据
    batch_size = 4
    input_dim = 8  # 临床数据维度
    causal_dim = 64
    
    # 创建模型
    causal_gnn = CausalGNN(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=1024,
        causal_dim=causal_dim,
        num_heads=4,
        dropout=0.1
    )
    
    # 创建测试输入
    x = torch.randn(batch_size, input_dim)
    intervention_targets = torch.tensor([0, -1, 2, -1])  # 部分样本进行干预
    intervention_values = torch.tensor([1.0, 0.0, -0.5, 0.0])
    
    print(f"输入形状: {x.shape}")
    print(f"干预目标: {intervention_targets}")
    print(f"干预值: {intervention_values}")
    
    # 前向传播
    outputs = causal_gnn(x, intervention_targets, intervention_values, generate_counterfactual=True)
    
    print("=== 输出检查 ===")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {type(value)}")
    
    # 计算因果损失
    targets = torch.randint(0, 2, (batch_size,))
    losses = causal_gnn.compute_causal_losses(outputs, targets)
    
    print("=== 因果损失检查 ===")
    for loss_name, loss_value in losses.items():
        print(f"{loss_name}: {loss_value.item():.4f}")
    
    # 解释因果效应
    effects = causal_gnn.interpret_causal_effects(outputs)
    
    print("=== 因果效应分析 ===")
    for effect_name, effect_value in effects.items():
        if isinstance(effect_value, torch.Tensor):
            print(f"{effect_name}: {effect_value.shape}")
        else:
            print(f"{effect_name}: {effect_value}")
    
    print("✓ 因果GNN组件测试完成")

def test_causal_multimodal():
    """测试因果多模态Transformer"""
    print("\n=== 测试因果多模态Transformer ===")
    
    # 创建模拟的backbone模型
    class MockBackbone(nn.Module):
        def __init__(self, feature_dim=1280):
            super().__init__()
            self.feature_dim = feature_dim
            self.classifier = nn.Linear(feature_dim, 1000)
        
        def forward(self, x):
            batch_size = x.size(0)
            features = torch.randn(batch_size, self.feature_dim)
            return {'features': features}
    
    # 创建模型
    oct_model = MockBackbone(1280)
    col_model = MockBackbone(1280)
    
    model = CausalMultimodalTransformer(
        oct_model=oct_model,
        col_model=col_model,
        num_classes=2,
        embed_dim=1024,
        causal_dim=64,
        dropout_rate=0.2
    )
    
    # 创建测试输入
    batch_size = 4
    oct_img = torch.randn(batch_size, 3, 224, 224)
    col_img = torch.randn(batch_size, 3, 3, 224, 224)  # [B, 3, C, H, W]
    clinical_data = torch.randn(batch_size, 8)  # 临床数据
    
    intervention_targets = torch.tensor([0, -1, 2, -1])
    intervention_values = torch.tensor([1.0, 0.0, -0.5, 0.0])
    
    print(f"OCT图像形状: {oct_img.shape}")
    print(f"COL图像形状: {col_img.shape}")
    print(f"临床数据形状: {clinical_data.shape}")
    
    # 前向传播
    outputs = model(oct_img, col_img, clinical_data, 
                   intervention_targets, intervention_values,
                   generate_counterfactual=True)
    
    print("=== 输出检查 ===")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {type(value)}")
    
    # 计算损失
    targets = torch.randint(0, 2, (batch_size,))
    losses = model.compute_losses(outputs, targets)
    
    print("=== 损失检查 ===")
    for loss_name, loss_value in losses.items():
        print(f"{loss_name}: {loss_value.item():.4f}")
    
    # 解释因果效应
    effects = model.interpret_causal_effects(outputs)
    
    print("=== 因果效应分析 ===")
    for effect_name, effect_value in effects.items():
        if isinstance(effect_value, torch.Tensor):
            print(f"{effect_name}: {effect_value.shape}")
        else:
            print(f"{effect_name}: {effect_value}")
    
    print("✓ 因果多模态Transformer测试完成")

def test_with_real_data():
    """使用真实数据测试"""
    print("\n=== 使用真实数据测试 ===")
    
    # 创建参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./5centers_multi')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--input_size', type=int, default=224)
    args = parser.parse_args([])  # 空参数列表
    
    try:
        # 加载数据集
        dataset = MultimodalCervicalDataset(
            root=os.path.join(args.data_path, 'train'),
            is_train='train',
            args=args,
            transform=None
        )
        
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        # 获取一个batch
        sample_batch = next(iter(dataloader))
        oct_img, col_img, meta, label, temporal, _ = sample_batch
        
        print(f"真实数据形状:")
        print(f"OCT: {oct_img.shape}")
        print(f"COL: {col_img.shape}")
        print(f"Meta: {meta.shape}")
        print(f"Label: {label.shape}")
        print(f"Temporal: {temporal.shape}")
        
        # 创建模型
        class MockBackbone(nn.Module):
            def __init__(self, feature_dim=1280):
                super().__init__()
                self.feature_dim = feature_dim
                self.classifier = nn.Linear(feature_dim, 1000)
            
            def forward(self, x):
                batch_size = x.size(0)
                features = torch.randn(batch_size, self.feature_dim)
                return {'features': features}
        
        oct_model = MockBackbone(1280)
        col_model = MockBackbone(1280)
        
        model = CausalMultimodalTransformer(
            oct_model=oct_model,
            col_model=col_model,
            num_classes=2,
            embed_dim=1024,
            causal_dim=64
        )
        
        # 处理临床数据
        clinical_data = torch.cat((meta, temporal), dim=1)
        
        # 前向传播
        outputs = model(oct_img, col_img, clinical_data, 
                       generate_counterfactual=True)
        
        print("=== 真实数据输出检查 ===")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
        
        # 计算损失
        losses = model.compute_losses(outputs, label)
        
        print("=== 真实数据损失检查 ===")
        for loss_name, loss_value in losses.items():
            print(f"{loss_name}: {loss_value.item():.4f}")
        
        print("✓ 真实数据测试完成")
        
    except Exception as e:
        print(f"真实数据测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("开始因果学习组件测试...")
    
    # 测试因果GNN
    test_causal_gnn()
    
    # 测试因果多模态Transformer
    test_causal_multimodal()
    
    # 测试真实数据
    test_with_real_data()
    
    print("\n=== 所有测试完成 ===") 