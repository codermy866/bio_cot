#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速生成因果图可视化结果
从已训练模型或测试数据生成可视化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from utils.visualize_causal_graph import (
    visualize_causal_adjacency_matrix,
    visualize_causal_graph_networkx
)

def create_output_directory(output_dir='./causal_analysis'):
    """创建输出目录"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录: {output_dir}")
    return output_dir

def generate_test_visualization(output_dir='./causal_analysis'):
    """生成测试可视化（使用模拟数据）"""
    print("\n" + "="*60)
    print("🎨 生成测试因果图可视化")
    print("="*60)
    
    # 创建输出目录
    output_dir = create_output_directory(output_dir)
    
    # 创建模拟因果邻接矩阵（3×3，符合医学先验）
    print("\n📊 创建模拟因果邻接矩阵...")
    causal_adj = np.array([
        [0.0, 0.0, 0.8],  # OCT ← Clinical (强度0.8)
        [0.0, 0.0, 0.7],  # Colposcopy ← Clinical (强度0.7)
        [0.0, 0.0, 0.0]   # Clinical (根源)
    ])
    
    print(f"   矩阵形状: {causal_adj.shape}")
    print(f"   矩阵内容:\n{causal_adj}")
    
    # 转换为torch tensor（模拟模型输出）
    causal_adj_tensor = torch.from_numpy(causal_adj).float()
    
    # 1. 可视化因果邻接矩阵
    print("\n📈 生成因果邻接矩阵热力图...")
    save_path = os.path.join(output_dir, 'causal_adjacency_matrix.png')
    visualize_causal_adjacency_matrix(
        causal_adj_tensor,
        save_path=save_path,
        title="Causal Adjacency Matrix (医学先验知识)"
    )
    print(f"   ✅ 已保存: {save_path}")
    
    # 2. 可视化因果图网络
    print("\n🕸️  生成因果图网络图...")
    save_path = os.path.join(output_dir, 'causal_graph_network.png')
    visualize_causal_graph_networkx(
        causal_adj_tensor,
        save_path=save_path,
        title="Causal Graph Network (医学先验知识)"
    )
    print(f"   ✅ 已保存: {save_path}")
    
    # 3. 保存数值数据
    print("\n💾 保存数值数据...")
    npy_path = os.path.join(output_dir, 'causal_adjacency_matrix.npy')
    np.save(npy_path, causal_adj)
    print(f"   ✅ 已保存: {npy_path}")
    
    print("\n" + "="*60)
    print("✅ 测试可视化生成完成！")
    print("="*60)
    print(f"\n📂 所有文件保存在: {os.path.abspath(output_dir)}")
    print("\n生成的文件:")
    print(f"  - causal_adjacency_matrix.png")
    print(f"  - causal_graph_network.png")
    print(f"  - causal_adjacency_matrix.npy")

def generate_from_model_checkpoint(model_path, output_dir='./causal_analysis'):
    """从模型检查点生成可视化"""
    print("\n" + "="*60)
    print("🎨 从模型检查点生成可视化")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    print(f"📦 加载模型: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 检查是否有保存的因果邻接矩阵
        causal_adj = None
        
        # 方法1: 直接从checkpoint获取
        if 'causal_adj' in checkpoint:
            causal_adj = checkpoint['causal_adj']
            print("   ✅ 从checkpoint中找到因果邻接矩阵")
        
        # 方法2: 从模型状态中提取（如果模型支持）
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # 查找因果相关的参数
            causal_keys = [k for k in state_dict.keys() if 'causal' in k.lower()]
            if causal_keys:
                print(f"   ℹ️  找到因果相关参数: {causal_keys[:5]}...")
                # 尝试提取因果权重
                for key in causal_keys:
                    if 'weight' in key or 'adj' in key:
                        causal_adj = state_dict[key]
                        print(f"   ✅ 从参数 '{key}' 提取因果矩阵")
                        break
        
        if causal_adj is not None:
            # 确保是numpy或tensor格式
            if isinstance(causal_adj, torch.Tensor):
                causal_adj = causal_adj.detach().cpu().numpy()
            
            # 如果是batch维度，取平均
            if causal_adj.ndim == 3:
                causal_adj = causal_adj.mean(axis=0)
            
            print(f"   📊 因果矩阵形状: {causal_adj.shape}")
            
            # 创建输出目录
            output_dir = create_output_directory(output_dir)
            
            # 可视化
            causal_adj_tensor = torch.from_numpy(causal_adj).float()
            
            # 1. 热力图
            save_path = os.path.join(output_dir, 'causal_adjacency_from_model.png')
            visualize_causal_adjacency_matrix(
                causal_adj_tensor,
                save_path=save_path,
                title="Causal Adjacency Matrix (From Model)"
            )
            print(f"   ✅ 已保存: {save_path}")
            
            # 2. 网络图
            save_path = os.path.join(output_dir, 'causal_graph_from_model.png')
            visualize_causal_graph_networkx(
                causal_adj_tensor,
                save_path=save_path,
                title="Causal Graph Network (From Model)"
            )
            print(f"   ✅ 已保存: {save_path}")
            
            # 3. 保存数值
            npy_path = os.path.join(output_dir, 'causal_adjacency_from_model.npy')
            np.save(npy_path, causal_adj)
            print(f"   ✅ 已保存: {npy_path}")
            
        else:
            print("   ⚠️  模型检查点中没有找到因果邻接矩阵")
            print("   💡 建议: 运行训练脚本生成可视化，或使用测试可视化")
            
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()

def generate_from_enhanced_clip_results():
    """从增强因果CLIP结果生成可视化"""
    print("\n" + "="*60)
    print("🎨 从增强因果CLIP结果生成可视化")
    print("="*60)
    
    model_path = 'enhanced_causal_clip_results/best_model.pth'
    
    if os.path.exists(model_path):
        generate_from_model_checkpoint(model_path, output_dir='./causal_analysis/enhanced_clip')
    else:
        print(f"❌ 未找到模型: {model_path}")
        print("   💡 提示: 需要先运行增强因果CLIP训练")

def generate_from_adaptive_causal_results():
    """从自适应因果干预结果生成可视化"""
    print("\n" + "="*60)
    print("🎨 从自适应因果干预结果生成可视化")
    print("="*60)
    
    model_path = 'adaptive_causal_intervention_results/best_model.pth'
    
    if os.path.exists(model_path):
        generate_from_model_checkpoint(model_path, output_dir='./causal_analysis/adaptive_causal')
    else:
        print(f"❌ 未找到模型: {model_path}")
        print("   💡 提示: 需要先运行自适应因果干预训练")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='生成因果图可视化')
    parser.add_argument('--mode', type=str, default='test', 
                       choices=['test', 'enhanced_clip', 'adaptive_causal', 'all'],
                       help='生成模式: test(测试), enhanced_clip(增强CLIP), adaptive_causal(自适应因果), all(全部)')
    parser.add_argument('--output_dir', type=str, default='./causal_analysis',
                       help='输出目录')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型检查点路径（如果指定，从此模型生成）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🎨 因果图可视化生成器")
    print("="*60)
    
    # 如果指定了模型路径，直接使用
    if args.model_path:
        generate_from_model_checkpoint(args.model_path, args.output_dir)
        return
    
    # 根据模式生成
    if args.mode == 'test' or args.mode == 'all':
        generate_test_visualization(args.output_dir)
    
    if args.mode == 'enhanced_clip' or args.mode == 'all':
        generate_from_enhanced_clip_results()
    
    if args.mode == 'adaptive_causal' or args.mode == 'all':
        generate_from_adaptive_causal_results()
    
    print("\n" + "="*60)
    print("✅ 所有可视化生成完成！")
    print("="*60)
    print(f"\n📂 输出目录: {os.path.abspath(args.output_dir)}")
    print("\n💡 提示:")
    print("  - 查看PNG文件: 使用图片查看器")
    print("  - 查看NPY文件: 使用 np.load() 加载")
    print("  - 论文使用: PNG文件（300 DPI）")

if __name__ == '__main__':
    main()

