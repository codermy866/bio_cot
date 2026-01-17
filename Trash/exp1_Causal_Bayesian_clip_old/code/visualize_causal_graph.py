#!/usr/bin/env python3
"""
因果图可视化脚本
可视化因果图结构、项目文件结构和实验流程
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models_causal_gnn import CausalGNN, CausalMultimodalTransformer
import argparse

def visualize_causal_adjacency_matrix(causal_adj, save_path=None, title="Causal Adjacency Matrix"):
    """可视化因果邻接矩阵"""
    plt.figure(figsize=(12, 10))
    
    # 转换为numpy数组
    if isinstance(causal_adj, torch.Tensor):
        causal_adj = causal_adj.detach().cpu().numpy()
    
    # 如果是batch，取平均值
    if causal_adj.ndim == 3:
        causal_adj = causal_adj.mean(axis=0)
    
    # 创建热力图
    sns.heatmap(causal_adj, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                cbar_kws={'label': 'Causal Strength'},
                xticklabels=range(causal_adj.shape[0]),
                yticklabels=range(causal_adj.shape[1]))
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Causal Factors', fontsize=12)
    plt.ylabel('Causal Factors', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"因果邻接矩阵已保存到: {save_path}")
    
    plt.show()

def visualize_causal_graph_networkx(causal_adj, save_path=None, title="Causal Graph"):
    """使用NetworkX可视化因果图"""
    plt.figure(figsize=(15, 12))
    
    # 转换为numpy数组
    if isinstance(causal_adj, torch.Tensor):
        causal_adj = causal_adj.detach().cpu().numpy()
    
    # 如果是batch，取平均值
    if causal_adj.ndim == 3:
        causal_adj = causal_adj.mean(axis=0)
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点
    for i in range(causal_adj.shape[0]):
        G.add_node(i, label=f'Factor_{i}')
    
    # 添加边（只添加强度大于阈值的边）
    threshold = 0.1
    for i in range(causal_adj.shape[0]):
        for j in range(causal_adj.shape[1]):
            if i != j and abs(causal_adj[i, j]) > threshold:
                G.add_edge(i, j, weight=causal_adj[i, j])
    
    # 设置布局
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue',
                          node_size=1000,
                          alpha=0.8)
    
    # 绘制边
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # 根据权重设置边的颜色和宽度
    edge_colors = ['red' if w < 0 else 'green' for w in weights]
    edge_widths = [abs(w) * 3 for w in weights]
    
    nx.draw_networkx_edges(G, pos,
                          edge_color=edge_colors,
                          width=edge_widths,
                          alpha=0.6,
                          arrows=True,
                          arrowsize=20)
    
    # 添加标签
    labels = {node: f'F{node}' for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold')
    
    # 添加边标签
    edge_labels = {(u, v): f'{w:.2f}' for (u, v), w in zip(edges, weights)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"因果图已保存到: {save_path}")
    
    plt.show()

def create_project_structure_diagram():
    """创建项目结构图"""
    plt.figure(figsize=(16, 12))
    
    # 定义项目结构
    structure = {
        'HPV-TCT3multi-RETFound_MAE': {
            '📁 5centers_multi/': {
                '📄 train_labels.csv': '训练标签',
                '📄 test_labels.csv': '测试标签',
                '📁 train/': {
                    '📁 oct/': 'OCT图像',
                    '📁 col/': 'COL图像'
                },
                '📁 test/': {
                    '📁 oct/': 'OCT图像',
                    '📁 col/': 'COL图像'
                }
            },
            '📁 util/': {
                '📄 datasets.py': '数据集加载',
                '📄 transforms.py': '数据变换'
            },
            '📁 models/': {
                '📄 models_causal_gnn.py': '因果GNN模型',
                '📄 models_vit.py': 'Vision Transformer',
                '📄 models_causal_directed_graph.py': '因果有向图'
            },
            '📁 training/': {
                '📄 main_causal_finetune.py': '因果微调训练',
                '📄 engine_finetune.py': '训练引擎'
            },
            '📁 analysis/': {
                '📄 causal_analysis.py': '因果分析',
                '📄 visualize_causal_graph.py': '因果图可视化'
            },
            '📁 logs/': '训练日志',
            '📁 checkpoints/': '模型检查点',
            '📁 causal_results/': '因果分析结果'
        }
    }
    
    # 创建树状图
    def plot_tree(data, x=0, y=0, level=0):
        for key, value in data.items():
            if isinstance(value, dict):
                plt.text(x, y, key, fontsize=10, ha='left', va='center')
                plot_tree(value, x + 2, y - 1, level + 1)
            else:
                plt.text(x, y, f"{key} - {value}", fontsize=8, ha='left', va='center')
            y -= 1.5
    
    plt.subplot(111)
    plot_tree(structure)
    plt.title('项目文件结构图', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.xlim(-1, 20)
    plt.ylim(-20, 2)
    
    plt.tight_layout()
    plt.savefig('project_structure.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_experiment_workflow():
    """创建实验流程图"""
    plt.figure(figsize=(18, 12))
    
    # 定义流程步骤
    steps = [
        ('数据准备', '📊 加载5centers_multi数据集\n📄 CSV标签文件\n🖼️ OCT/COL图像'),
        ('数据预处理', '🔄 图像变换\n📏 尺寸调整\n🔢 标签编码'),
        ('模型构建', '🧠 CausalGNN\n🔗 CausalIntervention\n🔄 CounterfactualGenerator'),
        ('因果编码', '📊 临床数据编码\n🔍 因果因子提取\n📈 变分自编码'),
        ('因果图发现', '🕸️ 邻接矩阵生成\n📊 DAG约束\n🔍 稀疏性约束'),
        ('干预分析', '🎯 目标选择\n💉 干预执行\n📊 效果评估'),
        ('反事实生成', '🔄 事实vs反事实\n🎲 场景生成\n📊 差异分析'),
        ('多模态融合', '🖼️ OCT特征\n🔍 COL特征\n📊 临床特征'),
        ('训练优化', '📈 损失计算\n🔄 反向传播\n💾 模型保存'),
        ('结果分析', '📊 因果效应\n📈 性能评估\n🎨 可视化')
    ]
    
    # 绘制流程图
    for i, (step, description) in enumerate(steps):
        x = (i % 5) * 3.5
        y = -(i // 5) * 4
        
        # 绘制节点
        circle = plt.Circle((x, y), 0.8, color='lightblue', alpha=0.8)
        plt.gca().add_patch(circle)
        
        # 添加文本
        plt.text(x, y, step, ha='center', va='center', fontsize=10, fontweight='bold')
        plt.text(x, y-1.5, description, ha='center', va='top', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 绘制箭头
        if i < len(steps) - 1:
            next_x = ((i + 1) % 5) * 3.5
            next_y = -((i + 1) // 5) * 4
            plt.arrow(x + 0.8, y, next_x - x - 1.6, next_y - y, 
                     head_width=0.2, head_length=0.2, fc='black', ec='black')
    
    plt.title('因果学习实验流程图', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.axis('off')
    plt.xlim(-1, 18)
    plt.ylim(-8, 2)
    
    plt.tight_layout()
    plt.savefig('experiment_workflow.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_causal_graph_visualization():
    """测试因果图可视化"""
    print("=== 测试因果图可视化 ===")
    
    # 创建测试模型
    causal_gnn = CausalGNN(
        input_dim=8,
        hidden_dim=128,
        output_dim=1024,
        causal_dim=16,  # 使用较小的维度便于可视化
        num_heads=4,
        dropout=0.1
    )
    
    # 创建测试数据
    batch_size = 4
    x = torch.randn(batch_size, 8)
    
    # 前向传播
    outputs = causal_gnn(x, generate_counterfactual=True)
    causal_adj = outputs['causal_adj']
    
    print(f"因果邻接矩阵形状: {causal_adj.shape}")
    print(f"因果邻接矩阵值范围: [{causal_adj.min().item():.4f}, {causal_adj.max().item():.4f}]")
    
    # 可视化因果邻接矩阵
    visualize_causal_adjacency_matrix(
        causal_adj, 
        save_path='causal_adjacency_matrix.png',
        title="Causal Adjacency Matrix (Test)"
    )
    
    # 可视化因果图网络
    visualize_causal_graph_networkx(
        causal_adj,
        save_path='causal_graph_network.png',
        title="Causal Graph Network (Test)"
    )
    
    print("✓ 因果图可视化测试完成")

def ensure_gpu_training():
    """确保GPU训练正常运行"""
    print("=== GPU训练检查 ===")
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"✓ CUDA可用，设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  设备 {i}: {torch.cuda.get_device_name(i)}")
        
        # 设置默认设备
        device = torch.device('cuda:0')
        print(f"✓ 使用设备: {device}")
        
        # 测试GPU内存
        try:
            test_tensor = torch.randn(1000, 1000).to(device)
            print(f"✓ GPU内存测试成功，张量形状: {test_tensor.shape}")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ GPU内存测试失败: {e}")
            return False
        
        return True
    else:
        print("✗ CUDA不可用，将使用CPU")
        return False

def create_training_monitor_script():
    """创建训练监控脚本"""
    monitor_script = '''#!/bin/bash
# 训练监控脚本

echo "=== 因果学习训练监控 ==="

# 检查GPU使用情况
echo "GPU使用情况:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

# 检查训练进程
echo -e "\\n训练进程:"
ps aux | grep main_causal_finetune.py | grep -v grep

# 检查最新日志
echo -e "\\n最新训练日志:"
latest_log=$(ls -t logs/training_5centers_enhanced_*.log 2>/dev/null | head -1)
if [ -n "$latest_log" ]; then
    echo "日志文件: $latest_log"
    echo "文件大小: $(ls -lh $latest_log | awk '{print $5}')"
    echo "最后10行:"
    tail -10 $latest_log
else
    echo "未找到训练日志文件"
fi

# 检查因果分析结果
echo -e "\\n因果分析结果:"
if [ -d "causal_analysis" ]; then
    ls -la causal_analysis/
else
    echo "因果分析目录不存在"
fi

echo -e "\\n=== 监控完成 ==="
'''
    
    with open('monitor_causal_training.sh', 'w') as f:
        f.write(monitor_script)
    
    os.chmod('monitor_causal_training.sh', 0o755)
    print("✓ 训练监控脚本已创建: monitor_causal_training.sh")

if __name__ == "__main__":
    print("开始因果图可视化和项目分析...")
    
    # 1. 测试因果图可视化
    test_causal_graph_visualization()
    
    # 2. 创建项目结构图
    print("\n=== 创建项目结构图 ===")
    create_project_structure_diagram()
    
    # 3. 创建实验流程图
    print("\n=== 创建实验流程图 ===")
    create_experiment_workflow()
    
    # 4. 检查GPU训练
    print("\n=== 检查GPU训练 ===")
    gpu_available = ensure_gpu_training()
    
    # 5. 创建监控脚本
    print("\n=== 创建训练监控脚本 ===")
    create_training_monitor_script()
    
    print("\n=== 所有可视化完成 ===")
    print("生成的文件:")
    print("- causal_adjacency_matrix.png: 因果邻接矩阵")
    print("- causal_graph_network.png: 因果图网络")
    print("- project_structure.png: 项目结构图")
    print("- experiment_workflow.png: 实验流程图")
    print("- monitor_causal_training.sh: 训练监控脚本") 