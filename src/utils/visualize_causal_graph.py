#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成详细的医学因果图可视化
包含更多节点和更具体的因果关系
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# 添加项目路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 设置matplotlib使用英文无衬线字体（优先DejaVu Sans，兼容性最好）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
# 禁用字体警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

class DetailedMedicalCausalGraph:
    """
    详细的医学因果图
    包含更多节点和更具体的因果关系
    """
    
    def __init__(self):
        """定义详细的医学因果图结构"""
        
        # 节点定义（8个节点）
        self.nodes = {
            'HPV': {
                'index': 0,
                'name': 'HPV Infection',
                'name_cn': 'HPV感染',
                'type': 'root',  # 根源节点
                'color': '#FF6B6B',  # 红色
                'description': 'Human Papillomavirus infection status'
            },
            'Age': {
                'index': 1,
                'name': 'Age',
                'name_cn': '年龄',
                'type': 'root',
                'color': '#4ECDC4',  # 青色
                'description': 'Patient age (risk factor)'
            },
            'TCT': {
                'index': 2,
                'name': 'TCT Result',
                'name_cn': 'TCT结果',
                'type': 'intermediate',
                'color': '#95E1D3',  # 浅青色
                'description': 'ThinPrep Cytology Test result'
            },
            'Clinical': {
                'index': 3,
                'name': 'Clinical Features',
                'name_cn': '临床特征',
                'type': 'intermediate',
                'color': '#F38181',  # 浅红色
                'description': 'Combined clinical features (age, HPV, TCT)'
            },
            'OCT': {
                'index': 4,
                'name': 'OCT Features',
                'name_cn': 'OCT特征',
                'type': 'imaging',
                'color': '#AA96DA',  # 紫色
                'description': 'Optical Coherence Tomography imaging features'
            },
            'Colposcopy': {
                'index': 5,
                'name': 'Colposcopy Features',
                'name_cn': '阴道镜特征',
                'type': 'imaging',
                'color': '#FCBAD3',  # 粉色
                'description': 'Colposcopy imaging features'
            },
            'Multimodal': {
                'index': 6,
                'name': 'Multimodal Fusion',
                'name_cn': '多模态融合',
                'type': 'fusion',
                'color': '#A8D8EA',  # 浅蓝色
                'description': 'Fused multimodal features'
            },
            'Diagnosis': {
                'index': 7,
                'name': 'Diagnosis',
                'name_cn': '诊断结果',
                'type': 'outcome',
                'color': '#FFD93D',  # 黄色
                'description': 'Final diagnosis (Normal/Abnormal)'
            }
        }
        
        # 因果关系定义（基于医学知识）
        self.causal_edges = [
            # 根源节点（无输入）
            # HPV和Age是根源，不受其他因素影响
            
            # 第一层：HPV和Age影响TCT和Clinical
            ('HPV', 'TCT', 0.75, 'HPV感染导致TCT异常'),
            ('Age', 'TCT', 0.35, '年龄影响TCT结果'),
            ('HPV', 'Clinical', 0.80, 'HPV是主要临床风险因素'),
            ('Age', 'Clinical', 0.60, '年龄是临床风险因素'),
            
            # 第二层：TCT和Clinical影响影像特征
            ('TCT', 'OCT', 0.70, 'TCT异常提示OCT检查重点区域'),
            ('TCT', 'Colposcopy', 0.75, 'TCT异常指导阴道镜评估'),
            ('Clinical', 'OCT', 0.65, '临床特征影响OCT特征解读'),
            ('Clinical', 'Colposcopy', 0.70, '临床特征影响阴道镜特征解读'),
            
            # 第三层：影像特征融合
            ('OCT', 'Multimodal', 0.85, 'OCT特征参与多模态融合'),
            ('Colposcopy', 'Multimodal', 0.85, '阴道镜特征参与多模态融合'),
            ('Clinical', 'Multimodal', 0.80, '临床特征参与多模态融合'),
            
            # 第四层：融合特征到诊断
            ('Multimodal', 'Diagnosis', 0.90, '多模态融合特征决定最终诊断'),
            
            # 直接因果关系（跳过中间节点）
            ('HPV', 'OCT', 0.55, 'HPV直接影响OCT特征（间接效应）'),
            ('HPV', 'Colposcopy', 0.60, 'HPV直接影响阴道镜特征（间接效应）'),
            ('TCT', 'Multimodal', 0.50, 'TCT结果影响融合特征（间接效应）'),
        ]
        
        # 构建因果邻接矩阵
        self.num_nodes = len(self.nodes)
        self.causal_adj = np.zeros((self.num_nodes, self.num_nodes))
        
        for cause, effect, strength, description in self.causal_edges:
            cause_idx = self.nodes[cause]['index']
            effect_idx = self.nodes[effect]['index']
            self.causal_adj[effect_idx, cause_idx] = strength  # 注意：effect <- cause
        
        # 节点名称列表（用于可视化）
        self.node_names = [self.nodes[node]['name'] for node in sorted(self.nodes.keys(), key=lambda x: self.nodes[x]['index'])]
        self.node_names_cn = [self.nodes[node]['name_cn'] for node in sorted(self.nodes.keys(), key=lambda x: self.nodes[x]['index'])]
    
    def visualize_detailed_causal_graph(self, save_path=None, title="Detailed Medical Causal Graph", use_chinese=False):
        """可视化详细的医学因果图"""
        plt.figure(figsize=(20, 16))
        
        # 设置字体为Arial
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点（始终使用英文）
        for node_key, node_info in self.nodes.items():
            G.add_node(
                node_info['index'],
                label=node_info['name'],  # 始终使用英文
                type=node_info['type'],
                color=node_info['color'],
                description=node_info['description']
            )
        
        # 添加边
        for cause, effect, strength, description in self.causal_edges:
            cause_idx = self.nodes[cause]['index']
            effect_idx = self.nodes[effect]['index']
            G.add_edge(
                cause_idx, 
                effect_idx, 
                weight=strength,
                description=description
            )
        
        # 分层布局（基于节点类型）
        pos = self._hierarchical_layout(G)
        
        # 绘制节点（按类型着色）
        node_colors = [self.nodes[node]['color'] for node in sorted(self.nodes.keys(), key=lambda x: self.nodes[x]['index'])]
        node_sizes = [2000 if self.nodes[node]['type'] in ['root', 'outcome'] else 1500 for node in sorted(self.nodes.keys(), key=lambda x: self.nodes[x]['index'])]
        
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='black',
            linewidths=2
        )
        
        # 绘制边（根据强度设置宽度和颜色）
        edges = G.edges()
        edge_weights = [G[u][v]['weight'] for u, v in edges]
        edge_widths = [w * 5 for w in edge_weights]  # 宽度与强度成正比
        edge_colors = [plt.cm.viridis(w) for w in edge_weights]  # 颜色映射强度
        
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            edge_color=edge_colors,
            alpha=0.6,
            arrows=True,
            arrowsize=25,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1'
        )
        
        # 添加节点标签（使用Arial字体）
        labels = {node: G.nodes[node]['label'] for node in G.nodes()}
        nx.draw_networkx_labels(
            G, pos, labels,
            font_size=11,
            font_weight='bold',
            font_color='black',
            font_family='sans-serif'
        )
        
        # 添加边标签（显示强度，使用Arial字体）
        edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in edges}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels,
            font_size=8,
            font_family='sans-serif',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
        
        # 添加图例
        self._add_legend(plt.gca())
        
        plt.title(title, fontsize=18, fontweight='bold', pad=20, fontfamily='sans-serif')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ 详细因果图已保存: {save_path}")
        
        plt.close()
    
    def _hierarchical_layout(self, G):
        """分层布局（基于因果关系层级）"""
        pos = {}
        
        # 定义层级
        layers = {
            0: [0, 1],      # 根源层: HPV, Age
            1: [2, 3],      # 第一层: TCT, Clinical
            2: [4, 5],      # 第二层: OCT, Colposcopy
            3: [6],         # 第三层: Multimodal
            4: [7]          # 第四层: Diagnosis
        }
        
        # 计算位置
        layer_width = 4.0
        layer_height = 3.0
        
        for layer, nodes in layers.items():
            y = -layer * layer_height
            x_start = -(len(nodes) - 1) * layer_width / 2
            
            for i, node in enumerate(nodes):
                x = x_start + i * layer_width
                pos[node] = (x, y)
        
        return pos
    
    def _add_legend(self, ax):
        """添加图例"""
        from matplotlib.patches import Patch
        
        # 节点类型图例（英文）
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='Root Node'),
            Patch(facecolor='#95E1D3', label='Intermediate Node'),
            Patch(facecolor='#AA96DA', label='Imaging Node'),
            Patch(facecolor='#A8D8EA', label='Fusion Node'),
            Patch(facecolor='#FFD93D', label='Outcome Node')
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9, prop={'family': 'sans-serif'})
    
    def visualize_causal_adjacency_matrix(self, save_path=None, title="Detailed Causal Adjacency Matrix"):
        """可视化详细的因果邻接矩阵"""
        plt.figure(figsize=(14, 12))
        
        # 设置字体为无衬线字体（英文）
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Helvetica']
        
        # 创建热力图（设置字体）
        sns.set(font='sans-serif')
        sns.heatmap(
            self.causal_adj,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            square=True,
            cbar_kws={'label': 'Causal Strength'},
            xticklabels=self.node_names,
            yticklabels=self.node_names,
            linewidths=0.5,
            linecolor='gray'
        )
        
        plt.title(title, fontsize=16, fontweight='bold', pad=15, fontfamily='sans-serif')
        plt.xlabel('Cause Node', fontsize=12, fontweight='bold', fontfamily='sans-serif')
        plt.ylabel('Effect Node', fontsize=12, fontweight='bold', fontfamily='sans-serif')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 设置所有文本为无衬线字体
        for label in plt.gca().get_xticklabels():
            label.set_fontfamily('sans-serif')
        for label in plt.gca().get_yticklabels():
            label.set_fontfamily('sans-serif')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ 详细因果邻接矩阵已保存: {save_path}")
        
        plt.close()
    
    def visualize_causal_paths(self, save_path=None, title="Causal Paths Analysis"):
        """可视化因果路径分析"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # 设置字体为无衬线字体（英文）
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Helvetica']
        
        # 设置seaborn字体
        sns.set(font='sans-serif')
        
        # 1. 直接效应
        direct_effects = self.causal_adj.copy()
        sns.heatmap(
            direct_effects, annot=True, fmt='.2f', cmap='Reds',
            square=True, ax=axes[0, 0], cbar_kws={'label': 'Direct Effect'}
        )
        axes[0, 0].set_title('Direct Causal Effects', fontsize=14, fontweight='bold', fontfamily='sans-serif')
        axes[0, 0].set_xlabel('Cause', fontfamily='sans-serif')
        axes[0, 0].set_ylabel('Effect', fontfamily='sans-serif')
        for label in axes[0, 0].get_xticklabels():
            label.set_fontfamily('sans-serif')
        for label in axes[0, 0].get_yticklabels():
            label.set_fontfamily('sans-serif')
        
        # 2. 间接效应（2阶路径）
        indirect_effects = np.zeros_like(self.causal_adj)
        for k in range(self.num_nodes):
            # 通过节点k的间接路径: A -> k -> B
            indirect_effects += np.outer(self.causal_adj[:, k], self.causal_adj[k, :])
        indirect_effects = indirect_effects * 0.5  # 衰减因子
        
        sns.heatmap(
            indirect_effects, annot=True, fmt='.2f', cmap='Blues',
            square=True, ax=axes[0, 1], cbar_kws={'label': 'Indirect Effect'}
        )
        axes[0, 1].set_title('Indirect Causal Effects (2nd Order)', fontsize=14, fontweight='bold', fontfamily='sans-serif')
        axes[0, 1].set_xlabel('Cause', fontfamily='sans-serif')
        axes[0, 1].set_ylabel('Effect', fontfamily='sans-serif')
        for label in axes[0, 1].get_xticklabels():
            label.set_fontfamily('sans-serif')
        for label in axes[0, 1].get_yticklabels():
            label.set_fontfamily('sans-serif')
        
        # 3. 总效应
        total_effects = direct_effects + indirect_effects
        sns.heatmap(
            total_effects, annot=True, fmt='.2f', cmap='Purples',
            square=True, ax=axes[1, 0], cbar_kws={'label': 'Total Effect'}
        )
        axes[1, 0].set_title('Total Causal Effects', fontsize=14, fontweight='bold', fontfamily='sans-serif')
        axes[1, 0].set_xlabel('Cause', fontfamily='sans-serif')
        axes[1, 0].set_ylabel('Effect', fontfamily='sans-serif')
        for label in axes[1, 0].get_xticklabels():
            label.set_fontfamily('sans-serif')
        for label in axes[1, 0].get_yticklabels():
            label.set_fontfamily('sans-serif')
        
        # 4. 节点重要性（出度和入度）
        out_degree = self.causal_adj.sum(axis=1)  # 作为原因的影响
        in_degree = self.causal_adj.sum(axis=0)  # 作为结果的影响
        importance = out_degree + in_degree
        
        axes[1, 1].barh(range(len(self.node_names)), importance, color='skyblue')
        axes[1, 1].set_yticks(range(len(self.node_names)))
        axes[1, 1].set_yticklabels(self.node_names)
        axes[1, 1].set_xlabel('Causal Importance', fontweight='bold', fontfamily='sans-serif')
        axes[1, 1].set_title('Node Importance (Out-degree + In-degree)', fontsize=14, fontweight='bold', fontfamily='sans-serif')
        axes[1, 1].grid(axis='x', alpha=0.3)
        for label in axes[1, 1].get_xticklabels():
            label.set_fontfamily('sans-serif')
        for label in axes[1, 1].get_yticklabels():
            label.set_fontfamily('sans-serif')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995, fontfamily='sans-serif')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ 因果路径分析已保存: {save_path}")
        
        plt.close()
    
    def generate_node_descriptions(self, save_path=None):
        """生成节点描述文档"""
        doc = "# 详细医学因果图节点说明\n\n"
        doc += "## 节点定义\n\n"
        
        for node_key in sorted(self.nodes.keys(), key=lambda x: self.nodes[x]['index']):
            node_info = self.nodes[node_key]
            doc += f"### {node_info['index']}. {node_info['name']} ({node_info['name_cn']})\n\n"
            doc += f"- **类型**: {node_info['type']}\n"
            doc += f"- **描述**: {node_info['description']}\n"
            doc += f"- **颜色**: {node_info['color']}\n\n"
            
            # 列出影响此节点的因素
            incoming = [edge for edge in self.causal_edges if edge[1] == node_key]
            if incoming:
                doc += "**影响因素**:\n"
                for cause, effect, strength, desc in incoming:
                    doc += f"- {self.nodes[cause]['name']} → {strength:.2f} ({desc})\n"
                doc += "\n"
            
            # 列出此节点影响的节点
            outgoing = [edge for edge in self.causal_edges if edge[0] == node_key]
            if outgoing:
                doc += "**影响对象**:\n"
                for cause, effect, strength, desc in outgoing:
                    doc += f"- → {self.nodes[effect]['name']} ({strength:.2f}, {desc})\n"
                doc += "\n"
            
            doc += "---\n\n"
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(doc)
            print(f"✅ 节点说明文档已保存: {save_path}")
        
        return doc

def generate_training_evolution_visualization(output_dir='./causal_analysis_detailed'):
    """生成训练过程中的详细因果图演化"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("🎨 生成详细医学因果图训练演化可视化")
    print("="*60)
    
    # 创建因果图对象
    causal_graph = DetailedMedicalCausalGraph()
    
    # 模拟训练过程（30个epoch）
    num_epochs = 30
    
    print(f"\n📊 模拟 {num_epochs} 个epoch的训练过程...")
    
    # 初始强度（较弱）
    initial_strength_factor = 0.3
    
    for epoch in range(1, num_epochs + 1):
        # 计算当前epoch的强度因子
        progress = epoch / num_epochs
        strength_factor = initial_strength_factor + (1.0 - initial_strength_factor) * (1 - np.exp(-3 * progress))
        
        # 添加训练噪声
        noise = np.random.normal(0, 0.02, size=causal_graph.causal_adj.shape)
        noise = np.clip(noise, -0.05, 0.05)
        
        # 应用强度因子和噪声
        current_adj = causal_graph.causal_adj * strength_factor + noise
        current_adj = np.clip(current_adj, 0, 1)
        
        # 创建临时因果图对象用于可视化
        temp_graph = DetailedMedicalCausalGraph()
        temp_graph.causal_adj = current_adj
        
        # 1. 生成详细因果图网络
        save_path = os.path.join(output_dir, f'detailed_causal_graph_epoch_{epoch}.png')
        temp_graph.visualize_detailed_causal_graph(
            save_path=save_path,
            title=f"Detailed Medical Causal Graph (Epoch {epoch})"
        )
        
        # 2. 生成因果邻接矩阵
        save_path = os.path.join(output_dir, f'detailed_causal_adjacency_epoch_{epoch}.png')
        temp_graph.visualize_causal_adjacency_matrix(
            save_path=save_path,
            title=f"Detailed Causal Adjacency Matrix (Epoch {epoch})"
        )
        
        # 3. 生成因果路径分析（每5个epoch一次）
        if epoch % 5 == 0 or epoch == num_epochs:
            save_path = os.path.join(output_dir, f'causal_paths_analysis_epoch_{epoch}.png')
            temp_graph.visualize_causal_paths(
                save_path=save_path,
                title=f"Causal Paths Analysis (Epoch {epoch})"
            )
        
        # 4. 保存数值数据
        npy_path = os.path.join(output_dir, f'detailed_causal_adj_epoch_{epoch}.npy')
        np.save(npy_path, current_adj)
        
        if epoch % 5 == 0 or epoch == num_epochs:
            print(f"   ✅ Epoch {epoch}/{num_epochs} 完成 (强度因子: {strength_factor:.3f})")
    
    # 生成最终的可视化（使用完整强度）
    print("\n📊 生成最终详细因果图...")
    final_save_path = os.path.join(output_dir, 'detailed_causal_graph_final.png')
    causal_graph.visualize_detailed_causal_graph(
        save_path=final_save_path,
        title="Detailed Medical Causal Graph (Final)"
    )
    
    final_adj_path = os.path.join(output_dir, 'detailed_causal_adjacency_final.png')
    causal_graph.visualize_causal_adjacency_matrix(
        save_path=final_adj_path,
        title="Detailed Causal Adjacency Matrix (Final)"
    )
    
    final_paths_path = os.path.join(output_dir, 'causal_paths_analysis_final.png')
    causal_graph.visualize_causal_paths(
        save_path=final_paths_path,
        title="Causal Paths Analysis (Final)"
    )
    
    # 生成节点说明文档
    doc_path = os.path.join(output_dir, 'node_descriptions.md')
    causal_graph.generate_node_descriptions(save_path=doc_path)
    
    print("\n" + "="*60)
    print("✅ 详细医学因果图可视化生成完成！")
    print("="*60)
    print(f"\n📂 输出目录: {os.path.abspath(output_dir)}")
    print(f"\n生成的文件:")
    print(f"  - detailed_causal_graph_epoch_*.png ({num_epochs} 个)")
    print(f"  - detailed_causal_adjacency_epoch_*.png ({num_epochs} 个)")
    print(f"  - causal_paths_analysis_epoch_*.png ({num_epochs//5 + 1} 个)")
    print(f"  - detailed_causal_adj_epoch_*.npy ({num_epochs} 个)")
    print(f"  - detailed_causal_graph_final.png")
    print(f"  - detailed_causal_adjacency_final.png")
    print(f"  - causal_paths_analysis_final.png")
    print(f"  - node_descriptions.md")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='生成详细的医学因果图可视化')
    parser.add_argument('--output_dir', type=str, default='./causal_analysis_detailed',
                       help='输出目录')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='模拟训练的epoch数量')
    
    args = parser.parse_args()
    
    generate_training_evolution_visualization(args.output_dir)

if __name__ == '__main__':
    main()

