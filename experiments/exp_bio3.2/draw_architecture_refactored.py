#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2 Refactored Architecture Visualization
重构后的模型架构图：A+B1合并、B2+B3合并、C1+C2合并
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

# 设置字体和样式（适合SCI论文）
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 300  # 高分辨率

def draw_refactored_architecture():
    """绘制重构后的Bio-COT 3.2架构图"""
    
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # ================== 定义颜色方案 ==================
    colors = {
        'input': '#E8F4F8',       # 浅蓝色 - 输入层
        'module1': '#E6F7E6',     # 浅绿色 - 模块1：特征编码+证据累积
        'module2': '#FFF4E6',     # 浅橙色 - 模块2：融合+验证
        'module3': '#FFE6E6',     # 浅红色 - 模块3：诊断+正则化
        'frozen': '#D0D0D0',      # 灰色 - 冻结模块
        'trainable': '#B8E6B8',   # 亮绿色 - 可训练模块
        'loss': '#FF6B6B'         # 红色 - 损失函数
    }
    
    # ================== 标题 ==================
    ax.text(10, 11.5, 'BioLCoT: Latent Chain-of-Thought Reasoning Engine', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    ax.text(10, 11.0, 'Reliability-Aware Clinical Decision Support', 
            ha='center', va='center', fontsize=12, style='italic', color='#555')
    
    # ================== 输入层 ==================
    y_input = 9.5
    
    # OCT/Colpo Images
    draw_module(ax, 1, y_input, 3.5, 0.8, 
                'OCT / Colpo Images\n(Center A/B/C...)', 
                colors['input'], edgecolor='#333', linewidth=1.5)
    
    # Clinical Text
    draw_module(ax, 5.5, y_input, 3.5, 0.8, 
                'Clinical Text\n(Center A/B/C...)', 
                colors['input'], edgecolor='#333', linewidth=1.5)
    
    # Medical Knowledge Base
    draw_module(ax, 10, y_input, 3.5, 0.8, 
                'Medical Knowledge Base\n(VLM Cache)', 
                colors['input'], edgecolor='#333', linewidth=1.5)
    
    # ================== 模块1：特征编码 + 证据累积 (A + B1) ==================
    y_module1 = 7.0
    module1_height = 1.8
    
    # 模块1外框
    module1_box = FancyBboxPatch(
        (0.5, y_module1 - module1_height/2), 19, module1_height,
        boxstyle="round,pad=0.2", 
        edgecolor='#2E7D32', facecolor='#F1F8E9', 
        linewidth=2.5, zorder=0, alpha=0.3
    )
    ax.add_patch(module1_box)
    ax.text(10, y_module1 + module1_height/2 - 0.1, 
            'Module 1: Multimodal Feature Encoding & Evidence Accumulation', 
            ha='center', va='top', fontsize=11, fontweight='bold', color='#2E7D32')
    
    # 视觉流
    y_vis = y_module1 - 0.3
    draw_module(ax, 1.5, y_vis, 3, 0.6, 
                'Vision Backbone\n(ViT/CNN)\n[Frozen]', 
                colors['frozen'], edgecolor='#555', linewidth=2)
    draw_arrow(ax, 2.5, y_input, 3, y_vis+0.3, style='solid', color='#333', linewidth=1.5)
    
    # Visual Adapter (B1的一部分)
    draw_module(ax, 5, y_vis, 2.5, 0.6, 
                'Visual Adapter\n[Trainable]\nLinear-GELU\nLayerNorm', 
                colors['trainable'], edgecolor='#2E7D32', linewidth=1.5)
    draw_arrow(ax, 4.5, y_vis+0.3, 5, y_vis+0.3, style='solid', color='#333', linewidth=1.5)
    
    # Memory Bank (B1的核心)
    draw_module(ax, 8, y_vis, 2.5, 0.6, 
                'Memory Bank (M)\n[Trainable]\nLearnable Slots\nRead/Write Heads', 
                colors['trainable'], edgecolor='#2E7D32', linewidth=1.5)
    draw_arrow(ax, 7.5, y_vis+0.3, 8, y_vis+0.3, style='solid', color='#333', linewidth=1.5)
    
    # 文本流
    y_text = y_module1 + 0.3
    draw_module(ax, 1.5, y_text, 3, 0.6, 
                'Text Encoder\n(BioBERT)\n[Frozen]', 
                colors['frozen'], edgecolor='#555', linewidth=2)
    draw_arrow(ax, 7, y_input, 3, y_text, style='solid', color='#333', linewidth=1.5)
    
    # Text Adapter
    draw_module(ax, 5, y_text, 2.5, 0.6, 
                'Text Adapter\n[Trainable]', 
                colors['trainable'], edgecolor='#2E7D32', linewidth=1.5)
    draw_arrow(ax, 4.5, y_text, 5, y_text, style='solid', color='#333', linewidth=1.5)
    
    # 知识检索
    draw_module(ax, 11.5, y_module1, 3, 0.6, 
                'Knowledge Retrieval\n[Trainable]\nVLM Retriever', 
                colors['trainable'], edgecolor='#2E7D32', linewidth=1.5)
    draw_arrow(ax, 11.75, y_input, 11.75, y_module1+0.3, style='solid', color='#333', linewidth=1.5)
    
    # 输出：Refined Visual Features (F_v)
    draw_module(ax, 15.5, y_module1, 2.5, 0.8, 
                'Refined Visual\nFeatures (F_v)', 
                '#FFE6B3', edgecolor='#333', linewidth=1.5)
    draw_arrow(ax, 10.5, y_vis+0.3, 15.5, y_module1, style='solid', color='#4CAF50', linewidth=2)
    
    # 输出：Text Prototypes (P_t)
    draw_module(ax, 15.5, y_text, 2.5, 0.6, 
                'Text Prototypes (P_t)', 
                '#E1BEE7', edgecolor='#333', linewidth=1.5)
    draw_arrow(ax, 7.5, y_text, 15.5, y_text, style='solid', color='#9C27B0', linewidth=1.5)
    
    # 输出：Knowledge Embeddings (K_e)
    draw_module(ax, 15.5, y_vis, 2.5, 0.6, 
                'Knowledge\nEmbeddings (K_e)', 
                '#BBDEFB', edgecolor='#333', linewidth=1.5)
    draw_arrow(ax, 14.5, y_module1, 15.5, y_vis+0.3, style='solid', color='#2196F3', linewidth=1.5)
    
    # ================== 模块2：自适应融合 + 假设验证 (B2 + B3) ==================
    y_module2 = 4.5
    module2_height = 1.8
    
    # 模块2外框
    module2_box = FancyBboxPatch(
        (0.5, y_module2 - module2_height/2), 19, module2_height,
        boxstyle="round,pad=0.2", 
        edgecolor='#F57C00', facecolor='#FFF3E0', 
        linewidth=2.5, zorder=0, alpha=0.3
    )
    ax.add_patch(module2_box)
    ax.text(10, y_module2 + module2_height/2 - 0.1, 
            'Module 2: Adaptive Reliability-Weighted Fusion & Hypothesis Verification', 
            ha='center', va='top', fontsize=11, fontweight='bold', color='#F57C00')
    
    # 输入连接
    draw_arrow(ax, 16.75, y_module1 - module1_height/2, 9, y_module2 + module2_height/2, 
              style='solid', color='#4CAF50', linewidth=2)
    draw_arrow(ax, 16.75, y_text, 11, y_module2 + module2_height/2, 
              style='solid', color='#9C27B0', linewidth=2)
    
    # Adaptive Gating (B2的一部分)
    y_gating = y_module2 + 0.3
    draw_module(ax, 1.5, y_gating, 3, 0.6, 
                'Adaptive Gating\n[Trainable]\nConcat-MLP-Sigmoid\nα = Gate(F_v, Z_noise)', 
                colors['trainable'], edgecolor='#F57C00', linewidth=1.5)
    
    # Z_noise采样
    draw_module(ax, 5.5, y_gating, 2, 0.6, 
                'Z_noise\n~N(0,1)', 
                '#E0E0E0', edgecolor='#333', linewidth=1)
    draw_arrow(ax, 6.5, y_gating+0.3, 4.5, y_gating+0.3, style='dashed', color='#666', linewidth=1.5)
    
    # Weighted Sum (B2)
    draw_module(ax, 8.5, y_gating, 2.5, 0.6, 
                'Weighted Sum\nF_cm = α·P_t + (1-α)·F_v', 
                colors['module2'], edgecolor='#F57C00', linewidth=1.5)
    draw_arrow(ax, 4.5, y_gating+0.3, 8.5, y_gating+0.3, style='solid', color='#333', linewidth=1.5)
    draw_arrow(ax, 16.75, y_module1 - module1_height/2, 9.75, y_gating+0.3, 
              style='solid', color='#4CAF50', linewidth=1.5)
    
    # OT Loss (B2)
    draw_loss_diamond(ax, 11.5, y_gating, 'OT Loss\nSinkhorn Dist', colors['loss'])
    draw_arrow(ax, 11, y_gating+0.3, 11.5, y_gating+0.3, style='dashed', color='#FF6B6B', linewidth=1.5)
    
    # Reasoning Unit (B3)
    y_reasoning = y_module2 - 0.3
    draw_module(ax, 1.5, y_reasoning, 3, 0.6, 
                'Reasoning Unit\n[Trainable]\nHidden State h_0', 
                colors['trainable'], edgecolor='#F57C00', linewidth=1.5)
    draw_arrow(ax, 11, y_gating, 3, y_reasoning+0.3, style='solid', color='#333', linewidth=1.5)
    
    # Iterative Reasoning Loop (B3)
    draw_module(ax, 5.5, y_reasoning, 4, 0.6, 
                'Iterative Reasoning Loop\n[Trainable]\nRefinement Steps (T=1...N)', 
                colors['trainable'], edgecolor='#F57C00', linewidth=1.5)
    draw_arrow(ax, 4.5, y_reasoning+0.3, 5.5, y_reasoning+0.3, style='solid', color='#333', linewidth=1.5)
    
    # 循环箭头（表示迭代）
    draw_arrow(ax, 7.5, y_reasoning, 7.5, y_reasoning+0.6, 
              style='solid', color='#F57C00', linewidth=1.5, alpha=0.6)
    draw_arrow(ax, 7.5, y_reasoning+0.6, 5.5, y_reasoning+0.6, 
              style='solid', color='#F57C00', linewidth=1.5, alpha=0.6)
    
    # 输出：Final Reasoning Feature (h_N)
    draw_module(ax, 11.5, y_reasoning, 2.5, 0.6, 
                'Final Reasoning\nFeature (h_N)', 
                '#FFE6B3', edgecolor='#333', linewidth=1.5)
    draw_arrow(ax, 9.5, y_reasoning+0.3, 11.5, y_reasoning+0.3, style='solid', color='#4CAF50', linewidth=2)
    
    # ================== 模块3：诊断推理 + 正则化 (C1 + C2) ==================
    y_module3 = 1.5
    module3_height = 1.5
    
    # 模块3外框
    module3_box = FancyBboxPatch(
        (0.5, y_module3 - module3_height/2), 19, module3_height,
        boxstyle="round,pad=0.2", 
        edgecolor='#C62828', facecolor='#FFEBEE', 
        linewidth=2.5, zorder=0, alpha=0.3
    )
    ax.add_patch(module3_box)
    ax.text(10, y_module3 + module3_height/2 - 0.1, 
            'Module 3: Joint Diagnostic Inference & Manifold Regularization', 
            ha='center', va='top', fontsize=11, fontweight='bold', color='#C62828')
    
    # 输入连接
    draw_arrow(ax, 12.75, y_module2 - module2_height/2, 9, y_module3 + module3_height/2, 
              style='solid', color='#4CAF50', linewidth=2)
    draw_arrow(ax, 16.75, y_vis, 15, y_module3 + module3_height/2, 
              style='solid', color='#2196F3', linewidth=2)
    
    # Global Pool (C1)
    y_pool = y_module3 + 0.2
    draw_module(ax, 1.5, y_pool, 2.5, 0.5, 
                'Global Pool', 
                colors['module3'], edgecolor='#C62828', linewidth=1.5)
    draw_arrow(ax, 12.75, y_module2 - module2_height/2, 2.75, y_pool+0.25, 
              style='solid', color='#333', linewidth=1.5)
    
    # Classifier (C1)
    draw_module(ax, 4.5, y_pool, 2.5, 0.5, 
                'Classifier\n[Trainable]\nDiagnosis Probability', 
                colors['trainable'], edgecolor='#C62828', linewidth=1.5)
    draw_arrow(ax, 4, y_pool+0.25, 4.5, y_pool+0.25, style='solid', color='#333', linewidth=1.5)
    
    # 诊断输出
    draw_module(ax, 7.5, y_pool, 2, 0.5, 
                'Diagnosis\nProbability', 
                '#FFE6CC', edgecolor='#FF9800', linewidth=2)
    draw_arrow(ax, 7, y_pool+0.25, 7.5, y_pool+0.25, style='solid', color='#FF9800', linewidth=2)
    
    # Manifold Regularization Branch (C2)
    y_align = y_module3 - 0.2
    draw_module(ax, 11, y_align, 2.5, 0.5, 
                'Linear Projection\n[Trainable]\nh_N → Shared Space', 
                colors['trainable'], edgecolor='#C62828', linewidth=1.5)
    draw_arrow(ax, 12.75, y_module2 - module2_height/2, 12.25, y_align+0.25, 
              style='solid', color='#333', linewidth=1.5)
    
    # Align Loss (C2)
    draw_loss_diamond(ax, 14.5, y_align, 'Align Loss\nContrastive', colors['loss'])
    draw_arrow(ax, 13.5, y_align+0.25, 14.5, y_align+0.25, style='dashed', color='#FF6B6B', linewidth=1.5)
    draw_arrow(ax, 16.75, y_vis, 15.5, y_align+0.25, style='dashed', color='#2196F3', linewidth=1.5)
    
    # ================== 图例 ==================
    legend_elements = [
        mpatches.Patch(facecolor=colors['frozen'], edgecolor='#555', linewidth=2, 
                      label='[Frozen] Components'),
        mpatches.Patch(facecolor=colors['trainable'], edgecolor='black', 
                      label='[Trainable] Components'),
        mpatches.Patch(facecolor=colors['module1'], edgecolor='#2E7D32', linewidth=2, 
                      label='Module 1: Encoding & Accumulation'),
        mpatches.Patch(facecolor=colors['module2'], edgecolor='#F57C00', linewidth=2, 
                      label='Module 2: Fusion & Verification'),
        mpatches.Patch(facecolor=colors['module3'], edgecolor='#C62828', linewidth=2, 
                      label='Module 3: Inference & Regularization'),
        FancyArrowPatch((0, 0), (0.5, 0), arrowstyle='->', lw=2, color='#333', 
                       label='→ Data Flow'),
        FancyArrowPatch((0, 0), (0.5, 0), arrowstyle='->', lw=1.5, color='#FF6B6B', 
                       linestyle='dashed', label='⋯ Gradient Flow / Loss'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', 
             bbox_to_anchor=(0.01, 0.99), fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('Bio_COT_3.2_Refactored_Architecture.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('Bio_COT_3.2_Refactored_Architecture.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✅ Refactored architecture diagram saved: Bio_COT_3.2_Refactored_Architecture.png & .pdf")
    plt.show()


def draw_module(ax, x, y, width, height, text, color, edgecolor='black', linewidth=1.5):
    """绘制模块框"""
    box = FancyBboxPatch((x, y), width, height, 
                         boxstyle="round,pad=0.1", 
                         edgecolor=edgecolor, facecolor=color, 
                         linewidth=linewidth, zorder=2)
    ax.add_patch(box)
    # 处理多行文本
    lines = text.split('\n')
    y_pos = y + height/2 + (len(lines)-1) * 0.08
    for i, line in enumerate(lines):
        ax.text(x + width/2, y_pos - i*0.16, line, 
               ha='center', va='center', fontsize=8, 
               fontweight='normal', wrap=True)


def draw_loss_diamond(ax, x, y, text, color):
    """绘制损失函数菱形"""
    diamond = mpatches.RegularPolygon((x, y), 4, radius=0.3, 
                                     orientation=np.pi/4,
                                     facecolor=color, 
                                     edgecolor='#D32F2F', 
                                     linewidth=2, zorder=3)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=7, 
           fontweight='bold', color='white')


def draw_arrow(ax, x1, y1, x2, y2, style='solid', color='black', alpha=1.0, linewidth=1.5):
    """绘制箭头"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                           arrowstyle='->', lw=linewidth, 
                           color=color, alpha=alpha, 
                           linestyle=style, zorder=3,
                           mutation_scale=20)
    ax.add_patch(arrow)


if __name__ == '__main__':
    print("🎨 Generating Bio-COT 3.2 Refactored Architecture Diagram...")
    print("=" * 60)
    
    draw_refactored_architecture()
    
    print("\n" + "=" * 60)
    print("✅ Refactored diagram generated successfully!")
    print("\nGenerated files:")
    print("  1. Bio_COT_3.2_Refactored_Architecture.png (High-res)")
    print("  2. Bio_COT_3.2_Refactored_Architecture.pdf (Vector format)")

