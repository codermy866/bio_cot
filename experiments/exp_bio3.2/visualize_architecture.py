#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2 模型架构可视化
绘制详细的模型架构图，包含所有模块和细节
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch, Rectangle, Circle, Ellipse
from matplotlib.patches import Arrow, FancyArrow
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def draw_bio_cot_v3_2_architecture():
    """
    绘制Bio-COT 3.2的详细架构图
    """
    fig = plt.figure(figsize=(24, 32))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 32)
    ax.axis('off')
    
    # 定义颜色方案
    colors = {
        'input': '#E8F4F8',      # 浅蓝色 - 输入
        'vlm': '#FFE5B4',        # 浅橙色 - VLM模块（4.0优势）
        'visual_notes': '#E6F3FF', # 浅蓝色 - Visual Notes（3.1优势）
        'adaptive': '#E8F5E9',    # 浅绿色 - 自适应模块（3.1优势）
        'dual_head': '#FFF3E0',   # 浅橙色 - 双头编码器
        'alignment': '#F3E5F5',   # 浅紫色 - 对齐模块（3.1优势）
        'fusion': '#E0F2F1',      # 浅青色 - 融合模块
        'classifier': '#FFEBEE',  # 浅红色 - 分类器
        'loss': '#FCE4EC',        # 浅粉色 - 损失函数
        'arrow': '#424242',       # 深灰色 - 箭头
        'text': '#212121'         # 深灰色 - 文本
    }
    
    # ==================== 输入层 ====================
    y_start = 30.5
    
    # OCT图像特征
    oct_box = FancyBboxPatch((0.5, y_start-0.8), 3.5, 0.6, 
                             boxstyle="round,pad=0.05", 
                             facecolor=colors['input'], 
                             edgecolor='#1976D2', linewidth=2)
    ax.add_patch(oct_box)
    ax.text(2.25, y_start-0.5, 'OCT图像特征\n[B, N, 768]', 
            ha='center', va='center', fontsize=10, weight='bold', color=colors['text'])
    
    # Colposcopy图像特征
    colpo_box = FancyBboxPatch((4.5, y_start-0.8), 3.5, 0.6,
                               boxstyle="round,pad=0.05",
                               facecolor=colors['input'],
                               edgecolor='#1976D2', linewidth=2)
    ax.add_patch(colpo_box)
    ax.text(6.25, y_start-0.5, 'Colposcopy图像特征\n[B, N, 768]',
            ha='center', va='center', fontsize=10, weight='bold', color=colors['text'])
    
    # 图像文件名和临床信息
    meta_box = FancyBboxPatch((8.5, y_start-0.8), 3.5, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor=colors['input'],
                              edgecolor='#1976D2', linewidth=2)
    ax.add_patch(meta_box)
    ax.text(10.25, y_start-0.5, '图像文件名\n临床信息',
            ha='center', va='center', fontsize=10, weight='bold', color=colors['text'])
    
    # VLM缓存
    vlm_cache_box = FancyBboxPatch((12.5, y_start-0.8), 4.5, 0.6,
                                    boxstyle="round,pad=0.05",
                                    facecolor=colors['vlm'],
                                    edgecolor='#FF6F00', linewidth=2)
    ax.add_patch(vlm_cache_box)
    ax.text(14.75, y_start-0.5, 'VLM缓存\n(102,705个描述)',
            ha='center', va='center', fontsize=9, weight='bold', color=colors['text'])
    
    y_current = y_start - 1.5
    
    # ==================== Step 1: VLM增强知识检索器（4.0优势）====================
    y_vlm = y_current - 1.0
    
    # VLM检索器主框
    vlm_main_box = FancyBboxPatch((2, y_vlm-2.5), 20, 2.5,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['vlm'],
                                   edgecolor='#FF6F00', linewidth=3)
    ax.add_patch(vlm_main_box)
    ax.text(12, y_vlm-0.3, 'Step 1: VLM增强知识检索器 (4.0优势)', 
            ha='center', va='center', fontsize=14, weight='bold', color='#E65100')
    
    # Frozen Text Encoder
    frozen_box = FancyBboxPatch((3, y_vlm-1.8), 5.5, 1.0,
                                 boxstyle="round,pad=0.05",
                                 facecolor='#FFF9C4',
                                 edgecolor='#F57F17', linewidth=2)
    ax.add_patch(frozen_box)
    ax.text(5.75, y_vlm-1.3, '❄️ Frozen Text Encoder\nPubMedBERT (768维)', 
            ha='center', va='center', fontsize=9, color=colors['text'])
    
    # VLM描述检索
    vlm_desc_box = FancyBboxPatch((9, y_vlm-1.8), 4, 1.0,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#E1F5FE',
                                   edgecolor='#0277BD', linewidth=2)
    ax.add_patch(vlm_desc_box)
    ax.text(11, y_vlm-1.3, 'VLM描述检索\n(基于图像文件名)',
            ha='center', va='center', fontsize=9, color=colors['text'])
    
    # Trainable Adapter
    adapter_box = FancyBboxPatch((13.5, y_vlm-1.8), 5.5, 1.0,
                                  boxstyle="round,pad=0.05",
                                  facecolor='#FFE0B2',
                                  edgecolor='#E65100', linewidth=2)
    ax.add_patch(adapter_box)
    ax.text(16.25, y_vlm-1.3, '🔥 Trainable Adapter\nLinear→LayerNorm→ReLU→Dropout→Linear\n(768→768→768)',
            ha='center', va='center', fontsize=8, color=colors['text'])
    
    # 输出：z_sem
    z_sem_box = FancyBboxPatch((19.5, y_vlm-1.8), 2, 1.0,
                                boxstyle="round,pad=0.05",
                                facecolor='#C8E6C9',
                                edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(z_sem_box)
    ax.text(20.5, y_vlm-1.3, 'z_sem\n[B, 768]',
            ha='center', va='center', fontsize=9, weight='bold', color=colors['text'])
    
    # 箭头：输入到VLM
    arrow1 = FancyArrowPatch((10.25, y_start-1.4), (11, y_vlm-0.5),
                             arrowstyle='->', lw=2, color=colors['arrow'], 
                             connectionstyle="arc3,rad=0.2")
    ax.add_patch(arrow1)
    
    arrow2 = FancyArrowPatch((14.75, y_start-1.4), (11, y_vlm-0.5),
                             arrowstyle='->', lw=2, color=colors['arrow'],
                             connectionstyle="arc3,rad=0.2")
    ax.add_patch(arrow2)
    
    y_current = y_vlm - 3.0
    
    # ==================== Step 2: Visual Notes模块（3.1优势）====================
    y_visual = y_current - 1.0
    
    # Visual Notes主框
    visual_main_box = FancyBboxPatch((0.5, y_visual-2.5), 10.5, 2.5,
                                      boxstyle="round,pad=0.1",
                                      facecolor=colors['visual_notes'],
                                      edgecolor='#1976D2', linewidth=3)
    ax.add_patch(visual_main_box)
    ax.text(5.75, y_visual-0.3, 'Step 2: 增强型Visual Notes (3.1优势)', 
            ha='center', va='center', fontsize=14, weight='bold', color='#0D47A1')
    
    # OCT Visual Notes
    oct_visual_box = FancyBboxPatch((1, y_visual-1.8), 4.5, 1.8,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#BBDEFB',
                                      edgecolor='#1976D2', linewidth=2)
    ax.add_patch(oct_visual_box)
    ax.text(3.25, y_visual-0.9, 'OCT Visual Notes\nCross-Attention:\nText(Query) → Image(Key/Value)\nGating + Feature Refine',
            ha='center', va='center', fontsize=8, color=colors['text'])
    
    # Colpo Visual Notes
    colpo_visual_box = FancyBboxPatch((6, y_visual-1.8), 4.5, 1.8,
                                       boxstyle="round,pad=0.05",
                                       facecolor='#BBDEFB',
                                       edgecolor='#1976D2', linewidth=2)
    ax.add_patch(colpo_visual_box)
    ax.text(8.25, y_visual-0.9, 'Colposcopy Visual Notes\nCross-Attention:\nText(Query) → Image(Key/Value)\nGating + Feature Refine',
            ha='center', va='center', fontsize=8, color=colors['text'])
    
    # 输出：f_oct_pooled, f_colpo_pooled
    oct_pooled_box = FancyBboxPatch((1, y_visual-2.3), 4.5, 0.4,
                                     boxstyle="round,pad=0.03",
                                     facecolor='#90CAF9',
                                     edgecolor='#1565C0', linewidth=1.5)
    ax.add_patch(oct_pooled_box)
    ax.text(3.25, y_visual-2.1, 'f_oct_pooled [B, 768]', 
            ha='center', va='center', fontsize=8, weight='bold', color=colors['text'])
    
    colpo_pooled_box = FancyBboxPatch((6, y_visual-2.3), 4.5, 0.4,
                                       boxstyle="round,pad=0.03",
                                       facecolor='#90CAF9',
                                       edgecolor='#1565C0', linewidth=1.5)
    ax.add_patch(colpo_pooled_box)
    ax.text(8.25, y_visual-2.1, 'f_colpo_pooled [B, 768]',
            ha='center', va='center', fontsize=8, weight='bold', color=colors['text'])
    
    # 箭头：OCT和Colpo到Visual Notes
    arrow3 = FancyArrowPatch((2.25, y_start-1.4), (3.25, y_visual-0.5),
                             arrowstyle='->', lw=2, color=colors['arrow'],
                             connectionstyle="arc3,rad=0.3")
    ax.add_patch(arrow3)
    
    arrow4 = FancyArrowPatch((6.25, y_start-1.4), (8.25, y_visual-0.5),
                             arrowstyle='->', lw=2, color=colors['arrow'],
                             connectionstyle="arc3,rad=0.3")
    ax.add_patch(arrow4)
    
    # 箭头：z_sem到Visual Notes
    arrow5 = FancyArrowPatch((20.5, y_vlm-1.3), (5.75, y_visual-0.5),
                             arrowstyle='->', lw=2, color='#4CAF50', linestyle='--',
                             connectionstyle="arc3,rad=-0.4")
    ax.add_patch(arrow5)
    
    y_current = y_visual - 3.0
    
    # ==================== Step 3: 自适应模态门控（3.1优势）====================
    y_adaptive = y_current - 0.8
    
    adaptive_box = FancyBboxPatch((11.5, y_adaptive-1.5), 11, 1.5,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['adaptive'],
                                   edgecolor='#388E3C', linewidth=3)
    ax.add_patch(adaptive_box)
    ax.text(17, y_adaptive-0.2, 'Step 3: 自适应模态门控 (3.1优势)', 
            ha='center', va='center', fontsize=14, weight='bold', color='#1B5E20')
    
    # 内部结构
    adaptive_inner = FancyBboxPatch((12.5, y_adaptive-1.2), 9, 0.9,
                                     boxstyle="round,pad=0.05",
                                     facecolor='#C8E6C9',
                                     edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(adaptive_inner)
    ax.text(17, y_adaptive-0.75, 'Concat → Linear(1536→384) → LayerNorm → ReLU → Linear(384→2) → Softmax',
            ha='center', va='center', fontsize=8, color=colors['text'])
    ax.text(17, y_adaptive-1.0, '动态权重: w_oct, w_colpo → 加权融合',
            ha='center', va='center', fontsize=9, weight='bold', color=colors['text'])
    
    # 输出：f_fused
    fused_box = FancyBboxPatch((19.5, y_adaptive-1.2), 2, 0.9,
                                boxstyle="round,pad=0.05",
                                facecolor='#81C784',
                                edgecolor='#1B5E20', linewidth=2)
    ax.add_patch(fused_box)
    ax.text(20.5, y_adaptive-0.75, 'f_fused\n[B, 768]',
            ha='center', va='center', fontsize=9, weight='bold', color=colors['text'])
    
    # 箭头：Visual Notes到Adaptive Gating
    arrow6 = FancyArrowPatch((3.25, y_visual-2.1), (15, y_adaptive-0.75),
                             arrowstyle='->', lw=2, color=colors['arrow'],
                             connectionstyle="arc3,rad=0.2")
    ax.add_patch(arrow6)
    
    arrow7 = FancyArrowPatch((8.25, y_visual-2.1), (17, y_adaptive-0.75),
                             arrowstyle='->', lw=2, color=colors['arrow'],
                             connectionstyle="arc3,rad=0.2")
    ax.add_patch(arrow7)
    
    y_current = y_adaptive - 2.0
    
    # ==================== Step 4: 双头因果编码器 ====================
    y_dual = y_current - 0.8
    
    dual_box = FancyBboxPatch((0.5, y_dual-2.0), 11, 2.0,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['dual_head'],
                               edgecolor='#F57C00', linewidth=3)
    ax.add_patch(dual_box)
    ax.text(6, y_dual-0.2, 'Step 4: 双头因果编码器', 
            ha='center', va='center', fontsize=14, weight='bold', color='#E65100')
    
    # Feature Projection
    proj_box = FancyBboxPatch((1, y_dual-1.5), 4.5, 0.8,
                               boxstyle="round,pad=0.05",
                               facecolor='#FFE0B2',
                               edgecolor='#E65100', linewidth=2)
    ax.add_patch(proj_box)
    ax.text(3.25, y_dual-1.1, 'Feature Projection\nLinear(768→1536) → LayerNorm\n→ GELU → Dropout → Linear(1536→768)',
            ha='center', va='center', fontsize=8, color=colors['text'])
    
    # Causal Head
    causal_box = FancyBboxPatch((6, y_dual-1.5), 4.5, 0.8,
                                 boxstyle="round,pad=0.05",
                                 facecolor='#C5E1A5',
                                 edgecolor='#558B2F', linewidth=2)
    ax.add_patch(causal_box)
    ax.text(8.25, y_dual-1.1, 'Causal Head\nLinear → LayerNorm → GELU\n→ Dropout → Linear',
            ha='center', va='center', fontsize=8, color=colors['text'])
    
    # Noise Head
    noise_box = FancyBboxPatch((1, y_dual-2.3), 4.5, 0.7,
                                boxstyle="round,pad=0.05",
                                facecolor='#FFCCBC',
                                edgecolor='#D84315', linewidth=2)
    ax.add_patch(noise_box)
    ax.text(3.25, y_dual-1.95, 'Noise Head\nLinear → LayerNorm → GELU\n→ Dropout → Linear',
            ha='center', va='center', fontsize=8, color=colors['text'])
    
    # 输出
    z_causal_box = FancyBboxPatch((6, y_dual-2.3), 4.5, 0.7,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#A5D6A7',
                                   edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(z_causal_box)
    ax.text(8.25, y_dual-1.95, 'z_causal [B, 768]',
            ha='center', va='center', fontsize=9, weight='bold', color=colors['text'])
    
    z_noise_box = FancyBboxPatch((11.5, y_dual-2.3), 4.5, 0.7,
                                  boxstyle="round,pad=0.05",
                                  facecolor='#FFAB91',
                                  edgecolor='#BF360C', linewidth=2)
    ax.add_patch(z_noise_box)
    ax.text(13.75, y_dual-1.95, 'z_noise [B, 768]',
            ha='center', va='center', fontsize=9, weight='bold', color=colors['text'])
    
    # 箭头：Adaptive Gating到Dual Head
    arrow8 = FancyArrowPatch((20.5, y_adaptive-0.75), (3.25, y_dual-1.1),
                             arrowstyle='->', lw=2, color=colors['arrow'],
                             connectionstyle="arc3,rad=0.2")
    ax.add_patch(arrow8)
    
    y_current = y_dual - 2.5
    
    # ==================== Step 5: 深度对齐模块（3.1优势）====================
    y_align = y_current - 1.0
    
    align_main_box = FancyBboxPatch((16, y_align-3.5), 7.5, 3.5,
                                     boxstyle="round,pad=0.1",
                                     facecolor=colors['alignment'],
                                     edgecolor='#7B1FA2', linewidth=3)
    ax.add_patch(align_main_box)
    ax.text(19.75, y_align-0.3, 'Step 6: 深度对齐模块 (3.1优势)', 
            ha='center', va='center', fontsize=14, weight='bold', color='#4A148C')
    
    # Image投影头
    img_proj_box = FancyBboxPatch((16.5, y_align-1.8), 3, 1.3,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#E1BEE7',
                                   edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(img_proj_box)
    ax.text(18, y_align-1.15, 'Image投影头\nLinear(768→768)\nLayerNorm → GELU\nDropout → Linear(768→256)\nLayerNorm',
            ha='center', va='center', fontsize=7, color=colors['text'])
    
    # Text投影头
    txt_proj_box = FancyBboxPatch((20, y_align-1.8), 3, 1.3,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#E1BEE7',
                                   edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(txt_proj_box)
    ax.text(21.5, y_align-1.15, 'Text投影头\nLinear(768→768)\nLayerNorm → GELU\nDropout → Linear(768→256)\nLayerNorm',
            ha='center', va='center', fontsize=7, color=colors['text'])
    
    # 共享语义空间投影
    shared_box = FancyBboxPatch((16.5, y_align-2.8), 6.5, 0.8,
                                 boxstyle="round,pad=0.05",
                                 facecolor='#CE93D8',
                                 edgecolor='#6A1B9A', linewidth=2)
    ax.add_patch(shared_box)
    ax.text(19.75, y_align-2.4, '共享语义空间投影: Linear(256→256) → LayerNorm → GELU',
            ha='center', va='center', fontsize=8, color=colors['text'])
    
    # 归一化和温度
    norm_box = FancyBboxPatch((16.5, y_align-3.3), 6.5, 0.4,
                               boxstyle="round,pad=0.03",
                               facecolor='#BA68C8',
                               edgecolor='#4A148C', linewidth=1.5)
    ax.add_patch(norm_box)
    ax.text(19.75, y_align-3.1, 'L2归一化 + 温度系数(τ) → InfoNCE Loss + L2 Loss',
            ha='center', va='center', fontsize=8, weight='bold', color=colors['text'])
    
    # 箭头：z_causal和z_sem到对齐模块
    arrow9 = FancyArrowPatch((8.25, y_dual-1.95), (18, y_align-1.15),
                             arrowstyle='->', lw=2, color='#4CAF50', linestyle='--',
                             connectionstyle="arc3,rad=-0.3")
    ax.add_patch(arrow9)
    
    arrow10 = FancyArrowPatch((20.5, y_vlm-1.3), (21.5, y_align-1.15),
                              arrowstyle='->', lw=2, color='#4CAF50', linestyle='--',
                              connectionstyle="arc3,rad=-0.5")
    ax.add_patch(arrow10)
    
    y_current = y_align - 4.0
    
    # ==================== Step 6: 跨模态融合（3.1优势）====================
    y_fusion = y_current - 0.8
    
    fusion_box = FancyBboxPatch((0.5, y_fusion-1.5), 11, 1.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['fusion'],
                                 edgecolor='#00695C', linewidth=3)
    ax.add_patch(fusion_box)
    ax.text(6, y_fusion-0.2, 'Step 5: 跨模态融合 (3.1优势)', 
            ha='center', va='center', fontsize=14, weight='bold', color='#004D40')
    
    # MultiheadAttention
    attn_box = FancyBboxPatch((1, y_fusion-1.2), 9, 0.9,
                               boxstyle="round,pad=0.05",
                               facecolor='#B2DFDB',
                               edgecolor='#00695C', linewidth=2)
    ax.add_patch(attn_box)
    ax.text(5.5, y_fusion-0.75, 'MultiheadAttention (4 heads)\nQ=z_causal, K=V=z_sem → Residual Connection',
            ha='center', va='center', fontsize=9, color=colors['text'])
    
    # 输出：f_final
    final_box = FancyBboxPatch((10.5, y_fusion-1.2), 2, 0.9,
                                boxstyle="round,pad=0.05",
                                facecolor='#4DB6AC',
                                edgecolor='#004D40', linewidth=2)
    ax.add_patch(final_box)
    ax.text(11.5, y_fusion-0.75, 'f_final\n[B, 768]',
            ha='center', va='center', fontsize=9, weight='bold', color=colors['text'])
    
    # 箭头：z_causal和z_sem到Fusion
    arrow11 = FancyArrowPatch((8.25, y_dual-1.95), (5.5, y_fusion-0.75),
                             arrowstyle='->', lw=2, color=colors['arrow'],
                             connectionstyle="arc3,rad=0.2")
    ax.add_patch(arrow11)
    
    arrow12 = FancyArrowPatch((20.5, y_vlm-1.3), (5.5, y_fusion-0.75),
                             arrowstyle='->', lw=2, color='#4CAF50', linestyle='--',
                             connectionstyle="arc3,rad=-0.6")
    ax.add_patch(arrow12)
    
    y_current = y_fusion - 2.0
    
    # ==================== Step 7: 分类器 ====================
    y_cls = y_current - 0.8
    
    cls_box = FancyBboxPatch((0.5, y_cls-1.2), 11, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['classifier'],
                              edgecolor='#C62828', linewidth=3)
    ax.add_patch(cls_box)
    ax.text(6, y_cls-0.2, 'Step 7: 分类器', 
            ha='center', va='center', fontsize=14, weight='bold', color='#B71C1C')
    
    cls_inner = FancyBboxPatch((1, y_cls-0.9), 9, 0.6,
                                boxstyle="round,pad=0.05",
                                facecolor='#FFCDD2',
                                edgecolor='#C62828', linewidth=2)
    ax.add_patch(cls_inner)
    ax.text(5.5, y_cls-0.6, 'Linear(768→384) → LayerNorm → GELU → Dropout(0.2) → Linear(384→2)',
            ha='center', va='center', fontsize=9, color=colors['text'])
    
    # 输出：预测
    pred_box = FancyBboxPatch((10.5, y_cls-0.9), 2, 0.6,
                               boxstyle="round,pad=0.05",
                               facecolor='#EF5350',
                               edgecolor='#B71C1C', linewidth=2)
    ax.add_patch(pred_box)
    ax.text(11.5, y_cls-0.6, '预测\n[B, 2]',
            ha='center', va='center', fontsize=9, weight='bold', color='white')
    
    # 箭头：Fusion到Classifier
    arrow13 = FancyArrowPatch((11.5, y_fusion-0.75), (11.5, y_cls-0.6),
                             arrowstyle='->', lw=2, color=colors['arrow'])
    ax.add_patch(arrow13)
    
    y_current = y_cls - 1.5
    
    # ==================== 损失函数模块 ====================
    y_loss = y_current - 0.5
    
    loss_main_box = FancyBboxPatch((13, y_loss-4.5), 10.5, 4.5,
                                     boxstyle="round,pad=0.1",
                                     facecolor=colors['loss'],
                                     edgecolor='#AD1457', linewidth=3)
    ax.add_patch(loss_main_box)
    ax.text(18.25, y_loss-0.3, '损失函数模块', 
            ha='center', va='center', fontsize=14, weight='bold', color='#880E4F')
    
    # OT Loss
    ot_loss_box = FancyBboxPatch((13.5, y_loss-1.2), 4.5, 0.8,
                                  boxstyle="round,pad=0.05",
                                  facecolor='#F8BBD0',
                                  edgecolor='#AD1457', linewidth=2)
    ax.add_patch(ot_loss_box)
    ax.text(15.75, y_loss-0.8, 'OT Loss (Sinkhorn)\nλ_ot = 0.5',
            ha='center', va='center', fontsize=9, color=colors['text'])
    
    # Alignment Loss
    align_loss_box = FancyBboxPatch((18.5, y_loss-1.2), 4.5, 0.8,
                                    boxstyle="round,pad=0.05",
                                    facecolor='#F48FB1',
                                    edgecolor='#AD1457', linewidth=2)
    ax.add_patch(align_loss_box)
    ax.text(20.75, y_loss-0.8, 'Alignment Loss\n(InfoNCE + L2)\nλ_align = 0.5',
            ha='center', va='center', fontsize=9, color=colors['text'])
    
    # Consistency Loss
    consist_loss_box = FancyBboxPatch((13.5, y_loss-2.2), 4.5, 0.8,
                                       boxstyle="round,pad=0.05",
                                       facecolor='#F8BBD0',
                                       edgecolor='#AD1457', linewidth=2)
    ax.add_patch(consist_loss_box)
    ax.text(15.75, y_loss-1.8, 'Consistency Loss\n(Counterfactual)\nλ_consist = 0.2',
            ha='center', va='center', fontsize=9, color=colors['text'])
    
    # Adversarial Loss
    adv_loss_box = FancyBboxPatch((18.5, y_loss-2.2), 4.5, 0.8,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#F48FB1',
                                   edgecolor='#AD1457', linewidth=2)
    ax.add_patch(adv_loss_box)
    ax.text(20.75, y_loss-1.8, 'Adversarial Loss\n(Center Discriminator)\nλ_adv = 0.5',
            ha='center', va='center', fontsize=9, color=colors['text'])
    
    # Sparse Loss
    sparse_loss_box = FancyBboxPatch((13.5, y_loss-3.2), 4.5, 0.8,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#F8BBD0',
                                      edgecolor='#AD1457', linewidth=2)
    ax.add_patch(sparse_loss_box)
    ax.text(15.75, y_loss-2.8, 'Sparse Loss\n(Attention Entropy)\nλ_sparse = 0.05',
            ha='center', va='center', fontsize=9, color=colors['text'])
    
    # Classification Loss
    cls_loss_box = FancyBboxPatch((18.5, y_loss-3.2), 4.5, 0.8,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#F48FB1',
                                   edgecolor='#AD1457', linewidth=2)
    ax.add_patch(cls_loss_box)
    ax.text(20.75, y_loss-2.8, 'Classification Loss\n(Focal Loss)\nλ_cls = 2.0',
            ha='center', va='center', fontsize=9, color=colors['text'])
    
    # Memory Bank
    memory_box = FancyBboxPatch((13.5, y_loss-4.2), 9.5, 0.8,
                                 boxstyle="round,pad=0.05",
                                 facecolor='#EC407A',
                                 edgecolor='#880E4F', linewidth=2)
    ax.add_patch(memory_box)
    ax.text(18.25, y_loss-3.8, 'Memory Bank: 存储z_noise用于生成反事实噪声',
            ha='center', va='center', fontsize=9, weight='bold', color='white')
    
    # 箭头：各个模块到损失函数
    arrow14 = FancyArrowPatch((8.25, y_dual-1.95), (15.75, y_loss-0.8),
                             arrowstyle='->', lw=1.5, color='#E91E63', linestyle=':',
                             connectionstyle="arc3,rad=0.4")
    ax.add_patch(arrow14)
    
    arrow15 = FancyArrowPatch((20.5, y_vlm-1.3), (20.75, y_loss-0.8),
                             arrowstyle='->', lw=1.5, color='#E91E63', linestyle=':',
                             connectionstyle="arc3,rad=-0.5")
    ax.add_patch(arrow15)
    
    arrow16 = FancyArrowPatch((13.75, y_dual-1.95), (15.75, y_loss-1.8),
                             arrowstyle='->', lw=1.5, color='#E91E63', linestyle=':',
                             connectionstyle="arc3,rad=0.3")
    ax.add_patch(arrow16)
    
    arrow17 = FancyArrowPatch((11.5, y_cls-0.6), (20.75, y_loss-2.8),
                             arrowstyle='->', lw=1.5, color='#E91E63', linestyle=':',
                             connectionstyle="arc3,rad=-0.3")
    ax.add_patch(arrow17)
    
    # ==================== 图例 ====================
    legend_y = 2.5
    legend_items = [
        ('输入层', colors['input']),
        ('VLM模块 (4.0优势)', colors['vlm']),
        ('Visual Notes (3.1优势)', colors['visual_notes']),
        ('自适应门控 (3.1优势)', colors['adaptive']),
        ('双头编码器', colors['dual_head']),
        ('对齐模块 (3.1优势)', colors['alignment']),
        ('融合模块', colors['fusion']),
        ('分类器', colors['classifier']),
        ('损失函数', colors['loss'])
    ]
    
    for i, (label, color) in enumerate(legend_items):
        x_pos = 0.5 + (i % 3) * 7.5
        y_pos = legend_y - (i // 3) * 0.4
        rect = Rectangle((x_pos, y_pos), 0.3, 0.25, facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x_pos + 0.4, y_pos + 0.125, label, fontsize=9, va='center', color=colors['text'])
    
    # 标题
    ax.text(12, 31.5, 'Bio-COT 3.2 Enhanced Architecture', 
            ha='center', va='center', fontsize=20, weight='bold', color='#1A237E')
    ax.text(12, 31.0, '融合3.1和4.0的优势：保留3.1的所有优点 + 引入4.0的计算效率和知识复杂度', 
            ha='center', va='center', fontsize=12, style='italic', color='#424242')
    
    # 保存图片
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    # 保存PNG
    png_path = output_dir / 'bio_cot_v3_2_architecture.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 架构图已保存: {png_path}")
    
    # 保存PDF
    pdf_path = output_dir / 'bio_cot_v3_2_architecture.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✅ PDF已保存: {pdf_path}")
    
    plt.close()
    
    return png_path, pdf_path


if __name__ == '__main__':
    print("🎨 开始绘制Bio-COT 3.2架构图...")
    png_path, pdf_path = draw_bio_cot_v3_2_architecture()
    print(f"✅ 完成！")
    print(f"   PNG: {png_path}")
    print(f"   PDF: {pdf_path}")

