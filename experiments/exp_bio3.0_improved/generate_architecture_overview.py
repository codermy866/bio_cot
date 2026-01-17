#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成Bio-COT 3.0 Improved整体架构概览图（SCI论文Figure 1）
包含详细的模块说明和数据流标注
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import matplotlib.patheffects as path_effects
import numpy as np
from pathlib import Path
from datetime import datetime

# 设置matplotlib（适合SCI论文）
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 15,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'pdf.fonttype': 42,
    'mathtext.fontset': 'stix',  # 数学公式字体
})


def create_architecture_overview(output_dir, timestamp):
    """
    创建Bio-COT 3.0 Improved整体架构概览图
    
    特点:
    - 清晰的模块划分和标注
    - 完整的数据流和维度标注
    - 关键公式标注
    - 损失函数说明
    - 适合SCI论文Figure 1
    """
    print("📊 生成Bio-COT 3.0 Improved架构概览图...")
    
    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # 定义颜色方案（论文风格）
    colors = {
        'input': '#E8F4F8',      # 浅蓝色 - 输入
        'knowledge': '#FFF4E6',  # 浅橙色 - Knowledge Notes
        'visual': '#E6F3FF',     # 浅蓝色 - Visual Notes
        'encoder': '#F0F8E8',    # 浅绿色 - 编码器
        'fusion': '#F5E6FF',     # 浅紫色 - 融合
        'causal': '#FFE6E6',     # 浅红色 - 因果特征
        'noise': '#E6E6E6',      # 浅灰色 - 噪声特征
        'ot': '#FFF0E6',         # 浅橙色 - OT
        'output': '#E8F8E8',     # 浅绿色 - 输出
        'border': '#333333',      # 深灰色 - 边框
        'arrow': '#666666',       # 灰色 - 箭头
        'formula': '#2C3E50',     # 深蓝色 - 公式
    }
    
    # ==================== 标题 ====================
    ax.text(10, 13.5, 'Bio-COT 3.0 Improved: Knowledge Notes Guided Causal Optimal Transport',
           ha='center', va='top', fontsize=16, fontweight='bold')
    ax.text(10, 13.1, 'Architecture Overview',
           ha='center', va='top', fontsize=12, style='italic')
    
    # ==================== 第一部分：输入层 ====================
    input_y = 11.5
    input_width = 2.5
    input_height = 1.2
    
    # OCT Images
    oct_box = FancyBboxPatch((0.5, input_y), input_width, input_height,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['input'],
                            edgecolor=colors['border'],
                            linewidth=2)
    ax.add_patch(oct_box)
    ax.text(1.75, input_y + 0.6, 'OCT Images', ha='center', va='center',
           fontsize=10, fontweight='bold')
    ax.text(1.75, input_y + 0.2, '[B, F, C, H, W]', ha='center', va='center',
           fontsize=8, style='italic')
    ax.text(1.75, input_y - 0.2, 'F=20 frames', ha='center', va='center',
           fontsize=8)
    
    # Colposcopy Images
    colpo_box = FancyBboxPatch((0.5, input_y - 1.5), input_width, input_height,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['input'],
                               edgecolor=colors['border'],
                               linewidth=2)
    ax.add_patch(colpo_box)
    ax.text(1.75, input_y - 0.9, 'Colposcopy Images', ha='center', va='center',
           fontsize=10, fontweight='bold')
    ax.text(1.75, input_y - 1.3, '[B, N, C, H, W]', ha='center', va='center',
           fontsize=8, style='italic')
    ax.text(1.75, input_y - 1.7, 'N=3 images', ha='center', va='center',
           fontsize=8)
    
    # Clinical Data
    clinical_box = FancyBboxPatch((0.5, input_y - 3), input_width, input_height,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['input'],
                                 edgecolor=colors['border'],
                                 linewidth=2)
    ax.add_patch(clinical_box)
    ax.text(1.75, input_y - 2.4, 'Clinical Data', ha='center', va='center',
           fontsize=10, fontweight='bold')
    ax.text(1.75, input_y - 2.8, '(HPV, TCT, Age)', ha='center', va='center',
           fontsize=8, style='italic')
    
    # Medical Knowledge Base
    kb_box = FancyBboxPatch((0.5, input_y - 4.5), input_width, input_height,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['input'],
                           edgecolor=colors['border'],
                           linewidth=2)
    ax.add_patch(kb_box)
    ax.text(1.75, input_y - 3.9, 'Medical KB', ha='center', va='center',
           fontsize=10, fontweight='bold')
    ax.text(1.75, input_y - 4.3, '(JSON)', ha='center', va='center',
           fontsize=8, style='italic')
    
    # ==================== 第二部分：ViT特征提取 ====================
    vit_x = 4
    vit_y = 11.5
    vit_width = 2.5
    vit_height = 1.2
    
    # ViT for OCT
    vit_oct_box = FancyBboxPatch((vit_x, vit_y), vit_width, vit_height,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['encoder'],
                                edgecolor=colors['border'],
                                linewidth=2)
    ax.add_patch(vit_oct_box)
    ax.text(vit_x + 1.25, vit_y + 0.6, 'ViT Encoder', ha='center', va='center',
           fontsize=10, fontweight='bold')
    ax.text(vit_x + 1.25, vit_y + 0.2, '(OCT)', ha='center', va='center',
           fontsize=9)
    ax.text(vit_x + 1.25, vit_y - 0.2, '[B,196,768]', ha='center', va='center',
           fontsize=8, style='italic')
    
    # ViT for Colposcopy
    vit_colpo_box = FancyBboxPatch((vit_x, vit_y - 1.5), vit_width, vit_height,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['encoder'],
                                   edgecolor=colors['border'],
                                   linewidth=2)
    ax.add_patch(vit_colpo_box)
    ax.text(vit_x + 1.25, vit_y - 0.9, 'ViT Encoder', ha='center', va='center',
           fontsize=10, fontweight='bold')
    ax.text(vit_x + 1.25, vit_y - 1.3, '(Colposcopy)', ha='center', va='center',
           fontsize=9)
    ax.text(vit_x + 1.25, vit_y - 1.7, '[B,196,768]', ha='center', va='center',
           fontsize=8, style='italic')
    
    # ==================== 第三部分：Knowledge Notes ====================
    kn_x = 4
    kn_y = 8.5
    kn_width_small = 1.2
    kn_width_large = 2
    
    
    # Medical KB
    kb_small_box = FancyBboxPatch((kn_x, kn_y), kn_width_small, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['knowledge'],
                                 edgecolor=colors['border'],
                                 linewidth=2)
    ax.add_patch(kb_small_box)
    ax.text(kn_x + 0.6, kn_y + 0.6, 'Medical\nKB', ha='center', va='center',
           fontsize=9, fontweight='bold')
    
    # RAG Retrieval
    rag_box = FancyBboxPatch((kn_x + 1.3, kn_y), kn_width_small, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['knowledge'],
                            edgecolor=colors['border'],
                            linewidth=2)
    ax.add_patch(rag_box)
    ax.text(kn_x + 1.9, kn_y + 0.6, 'RAG\nRetrieval', ha='center', va='center',
           fontsize=9, fontweight='bold')
    
    # Knowledge Notes Embedding
    kn_embed_box = FancyBboxPatch((kn_x, kn_y - 1.5), kn_width_large + kn_width_small + 0.1, 1.2,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['knowledge'],
                                  edgecolor=colors['border'],
                                  linewidth=2)
    ax.add_patch(kn_embed_box)
    ax.text(kn_x + 1.25, kn_y - 0.9, 'Knowledge Notes Embedding', ha='center', va='center',
           fontsize=10, fontweight='bold')
    ax.text(kn_x + 1.25, kn_y - 1.3, 'z_sem [B,768]', ha='center', va='center',
           fontsize=9, style='italic')
    
    # ==================== 第四部分：Visual Notes ====================
    vn_x = 7.5
    vn_y = 10
    
    # Cross-Modal Attention
    cross_attn_box = FancyBboxPatch((vn_x, vn_y), 2, 1.2,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['visual'],
                                   edgecolor=colors['border'],
                                   linewidth=2)
    ax.add_patch(cross_attn_box)
    ax.text(vn_x + 1, vn_y + 0.6, 'Cross-Modal', ha='center', va='center',
           fontsize=10, fontweight='bold')
    ax.text(vn_x + 1, vn_y + 0.2, 'Attention', ha='center', va='center',
           fontsize=9)
    ax.text(vn_x + 1, vn_y - 0.2, 'A=σ(QK^T/√d)', ha='center', va='center',
           fontsize=8, style='italic', color=colors['formula'])
    
    # Visual Notes (OCT)
    vn_oct_box = FancyBboxPatch((vn_x, vit_y), 2, vit_height,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['visual'],
                               edgecolor=colors['border'],
                               linewidth=2)
    ax.add_patch(vn_oct_box)
    ax.text(vn_x + 1, vit_y + 0.6, 'Visual Notes', ha='center', va='center',
           fontsize=10, fontweight='bold')
    ax.text(vn_x + 1, vit_y + 0.2, '(OCT)', ha='center', va='center',
           fontsize=9)
    ax.text(vn_x + 1, vit_y - 0.2, 'F_note = F⊙(M+(1-M)β)', ha='center', va='center',
           fontsize=8, style='italic', color=colors['formula'])
    
    # Visual Notes (Colposcopy)
    vn_colpo_box = FancyBboxPatch((vn_x, vit_y - 1.5), 2, vit_height,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['visual'],
                                 edgecolor=colors['border'],
                                 linewidth=2)
    ax.add_patch(vn_colpo_box)
    ax.text(vn_x + 1, vit_y - 0.9, 'Visual Notes', ha='center', va='center',
           fontsize=10, fontweight='bold')
    ax.text(vn_x + 1, vit_y - 1.3, '(Colposcopy)', ha='center', va='center',
           fontsize=9)
    ax.text(vn_x + 1, vit_y - 1.7, 'F_note = F⊙(M+(1-M)β)', ha='center', va='center',
           fontsize=8, style='italic', color=colors['formula'])
    
    # ==================== 第五部分：多模态融合 ====================
    mm_fusion_x = 10.5
    mm_fusion_y = 10.5
    mm_fusion_width = 2.5
    mm_fusion_height = 1.2
    
    mm_fusion_box = FancyBboxPatch((mm_fusion_x, mm_fusion_y), mm_fusion_width, mm_fusion_height,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['fusion'],
                                   edgecolor=colors['border'],
                                   linewidth=2)
    ax.add_patch(mm_fusion_box)
    ax.text(mm_fusion_x + 1.25, mm_fusion_y + 0.6, 'Multimodal', ha='center', va='center',
           fontsize=10, fontweight='bold')
    ax.text(mm_fusion_x + 1.25, mm_fusion_y + 0.2, 'Fusion', ha='center', va='center',
           fontsize=9)
    ax.text(mm_fusion_x + 1.25, mm_fusion_y - 0.2, '0.6·F_oct+0.4·F_colpo', ha='center', va='center',
           fontsize=8, style='italic', color=colors['formula'])
    
    # ==================== 第六部分：Dual-Head编码器 ====================
    dual_x = 13.5
    dual_y = 11
    dual_width = 2.5
    dual_height = 1.8
    
    # Causal Head
    causal_box = FancyBboxPatch((dual_x, dual_y), dual_width, dual_height,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['causal'],
                               edgecolor=colors['border'],
                               linewidth=2)
    ax.add_patch(causal_box)
    ax.text(dual_x + 1.25, dual_y + 1.2, 'Causal Head', ha='center', va='center',
           fontsize=10, fontweight='bold')
    ax.text(dual_x + 1.25, dual_y + 0.6, 'Encoder', ha='center', va='center',
           fontsize=9)
    ax.text(dual_x + 1.25, dual_y + 0.1, 'z_causal', ha='center', va='center',
           fontsize=9, style='italic', fontweight='bold')
    ax.text(dual_x + 1.25, dual_y - 0.4, '[B,768]', ha='center', va='center',
           fontsize=8, style='italic')
    
    # Noise Head
    noise_box = FancyBboxPatch((dual_x, dual_y - 2.2), dual_width, dual_height,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['noise'],
                              edgecolor=colors['border'],
                              linewidth=2)
    ax.add_patch(noise_box)
    ax.text(dual_x + 1.25, dual_y - 1, 'Noise Head', ha='center', va='center',
           fontsize=10, fontweight='bold')
    ax.text(dual_x + 1.25, dual_y - 1.6, 'Encoder', ha='center', va='center',
           fontsize=9)
    ax.text(dual_x + 1.25, dual_y - 2.1, 'z_noise', ha='center', va='center',
           fontsize=9, style='italic', fontweight='bold')
    ax.text(dual_x + 1.25, dual_y - 2.6, '[B,768]', ha='center', va='center',
           fontsize=8, style='italic')
    
    # ==================== 第七部分：Cross-Attention融合 ====================
    cross_fusion_x = 7.5
    cross_fusion_y = 6.5
    cross_fusion_width = 3
    cross_fusion_height = 1.5
    
    cross_fusion_box = FancyBboxPatch((cross_fusion_x, cross_fusion_y), cross_fusion_width, cross_fusion_height,
                                     boxstyle="round,pad=0.1",
                                     facecolor=colors['fusion'],
                                     edgecolor=colors['border'],
                                     linewidth=2)
    ax.add_patch(cross_fusion_box)
    ax.text(cross_fusion_x + 1.5, cross_fusion_y + 0.9, 'Cross-Attention', ha='center', va='center',
           fontsize=11, fontweight='bold')
    ax.text(cross_fusion_x + 1.5, cross_fusion_y + 0.4, 'Fusion', ha='center', va='center',
           fontsize=10)
    ax.text(cross_fusion_x + 1.5, cross_fusion_y - 0.1, 'z_final = CrossAttn', ha='center', va='center',
           fontsize=9, style='italic', color=colors['formula'])
    ax.text(cross_fusion_x + 1.5, cross_fusion_y - 0.5, '(z_causal, z_sem)', ha='center', va='center',
           fontsize=8, style='italic', color=colors['formula'])
    
    # ==================== 第八部分：Optimal Transport ====================
    ot_x = 11.5
    ot_y = 6.5
    ot_width = 2.5
    ot_height = 1.5
    
    ot_box = FancyBboxPatch((ot_x, ot_y), ot_width, ot_height,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['ot'],
                           edgecolor=colors['border'],
                           linewidth=2)
    ax.add_patch(ot_box)
    ax.text(ot_x + 1.25, ot_y + 0.9, 'Sinkhorn', ha='center', va='center',
           fontsize=11, fontweight='bold')
    ax.text(ot_x + 1.25, ot_y + 0.4, 'OT', ha='center', va='center',
           fontsize=10)
    ax.text(ot_x + 1.25, ot_y - 0.1, 'L_OT = ⟨P,C⟩', ha='center', va='center',
           fontsize=9, style='italic', color=colors['formula'])
    ax.text(ot_x + 1.25, ot_y - 0.5, '-εH(P)', ha='center', va='center',
           fontsize=9, style='italic', color=colors['formula'])
    
    # Memory Bank
    mb_x = 11.5
    mb_y = 4
    mb_width = 2.5
    mb_height = 1.5
    
    mb_box = FancyBboxPatch((mb_x, mb_y), mb_width, mb_height,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['ot'],
                           edgecolor=colors['border'],
                           linewidth=2)
    ax.add_patch(mb_box)
    ax.text(mb_x + 1.25, mb_y + 0.9, 'Memory Bank', ha='center', va='center',
           fontsize=10, fontweight='bold')
    ax.text(mb_x + 1.25, mb_y + 0.4, 'Counterfactual', ha='center', va='center',
           fontsize=9)
    ax.text(mb_x + 1.25, mb_y - 0.1, 'Intervention', ha='center', va='center',
           fontsize=9)
    
    # ==================== 第九部分：分类器 ====================
    cls_x = 15
    cls_y = 6.5
    cls_width = 2.5
    cls_height = 1.5
    
    cls_box = FancyBboxPatch((cls_x, cls_y), cls_width, cls_height,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['output'],
                            edgecolor=colors['border'],
                            linewidth=2)
    ax.add_patch(cls_box)
    ax.text(cls_x + 1.25, cls_y + 0.9, 'Classifier', ha='center', va='center',
           fontsize=11, fontweight='bold')
    ax.text(cls_x + 1.25, cls_y + 0.4, 'MLP', ha='center', va='center',
           fontsize=10)
    ax.text(cls_x + 1.25, cls_y - 0.1, 'P(y|x)', ha='center', va='center',
           fontsize=9, style='italic')
    ax.text(cls_x + 1.25, cls_y - 0.5, '[B,2]', ha='center', va='center',
           fontsize=8, style='italic')
    
    # Output
    output_x = 15
    output_y = 4
    output_width = 2.5
    output_height = 1.5
    
    output_box = FancyBboxPatch((output_x, output_y), output_width, output_height,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['output'],
                               edgecolor=colors['border'],
                               linewidth=2)
    ax.add_patch(output_box)
    ax.text(output_x + 1.25, output_y + 0.9, 'Prediction', ha='center', va='center',
           fontsize=10, fontweight='bold')
    ax.text(output_x + 1.25, output_y + 0.4, 'Threshold=0.580', ha='center', va='center',
           fontsize=9)
    ax.text(output_x + 1.25, output_y - 0.1, 'y ∈ {0,1}', ha='center', va='center',
           fontsize=9, style='italic')
    
    # ==================== 绘制箭头 ====================
    arrow_kwargs = dict(lw=2, color=colors['arrow'], head_width=0.15, head_length=0.2)
    
    # 输入到ViT
    ax.arrow(3, input_y + 0.6, 1, 0, **arrow_kwargs)
    ax.arrow(3, input_y - 0.9, 1, 0, **arrow_kwargs)
    ax.arrow(3, input_y - 2.4, 1, 0, **arrow_kwargs)
    ax.arrow(3, input_y - 3.9, 1, 0, **arrow_kwargs)
    
    # ViT到Visual Notes
    ax.arrow(6.5, vit_y + 0.6, 1, 0, **arrow_kwargs)
    ax.arrow(6.5, vit_y - 0.9, 1, 0, **arrow_kwargs)
    
    # Knowledge Notes到Cross-Attention
    ax.arrow(kn_x + 1.25, kn_y - 0.3, 0, 1.3, **arrow_kwargs)
    
    # Visual Notes到多模态融合
    ax.arrow(9.5, vit_y + 0.6, 1, 0.9, **arrow_kwargs)
    ax.arrow(9.5, vit_y - 0.9, 1, 1.4, **arrow_kwargs)
    
    # 多模态融合到Dual-Head
    ax.arrow(13, mm_fusion_y + 0.6, 0.5, 0.4, **arrow_kwargs)
    
    # Causal Head到Cross-Attention
    ax.arrow(dual_x + 1.25, dual_y, -4.25, -3.5, **arrow_kwargs)
    
    # Knowledge Notes到Cross-Attention（另一条路径）
    ax.arrow(kn_x + 1.25, kn_y - 0.3, 6.25, 0.8, **arrow_kwargs)
    
    # Cross-Attention到OT
    ax.arrow(10.5, cross_fusion_y + 0.75, 1, 0, **arrow_kwargs)
    
    # OT到分类器
    ax.arrow(14, ot_y + 0.75, 1, 0, **arrow_kwargs)
    
    # 分类器到输出
    ax.arrow(cls_x + 1.25, cls_y, 0, -1.5, **arrow_kwargs)
    
    # Memory Bank到OT（双向）
    ax.arrow(mb_x + 1.25, mb_y + 1.5, 0, 0.5, **arrow_kwargs)
    ax.arrow(ot_x + 1.25, ot_y, 0, -0.5, **arrow_kwargs)
    
    # ==================== 添加损失函数说明 ====================
    loss_box = FancyBboxPatch((0.5, 1.5), 6, 1.8,
                             boxstyle="round,pad=0.1",
                             facecolor='wheat',
                             edgecolor=colors['border'],
                             linewidth=2)
    ax.add_patch(loss_box)
    
    loss_title = "Loss Functions:"
    ax.text(3.5, 2.8, loss_title, ha='center', va='center',
           fontsize=10, fontweight='bold')
    
    loss_text = [
        "L_total = λ_cls·L_cls + λ_ot·L_ot + λ_sparse·L_sparse",
        "         + λ_consist·L_consist + λ_adv·L_adv",
        "λ_cls=2.0, λ_ot=0.5, λ_sparse=0.01, λ_consist=0.2, λ_adv=0.5"
    ]
    for i, text in enumerate(loss_text):
        ax.text(3.5, 2.4 - i*0.3, text, ha='center', va='center',
               fontsize=8, family='monospace')
    
    # ==================== 添加模块分组标签 ====================
    module_labels = [
        (1.75, 13.2, 'Input Layer', 'left'),
        (5.25, 13.2, 'Feature Extraction', 'center'),
        (9, 13.2, 'Knowledge & Visual Notes', 'center'),
        (14.75, 13.2, 'Dual-Head Encoder', 'center'),
        (9, 8.5, 'Cross-Modal Fusion & OT', 'center'),
        (16.25, 8.5, 'Classification', 'center'),
    ]
    
    for x, y, text, ha in module_labels:
        ax.text(x, y, text, ha=ha, va='bottom',
               fontsize=11, fontweight='bold', style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=colors['border']))
    
    # ==================== 添加图例 ====================
    legend_elements = [
        mpatches.Patch(facecolor=colors['input'], edgecolor=colors['border'], label='Input'),
        mpatches.Patch(facecolor=colors['knowledge'], edgecolor=colors['border'], label='Knowledge Notes'),
        mpatches.Patch(facecolor=colors['visual'], edgecolor=colors['border'], label='Visual Notes'),
        mpatches.Patch(facecolor=colors['encoder'], edgecolor=colors['border'], label='Encoder'),
        mpatches.Patch(facecolor=colors['causal'], edgecolor=colors['border'], label='Causal Features'),
        mpatches.Patch(facecolor=colors['noise'], edgecolor=colors['border'], label='Noise Features'),
        mpatches.Patch(facecolor=colors['fusion'], edgecolor=colors['border'], label='Fusion'),
        mpatches.Patch(facecolor=colors['ot'], edgecolor=colors['border'], label='Optimal Transport'),
        mpatches.Patch(facecolor=colors['output'], edgecolor=colors['border'], label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8, ncol=3, framealpha=0.9)
    
    plt.tight_layout()
    
    # 保存
    output_path_pdf = output_dir / f"architecture_overview_{timestamp}.pdf"
    output_path_png = output_dir / f"architecture_overview_{timestamp}.png"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 架构概览图已保存: {output_path_pdf}")
    return output_path_pdf


def main():
    """主函数"""
    print("=" * 80)
    print("生成Bio-COT 3.0 Improved架构概览图（SCI论文Figure 1）")
    print("=" * 80)
    
    from config import BioCOT_v3_Config
    config = BioCOT_v3_Config()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成架构图
    print("\n生成架构概览图...")
    fig_path = create_architecture_overview(output_dir, timestamp)
    
    print("\n" + "=" * 80)
    print("✅ 架构图生成完成！")
    print("=" * 80)
    print(f"输出目录: {output_dir}")
    print(f"\n生成的文件:")
    print(f"  - architecture_overview_{timestamp}.pdf (PDF格式，适合论文)")
    print(f"  - architecture_overview_{timestamp}.png (PNG格式，适合展示)")
    print(f"\n详细说明文档:")
    print(f"  - ARCHITECTURE_OVERVIEW.md (架构详细说明)")


if __name__ == '__main__':
    main()

