#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成Bio-COT 3.0模型架构图（SCI论文Figure 1）
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Ellipse
import matplotlib.patheffects as path_effects
import numpy as np
from pathlib import Path
from datetime import datetime

# 设置matplotlib
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'pdf.fonttype': 42,
})

def create_architecture_figure(output_dir, timestamp):
    """创建Bio-COT 3.0架构图"""
    print("📊 生成Bio-COT 3.0架构图...")
    
    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # 定义颜色方案
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
    }
    
    # ==================== 第一部分：输入层 ====================
    # OCT Images
    oct_box = FancyBboxPatch((0.5, 11.5), 2.5, 1.5, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['input'],
                            edgecolor=colors['border'],
                            linewidth=2)
    ax.add_patch(oct_box)
    ax.text(1.75, 12.5, 'OCT Images\n[B, F, C, H, W]', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Colposcopy Images
    colpo_box = FancyBboxPatch((0.5, 9), 2.5, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['input'],
                              edgecolor=colors['border'],
                              linewidth=2)
    ax.add_patch(colpo_box)
    ax.text(1.75, 9.75, 'Colposcopy Images\n[B, N, C, H, W]',
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Clinical Data
    clinical_box = FancyBboxPatch((0.5, 6.5), 2.5, 1.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['input'],
                                 edgecolor=colors['border'],
                                 linewidth=2)
    ax.add_patch(clinical_box)
    ax.text(1.75, 7.25, 'Clinical Data\n(HPV, TCT, Age)',
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ==================== 第二部分：ViT特征提取 ====================
    # ViT Encoder for OCT
    vit_oct_box = FancyBboxPatch((4, 11.5), 2.5, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['encoder'],
                                edgecolor=colors['border'],
                                linewidth=2)
    ax.add_patch(vit_oct_box)
    ax.text(5.25, 12.5, 'ViT Encoder\n(Patch Features)',
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ViT Encoder for Colposcopy
    vit_colpo_box = FancyBboxPatch((4, 9), 2.5, 1.5,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['encoder'],
                                  edgecolor=colors['border'],
                                  linewidth=2)
    ax.add_patch(vit_colpo_box)
    ax.text(5.25, 9.75, 'ViT Encoder\n(Patch Features)',
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ==================== 第三部分：Knowledge Notes ====================
    # Knowledge Base
    kb_box = FancyBboxPatch((4, 6.5), 1.2, 1.5,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['knowledge'],
                           edgecolor=colors['border'],
                           linewidth=2)
    ax.add_patch(kb_box)
    ax.text(4.6, 7.25, 'Medical\nKnowledge\nBase',
           ha='center', va='center', fontsize=9, fontweight='bold')
    
    # RAG Retrieval
    rag_box = FancyBboxPatch((5.3, 6.5), 1.2, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['knowledge'],
                            edgecolor=colors['border'],
                            linewidth=2)
    ax.add_patch(rag_box)
    ax.text(5.9, 7.25, 'RAG\nRetrieval',
           ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Knowledge Notes Embedding
    kn_embed_box = FancyBboxPatch((7.5, 6.5), 2, 1.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['knowledge'],
                                 edgecolor=colors['border'],
                                 linewidth=2)
    ax.add_patch(kn_embed_box)
    ax.text(8.5, 7.25, 'Knowledge Notes\nEmbedding\n[768-dim]',
           ha='center', va='center', fontsize=9, fontweight='bold')
    
    # ==================== 第四部分：Visual Notes ====================
    # Cross-Modal Attention
    cross_attn_box = FancyBboxPatch((7.5, 9), 2, 1.5,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['visual'],
                                   edgecolor=colors['border'],
                                   linewidth=2)
    ax.add_patch(cross_attn_box)
    ax.text(8.5, 9.75, 'Cross-Modal\nAttention',
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Visual Notes (OCT)
    vn_oct_box = FancyBboxPatch((10.5, 11.5), 2, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['visual'],
                               edgecolor=colors['border'],
                               linewidth=2)
    ax.add_patch(vn_oct_box)
    ax.text(11.5, 12.5, 'Visual Notes\n(OCT)\nAttention Map',
           ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Visual Notes (Colposcopy)
    vn_colpo_box = FancyBboxPatch((10.5, 9), 2, 1.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['visual'],
                                 edgecolor=colors['border'],
                                 linewidth=2)
    ax.add_patch(vn_colpo_box)
    ax.text(11.5, 9.75, 'Visual Notes\n(Colposcopy)\nAttention Map',
           ha='center', va='center', fontsize=9, fontweight='bold')
    
    # ==================== 第五部分：Dual-Head Image Encoder ====================
    # Causal Head
    causal_head_box = FancyBboxPatch((13.5, 11), 2.5, 2,
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['causal'],
                                    edgecolor=colors['border'],
                                    linewidth=2)
    ax.add_patch(causal_head_box)
    ax.text(14.75, 12.5, 'Causal Head\nEncoder',
           ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(14.75, 11.8, 'z_causal',
           ha='center', va='center', fontsize=9, style='italic')
    
    # Noise Head
    noise_head_box = FancyBboxPatch((13.5, 8.5), 2.5, 2,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['noise'],
                                   edgecolor=colors['border'],
                                   linewidth=2)
    ax.add_patch(noise_head_box)
    ax.text(14.75, 9.75, 'Noise Head\nEncoder',
           ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(14.75, 9.05, 'z_noise',
           ha='center', va='center', fontsize=9, style='italic')
    
    # ==================== 第六部分：特征融合 ====================
    # Cross-Attention Fusion
    fusion_box = FancyBboxPatch((7.5, 3.5), 3, 2,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['fusion'],
                               edgecolor=colors['border'],
                               linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(9, 5, 'Cross-Attention\nFusion',
           ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(9, 4.3, 'z_causal + z_sem',
           ha='center', va='center', fontsize=9, style='italic')
    
    # ==================== 第七部分：Optimal Transport ====================
    # OT Alignment
    ot_box = FancyBboxPatch((11.5, 3.5), 2.5, 2,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['ot'],
                           edgecolor=colors['border'],
                           linewidth=2)
    ax.add_patch(ot_box)
    ax.text(12.75, 4.8, 'Sinkhorn\nOT',
           ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(12.75, 4.2, 'Domain\nAlignment',
           ha='center', va='center', fontsize=9)
    
    # Memory Bank
    mb_box = FancyBboxPatch((11.5, 1), 2.5, 1.8,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['ot'],
                           edgecolor=colors['border'],
                           linewidth=2)
    ax.add_patch(mb_box)
    ax.text(12.75, 1.9, 'Memory Bank\n(Counterfactual)',
           ha='center', va='center', fontsize=9, fontweight='bold')
    
    # ==================== 第八部分：分类器 ====================
    # Classifier
    cls_box = FancyBboxPatch((15, 3.5), 2.5, 2,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['output'],
                            edgecolor=colors['border'],
                            linewidth=2)
    ax.add_patch(cls_box)
    ax.text(16.25, 5, 'Classifier',
           ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(16.25, 4.3, 'MLP',
           ha='center', va='center', fontsize=9)
    
    # Output
    output_box = FancyBboxPatch((15, 1), 2.5, 1.8,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['output'],
                               edgecolor=colors['border'],
                               linewidth=2)
    ax.add_patch(output_box)
    ax.text(16.25, 1.9, 'Prediction\nP(y|x)',
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ==================== 绘制箭头 ====================
    # 输入到ViT
    arrow1 = FancyArrowPatch((3, 12.25), (4, 12.25),
                            arrowstyle='->', lw=2, color=colors['arrow'],
                            mutation_scale=20)
    ax.add_patch(arrow1)
    
    arrow2 = FancyArrowPatch((3, 9.75), (4, 9.75),
                            arrowstyle='->', lw=2, color=colors['arrow'],
                            mutation_scale=20)
    ax.add_patch(arrow2)
    
    # Clinical Data到Knowledge Base
    arrow3 = FancyArrowPatch((3, 7.25), (4, 7.25),
                            arrowstyle='->', lw=2, color=colors['arrow'],
                            mutation_scale=20)
    ax.add_patch(arrow3)
    
    # ViT到Visual Notes
    arrow4 = FancyArrowPatch((6.5, 12.25), (10.5, 12.25),
                            arrowstyle='->', lw=2, color=colors['arrow'],
                            mutation_scale=20)
    ax.add_patch(arrow4)
    
    arrow5 = FancyArrowPatch((6.5, 9.75), (10.5, 9.75),
                            arrowstyle='->', lw=2, color=colors['arrow'],
                            mutation_scale=20)
    ax.add_patch(arrow5)
    
    # Knowledge Notes到Cross-Attention
    arrow6 = FancyArrowPatch((9.5, 7.25), (8.5, 9.25),
                            arrowstyle='->', lw=2, color=colors['arrow'],
                            mutation_scale=20)
    ax.add_patch(arrow6)
    
    # Visual Notes到Dual-Head
    arrow7 = FancyArrowPatch((12.5, 12.25), (13.5, 12.5),
                            arrowstyle='->', lw=2, color=colors['arrow'],
                            mutation_scale=20)
    ax.add_patch(arrow7)
    
    arrow8 = FancyArrowPatch((12.5, 9.75), (13.5, 9.75),
                            arrowstyle='->', lw=2, color=colors['arrow'],
                            mutation_scale=20)
    ax.add_patch(arrow8)
    
    # Knowledge Notes到Fusion
    arrow9 = FancyArrowPatch((9.5, 6.5), (9, 5.5),
                            arrowstyle='->', lw=2, color=colors['arrow'],
                            mutation_scale=20)
    ax.add_patch(arrow9)
    
    # Causal Head到Fusion
    arrow10 = FancyArrowPatch((14.75, 11), (10.5, 4.5),
                             arrowstyle='->', lw=2, color=colors['arrow'],
                             mutation_scale=20)
    ax.add_patch(arrow10)
    
    # Fusion到OT
    arrow11 = FancyArrowPatch((10.5, 4.5), (11.5, 4.5),
                             arrowstyle='->', lw=2, color=colors['arrow'],
                             mutation_scale=20)
    ax.add_patch(arrow11)
    
    # OT到Classifier
    arrow12 = FancyArrowPatch((14, 4.5), (15, 4.5),
                             arrowstyle='->', lw=2, color=colors['arrow'],
                             mutation_scale=20)
    ax.add_patch(arrow12)
    
    # Classifier到Output
    arrow13 = FancyArrowPatch((16.25, 3.5), (16.25, 2.8),
                             arrowstyle='->', lw=2, color=colors['arrow'],
                             mutation_scale=20)
    ax.add_patch(arrow13)
    
    # Memory Bank到OT（双向）
    arrow14 = FancyArrowPatch((12.75, 2.8), (12.75, 3.5),
                             arrowstyle='<->', lw=2, color=colors['arrow'],
                             mutation_scale=20)
    ax.add_patch(arrow14)
    
    # ==================== 添加标签和说明 ====================
    # 模块分组标签
    ax.text(1.75, 13.5, 'Input Layer', ha='center', va='bottom',
           fontsize=12, fontweight='bold', style='italic')
    
    ax.text(5.25, 13.5, 'Feature Extraction', ha='center', va='bottom',
           fontsize=12, fontweight='bold', style='italic')
    
    ax.text(8.5, 13.5, 'Knowledge & Visual Notes', ha='center', va='bottom',
           fontsize=12, fontweight='bold', style='italic')
    
    ax.text(14.75, 13.5, 'Dual-Head Encoder', ha='center', va='bottom',
           fontsize=12, fontweight='bold', style='italic')
    
    ax.text(9, 6, 'Feature Fusion & OT', ha='center', va='top',
           fontsize=12, fontweight='bold', style='italic')
    
    ax.text(16.25, 6, 'Classification', ha='center', va='top',
           fontsize=12, fontweight='bold', style='italic')
    
    # 添加损失函数标注
    loss_text = """
    Loss Functions:
    • Classification Loss (λ_cls=2.0)
    • OT Loss (λ_ot=0.5)
    • Sparse Loss (λ_sparse=0.01)
    • Consistency Loss (λ_consist=0.2)
    • Adversarial Loss (λ_adv=0.5)
    """
    ax.text(0.5, 4, loss_text, ha='left', va='top',
           fontsize=9, family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 添加标题
    ax.text(10, 13.8, 'Bio-COT 3.0: Knowledge Notes Guided Causal Optimal Transport Architecture',
           ha='center', va='top', fontsize=16, fontweight='bold')
    
    # 添加图例
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
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9, ncol=3)
    
    plt.tight_layout()
    
    # 保存
    output_path_pdf = output_dir / f"architecture_figure1_{timestamp}.pdf"
    output_path_png = output_dir / f"architecture_figure1_{timestamp}.png"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 架构图已保存: {output_path_pdf}")
    return output_path_pdf


def create_detailed_architecture_figure(output_dir, timestamp):
    """创建更详细的架构图（包含数学公式）"""
    print("📊 生成详细架构图（含公式）...")
    
    fig = plt.figure(figsize=(22, 16))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    colors = {
        'input': '#E8F4F8',
        'knowledge': '#FFF4E6',
        'visual': '#E6F3FF',
        'encoder': '#F0F8E8',
        'fusion': '#F5E6FF',
        'causal': '#FFE6E6',
        'noise': '#E6E6E6',
        'ot': '#FFF0E6',
        'output': '#E8F8E8',
        'border': '#333333',
        'arrow': '#666666',
    }
    
    # 标题
    ax.text(11, 15.5, 'Bio-COT 3.0: Complete Architecture Overview',
           ha='center', va='top', fontsize=18, fontweight='bold')
    
    # ==================== 左侧：数据流 ====================
    # 输入
    input_y = 13
    oct_box = FancyBboxPatch((0.5, input_y), 3, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['input'],
                            edgecolor=colors['border'],
                            linewidth=2)
    ax.add_patch(oct_box)
    ax.text(2, input_y+0.6, 'OCT Images\n[B, F, C, H, W]',
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    colpo_box = FancyBboxPatch((0.5, input_y-1.5), 3, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['input'],
                              edgecolor=colors['border'],
                              linewidth=2)
    ax.add_patch(colpo_box)
    ax.text(2, input_y-0.9, 'Colposcopy Images\n[B, N, C, H, W]',
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    clinical_box = FancyBboxPatch((0.5, input_y-3), 3, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['input'],
                                 edgecolor=colors['border'],
                                 linewidth=2)
    ax.add_patch(clinical_box)
    ax.text(2, input_y-2.4, 'Clinical Data\n(HPV, TCT, Age)',
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ViT编码器
    vit_y = 11.5
    vit_oct_box = FancyBboxPatch((4.5, vit_y), 3, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['encoder'],
                                edgecolor=colors['border'],
                                linewidth=2)
    ax.add_patch(vit_oct_box)
    ax.text(6, vit_y+0.6, 'ViT Encoder (OCT)\nf_oct [B×196×768]',
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    vit_colpo_box = FancyBboxPatch((4.5, vit_y-1.5), 3, 1.2,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['encoder'],
                                  edgecolor=colors['border'],
                                  linewidth=2)
    ax.add_patch(vit_colpo_box)
    ax.text(6, vit_y-0.9, 'ViT Encoder (Colpo)\nf_colpo [B×196×768]',
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Knowledge Notes
    kn_y = 8.5
    kb_box = FancyBboxPatch((4.5, kn_y), 1.4, 1.2,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['knowledge'],
                           edgecolor=colors['border'],
                           linewidth=2)
    ax.add_patch(kb_box)
    ax.text(5.2, kn_y+0.6, 'Medical\nKB',
           ha='center', va='center', fontsize=9, fontweight='bold')
    
    rag_box = FancyBboxPatch((6, kn_y), 1.5, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['knowledge'],
                            edgecolor=colors['border'],
                            linewidth=2)
    ax.add_patch(rag_box)
    ax.text(6.75, kn_y+0.6, 'RAG\nRetrieval',
           ha='center', va='center', fontsize=9, fontweight='bold')
    
    kn_embed_box = FancyBboxPatch((4.5, kn_y-1.5), 3, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['knowledge'],
                                 edgecolor=colors['border'],
                                 linewidth=2)
    ax.add_patch(kn_embed_box)
    ax.text(6, kn_y-0.9, 'Knowledge Notes Embedding\nz_sem [B×768]',
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Visual Notes
    vn_y = 6
    cross_attn_box = FancyBboxPatch((8.5, vn_y), 3, 1.2,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['visual'],
                                   edgecolor=colors['border'],
                                   linewidth=2)
    ax.add_patch(cross_attn_box)
    ax.text(10, vn_y+0.6, 'Cross-Modal Attention\nA = softmax(QK^T/√d)',
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    vn_oct_box = FancyBboxPatch((8.5, vn_y-1.5), 1.4, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['visual'],
                               edgecolor=colors['border'],
                               linewidth=2)
    ax.add_patch(vn_oct_box)
    ax.text(9.2, vn_y-0.9, 'Visual Notes\n(OCT)\nA_oct',
           ha='center', va='center', fontsize=9, fontweight='bold')
    
    vn_colpo_box = FancyBboxPatch((10, vn_y-1.5), 1.4, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['visual'],
                                 edgecolor=colors['border'],
                                 linewidth=2)
    ax.add_patch(vn_colpo_box)
    ax.text(10.7, vn_y-0.9, 'Visual Notes\n(Colpo)\nA_colpo',
           ha='center', va='center', fontsize=9, fontweight='bold')
    
    # ==================== 中间：Dual-Head编码器 ====================
    dual_y = 10
    causal_box = FancyBboxPatch((12.5, dual_y+0.6), 3, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['causal'],
                               edgecolor=colors['border'],
                               linewidth=2)
    ax.add_patch(causal_box)
    ax.text(14, dual_y+1.2, 'Causal Head\nz_causal = Enc_causal(f_masked)',
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    noise_box = FancyBboxPatch((12.5, dual_y-0.9), 3, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['noise'],
                              edgecolor=colors['border'],
                              linewidth=2)
    ax.add_patch(noise_box)
    ax.text(14, dual_y-0.3, 'Noise Head\nz_noise = Enc_noise(f_masked)',
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ==================== 右侧：融合和分类 ====================
    fusion_y = 7
    fusion_box = FancyBboxPatch((16.5, fusion_y), 3.5, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['fusion'],
                               edgecolor=colors['border'],
                               linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(18.25, fusion_y+0.75, 'Cross-Attention Fusion',
           ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(18.25, fusion_y+0.2, 'z_fused = CrossAttn(z_causal, z_sem)',
           ha='center', va='center', fontsize=9, style='italic')
    
    # OT
    ot_y = 5
    ot_box = FancyBboxPatch((16.5, ot_y), 3.5, 1.5,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['ot'],
                           edgecolor=colors['border'],
                           linewidth=2)
    ax.add_patch(ot_box)
    ax.text(18.25, ot_y+0.75, 'Sinkhorn Optimal Transport',
           ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(18.25, ot_y+0.2, 'L_OT = ⟨P, C⟩ + λH(P)',
           ha='center', va='center', fontsize=9, style='italic')
    
    # Memory Bank
    mb_y = 2.5
    mb_box = FancyBboxPatch((16.5, mb_y), 3.5, 1.5,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['ot'],
                           edgecolor=colors['border'],
                           linewidth=2)
    ax.add_patch(mb_box)
    ax.text(18.25, mb_y+0.75, 'Memory Bank',
           ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(18.25, mb_y+0.2, 'Counterfactual Intervention',
           ha='center', va='center', fontsize=9)
    
    # Classifier
    cls_y = 0.5
    cls_box = FancyBboxPatch((16.5, cls_y), 3.5, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['output'],
                            edgecolor=colors['border'],
                            linewidth=2)
    ax.add_patch(cls_box)
    ax.text(18.25, cls_y+0.75, 'Classifier',
           ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(18.25, cls_y+0.2, 'P(y|x) = softmax(MLP(z_fused))',
           ha='center', va='center', fontsize=9, style='italic')
    
    # ==================== 绘制箭头 ====================
    # 输入到ViT
    ax.arrow(3.5, input_y+0.6, 1, 0, head_width=0.15, head_length=0.2,
            fc=colors['arrow'], ec=colors['arrow'], lw=2)
    ax.arrow(3.5, input_y-0.9, 1, 0, head_width=0.15, head_length=0.2,
            fc=colors['arrow'], ec=colors['arrow'], lw=2)
    ax.arrow(3.5, input_y-2.4, 1, 0, head_width=0.15, head_length=0.2,
            fc=colors['arrow'], ec=colors['arrow'], lw=2)
    
    # ViT到Visual Notes
    ax.arrow(7.5, vit_y+0.6, 1, -1.1, head_width=0.15, head_length=0.2,
            fc=colors['arrow'], ec=colors['arrow'], lw=2)
    ax.arrow(7.5, vit_y-0.9, 1, -0.6, head_width=0.15, head_length=0.2,
            fc=colors['arrow'], ec=colors['arrow'], lw=2)
    
    # Knowledge Notes到Cross-Attention
    ax.arrow(7.5, kn_y-0.9, 1, 1.4, head_width=0.15, head_length=0.2,
            fc=colors['arrow'], ec=colors['arrow'], lw=2)
    
    # Visual Notes到Dual-Head
    ax.arrow(11.5, vn_y-0.9, 1, 1.9, head_width=0.15, head_length=0.2,
            fc=colors['arrow'], ec=colors['arrow'], lw=2)
    
    # Causal Head到Fusion
    ax.arrow(15.5, dual_y+1.2, 1, -3.45, head_width=0.15, head_length=0.2,
            fc=colors['arrow'], ec=colors['arrow'], lw=2)
    
    # Knowledge Notes到Fusion
    ax.arrow(7.5, kn_y-0.9, 9, -2.4, head_width=0.15, head_length=0.2,
            fc=colors['arrow'], ec=colors['arrow'], lw=2)
    
    # Fusion到OT
    ax.arrow(18.25, fusion_y, 0, -1.5, head_width=0.15, head_length=0.2,
            fc=colors['arrow'], ec=colors['arrow'], lw=2)
    
    # OT到Classifier
    ax.arrow(18.25, ot_y, 0, -1.5, head_width=0.15, head_length=0.2,
            fc=colors['arrow'], ec=colors['arrow'], lw=2)
    
    # Memory Bank到OT（双向）
    ax.arrow(18.25, mb_y+1.5, 0, 0.5, head_width=0.15, head_length=0.2,
            fc=colors['arrow'], ec=colors['arrow'], lw=2)
    ax.arrow(18.25, ot_y, 0, -0.5, head_width=0.15, head_length=0.2,
            fc=colors['arrow'], ec=colors['arrow'], lw=2)
    
    # ==================== 添加损失函数框 ====================
    loss_box = FancyBboxPatch((0.5, 0.5), 8, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor='wheat',
                             edgecolor=colors['border'],
                             linewidth=2)
    ax.add_patch(loss_box)
    loss_text = "Total Loss: L = λ_cls·L_cls + λ_ot·L_ot + λ_sparse·L_sparse + λ_consist·L_consist + λ_adv·L_adv"
    ax.text(4.5, 1.25, loss_text, ha='center', va='center',
           fontsize=10, fontweight='bold')
    
    # ==================== 添加模块说明 ====================
    module_text = """
    Key Modules:
    • Knowledge Notes: RAG-based semantic embedding from medical KB
    • Visual Notes: Cross-modal attention for lesion localization
    • Dual-Head Encoder: Causal/Noise feature decoupling
    • Optimal Transport: Cross-center domain alignment
    • Memory Bank: Counterfactual intervention for causal consistency
    """
    ax.text(0.5, 3, module_text, ha='left', va='top',
           fontsize=9, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    output_path_pdf = output_dir / f"architecture_detailed_{timestamp}.pdf"
    output_path_png = output_dir / f"architecture_detailed_{timestamp}.png"
    plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 详细架构图已保存: {output_path_pdf}")
    return output_path_pdf


def main():
    """主函数"""
    print("=" * 80)
    print("生成Bio-COT 3.0架构图（SCI论文Figure 1）")
    print("=" * 80)
    
    from config import BioCOT_v3_Config
    config = BioCOT_v3_Config()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成架构图
    print("\n生成架构图...")
    fig1 = create_architecture_figure(output_dir, timestamp)
    fig2 = create_detailed_architecture_figure(output_dir, timestamp)
    
    print("\n" + "=" * 80)
    print("✅ 架构图生成完成！")
    print("=" * 80)
    print(f"输出目录: {output_dir}")
    print(f"\n生成的文件:")
    print(f"  - architecture_figure1_{timestamp}.pdf (简洁版，适合Figure 1)")
    print(f"  - architecture_detailed_{timestamp}.pdf (详细版，含公式)")


if __name__ == '__main__':
    main()

