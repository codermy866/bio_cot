#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2 Architecture Visualization for SCI Paper
完整的模型架构图绘制代码（适合学术论文发表）
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# 设置字体和样式（适合SCI论文）
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 300  # 高分辨率

def draw_bio_cot_architecture():
    """绘制Bio-COT 3.2完整架构图"""
    
    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # ================== 定义颜色方案 ==================
    colors = {
        'input': '#E8F4F8',       # 浅蓝色 - 输入层
        'feature': '#FFF4E6',     # 浅橙色 - 特征提取
        'semantic': '#F0E6FF',    # 浅紫色 - 语义增强
        'fusion': '#E6F7E6',      # 浅绿色 - 多模态融合
        'align': '#FFE6E6',       # 浅红色 - 对齐机制
        'output': '#FFE6CC',      # 浅金色 - 输出层
        'frozen': '#D0D0D0',      # 灰色 - 冻结模块
        'trainable': '#B8E6B8'    # 亮绿色 - 可训练模块
    }
    
    # ================== 第1层：输入层 ==================
    y_start = 12.5
    
    # 标题
    ax.text(10, 13.5, 'Bio-COT 3.2: Biomedical Chain-of-Thought Architecture', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(10, 13.0, 'Multi-Modal Medical Image Analysis with Frozen VLM and Adaptive Fusion',
            ha='center', va='center', fontsize=12, style='italic', color='#555')
    
    # OCT输入
    draw_module(ax, 1, y_start, 3, 1, 'OCT Images\n[B, 20, 3, 224, 224]', colors['input'])
    
    # Colposcopy输入
    draw_module(ax, 5.5, y_start, 3, 1, 'Colposcopy\n[B, 3, 3, 224, 224]', colors['input'])
    
    # Clinical输入
    draw_module(ax, 10, y_start, 3, 1, 'Clinical Info\n[B, 7]\n(Age, HPV, TCT)', colors['input'])
    
    # VLM Cache (Frozen)
    draw_module(ax, 14.5, y_start, 4, 1, 
                'VLM Cache (Frozen)\n102,705 Descriptions\n❄️ Pre-generated', 
                colors['frozen'], edgecolor='#555', linewidth=2)
    
    # ================== 第2层：特征提取 ==================
    y_level2 = 10.5
    
    # HierarchicalViT (4 stages)
    hierarchical_x = 2.5
    draw_module(ax, hierarchical_x-0.5, y_level2, 5, 1.5, 
                'HierarchicalViT\n(Frozen Backbone)', 
                colors['frozen'], edgecolor='#555', linewidth=2)
    
    # 4个stage的输出
    stage_y = y_level2 - 0.5
    for i, layer in enumerate([2, 5, 8, 11]):
        x_pos = hierarchical_x + i * 1.2
        draw_small_box(ax, x_pos, stage_y, 0.8, 0.4, 
                      f'L{layer}\n[196,768]', '#FFE6B3')
        # 箭头从输入到stage
        draw_arrow(ax, 2.5, y_start-0.5, x_pos+0.4, stage_y+0.4, style='solid', color='#333')
    
    # Clinical Encoder
    draw_module(ax, 10, y_level2, 3, 0.8, 
                'Clinical Encoder\n🔥 MLP [7→768]', colors['trainable'])
    draw_arrow(ax, 11.5, y_start-0.5, 11.5, y_level2+0.4, style='solid', color='#333')
    
    # Text Encoder (Frozen)
    text_enc_x = 15.5
    draw_module(ax, text_enc_x, y_level2+0.4, 3, 0.6, 
                'PubMedBERT\n❄️ 109M params (Frozen)', 
                colors['frozen'], edgecolor='#555', linewidth=2)
    draw_arrow(ax, 16.5, y_start-0.5, 16.5, y_level2+1.0, style='solid', color='#555')
    
    # Trainable Adapter
    draw_module(ax, text_enc_x, y_level2-0.4, 3, 0.6, 
                'Trainable Adapter\n🔥 5.7M params', colors['trainable'])
    draw_arrow(ax, 16.5, y_level2+0.4, 16.5, y_level2+0.2, style='solid', color='#2E7D32')
    
    # ================== 第3层：多阶段融合 ==================
    y_level3 = 8.0
    
    # Stage 1: NA-MHC + Evolver
    stage1_x = 2
    draw_fusion_stage(ax, stage1_x, y_level3, 1, 
                     'Stage 1\nNA-MHC₁\n(Layer 2)', colors['fusion'])
    draw_evolver(ax, stage1_x+2.5, y_level3, 1, 'Evolver₁', colors['trainable'])
    
    # 箭头：Layer 2 → Stage 1
    draw_arrow(ax, hierarchical_x, stage_y, stage1_x+1, y_level3+0.5, style='solid', color='#333')
    # 箭头：Clinical → Stage 1
    draw_arrow(ax, 10.5, y_level2, stage1_x+1, y_level3+0.5, style='dashed', color='#FF6B6B')
    
    # Stage 2: NA-MHC + Evolver
    stage2_x = 5.5
    draw_fusion_stage(ax, stage2_x, y_level3, 2, 
                     'Stage 2\nNA-MHC₂\n(Layer 5)', colors['fusion'])
    draw_evolver(ax, stage2_x+2.5, y_level3, 2, 'Evolver₂', colors['trainable'])
    
    # 箭头：Layer 5 → Stage 2
    draw_arrow(ax, hierarchical_x+1.2, stage_y, stage2_x+1, y_level3+0.5, style='solid', color='#333')
    # 箭头：Evolver₁ → Stage 2
    draw_arrow(ax, stage1_x+3.5, y_level3+0.2, stage2_x+1, y_level3+0.5, style='dashed', color='#FF6B6B')
    
    # Stage 3: NA-MHC + Evolver
    stage3_x = 9
    draw_fusion_stage(ax, stage3_x, y_level3, 3, 
                     'Stage 3\nNA-MHC₃\n(Layer 8)', colors['fusion'])
    draw_evolver(ax, stage3_x+2.5, y_level3, 3, 'Evolver₃', colors['trainable'])
    
    # 箭头：Layer 8 → Stage 3
    draw_arrow(ax, hierarchical_x+2.4, stage_y, stage3_x+1, y_level3+0.5, style='solid', color='#333')
    # 箭头：Evolver₂ → Stage 3
    draw_arrow(ax, stage2_x+3.5, y_level3+0.2, stage3_x+1, y_level3+0.5, style='dashed', color='#FF6B6B')
    
    # Stage 4: NA-MHC (Final)
    stage4_x = 12.5
    draw_fusion_stage(ax, stage4_x, y_level3, 4, 
                     'Stage 4\nNA-MHC₄\n(Layer 11)', colors['fusion'])
    
    # 箭头：Layer 11 → Stage 4
    draw_arrow(ax, hierarchical_x+3.6, stage_y, stage4_x+1, y_level3+0.5, style='solid', color='#333')
    # 箭头：Evolver₃ → Stage 4
    draw_arrow(ax, stage3_x+3.5, y_level3+0.2, stage4_x+1, y_level3+0.5, style='dashed', color='#FF6B6B')
    
    # ================== 第4层：对齐与融合 ==================
    y_level4 = 5.5
    
    # Visual Notes Module (左侧)
    vn_x = 2
    draw_module(ax, vn_x, y_level4, 3, 1, 
                'Visual Notes Module\n🔥 Cross-Attention\n[B, 768]', 
                colors['trainable'])
    # 箭头从Stage 4和VLM Adapter
    draw_arrow(ax, stage4_x+1, y_level3, vn_x+1.5, y_level4+0.5, style='solid', color='#333')
    draw_arrow(ax, 16.5, y_level2-0.4, vn_x+1.5, y_level4+0.5, style='solid', color='#9C27B0')
    
    # Alignment Module (右侧上)
    align_x = 14
    draw_module(ax, align_x, y_level4+0.8, 4, 0.8, 
                'Semantic-Visual Alignment\n📐 InfoNCE Loss\nRecall@1: 25.6%', 
                colors['align'])
    # 箭头：VLM → Alignment
    draw_arrow(ax, 16.5, y_level2-0.4, align_x+2, y_level4+1.6, style='dashed', color='#9C27B0')
    # 箭头：Visual Features → Alignment
    draw_arrow(ax, stage4_x+1, y_level3, align_x+2, y_level4+1.6, style='dashed', color='#333')
    
    # Adaptive Modality Gating (右侧下)
    amcg_x = 14
    draw_module(ax, amcg_x, y_level4-0.4, 4, 0.8, 
                'Adaptive Modality Gating\n⚖️ AMCG\nw_oct ⊕ w_colpo', 
                colors['fusion'])
    # 箭头：Stage outputs → AMCG
    draw_arrow(ax, vn_x+1.5, y_level4, amcg_x, y_level4+0.2, style='solid', color='#4CAF50')
    
    # Aggregation (中央)
    agg_x = 8
    draw_module(ax, agg_x, y_level4, 2.5, 0.8, 
                'Multi-Stage\nAggregation\nΣ M₁..M₄', 
                colors['fusion'])
    # 箭头从各个stage
    for i, sx in enumerate([stage1_x, stage2_x, stage3_x, stage4_x]):
        draw_arrow(ax, sx+1, y_level3, agg_x+1.25, y_level4+0.4, 
                  style='solid', color='#4CAF50', alpha=0.6)
    
    # ================== 第5层：最终决策 ==================
    y_level5 = 3.5
    
    # Dual-Head Encoder
    dual_x = 3
    draw_module(ax, dual_x, y_level5, 3, 0.8, 
                'Dual-Head Encoder\n🔥 Causal + Contextual', 
                colors['trainable'])
    draw_arrow(ax, agg_x+1.25, y_level4, dual_x+1.5, y_level5+0.4, style='solid', color='#4CAF50')
    
    # Final Fusion
    fusion_x = 9
    draw_module(ax, fusion_x, y_level5, 3, 0.8, 
                'Final Feature Fusion\n[B, 768]', 
                colors['fusion'])
    draw_arrow(ax, dual_x+3, y_level5+0.4, fusion_x, y_level5+0.4, style='solid', color='#4CAF50')
    draw_arrow(ax, amcg_x, y_level4-0.4, fusion_x+1.5, y_level5+0.4, style='solid', color='#4CAF50')
    
    # ================== 第6层：输出与损失 ==================
    y_level6 = 1.5
    
    # Classification Head
    cls_x = 6
    draw_module(ax, cls_x, y_level6, 2.5, 0.8, 
                'Classification\nHead\n[B, 2]', colors['output'])
    draw_arrow(ax, fusion_x+1.5, y_level5, cls_x+1.25, y_level6+0.4, 
              style='solid', color='#FF9800', linewidth=2)
    
    # Center Discriminator (Adversarial)
    disc_x = 13
    draw_module(ax, disc_x, y_level6, 2.5, 0.8, 
                'Center\nDiscriminator\n(Adversarial)', colors['output'])
    draw_arrow(ax, fusion_x+1.5, y_level5, disc_x+1.25, y_level6+0.4, 
              style='dashed', color='#F44336', linewidth=1.5)
    
    # Final Output
    output_x = 9.5
    draw_module(ax, output_x, y_level6-1, 4, 0.6, 
                '📊 Output: Positive/Negative Prediction', 
                colors['output'], edgecolor='#FF9800', linewidth=3)
    draw_arrow(ax, cls_x+1.25, y_level6, output_x+2, y_level6-0.4, 
              style='solid', color='#FF9800', linewidth=2)
    
    # ================== 损失函数标注 ==================
    y_loss = 0.3
    loss_text = (
        'Multi-Task Loss: ℒ = λ_cls·ℒ_cls + λ_align·ℒ_align + λ_ot·ℒ_ot '
        '+ λ_consist·ℒ_consist + λ_adv·ℒ_adv + λ_sparse·ℒ_sparse + λ_ortho·ℒ_ortho'
    )
    ax.text(10, y_loss, loss_text, ha='center', va='center', 
            fontsize=9, style='italic', color='#D32F2F',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE', edgecolor='#D32F2F'))
    
    # ================== 图例 ==================
    legend_elements = [
        mpatches.Patch(facecolor=colors['frozen'], edgecolor='#555', linewidth=2, 
                      label='❄️ Frozen (109M params)'),
        mpatches.Patch(facecolor=colors['trainable'], edgecolor='black', 
                      label='🔥 Trainable (115M params)'),
        mpatches.Patch(facecolor=colors['fusion'], edgecolor='black', 
                      label='🔀 Fusion Modules'),
        mpatches.Patch(facecolor=colors['align'], edgecolor='black', 
                      label='📐 Alignment'),
        FancyArrowPatch((0, 0), (0.5, 0), arrowstyle='->', lw=1.5, color='#333', 
                       label='→ Feature Flow'),
        FancyArrowPatch((0, 0), (0.5, 0), arrowstyle='->', lw=1.5, color='#FF6B6B', 
                       linestyle='dashed', label='⋯ Clinical Evolution'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', 
             bbox_to_anchor=(0.01, 0.99), fontsize=9, framealpha=0.9)
    
    # ================== 标注关键创新点 ==================
    # Innovation markers
    innovations = [
        (16.5, y_level2, '💡1', 'Frozen VLM\n+ Adapter'),
        (align_x+2, y_level4+1.2, '💡2', 'Explicit\nAlignment'),
        (amcg_x+2, y_level4, '💡3', 'Adaptive\nGating'),
        (stage2_x+1, y_level3+0.2, '💡4', 'Multi-Scale\nHierarchy'),
        (stage1_x+1, y_level3+0.2, '💡5', 'NA-mHC\nFusion'),
        (stage1_x+3, y_level3+0.2, '💡6', 'Clinical\nEvolution'),
    ]
    
    for x, y, marker, text in innovations:
        circle = Circle((x, y), 0.15, color='#FFD700', ec='#FF6B00', linewidth=2, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, marker, ha='center', va='center', fontsize=8, 
               fontweight='bold', color='#D32F2F', zorder=11)
        ax.text(x+0.6, y, text, ha='left', va='center', fontsize=7, 
               color='#555', style='italic')
    
    plt.tight_layout()
    plt.savefig('Bio_COT_3.2_Architecture.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('Bio_COT_3.2_Architecture.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✅ Architecture diagram saved: Bio_COT_3.2_Architecture.png & .pdf")
    plt.show()


def draw_module(ax, x, y, width, height, text, color, edgecolor='black', linewidth=1.5):
    """绘制模块框"""
    box = FancyBboxPatch((x, y), width, height, 
                         boxstyle="round,pad=0.1", 
                         edgecolor=edgecolor, facecolor=color, 
                         linewidth=linewidth, zorder=1)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
           ha='center', va='center', fontsize=9, 
           fontweight='normal', wrap=True)


def draw_small_box(ax, x, y, width, height, text, color):
    """绘制小框（用于stage输出）"""
    rect = mpatches.Rectangle((x, y), width, height, 
                              edgecolor='#333', facecolor=color, 
                              linewidth=1, zorder=2)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, 
           ha='center', va='center', fontsize=7)


def draw_fusion_stage(ax, x, y, stage_num, text, color):
    """绘制融合stage"""
    draw_module(ax, x, y, 2, 0.8, text, color, linewidth=1.5)


def draw_evolver(ax, x, y, stage_num, text, color):
    """绘制Clinical Evolver"""
    draw_small_box(ax, x, y, 1, 0.4, text, color)


def draw_arrow(ax, x1, y1, x2, y2, style='solid', color='black', alpha=1.0, linewidth=1.5):
    """绘制箭头"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                           arrowstyle='->', lw=linewidth, 
                           color=color, alpha=alpha, 
                           linestyle=style, zorder=3,
                           mutation_scale=20)
    ax.add_patch(arrow)


def draw_simplified_architecture():
    """绘制简化版架构图（适合PPT/海报）"""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(8, 9.5, 'Bio-COT 3.2: Simplified Architecture', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    # 定义颜色
    c_input = '#E3F2FD'
    c_process = '#FFF9C4'
    c_output = '#C8E6C9'
    
    # 输入层
    draw_module(ax, 1, 7.5, 3, 1, 'OCT\n+\nColposcopy', c_input)
    draw_module(ax, 5, 7.5, 2, 1, 'Clinical\nInfo', c_input)
    draw_module(ax, 11, 7.5, 3, 1, 'VLM\nDescriptions', c_input)
    
    # 特征提取
    draw_module(ax, 1.5, 5.5, 2, 1, 'ViT\nBackbone\n❄️ Frozen', '#E0E0E0')
    draw_module(ax, 11.5, 5.5, 2, 1, 'Text Enc\n❄️ Frozen', '#E0E0E0')
    draw_arrow(ax, 2.5, 7.5, 2.5, 6.5, style='solid', color='#333', linewidth=2)
    draw_arrow(ax, 12.5, 7.5, 12.5, 6.5, style='solid', color='#333', linewidth=2)
    
    # Adapter
    draw_module(ax, 11.5, 4, 2, 0.8, 'Adapter\n🔥 Train', '#B8E6B8')
    draw_arrow(ax, 12.5, 5.5, 12.5, 4.8, style='solid', color='#2E7D32', linewidth=2)
    
    # 多阶段融合
    draw_module(ax, 1, 3, 5, 1.5, 
                'Multi-Stage Fusion\n(NA-MHC + Evolver × 4)\n🔥 Trainable', 
                c_process)
    draw_arrow(ax, 2.5, 5.5, 3.5, 4.5, style='solid', color='#333', linewidth=2)
    draw_arrow(ax, 6, 7.5, 3.5, 4.5, style='dashed', color='#FF6B6B', linewidth=2)
    
    # 对齐模块
    draw_module(ax, 8, 3.5, 3, 1, 
                'Alignment\n📐 InfoNCE', 
                '#FFEBEE')
    draw_arrow(ax, 6, 3.75, 8, 4, style='dashed', color='#9C27B0', linewidth=1.5)
    draw_arrow(ax, 12.5, 4, 11, 4, style='dashed', color='#9C27B0', linewidth=1.5)
    
    # 自适应门控
    draw_module(ax, 11.5, 3, 2, 1, 
                'AMCG\n⚖️ Gating', 
                c_process)
    draw_arrow(ax, 6, 3.75, 12.5, 4, style='solid', color='#4CAF50', linewidth=1.5)
    
    # 输出
    draw_module(ax, 6, 0.5, 4, 1, 
                '📊 Classification\nPositive / Negative', 
                c_output, edgecolor='#FF9800', linewidth=3)
    draw_arrow(ax, 3.5, 3, 8, 1.5, style='solid', color='#FF9800', linewidth=3)
    draw_arrow(ax, 12.5, 3, 8, 1.5, style='solid', color='#4CAF50', linewidth=2)
    
    # 关键创新标注
    innovations_simple = [
        (12.5, 4.5, 'Innovation 1:\nFrozen VLM'),
        (10, 4, 'Innovation 2:\nAlignment'),
        (12.5, 3.5, 'Innovation 3:\nAdaptive Gating'),
        (3.5, 3.75, 'Innovation 4-6:\nHierarchy + Fusion + Evolution'),
    ]
    
    for x, y, text in innovations_simple:
        ax.annotate(text, xy=(x, y), xytext=(x+1.5, y+0.5),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='#D32F2F'),
                   fontsize=9, color='#D32F2F', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', 
                            edgecolor='#D32F2F', linewidth=1.5))
    
    plt.tight_layout()
    plt.savefig('Bio_COT_3.2_Simplified.png', dpi=300, bbox_inches='tight', 
                facecolor='white')
    print("✅ Simplified diagram saved: Bio_COT_3.2_Simplified.png")
    plt.show()


if __name__ == '__main__':
    print("🎨 Generating Bio-COT 3.2 Architecture Diagrams...")
    print("=" * 60)
    
    # 生成完整架构图
    print("\n📐 Drawing complete architecture...")
    draw_bio_cot_architecture()
    
    # 生成简化架构图
    print("\n📐 Drawing simplified architecture...")
    draw_simplified_architecture()
    
    print("\n" + "=" * 60)
    print("✅ All diagrams generated successfully!")
    print("\nGenerated files:")
    print("  1. Bio_COT_3.2_Architecture.png (Complete, high-res)")
    print("  2. Bio_COT_3.2_Architecture.pdf (Vector format for papers)")
    print("  3. Bio_COT_3.2_Simplified.png (Simplified for presentations)")

