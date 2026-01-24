#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制自适应因果干预CLIP模型架构图（真正的CLIP方法）
展示所有创新点
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def draw_clip_architecture():
    """绘制自适应因果干预CLIP架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # 定义颜色
    colors = {
        'input': '#E8F4F8',
        'encoder': '#B3E5FC',
        'bayesian': '#81D4FA',
        'causal': '#4FC3F7',
        'fusion': '#29B6F6',
        'contrastive': '#03A9F4',
        'classifier': '#0288D1',
        'output': '#01579B',
        'innovation': '#FF6B6B'
    }
    
    # 标题
    ax.text(6, 15.5, 'Adaptive Causal Intervention CLIP Architecture', 
            ha='center', va='center', fontsize=18, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    # 1. 输入层
    input_y = 13.5
    input_width = 1.5
    input_height = 0.7
    
    oct_input = FancyBboxPatch((0.5, input_y), input_width, input_height,
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'],
                               edgecolor='black', linewidth=1.5)
    ax.add_patch(oct_input)
    ax.text(1.25, input_y + input_height/2, 'OCT\nImages', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    col_input = FancyBboxPatch((2.5, input_y), input_width, input_height,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['input'],
                               edgecolor='black', linewidth=1.5)
    ax.add_patch(col_input)
    ax.text(3.25, input_y + input_height/2, 'Colposcopy\nImages',
            ha='center', va='center', fontsize=9, weight='bold')
    
    clinical_input = FancyBboxPatch((4.5, input_y), input_width, input_height,
                                     boxstyle="round,pad=0.1",
                                     facecolor=colors['input'],
                                     edgecolor='black', linewidth=1.5)
    ax.add_patch(clinical_input)
    ax.text(5.25, input_y + input_height/2, 'Clinical\nFeatures',
            ha='center', va='center', fontsize=9, weight='bold')
    
    # 2. 特征提取器（Swin-T）
    encoder_y = 11.5
    encoder_width = 1.5
    encoder_height = 1.0
    
    oct_encoder = FancyBboxPatch((0.5, encoder_y), encoder_width, encoder_height,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['encoder'],
                                 edgecolor='black', linewidth=1.5)
    ax.add_patch(oct_encoder)
    ax.text(1.25, encoder_y + encoder_height/2, 
            'OCT Encoder\n(Swin-T)',
            ha='center', va='center', fontsize=8, weight='bold')
    
    col_encoder = FancyBboxPatch((2.5, encoder_y), encoder_width, encoder_height,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['encoder'],
                                  edgecolor='black', linewidth=1.5)
    ax.add_patch(col_encoder)
    ax.text(3.25, encoder_y + encoder_height/2,
            'Col Encoder\n(Swin-T)',
            ha='center', va='center', fontsize=8, weight='bold')
    
    clinical_encoder = FancyBboxPatch((4.5, encoder_y), encoder_width, encoder_height,
                                       boxstyle="round,pad=0.1",
                                       facecolor=colors['encoder'],
                                       edgecolor='black', linewidth=1.5)
    ax.add_patch(clinical_encoder)
    ax.text(5.25, encoder_y + encoder_height/2,
            'Clinical\nEncoder (MLP)',
            ha='center', va='center', fontsize=8, weight='bold')
    
    # 输入到编码器的箭头
    arrow_props = dict(arrowstyle='->', lw=2, color='#1976D2')
    for x in [1.25, 3.25, 5.25]:
        ax.annotate('', xy=(x, encoder_y + encoder_height), xytext=(x, input_y),
                    arrowprops=arrow_props)
    
    # 3. 创新点1: 贝叶斯编码器（不确定性量化）
    bayesian_y = 9.5
    bayesian_width = 1.5
    bayesian_height = 1.0
    
    oct_bayesian = FancyBboxPatch((0.5, bayesian_y), bayesian_width, bayesian_height,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['bayesian'],
                                  edgecolor='red', linewidth=2)
    ax.add_patch(oct_bayesian)
    ax.text(1.25, bayesian_y + bayesian_height/2,
            'Bayesian\nEncoder\n(μ, σ²)',
            ha='center', va='center', fontsize=8, weight='bold')
    
    col_bayesian = FancyBboxPatch((2.5, bayesian_y), bayesian_width, bayesian_height,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['bayesian'],
                                   edgecolor='red', linewidth=2)
    ax.add_patch(col_bayesian)
    ax.text(3.25, bayesian_y + bayesian_height/2,
            'Bayesian\nEncoder\n(μ, σ²)',
            ha='center', va='center', fontsize=8, weight='bold')
    
    clinical_bayesian = FancyBboxPatch((4.5, bayesian_y), bayesian_width, bayesian_height,
                                        boxstyle="round,pad=0.1",
                                        facecolor=colors['bayesian'],
                                        edgecolor='red', linewidth=2)
    ax.add_patch(clinical_bayesian)
    ax.text(5.25, bayesian_y + bayesian_height/2,
            'Bayesian\nEncoder\n(μ, σ²)',
            ha='center', va='center', fontsize=8, weight='bold')
    
    # 创新点标注
    ax.text(6.5, bayesian_y + bayesian_height/2, 
            'INNOVATION 1:\nUncertainty Quantification\n(Bayesian Encoding)',
            ha='left', va='center', fontsize=9, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['innovation'], alpha=0.3))
    
    # 编码器到贝叶斯的箭头
    for x in [1.25, 3.25, 5.25]:
        ax.annotate('', xy=(x, bayesian_y + bayesian_height), xytext=(x, encoder_y),
                    arrowprops=arrow_props)
    
    # 4. 创新点2: 自适应因果图
    causal_graph_y = 7.5
    causal_graph_width = 2.5
    causal_graph_height = 1.0
    
    causal_graph = FancyBboxPatch((1.0, causal_graph_y), causal_graph_width, causal_graph_height,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['causal'],
                                  edgecolor='red', linewidth=2)
    ax.add_patch(causal_graph)
    ax.text(2.25, causal_graph_y + causal_graph_height/2,
            'Adaptive Causal Graph\nLearning Causal Relationships',
            ha='center', va='center', fontsize=9, weight='bold')
    
    ax.text(4.5, causal_graph_y + causal_graph_height/2,
            'INNOVATION 2:\nAdaptive Causal Graph\n(Learnable Causal Structure)',
            ha='left', va='center', fontsize=9, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['innovation'], alpha=0.3))
    
    # 贝叶斯到因果图的箭头
    ax.annotate('', xy=(2.25, causal_graph_y + causal_graph_height), 
                xytext=(2.25, bayesian_y),
                arrowprops=arrow_props)
    
    # 5. 创新点3: 因果干预
    intervention_y = 5.5
    intervention_width = 2.5
    intervention_height = 1.0
    
    intervention = FancyBboxPatch((1.0, intervention_y), intervention_width, intervention_height,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['causal'],
                                  edgecolor='red', linewidth=2)
    ax.add_patch(intervention)
    ax.text(2.25, intervention_y + intervention_height/2,
            'Causal Intervention\nDo(X=x) Operation',
            ha='center', va='center', fontsize=9, weight='bold')
    
    ax.text(4.5, intervention_y + intervention_height/2,
            'INNOVATION 3:\nCausal Intervention\n(Do-Calculus)',
            ha='left', va='center', fontsize=9, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['innovation'], alpha=0.3))
    
    # 因果图到干预的箭头
    ax.annotate('', xy=(2.25, intervention_y + intervention_height), 
                xytext=(2.25, causal_graph_y),
                arrowprops=arrow_props)
    
    # 6. 跨模态融合
    fusion_y = 3.5
    fusion_width = 5.0
    fusion_height = 1.0
    
    fusion = FancyBboxPatch((1.0, fusion_y), fusion_width, fusion_height,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['fusion'],
                            edgecolor='black', linewidth=1.5)
    ax.add_patch(fusion)
    ax.text(3.5, fusion_y + fusion_height/2,
            'Cross-Modal Fusion\n(Multi-Head Attention)',
            ha='center', va='center', fontsize=10, weight='bold')
    
    # 干预到融合的箭头
    ax.annotate('', xy=(3.5, fusion_y + fusion_height), 
                xytext=(2.25, intervention_y),
                arrowprops=arrow_props)
    
    # 7. 创新点4: 分层对比学习（CLIP核心）
    contrastive_y = 1.5
    contrastive_width = 5.0
    contrastive_height = 1.0
    
    contrastive = FancyBboxPatch((1.0, contrastive_y), contrastive_width, contrastive_height,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['contrastive'],
                                  edgecolor='red', linewidth=2)
    ax.add_patch(contrastive)
    ax.text(3.5, contrastive_y + contrastive_height/2,
            'Hierarchical Contrastive Learning (CLIP)\nInfoNCE Loss: Align Modalities',
            ha='center', va='center', fontsize=9, weight='bold')
    
    ax.text(7.0, contrastive_y + contrastive_height/2,
            'INNOVATION 4:\nHierarchical CLIP\n(Multi-level Alignment)',
            ha='left', va='center', fontsize=9, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['innovation'], alpha=0.3))
    
    # 融合到对比学习的箭头
    ax.annotate('', xy=(3.5, contrastive_y + contrastive_height), 
                xytext=(3.5, fusion_y),
                arrowprops=arrow_props)
    
    # 8. 分类器
    classifier_y = 0.2
    classifier_width = 2.0
    classifier_height = 0.8
    
    classifier = FancyBboxPatch((2.5, classifier_y), classifier_width, classifier_height,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['classifier'],
                                edgecolor='black', linewidth=2)
    ax.add_patch(classifier)
    ax.text(3.5, classifier_y + classifier_height/2, 'Classifier\n[B, 2]',
            ha='center', va='center', fontsize=10, weight='bold')
    
    # 对比学习到分类器的箭头
    ax.annotate('', xy=(3.5, classifier_y + classifier_height), 
                xytext=(3.5, contrastive_y),
                arrowprops=arrow_props)
    
    # 9. 损失函数说明（右侧）
    loss_x = 7.5
    loss_y = 12.0
    
    loss_text = [
        'Loss Function:',
        'L_total = L_cls + λ_kl·L_KL +',
        '          λ_contrast·L_CLIP +',
        '          λ_interv·L_intervention',
        '',
        'Components:',
        '• L_cls: Focal Loss (γ=3.0)',
        '• L_KL: KL Divergence',
        '• L_CLIP: InfoNCE Loss',
        '• L_intervention: Causal Loss',
        '',
        'Key Innovations:',
        '1. Bayesian Uncertainty',
        '2. Causal Graph Learning',
        '3. Causal Intervention',
        '4. Hierarchical CLIP'
    ]
    
    for i, text in enumerate(loss_text):
        ax.text(loss_x, loss_y - i*0.35, text,
                ha='left', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.5) if i == 0 else None)
    
    plt.tight_layout()
    return fig

def main():
    """主函数"""
    print("🎨 正在绘制自适应因果干预CLIP架构图...")
    fig = draw_clip_architecture()
    
    # 保存图片
    output_path = '/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/clip_architecture.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ CLIP架构图已保存至: {output_path}")
    
    # PDF版本
    pdf_path = '/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/clip_architecture.pdf'
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ PDF版本已保存至: {pdf_path}")
    
    plt.close(fig)
    print("🎉 完成！")

if __name__ == '__main__':
    main()

