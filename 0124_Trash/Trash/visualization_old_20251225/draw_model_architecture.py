#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制优化的Swin-T多模态模型架构图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def draw_model_architecture():
    """绘制模型架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # 定义颜色
    colors = {
        'input': '#E8F4F8',
        'encoder': '#B3E5FC',
        'fusion': '#81D4FA',
        'classifier': '#4FC3F7',
        'output': '#29B6F6',
        'arrow': '#1976D2'
    }
    
    # 1. 输入层
    input_y = 12.5
    input_width = 1.8
    input_height = 0.8
    
    # OCT输入
    oct_box = FancyBboxPatch((0.5, input_y), input_width, input_height,
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['input'],
                             edgecolor='black', linewidth=1.5)
    ax.add_patch(oct_box)
    ax.text(1.4, input_y + input_height/2, 'OCT图像\n[B, T, C, H, W]', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Colposcopy输入
    col_box = FancyBboxPatch((4.1, input_y), input_width, input_height,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['input'],
                             edgecolor='black', linewidth=1.5)
    ax.add_patch(col_box)
    ax.text(5.0, input_y + input_height/2, 'Colposcopy图像\n[B, T, C, H, W]',
            ha='center', va='center', fontsize=10, weight='bold')
    
    # 临床特征输入
    clinical_box = FancyBboxPatch((7.7, input_y), input_width, input_height,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['input'],
                                  edgecolor='black', linewidth=1.5)
    ax.add_patch(clinical_box)
    ax.text(8.6, input_y + input_height/2, '临床特征\n[B, 7]',
            ha='center', va='center', fontsize=10, weight='bold')
    
    # 2. 编码器层
    encoder_y = 10.0
    encoder_width = 1.8
    encoder_height = 1.2
    
    # OCT编码器
    oct_encoder_box = FancyBboxPatch((0.5, encoder_y), encoder_width, encoder_height,
                                     boxstyle="round,pad=0.1",
                                     facecolor=colors['encoder'],
                                     edgecolor='black', linewidth=1.5)
    ax.add_patch(oct_encoder_box)
    ax.text(1.4, encoder_y + encoder_height/2, 
            'OCT编码器\n(Swin-T)\n→ [B, 768]',
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Colposcopy编码器
    col_encoder_box = FancyBboxPatch((4.1, encoder_y), encoder_width, encoder_height,
                                     boxstyle="round,pad=0.1",
                                     facecolor=colors['encoder'],
                                     edgecolor='black', linewidth=1.5)
    ax.add_patch(col_encoder_box)
    ax.text(5.0, encoder_y + encoder_height/2,
            'Colposcopy编码器\n(Swin-T)\n→ [B, 768]',
            ha='center', va='center', fontsize=9, weight='bold')
    
    # 临床特征编码器
    clinical_encoder_box = FancyBboxPatch((7.7, encoder_y), encoder_width, encoder_height,
                                          boxstyle="round,pad=0.1",
                                          facecolor=colors['encoder'],
                                          edgecolor='black', linewidth=1.5)
    ax.add_patch(clinical_encoder_box)
    ax.text(8.6, encoder_y + encoder_height/2,
            '临床特征编码器\n(MLP)\n→ [B, 768]',
            ha='center', va='center', fontsize=9, weight='bold')
    
    # 3. 输入到编码器的箭头
    arrow_props = dict(arrowstyle='->', lw=2, color=colors['arrow'])
    ax.annotate('', xy=(1.4, encoder_y + encoder_height), xytext=(1.4, input_y),
                arrowprops=arrow_props)
    ax.annotate('', xy=(5.0, encoder_y + encoder_height), xytext=(5.0, input_y),
                arrowprops=arrow_props)
    ax.annotate('', xy=(8.6, encoder_y + encoder_height), xytext=(8.6, input_y),
                arrowprops=arrow_props)
    
    # 4. 跨模态融合层（详细展开）
    fusion_y = 6.5
    fusion_width = 7.0
    fusion_height = 2.5
    
    fusion_box = FancyBboxPatch((1.5, fusion_y), fusion_width, fusion_height,
                                boxstyle="round,pad=0.15",
                                facecolor=colors['fusion'],
                                edgecolor='black', linewidth=2)
    ax.add_patch(fusion_box)
    
    # 融合层标题
    ax.text(5.0, fusion_y + fusion_height - 0.2, '跨模态融合层 (CrossModalFusion)',
            ha='center', va='center', fontsize=12, weight='bold')
    
    # 融合步骤
    step_y = fusion_y + fusion_height - 0.6
    step_height = 0.35
    step_width = 6.5
    
    # 步骤1: 堆叠tokens
    step1_box = FancyBboxPatch((1.7, step_y), step_width, step_height,
                               boxstyle="round,pad=0.05",
                               facecolor='white',
                               edgecolor='gray', linewidth=1)
    ax.add_patch(step1_box)
    ax.text(5.0, step_y + step_height/2, '1. 堆叠tokens: [OCT, COL, Clinical] → [B, 3, 768]',
            ha='center', va='center', fontsize=8)
    
    # 步骤2: 自注意力
    step_y -= 0.45
    step2_box = FancyBboxPatch((1.7, step_y), step_width, step_height,
                               boxstyle="round,pad=0.05",
                               facecolor='white',
                               edgecolor='gray', linewidth=1)
    ax.add_patch(step2_box)
    ax.text(5.0, step_y + step_height/2, '2. 自注意力 (Self-Attention): 所有tokens相互关注',
            ha='center', va='center', fontsize=8)
    
    # 步骤3: OCT-COL交叉注意力
    step_y -= 0.45
    step3_box = FancyBboxPatch((1.7, step_y), step_width, step_height,
                               boxstyle="round,pad=0.05",
                               facecolor='white',
                               edgecolor='gray', linewidth=1)
    ax.add_patch(step3_box)
    ax.text(5.0, step_y + step_height/2, '3. OCT-COL交叉注意力: 图像模态间交互',
            ha='center', va='center', fontsize=8)
    
    # 步骤4: 图像-临床交叉注意力
    step_y -= 0.45
    step4_box = FancyBboxPatch((1.7, step_y), step_width, step_height,
                               boxstyle="round,pad=0.05",
                               facecolor='white',
                               edgecolor='gray', linewidth=1)
    ax.add_patch(step4_box)
    ax.text(5.0, step_y + step_height/2, '4. 图像-临床交叉注意力: 图像与临床信息交互',
            ha='center', va='center', fontsize=8)
    
    # 步骤5: 前馈网络
    step_y -= 0.45
    step5_box = FancyBboxPatch((1.7, step_y), step_width, step_height,
                               boxstyle="round,pad=0.05",
                               facecolor='white',
                               edgecolor='gray', linewidth=1)
    ax.add_patch(step5_box)
    ax.text(5.0, step_y + step_height/2, '5. 前馈网络 (FFN): GELU + Dropout',
            ha='center', va='center', fontsize=8)
    
    # 步骤6: 自适应融合
    step_y -= 0.45
    step6_box = FancyBboxPatch((1.7, step_y), step_width, step_height,
                               boxstyle="round,pad=0.05",
                               facecolor='white',
                               edgecolor='gray', linewidth=1)
    ax.add_patch(step6_box)
    ax.text(5.0, step_y + step_height/2, '6. 自适应权重融合: [B, 3×768] → [B, 768]',
            ha='center', va='center', fontsize=8)
    
    # 编码器到融合层的箭头
    ax.annotate('', xy=(5.0, fusion_y + fusion_height), xytext=(1.4, encoder_y),
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['arrow']))
    ax.annotate('', xy=(5.0, fusion_y + fusion_height), xytext=(5.0, encoder_y),
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['arrow']))
    ax.annotate('', xy=(5.0, fusion_y + fusion_height), xytext=(8.6, encoder_y),
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['arrow']))
    
    # 5. 分类器层
    classifier_y = 3.5
    classifier_width = 7.0
    classifier_height = 1.0
    
    classifier_box = FancyBboxPatch((1.5, classifier_y), classifier_width, classifier_height,
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['classifier'],
                                    edgecolor='black', linewidth=2)
    ax.add_patch(classifier_box)
    
    classifier_text = '分类器: Linear(768→384) → LayerNorm → GELU → Dropout(0.25) → Linear(384→2)'
    ax.text(5.0, classifier_y + classifier_height/2, classifier_text,
            ha='center', va='center', fontsize=10, weight='bold')
    
    # 融合层到分类器的箭头
    ax.annotate('', xy=(5.0, classifier_y + classifier_height), xytext=(5.0, fusion_y),
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['arrow']))
    
    # 6. 输出层
    output_y = 2.0
    output_width = 2.0
    output_height = 0.8
    
    output_box = FancyBboxPatch((4.0, output_y), output_width, output_height,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['output'],
                                edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5.0, output_y + output_height/2, 'Logits\n[B, 2]',
            ha='center', va='center', fontsize=11, weight='bold')
    
    # 分类器到输出的箭头
    ax.annotate('', xy=(5.0, output_y + output_height), xytext=(5.0, classifier_y),
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['arrow']))
    
    # 7. 训练配置信息（右侧）
    config_x = 9.0
    config_y = 11.0
    
    config_text = [
        '训练配置:',
        '• 学习率: 2.1e-5',
        '• Batch Size: 5',
        '• 输入尺寸: 192×192',
        '• OCT帧数: 48',
        '• Loss: Focal Loss (γ=3.0)',
        '• Label Smoothing: 0.1',
        '• Weight Decay: 6e-4',
        '• Dropout: 0.25',
        '• Optimizer: AdamW',
        '• Scheduler: Warmup + Cosine',
        '',
        '模型参数:',
        '• Embed Dim: 768',
        '• Num Heads: 8',
        '• 总参数量: 70.11M'
    ]
    
    for i, text in enumerate(config_text):
        ax.text(config_x, config_y - i*0.3, text,
                ha='left', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.5) if i == 0 else None)
    
    # 8. 添加标题
    ax.text(5.0, 13.5, '优化的Swin-T多模态宫颈癌筛查模型架构',
            ha='center', va='center', fontsize=16, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    return fig

def main():
    """主函数"""
    print("🎨 正在绘制模型架构图...")
    fig = draw_model_architecture()
    
    # 保存图片
    output_path = '/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/model_architecture.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 模型架构图已保存至: {output_path}")
    
    # 同时保存PDF版本（适合论文）
    pdf_path = '/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/model_architecture.pdf'
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ PDF版本已保存至: {pdf_path}")
    
    plt.close(fig)
    print("🎉 完成！")

if __name__ == '__main__':
    main()

