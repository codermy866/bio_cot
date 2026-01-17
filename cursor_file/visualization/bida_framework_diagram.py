#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BIDA框架图生成脚本
生成PDF格式的框架图，展示分布锚定机制和先验约束的域不变性学习
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_bida_framework_diagram():
    """创建BIDA框架图"""
    
    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 定义颜色
    colors = {
        'input': '#E8F4F8',
        'branch_a': '#FFE5E5',
        'branch_b': '#E5F5E5',
        'constraint': '#FFF5E5',
        'output': '#F0E8FF',
        'arrow': '#333333',
        'text': '#000000'
    }
    
    # ==================== 输入层 ====================
    # OCT图像
    oct_box = FancyBboxPatch((0.5, 9.5), 2, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['input'], 
                             edgecolor='black', linewidth=2)
    ax.add_patch(oct_box)
    ax.text(1.5, 10.5, 'OCT图像', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(1.5, 10.0, '[B, C, H, W]', ha='center', va='center', fontsize=9)
    
    # Colposcopy图像
    col_box = FancyBboxPatch((3.5, 9.5), 2, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['input'], 
                             edgecolor='black', linewidth=2)
    ax.add_patch(col_box)
    ax.text(4.5, 10.5, 'Colposcopy图像', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(4.5, 10.0, '[B, C, H, W]', ha='center', va='center', fontsize=9)
    
    # 临床数据
    clinical_box = FancyBboxPatch((6.5, 9.5), 2.5, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['input'], 
                                  edgecolor='black', linewidth=2)
    ax.add_patch(clinical_box)
    ax.text(7.75, 10.5, '临床数据', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(7.75, 10.0, 'HPV, TCT, Age', ha='center', va='center', fontsize=9)
    ax.text(7.75, 9.7, '(先验知识)', ha='center', va='center', fontsize=8, style='italic', color='blue')
    
    # 中心标签
    center_box = FancyBboxPatch((10, 9.5), 2, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['input'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(center_box)
    ax.text(11, 10.5, '中心标签', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(11, 10.0, 'Center ID', ha='center', va='center', fontsize=9)
    ax.text(11, 9.7, '[0-4]', ha='center', va='center', fontsize=9)
    
    # ==================== Branch A: Distributional Anchor ====================
    # Branch A标题
    ax.text(2, 8.5, 'Branch A: Distributional Anchor (分布锚定机制)', 
            ha='left', va='center', fontsize=12, weight='bold', color='red')
    
    # 临床数据 → 文本描述
    text_box = FancyBboxPatch((0.5, 7), 3, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['branch_a'], 
                              edgecolor='red', linewidth=2)
    ax.add_patch(text_box)
    ax.text(2, 7.5, '临床数据 → 文本描述', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(2, 7.2, 'clinical_to_text()', ha='center', va='center', fontsize=8, style='italic')
    
    # VLM处理
    vlm_box = FancyBboxPatch((4.5, 7), 3.5, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['branch_a'], 
                             edgecolor='red', linewidth=2)
    ax.add_patch(vlm_box)
    ax.text(6.25, 7.5, 'VLM处理图像+文本', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(6.25, 7.2, 'Qwen2-VL (Frozen)', ha='center', va='center', fontsize=8, style='italic')
    
    # 分布参数生成
    dist_box = FancyBboxPatch((9, 7), 3, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['branch_a'], 
                              edgecolor='red', linewidth=2)
    ax.add_patch(dist_box)
    ax.text(10.5, 7.5, '分布参数生成', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(10.5, 7.2, 'distribution_head()', ha='center', va='center', fontsize=8, style='italic')
    
    # 生物流形分布
    bio_dist_box = FancyBboxPatch((5, 5.5), 6, 1, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#FFD700', 
                                  edgecolor='red', linewidth=3)
    ax.add_patch(bio_dist_box)
    ax.text(8, 6.0, '生物流形分布', ha='center', va='center', fontsize=12, weight='bold', color='red')
    ax.text(8, 5.7, 'P_bio = N(μ_bio, σ_bio)', ha='center', va='center', fontsize=10, style='italic')
    ax.text(6.5, 5.4, 'μ_bio [B, 768]', ha='center', va='center', fontsize=9)
    ax.text(9.5, 5.4, 'σ_bio [B, 768]', ha='center', va='center', fontsize=9)
    
    # ==================== Branch B: Dual Head Image Encoder ====================
    # Branch B标题
    ax.text(2, 4.5, 'Branch B: Dual Head Image Encoder (双头图像编码器)', 
            ha='left', va='center', fontsize=12, weight='bold', color='green')
    
    # 图像特征提取
    img_feat_box = FancyBboxPatch((0.5, 3), 3, 1, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['branch_b'], 
                                  edgecolor='green', linewidth=2)
    ax.add_patch(img_feat_box)
    ax.text(2, 3.5, '图像特征提取', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(2, 3.2, 'ResNet50 → [B, 512]', ha='center', va='center', fontsize=8, style='italic')
    
    # 双头网络
    dual_head_box = FancyBboxPatch((4.5, 2.5), 7, 2, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['branch_b'], 
                                   edgecolor='green', linewidth=2)
    ax.add_patch(dual_head_box)
    ax.text(8, 4.0, '双头网络', ha='center', va='center', fontsize=11, weight='bold')
    
    # Head 1: z_causal
    causal_head_box = FancyBboxPatch((5, 3), 2.5, 0.8, 
                                     boxstyle="round,pad=0.05", 
                                     facecolor='#90EE90', 
                                     edgecolor='green', linewidth=2)
    ax.add_patch(causal_head_box)
    ax.text(6.25, 3.4, 'Head 1: z_causal', ha='center', va='center', fontsize=9, weight='bold')
    ax.text(6.25, 3.1, '[B, 768] 因果特征', ha='center', va='center', fontsize=8)
    
    # Head 2: z_noise
    noise_head_box = FancyBboxPatch((8.5, 3), 2.5, 0.8, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor='#FFB6C1', 
                                    edgecolor='green', linewidth=2)
    ax.add_patch(noise_head_box)
    ax.text(9.75, 3.4, 'Head 2: z_noise', ha='center', va='center', fontsize=9, weight='bold')
    ax.text(9.75, 3.1, '[B, 768] 噪声特征', ha='center', va='center', fontsize=8)
    
    # ==================== 约束机制 ====================
    # 约束标题
    ax.text(2, 1.8, '约束机制 (Constraint Mechanisms)', 
            ha='left', va='center', fontsize=12, weight='bold', color='orange')
    
    # Constraint 1: Distribution Matching
    constraint1_box = FancyBboxPatch((0.5, 0.5), 4.5, 1, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=colors['constraint'], 
                                     edgecolor='orange', linewidth=2)
    ax.add_patch(constraint1_box)
    ax.text(2.75, 1.0, 'Constraint 1: 分布锚定', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(2.75, 0.7, 'L_dist = D_KL(Q(z_causal) || P_bio)', ha='center', va='center', fontsize=9, style='italic')
    ax.text(2.75, 0.4, 'z_causal 必须在 N(μ_bio, σ_bio) 内', ha='center', va='center', fontsize=8)
    
    # Constraint 2: Orthogonal Disentanglement
    constraint2_box = FancyBboxPatch((5.5, 0.5), 4.5, 1, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=colors['constraint'], 
                                     edgecolor='orange', linewidth=2)
    ax.add_patch(constraint2_box)
    ax.text(7.75, 1.0, 'Constraint 2: 正交解耦', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(7.75, 0.7, 'L_orth = |z_causal^T · z_noise|', ha='center', va='center', fontsize=9, style='italic')
    ax.text(7.75, 0.4, 'z_causal ⊥ z_noise', ha='center', va='center', fontsize=8)
    
    # Constraint 3: Noise Supervision
    constraint3_box = FancyBboxPatch((10.5, 0.5), 4.5, 1, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=colors['constraint'], 
                                     edgecolor='orange', linewidth=2)
    ax.add_patch(constraint3_box)
    ax.text(12.75, 1.0, 'Constraint 3: 噪声监督', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(12.75, 0.7, 'L_noise = CE(CenterPred(z_noise), d)', ha='center', va='center', fontsize=9, style='italic')
    ax.text(12.75, 0.4, 'z_noise → Center ID', ha='center', va='center', fontsize=8)
    
    # ==================== 输出层 ====================
    output_box = FancyBboxPatch((13.5, 2.5), 2, 2, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['output'], 
                                edgecolor='purple', linewidth=3)
    ax.add_patch(output_box)
    ax.text(14.5, 4.0, '输出', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(14.5, 3.5, 'Classifier', ha='center', va='center', fontsize=10)
    ax.text(14.5, 3.0, 'z_causal →', ha='center', va='center', fontsize=9)
    ax.text(14.5, 2.7, 'Diagnosis', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(14.5, 2.4, '[B, num_classes]', ha='center', va='center', fontsize=8)
    
    # ==================== 箭头连接 ====================
    # 输入 → Branch A
    arrow1 = FancyArrowPatch((7.75, 9.5), (2, 7.5), 
                            arrowstyle='->', lw=2, color='red', 
                            connectionstyle="arc3,rad=0.2")
    ax.add_patch(arrow1)
    
    arrow1_img = FancyArrowPatch((1.5, 9.5), (6.25, 7.5), 
                                 arrowstyle='->', lw=2, color='red', 
                                 connectionstyle="arc3,rad=-0.2")
    ax.add_patch(arrow1_img)
    
    # Branch A → 生物流形分布
    arrow2 = FancyArrowPatch((10.5, 7), (8, 6.5), 
                            arrowstyle='->', lw=2, color='red')
    ax.add_patch(arrow2)
    
    # 输入 → Branch B
    arrow3 = FancyArrowPatch((1.5, 9.5), (2, 3.5), 
                            arrowstyle='->', lw=2, color='green', 
                            connectionstyle="arc3,rad=0.3")
    ax.add_patch(arrow3)
    
    arrow4 = FancyArrowPatch((4.5, 9.5), (2, 3.5), 
                            arrowstyle='->', lw=2, color='green', 
                            connectionstyle="arc3,rad=0.3")
    ax.add_patch(arrow4)
    
    # Branch B → 双头网络
    arrow5 = FancyArrowPatch((3.5, 3.5), (5, 3.4), 
                            arrowstyle='->', lw=2, color='green')
    ax.add_patch(arrow5)
    
    arrow6 = FancyArrowPatch((3.5, 3.5), (8.5, 3.4), 
                            arrowstyle='->', lw=2, color='green')
    ax.add_patch(arrow6)
    
    # 生物流形分布 → Constraint 1
    arrow7 = FancyArrowPatch((8, 5.5), (2.75, 1.5), 
                            arrowstyle='->', lw=2, color='orange', 
                            connectionstyle="arc3,rad=0.3")
    ax.add_patch(arrow7)
    
    # z_causal → Constraint 1
    arrow8 = FancyArrowPatch((6.25, 3), (2.75, 1.5), 
                            arrowstyle='->', lw=2, color='orange', 
                            connectionstyle="arc3,rad=0.2")
    ax.add_patch(arrow8)
    
    # z_causal → Constraint 2
    arrow9 = FancyArrowPatch((6.25, 3), (7.75, 1.5), 
                            arrowstyle='->', lw=2, color='orange', 
                            connectionstyle="arc3,rad=0.2")
    ax.add_patch(arrow9)
    
    # z_noise → Constraint 2
    arrow10 = FancyArrowPatch((9.75, 3), (7.75, 1.5), 
                             arrowstyle='->', lw=2, color='orange', 
                             connectionstyle="arc3,rad=-0.2")
    ax.add_patch(arrow10)
    
    # z_noise → Constraint 3
    arrow11 = FancyArrowPatch((9.75, 3), (12.75, 1.5), 
                             arrowstyle='->', lw=2, color='orange', 
                             connectionstyle="arc3,rad=-0.2")
    ax.add_patch(arrow11)
    
    # Center ID → Constraint 3
    arrow12 = FancyArrowPatch((11, 9.5), (12.75, 1.5), 
                             arrowstyle='->', lw=2, color='orange', 
                             connectionstyle="arc3,rad=-0.4")
    ax.add_patch(arrow12)
    
    # z_causal → 输出
    arrow13 = FancyArrowPatch((7.5, 3.4), (13.5, 3.5), 
                             arrowstyle='->', lw=3, color='purple')
    ax.add_patch(arrow13)
    
    # ==================== 损失函数 ====================
    loss_box = FancyBboxPatch((0.5, 11), 15, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#F0F0F0', 
                              edgecolor='black', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(8, 11.4, '总损失函数', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(8, 11.1, 'L = L_cls + λ_KL·L_dist + λ_orth·L_orth + λ_adv·L_noise', 
            ha='center', va='center', fontsize=10, style='italic')
    ax.text(3, 10.8, 'λ_KL = 0.005', ha='center', va='center', fontsize=9)
    ax.text(8, 10.8, 'λ_orth = 0.01', ha='center', va='center', fontsize=9)
    ax.text(13, 10.8, 'λ_adv = 0.05', ha='center', va='center', fontsize=9)
    
    # ==================== 图例 ====================
    legend_elements = [
        mpatches.Patch(facecolor=colors['input'], edgecolor='black', label='输入 (Input)'),
        mpatches.Patch(facecolor=colors['branch_a'], edgecolor='red', label='Branch A: 分布锚定机制'),
        mpatches.Patch(facecolor=colors['branch_b'], edgecolor='green', label='Branch B: 双头图像编码器'),
        mpatches.Patch(facecolor=colors['constraint'], edgecolor='orange', label='约束机制 (Constraints)'),
        mpatches.Patch(facecolor=colors['output'], edgecolor='purple', label='输出 (Output)'),
        mpatches.Patch(facecolor='#FFD700', edgecolor='red', label='生物流形分布 (Bio-Invariant Distribution)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    
    # ==================== 标题 ====================
    ax.text(8, 11.8, 'BIDA Framework: Distributional Anchoring for Zero-Shot Cross-Center Generalization', 
            ha='center', va='center', fontsize=14, weight='bold')
    
    plt.tight_layout()
    return fig

def create_innovation_points_diagram():
    """创建创新点说明图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('BIDA Framework: 核心创新点详解', fontsize=16, weight='bold', y=0.98)
    
    # 创新点1: 分布锚定机制
    ax1 = axes[0, 0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    ax1.text(5, 9, '创新点1: 分布锚定机制', ha='center', va='center', 
            fontsize=14, weight='bold', color='red')
    
    # 先验约束
    prior_box = FancyBboxPatch((1, 7), 3, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#FFE5E5', 
                              edgecolor='red', linewidth=2)
    ax1.add_patch(prior_box)
    ax1.text(2.5, 8, '先验约束', ha='center', va='center', fontsize=11, weight='bold')
    ax1.text(2.5, 7.5, '临床数据 (HPV/TCT)', ha='center', va='center', fontsize=9)
    ax1.text(2.5, 7.1, '→ 生物流形分布', ha='center', va='center', fontsize=9)
    
    # 分布锚定
    anchor_box = FancyBboxPatch((6, 7), 3, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#E5F5E5', 
                               edgecolor='green', linewidth=2)
    ax1.add_patch(anchor_box)
    ax1.text(7.5, 8, '分布锚定', ha='center', va='center', fontsize=11, weight='bold')
    ax1.text(7.5, 7.5, 'z_causal → N(μ_bio, σ_bio)', ha='center', va='center', fontsize=9)
    ax1.text(7.5, 7.1, 'KL散度约束', ha='center', va='center', fontsize=9)
    
    # 箭头
    arrow = FancyArrowPatch((4, 7.75), (6, 7.75), 
                           arrowstyle='->', lw=2, color='black')
    ax1.add_patch(arrow)
    
    # 说明
    ax1.text(5, 5.5, '核心思想:', ha='center', va='center', fontsize=11, weight='bold')
    ax1.text(5, 4.5, '使用临床模态定义分布 N(μ_bio, σ_bio)', ha='center', va='center', fontsize=10)
    ax1.text(5, 4, '约束图像特征 z_causal 必须落在这个分布内', ha='center', va='center', fontsize=10)
    ax1.text(5, 3, '允许对齐误差，提高泛化能力', ha='center', va='center', fontsize=10, style='italic')
    
    # 代码位置
    code_box = FancyBboxPatch((1, 1), 8, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#F0F0F0', 
                              edgecolor='gray', linewidth=1)
    ax1.add_patch(code_box)
    ax1.text(5, 2, '代码位置:', ha='center', va='center', fontsize=10, weight='bold')
    ax1.text(2.5, 1.5, 'DistributionalAnchor.forward()', ha='center', va='center', fontsize=8)
    ax1.text(7.5, 1.5, 'DistributionMatchingLoss.forward()', ha='center', va='center', fontsize=8)
    
    # 创新点2: 正交解耦机制
    ax2 = axes[0, 1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    ax2.text(5, 9, '创新点2: 正交解耦机制', ha='center', va='center', 
            fontsize=14, weight='bold', color='green')
    
    # 双头网络
    dual_box = FancyBboxPatch((2, 7), 6, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#E5F5E5', 
                              edgecolor='green', linewidth=2)
    ax2.add_patch(dual_box)
    ax2.text(5, 8, '双头图像编码器', ha='center', va='center', fontsize=11, weight='bold')
    ax2.text(3.5, 7.5, 'Head 1: z_causal', ha='center', va='center', fontsize=9)
    ax2.text(6.5, 7.5, 'Head 2: z_noise', ha='center', va='center', fontsize=9)
    
    # 正交约束
    orth_box = FancyBboxPatch((2, 5), 6, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor='#FFF5E5', 
                             edgecolor='orange', linewidth=2)
    ax2.add_patch(orth_box)
    ax2.text(5, 5.5, '正交约束: z_causal ⊥ z_noise', ha='center', va='center', fontsize=10, weight='bold')
    
    # 说明
    ax2.text(5, 3.5, '核心思想:', ha='center', va='center', fontsize=11, weight='bold')
    ax2.text(5, 2.8, '显式分离因果特征和噪声特征', ha='center', va='center', fontsize=10)
    ax2.text(5, 2.3, '物理上确保 z_causal 不包含设备信息', ha='center', va='center', fontsize=10)
    ax2.text(5, 1.8, '通过正交约束实现可解释的解耦', ha='center', va='center', fontsize=10, style='italic')
    
    # 代码位置
    code_box2 = FancyBboxPatch((1, 0.5), 8, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#F0F0F0', 
                               edgecolor='gray', linewidth=1)
    ax2.add_patch(code_box2)
    ax2.text(5, 1, '代码位置: OrthogonalLoss.forward()', ha='center', va='center', fontsize=9)
    
    # 创新点3: 先验约束的域不变性学习
    ax3 = axes[1, 0]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    
    ax3.text(5, 9, '创新点3: 先验约束的域不变性学习', ha='center', va='center', 
            fontsize=14, weight='bold', color='blue')
    
    # 先验知识
    prior_box3 = FancyBboxPatch((1, 7), 3.5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#E8F4F8', 
                               edgecolor='blue', linewidth=2)
    ax3.add_patch(prior_box3)
    ax3.text(2.75, 8, '先验知识', ha='center', va='center', fontsize=11, weight='bold')
    ax3.text(2.75, 7.5, '临床数据', ha='center', va='center', fontsize=9)
    ax3.text(2.75, 7.1, '(HPV/TCT)', ha='center', va='center', fontsize=9)
    
    # 域不变性
    inv_box = FancyBboxPatch((5.5, 7), 3.5, 1.5, 
                            boxstyle="round,pad=0.1", 
                            facecolor='#E5F5E5', 
                            edgecolor='green', linewidth=2)
    ax3.add_patch(inv_box)
    ax3.text(7.25, 8, '域不变性', ha='center', va='center', fontsize=11, weight='bold')
    ax3.text(7.25, 7.5, '约束到生物流形', ha='center', va='center', fontsize=9)
    ax3.text(7.25, 7.1, '消除设备噪声', ha='center', va='center', fontsize=9)
    
    # 箭头
    arrow3 = FancyArrowPatch((4.5, 7.75), (5.5, 7.75), 
                             arrowstyle='->', lw=2, color='black')
    ax3.add_patch(arrow3)
    
    # 说明
    ax3.text(5, 5.5, '核心思想:', ha='center', va='center', fontsize=11, weight='bold')
    ax3.text(5, 4.8, '使用临床模态作为先验，定义生物流形', ha='center', va='center', fontsize=10)
    ax3.text(5, 4.3, '通过约束图像特征到生物流形，实现域不变性', ha='center', va='center', fontsize=10)
    ax3.text(5, 3.8, '临床数据在不同医院含义相同（生物不变性）', ha='center', va='center', fontsize=10, style='italic')
    
    # 代码位置
    code_box3 = FancyBboxPatch((1, 1), 8, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#F0F0F0', 
                               edgecolor='gray', linewidth=1)
    ax3.add_patch(code_box3)
    ax3.text(5, 2, '代码位置:', ha='center', va='center', fontsize=10, weight='bold')
    ax3.text(2.5, 1.5, 'clinical_to_text()', ha='center', va='center', fontsize=8)
    ax3.text(7.5, 1.5, 'BIDAModel.forward()', ha='center', va='center', fontsize=8)
    
    # 创新点4: VLM增强的语义理解
    ax4 = axes[1, 1]
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    
    ax4.text(5, 9, '创新点4: VLM增强的语义理解', ha='center', va='center', 
            fontsize=14, weight='bold', color='purple')
    
    # VLM处理
    vlm_box4 = FancyBboxPatch((2, 7), 6, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#F0E8FF', 
                              edgecolor='purple', linewidth=2)
    ax4.add_patch(vlm_box4)
    ax4.text(5, 8, 'VLM处理图像+文本', ha='center', va='center', fontsize=11, weight='bold')
    ax4.text(3.5, 7.5, 'OCT图像', ha='center', va='center', fontsize=9)
    ax4.text(5, 7.5, '+', ha='center', va='center', fontsize=12, weight='bold')
    ax4.text(6.5, 7.5, '临床文本', ha='center', va='center', fontsize=9)
    ax4.text(5, 7.1, '→ 语义特征 [B, 1536]', ha='center', va='center', fontsize=9)
    
    # 说明
    ax4.text(5, 5.5, '核心思想:', ha='center', va='center', fontsize=11, weight='bold')
    ax4.text(5, 4.8, 'VLM同时理解图像内容和文本语义', ha='center', va='center', fontsize=10)
    ax4.text(5, 4.3, '提取融合了图像和文本的语义特征', ha='center', va='center', fontsize=10)
    ax4.text(5, 3.8, '理解"HPV阳性"和图像中病理特征的关联', ha='center', va='center', fontsize=10, style='italic')
    
    # 代码位置
    code_box4 = FancyBboxPatch((1, 1), 8, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#F0F0F0', 
                               edgecolor='gray', linewidth=1)
    ax4.add_patch(code_box4)
    ax4.text(5, 2, '代码位置:', ha='center', va='center', fontsize=10, weight='bold')
    ax4.text(5, 1.5, 'DistributionalAnchor.forward() - VLM处理部分', ha='center', va='center', fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def main():
    """主函数：生成PDF"""
    
    # 创建PDF文件
    pdf_path = '/data2/hmy/VLM_Caus_Rm_Mics/cursor_file/visualization/BIDA_Framework_Diagram.pdf'
    
    with PdfPages(pdf_path) as pdf:
        # 图1: 整体框架图
        print("正在生成整体框架图...")
        fig1 = create_bida_framework_diagram()
        pdf.savefig(fig1, bbox_inches='tight', dpi=300)
        plt.close(fig1)
        print("✅ 整体框架图已生成")
        
        # 图2: 创新点详解图
        print("正在生成创新点详解图...")
        fig2 = create_innovation_points_diagram()
        pdf.savefig(fig2, bbox_inches='tight', dpi=300)
        plt.close(fig2)
        print("✅ 创新点详解图已生成")
    
    print(f"\n🎉 PDF文件已保存至: {pdf_path}")
    print(f"📄 包含2页：")
    print(f"   1. BIDA整体框架图")
    print(f"   2. 核心创新点详解图")

if __name__ == '__main__':
    main()


