#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2 模型架构可视化（顶刊风格）
参考CVPR、MICCAI等顶刊的架构图风格，绘制专业的模型架构图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon, PathPatch
from matplotlib.patches import Path as MPath
import numpy as np
from pathlib import Path

# 设置中文字体和专业字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9
plt.rcParams['font.weight'] = 'normal'

def draw_module_box(ax, x, y, width, height, text, color, edge_color, linewidth=2, 
                    subtext=None, detail_text=None, text_color='black'):
    """绘制模块框（顶刊风格）"""
    # 主框
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.08", 
                         facecolor=color,
                         edgecolor=edge_color,
                         linewidth=linewidth,
                         zorder=2)
    ax.add_patch(box)
    
    # 主文本
    if subtext:
        ax.text(x + width/2, y + height - 0.15, text,
                ha='center', va='top', fontsize=11, weight='bold', color=text_color)
        ax.text(x + width/2, y + height/2, subtext,
                ha='center', va='center', fontsize=9, color=text_color)
    else:
        ax.text(x + width/2, y + height/2, text,
                ha='center', va='center', fontsize=10, weight='bold', color=text_color)
    
    # 详细文本
    if detail_text:
        ax.text(x + width/2, y + 0.1, detail_text,
                ha='center', va='bottom', fontsize=7, style='italic', color=text_color)
    
    return box

def draw_arrow(ax, x1, y1, x2, y2, color='#333333', linewidth=1.5, style='solid', 
               label=None, label_pos=0.5, label_offset=0.15):
    """绘制箭头（顶刊风格）"""
    if style == 'dashed':
        linestyle = '--'
    elif style == 'dotted':
        linestyle = ':'
    else:
        linestyle = '-'
    
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='->', 
                            lw=linewidth, 
                            color=color,
                            linestyle=linestyle,
                            zorder=1,
                            mutation_scale=15)
    ax.add_patch(arrow)
    
    if label:
        mid_x = x1 + (x2 - x1) * label_pos
        mid_y = y1 + (y2 - y1) * label_pos
        # 计算垂直偏移
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            perp_x = -dy / length * label_offset
            perp_y = dx / length * label_offset
            ax.text(mid_x + perp_x, mid_y + perp_y, label,
                   ha='center', va='center', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='none', alpha=0.8),
                   zorder=3)
    return arrow

def draw_dimension_label(ax, x, y, text, color='#666666'):
    """绘制维度标注"""
    ax.text(x, y, text, ha='center', va='center', fontsize=7,
           style='italic', color=color, weight='normal')

def draw_bio_cot_v3_2_architecture():
    """
    绘制Bio-COT 3.2的详细架构图（顶刊风格）
    """
    fig = plt.figure(figsize=(20, 28))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 28)
    ax.axis('off')
    
    # 专业配色方案（参考顶刊）
    colors = {
        'input': '#E3F2FD',           # 浅蓝色 - 输入
        'vlm_frozen': '#FFF3E0',      # 浅橙色 - 冻结模块
        'vlm_trainable': '#FFE0B2',    # 橙色 - 可训练模块
        'visual_notes': '#E1F5FE',     # 浅青色 - Visual Notes
        'adaptive': '#E8F5E9',        # 浅绿色 - 自适应
        'dual_head': '#F3E5F5',       # 浅紫色 - 双头
        'alignment': '#FCE4EC',       # 浅粉色 - 对齐
        'fusion': '#E0F2F1',          # 浅青色 - 融合
        'classifier': '#FFF9C4',      # 浅黄色 - 分类器
        'loss': '#FFEBEE',            # 浅红色 - 损失
        'arrow_main': '#1976D2',      # 蓝色 - 主数据流
        'arrow_aux': '#4CAF50',       # 绿色 - 辅助连接
        'arrow_loss': '#F44336',      # 红色 - 损失连接
        'text': '#212121'             # 深灰色 - 文本
    }
    
    # ==================== 标题区域 ====================
    title_box = Rectangle((0, 26.5), 20, 1.3, 
                         facecolor='#1A237E', edgecolor='none', zorder=0)
    ax.add_patch(title_box)
    ax.text(10, 27.5, 'Bio-COT 3.2: Enhanced Logic Loop Architecture', 
            ha='center', va='center', fontsize=16, weight='bold', color='white')
    ax.text(10, 27.0, 'Integrating 3.1 Advantages (Explicit Alignment, Adaptive Fusion) + 4.0 Advantages (Frozen VLM, Dynamic Knowledge)', 
            ha='center', va='center', fontsize=10, color='#E3F2FD')
    
    y_pos = 26.0
    
    # ==================== 输入层 ====================
    y_input = y_pos - 0.8
    
    # OCT特征
    draw_module_box(ax, 0.3, y_input-0.4, 3.5, 0.4, 
                   'OCT Features', colors['input'], '#1976D2', 2,
                   detail_text='[B, N, 768]')
    
    # Colposcopy特征
    draw_module_box(ax, 4.3, y_input-0.4, 3.5, 0.4,
                   'Colposcopy Features', colors['input'], '#1976D2', 2,
                   detail_text='[B, N, 768]')
    
    # 图像元数据
    draw_module_box(ax, 8.3, y_input-0.4, 3.5, 0.4,
                   'Image Names\nClinical Info', colors['input'], '#1976D2', 2)
    
    # VLM缓存
    draw_module_box(ax, 12.3, y_input-0.4, 3.5, 0.4,
                   'VLM Cache', colors['vlm_frozen'], '#FF6F00', 2,
                   detail_text='102,705 descriptions')
    
    # 中心标签（用于损失）
    draw_module_box(ax, 16.3, y_input-0.4, 3.2, 0.4,
                   'Center Labels', colors['input'], '#1976D2', 2,
                   detail_text='[B]')
    
    y_pos = y_input - 0.9
    
    # ==================== Step 1: VLM增强知识检索器（4.0优势）====================
    y_vlm = y_pos - 0.5
    
    # 主框架
    vlm_frame = Rectangle((0.2, y_vlm-2.8), 19.6, 2.8,
                          facecolor='white', edgecolor='#FF6F00', linewidth=2.5,
                          linestyle='--', zorder=0)
    ax.add_patch(vlm_frame)
    ax.text(10, y_vlm-0.1, 'Step 1: VLM-Augmented Knowledge Retriever (4.0 Advantage)', 
            ha='center', va='top', fontsize=12, weight='bold', color='#E65100')
    
    # 1.1 VLM描述检索
    draw_module_box(ax, 1, y_vlm-1.2, 4, 0.9,
                   'VLM Description\nRetrieval', colors['vlm_frozen'], '#FF6F00', 2,
                   detail_text='Lookup: image_name → VLM description')
    
    # 1.2 文本构造
    draw_module_box(ax, 5.5, y_vlm-1.2, 3.5, 0.9,
                   'Text Prompt\nConstruction', colors['vlm_frozen'], '#FF6F00', 2,
                   detail_text='"Findings: {vlm_desc}. Clinical: {clinical_info}"')
    
    # 1.3 Frozen Text Encoder
    draw_module_box(ax, 9.5, y_vlm-1.2, 4.5, 0.9,
                   'Frozen Text Encoder\nPubMedBERT', colors['vlm_frozen'], '#F57C00', 2.5,
                   detail_text='❄️ Frozen (no grad) | Output: [B, 768]')
    
    # 1.4 Trainable Adapter
    adapter_detail = 'Linear(768→768)\nLayerNorm → ReLU\nDropout(0.1)\nLinear(768→768)'
    draw_module_box(ax, 14.5, y_vlm-1.2, 4.5, 0.9,
                   'Trainable Adapter', colors['vlm_trainable'], '#E65100', 2.5,
                   detail_text='🔥 Trainable | Maps: text → visual space')
    
    # 输出：z_sem
    z_sem_box = draw_module_box(ax, 15.5, y_vlm-2.5, 3, 0.5,
                               'z_sem', '#C8E6C9', '#2E7D32', 2,
                               detail_text='[B, 768]')
    
    # 箭头
    draw_arrow(ax, 2.3, y_input-0.2, 3, y_vlm-0.75, colors['arrow_main'], 2)
    draw_arrow(ax, 10.1, y_input-0.2, 7.25, y_vlm-0.75, colors['arrow_main'], 2)
    draw_arrow(ax, 14.05, y_input-0.2, 11.75, y_vlm-0.75, colors['arrow_main'], 2)
    draw_arrow(ax, 3, y_vlm-0.75, 7.25, y_vlm-0.75, colors['arrow_main'], 1.5)
    draw_arrow(ax, 9.5, y_vlm-0.75, 11.75, y_vlm-0.75, colors['arrow_main'], 1.5)
    draw_arrow(ax, 14, y_vlm-0.75, 17, y_vlm-2.25, colors['arrow_main'], 2)
    
    y_pos = y_vlm - 3.2
    
    # ==================== Step 2: Visual Notes模块（3.1优势）====================
    y_visual = y_pos - 0.5
    
    visual_frame = Rectangle((0.2, y_visual-3.0), 19.6, 3.0,
                             facecolor='white', edgecolor='#1976D2', linewidth=2.5,
                             linestyle='--', zorder=0)
    ax.add_patch(visual_frame)
    ax.text(10, y_visual-0.1, 'Step 2: Enhanced Visual Notes with Cross-Attention (3.1 Advantage)', 
            ha='center', va='top', fontsize=12, weight='bold', color='#0D47A1')
    
    # OCT Visual Notes
    oct_visual_detail = 'Q_proj: Linear(768→768)\nK_proj: Linear(768→768)\nAttention: Q(text) @ K(image)\nGate: Sigmoid → Clamp(0.05, 1.0)\nModulation: feat × (mask + (1-mask)×β)\nRefine: LayerNorm → Linear → GELU'
    draw_module_box(ax, 0.5, y_visual-2.5, 8.5, 2.3,
                   'OCT Visual Notes\n(Cross-Attention)', colors['visual_notes'], '#1976D2', 2,
                   detail_text=oct_visual_detail)
    
    # Colposcopy Visual Notes
    colpo_visual_detail = 'Same as OCT Visual Notes\nText-guided feature enhancement\nDynamic β (warm-up strategy)'
    draw_module_box(ax, 9.5, y_visual-2.5, 8.5, 2.3,
                   'Colposcopy Visual Notes\n(Cross-Attention)', colors['visual_notes'], '#1976D2', 2,
                   detail_text=colpo_visual_detail)
    
    # 输出
    draw_module_box(ax, 2, y_visual-2.8, 5.5, 0.25,
                   'f_oct_pooled [B, 768]', '#90CAF9', '#1565C0', 1.5)
    draw_module_box(ax, 11, y_visual-2.8, 5.5, 0.25,
                   'f_colpo_pooled [B, 768]', '#90CAF9', '#1565C0', 1.5)
    
    # 箭头
    draw_arrow(ax, 2.05, y_input-0.2, 4.75, y_visual-1.35, colors['arrow_main'], 2)
    draw_arrow(ax, 6.05, y_input-0.2, 13.75, y_visual-1.35, colors['arrow_main'], 2)
    draw_arrow(ax, 17, y_vlm-2.25, 4.75, y_visual-1.35, colors['arrow_aux'], 2, 'dashed', 'z_sem')
    draw_arrow(ax, 17, y_vlm-2.25, 13.75, y_visual-1.35, colors['arrow_aux'], 2, 'dashed', 'z_sem')
    
    y_pos = y_visual - 3.5
    
    # ==================== Step 3: 自适应模态门控（3.1优势）====================
    y_adaptive = y_pos - 0.5
    
    adaptive_frame = Rectangle((0.2, y_adaptive-1.5), 19.6, 1.5,
                               facecolor='white', edgecolor='#388E3C', linewidth=2.5,
                               linestyle='--', zorder=0)
    ax.add_patch(adaptive_frame)
    ax.text(10, y_adaptive-0.1, 'Step 3: Adaptive Modality Gating (3.1 Advantage)', 
            ha='center', va='top', fontsize=12, weight='bold', color='#1B5E20')
    
    adaptive_detail = 'Concat: [f_oct, f_colpo] → [B, 1536]\nScore Network: Linear(1536→384) → LayerNorm → ReLU → Linear(384→2)\nSoftmax: w_oct, w_colpo\nFusion: w_oct × f_oct + w_colpo × f_colpo'
    draw_module_box(ax, 0.5, y_adaptive-1.2, 18, 1.0,
                   'Adaptive Modality Gating', colors['adaptive'], '#388E3C', 2,
                   detail_text=adaptive_detail)
    
    # 输出
    draw_module_box(ax, 15.5, y_adaptive-1.3, 3, 0.3,
                   'f_fused [B, 768]', '#81C784', '#1B5E20', 1.5)
    
    # 箭头
    draw_arrow(ax, 4.75, y_visual-2.8, 9.5, y_adaptive-0.7, colors['arrow_main'], 2)
    draw_arrow(ax, 13.75, y_visual-2.8, 9.5, y_adaptive-0.7, colors['arrow_main'], 2)
    draw_arrow(ax, 9.5, y_adaptive-0.7, 17, y_adaptive-1.15, colors['arrow_main'], 2)
    
    y_pos = y_adaptive - 2.0
    
    # ==================== Step 4: 双头因果编码器 ====================
    y_dual = y_pos - 0.5
    
    dual_frame = Rectangle((0.2, y_dual-2.5), 19.6, 2.5,
                          facecolor='white', edgecolor='#F57C00', linewidth=2.5,
                          linestyle='--', zorder=0)
    ax.add_patch(dual_frame)
    ax.text(10, y_dual-0.1, 'Step 4: Dual-Head Causal Encoder', 
            ha='center', va='top', fontsize=12, weight='bold', color='#E65100')
    
    # Feature Projection
    proj_detail = 'Linear(768→1536)\nLayerNorm(1536)\nGELU\nDropout(0.1)\nLinear(1536→768)'
    draw_module_box(ax, 0.5, y_dual-2.0, 5, 1.7,
                   'Feature Projection', colors['dual_head'], '#F57C00', 2,
                   detail_text=proj_detail)
    
    # Causal Head
    causal_detail = 'Linear(768→768)\nLayerNorm(768)\nGELU\nDropout(0.1)\nLinear(768→768)'
    draw_module_box(ax, 6, y_dual-2.0, 5, 1.7,
                   'Causal Head', '#C5E1A5', '#558B2F', 2,
                   detail_text=causal_detail)
    
    # Noise Head
    noise_detail = 'Linear(768→768)\nLayerNorm(768)\nGELU\nDropout(0.1)\nLinear(768→768)'
    draw_module_box(ax, 11.5, y_dual-2.0, 5, 1.7,
                   'Noise Head', '#FFCCBC', '#D84315', 2,
                   detail_text=noise_detail)
    
    # 输出
    draw_module_box(ax, 6.5, y_dual-2.3, 4, 0.25,
                   'z_causal [B, 768]', '#A5D6A7', '#2E7D32', 1.5)
    draw_module_box(ax, 12, y_dual-2.3, 4, 0.25,
                   'z_noise [B, 768]', '#FFAB91', '#BF360C', 1.5)
    
    # 箭头
    draw_arrow(ax, 17, y_adaptive-1.15, 3, y_dual-1.15, colors['arrow_main'], 2)
    draw_arrow(ax, 3, y_dual-1.15, 8.5, y_dual-1.15, colors['arrow_main'], 1.5)
    draw_arrow(ax, 8.5, y_dual-1.15, 14, y_dual-1.15, colors['arrow_main'], 1.5)
    draw_arrow(ax, 8.5, y_dual-1.15, 8.5, y_dual-2.15, colors['arrow_main'], 1.5)
    draw_arrow(ax, 14, y_dual-1.15, 14, y_dual-2.15, colors['arrow_main'], 1.5)
    
    y_pos = y_dual - 3.0
    
    # ==================== Step 5: 跨模态融合（3.1优势）====================
    y_fusion = y_pos - 0.5
    
    fusion_frame = Rectangle((0.2, y_fusion-1.5), 19.6, 1.5,
                            facecolor='white', edgecolor='#00695C', linewidth=2.5,
                            linestyle='--', zorder=0)
    ax.add_patch(fusion_frame)
    ax.text(10, y_fusion-0.1, 'Step 5: Cross-Modal Fusion (3.1 Advantage)', 
            ha='center', va='top', fontsize=12, weight='bold', color='#004D40')
    
    fusion_detail = 'MultiheadAttention (4 heads, batch_first=True)\nQ = z_causal [B, 1, 768]\nK = V = z_sem [B, 1, 768]\nOutput + Residual: f_final = Attention(Q,K,V) + z_causal'
    draw_module_box(ax, 0.5, y_fusion-1.2, 18, 1.0,
                   'Cross-Modal Fusion', colors['fusion'], '#00695C', 2,
                   detail_text=fusion_detail)
    
    # 输出
    draw_module_box(ax, 15.5, y_fusion-1.3, 3, 0.3,
                   'f_final [B, 768]', '#4DB6AC', '#004D40', 1.5)
    
    # 箭头
    draw_arrow(ax, 8.5, y_dual-2.15, 9.5, y_fusion-0.7, colors['arrow_main'], 2)
    draw_arrow(ax, 17, y_vlm-2.25, 9.5, y_fusion-0.7, colors['arrow_aux'], 2, 'dashed', 'z_sem')
    draw_arrow(ax, 9.5, y_fusion-0.7, 17, y_fusion-1.15, colors['arrow_main'], 2)
    
    y_pos = y_fusion - 2.0
    
    # ==================== Step 6: 分类器 ====================
    y_cls = y_pos - 0.5
    
    cls_frame = Rectangle((0.2, y_cls-1.2), 19.6, 1.2,
                         facecolor='white', edgecolor='#C62828', linewidth=2.5,
                         linestyle='--', zorder=0)
    ax.add_patch(cls_frame)
    ax.text(10, y_cls-0.1, 'Step 6: Classifier', 
            ha='center', va='top', fontsize=12, weight='bold', color='#B71C1C')
    
    cls_detail = 'Linear(768→384) → LayerNorm(384) → GELU → Dropout(0.2) → Linear(384→2)'
    draw_module_box(ax, 0.5, y_cls-0.9, 18, 0.7,
                   'Classifier', colors['classifier'], '#C62828', 2,
                   detail_text=cls_detail)
    
    # 输出
    draw_module_box(ax, 15.5, y_cls-1.0, 3, 0.3,
                   'Prediction [B, 2]', '#EF5350', '#B71C1C', 1.5)
    
    # 箭头
    draw_arrow(ax, 17, y_fusion-1.15, 17, y_cls-0.55, colors['arrow_main'], 2)
    
    y_pos = y_cls - 1.7
    
    # ==================== Step 7: 深度对齐模块（3.1优势）====================
    y_align = y_pos - 0.5
    
    align_frame = Rectangle((0.2, y_align-4.0), 19.6, 4.0,
                           facecolor='white', edgecolor='#7B1FA2', linewidth=2.5,
                           linestyle='--', zorder=0)
    ax.add_patch(align_frame)
    ax.text(10, y_align-0.1, 'Step 7: Deep Alignment Module (3.1 Advantage)', 
            ha='center', va='top', fontsize=12, weight='bold', color='#4A148C')
    
    # Image投影头
    img_proj_detail = 'Linear(768→768, bias=False)\nLayerNorm(768)\nGELU\nDropout(0.1)\nLinear(768→256, bias=False)\nLayerNorm(256)'
    draw_module_box(ax, 0.5, y_align-2.5, 4.5, 2.2,
                   'Image Projection\nHead', '#E1BEE7', '#7B1FA2', 2,
                   detail_text=img_proj_detail)
    
    # Text投影头
    txt_proj_detail = 'Linear(768→768, bias=False)\nLayerNorm(768)\nGELU\nDropout(0.1)\nLinear(768→256, bias=False)\nLayerNorm(256)'
    draw_module_box(ax, 5.5, y_align-2.5, 4.5, 2.2,
                   'Text Projection\nHead', '#E1BEE7', '#7B1FA2', 2,
                   detail_text=txt_proj_detail)
    
    # 共享语义空间投影
    shared_detail = 'Shared Projection:\nLinear(256→256, bias=False)\nLayerNorm(256)\nGELU'
    draw_module_box(ax, 10.5, y_align-2.5, 4.5, 1.0,
                   'Shared Semantic\nSpace Projection', '#CE93D8', '#6A1B9A', 2,
                   detail_text=shared_detail)
    
    # 归一化和温度
    norm_detail = 'L2 Normalization\nTemperature: τ = exp(logit_scale)\nClamp: [0.1, 50.0]'
    draw_module_box(ax, 10.5, y_align-1.4, 4.5, 0.8,
                   'Normalization &\nTemperature', '#BA68C8', '#4A148C', 2,
                   detail_text=norm_detail)
    
    # InfoNCE Loss
    infonce_detail = 'Similarity Matrix:\nlogits = τ × (z_img_norm @ z_txt_norm^T)\nInfoNCE Loss:\nL_ce = (CE(logits_img, labels) + CE(logits_txt, labels)) / 2'
    draw_module_box(ax, 15.5, y_align-2.5, 4, 1.3,
                   'InfoNCE Loss', '#F8BBD0', '#AD1457', 2,
                   detail_text=infonce_detail)
    
    # L2辅助损失
    l2_detail = 'L2 Distance Loss:\nL_l2 = mean(||z_img_norm - z_txt_norm||_2)\nCombined: L_align = L_ce + 0.1 × L_l2'
    draw_module_box(ax, 15.5, y_align-1.1, 4, 1.3,
                   'L2 Auxiliary Loss', '#F48FB1', '#AD1457', 2,
                   detail_text=l2_detail)
    
    # Recall@1计算
    recall_detail = 'Recall@1:\npred_i2t = argmax(logits_img, dim=1)\nRecall = mean(pred_i2t == labels)'
    draw_module_box(ax, 10.5, y_align-3.7, 9, 0.5,
                   'Recall@1 Metric', '#FCE4EC', '#C2185B', 1.5,
                   detail_text=recall_detail)
    
    # 箭头
    draw_arrow(ax, 8.5, y_dual-2.15, 2.75, y_align-1.4, colors['arrow_aux'], 2, 'dashed', 'z_causal')
    draw_arrow(ax, 17, y_vlm-2.25, 7.75, y_align-1.4, colors['arrow_aux'], 2, 'dashed', 'z_sem')
    draw_arrow(ax, 2.75, y_align-1.4, 7.75, y_align-1.4, colors['arrow_main'], 1.5)
    draw_arrow(ax, 7.75, y_align-1.4, 12.75, y_align-2.0, colors['arrow_main'], 1.5)
    draw_arrow(ax, 2.75, y_align-1.4, 12.75, y_align-2.0, colors['arrow_main'], 1.5)
    draw_arrow(ax, 12.75, y_align-2.0, 12.75, y_align-1.0, colors['arrow_main'], 1.5)
    draw_arrow(ax, 12.75, y_align-1.0, 17.5, y_align-1.85, colors['arrow_main'], 1.5)
    draw_arrow(ax, 12.75, y_align-1.0, 17.5, y_align-0.45, colors['arrow_main'], 1.5)
    
    y_pos = y_align - 4.5
    
    # ==================== 损失函数模块 ====================
    y_loss = y_pos - 0.3
    
    loss_frame = Rectangle((0.2, y_loss-3.5), 19.6, 3.5,
                          facecolor='white', edgecolor='#AD1457', linewidth=2.5,
                          linestyle='--', zorder=0)
    ax.add_patch(loss_frame)
    ax.text(10, y_loss-0.1, 'Loss Functions', 
            ha='center', va='top', fontsize=12, weight='bold', color='#880E4F')
    
    # 损失函数网格布局
    loss_modules = [
        ('OT Loss\n(Sinkhorn)', 'λ_ot = 0.5', 0.5, y_loss-1.0, 3.5, 0.8),
        ('Alignment Loss\n(InfoNCE + L2)', 'λ_align = 0.5', 4.5, y_loss-1.0, 3.5, 0.8),
        ('Consistency Loss\n(Counterfactual)', 'λ_consist = 0.2', 8.5, y_loss-1.0, 3.5, 0.8),
        ('Adversarial Loss\n(Center Discriminator)', 'λ_adv = 0.5', 12.5, y_loss-1.0, 3.5, 0.8),
        ('Sparse Loss\n(Attention Entropy)', 'λ_sparse = 0.05', 16.5, y_loss-1.0, 3.5, 0.8),
        ('Classification Loss\n(Focal Loss)', 'λ_cls = 2.0', 0.5, y_loss-2.0, 3.5, 0.8),
        ('Memory Bank\nUpdate', 'Store z_noise\nby center', 4.5, y_loss-2.0, 3.5, 0.8),
        ('Counterfactual\nGeneration', 'z_causal_cf = z_causal\n+ z_noise_cf', 8.5, y_loss-2.0, 3.5, 0.8),
        ('Total Loss', 'L = λ_cls×L_cls + λ_ot×L_ot\n+ λ_align×L_align + λ_consist×L_consist\n+ λ_adv×L_adv + λ_sparse×L_sparse', 12.5, y_loss-2.0, 7.5, 0.8),
    ]
    
    for i, (title, detail, x, y, w, h) in enumerate(loss_modules):
        if i < 5:
            color = '#F8BBD0'
            edge = '#AD1457'
        elif i < 8:
            color = '#F48FB1'
            edge = '#C2185B'
        else:
            color = '#EC407A'
            edge = '#880E4F'
        
        draw_module_box(ax, x, y, w, h, title, color, edge, 2, detail_text=detail)
    
    # 损失连接箭头（虚线）
    draw_arrow(ax, 8.5, y_dual-2.15, 2.25, y_loss-0.6, colors['arrow_loss'], 1.5, 'dotted', 'z_causal')
    draw_arrow(ax, 17, y_vlm-2.25, 6.25, y_loss-0.6, colors['arrow_loss'], 1.5, 'dotted', 'z_sem')
    draw_arrow(ax, 14, y_dual-2.15, 10.25, y_loss-0.6, colors['arrow_loss'], 1.5, 'dotted', 'z_noise')
    draw_arrow(ax, 17, y_cls-0.55, 2.25, y_loss-1.6, colors['arrow_loss'], 1.5, 'dotted', 'pred')
    draw_arrow(ax, 4.75, y_visual-1.35, 18.25, y_loss-1.6, colors['arrow_loss'], 1.5, 'dotted', 'attn')
    
    # ==================== 图例 ====================
    legend_y = 1.5
    legend_x = 0.5
    
    # 图例框
    legend_frame = Rectangle((legend_x, legend_y-1.2), 19, 1.2,
                            facecolor='#FAFAFA', edgecolor='#BDBDBD', linewidth=1.5,
                            zorder=0)
    ax.add_patch(legend_frame)
    ax.text(legend_x + 9.5, legend_y-0.2, 'Legend', 
            ha='center', va='top', fontsize=11, weight='bold', color=colors['text'])
    
    # 图例项
    legend_items = [
        ('Main Data Flow', colors['arrow_main'], '-'),
        ('Auxiliary Connection', colors['arrow_aux'], '--'),
        ('Loss Connection', colors['arrow_loss'], ':'),
        ('3.1 Advantage', '#1976D2', 'solid'),
        ('4.0 Advantage', '#FF6F00', 'solid'),
    ]
    
    for i, (label, color, style) in enumerate(legend_items):
        x = legend_x + 0.3 + (i % 3) * 6
        y = legend_y - 0.5 - (i // 3) * 0.4
        
        if style == '-':
            arrow = FancyArrowPatch((x, y), (x + 0.8, y),
                                   arrowstyle='->', lw=2, color=color)
        elif style == '--':
            arrow = FancyArrowPatch((x, y), (x + 0.8, y),
                                   arrowstyle='->', lw=2, color=color, linestyle='--')
        else:
            arrow = FancyArrowPatch((x, y), (x + 0.8, y),
                                   arrowstyle='->', lw=2, color=color, linestyle=':')
        ax.add_patch(arrow)
        ax.text(x + 1.0, y, label, fontsize=8, va='center', color=colors['text'])
    
    # 保存图片
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    # 保存PNG（高分辨率）
    png_path = output_dir / 'bio_cot_v3_2_architecture.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.1)
    print(f"✅ 架构图已保存: {png_path}")
    
    # 保存PDF
    pdf_path = output_dir / 'bio_cot_v3_2_architecture.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.1)
    print(f"✅ PDF已保存: {pdf_path}")
    
    plt.close()
    
    return png_path, pdf_path


if __name__ == '__main__':
    print("🎨 开始绘制Bio-COT 3.2架构图（顶刊风格）...")
    png_path, pdf_path = draw_bio_cot_v3_2_architecture()
    print(f"✅ 完成！")
    print(f"   PNG: {png_path}")
    print(f"   PDF: {pdf_path}")
