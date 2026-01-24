#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT Model Architecture Diagram - Standard SCI Paper Style
参考标准SCI论文框架图风格，详细展示每个模块的内部结构
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.patches import FancyBboxPatch as RoundedRect
import numpy as np
import os

# Set high DPI for publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

# Create figure (wider for detailed structure)
fig = plt.figure(figsize=(24, 16))
ax = fig.add_subplot(111)
ax.set_xlim(0, 24)
ax.set_ylim(0, 16)
ax.axis('off')

# Color scheme (matching standard SCI style)
colors = {
    'input': '#E8F5E9',          # Light green
    'input_oct': '#C8E6C9',      # Green
    'input_colpo': '#A5D6A7',    # Light green
    'input_clinical': '#FFCDD2',  # Light red
    'stem': '#BBDEFB',           # Light blue
    'conv': '#90CAF9',           # Blue
    'norm': '#CE93D8',           # Purple
    'activation': '#FFB74D',     # Orange
    'linear': '#64B5F6',         # Light blue
    'attention': '#F48FB1',       # Pink
    'fusion': '#81C784',         # Green
    'output': '#FFF9C4',         # Yellow
    'skip': '#9E9E9E',          # Gray
    'vlm': '#BA68C8',            # Purple
    'memory': '#FF8A65',         # Deep orange
    'loss': '#B0BEC5',           # Blue gray
}

# Font settings
title_font = {'fontsize': 18, 'fontweight': 'bold', 'ha': 'center'}
section_font = {'fontsize': 12, 'fontweight': 'bold', 'ha': 'center'}
label_font = {'fontsize': 10, 'ha': 'center'}
small_font = {'fontsize': 8, 'ha': 'center'}
formula_font = {'fontsize': 9, 'family': 'monospace', 'ha': 'center'}

# ==================== Title ====================
ax.text(12, 15.5, 'Bio-COT: Biological Causal Optimal Transport Framework', 
        **title_font, va='center')
ax.text(12, 15.0, 'for Multi-Center Cervical Lesion Classification', 
        fontsize=14, style='italic', ha='center', va='center')

# ==================== Helper Functions ====================
def draw_module_box(ax, x, y, width, height, color, text, text_color='black', 
                    fontsize=10, fontweight='normal', alpha=0.8):
    """绘制模块框"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.05",
                         facecolor=color,
                         edgecolor='black', linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=fontsize, 
            fontweight=fontweight, color=text_color)

def draw_layer_stack(ax, x, y, width, height, layers, color_base, direction='vertical'):
    """绘制层堆叠（类似3D tensor可视化）"""
    if direction == 'vertical':
        layer_height = height / len(layers)
        for i, layer in enumerate(layers):
            y_pos = y + i * layer_height
            draw_module_box(ax, x, y_pos, width, layer_height, 
                          color_base, layer, fontsize=8)
    else:
        layer_width = width / len(layers)
        for i, layer in enumerate(layers):
            x_pos = x + i * layer_width
            draw_module_box(ax, x_pos, y, layer_width, height,
                          color_base, layer, fontsize=8)

def draw_arrow(ax, x1, y1, x2, y2, color='black', width=2, style='-', alpha=1.0, 
               connection=None, arrowstyle='->'):
    """绘制箭头"""
    if connection:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle=arrowstyle, lw=width, color=color,
                               linestyle=style, alpha=alpha,
                               connectionstyle=connection)
    else:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle=arrowstyle, lw=width, color=color,
                               linestyle=style, alpha=alpha)
    ax.add_patch(arrow)

# ==================== Input Layer ====================
input_y = 13.5
input_width = 1.5
input_height = 0.8

# OCT Input
oct_input = FancyBboxPatch((0.5, input_y), input_width, input_height,
                          boxstyle="round,pad=0.05",
                          facecolor=colors['input_oct'],
                          edgecolor='black', linewidth=1.5, alpha=0.9)
ax.add_patch(oct_input)
ax.text(1.25, input_y + 0.5, 'OCT\nFeatures', ha='center', va='center', 
        fontsize=10, fontweight='bold')
ax.text(1.25, input_y + 0.15, '[B, 512]', ha='center', va='center', 
        fontsize=8, family='monospace')

# Colposcopy Input
colpo_input = FancyBboxPatch((0.5, input_y - 1.0), input_width, input_height,
                             boxstyle="round,pad=0.05",
                             facecolor=colors['input_colpo'],
                             edgecolor='black', linewidth=1.5, alpha=0.9)
ax.add_patch(colpo_input)
ax.text(1.25, input_y - 0.5, 'Colposcopy\nFeatures', ha='center', va='center',
        fontsize=10, fontweight='bold')
ax.text(1.25, input_y - 0.85, '[B, 512]', ha='center', va='center',
        fontsize=8, family='monospace')

# Clinical Input
clinical_input = FancyBboxPatch((0.5, input_y - 2.0), input_width, input_height,
                                boxstyle="round,pad=0.05",
                                facecolor=colors['input_clinical'],
                                edgecolor='black', linewidth=1.5, alpha=0.9)
ax.add_patch(clinical_input)
ax.text(1.25, input_y - 1.5, 'Clinical\nData', ha='center', va='center',
        fontsize=10, fontweight='bold')
ax.text(1.25, input_y - 1.85, '[B, 7]', ha='center', va='center',
        fontsize=8, family='monospace')

# ==================== Module 1: Multimodal Fusion (Detailed) ====================
fusion_x = 2.8
fusion_y = 12.5
fusion_width = 2.5
fusion_height = 2.0

# Fusion Module Box
fusion_box = FancyBboxPatch((fusion_x, fusion_y), fusion_width, fusion_height,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['fusion'],
                            edgecolor='black', linewidth=2, alpha=0.9)
ax.add_patch(fusion_box)
ax.text(fusion_x + fusion_width/2, fusion_y + 1.8, 'Multimodal Fusion',
        ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# Internal structure
# Weighted Fusion
ax.text(fusion_x + fusion_width/2, fusion_y + 1.4, 'Weighted Fusion',
        ha='center', va='center', fontsize=9, color='white')
ax.text(fusion_x + 0.3, fusion_y + 1.1, 'w₁·OCT', ha='left', va='center',
        fontsize=8, color='white', family='monospace')
ax.text(fusion_x + 0.3, fusion_y + 0.9, '+ w₂·Colpo', ha='left', va='center',
        fontsize=8, color='white', family='monospace')
ax.text(fusion_x + fusion_width/2, fusion_y + 0.6, 'Concat',
        ha='center', va='center', fontsize=9, color='white', fontweight='bold')
ax.text(fusion_x + fusion_width/2, fusion_y + 0.3, '[B, 1024]',
        ha='center', va='center', fontsize=8, color='white', family='monospace')
ax.text(fusion_x + fusion_width/2, fusion_y + 0.05, '→ [B, 768]',
        ha='center', va='center', fontsize=8, color='white', family='monospace')

# ==================== Module 2: Student Prior Network (Detailed) ====================
student_x = 2.8
student_y = 9.5
student_width = 2.5
student_height = 2.5

# Student Prior Box
student_box = FancyBboxPatch((student_x, student_y), student_width, student_height,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['stem'],
                             edgecolor='black', linewidth=2, alpha=0.9)
ax.add_patch(student_box)
ax.text(student_x + student_width/2, student_y + 2.3, 'Student Prior Network',
        ha='center', va='center', fontsize=12, fontweight='bold')

# Internal layers (detailed)
layers = [
    ('Linear(7→256)', colors['linear']),
    ('BatchNorm1d', colors['norm']),
    ('LeakyReLU(0.2)', colors['activation']),
    ('Dropout(0.2)', colors['norm']),
    ('Linear(256→512)', colors['linear']),
    ('BatchNorm1d', colors['norm']),
    ('LeakyReLU(0.2)', colors['activation']),
    ('Dropout(0.2)', colors['norm']),
    ('Linear(512→768)', colors['linear']),
]

layer_height = student_height / (len(layers) + 1)
for i, (layer_name, layer_color) in enumerate(layers):
    y_pos = student_y + student_height - (i + 1) * layer_height - 0.1
    draw_module_box(ax, student_x + 0.1, y_pos, student_width - 0.2, layer_height - 0.05,
                   layer_color, layer_name, fontsize=7, alpha=0.8)

# Output
ax.text(student_x + student_width/2, student_y + 0.05, 'z_sem [B, 768]',
        ha='center', va='center', fontsize=9, fontweight='bold', family='monospace')

# ==================== Module 3: Dual Head Image Encoder (Detailed) ====================
encoder_x = 6.0
encoder_y = 12.0
encoder_width = 3.5
encoder_height = 3.0

# Encoder Box
encoder_box = FancyBboxPatch((encoder_x, encoder_y), encoder_width, encoder_height,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['conv'],
                             edgecolor='black', linewidth=2, alpha=0.9)
ax.add_patch(encoder_box)
ax.text(encoder_x + encoder_width/2, encoder_y + 2.8, 'Dual Head Image Encoder',
        ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# Feature Projection
ax.text(encoder_x + encoder_width/2, encoder_y + 2.4, 'Feature Projection',
        ha='center', va='center', fontsize=10, fontweight='bold', color='white')
proj_layers = ['Linear(512→1536)', 'LayerNorm', 'GELU', 'Dropout(0.1)', 'Linear(1536→768)']
proj_y = encoder_y + 1.8
proj_height = 0.5
for i, layer in enumerate(proj_layers):
    x_pos = encoder_x + 0.2 + i * (encoder_width - 0.4) / len(proj_layers)
    draw_module_box(ax, x_pos, proj_y, (encoder_width - 0.4) / len(proj_layers) - 0.05, proj_height,
                   colors['linear'], layer, fontsize=7, text_color='white', alpha=0.8)

# Dual Heads
head_y = encoder_y + 1.0
head_height = 0.6

# Causal Head
causal_head_x = encoder_x + 0.2
causal_head_width = (encoder_width - 0.5) / 2
draw_module_box(ax, causal_head_x, head_y, causal_head_width, head_height,
               colors['output'], 'Causal Head', fontsize=9, fontweight='bold')
causal_layers = ['Linear(768→768)', 'LayerNorm', 'GELU', 'Dropout(0.1)', 'Linear(768→768)']
for i, layer in enumerate(causal_layers):
    y_pos = head_y + head_height - (i + 1) * (head_height / len(causal_layers))
    draw_module_box(ax, causal_head_x + 0.05, y_pos, causal_head_width - 0.1, 
                   head_height / len(causal_layers) - 0.02,
                   colors['linear'], layer, fontsize=6, alpha=0.7)

# Noise Head
noise_head_x = encoder_x + encoder_width - causal_head_width - 0.2
draw_module_box(ax, noise_head_x, head_y, causal_head_width, head_height,
               colors['memory'], 'Noise Head', fontsize=9, fontweight='bold')
for i, layer in enumerate(causal_layers):  # Same structure
    y_pos = head_y + head_height - (i + 1) * (head_height / len(causal_layers))
    draw_module_box(ax, noise_head_x + 0.05, y_pos, causal_head_width - 0.1,
                   head_height / len(causal_layers) - 0.02,
                   colors['linear'], layer, fontsize=6, alpha=0.7)

# Outputs
ax.text(causal_head_x + causal_head_width/2, encoder_y + 0.2, 'z_causal\n[B, 768]',
        ha='center', va='center', fontsize=9, fontweight='bold', family='monospace')
ax.text(noise_head_x + causal_head_width/2, encoder_y + 0.2, 'z_noise\n[B, 768]',
        ha='center', va='center', fontsize=9, fontweight='bold', family='monospace')

# ==================== Module 4: VLM Encoder (Detailed) ====================
vlm_x = 6.0
vlm_y = 8.5
vlm_width = 3.5
vlm_height = 2.5

# VLM Box
vlm_box = FancyBboxPatch((vlm_x, vlm_y), vlm_width, vlm_height,
                         boxstyle="round,pad=0.1",
                         facecolor=colors['vlm'],
                         edgecolor='black', linewidth=2, alpha=0.9)
ax.add_patch(vlm_box)
ax.text(vlm_x + vlm_width/2, vlm_y + 2.3, 'VLM Image Encoder (Optional)',
        ha='center', va='center', fontsize=12, fontweight='bold', color='white')
ax.text(vlm_x + vlm_width/2, vlm_y + 2.0, 'Qwen2-VL-2B-Instruct',
        ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# VLM Internal Structure
vlm_layers = [
    ('Image Input\n[B, C, H, W]', colors['input']),
    ('VLM Processor\n(Text + Image)', colors['vlm']),
    ('Hidden States\n[B, seq_len, 2048]', colors['linear']),
    ('[CLS] Token\n[B, 2048]', colors['attention']),
    ('Linear(2048→768)', colors['linear']),
    ('Dual Heads\n(Same as above)', colors['conv']),
]

vlm_layer_height = vlm_height / (len(vlm_layers) + 1)
for i, (layer_name, layer_color) in enumerate(vlm_layers):
    y_pos = vlm_y + vlm_height - (i + 1) * vlm_layer_height - 0.1
    draw_module_box(ax, vlm_x + 0.1, y_pos, vlm_width - 0.2, vlm_layer_height - 0.05,
                   layer_color, layer_name, fontsize=7, text_color='white', alpha=0.8)

# ==================== Module 5: Memory Bank (Detailed) ====================
memory_x = 6.0
memory_y = 5.5
memory_width = 3.5
memory_height = 2.0

# Memory Bank Box
memory_box = FancyBboxPatch((memory_x, memory_y), memory_width, memory_height,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['memory'],
                            edgecolor='black', linewidth=2, alpha=0.9)
ax.add_patch(memory_box)
ax.text(memory_x + memory_width/2, memory_y + 1.8, 'Noise Memory Bank',
        ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# Internal structure
ax.text(memory_x + memory_width/2, memory_y + 1.4, 'FIFO Queue per Center',
        ha='center', va='center', fontsize=9, color='white')
ax.text(memory_x + 0.2, memory_y + 1.0, 'Update: z_noise + center_id',
        ha='left', va='center', fontsize=8, color='white', family='monospace')
ax.text(memory_x + 0.2, memory_y + 0.6, 'Sample: Random/Mean/Nearest',
        ha='left', va='center', fontsize=8, color='white', family='monospace')
ax.text(memory_x + memory_width/2, memory_y + 0.2, 'z_noise_cf [B, 768]',
        ha='center', va='center', fontsize=9, fontweight='bold', color='white', family='monospace')

# ==================== Module 6: Three-Modal Fusion ====================
multimodal_x = 10.5
multimodal_y = 12.0
multimodal_width = 2.5
multimodal_height = 2.0

multimodal_box = FancyBboxPatch((multimodal_x, multimodal_y), multimodal_width, multimodal_height,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['fusion'],
                                edgecolor='black', linewidth=2, alpha=0.9)
ax.add_patch(multimodal_box)
ax.text(multimodal_x + multimodal_width/2, multimodal_y + 1.8, 'Three-Modal Fusion',
        ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# Internal structure
ax.text(multimodal_x + multimodal_width/2, multimodal_y + 1.4, 'Concat',
        ha='center', va='center', fontsize=10, fontweight='bold', color='white')
ax.text(multimodal_x + 0.2, multimodal_y + 1.0, 'z_causal ⊕ z_sem',
        ha='left', va='center', fontsize=9, color='white', family='monospace')
ax.text(multimodal_x + multimodal_width/2, multimodal_y + 0.6, 'Linear(1536→768)',
        ha='center', va='center', fontsize=9, color='white', family='monospace')
ax.text(multimodal_x + multimodal_width/2, multimodal_y + 0.3, 'LayerNorm + GELU',
        ha='center', va='center', fontsize=8, color='white', family='monospace')
ax.text(multimodal_x + multimodal_width/2, multimodal_y + 0.05, '[B, 768]',
        ha='center', va='center', fontsize=8, color='white', family='monospace')

# ==================== Module 7: Classifier (Detailed) ====================
classifier_x = 10.5
classifier_y = 9.0
classifier_width = 2.5
classifier_height = 2.5

classifier_box = FancyBboxPatch((classifier_x, classifier_y), classifier_width, classifier_height,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['output'],
                                edgecolor='black', linewidth=2, alpha=0.9)
ax.add_patch(classifier_box)
ax.text(classifier_x + classifier_width/2, classifier_y + 2.3, 'Classifier',
        ha='center', va='center', fontsize=12, fontweight='bold')

# Internal layers
cls_layers = [
    ('Linear(768→768)', colors['linear']),
    ('LayerNorm', colors['norm']),
    ('GELU', colors['activation']),
    ('Dropout(0.5)', colors['norm']),
    ('Linear(768→384)', colors['linear']),
    ('LayerNorm', colors['norm']),
    ('GELU', colors['activation']),
    ('Dropout(0.4)', colors['norm']),
    ('Linear(384→2)', colors['linear']),
]

cls_layer_height = classifier_height / (len(cls_layers) + 1)
for i, (layer_name, layer_color) in enumerate(cls_layers):
    y_pos = classifier_y + classifier_height - (i + 1) * cls_layer_height - 0.1
    draw_module_box(ax, classifier_x + 0.1, y_pos, classifier_width - 0.2, cls_layer_height - 0.05,
                   layer_color, layer_name, fontsize=7, alpha=0.8)

# Output
ax.text(classifier_x + classifier_width/2, classifier_y + 0.05, 'Logits [B, 2]',
        ha='center', va='center', fontsize=9, fontweight='bold', family='monospace')

# ==================== Module 8: Sinkhorn OT Loss (Detailed) ====================
sinkhorn_x = 13.8
sinkhorn_y = 12.0
sinkhorn_width = 3.0
sinkhorn_height = 2.0

sinkhorn_box = FancyBboxPatch((sinkhorn_x, sinkhorn_y), sinkhorn_width, sinkhorn_height,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['loss'],
                              edgecolor='black', linewidth=2, alpha=0.9)
ax.add_patch(sinkhorn_box)
ax.text(sinkhorn_x + sinkhorn_width/2, sinkhorn_y + 1.8, 'Sinkhorn OT Loss',
        ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# Algorithm steps
ax.text(sinkhorn_x + sinkhorn_width/2, sinkhorn_y + 1.4, '1. Normalize Features',
        ha='center', va='center', fontsize=9, color='white')
ax.text(sinkhorn_x + sinkhorn_width/2, sinkhorn_y + 1.1, '2. Cost Matrix C',
        ha='center', va='center', fontsize=9, color='white')
ax.text(sinkhorn_x + sinkhorn_width/2, sinkhorn_y + 0.8, '3. Sinkhorn Iter (100)',
        ha='center', va='center', fontsize=9, color='white')
ax.text(sinkhorn_x + sinkhorn_width/2, sinkhorn_y + 0.5, '4. Transport Plan γ',
        ha='center', va='center', fontsize=9, color='white')
ax.text(sinkhorn_x + sinkhorn_width/2, sinkhorn_y + 0.2, 'L_OT = Σ(γ ⊙ C)',
        ha='center', va='center', fontsize=9, fontweight='bold', color='white', family='monospace')

# ==================== Arrows (Data Flow) ====================
# Input to Fusion
draw_arrow(ax, 2.0, input_y + 0.4, fusion_x, fusion_y + 1.5, colors['skip'], width=2)
draw_arrow(ax, 2.0, input_y - 0.6, fusion_x, fusion_y + 0.5, colors['skip'], width=2)

# Fusion to Encoder
draw_arrow(ax, fusion_x + fusion_width, fusion_y + 1.0, encoder_x, encoder_y + 1.5, 
          colors['skip'], width=2.5)

# Clinical to Student Prior
draw_arrow(ax, 2.0, input_y - 1.6, student_x, student_y + 2.0, colors['skip'], width=2)

# Student Prior to z_sem
draw_arrow(ax, student_x + student_width, student_y + 1.25, multimodal_x, multimodal_y + 0.8,
          colors['skip'], width=2, connection="arc3,rad=0.2")

# Encoder to z_causal, z_noise
draw_arrow(ax, encoder_x + encoder_width, encoder_y + 0.5, multimodal_x, multimodal_y + 1.5,
          colors['skip'], width=2.5)
draw_arrow(ax, encoder_x + encoder_width, encoder_y + 0.5, memory_x, memory_y + 1.5,
          colors['skip'], width=2, connection="arc3,rad=-0.2")

# Three-Modal Fusion to Classifier
draw_arrow(ax, multimodal_x + multimodal_width/2, multimodal_y, 
          classifier_x + classifier_width/2, classifier_y + classifier_height,
          colors['skip'], width=2.5)

# Classifier to Output
output_x = 13.8
output_y = 9.5
output_box = FancyBboxPatch((output_x, output_y), 1.5, 0.8,
                            boxstyle="round,pad=0.05",
                            facecolor=colors['output'],
                            edgecolor='black', linewidth=2, alpha=0.95)
ax.add_patch(output_box)
ax.text(output_x + 0.75, output_y + 0.4, 'Logits\n[B, 2]',
        ha='center', va='center', fontsize=10, fontweight='bold', family='monospace')

draw_arrow(ax, classifier_x + classifier_width, classifier_y + 1.25, output_x, output_y + 0.4,
          colors['skip'], width=2.5)

# Loss arrows (dashed)
draw_arrow(ax, encoder_x + encoder_width/2, encoder_y + 0.5, sinkhorn_x, sinkhorn_y + 1.5,
          colors['loss'], width=1.5, style='--', alpha=0.7, connection="arc3,rad=0.3")
draw_arrow(ax, student_x + student_width/2, student_y + 1.25, sinkhorn_x, sinkhorn_y + 1.0,
          colors['loss'], width=1.5, style='--', alpha=0.7, connection="arc3,rad=-0.3")

# ==================== Skip Connections Visualization ====================
# Show skip connection in Dual Head Encoder
skip_y = encoder_y + 1.5
draw_arrow(ax, encoder_x + 0.3, skip_y, encoder_x + encoder_width - 0.3, skip_y,
          colors['skip'], width=1.5, style=':', alpha=0.5)
ax.text(encoder_x + encoder_width/2, skip_y + 0.1, 'Skip',
        ha='center', va='bottom', fontsize=7, style='italic', color=colors['skip'])

# ==================== Center Discriminator ====================
disc_x = 13.8
disc_y = 9.0
disc_width = 3.0
disc_height = 1.5

disc_box = FancyBboxPatch((disc_x, disc_y), disc_width, disc_height,
                          boxstyle="round,pad=0.1",
                          facecolor=colors['memory'],
                          edgecolor='black', linewidth=2, alpha=0.9)
ax.add_patch(disc_box)
ax.text(disc_x + disc_width/2, disc_y + 1.3, 'Center Discriminator',
        ha='center', va='center', fontsize=11, fontweight='bold', color='white')
ax.text(disc_x + disc_width/2, disc_y + 0.8, 'Linear(768→256→128→5)',
        ha='center', va='center', fontsize=9, color='white', family='monospace')
ax.text(disc_x + disc_width/2, disc_y + 0.3, 'L_adv = CE(D(z_noise), center_id)',
        ha='center', va='center', fontsize=8, color='white', family='monospace')

# ==================== Total Loss Formula ====================
loss_formula_y = 1.5
loss_formula_box = FancyBboxPatch((2.0, loss_formula_y), 20.0, 1.0,
                                  boxstyle="round,pad=0.15",
                                  facecolor='white',
                                  edgecolor='black', linewidth=2.5, alpha=0.95)
ax.add_patch(loss_formula_box)
ax.text(12, loss_formula_y + 0.65,
        r'$\mathcal{L}_{total} = \lambda_{cls} \cdot \mathcal{L}_{cls} + \lambda_{ot} \cdot \mathcal{L}_{ot} + \lambda_{consist} \cdot \mathcal{L}_{consist} + \lambda_{adv} \cdot \mathcal{L}_{adv}$',
        ha='center', va='center', fontsize=14, fontweight='bold')

ax.text(12, loss_formula_y + 0.15,
        r'$\lambda_{cls}=1.0, \quad \lambda_{ot}=1.0, \quad \lambda_{consist}=0.5, \quad \lambda_{adv}=0.1$',
        ha='center', va='center', fontsize=11, style='italic')

# ==================== Legend ====================
legend_x = 0.5
legend_y = 0.5
legend_width = 3.0
legend_height = 0.8

legend_box = FancyBboxPatch((legend_x, legend_y), legend_width, legend_height,
                            boxstyle="round,pad=0.1",
                            facecolor='white',
                            edgecolor='black', linewidth=1.5, alpha=0.9)
ax.add_patch(legend_box)
ax.text(legend_x + legend_width/2, legend_y + 0.6, 'Legend',
        ha='center', va='center', fontsize=10, fontweight='bold')

legend_elements = [
    mpatches.Patch(facecolor=colors['skip'], edgecolor='black', label='Data Flow'),
    mpatches.Patch(facecolor=colors['loss'], edgecolor='black', label='Loss Flow', linestyle='--'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=8, framealpha=0.9,
          bbox_to_anchor=(legend_x + 0.1, legend_y + 0.1))

# ==================== Save ====================
output_dir = '/data2/hmy/VLM_Caus_Rm_Mics/docs/figures'
os.makedirs(output_dir, exist_ok=True)

# Save as PDF
pdf_path = os.path.join(output_dir, 'BioCOT_Architecture_Standard_SCI.pdf')
plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white',
            edgecolor='none', pad_inches=0.1)
print(f"✅ PDF架构图已保存: {pdf_path}")

# Save as PNG
png_path = os.path.join(output_dir, 'BioCOT_Architecture_Standard_SCI.png')
plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white',
            edgecolor='none', pad_inches=0.1)
print(f"✅ PNG架构图已保存: {png_path}")

# Save as SVG
svg_path = os.path.join(output_dir, 'BioCOT_Architecture_Standard_SCI.svg')
plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white',
            edgecolor='none', pad_inches=0.1)
print(f"✅ SVG架构图已保存: {svg_path}")

plt.close()
print("\n🎉 标准SCI风格Bio-COT模型架构图生成完成！")
print("📄 所有格式已保存到: docs/figures/")
print("   - PDF: 适合论文提交（矢量格式）")
print("   - PNG: 高分辨率预览（300 DPI）")
print("   - SVG: 可缩放矢量图（可编辑）")

