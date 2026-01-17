#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT Model Architecture Diagram for Scientific Paper
Professional visualization with complete details for reviewers
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch, Rectangle, Circle
import numpy as np
import os

# Set high DPI for publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

# Create figure (16:10 ratio, suitable for 2-column paper)
fig = plt.figure(figsize=(20, 13))
ax = fig.add_subplot(111)
ax.set_xlim(0, 20)
ax.set_ylim(0, 13)
ax.axis('off')

# Professional color scheme (colorblind-friendly, publication-ready)
colors = {
    # Input modules
    'input_oct': '#4A90E2',           # Blue
    'input_colpo': '#50C878',          # Green
    'input_clinical': '#E74C3C',       # Red
    'input_center': '#95A5A6',         # Gray
    
    # Processing modules
    'student_prior': '#F39C12',        # Orange
    'encoder': '#3498DB',              # Light Blue
    'vlm_encoder': '#9B59B6',          # Purple
    'fusion': '#1ABC9C',               # Teal
    
    # Feature representations
    'z_causal': '#2ECC71',            # Green
    'z_noise': '#E67E22',             # Orange-Red
    'z_sem': '#F1C40F',               # Yellow
    
    # Advanced modules
    'memory_bank': '#E74C3C',          # Red
    'discriminator': '#8E44AD',       # Dark Purple
    'classifier': '#16A085',          # Dark Teal
    
    # Loss functions
    'loss': '#34495E',                 # Dark Gray
    
    # Output
    'output': '#2C3E50',               # Very Dark Gray
    
    # Arrows
    'arrow_main': '#2C3E50',           # Dark Blue-Black
    'arrow_loss': '#E67E22',           # Orange
    'arrow_aux': '#7F8C8D',           # Gray
    
    # Highlights
    'highlight': '#E74C3C',            # Red
    'innovation': '#27AE60'           # Green
}

# Font settings
title_font = {'fontsize': 20, 'fontweight': 'bold', 'ha': 'center'}
subtitle_font = {'fontsize': 14, 'style': 'italic', 'ha': 'center'}
section_font = {'fontsize': 12, 'fontweight': 'bold', 'ha': 'center'}
label_font = {'fontsize': 10, 'ha': 'center'}
formula_font = {'fontsize': 9, 'family': 'monospace', 'style': 'italic', 'ha': 'center'}
small_font = {'fontsize': 8, 'ha': 'center'}

# ==================== Title ====================
ax.text(10, 12.5, 'Bio-COT: Biological Causal Optimal Transport Framework', 
        **title_font, va='center')
ax.text(10, 12.0, 'for Multi-Center Cervical Lesion Classification', 
        **subtitle_font, va='center')

# ==================== Input Layer (Left) ====================
input_x = 0.5
input_width = 2.0
input_height = 0.9

# OCT Features
oct_y = 10.0
oct_box = FancyBboxPatch((input_x, oct_y), input_width, input_height,
                         boxstyle="round,pad=0.1", 
                         facecolor=colors['input_oct'],
                         edgecolor='black', linewidth=2, alpha=0.85)
ax.add_patch(oct_box)
ax.text(input_x + input_width/2, oct_y + 0.6, 'OCT Features', 
        **section_font, color='white', va='center')
ax.text(input_x + input_width/2, oct_y + 0.3, '[B, 512]', 
        **formula_font, color='white', va='center')
ax.text(input_x + input_width/2, oct_y + 0.05, 'or [B, F, C, H, W]', 
        **small_font, color='white', va='center')

# Colposcopy Features
colpo_y = 8.8
colpo_box = FancyBboxPatch((input_x, colpo_y), input_width, input_height,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['input_colpo'],
                            edgecolor='black', linewidth=2, alpha=0.85)
ax.add_patch(colpo_box)
ax.text(input_x + input_width/2, colpo_y + 0.6, 'Colposcopy', 
        **section_font, color='white', va='center')
ax.text(input_x + input_width/2, colpo_y + 0.3, 'Features', 
        **section_font, color='white', va='center')
ax.text(input_x + input_width/2, colpo_y + 0.05, '[B, 512]', 
        **formula_font, color='white', va='center')

# Clinical Data
clinical_y = 7.6
clinical_box = FancyBboxPatch((input_x, clinical_y), input_width, input_height,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['input_clinical'],
                               edgecolor='black', linewidth=2, alpha=0.85)
ax.add_patch(clinical_box)
ax.text(input_x + input_width/2, clinical_y + 0.6, 'Clinical Data', 
        **section_font, color='white', va='center')
ax.text(input_x + input_width/2, clinical_y + 0.3, 'HPV, TCT, Age', 
        **label_font, color='white', va='center')
ax.text(input_x + input_width/2, clinical_y + 0.05, '[B, 7]', 
        **formula_font, color='white', va='center')

# Center Labels
center_y = 6.4
center_box = FancyBboxPatch((input_x, center_y), input_width, input_height,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['input_center'],
                            edgecolor='black', linewidth=2, alpha=0.7)
ax.add_patch(center_box)
ax.text(input_x + input_width/2, center_y + 0.5, 'Center ID', 
        **label_font, color='white', va='center')
ax.text(input_x + input_width/2, center_y + 0.05, '[B]', 
        **formula_font, color='white', va='center')

# ==================== Module 1: Multimodal Fusion ====================
fusion_x = 3.2
fusion_y = 9.0
fusion_width = 2.8
fusion_height = 1.8

fusion_box = FancyBboxPatch((fusion_x, fusion_y), fusion_width, fusion_height,
                             boxstyle="round,pad=0.12",
                             facecolor=colors['fusion'],
                             edgecolor='black', linewidth=2.5, alpha=0.9)
ax.add_patch(fusion_box)
ax.text(fusion_x + fusion_width/2, fusion_y + 1.4, 'Multimodal Fusion', 
        **section_font, color='white', va='center')
ax.text(fusion_x + fusion_width/2, fusion_y + 1.0, 'Weighted + Concat', 
        **label_font, color='white', va='center')
ax.text(fusion_x + fusion_width/2, fusion_y + 0.6, 'Learnable Weights', 
        **small_font, color='white', va='center')
ax.text(fusion_x + fusion_width/2, fusion_y + 0.3, '[B, 512×2] → [B, 768]', 
        **formula_font, color='white', va='center')

# ==================== Module 2: Student Prior Network ====================
student_x = 3.2
student_y = 6.5
student_width = 2.8
student_height = 1.8

student_box = FancyBboxPatch((student_x, student_y), student_width, student_height,
                             boxstyle="round,pad=0.12",
                             facecolor=colors['student_prior'],
                             edgecolor='black', linewidth=2.5, alpha=0.9)
ax.add_patch(student_box)
ax.text(student_x + student_width/2, student_y + 1.4, 'Student Prior Network', 
        **section_font, va='center')
ax.text(student_x + student_width/2, student_y + 1.0, 'MLP: 7→256→512→768', 
        **formula_font, va='center')
ax.text(student_x + student_width/2, student_y + 0.6, 'Replaces Online VLM', 
        **small_font, style='italic', va='center')
ax.text(student_x + student_width/2, student_y + 0.2, 'Fast Training', 
        **small_font, style='italic', va='center')

# Output: z_sem
z_sem_x = 6.5
z_sem_y = 7.4
z_sem_width = 1.5
z_sem_height = 1.0

z_sem_box = FancyBboxPatch((z_sem_x, z_sem_y), z_sem_width, z_sem_height,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['z_sem'],
                           edgecolor='black', linewidth=2, alpha=0.9)
ax.add_patch(z_sem_box)
ax.text(z_sem_x + z_sem_width/2, z_sem_y + 0.7, 'z_sem', 
        **section_font, color='black', va='center')
ax.text(z_sem_x + z_sem_width/2, z_sem_y + 0.4, 'Semantic', 
        **label_font, color='black', va='center')
ax.text(z_sem_x + z_sem_width/2, z_sem_y + 0.1, '[B, 768]', 
        **formula_font, color='black', va='center')

# ==================== Module 3: Image Encoder (Two Modes) ====================
encoder_x = 6.5
encoder_y = 9.3
encoder_width = 3.5
encoder_height = 1.5

encoder_box = FancyBboxPatch((encoder_x, encoder_y), encoder_width, encoder_height,
                             boxstyle="round,pad=0.12",
                             facecolor=colors['encoder'],
                             edgecolor='black', linewidth=2.5, alpha=0.9)
ax.add_patch(encoder_box)
ax.text(encoder_x + encoder_width/2, encoder_y + 1.1, 'Image Encoder', 
        **section_font, color='white', va='center')
ax.text(encoder_x + encoder_width/2, encoder_y + 0.85, 'Mode A: VLM Encoder (Qwen2-VL-2B)', 
        ha='center', va='center', fontsize=9, color='white', weight='bold')
ax.text(encoder_x + encoder_width/2, encoder_y + 0.6, 'Mode B: MLP Encoder (512→1536→768)', 
        ha='center', va='center', fontsize=9, color='white')
ax.text(encoder_x + encoder_width/2, encoder_y + 0.3, 'Dual Heads → z_causal, z_noise', 
        ha='center', va='center', fontsize=8, color='white', family='monospace', style='italic')

# VLM Encoder Detail Box (Optional, shown as alternative path)
vlm_detail_x = 0.5
vlm_detail_y = 4.8
vlm_detail_width = 2.0
vlm_detail_height = 1.2

vlm_detail_box = FancyBboxPatch((vlm_detail_x, vlm_detail_y), vlm_detail_width, vlm_detail_height,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['vlm_encoder'],
                                 edgecolor='black', linewidth=2.5, alpha=0.9)
ax.add_patch(vlm_detail_box)
ax.text(vlm_detail_x + vlm_detail_width/2, vlm_detail_y + 0.9, 'VLM Encoder', 
        **section_font, color='white', va='center')
ax.text(vlm_detail_x + vlm_detail_width/2, vlm_detail_y + 0.6, 'Qwen2-VL-2B', 
        ha='center', va='center', fontsize=9, color='white', weight='bold')
ax.text(vlm_detail_x + vlm_detail_width/2, vlm_detail_y + 0.3, 'Instruct', 
        ha='center', va='center', fontsize=9, color='white', weight='bold')
ax.text(vlm_detail_x + vlm_detail_width/2, vlm_detail_y + 0.05, 'Frozen', 
        ha='center', va='center', fontsize=8, color='white', style='italic')

# Causal Head Output
causal_x = 10.5
causal_y = 9.7
causal_width = 1.4
causal_height = 0.6

causal_box = FancyBboxPatch((causal_x, causal_y), causal_width, causal_height,
                            boxstyle="round,pad=0.08",
                            facecolor=colors['z_causal'],
                            edgecolor='black', linewidth=2, alpha=0.9)
ax.add_patch(causal_box)
ax.text(causal_x + causal_width/2, causal_y + 0.4, 'z_causal', 
        **section_font, color='white', va='center')
ax.text(causal_x + causal_width/2, causal_y + 0.1, '[B, 768]', 
        **formula_font, color='white', va='center')

# Noise Head Output
noise_x = 10.5
noise_y = 8.5
noise_width = 1.4
noise_height = 0.6

noise_box = FancyBboxPatch((noise_x, noise_y), noise_width, noise_height,
                           boxstyle="round,pad=0.08",
                           facecolor=colors['z_noise'],
                           edgecolor='black', linewidth=2, alpha=0.9)
ax.add_patch(noise_box)
ax.text(noise_x + noise_width/2, noise_y + 0.4, 'z_noise', 
        **section_font, color='white', va='center')
ax.text(noise_x + noise_width/2, noise_y + 0.1, '[B, 768]', 
        **formula_font, color='white', va='center')

# ==================== Module 4: Memory Bank ====================
memory_x = 6.5
memory_y = 5.0
memory_width = 3.5
memory_height = 1.3

memory_box = FancyBboxPatch((memory_x, memory_y), memory_width, memory_height,
                             boxstyle="round,pad=0.12",
                             facecolor=colors['memory_bank'],
                             edgecolor='black', linewidth=2.5, alpha=0.9)
ax.add_patch(memory_box)
ax.text(memory_x + memory_width/2, memory_y + 0.9, 'Noise Memory Bank', 
        **section_font, color='white', va='center')
ax.text(memory_x + memory_width/2, memory_y + 0.5, 'FIFO Queue per Center', 
        **label_font, color='white', va='center')
ax.text(memory_x + memory_width/2, memory_y + 0.1, 'Capacity: 100/center', 
        **formula_font, color='white', va='center')

# Counterfactual Noise Output
cf_x = 10.5
cf_y = 5.2
cf_width = 1.4
cf_height = 0.9

cf_box = FancyBboxPatch((cf_x, cf_y), cf_width, cf_height,
                        boxstyle="round,pad=0.08",
                        facecolor=colors['z_noise'],
                        edgecolor='black', linewidth=2, alpha=0.8)
ax.add_patch(cf_box)
ax.text(cf_x + cf_width/2, cf_y + 0.7, 'z_noise_cf', 
        **section_font, color='white', va='center')
ax.text(cf_x + cf_width/2, cf_y + 0.4, 'Counter-', 
        **label_font, color='white', va='center')
ax.text(cf_x + cf_width/2, cf_y + 0.1, 'factual', 
        **label_font, color='white', va='center')

# ==================== Module 5: Center Discriminator ====================
disc_x = 6.5
disc_y = 3.5
disc_width = 3.5
disc_height = 1.0

disc_box = FancyBboxPatch((disc_x, disc_y), disc_width, disc_height,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['discriminator'],
                           edgecolor='black', linewidth=2, alpha=0.85)
ax.add_patch(disc_box)
ax.text(disc_x + disc_width/2, disc_y + 0.6, 'Center Discriminator', 
        **section_font, color='white', va='center')
ax.text(disc_x + disc_width/2, disc_y + 0.2, '768 → 256 → 128 → 5', 
        **formula_font, color='white', va='center')

# ==================== Module 6: Three-Modal Fusion ====================
multimodal_x = 12.5
multimodal_y = 8.0
multimodal_width = 2.8
multimodal_height = 1.8

multimodal_box = FancyBboxPatch((multimodal_x, multimodal_y), multimodal_width, multimodal_height,
                                boxstyle="round,pad=0.12",
                                facecolor=colors['fusion'],
                                edgecolor='black', linewidth=2.5, alpha=0.9)
ax.add_patch(multimodal_box)
ax.text(multimodal_x + multimodal_width/2, multimodal_y + 1.4, 'Three-Modal', 
        **section_font, color='white', va='center')
ax.text(multimodal_x + multimodal_width/2, multimodal_y + 1.0, 'Fusion', 
        **section_font, color='white', va='center')
ax.text(multimodal_x + multimodal_width/2, multimodal_y + 0.6, 'z_causal ⊕ z_sem', 
        **formula_font, color='white', va='center')
ax.text(multimodal_x + multimodal_width/2, multimodal_y + 0.2, '[B, 1536] → [B, 768]', 
        **formula_font, color='white', va='center')

# ==================== Module 7: Classifier ====================
classifier_x = 12.5
classifier_y = 5.5
classifier_width = 2.8
classifier_height = 1.8

classifier_box = FancyBboxPatch((classifier_x, classifier_y), classifier_width, classifier_height,
                                boxstyle="round,pad=0.12",
                                facecolor=colors['classifier'],
                                edgecolor='black', linewidth=2.5, alpha=0.9)
ax.add_patch(classifier_box)
ax.text(classifier_x + classifier_width/2, classifier_y + 1.4, 'Classifier', 
        **section_font, color='white', va='center')
ax.text(classifier_x + classifier_width/2, classifier_y + 1.0, 'MLP + Dropout', 
        **label_font, color='white', va='center')
ax.text(classifier_x + classifier_width/2, classifier_y + 0.6, '768 → 768 → 384 → 2', 
        **formula_font, color='white', va='center')
ax.text(classifier_x + classifier_width/2, classifier_y + 0.2, 'Dropout: 0.5, 0.4', 
        **formula_font, color='white', va='center')

# Output: Logits
logits_x = 15.8
logits_y = 6.4
logits_width = 1.2
logits_height = 1.0

logits_box = FancyBboxPatch((logits_x, logits_y), logits_width, logits_height,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['output'],
                             edgecolor='black', linewidth=2.5, alpha=0.95)
ax.add_patch(logits_box)
ax.text(logits_x + logits_width/2, logits_y + 0.7, 'Logits', 
        **section_font, color='white', va='center')
ax.text(logits_x + logits_width/2, logits_y + 0.3, '[B, 2]', 
        **formula_font, color='white', va='center')

# ==================== Loss Functions (Right Side) ====================
loss_x = 12.5
loss_width = 4.5
loss_height = 0.7

# Sinkhorn OT Loss
ot_y = 10.5
ot_box = FancyBboxPatch((loss_x, ot_y), loss_width, loss_height,
                        boxstyle="round,pad=0.08",
                        facecolor=colors['loss'],
                        edgecolor='black', linewidth=1.5, alpha=0.85)
ax.add_patch(ot_box)
ax.text(loss_x + loss_width/2, ot_y + 0.45, r'$\mathcal{L}_{OT} = \text{Sinkhorn}(z_{causal}, z_{sem})$', 
        ha='center', va='center', fontsize=10, color='white', weight='bold')
ax.text(loss_x + loss_width/2, ot_y + 0.1, r'$\lambda_{OT} = 1.0$', 
        ha='center', va='center', fontsize=9, color='white', style='italic')

# Counterfactual Consistency Loss
consist_y = 9.5
consist_box = FancyBboxPatch((loss_x, consist_y), loss_width, loss_height,
                             boxstyle="round,pad=0.08",
                             facecolor=colors['loss'],
                             edgecolor='black', linewidth=1.5, alpha=0.85)
ax.add_patch(consist_box)
ax.text(loss_x + loss_width/2, consist_y + 0.45, r'$\mathcal{L}_{consist} = \text{MSE}(\text{logits}, \text{logits}_{cf})$', 
        ha='center', va='center', fontsize=10, color='white', weight='bold')
ax.text(loss_x + loss_width/2, consist_y + 0.1, r'$\lambda_{consist} = 0.5$', 
        ha='center', va='center', fontsize=9, color='white', style='italic')

# Adversarial Loss
adv_y = 8.5
adv_box = FancyBboxPatch((loss_x, adv_y), loss_width, loss_height,
                          boxstyle="round,pad=0.08",
                          facecolor=colors['loss'],
                          edgecolor='black', linewidth=1.5, alpha=0.85)
ax.add_patch(adv_box)
ax.text(loss_x + loss_width/2, adv_y + 0.45, r'$\mathcal{L}_{adv} = \text{CE}(D(z_{noise}), \text{center\_id})$', 
        ha='center', va='center', fontsize=10, color='white', weight='bold')
ax.text(loss_x + loss_width/2, adv_y + 0.1, r'$\lambda_{adv} = 0.1$', 
        ha='center', va='center', fontsize=9, color='white', style='italic')

# Classification Loss
cls_y = 7.5
cls_box = FancyBboxPatch((loss_x, cls_y), loss_width, loss_height,
                         boxstyle="round,pad=0.08",
                         facecolor=colors['loss'],
                         edgecolor='black', linewidth=1.5, alpha=0.85)
ax.add_patch(cls_box)
ax.text(loss_x + loss_width/2, cls_y + 0.45, r'$\mathcal{L}_{cls} = \text{CE}(\text{logits}, \text{labels})$', 
        ha='center', va='center', fontsize=10, color='white', weight='bold')
ax.text(loss_x + loss_width/2, cls_y + 0.1, r'$\lambda_{cls} = 1.0$', 
        ha='center', va='center', fontsize=9, color='white', style='italic')

# ==================== Main Data Flow Arrows ====================
def draw_arrow(x1, y1, x2, y2, color, width=2.5, style='-', alpha=1.0, connection=None):
    """Helper function to draw arrows"""
    if connection:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', lw=width, color=color,
                               linestyle=style, alpha=alpha,
                               connectionstyle=connection)
    else:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', lw=width, color=color,
                               linestyle=style, alpha=alpha)
    ax.add_patch(arrow)

# Input to Fusion
draw_arrow(2.5, oct_y + 0.45, fusion_x, fusion_y + 1.2, colors['arrow_main'], connection="arc3,rad=0.1")
draw_arrow(2.5, colpo_y + 0.45, fusion_x, fusion_y + 0.6, colors['arrow_main'], connection="arc3,rad=-0.1")

# Fusion to Encoder (Main path: MLP mode)
draw_arrow(fusion_x + fusion_width, fusion_y + 0.9, encoder_x, encoder_y + 0.75, colors['arrow_main'])
# Note: VLM path shown separately above

# Encoder to Dual Heads
draw_arrow(encoder_x + encoder_width, encoder_y + 0.75, causal_x, causal_y + 0.3, colors['arrow_main'])
draw_arrow(encoder_x + encoder_width, encoder_y + 0.75, noise_x, noise_y + 0.3, colors['arrow_main'])

# Clinical to Student Prior
draw_arrow(2.5, clinical_y + 0.45, student_x, student_y + 0.9, colors['arrow_main'])

# Student Prior to z_sem
draw_arrow(student_x + student_width, student_y + 0.9, z_sem_x, z_sem_y + 0.5, colors['arrow_main'])

# z_noise to Memory Bank
draw_arrow(noise_x, noise_y + 0.3, noise_x, memory_y + 0.65, colors['arrow_main'], connection="arc3,rad=0.2")
draw_arrow(noise_x, memory_y + 0.65, memory_x + memory_width, memory_y + 0.65, colors['arrow_main'])

# Center Labels to Memory Bank
draw_arrow(2.5, center_y + 0.45, memory_x, memory_y + 0.65, colors['arrow_aux'], style='--', alpha=0.6)

# Memory Bank to Counterfactual
draw_arrow(memory_x + memory_width, memory_y + 0.65, cf_x, cf_y + 0.45, colors['arrow_main'])

# z_causal and z_sem to Three-Modal Fusion
draw_arrow(causal_x + causal_width, causal_y + 0.3, multimodal_x, multimodal_y + 1.2, 
           colors['arrow_main'], connection="arc3,rad=0.2")
draw_arrow(z_sem_x + z_sem_width, z_sem_y + 0.5, multimodal_x, multimodal_y + 0.6, 
           colors['arrow_main'], connection="arc3,rad=-0.2")

# Three-Modal Fusion to Classifier
draw_arrow(multimodal_x + multimodal_width/2, multimodal_y, 
           classifier_x + classifier_width/2, classifier_y + classifier_height, 
           colors['arrow_main'])

# Classifier to Logits
draw_arrow(classifier_x + classifier_width, classifier_y + 0.9, logits_x, logits_y + 0.5, colors['arrow_main'])

# ==================== Loss Function Arrows (Dashed) ====================
# z_causal -> OT Loss
draw_arrow(causal_x + causal_width/2, causal_y, loss_x + 1.0, ot_y + 0.35, 
           colors['arrow_loss'], width=1.5, style='--', alpha=0.7, connection="arc3,rad=0.3")
# z_sem -> OT Loss
draw_arrow(z_sem_x + z_sem_width/2, z_sem_y, loss_x + 1.0, ot_y + 0.35, 
           colors['arrow_loss'], width=1.5, style='--', alpha=0.7, connection="arc3,rad=-0.3")

# Counterfactual -> Consistency Loss
draw_arrow(cf_x + cf_width/2, cf_y + cf_height, loss_x + 1.0, consist_y + 0.35, 
           colors['arrow_loss'], width=1.5, style='--', alpha=0.7, connection="arc3,rad=0.2")

# z_noise -> Adversarial Loss
draw_arrow(noise_x + noise_width/2, noise_y, disc_x + disc_width/2, disc_y + disc_height, 
           colors['arrow_main'], width=2)
draw_arrow(disc_x + disc_width, disc_y + disc_height/2, loss_x + 1.0, adv_y + 0.35, 
           colors['arrow_loss'], width=1.5, style='--', alpha=0.7)

# Logits -> Classification Loss
draw_arrow(logits_x, logits_y + 0.5, loss_x + 1.0, cls_y + 0.35, 
           colors['arrow_loss'], width=1.5, style='--', alpha=0.7, connection="arc3,rad=-0.2")

# ==================== Innovation Highlights ====================
# Innovation 1: Student Prior
innovation1 = FancyBboxPatch((student_x - 0.2, student_y - 0.2), 
                              student_width + 0.4, student_height + 0.4,
                              boxstyle="round,pad=0.05",
                              facecolor='none',
                              edgecolor=colors['innovation'], linewidth=2.5, linestyle='--')
ax.add_patch(innovation1)
ax.text(student_x + student_width/2, student_y - 0.5, '① Student Prior', 
        ha='center', va='center', fontsize=10, fontweight='bold', color=colors['innovation'])

# Innovation 2: Memory Bank
innovation2 = FancyBboxPatch((memory_x - 0.2, memory_y - 0.2), 
                              memory_width + 5.4, memory_height + 0.4,
                              boxstyle="round,pad=0.05",
                              facecolor='none',
                              edgecolor=colors['innovation'], linewidth=2.5, linestyle='--')
ax.add_patch(innovation2)
ax.text(memory_x + (memory_width + 5.4)/2, memory_y - 0.5, '② Memory Bank + Counterfactual', 
        ha='center', va='center', fontsize=10, fontweight='bold', color=colors['innovation'])

# Innovation: VLM Encoder (Optional but important)
innovation_vlm = FancyBboxPatch((vlm_detail_x - 0.2, vlm_detail_y - 0.2), 
                                 vlm_detail_width + 0.4, vlm_detail_height + 0.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor='none',
                                 edgecolor=colors['highlight'], linewidth=2.5, linestyle='--')
ax.add_patch(innovation_vlm)
ax.text(vlm_detail_x + vlm_detail_width/2, vlm_detail_y - 0.4, 'VLM Encoder (Optional)', 
        ha='center', va='center', fontsize=9, fontweight='bold', color=colors['highlight'])

# Innovation 4: Sinkhorn OT
innovation3 = FancyBboxPatch((loss_x - 0.2, ot_y - 0.2), 
                              loss_width + 0.4, loss_height + 0.4,
                              boxstyle="round,pad=0.05",
                              facecolor='none',
                              edgecolor=colors['innovation'], linewidth=2.5, linestyle='--')
ax.add_patch(innovation3)
ax.text(loss_x + loss_width/2, ot_y - 0.5, '④ Sinkhorn Optimal Transport', 
        ha='center', va='center', fontsize=10, fontweight='bold', color=colors['innovation'])

# ==================== Total Loss Formula ====================
total_loss_y = 1.5
total_loss_box = FancyBboxPatch((2.0, total_loss_y), 16.0, 1.0,
                                 boxstyle="round,pad=0.15",
                                 facecolor='white',
                                 edgecolor='black', linewidth=2.5, alpha=0.95)
ax.add_patch(total_loss_box)
ax.text(10, total_loss_y + 0.65, 
        r'$\mathcal{L}_{total} = \lambda_{cls} \cdot \mathcal{L}_{cls} + \lambda_{ot} \cdot \mathcal{L}_{ot} + \lambda_{consist} \cdot \mathcal{L}_{consist} + \lambda_{adv} \cdot \mathcal{L}_{adv}$', 
        ha='center', va='center', fontsize=14, fontweight='bold')

param_text = r'$\lambda_{cls}=1.0, \quad \lambda_{ot}=1.0, \quad \lambda_{consist}=0.5, \quad \lambda_{adv}=0.1$'
ax.text(10, total_loss_y + 0.15, param_text, 
        ha='center', va='center', fontsize=11, style='italic')

# ==================== Legend ====================
legend_elements = [
    mpatches.Patch(facecolor=colors['arrow_main'], edgecolor='black', label='Data Flow'),
    mpatches.Patch(facecolor=colors['arrow_loss'], edgecolor='black', label='Loss Flow', linestyle='--'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.95, 
          bbox_to_anchor=(0.02, 0.98))

# ==================== Save ====================
output_dir = '/data2/hmy/VLM_Caus_Rm_Mics/docs/figures'
os.makedirs(output_dir, exist_ok=True)

# Save as PDF (publication quality)
pdf_path = os.path.join(output_dir, 'BioCOT_Architecture_Paper.pdf')
plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white', 
            edgecolor='none', pad_inches=0.1)
print(f"✅ PDF架构图已保存: {pdf_path}")

# Save as PNG (high resolution for preview)
png_path = os.path.join(output_dir, 'BioCOT_Architecture_Paper.png')
plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white', 
            edgecolor='none', pad_inches=0.1)
print(f"✅ PNG架构图已保存: {png_path}")

# Save as SVG (vector format, scalable)
svg_path = os.path.join(output_dir, 'BioCOT_Architecture_Paper.svg')
plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white', 
            edgecolor='none', pad_inches=0.1)
print(f"✅ SVG架构图已保存: {svg_path}")

plt.close()
print("\n🎉 Bio-COT模型架构图生成完成！")
print("📄 所有格式已保存到: docs/figures/")
print("   - PDF: 适合论文提交（矢量格式）")
print("   - PNG: 高分辨率预览（300 DPI）")
print("   - SVG: 可缩放矢量图（可编辑）")

