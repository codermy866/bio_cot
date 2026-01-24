#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT Framework Architecture Diagram
Professional visualization with clear layout and English labels
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

# Set up the figure with high resolution (wider to avoid overlap)
fig = plt.figure(figsize=(24, 16))
ax = fig.add_subplot(111)
ax.set_xlim(0, 24)
ax.set_ylim(0, 16)
ax.axis('off')

# Color scheme
colors = {
    'input': '#E8F4F8',
    'student_prior': '#FFE5B4',
    'encoder': '#D4EDDA',
    'memory': '#F8D7DA',
    'loss': '#E2E3E5',
    'output': '#FFF9C4',
    'arrow': '#2C3E50',
    'highlight': '#E74C3C'
}

# Font settings
title_font = {'fontsize': 18, 'fontweight': 'bold', 'fontfamily': 'sans-serif'}
header_font = {'fontsize': 14, 'fontweight': 'bold', 'fontfamily': 'sans-serif'}
text_font = {'fontsize': 11, 'fontfamily': 'sans-serif'}
formula_font = {'fontsize': 10, 'fontfamily': 'monospace', 'style': 'italic'}

# Title
ax.text(12, 15.2, 'Bio-COT: Bio-Invariant Causal Optimal Transport Framework', 
        ha='center', va='center', **title_font)

# ==================== Input Layer ====================
# Clinical Data Input
clinical_box = FancyBboxPatch((0.5, 12), 3.5, 2, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], 
                              edgecolor='black', linewidth=1.5)
ax.add_patch(clinical_box)
ax.text(2.25, 13.5, 'Clinical Data', ha='center', va='center', **header_font)
ax.text(2.25, 13, 'HPV, TCT, Age', ha='center', va='center', **text_font)
ax.text(2.25, 12.5, '[B, 7]', ha='center', va='center', **formula_font)

# Image Features Input
image_box = FancyBboxPatch((0.5, 9), 3.5, 2,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['input'],
                           edgecolor='black', linewidth=1.5)
ax.add_patch(image_box)
ax.text(2.25, 10.5, 'Image Features', ha='center', va='center', **header_font)
ax.text(2.25, 10, 'OCT + Colposcopy', ha='center', va='center', **text_font)
ax.text(2.25, 9.5, '[B, 512]', ha='center', va='center', **formula_font)

# Center Labels Input
center_box = FancyBboxPatch((0.5, 6), 3.5, 2,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['input'],
                            edgecolor='black', linewidth=1.5)
ax.add_patch(center_box)
ax.text(2.25, 7.5, 'Center Labels', ha='center', va='center', **header_font)
ax.text(2.25, 7, 'Hospital ID', ha='center', va='center', **text_font)
ax.text(2.25, 6.5, '[B]', ha='center', va='center', **formula_font)

# ==================== Student Prior Network ====================
student_box = FancyBboxPatch((5, 12), 5, 2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['student_prior'],
                              edgecolor='black', linewidth=2)
ax.add_patch(student_box)
ax.text(7.5, 13.5, 'Student Prior Network', ha='center', va='center', **header_font)
ax.text(7.5, 13, 'Lightweight MLP', ha='center', va='center', **text_font)
ax.text(7.5, 12.5, '7 → 256 → 512 → 768', ha='center', va='center', **formula_font)

# Output: z_sem
z_sem_box = FancyBboxPatch((11, 12), 3, 2,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['output'],
                            edgecolor='black', linewidth=1.5)
ax.add_patch(z_sem_box)
ax.text(12.5, 13.5, 'z_sem', ha='center', va='center', **header_font)
ax.text(12.5, 13, 'Semantic Anchor', ha='center', va='center', **text_font)
ax.text(12.5, 12.5, '[B, 768]', ha='center', va='center', **formula_font)

# ==================== Dual Head Image Encoder ====================
encoder_box = FancyBboxPatch((5, 9), 5, 2,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['encoder'],
                             edgecolor='black', linewidth=2)
ax.add_patch(encoder_box)
ax.text(7.5, 10.5, 'Dual Head Encoder', ha='center', va='center', **header_font)
ax.text(7.5, 10, 'Feature Projection', ha='center', va='center', **text_font)
ax.text(7.5, 9.5, '512 → 1536 → 768', ha='center', va='center', **formula_font)

# Causal Head
causal_box = FancyBboxPatch((11, 9.5), 3, 1,
                             boxstyle="round,pad=0.05",
                             facecolor=colors['encoder'],
                             edgecolor='black', linewidth=1.5)
ax.add_patch(causal_box)
ax.text(12.5, 10, 'z_causal', ha='center', va='center', **header_font)
ax.text(12.5, 9.7, '[B, 768]', ha='center', va='center', **formula_font)

# Noise Head
noise_box = FancyBboxPatch((11, 9), 3, 1,
                           boxstyle="round,pad=0.05",
                           facecolor=colors['encoder'],
                           edgecolor='black', linewidth=1.5)
ax.add_patch(noise_box)
ax.text(12.5, 9.5, 'z_noise', ha='center', va='center', **header_font)
ax.text(12.5, 9.2, '[B, 768]', ha='center', va='center', **formula_font)

# ==================== Memory Bank ====================
memory_box = FancyBboxPatch((5, 6), 5, 2,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['memory'],
                            edgecolor='black', linewidth=2)
ax.add_patch(memory_box)
ax.text(7.5, 7.5, 'Noise Memory Bank', ha='center', va='center', **header_font)
ax.text(7.5, 7, 'Store z_noise by Center', ha='center', va='center', **text_font)
ax.text(7.5, 6.5, 'Capacity: 100 per center', ha='center', va='center', **formula_font)

# Counterfactual Noise
cf_noise_box = FancyBboxPatch((11, 6), 3, 2,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['memory'],
                               edgecolor='black', linewidth=1.5)
ax.add_patch(cf_noise_box)
ax.text(12.5, 7.5, 'z_noise_cf', ha='center', va='center', **header_font)
ax.text(12.5, 7, 'Counterfactual', ha='center', va='center', **text_font)
ax.text(12.5, 6.5, '[B, 768]', ha='center', va='center', **formula_font)

# ==================== Loss Functions ====================
# Sinkhorn OT Loss
ot_loss_box = FancyBboxPatch((15.5, 12), 6, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['loss'],
                              edgecolor='black', linewidth=1.5)
ax.add_patch(ot_loss_box)
ax.text(18.5, 13, 'Sinkhorn OT Loss', ha='center', va='center', **header_font)
ax.text(18.5, 12.5, 'L_ot = OT(z_causal, z_sem)', ha='center', va='center', **formula_font)

# Counterfactual Consistency Loss
consist_loss_box = FancyBboxPatch((15.5, 10), 6, 1.5,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['loss'],
                                   edgecolor='black', linewidth=1.5)
ax.add_patch(consist_loss_box)
ax.text(18.5, 10.75, 'Consistency Loss', ha='center', va='center', **header_font)
ax.text(18.5, 10.25, 'L_consist = ||logits - logits_cf||', ha='center', va='center', **formula_font)

# Adversarial Loss
adv_loss_box = FancyBboxPatch((15.5, 8), 6, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['loss'],
                               edgecolor='black', linewidth=1.5)
ax.add_patch(adv_loss_box)
ax.text(18.5, 8.75, 'Adversarial Loss', ha='center', va='center', **header_font)
ax.text(18.5, 8.25, 'L_adv = CE(D(z_noise), center_id)', ha='center', va='center', **formula_font)

# Classification Loss
cls_loss_box = FancyBboxPatch((15.5, 6), 6, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['loss'],
                               edgecolor='black', linewidth=1.5)
ax.add_patch(cls_loss_box)
ax.text(18.5, 6.75, 'Classification Loss', ha='center', va='center', **header_font)
ax.text(18.5, 6.25, 'L_cls = CE(logits, labels)', ha='center', va='center', **formula_font)

# ==================== Classifier ====================
classifier_box = FancyBboxPatch((5, 2.5), 5, 2,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['output'],
                                edgecolor='black', linewidth=2)
ax.add_patch(classifier_box)
ax.text(7.5, 4, 'Classifier', ha='center', va='center', **header_font)
ax.text(7.5, 3.5, 'MLP: 768 → 768 → 384 → 2', ha='center', va='center', **text_font)
ax.text(7.5, 3, 'Input: z_causal', ha='center', va='center', **formula_font)

# Output: Logits
logits_box = FancyBboxPatch((11, 2.5), 3, 2,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['output'],
                             edgecolor='black', linewidth=2)
ax.add_patch(logits_box)
ax.text(12.5, 4, 'Logits', ha='center', va='center', **header_font)
ax.text(12.5, 3.5, 'Classification', ha='center', va='center', **text_font)
ax.text(12.5, 3, '[B, 2]', ha='center', va='center', **formula_font)

# ==================== Arrows ====================
# Clinical Data -> Student Prior
arrow1 = FancyArrowPatch((4, 13), (5, 13),
                         arrowstyle='->', lw=2, color=colors['arrow'])
ax.add_patch(arrow1)

# Student Prior -> z_sem
arrow2 = FancyArrowPatch((10, 13), (11, 13),
                         arrowstyle='->', lw=2, color=colors['arrow'])
ax.add_patch(arrow2)

# Image Features -> Encoder
arrow3 = FancyArrowPatch((4, 10), (5, 10),
                         arrowstyle='->', lw=2, color=colors['arrow'])
ax.add_patch(arrow3)

# Encoder -> z_causal and z_noise
arrow4 = FancyArrowPatch((10, 10), (11, 10),
                         arrowstyle='->', lw=2, color=colors['arrow'])
ax.add_patch(arrow4)
arrow5 = FancyArrowPatch((10, 9.5), (11, 9.5),
                         arrowstyle='->', lw=2, color=colors['arrow'])
ax.add_patch(arrow5)

# z_noise -> Memory Bank
arrow6 = FancyArrowPatch((12.5, 9), (12.5, 8),
                         arrowstyle='->', lw=2, color=colors['arrow'])
ax.add_patch(arrow6)
arrow7 = FancyArrowPatch((12.5, 8), (10, 7),
                         arrowstyle='->', lw=2, color=colors['arrow'])
ax.add_patch(arrow7)

# Memory Bank -> Counterfactual Noise
arrow8 = FancyArrowPatch((10, 7), (11, 7),
                         arrowstyle='->', lw=2, color=colors['arrow'])
ax.add_patch(arrow8)

# Center Labels -> Memory Bank
arrow9 = FancyArrowPatch((4, 7), (5, 7),
                         arrowstyle='->', lw=2, color=colors['arrow'])
ax.add_patch(arrow9)

# z_causal -> Classifier
arrow10 = FancyArrowPatch((12.5, 9.5), (12.5, 4.5),
                          arrowstyle='->', lw=2, color=colors['arrow'])
ax.add_patch(arrow10)
arrow11 = FancyArrowPatch((12.5, 4.5), (10, 3.5),
                          arrowstyle='->', lw=2, color=colors['arrow'])
ax.add_patch(arrow11)

# Classifier -> Logits
arrow12 = FancyArrowPatch((10, 3.5), (11, 3.5),
                          arrowstyle='->', lw=2, color=colors['arrow'])
ax.add_patch(arrow12)

# Loss function arrows
# z_causal -> OT Loss
arrow13 = FancyArrowPatch((14, 10), (15.5, 12.75),
                          arrowstyle='->', lw=1.5, color=colors['highlight'], linestyle='--')
ax.add_patch(arrow13)
# z_sem -> OT Loss
arrow14 = FancyArrowPatch((14, 13), (15.5, 12.75),
                          arrowstyle='->', lw=1.5, color=colors['highlight'], linestyle='--')
ax.add_patch(arrow14)

# Counterfactual -> Consistency Loss
arrow15 = FancyArrowPatch((14, 7), (15.5, 10.75),
                          arrowstyle='->', lw=1.5, color=colors['highlight'], linestyle='--')
ax.add_patch(arrow15)

# z_noise -> Adversarial Loss
arrow16 = FancyArrowPatch((14, 9.5), (15.5, 8.75),
                          arrowstyle='->', lw=1.5, color=colors['highlight'], linestyle='--')
ax.add_patch(arrow16)

# Logits -> Classification Loss
arrow17 = FancyArrowPatch((14, 3.5), (15.5, 6.75),
                          arrowstyle='->', lw=1.5, color=colors['highlight'], linestyle='--')
ax.add_patch(arrow17)

# ==================== Innovation Highlights ====================
# Highlight boxes for innovations
innovation1 = FancyBboxPatch((0.2, 0.5), 5.1, 1,
                              boxstyle="round,pad=0.05",
                              facecolor='none',
                              edgecolor=colors['highlight'], linewidth=2, linestyle='--')
ax.add_patch(innovation1)
ax.text(2.75, 1, 'Innovation 1: Student Prior (Fast Training)', 
        ha='center', va='center', fontsize=10, fontweight='bold', color=colors['highlight'])

innovation2 = FancyBboxPatch((5.7, 0.5), 5.1, 1,
                              boxstyle="round,pad=0.05",
                              facecolor='none',
                              edgecolor=colors['highlight'], linewidth=2, linestyle='--')
ax.add_patch(innovation2)
ax.text(8.25, 1, 'Innovation 2: Sinkhorn OT (Flexible Alignment)', 
        ha='center', va='center', fontsize=10, fontweight='bold', color=colors['highlight'])

innovation3 = FancyBboxPatch((11.2, 0.5), 5.1, 1,
                              boxstyle="round,pad=0.05",
                              facecolor='none',
                              edgecolor=colors['highlight'], linewidth=2, linestyle='--')
ax.add_patch(innovation3)
ax.text(13.75, 1, 'Innovation 3: Memory Bank (Counterfactual)', 
        ha='center', va='center', fontsize=10, fontweight='bold', color=colors['highlight'])

# ==================== Total Loss Formula ====================
total_loss_text = 'Total Loss: L = λ_cls·L_cls + λ_ot·L_ot + λ_consist·L_consist + λ_adv·L_adv'
ax.text(12, 0.1, total_loss_text, ha='center', va='center', 
        fontsize=11, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', pad=0.5))

# Save as PDF
plt.tight_layout()
output_path = '/data2/hmy/VLM_Caus_Rm_Mics/cursor_file/visualization/bio_cot_framework_diagram.pdf'
plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
print(f"✅ Diagram saved to: {output_path}")

plt.close()

