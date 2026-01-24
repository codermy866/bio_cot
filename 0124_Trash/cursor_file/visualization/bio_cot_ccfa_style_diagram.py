#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT Framework Architecture Diagram (CCFA Style)
Professional visualization for SCI paper submission
Following CVPR/ICCV/ECCV/NeurIPS style guidelines
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch, Rectangle
from matplotlib.patches import Circle, Ellipse
import numpy as np

# Set up the figure with high resolution (CCFA standard: 2-column width)
fig = plt.figure(figsize=(16, 10))  # Standard paper width
ax = fig.add_subplot(111)
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# CCFA Color Scheme (professional, publication-ready)
colors = {
    'input_oct': '#4A90E2',      # Blue for OCT
    'input_colpo': '#50C878',     # Green for Colposcopy
    'input_clinical': '#FF6B6B',  # Red for Clinical
    'student_prior': '#FFD93D',   # Yellow for Student Prior
    'fusion': '#9B59B6',          # Purple for Fusion
    'encoder': '#3498DB',         # Light Blue for Encoder
    'causal': '#2ECC71',          # Green for Causal
    'noise': '#E74C3C',           # Red for Noise
    'memory': '#F39C12',          # Orange for Memory
    'classifier': '#1ABC9C',     # Teal for Classifier
    'loss': '#95A5A6',            # Gray for Loss
    'output': '#34495E',          # Dark Gray for Output
    'arrow_main': '#2C3E50',      # Dark Blue for main flow
    'arrow_loss': '#E67E22',      # Orange for loss flow
    'highlight': '#E74C3C'        # Red for highlights
}

# Font settings (CCFA style: clear, readable)
title_font = {'fontsize': 16, 'fontweight': 'bold', 'fontfamily': 'sans-serif'}
section_font = {'fontsize': 12, 'fontweight': 'bold', 'fontfamily': 'sans-serif'}
label_font = {'fontsize': 10, 'fontfamily': 'sans-serif'}
formula_font = {'fontsize': 9, 'fontfamily': 'monospace', 'style': 'italic'}

# ==================== Title ====================
ax.text(8, 9.5, 'Bio-COT: Bio-Invariant Causal Optimal Transport Framework', 
        ha='center', va='center', **title_font)
ax.text(8, 9.1, 'for Multi-Center Cervical Lesion Screening', 
        ha='center', va='center', fontsize=12, style='italic')

# ==================== Input Layer (Left) ====================
# OCT Features
oct_box = FancyBboxPatch((0.3, 7.5), 2.2, 1.2, 
                         boxstyle="round,pad=0.08", 
                         facecolor=colors['input_oct'], 
                         edgecolor='black', linewidth=1.5,
                         alpha=0.8)
ax.add_patch(oct_box)
ax.text(1.4, 8.3, 'OCT Features', ha='center', va='center', **section_font, color='white')
ax.text(1.4, 8.0, '48 frames', ha='center', va='center', **label_font, color='white')
ax.text(1.4, 7.7, '[B, 512]', ha='center', va='center', **formula_font, color='white')

# Colposcopy Features
colpo_box = FancyBboxPatch((0.3, 6.0), 2.2, 1.2,
                           boxstyle="round,pad=0.08",
                           facecolor=colors['input_colpo'],
                           edgecolor='black', linewidth=1.5,
                           alpha=0.8)
ax.add_patch(colpo_box)
ax.text(1.4, 6.8, 'Colposcopy', ha='center', va='center', **section_font, color='white')
ax.text(1.4, 6.5, 'Features', ha='center', va='center', **section_font, color='white')
ax.text(1.4, 6.2, '[B, 512]', ha='center', va='center', **formula_font, color='white')

# Clinical Data
clinical_box = FancyBboxPatch((0.3, 4.5), 2.2, 1.2,
                             boxstyle="round,pad=0.08",
                             facecolor=colors['input_clinical'],
                             edgecolor='black', linewidth=1.5,
                             alpha=0.8)
ax.add_patch(clinical_box)
ax.text(1.4, 5.3, 'Clinical Data', ha='center', va='center', **section_font, color='white')
ax.text(1.4, 5.0, 'HPV, TCT, Age', ha='center', va='center', **label_font, color='white')
ax.text(1.4, 4.7, '[B, 7]', ha='center', va='center', **formula_font, color='white')

# Center Labels
center_box = FancyBboxPatch((0.3, 3.0), 2.2, 1.0,
                            boxstyle="round,pad=0.08",
                            facecolor=colors['input_clinical'],
                            edgecolor='black', linewidth=1.5,
                            alpha=0.6)
ax.add_patch(center_box)
ax.text(1.4, 3.6, 'Center ID', ha='center', va='center', **label_font, color='white')
ax.text(1.4, 3.3, '[B]', ha='center', va='center', **formula_font, color='white')

# ==================== Multimodal Fusion Module ====================
fusion_box = FancyBboxPatch((3.0, 6.5), 2.5, 2.0,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['fusion'],
                            edgecolor='black', linewidth=2,
                            alpha=0.9)
ax.add_patch(fusion_box)
ax.text(4.25, 8.0, 'Multimodal', ha='center', va='center', **section_font, color='white')
ax.text(4.25, 7.7, 'Fusion', ha='center', va='center', **section_font, color='white')
ax.text(4.25, 7.3, 'Learnable Weights', ha='center', va='center', color='white', fontsize=9)
ax.text(4.25, 7.0, '+ Attention', ha='center', va='center', color='white', fontsize=9)
ax.text(4.25, 6.7, '[B, 512] → [B, 768]', ha='center', va='center', color='white', fontsize=8, family='monospace', style='italic')

# ==================== Student Prior Network ====================
student_box = FancyBboxPatch((3.0, 4.0), 2.5, 1.8,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['student_prior'],
                             edgecolor='black', linewidth=2,
                             alpha=0.9)
ax.add_patch(student_box)
ax.text(4.25, 5.3, 'Student Prior', ha='center', va='center', **section_font)
ax.text(4.25, 5.0, 'Network', ha='center', va='center', **section_font)
ax.text(4.25, 4.7, 'MLP: 7→256→512→768', ha='center', va='center', fontsize=8, family='monospace', style='italic')
ax.text(4.25, 4.4, 'Fast Training', ha='center', va='center', fontsize=8, style='italic')

# Output: z_sem
z_sem_box = FancyBboxPatch((6.0, 4.2), 1.8, 1.4,
                           boxstyle="round,pad=0.08",
                           facecolor=colors['output'],
                           edgecolor='black', linewidth=1.5,
                           alpha=0.7)
ax.add_patch(z_sem_box)
ax.text(6.9, 5.0, 'z_sem', ha='center', va='center', **section_font, color='white')
ax.text(6.9, 4.7, 'Semantic', ha='center', va='center', color='white', fontsize=9)
ax.text(6.9, 4.4, 'Anchor', ha='center', va='center', color='white', fontsize=9)
ax.text(6.9, 4.1, '[B, 768]', ha='center', va='center', color='white', fontsize=8, family='monospace', style='italic')

# ==================== Dual Head Image Encoder ====================
encoder_box = FancyBboxPatch((6.0, 6.8), 3.5, 1.6,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['encoder'],
                             edgecolor='black', linewidth=2,
                             alpha=0.9)
ax.add_patch(encoder_box)
ax.text(7.75, 7.8, 'Dual Head Image Encoder', ha='center', va='center', **section_font, color='white')
ax.text(7.75, 7.5, 'Feature Projection + Dual Heads', ha='center', va='center', color='white', fontsize=9)
ax.text(7.75, 7.2, '512 → 1536 → 768', ha='center', va='center', color='white', fontsize=8, family='monospace', style='italic')

# Causal Head Output
causal_box = FancyBboxPatch((10.0, 7.2), 1.5, 0.7,
                            boxstyle="round,pad=0.05",
                            facecolor=colors['causal'],
                            edgecolor='black', linewidth=1.5,
                            alpha=0.8)
ax.add_patch(causal_box)
ax.text(10.75, 7.6, 'z_causal', ha='center', va='center', color='white', fontsize=10, weight='bold')
ax.text(10.75, 7.3, '[B, 768]', ha='center', va='center', color='white', fontsize=8, family='monospace', style='italic')

# Noise Head Output
noise_box = FancyBboxPatch((10.0, 6.5), 1.5, 0.7,
                           boxstyle="round,pad=0.05",
                           facecolor=colors['noise'],
                           edgecolor='black', linewidth=1.5,
                           alpha=0.8)
ax.add_patch(noise_box)
ax.text(10.75, 6.9, 'z_noise', ha='center', va='center', color='white', fontsize=10, weight='bold')
ax.text(10.75, 6.6, '[B, 768]', ha='center', va='center', color='white', fontsize=8, family='monospace', style='italic')

# ==================== Memory Bank ====================
memory_box = FancyBboxPatch((6.0, 3.0), 3.5, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['memory'],
                            edgecolor='black', linewidth=2,
                            alpha=0.9)
ax.add_patch(memory_box)
ax.text(7.75, 4.0, 'Noise Memory Bank', ha='center', va='center', **section_font, color='white')
ax.text(7.75, 3.7, 'Store z_noise by Center', ha='center', va='center', color='white', fontsize=9)
ax.text(7.75, 3.4, 'Capacity: 100/center', ha='center', va='center', color='white', fontsize=8, family='monospace', style='italic')

# Counterfactual Noise Output
cf_noise_box = FancyBboxPatch((10.0, 3.2), 1.5, 1.0,
                              boxstyle="round,pad=0.08",
                              facecolor=colors['memory'],
                              edgecolor='black', linewidth=1.5,
                              alpha=0.7)
ax.add_patch(cf_noise_box)
ax.text(10.75, 3.9, 'z_noise_cf', ha='center', va='center', color='white', fontsize=10, weight='bold')
ax.text(10.75, 3.6, 'Counter-', ha='center', va='center', color='white', fontsize=8)
ax.text(10.75, 3.3, 'factual', ha='center', va='center', color='white', fontsize=8)

# ==================== Three-Modal Fusion ====================
multimodal_fusion_box = FancyBboxPatch((12.5, 5.5), 2.5, 2.0,
                                       boxstyle="round,pad=0.1",
                                       facecolor=colors['fusion'],
                                       edgecolor='black', linewidth=2,
                                       alpha=0.9)
ax.add_patch(multimodal_fusion_box)
ax.text(13.75, 7.0, 'Three-Modal', ha='center', va='center', **section_font, color='white')
ax.text(13.75, 6.7, 'Fusion', ha='center', va='center', **section_font, color='white')
ax.text(13.75, 6.3, 'z_causal ⊕ z_sem', ha='center', va='center', color='white', fontsize=9, family='monospace', style='italic')
ax.text(13.75, 6.0, 'Concat + Project', ha='center', va='center', color='white', fontsize=9)
ax.text(13.75, 5.7, '[B, 1536] → [B, 768]', ha='center', va='center', color='white', fontsize=8, family='monospace', style='italic')

# ==================== Classifier ====================
classifier_box = FancyBboxPatch((12.5, 2.5), 2.5, 2.0,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['classifier'],
                                edgecolor='black', linewidth=2,
                                alpha=0.9)
ax.add_patch(classifier_box)
ax.text(13.75, 4.0, 'Classifier', ha='center', va='center', **section_font, color='white')
ax.text(13.75, 3.6, 'MLP with Dropout', ha='center', va='center', color='white', fontsize=9)
ax.text(13.75, 3.3, '768 → 768 → 384 → 2', ha='center', va='center', color='white', fontsize=8, family='monospace', style='italic')
ax.text(13.75, 3.0, 'Dropout: 0.4, 0.3', ha='center', va='center', color='white', fontsize=8, family='monospace', style='italic')

# Output: Logits
logits_box = FancyBboxPatch((15.5, 3.5), 1.2, 1.0,
                            boxstyle="round,pad=0.08",
                            facecolor=colors['output'],
                            edgecolor='black', linewidth=2,
                            alpha=0.9)
ax.add_patch(logits_box)
ax.text(16.1, 4.2, 'Logits', ha='center', va='center', color='white', fontsize=11, weight='bold')
ax.text(16.1, 3.9, '[B, 2]', ha='center', va='center', color='white', fontsize=9, family='monospace', style='italic')
ax.text(16.1, 3.6, 'Prediction', ha='center', va='center', color='white', fontsize=8)

# ==================== Loss Functions (Right Side) ====================
# Sinkhorn OT Loss
ot_loss_box = FancyBboxPatch((12.5, 8.5), 3.0, 0.8,
                             boxstyle="round,pad=0.08",
                             facecolor=colors['loss'],
                             edgecolor='black', linewidth=1.5,
                             alpha=0.8)
ax.add_patch(ot_loss_box)
ax.text(14.0, 9.0, 'L_OT = Sinkhorn(z_causal, z_sem)', ha='center', va='center', fontsize=9, family='monospace', style='italic')
ax.text(14.0, 8.7, 'λ_OT = 0.5', ha='center', va='center', fontsize=8, style='italic')

# Counterfactual Consistency Loss
consist_loss_box = FancyBboxPatch((12.5, 7.5), 3.0, 0.8,
                                  boxstyle="round,pad=0.08",
                                  facecolor=colors['loss'],
                                  edgecolor='black', linewidth=1.5,
                                  alpha=0.8)
ax.add_patch(consist_loss_box)
ax.text(14.0, 8.0, 'L_consist = KL(logits, logits_cf)', ha='center', va='center', fontsize=9, family='monospace', style='italic')
ax.text(14.0, 7.7, 'λ_consist = 0.8', ha='center', va='center', fontsize=8, style='italic')

# Adversarial Loss
adv_loss_box = FancyBboxPatch((12.5, 6.5), 3.0, 0.8,
                              boxstyle="round,pad=0.08",
                              facecolor=colors['loss'],
                              edgecolor='black', linewidth=1.5,
                              alpha=0.8)
ax.add_patch(adv_loss_box)
ax.text(14.0, 7.0, 'L_adv = CE(D(z_noise), center_id)', ha='center', va='center', fontsize=9, family='monospace', style='italic')
ax.text(14.0, 6.7, 'λ_adv = 0.3', ha='center', va='center', fontsize=8, style='italic')

# Classification Loss
cls_loss_box = FancyBboxPatch((12.5, 5.5), 3.0, 0.8,
                              boxstyle="round,pad=0.08",
                              facecolor=colors['loss'],
                              edgecolor='black', linewidth=1.5,
                              alpha=0.8)
ax.add_patch(cls_loss_box)
ax.text(14.0, 6.0, 'L_cls = CE(logits, labels)', ha='center', va='center', fontsize=9, family='monospace', style='italic')
ax.text(14.0, 5.7, 'λ_cls = 1.0', ha='center', va='center', fontsize=8, style='italic')

# ==================== Arrows (Data Flow) ====================
# Input to Fusion
arrow1 = FancyArrowPatch((2.5, 8.1), (3.0, 7.5),
                         arrowstyle='->', lw=2.5, color=colors['arrow_main'],
                         connectionstyle="arc3,rad=0.1")
ax.add_patch(arrow1)
arrow2 = FancyArrowPatch((2.5, 6.6), (3.0, 7.0),
                         arrowstyle='->', lw=2.5, color=colors['arrow_main'],
                         connectionstyle="arc3,rad=-0.1")
ax.add_patch(arrow2)

# Fusion to Encoder
arrow3 = FancyArrowPatch((5.5, 7.5), (6.0, 7.6),
                         arrowstyle='->', lw=2.5, color=colors['arrow_main'])
ax.add_patch(arrow3)

# Encoder to Dual Heads
arrow4 = FancyArrowPatch((9.5, 7.55), (10.0, 7.55),
                         arrowstyle='->', lw=2.5, color=colors['arrow_main'])
ax.add_patch(arrow4)

# Clinical to Student Prior
arrow5 = FancyArrowPatch((2.5, 5.1), (3.0, 4.9),
                         arrowstyle='->', lw=2.5, color=colors['arrow_main'])
ax.add_patch(arrow5)

# Student Prior to z_sem
arrow6 = FancyArrowPatch((5.5, 4.9), (6.0, 4.9),
                         arrowstyle='->', lw=2.5, color=colors['arrow_main'])
ax.add_patch(arrow6)

# z_noise to Memory Bank
arrow7 = FancyArrowPatch((10.75, 6.5), (10.75, 4.5),
                         arrowstyle='->', lw=2, color=colors['arrow_main'],
                         connectionstyle="arc3,rad=0.2")
ax.add_patch(arrow7)
arrow8 = FancyArrowPatch((10.75, 4.5), (9.5, 3.75),
                         arrowstyle='->', lw=2, color=colors['arrow_main'])
ax.add_patch(arrow8)

# Center Labels to Memory Bank
arrow9 = FancyArrowPatch((2.5, 3.5), (6.0, 3.75),
                         arrowstyle='->', lw=2, color=colors['arrow_main'],
                         linestyle='--', alpha=0.6)
ax.add_patch(arrow9)

# Memory Bank to Counterfactual
arrow10 = FancyArrowPatch((9.5, 3.75), (10.0, 3.7),
                          arrowstyle='->', lw=2, color=colors['arrow_main'])
ax.add_patch(arrow10)

# z_causal and z_sem to Three-Modal Fusion
arrow11 = FancyArrowPatch((10.75, 7.55), (12.0, 6.5),
                          arrowstyle='->', lw=2.5, color=colors['arrow_main'],
                          connectionstyle="arc3,rad=0.3")
ax.add_patch(arrow11)
arrow12 = FancyArrowPatch((6.9, 4.9), (12.0, 6.0),
                          arrowstyle='->', lw=2.5, color=colors['arrow_main'],
                          connectionstyle="arc3,rad=0.2")
ax.add_patch(arrow12)

# Three-Modal Fusion to Classifier
arrow13 = FancyArrowPatch((13.75, 5.5), (13.75, 4.5),
                          arrowstyle='->', lw=2.5, color=colors['arrow_main'])
ax.add_patch(arrow13)

# Classifier to Logits
arrow14 = FancyArrowPatch((15.0, 3.5), (15.5, 4.0),
                          arrowstyle='->', lw=2.5, color=colors['arrow_main'])
ax.add_patch(arrow14)

# ==================== Loss Function Arrows (Dashed) ====================
# z_causal -> OT Loss
arrow_ot1 = FancyArrowPatch((10.75, 7.55), (13.0, 8.9),
                            arrowstyle='->', lw=1.5, color=colors['arrow_loss'],
                            linestyle='--', alpha=0.7)
ax.add_patch(arrow_ot1)
# z_sem -> OT Loss
arrow_ot2 = FancyArrowPatch((6.9, 4.9), (13.0, 8.9),
                            arrowstyle='->', lw=1.5, color=colors['arrow_loss'],
                            linestyle='--', alpha=0.7)
ax.add_patch(arrow_ot2)

# Counterfactual -> Consistency Loss
arrow_consist = FancyArrowPatch((10.75, 3.7), (13.0, 7.9),
                                arrowstyle='->', lw=1.5, color=colors['arrow_loss'],
                                linestyle='--', alpha=0.7)
ax.add_patch(arrow_consist)

# z_noise -> Adversarial Loss
arrow_adv = FancyArrowPatch((10.75, 6.85), (13.0, 6.9),
                            arrowstyle='->', lw=1.5, color=colors['arrow_loss'],
                            linestyle='--', alpha=0.7)
ax.add_patch(arrow_adv)

# Logits -> Classification Loss
arrow_cls = FancyArrowPatch((16.1, 3.5), (15.0, 5.9),
                            arrowstyle='->', lw=1.5, color=colors['arrow_loss'],
                            linestyle='--', alpha=0.7)
ax.add_patch(arrow_cls)

# ==================== Innovation Highlights ====================
# Innovation boxes with dashed borders
innovation1 = FancyBboxPatch((2.8, 3.8), 2.7, 2.0,
                             boxstyle="round,pad=0.05",
                             facecolor='none',
                             edgecolor=colors['highlight'], linewidth=2, linestyle='--')
ax.add_patch(innovation1)
ax.text(4.15, 5.5, '① Student Prior', 
        ha='center', va='center', fontsize=9, fontweight='bold', color=colors['highlight'])
ax.text(4.15, 5.2, '(Fast Training)', 
        ha='center', va='center', fontsize=8, style='italic', color=colors['highlight'])

innovation2 = FancyBboxPatch((2.8, 6.3), 2.7, 2.2,
                             boxstyle="round,pad=0.05",
                             facecolor='none',
                             edgecolor=colors['highlight'], linewidth=2, linestyle='--')
ax.add_patch(innovation2)
ax.text(4.15, 8.0, '② Multimodal Fusion', 
        ha='center', va='center', fontsize=9, fontweight='bold', color=colors['highlight'])
ax.text(4.15, 7.7, '(Learnable Weights)', 
        ha='center', va='center', fontsize=8, style='italic', color=colors['highlight'])

innovation3 = FancyBboxPatch((5.8, 2.8), 5.7, 1.7,
                             boxstyle="round,pad=0.05",
                             facecolor='none',
                             edgecolor=colors['highlight'], linewidth=2, linestyle='--')
ax.add_patch(innovation3)
ax.text(7.65, 4.0, '③ Memory Bank + Counterfactual', 
        ha='center', va='center', fontsize=9, fontweight='bold', color=colors['highlight'])
ax.text(7.65, 3.7, '(Causal Disentanglement)', 
        ha='center', va='center', fontsize=8, style='italic', color=colors['highlight'])

innovation4 = FancyBboxPatch((12.3, 8.3), 3.4, 0.6,
                             boxstyle="round,pad=0.05",
                             facecolor='none',
                             edgecolor=colors['highlight'], linewidth=2, linestyle='--')
ax.add_patch(innovation4)
ax.text(14.0, 8.7, '④ Sinkhorn OT (Flexible Alignment)', 
        ha='center', va='center', fontsize=9, fontweight='bold', color=colors['highlight'])

# ==================== Total Loss Formula ====================
total_loss_text = r'$\mathcal{L}_{total} = \lambda_{cls} \cdot \mathcal{L}_{cls} + \lambda_{ot} \cdot \mathcal{L}_{ot} + \lambda_{consist} \cdot \mathcal{L}_{consist} + \lambda_{adv} \cdot \mathcal{L}_{adv}$'
ax.text(8, 1.5, total_loss_text, ha='center', va='center', 
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5))

# Parameter values
param_text = r'$\lambda_{cls}=1.0, \lambda_{ot}=0.5, \lambda_{consist}=0.8, \lambda_{adv}=0.3$'
ax.text(8, 0.8, param_text, ha='center', va='center', 
        fontsize=10, style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F9FA', edgecolor='gray', linewidth=1, alpha=0.7))

# ==================== Legend ====================
legend_elements = [
    mpatches.Patch(facecolor=colors['arrow_main'], edgecolor='black', label='Data Flow'),
    mpatches.Patch(facecolor=colors['arrow_loss'], edgecolor='black', label='Loss Flow', linestyle='--'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.9)

# Save as high-resolution PDF (CCFA standard)
plt.tight_layout()
output_path = '/data2/hmy/VLM_Caus_Rm_Mics/cursor_file/visualization/BioCOT_Framework_CCFA_Style.pdf'
plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ CCFA风格框架图已保存: {output_path}")

# Also save as PNG for preview
png_path = '/data2/hmy/VLM_Caus_Rm_Mics/cursor_file/visualization/BioCOT_Framework_CCFA_Style.png'
plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ PNG预览图已保存: {png_path}")

plt.close()

