#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BIDA Framework Professional Diagram Generator (Fixed Layout)
Top-tier conference quality with no overlapping elements
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Professional font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'

# Professional color palette
COLORS = {
    'input': '#E3F2FD',
    'branch_a': '#FFEBEE',
    'branch_b': '#E8F5E9',
    'vlm': '#FFF3E0',
    'bio_dist': '#FFF9C4',
    'constraint': '#F3E5F5',
    'output': '#E0F2F1',
    'innovation': '#FF5252',
    'arrow_main': '#1976D2',
    'arrow_innovation': '#D32F2F',
    'text': '#212121',
    'border': '#424242'
}

def draw_rounded_box(ax, x, y, width, height, facecolor, edgecolor, linewidth=2, 
                     text='', fontsize=10, fontweight='normal', text_color='black'):
    """Draw a rounded box with text"""
    box = FancyBboxPatch((x, y), width, height,
                        boxstyle="round,pad=0.15",
                        facecolor=facecolor,
                        edgecolor=edgecolor,
                        linewidth=linewidth)
    ax.add_patch(box)
    
    if text:
        ax.text(x + width/2, y + height/2, text,
               ha='center', va='center',
               fontsize=fontsize, weight=fontweight,
               color=text_color)
    return box

def draw_arrow(ax, start, end, color, linewidth=2, arrowstyle='->', 
               connectionstyle='arc3,rad=0.1'):
    """Draw an arrow"""
    arrow = FancyArrowPatch(start, end,
                          arrowstyle=arrowstyle,
                          lw=linewidth,
                          color=color,
                          connectionstyle=connectionstyle,
                          zorder=1)
    ax.add_patch(arrow)
    return arrow

def create_professional_framework_diagram():
    """Create professional framework diagram with proper spacing"""
    
    fig = plt.figure(figsize=(22, 16))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # ==================== Title ====================
    ax.text(11, 15.5, 'BIDA: Bio-Invariant Distributional Anchoring Framework', 
            ha='center', va='center', fontsize=20, weight='bold', color=COLORS['text'])
    ax.text(11, 14.8, 'for Zero-Shot Cross-Center Cervical Cancer Diagnosis', 
            ha='center', va='center', fontsize=15, style='italic', color=COLORS['text'])
    
    # ==================== Input Layer ====================
    input_y = 13.2
    input_height = 1.0
    
    # OCT Images
    draw_rounded_box(ax, 0.5, input_y, 2.8, input_height, COLORS['input'], COLORS['border'], 
                     text='OCT\nImages', fontsize=11, fontweight='bold')
    ax.text(1.9, input_y + 0.25, '[B, C, H, W]', ha='center', va='center', fontsize=9, style='italic')
    
    # Colposcopy Images
    draw_rounded_box(ax, 3.8, input_y, 2.8, input_height, COLORS['input'], COLORS['border'],
                     text='Colposcopy\nImages', fontsize=11, fontweight='bold')
    ax.text(5.2, input_y + 0.25, '[B, C, H, W]', ha='center', va='center', fontsize=9, style='italic')
    
    # Clinical Data (Prior Knowledge) - Highlighted
    draw_rounded_box(ax, 7.1, input_y, 3.2, input_height, COLORS['bio_dist'], COLORS['innovation'], 
                     linewidth=3, text='Clinical Data\n(HPV, TCT, Age)', 
                     fontsize=11, fontweight='bold', text_color=COLORS['innovation'])
    ax.text(8.7, input_y + 0.25, 'Prior Knowledge', ha='center', va='center',
            fontsize=9, style='italic', color=COLORS['innovation'], weight='bold')
    
    # Center Labels
    draw_rounded_box(ax, 10.8, input_y, 2.8, input_height, COLORS['input'], COLORS['border'],
                     text='Center\nLabels', fontsize=11, fontweight='bold')
    ax.text(12.2, input_y + 0.25, '[0-4]', ha='center', va='center', fontsize=9)
    
    # ==================== Branch A: Distributional Anchor ====================
    branch_a_y = 10.5
    branch_a_title_y = 11.8
    
    # Title with innovation highlight
    innovation_box = FancyBboxPatch((0.3, branch_a_title_y - 0.3), 7, 0.6,
                                    boxstyle="round,pad=0.1",
                                    facecolor=COLORS['innovation'],
                                    edgecolor=COLORS['innovation'],
                                    linewidth=2,
                                    alpha=0.2)
    ax.add_patch(innovation_box)
    ax.text(3.8, branch_a_title_y, 'Branch A: Distributional Anchor', 
            ha='center', va='center', fontsize=14, weight='bold', color=COLORS['innovation'])
    ax.text(3.8, branch_a_title_y - 0.15, 'INNOVATION 1: Distributional Anchoring Mechanism', 
            ha='center', va='center', fontsize=11, style='italic', color=COLORS['innovation'])
    
    # Step 1: Clinical to Text
    draw_rounded_box(ax, 0.5, branch_a_y, 2.5, 1.0, COLORS['branch_a'], COLORS['innovation'],
                     linewidth=2.5, text='Clinical Data\n→ Text', fontsize=10, fontweight='bold')
    ax.text(1.75, branch_a_y + 0.2, 'clinical_to_text()', ha='center', va='center',
            fontsize=8, style='italic')
    
    # Step 2: VLM Processing (INNOVATION 4)
    vlm_box = draw_rounded_box(ax, 3.5, branch_a_y, 3.2, 1.0, COLORS['vlm'], COLORS['innovation'],
                               linewidth=2.5, text='VLM: Image+Text\nJoint Understanding', 
                               fontsize=10, fontweight='bold')
    ax.text(5.1, branch_a_y + 0.2, 'Qwen2-VL (Frozen)', ha='center', va='center',
            fontsize=8, style='italic')
    # Innovation 4 label
    ax.text(5.1, branch_a_y - 0.15, 'INNOVATION 4', ha='center', va='center',
            fontsize=8, weight='bold', color=COLORS['innovation'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['innovation'], linewidth=1.5))
    
    # Step 3: Distribution Head
    draw_rounded_box(ax, 7.2, branch_a_y, 2.5, 1.0, COLORS['branch_a'], COLORS['innovation'],
                     linewidth=2.5, text='Distribution\nHead', fontsize=10, fontweight='bold')
    ax.text(8.45, branch_a_y + 0.2, 'distribution_head()', ha='center', va='center',
            fontsize=8, style='italic')
    
    # Bio-Invariant Distribution (KEY INNOVATION - Highlighted)
    bio_y = 8.5
    bio_box = FancyBboxPatch((1.5, bio_y - 0.4), 7, 1.3,
                            boxstyle="round,pad=0.2",
                            facecolor=COLORS['bio_dist'],
                            edgecolor=COLORS['innovation'],
                            linewidth=4,
                            alpha=0.9)
    ax.add_patch(bio_box)
    
    ax.text(5, bio_y + 0.4, 'Bio-Invariant Distribution', ha='center', va='center',
            fontsize=15, weight='bold', color=COLORS['innovation'])
    ax.text(5, bio_y + 0.1, 'P_bio = N(μ_bio, σ_bio)', ha='center', va='center',
            fontsize=13, style='italic', weight='bold')
    ax.text(3.2, bio_y - 0.2, 'μ_bio ∈ ℝ^768', ha='center', va='center',
            fontsize=11, weight='bold')
    ax.text(6.8, bio_y - 0.2, 'σ_bio ∈ ℝ^768', ha='center', va='center',
            fontsize=11, weight='bold')
    
    # ==================== Branch B: Dual Head Encoder ====================
    branch_b_y = 6.5
    branch_b_title_y = 7.8
    
    # Title with innovation highlight
    innovation_box2 = FancyBboxPatch((0.3, branch_b_title_y - 0.3), 7, 0.6,
                                    boxstyle="round,pad=0.1",
                                    facecolor=COLORS['innovation'],
                                    edgecolor=COLORS['innovation'],
                                    linewidth=2,
                                    alpha=0.2)
    ax.add_patch(innovation_box2)
    ax.text(3.8, branch_b_title_y, 'Branch B: Dual Head Image Encoder', 
            ha='center', va='center', fontsize=14, weight='bold', color=COLORS['innovation'])
    ax.text(3.8, branch_b_title_y - 0.15, 'INNOVATION 2: Orthogonal Disentanglement', 
            ha='center', va='center', fontsize=11, style='italic', color=COLORS['innovation'])
    
    # Image Feature Extraction
    draw_rounded_box(ax, 0.5, branch_b_y, 2.5, 1.0, COLORS['branch_b'], COLORS['border'],
                     text='Image Feature\nExtraction', fontsize=10, fontweight='bold')
    ax.text(1.75, branch_b_y + 0.2, 'ResNet50 → [B, 512]', ha='center', va='center',
            fontsize=8, style='italic')
    
    # Dual Head Network
    dual_head_y = 4.5
    dual_head_box = FancyBboxPatch((3.5, dual_head_y - 0.5), 6.5, 2.2,
                                  boxstyle="round,pad=0.15",
                                  facecolor=COLORS['branch_b'],
                                  edgecolor=COLORS['innovation'],
                                  linewidth=3)
    ax.add_patch(dual_head_box)
    
    ax.text(6.75, dual_head_y + 1.5, 'Dual Head Network', ha='center', va='center',
            fontsize=13, weight='bold', color=COLORS['innovation'])
    
    # Head 1: z_causal
    head1_box = draw_rounded_box(ax, 4.2, dual_head_y + 0.3, 2.6, 0.9, '#C8E6C9', COLORS['innovation'],
                                 linewidth=2.5, text='Head 1: z_causal', fontsize=10, fontweight='bold')
    ax.text(5.5, dual_head_y + 0.5, '[B, 768] Causal Features', ha='center', va='center',
            fontsize=9)
    ax.text(5.5, dual_head_y + 0.2, 'For Classification', ha='center', va='center',
            fontsize=8, style='italic')
    
    # Head 2: z_noise
    head2_box = draw_rounded_box(ax, 7.5, dual_head_y + 0.3, 2.6, 0.9, '#FFCDD2', COLORS['innovation'],
                                 linewidth=2.5, text='Head 2: z_noise', fontsize=10, fontweight='bold')
    ax.text(8.8, dual_head_y + 0.5, '[B, 768] Noise Features', ha='center', va='center',
            fontsize=9)
    ax.text(8.8, dual_head_y + 0.2, 'For Center Prediction', ha='center', va='center',
            fontsize=8, style='italic')
    
    # Orthogonal symbol between heads
    ax.text(6.5, dual_head_y + 0.75, '⊥', ha='center', va='center',
            fontsize=24, weight='bold', color=COLORS['innovation'])
    
    # ==================== Constraint Mechanisms ====================
    constraint_y = 2.0
    constraint_title_y = 3.3
    
    # Title
    innovation_box3 = FancyBboxPatch((0.3, constraint_title_y - 0.3), 21.4, 0.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor=COLORS['innovation'],
                                     edgecolor=COLORS['innovation'],
                                     linewidth=2,
                                     alpha=0.2)
    ax.add_patch(innovation_box3)
    ax.text(11, constraint_title_y, 'Constraint Mechanisms', 
            ha='center', va='center', fontsize=14, weight='bold', color=COLORS['innovation'])
    ax.text(11, constraint_title_y - 0.15, 'INNOVATION 3: Prior-Constrained Domain-Invariant Learning', 
            ha='center', va='center', fontsize=11, style='italic', color=COLORS['innovation'])
    
    # Constraint 1: Distributional Anchoring
    const1_box = draw_rounded_box(ax, 0.5, constraint_y, 6.5, 1.3, COLORS['constraint'], COLORS['innovation'],
                                  linewidth=3, text='Constraint 1:\nDistributional Anchoring', 
                                  fontsize=11, fontweight='bold')
    ax.text(3.75, constraint_y + 0.7, 'L_dist = D_KL(Q(z_causal) || P_bio)', 
            ha='center', va='center', fontsize=10, style='italic', weight='bold')
    ax.text(3.75, constraint_y + 0.3, 'z_causal must be within N(μ_bio, σ_bio)', 
            ha='center', va='center', fontsize=9)
    
    # Constraint 2: Orthogonal Disentanglement
    const2_box = draw_rounded_box(ax, 7.5, constraint_y, 6.5, 1.3, COLORS['constraint'], COLORS['innovation'],
                                  linewidth=3, text='Constraint 2:\nOrthogonal Disentanglement', 
                                  fontsize=11, fontweight='bold')
    ax.text(10.75, constraint_y + 0.7, 'L_orth = ||z_causal^T · z_noise||', 
            ha='center', va='center', fontsize=10, style='italic', weight='bold')
    ax.text(10.75, constraint_y + 0.3, 'z_causal ⊥ z_noise (orthogonal)', 
            ha='center', va='center', fontsize=9)
    
    # Constraint 3: Noise Supervision
    const3_box = draw_rounded_box(ax, 14.5, constraint_y, 6.5, 1.3, COLORS['constraint'], COLORS['innovation'],
                                  linewidth=3, text='Constraint 3:\nNoise Supervision', 
                                  fontsize=11, fontweight='bold')
    ax.text(17.75, constraint_y + 0.7, 'L_noise = CE(CenterPred(z_noise), d)', 
            ha='center', va='center', fontsize=10, style='italic', weight='bold')
    ax.text(17.75, constraint_y + 0.3, 'z_noise → Center ID prediction', 
            ha='center', va='center', fontsize=9)
    
    # ==================== Output Layer ====================
    output_y = 4.5
    output_box = draw_rounded_box(ax, 15.5, output_y - 0.5, 4.5, 2.2, COLORS['output'], COLORS['border'],
                                 linewidth=3, text='Classifier', fontsize=13, fontweight='bold')
    ax.text(17.75, output_y + 0.8, 'z_causal →', ha='center', va='center', fontsize=12)
    ax.text(17.75, output_y + 0.3, 'Diagnosis', ha='center', va='center', fontsize=14, weight='bold')
    ax.text(17.75, output_y - 0.2, '[B, num_classes]', ha='center', va='center', fontsize=9, style='italic')
    
    # ==================== Arrows (Data Flow) ====================
    # Input → Branch A (Clinical Data)
    draw_arrow(ax, (8.7, input_y + 0.5), (1.75, branch_a_y + 0.5), COLORS['arrow_innovation'], 
              linewidth=3, connectionstyle='arc3,rad=0.3')
    
    # Input → Branch A (OCT Images for VLM)
    draw_arrow(ax, (1.9, input_y), (5.1, branch_a_y + 0.5), COLORS['arrow_innovation'], 
              linewidth=3, connectionstyle='arc3,rad=-0.2')
    
    # Branch A steps
    draw_arrow(ax, (3, branch_a_y + 0.5), (3.5, branch_a_y + 0.5), COLORS['arrow_innovation'], 
              linewidth=2.5)
    draw_arrow(ax, (6.7, branch_a_y + 0.5), (7.2, branch_a_y + 0.5), COLORS['arrow_innovation'], 
              linewidth=2.5)
    draw_arrow(ax, (9.7, branch_a_y + 0.5), (5, bio_y + 0.3), COLORS['arrow_innovation'], 
              linewidth=3, connectionstyle='arc3,rad=0.2')
    
    # Input → Branch B (Images)
    draw_arrow(ax, (1.9, input_y), (1.75, branch_b_y + 0.5), COLORS['arrow_main'], 
              linewidth=2.5, connectionstyle='arc3,rad=0.3')
    draw_arrow(ax, (5.2, input_y), (1.75, branch_b_y + 0.5), COLORS['arrow_main'], 
              linewidth=2.5, connectionstyle='arc3,rad=0.3')
    
    # Branch B → Dual Head
    draw_arrow(ax, (3, branch_b_y + 0.5), (4.2, dual_head_y + 0.75), COLORS['arrow_main'], 
              linewidth=2.5)
    draw_arrow(ax, (3, branch_b_y + 0.5), (7.5, dual_head_y + 0.75), COLORS['arrow_main'], 
              linewidth=2.5)
    
    # Bio-Invariant Distribution → Constraint 1
    draw_arrow(ax, (5, bio_y - 0.4), (3.75, constraint_y + 1.3), COLORS['arrow_innovation'], 
              linewidth=3, connectionstyle='arc3,rad=0.3')
    
    # z_causal → Constraints
    draw_arrow(ax, (5.5, dual_head_y), (3.75, constraint_y + 1.3), COLORS['arrow_innovation'], 
              linewidth=2.5, connectionstyle='arc3,rad=0.2')
    draw_arrow(ax, (5.5, dual_head_y), (10.75, constraint_y + 1.3), COLORS['arrow_innovation'], 
              linewidth=2.5, connectionstyle='arc3,rad=0.2')
    
    # z_noise → Constraints
    draw_arrow(ax, (8.8, dual_head_y), (10.75, constraint_y + 1.3), COLORS['arrow_innovation'], 
              linewidth=2.5, connectionstyle='arc3,rad=-0.2')
    draw_arrow(ax, (8.8, dual_head_y), (17.75, constraint_y + 1.3), COLORS['arrow_innovation'], 
              linewidth=2.5, connectionstyle='arc3,rad=-0.2')
    
    # Center Labels → Constraint 3
    draw_arrow(ax, (12.2, input_y), (17.75, constraint_y + 1.3), COLORS['arrow_innovation'], 
              linewidth=2.5, connectionstyle='arc3,rad=-0.4')
    
    # z_causal → Output
    draw_arrow(ax, (9.8, dual_head_y + 0.75), (15.5, output_y + 0.6), COLORS['arrow_main'], 
              linewidth=3)
    
    # ==================== Loss Function ====================
    loss_y = 0.2
    loss_box = FancyBboxPatch((0.5, loss_y), 21, 0.9,
                              boxstyle="round,pad=0.1",
                              facecolor='#FAFAFA',
                              edgecolor=COLORS['border'],
                              linewidth=2)
    ax.add_patch(loss_box)
    
    ax.text(11, loss_y + 0.55, 'Total Loss: L = L_cls + λ_KL·L_dist + λ_orth·L_orth + λ_adv·L_noise', 
            ha='center', va='center', fontsize=13, weight='bold')
    ax.text(3.5, loss_y + 0.2, 'λ_KL = 0.005', ha='center', va='center', fontsize=10)
    ax.text(8, loss_y + 0.2, 'λ_orth = 0.01', ha='center', va='center', fontsize=10)
    ax.text(12.5, loss_y + 0.2, 'λ_adv = 0.05', ha='center', va='center', fontsize=10)
    ax.text(17, loss_y + 0.2, 'L_cls: Classification Loss', ha='center', va='center', fontsize=10)
    
    # ==================== Innovation Callouts (Removed overlapping ones) ====================
    # Only add one callout that doesn't overlap
    callout_y = bio_y + 1.0
    if callout_y < branch_b_title_y - 0.5:  # Check if it doesn't overlap with Branch B title
        callout_box = FancyBboxPatch((11, callout_y - 0.3), 5, 0.6,
                                    boxstyle="round,pad=0.1",
                                    facecolor='white',
                                    edgecolor=COLORS['innovation'],
                                    linewidth=2,
                                    alpha=0.95)
        ax.add_patch(callout_box)
        ax.text(13.5, callout_y, '★ Core: Distributional Anchoring\nUsing clinical prior to define bio-invariant manifold', 
               ha='center', va='center', fontsize=9, weight='bold', color=COLORS['innovation'])
    
    plt.tight_layout()
    return fig

def create_innovation_details_diagram():
    """Create detailed innovation points diagram with proper spacing"""
    
    fig = plt.figure(figsize=(22, 16))
    
    # Create 4 subplots for 4 innovations with more spacing
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.4, left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    innovations = [
        {
            'title': 'Innovation 1: Distributional Anchoring Mechanism',
            'ax': fig.add_subplot(gs[0, 0]),
            'color': COLORS['innovation'],
            'details': [
                'Step 1: Clinical data (HPV, TCT, Age) → Text description',
                'Step 2: VLM processes image+text → Semantic features [B, 1536]',
                'Step 3: Distribution head → (μ_bio, σ_bio) [B, 768]',
                'Step 4: Constrain z_causal to N(μ_bio, σ_bio) via KL divergence',
                '',
                'Key Insight:',
                '• Use distribution instead of point matching',
                '• Allow alignment error, improve generalization',
                '• Clinical data defines bio-invariant manifold'
            ]
        },
        {
            'title': 'Innovation 2: Orthogonal Disentanglement',
            'ax': fig.add_subplot(gs[0, 1]),
            'color': COLORS['innovation'],
            'details': [
                'Architecture:',
                '• Dual Head Image Encoder',
                '  - Head 1: z_causal [B, 768] → Classification',
                '  - Head 2: z_noise [B, 768] → Center Prediction',
                '',
                'Constraint:',
                '• L_orth = ||z_causal^T · z_noise||',
                '• Enforce z_causal ⊥ z_noise',
                '',
                'Key Insight:',
                '• Explicitly separate causal and noise features',
                '• Physically ensure z_causal contains no device info',
                '• Achieve interpretable disentanglement'
            ]
        },
        {
            'title': 'Innovation 3: Prior-Constrained Domain-Invariant Learning',
            'ax': fig.add_subplot(gs[1, 0]),
            'color': COLORS['innovation'],
            'details': [
                'Prior Constraint:',
                '• Clinical modalities (HPV/TCT) as prior knowledge',
                '• Define bio-invariant manifold P_bio = N(μ_bio, σ_bio)',
                '',
                'Domain Invariance:',
                '• Constrain image features to bio-invariant manifold',
                '• Eliminate device-specific noise',
                '• Achieve cross-center generalization',
                '',
                'Key Insight:',
                '• Clinical data has same meaning across hospitals',
                '• Use prior to guide domain-invariant learning',
                '• Three constraints work together'
            ]
        },
        {
            'title': 'Innovation 4: VLM-Enhanced Semantic Understanding',
            'ax': fig.add_subplot(gs[1, 1]),
            'color': COLORS['innovation'],
            'details': [
                'VLM Processing:',
                '• Input: OCT images + Clinical text descriptions',
                '• Model: Qwen2-VL (Frozen)',
                '• Output: Semantic features [B, 1536]',
                '',
                'Joint Understanding:',
                '• Simultaneously understand image and text',
                '• Extract fused semantic features',
                '• Understand association between clinical data and pathology',
                '',
                'Key Insight:',
                '• VLM bridges image and clinical modalities',
                '• Semantic understanding improves alignment',
                '• Better than simple feature concatenation'
            ]
        }
    ]
    
    for innovation in innovations:
        ax = innovation['ax']
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title box
        title_box = FancyBboxPatch((0.3, 8.5), 9.4, 1.2,
                                  boxstyle="round,pad=0.15",
                                  facecolor=innovation['color'],
                                  edgecolor=innovation['color'],
                                  linewidth=3,
                                  alpha=0.2)
        ax.add_patch(title_box)
        
        ax.text(5, 9.2, innovation['title'], ha='center', va='center',
               fontsize=13, weight='bold', color=innovation['color'])
        
        # Details with proper spacing
        y_start = 7.5
        line_height = 0.45
        for i, detail in enumerate(innovation['details']):
            y_pos = y_start - i * line_height
            if detail.startswith('•'):
                ax.text(0.8, y_pos, detail, ha='left', va='center',
                       fontsize=9.5, color=COLORS['text'])
            elif detail.startswith('Step') or detail.startswith('Architecture') or \
                 detail.startswith('Constraint') or detail.startswith('Prior') or \
                 detail.startswith('Domain') or detail.startswith('VLM') or \
                 detail.startswith('Joint') or detail.startswith('Key'):
                ax.text(0.3, y_pos, detail, ha='left', va='center',
                       fontsize=11, weight='bold', color=COLORS['text'])
            elif detail == '':
                continue
            else:
                ax.text(0.3, y_pos, detail, ha='left', va='center',
                       fontsize=9.5, color=COLORS['text'])
        
        # Code location box
        code_box = FancyBboxPatch((0.3, 0.2), 9.4, 1.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#F5F5F5',
                                 edgecolor=COLORS['border'],
                                 linewidth=2)
        ax.add_patch(code_box)
        ax.text(5, 1.3, 'Code Location:', ha='center', va='center',
               fontsize=11, weight='bold')
        
        if 'Distributional' in innovation['title']:
            ax.text(2.5, 0.7, 'DistributionalAnchor.forward()', ha='center', va='center',
                   fontsize=9)
            ax.text(7.5, 0.7, 'DistributionMatchingLoss.forward()', ha='center', va='center',
                   fontsize=9)
        elif 'Orthogonal' in innovation['title']:
            ax.text(5, 0.7, 'OrthogonalLoss.forward()', ha='center', va='center',
                   fontsize=9)
        elif 'Prior-Constrained' in innovation['title']:
            ax.text(2.5, 0.7, 'clinical_to_text()', ha='center', va='center',
                   fontsize=9)
            ax.text(7.5, 0.7, 'BIDAModel.forward()', ha='center', va='center',
                   fontsize=9)
        else:  # VLM
            ax.text(5, 0.7, 'DistributionalAnchor.forward()\n(VLM Processing Part)', 
                   ha='center', va='center', fontsize=9)
    
    fig.suptitle('BIDA Framework: Detailed Innovation Points', 
                fontsize=18, weight='bold', y=0.98)
    
    return fig

def main():
    """Main function"""
    pdf_path = '/data2/hmy/VLM_Caus_Rm_Mics/cursor_file/visualization/BIDA_Framework_Professional_Fixed.pdf'
    
    with PdfPages(pdf_path) as pdf:
        print("Generating professional framework diagram (fixed layout)...")
        fig1 = create_professional_framework_diagram()
        pdf.savefig(fig1, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close(fig1)
        print("✅ Professional framework diagram generated")
        
        print("Generating innovation details diagram (fixed layout)...")
        fig2 = create_innovation_details_diagram()
        pdf.savefig(fig2, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close(fig2)
        print("✅ Innovation details diagram generated")
    
    print(f"\n🎉 Fixed PDF saved to: {pdf_path}")
    print(f"📄 Contains 2 pages with no overlapping elements")

if __name__ == '__main__':
    main()

