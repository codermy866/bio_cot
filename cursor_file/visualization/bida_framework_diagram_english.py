#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BIDA Framework Diagram Generator (English Version)
Generate PDF format framework diagram showing distributional anchoring mechanism
and prior-constrained domain-invariant learning
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Set font (use English to avoid font issues)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def create_bida_framework_diagram():
    """Create BIDA framework diagram"""
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    colors = {
        'input': '#E8F4F8',
        'branch_a': '#FFE5E5',
        'branch_b': '#E5F5E5',
        'constraint': '#FFF5E5',
        'output': '#F0E8FF',
        'bio_dist': '#FFD700',
        'arrow': '#333333'
    }
    
    # ==================== Title ====================
    ax.text(9, 11.5, 'BIDA Framework: Distributional Anchoring for Zero-Shot Cross-Center Generalization', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    # ==================== Input Layer ====================
    # OCT Images
    oct_box = FancyBboxPatch((0.5, 9), 2.5, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['input'], 
                             edgecolor='black', linewidth=2)
    ax.add_patch(oct_box)
    ax.text(1.75, 10, 'OCT Images', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(1.75, 9.5, '[B, C, H, W]', ha='center', va='center', fontsize=9)
    
    # Colposcopy Images
    col_box = FancyBboxPatch((3.5, 9), 2.5, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['input'], 
                             edgecolor='black', linewidth=2)
    ax.add_patch(col_box)
    ax.text(4.75, 10, 'Colposcopy', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(4.75, 9.5, 'Images [B, C, H, W]', ha='center', va='center', fontsize=9)
    
    # Clinical Data (Prior Knowledge)
    clinical_box = FancyBboxPatch((6.5, 9), 3, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['input'], 
                                  edgecolor='blue', linewidth=2)
    ax.add_patch(clinical_box)
    ax.text(8, 10, 'Clinical Data', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(8, 9.6, 'HPV, TCT, Age', ha='center', va='center', fontsize=9)
    ax.text(8, 9.2, '(Prior Knowledge)', ha='center', va='center', fontsize=8, style='italic', color='blue')
    
    # Center Labels
    center_box = FancyBboxPatch((10, 9), 2.5, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['input'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(center_box)
    ax.text(11.25, 10, 'Center ID', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(11.25, 9.5, '[0-4]', ha='center', va='center', fontsize=9)
    
    # ==================== Branch A: Distributional Anchor ====================
    # Branch A Title
    ax.text(1, 8.2, 'Branch A: Distributional Anchor (Distributional Anchoring Mechanism)', 
            ha='left', va='center', fontsize=12, weight='bold', color='red')
    
    # Clinical Data → Text Description
    text_box = FancyBboxPatch((0.5, 6.8), 3.5, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['branch_a'], 
                              edgecolor='red', linewidth=2)
    ax.add_patch(text_box)
    ax.text(2.25, 7.3, 'Clinical Data', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(2.25, 7.0, '→ Text Description', ha='center', va='center', fontsize=9)
    ax.text(2.25, 6.7, 'clinical_to_text()', ha='center', va='center', fontsize=8, style='italic')
    
    # VLM Processing
    vlm_box = FancyBboxPatch((4.5, 6.8), 4, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['branch_a'], 
                             edgecolor='red', linewidth=2)
    ax.add_patch(vlm_box)
    ax.text(6.5, 7.3, 'VLM: Image+Text', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(6.5, 7.0, 'Joint Understanding', ha='center', va='center', fontsize=9)
    ax.text(6.5, 6.7, 'Qwen2-VL (Frozen)', ha='center', va='center', fontsize=8, style='italic')
    
    # Distribution Parameter Generation
    dist_box = FancyBboxPatch((9, 6.8), 3.5, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['branch_a'], 
                              edgecolor='red', linewidth=2)
    ax.add_patch(dist_box)
    ax.text(10.75, 7.3, 'Distribution', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(10.75, 7.0, 'Parameter Generation', ha='center', va='center', fontsize=9)
    ax.text(10.75, 6.7, 'distribution_head()', ha='center', va='center', fontsize=8, style='italic')
    
    # Bio-Invariant Distribution
    bio_dist_box = FancyBboxPatch((4.5, 5.2), 9, 1.2, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['bio_dist'], 
                                  edgecolor='red', linewidth=3)
    ax.add_patch(bio_dist_box)
    ax.text(9, 5.9, 'Bio-Invariant Distribution (Anchor Point)', ha='center', va='center', 
            fontsize=13, weight='bold', color='red')
    ax.text(9, 5.5, 'P_bio = N(μ_bio, σ_bio)', ha='center', va='center', fontsize=11, style='italic', weight='bold')
    ax.text(6.5, 5.2, 'μ_bio [B, 768]', ha='center', va='center', fontsize=9)
    ax.text(11.5, 5.2, 'σ_bio [B, 768]', ha='center', va='center', fontsize=9)
    
    # ==================== Branch B: Dual Head Image Encoder ====================
    # Branch B Title
    ax.text(1, 4.5, 'Branch B: Dual Head Image Encoder', 
            ha='left', va='center', fontsize=12, weight='bold', color='green')
    
    # Image Feature Extraction
    img_feat_box = FancyBboxPatch((0.5, 3), 3.5, 1, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['branch_b'], 
                                  edgecolor='green', linewidth=2)
    ax.add_patch(img_feat_box)
    ax.text(2.25, 3.5, 'Image Feature', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(2.25, 3.2, 'Extraction', ha='center', va='center', fontsize=9)
    ax.text(2.25, 3.0, 'ResNet50 → [B, 512]', ha='center', va='center', fontsize=8, style='italic')
    
    # Dual Head Network
    dual_head_box = FancyBboxPatch((4.5, 2.2), 8, 2, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['branch_b'], 
                                   edgecolor='green', linewidth=2)
    ax.add_patch(dual_head_box)
    ax.text(8.5, 3.7, 'Dual Head Network', ha='center', va='center', fontsize=11, weight='bold')
    
    # Head 1: z_causal
    causal_head_box = FancyBboxPatch((5, 2.8), 3, 0.9, 
                                     boxstyle="round,pad=0.05", 
                                     facecolor='#90EE90', 
                                     edgecolor='green', linewidth=2)
    ax.add_patch(causal_head_box)
    ax.text(6.5, 3.25, 'Head 1: z_causal', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(6.5, 3.0, '[B, 768] Causal Features', ha='center', va='center', fontsize=8)
    ax.text(6.5, 2.75, '(For Classification)', ha='center', va='center', fontsize=7, style='italic')
    
    # Head 2: z_noise
    noise_head_box = FancyBboxPatch((9.5, 2.8), 3, 0.9, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor='#FFB6C1', 
                                    edgecolor='green', linewidth=2)
    ax.add_patch(noise_head_box)
    ax.text(11, 3.25, 'Head 2: z_noise', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(11, 3.0, '[B, 768] Noise Features', ha='center', va='center', fontsize=8)
    ax.text(11, 2.75, '(For Center Prediction)', ha='center', va='center', fontsize=7, style='italic')
    
    # ==================== Constraint Mechanisms ====================
    # Constraint Title
    ax.text(1, 1.6, 'Constraint Mechanisms (Prior-Constrained Domain-Invariant Learning)', 
            ha='left', va='center', fontsize=12, weight='bold', color='orange')
    
    # Constraint 1: Distribution Matching
    constraint1_box = FancyBboxPatch((0.5, 0.2), 5, 1.2, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=colors['constraint'], 
                                     edgecolor='orange', linewidth=2)
    ax.add_patch(constraint1_box)
    ax.text(3, 0.9, 'Constraint 1: Distributional Anchoring', ha='center', va='center', 
            fontsize=11, weight='bold')
    ax.text(3, 0.6, 'L_dist = D_KL(Q(z_causal) || P_bio)', ha='center', va='center', 
            fontsize=9, style='italic')
    ax.text(3, 0.3, 'z_causal must be within N(μ_bio, σ_bio)', ha='center', va='center', fontsize=8)
    
    # Constraint 2: Orthogonal Disentanglement
    constraint2_box = FancyBboxPatch((6, 0.2), 5, 1.2, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=colors['constraint'], 
                                     edgecolor='orange', linewidth=2)
    ax.add_patch(constraint2_box)
    ax.text(8.5, 0.9, 'Constraint 2: Orthogonal Disentanglement', ha='center', va='center', 
            fontsize=11, weight='bold')
    ax.text(8.5, 0.6, 'L_orth = |z_causal^T · z_noise|', ha='center', va='center', 
            fontsize=9, style='italic')
    ax.text(8.5, 0.3, 'z_causal ⊥ z_noise', ha='center', va='center', fontsize=8)
    
    # Constraint 3: Noise Supervision
    constraint3_box = FancyBboxPatch((11.5, 0.2), 5, 1.2, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=colors['constraint'], 
                                     edgecolor='orange', linewidth=2)
    ax.add_patch(constraint3_box)
    ax.text(14, 0.9, 'Constraint 3: Noise Supervision', ha='center', va='center', 
            fontsize=11, weight='bold')
    ax.text(14, 0.6, 'L_noise = CE(CenterPred(z_noise), d)', ha='center', va='center', 
            fontsize=9, style='italic')
    ax.text(14, 0.3, 'z_noise → Center ID', ha='center', va='center', fontsize=8)
    
    # ==================== Output Layer ====================
    output_box = FancyBboxPatch((14.5, 2.2), 3, 2, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['output'], 
                                edgecolor='purple', linewidth=3)
    ax.add_patch(output_box)
    ax.text(16, 3.7, 'Output', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(16, 3.2, 'Classifier', ha='center', va='center', fontsize=10)
    ax.text(16, 2.8, 'z_causal →', ha='center', va='center', fontsize=9)
    ax.text(16, 2.5, 'Diagnosis', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(16, 2.2, '[B, num_classes]', ha='center', va='center', fontsize=8)
    
    # ==================== Arrows ====================
    # Input → Branch A (Clinical Data)
    arrow1 = FancyArrowPatch((8, 9), (2.25, 7.3), 
                            arrowstyle='->', lw=2, color='red', 
                            connectionstyle="arc3,rad=0.2")
    ax.add_patch(arrow1)
    
    # Input → Branch A (OCT Images)
    arrow1_img = FancyArrowPatch((1.75, 9), (6.5, 7.3), 
                                 arrowstyle='->', lw=2, color='red', 
                                 connectionstyle="arc3,rad=-0.2")
    ax.add_patch(arrow1_img)
    
    # Branch A → Bio-Invariant Distribution
    arrow2 = FancyArrowPatch((10.75, 6.8), (9, 6.4), 
                            arrowstyle='->', lw=2, color='red')
    ax.add_patch(arrow2)
    
    # Input → Branch B (OCT)
    arrow3 = FancyArrowPatch((1.75, 9), (2.25, 4), 
                            arrowstyle='->', lw=2, color='green', 
                            connectionstyle="arc3,rad=0.3")
    ax.add_patch(arrow3)
    
    # Input → Branch B (Colposcopy)
    arrow4 = FancyArrowPatch((4.75, 9), (2.25, 4), 
                            arrowstyle='->', lw=2, color='green', 
                            connectionstyle="arc3,rad=0.3")
    ax.add_patch(arrow4)
    
    # Branch B → Dual Head
    arrow5 = FancyArrowPatch((4, 3.5), (5, 3.25), 
                            arrowstyle='->', lw=2, color='green')
    ax.add_patch(arrow5)
    
    arrow6 = FancyArrowPatch((4, 3.5), (9.5, 3.25), 
                            arrowstyle='->', lw=2, color='green')
    ax.add_patch(arrow6)
    
    # Bio-Invariant Distribution → Constraint 1
    arrow7 = FancyArrowPatch((9, 5.2), (3, 1.4), 
                            arrowstyle='->', lw=2, color='orange', 
                            connectionstyle="arc3,rad=0.3")
    ax.add_patch(arrow7)
    
    # z_causal → Constraint 1
    arrow8 = FancyArrowPatch((6.5, 2.8), (3, 1.4), 
                            arrowstyle='->', lw=2, color='orange', 
                            connectionstyle="arc3,rad=0.2")
    ax.add_patch(arrow8)
    
    # z_causal → Constraint 2
    arrow9 = FancyArrowPatch((6.5, 2.8), (8.5, 1.4), 
                            arrowstyle='->', lw=2, color='orange', 
                            connectionstyle="arc3,rad=0.2")
    ax.add_patch(arrow9)
    
    # z_noise → Constraint 2
    arrow10 = FancyArrowPatch((11, 2.8), (8.5, 1.4), 
                             arrowstyle='->', lw=2, color='orange', 
                             connectionstyle="arc3,rad=-0.2")
    ax.add_patch(arrow10)
    
    # z_noise → Constraint 3
    arrow11 = FancyArrowPatch((11, 2.8), (14, 1.4), 
                             arrowstyle='->', lw=2, color='orange', 
                             connectionstyle="arc3,rad=-0.2")
    ax.add_patch(arrow11)
    
    # Center ID → Constraint 3
    arrow12 = FancyArrowPatch((11.25, 9), (14, 1.4), 
                             arrowstyle='->', lw=2, color='orange', 
                             connectionstyle="arc3,rad=-0.4")
    ax.add_patch(arrow12)
    
    # z_causal → Output
    arrow13 = FancyArrowPatch((8, 3.25), (14.5, 3.2), 
                             arrowstyle='->', lw=3, color='purple')
    ax.add_patch(arrow13)
    
    # ==================== Loss Function ====================
    loss_box = FancyBboxPatch((0.5, 10.5), 17, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#F0F0F0', 
                              edgecolor='black', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(9, 10.9, 'Total Loss Function', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(9, 10.6, 'L = L_cls + λ_KL·L_dist + λ_orth·L_orth + λ_adv·L_noise', 
            ha='center', va='center', fontsize=10, style='italic')
    ax.text(3, 10.3, 'λ_KL = 0.005', ha='center', va='center', fontsize=9)
    ax.text(9, 10.3, 'λ_orth = 0.01', ha='center', va='center', fontsize=9)
    ax.text(15, 10.3, 'λ_adv = 0.05', ha='center', va='center', fontsize=9)
    
    # ==================== Legend ====================
    legend_elements = [
        mpatches.Patch(facecolor=colors['input'], edgecolor='black', label='Input'),
        mpatches.Patch(facecolor=colors['branch_a'], edgecolor='red', label='Branch A: Distributional Anchoring'),
        mpatches.Patch(facecolor=colors['branch_b'], edgecolor='green', label='Branch B: Dual Head Encoder'),
        mpatches.Patch(facecolor=colors['constraint'], edgecolor='orange', label='Constraints'),
        mpatches.Patch(facecolor=colors['output'], edgecolor='purple', label='Output'),
        mpatches.Patch(facecolor=colors['bio_dist'], edgecolor='red', label='Bio-Invariant Distribution')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    return fig

def create_innovation_points_diagram():
    """Create innovation points detailed diagram"""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('BIDA Framework: Core Innovation Points', fontsize=16, weight='bold', y=0.98)
    
    # Innovation 1: Distributional Anchoring
    ax1 = axes[0, 0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    ax1.text(5, 9, 'Innovation 1: Distributional Anchoring Mechanism', ha='center', va='center', 
            fontsize=14, weight='bold', color='red')
    
    # Prior Constraint
    prior_box = FancyBboxPatch((1, 7), 3.5, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#FFE5E5', 
                              edgecolor='red', linewidth=2)
    ax1.add_patch(prior_box)
    ax1.text(2.75, 8, 'Prior Constraint', ha='center', va='center', fontsize=11, weight='bold')
    ax1.text(2.75, 7.5, 'Clinical Data', ha='center', va='center', fontsize=9)
    ax1.text(2.75, 7.1, '(HPV/TCT)', ha='center', va='center', fontsize=9)
    ax1.text(2.75, 6.8, '→ Bio-Invariant Distribution', ha='center', va='center', fontsize=8)
    
    # Distributional Anchoring
    anchor_box = FancyBboxPatch((5.5, 7), 3.5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#E5F5E5', 
                               edgecolor='green', linewidth=2)
    ax1.add_patch(anchor_box)
    ax1.text(7.25, 8, 'Distributional', ha='center', va='center', fontsize=11, weight='bold')
    ax1.text(7.25, 7.7, 'Anchoring', ha='center', va='center', fontsize=11, weight='bold')
    ax1.text(7.25, 7.2, 'z_causal → N(μ_bio, σ_bio)', ha='center', va='center', fontsize=9)
    ax1.text(7.25, 6.8, 'KL Divergence Constraint', ha='center', va='center', fontsize=8)
    
    # Arrow
    arrow = FancyArrowPatch((4.5, 7.75), (5.5, 7.75), 
                           arrowstyle='->', lw=2, color='black')
    ax1.add_patch(arrow)
    
    # Description
    ax1.text(5, 5.5, 'Core Idea:', ha='center', va='center', fontsize=11, weight='bold')
    ax1.text(5, 4.8, 'Use clinical modalities to define distribution N(μ_bio, σ_bio)', 
            ha='center', va='center', fontsize=10)
    ax1.text(5, 4.3, 'Constrain image features z_causal to fall within this distribution', 
            ha='center', va='center', fontsize=10)
    ax1.text(5, 3.8, 'Allow alignment error, improve generalization', 
            ha='center', va='center', fontsize=10, style='italic')
    
    # Code Location
    code_box = FancyBboxPatch((1, 1), 8, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#F0F0F0', 
                              edgecolor='gray', linewidth=1)
    ax1.add_patch(code_box)
    ax1.text(5, 2, 'Code Location:', ha='center', va='center', fontsize=10, weight='bold')
    ax1.text(2.5, 1.5, 'DistributionalAnchor.forward()', ha='center', va='center', fontsize=8)
    ax1.text(7.5, 1.5, 'DistributionMatchingLoss.forward()', ha='center', va='center', fontsize=8)
    
    # Innovation 2: Orthogonal Disentanglement
    ax2 = axes[0, 1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    ax2.text(5, 9, 'Innovation 2: Orthogonal Disentanglement Mechanism', ha='center', va='center', 
            fontsize=14, weight='bold', color='green')
    
    # Dual Head Network
    dual_box = FancyBboxPatch((2, 7), 6, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#E5F5E5', 
                              edgecolor='green', linewidth=2)
    ax2.add_patch(dual_box)
    ax2.text(5, 8, 'Dual Head Image Encoder', ha='center', va='center', fontsize=11, weight='bold')
    ax2.text(3.5, 7.5, 'Head 1: z_causal', ha='center', va='center', fontsize=9)
    ax2.text(6.5, 7.5, 'Head 2: z_noise', ha='center', va='center', fontsize=9)
    
    # Orthogonal Constraint
    orth_box = FancyBboxPatch((2, 5), 6, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor='#FFF5E5', 
                             edgecolor='orange', linewidth=2)
    ax2.add_patch(orth_box)
    ax2.text(5, 5.5, 'Orthogonal Constraint: z_causal ⊥ z_noise', ha='center', va='center', 
            fontsize=10, weight='bold')
    
    # Description
    ax2.text(5, 3.5, 'Core Idea:', ha='center', va='center', fontsize=11, weight='bold')
    ax2.text(5, 2.8, 'Explicitly separate causal and noise features', ha='center', va='center', fontsize=10)
    ax2.text(5, 2.3, 'Physically ensure z_causal contains no device information', 
            ha='center', va='center', fontsize=10)
    ax2.text(5, 1.8, 'Achieve interpretable disentanglement via orthogonal constraint', 
            ha='center', va='center', fontsize=10, style='italic')
    
    # Code Location
    code_box2 = FancyBboxPatch((1, 0.5), 8, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#F0F0F0', 
                               edgecolor='gray', linewidth=1)
    ax2.add_patch(code_box2)
    ax2.text(5, 1, 'Code Location: OrthogonalLoss.forward()', ha='center', va='center', fontsize=9)
    
    # Innovation 3: Prior-Constrained Domain-Invariant Learning
    ax3 = axes[1, 0]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    
    ax3.text(5, 9, 'Innovation 3: Prior-Constrained Domain-Invariant Learning', 
            ha='center', va='center', fontsize=14, weight='bold', color='blue')
    
    # Prior Knowledge
    prior_box3 = FancyBboxPatch((1, 7), 3.5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#E8F4F8', 
                               edgecolor='blue', linewidth=2)
    ax3.add_patch(prior_box3)
    ax3.text(2.75, 8, 'Prior Knowledge', ha='center', va='center', fontsize=11, weight='bold')
    ax3.text(2.75, 7.5, 'Clinical Data', ha='center', va='center', fontsize=9)
    ax3.text(2.75, 7.1, '(HPV/TCT)', ha='center', va='center', fontsize=9)
    
    # Domain Invariance
    inv_box = FancyBboxPatch((5.5, 7), 3.5, 1.5, 
                            boxstyle="round,pad=0.1", 
                            facecolor='#E5F5E5', 
                            edgecolor='green', linewidth=2)
    ax3.add_patch(inv_box)
    ax3.text(7.25, 8, 'Domain Invariance', ha='center', va='center', fontsize=11, weight='bold')
    ax3.text(7.25, 7.5, 'Constraint to', ha='center', va='center', fontsize=9)
    ax3.text(7.25, 7.2, 'Bio-Invariant Manifold', ha='center', va='center', fontsize=9)
    ax3.text(7.25, 6.8, 'Eliminate Device Noise', ha='center', va='center', fontsize=8)
    
    # Arrow
    arrow3 = FancyArrowPatch((4.5, 7.75), (5.5, 7.75), 
                             arrowstyle='->', lw=2, color='black')
    ax3.add_patch(arrow3)
    
    # Description
    ax3.text(5, 5.5, 'Core Idea:', ha='center', va='center', fontsize=11, weight='bold')
    ax3.text(5, 4.8, 'Use clinical modalities as prior to define bio-invariant manifold', 
            ha='center', va='center', fontsize=10)
    ax3.text(5, 4.3, 'Achieve domain invariance by constraining image features to manifold', 
            ha='center', va='center', fontsize=10)
    ax3.text(5, 3.8, 'Clinical data has same meaning across hospitals (bio-invariance)', 
            ha='center', va='center', fontsize=10, style='italic')
    
    # Code Location
    code_box3 = FancyBboxPatch((1, 1), 8, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#F0F0F0', 
                               edgecolor='gray', linewidth=1)
    ax3.add_patch(code_box3)
    ax3.text(5, 2, 'Code Location:', ha='center', va='center', fontsize=10, weight='bold')
    ax3.text(2.5, 1.5, 'clinical_to_text()', ha='center', va='center', fontsize=8)
    ax3.text(7.5, 1.5, 'BIDAModel.forward()', ha='center', va='center', fontsize=8)
    
    # Innovation 4: VLM-Enhanced Semantic Understanding
    ax4 = axes[1, 1]
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    
    ax4.text(5, 9, 'Innovation 4: VLM-Enhanced Semantic Understanding', ha='center', va='center', 
            fontsize=14, weight='bold', color='purple')
    
    # VLM Processing
    vlm_box4 = FancyBboxPatch((2, 7), 6, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#F0E8FF', 
                              edgecolor='purple', linewidth=2)
    ax4.add_patch(vlm_box4)
    ax4.text(5, 8, 'VLM: Image+Text', ha='center', va='center', fontsize=11, weight='bold')
    ax4.text(3.5, 7.5, 'OCT Images', ha='center', va='center', fontsize=9)
    ax4.text(5, 7.5, '+', ha='center', va='center', fontsize=12, weight='bold')
    ax4.text(6.5, 7.5, 'Clinical Text', ha='center', va='center', fontsize=9)
    ax4.text(5, 7.1, '→ Semantic Features [B, 1536]', ha='center', va='center', fontsize=9)
    
    # Description
    ax4.text(5, 5.5, 'Core Idea:', ha='center', va='center', fontsize=11, weight='bold')
    ax4.text(5, 4.8, 'VLM simultaneously understands image content and text semantics', 
            ha='center', va='center', fontsize=10)
    ax4.text(5, 4.3, 'Extract semantic features fused from image and text', 
            ha='center', va='center', fontsize=10)
    ax4.text(5, 3.8, 'Understand association between "HPV positive" and pathological features', 
            ha='center', va='center', fontsize=10, style='italic')
    
    # Code Location
    code_box4 = FancyBboxPatch((1, 1), 8, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#F0F0F0', 
                               edgecolor='gray', linewidth=1)
    ax4.add_patch(code_box4)
    ax4.text(5, 2, 'Code Location:', ha='center', va='center', fontsize=10, weight='bold')
    ax4.text(5, 1.5, 'DistributionalAnchor.forward() - VLM Processing Part', ha='center', va='center', fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def main():
    """Main function: Generate PDF"""
    
    # Create PDF file
    pdf_path = '/data2/hmy/VLM_Caus_Rm_Mics/cursor_file/visualization/BIDA_Framework_Diagram.pdf'
    
    with PdfPages(pdf_path) as pdf:
        # Figure 1: Overall Framework Diagram
        print("Generating overall framework diagram...")
        fig1 = create_bida_framework_diagram()
        pdf.savefig(fig1, bbox_inches='tight', dpi=300)
        plt.close(fig1)
        print("✅ Overall framework diagram generated")
        
        # Figure 2: Innovation Points Detailed Diagram
        print("Generating innovation points detailed diagram...")
        fig2 = create_innovation_points_diagram()
        pdf.savefig(fig2, bbox_inches='tight', dpi=300)
        plt.close(fig2)
        print("✅ Innovation points detailed diagram generated")
    
    print(f"\n🎉 PDF file saved to: {pdf_path}")
    print(f"📄 Contains 2 pages:")
    print(f"   1. BIDA Overall Framework Diagram")
    print(f"   2. Core Innovation Points Detailed Diagram")

if __name__ == '__main__':
    main()


