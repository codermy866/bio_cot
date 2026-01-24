#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制四个创新点的超详细流程图
包含数学公式和细化的控制流程
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch, Circle, Rectangle, Polygon
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'  # 更好的数学公式字体

def draw_detailed_bayesian_uncertainty():
    """创新点1: 贝叶斯不确定性量化 - 超详细版"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    colors = {
        'input': '#E8F4F8',
        'process': '#B3E5FC',
        'decision': '#FFE082',
        'output': '#4FC3F7',
        'formula': '#FFF9C4'
    }
    
    # 标题
    title_box = FancyBboxPatch((1, 14.5), 10, 1, boxstyle="round,pad=0.2",
                               facecolor='lightblue', edgecolor='black', linewidth=3)
    ax.add_patch(title_box)
    ax.text(6, 15, 'Innovation 1: Bayesian Uncertainty Quantification\nDetailed Control Flow with Mathematical Formulas', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    y = 13
    
    # ========== 输入层 ==========
    input_box = FancyBboxPatch((1, y), 10, 0.8, boxstyle="round,pad=0.1",
                               facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(6, y+0.4, 'Input: Feature f ∈ ℝ^{B×D} from Swin-T Encoder', 
            ha='center', va='center', fontsize=11, weight='bold')
    y -= 1.2
    
    # ========== 分支：均值编码器 ==========
    # 决策节点（菱形）
    decision1 = Polygon([(3, y), (4, y+0.3), (3, y+0.6), (2, y+0.3)], 
                        facecolor=colors['decision'], edgecolor='black', linewidth=2)
    ax.add_patch(decision1)
    ax.text(3, y+0.3, 'Mean\nEncoder', ha='center', va='center', fontsize=9, weight='bold')
    
    # 均值编码器详细流程
    mean_process = FancyBboxPatch((5, y-0.3), 6, 1.2, boxstyle="round,pad=0.1",
                                  facecolor=colors['process'], edgecolor='red', linewidth=2)
    ax.add_patch(mean_process)
    mean_formula = r'$\mu = \text{MLP}_{\mu}(f)$'
    mean_detail = r'$\mu = W_2 \cdot \text{GELU}(\text{LayerNorm}(W_1 \cdot f + b_1)) + b_2$'
    ax.text(8, y+0.3, f'{mean_formula}\n{mean_detail}', ha='center', va='center', 
            fontsize=10, weight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # 箭头
    ax.annotate('', xy=(5, y+0.3), xytext=(4, y+0.3), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(3, y), xytext=(6, 13), arrowprops=dict(arrowstyle='->', lw=2))
    y -= 1.5
    
    # ========== 分支：方差编码器 ==========
    decision2 = Polygon([(3, y), (4, y+0.3), (3, y+0.6), (2, y+0.3)], 
                        facecolor=colors['decision'], edgecolor='black', linewidth=2)
    ax.add_patch(decision2)
    ax.text(3, y+0.3, 'Variance\nEncoder', ha='center', va='center', fontsize=9, weight='bold')
    
    var_process = FancyBboxPatch((5, y-0.3), 6, 1.2, boxstyle="round,pad=0.1",
                                 facecolor=colors['process'], edgecolor='red', linewidth=2)
    ax.add_patch(var_process)
    var_formula = r'$\sigma^2 = \text{Softplus}(\text{MLP}_{\sigma}(f)) + \epsilon$'
    var_detail = r'$\sigma^2 = \log(1 + \exp(W_2 \cdot \text{GELU}(\text{LayerNorm}(W_1 \cdot f)))) + 10^{-6}$'
    ax.text(8, y+0.3, f'{var_formula}\n{var_detail}', ha='center', va='center', 
            fontsize=10, weight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # 箭头
    ax.annotate('', xy=(5, y+0.3), xytext=(4, y+0.3), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(3, y), xytext=(6, 13), arrowprops=dict(arrowstyle='->', lw=2))
    y -= 1.5
    
    # ========== 采样决策 ==========
    decision3 = Polygon([(6, y), (7.5, y+0.4), (6, y+0.8), (4.5, y+0.4)], 
                        facecolor=colors['decision'], edgecolor='red', linewidth=3)
    ax.add_patch(decision3)
    ax.text(6, y+0.4, 'Training\nMode?', ha='center', va='center', fontsize=10, weight='bold')
    
    # 箭头到决策
    ax.annotate('', xy=(6, y+0.8), xytext=(8, y-0.3), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(6, y+0.8), xytext=(8, y-1.8), arrowprops=dict(arrowstyle='->', lw=2))
    y -= 1.2
    
    # 训练分支
    train_box = FancyBboxPatch((1, y-0.5), 4.5, 1.5, boxstyle="round,pad=0.1",
                               facecolor='#C8E6C9', edgecolor='green', linewidth=2)
    ax.add_patch(train_box)
    train_formula = r'$z = \mu + \epsilon \cdot \sqrt{\sigma^2}$'
    train_detail = r'where $\epsilon \sim \mathcal{N}(0, I)$ (reparameterization trick)'
    ax.text(3.25, y+0.25, f'Training: Sampling\n{train_formula}\n{train_detail}', 
            ha='center', va='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # 推理分支
    infer_box = FancyBboxPatch((6.5, y-0.5), 4.5, 1.5, boxstyle="round,pad=0.1",
                               facecolor='#FFCCBC', edgecolor='orange', linewidth=2)
    ax.add_patch(infer_box)
    infer_formula = r'$z = \mu$'
    infer_detail = r'Use mean (deterministic prediction)'
    ax.text(8.75, y+0.25, f'Inference: Mean\n{infer_formula}\n{infer_detail}', 
            ha='center', va='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # 箭头
    ax.annotate('', xy=(4.5, y+0.25), xytext=(6, y+0.4), arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.annotate('', xy=(6.5, y+0.25), xytext=(6, y+0.4), arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    y -= 2
    
    # ========== 输出 ==========
    output_box = FancyBboxPatch((1, y), 10, 0.8, boxstyle="round,pad=0.1",
                                facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(6, y+0.4, 'Output: Sampled Feature z ∈ ℝ^{B×D} + Uncertainty (μ, σ²)', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # 箭头
    ax.annotate('', xy=(6, y), xytext=(3.25, y-0.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(6, y), xytext=(8.75, y-0.5), arrowprops=dict(arrowstyle='->', lw=2, linestyle='--'))
    y -= 1.2
    
    # ========== KL损失计算 ==========
    kl_box = FancyBboxPatch((1, y-0.5), 10, 1.5, boxstyle="round,pad=0.1",
                            facecolor='#FF6B6B', edgecolor='black', linewidth=2)
    ax.add_patch(kl_box)
    kl_formula = r'$L_{KL} = \frac{1}{2} \sum_{i=1}^{D} (\mu_i^2 + \sigma_i^2 - \log(\sigma_i^2) - 1)$'
    kl_detail = r'Regularization: Prevents $\sigma^2$ from being too large or too small'
    ax.text(6, y+0.25, f'KL Divergence Loss:\n{kl_formula}\n{kl_detail}', 
            ha='center', va='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    # 箭头
    ax.annotate('', xy=(6, y-0.5), xytext=(6, y), arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    y -= 2
    
    # ========== 不确定性分解 ==========
    unc_box = FancyBboxPatch((1, y-0.5), 10, 1.5, boxstyle="round,pad=0.1",
                             facecolor='#FFE082', edgecolor='black', linewidth=2)
    ax.add_patch(unc_box)
    epistemic = r'$U_{epistemic} = \mathbb{E}[\sigma^2] = \frac{1}{D}\sum_{i=1}^{D} \sigma_i^2$'
    aleatoric = r'$U_{aleatoric} = \text{Var}(\mu) = \frac{1}{B}\sum_{b=1}^{B} (\mu_b - \bar{\mu})^2$'
    total = r'$U_{total} = U_{epistemic} + U_{aleatoric}$'
    ax.text(6, y+0.25, f'Uncertainty Decomposition:\nEpistemic (Model): {epistemic}\nAleatoric (Data): {aleatoric}\nTotal: {total}', 
            ha='center', va='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    # 箭头
    ax.annotate('', xy=(6, y-0.5), xytext=(6, y), arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    
    plt.tight_layout()
    return fig

def draw_detailed_adaptive_causal_graph():
    """创新点2: 自适应因果图学习 - 超详细版"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    colors = {
        'input': '#E8F4F8',
        'process': '#B3E5FC',
        'decision': '#FFE082',
        'output': '#4FC3F7',
        'formula': '#FFF9C4'
    }
    
    # 标题
    title_box = FancyBboxPatch((1, 14.5), 10, 1, boxstyle="round,pad=0.2",
                               facecolor='lightblue', edgecolor='black', linewidth=3)
    ax.add_patch(title_box)
    ax.text(6, 15, 'Innovation 2: Adaptive Causal Graph Learning\nDetailed Control Flow with Mathematical Formulas', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    y = 13
    
    # ========== 输入 ==========
    input_box = FancyBboxPatch((1, y), 10, 0.8, boxstyle="round,pad=0.1",
                               facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(6, y+0.4, 'Input: Features [z_oct, z_col, z_clin] ∈ ℝ^{B×D} each', 
            ha='center', va='center', fontsize=11, weight='bold')
    y -= 1.2
    
    # ========== Step 1: 拼接 ==========
    step1_box = FancyBboxPatch((1, y-0.3), 10, 1, boxstyle="round,pad=0.1",
                               facecolor=colors['process'], edgecolor='black', linewidth=2)
    ax.add_patch(step1_box)
    concat_formula = r'$\text{concat} = [z_{oct} \| z_{col} \| z_{clin}] \in \mathbb{R}^{B \times 3D}$'
    ax.text(6, y+0.2, f'Step 1: Concatenate Features\n{concat_formula}', 
            ha='center', va='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # 箭头
    ax.annotate('', xy=(6, y-0.3), xytext=(6, y), arrowprops=dict(arrowstyle='->', lw=2))
    y -= 1.5
    
    # ========== Step 2: 数据驱动学习 ==========
    step2_box = FancyBboxPatch((1, y-0.5), 10, 1.5, boxstyle="round,pad=0.1",
                                facecolor=colors['process'], edgecolor='red', linewidth=2)
    ax.add_patch(step2_box)
    learn_formula = r'$G_{learned} = \text{Reshape}(\text{MLP}(\text{concat})) \in \mathbb{R}^{B \times 3 \times 3}$'
    learn_detail = r'$G_{learned}[i,j,k] = \text{MLP}(\text{concat}_i)[j \cdot 3 + k]$'
    ax.text(6, y+0.25, f'Step 2: Data-Driven Learning\n{learn_formula}\n{learn_detail}', 
            ha='center', va='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # 箭头
    ax.annotate('', xy=(6, y-0.5), xytext=(6, y-0.3), arrowprops=dict(arrowstyle='->', lw=2))
    y -= 2
    
    # ========== Step 3: 先验约束 ==========
    decision1 = Polygon([(6, y), (7.5, y+0.4), (6, y+0.8), (4.5, y+0.4)], 
                        facecolor=colors['decision'], edgecolor='red', linewidth=3)
    ax.add_patch(decision1)
    ax.text(6, y+0.4, 'Use Prior\nConstraint?', ha='center', va='center', fontsize=10, weight='bold')
    
    # 先验约束分支
    prior_box = FancyBboxPatch((1, y-1.5), 4.5, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#C8E6C9', edgecolor='green', linewidth=2)
    ax.add_patch(prior_box)
    prior_formula = r'$G = G_{learned} \odot (1 - P) + P$'
    prior_detail = r'$P[i,j] = 1$: Must exist, $P[i,j] = 0$: Forbidden, $P[i,j] = -1$: Unknown'
    ax.text(3.25, y-0.75, f'Step 3: Prior Constraint\n{prior_formula}\n{prior_detail}', 
            ha='center', va='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # 无约束分支
    no_prior_box = FancyBboxPatch((6.5, y-1.5), 4.5, 1.5, boxstyle="round,pad=0.1",
                                   facecolor='#FFCCBC', edgecolor='orange', linewidth=2)
    ax.add_patch(no_prior_box)
    ax.text(8.75, y-0.75, 'No Prior Constraint\n$G = G_{learned}$', 
            ha='center', va='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # 箭头
    ax.annotate('', xy=(4.5, y-0.75), xytext=(6, y+0.4), arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.annotate('', xy=(6.5, y-0.75), xytext=(6, y+0.4), arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    y -= 2.5
    
    # ========== Step 4: DAG约束 ==========
    decision2 = Polygon([(6, y), (7.5, y+0.4), (6, y+0.8), (4.5, y+0.4)], 
                        facecolor=colors['decision'], edgecolor='red', linewidth=3)
    ax.add_patch(decision2)
    ax.text(6, y+0.4, 'Enforce\nDAG?', ha='center', va='center', fontsize=10, weight='bold')
    
    # DAG方法1: 上三角
    dag1_box = FancyBboxPatch((1, y-1.5), 4.5, 1.5, boxstyle="round,pad=0.1",
                              facecolor='#C8E6C9', edgecolor='green', linewidth=2)
    ax.add_patch(dag1_box)
    dag1_formula = r'$G[i,j] = 0$ if $i > j$ (Upper Triangular)'
    ax.text(3.25, y-0.75, f'Method 1: Upper Triangular\n{dag1_formula}', 
            ha='center', va='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # DAG方法2: 矩阵指数
    dag2_box = FancyBboxPatch((6.5, y-1.5), 4.5, 1.5, boxstyle="round,pad=0.1",
                              facecolor='#FFCCBC', edgecolor='orange', linewidth=2)
    ax.add_patch(dag2_box)
    dag2_formula = r'$L_{DAG} = \|\text{tr}(\exp(G)) - 3\|^2$'
    dag2_detail = r'Differentiable DAG constraint'
    ax.text(8.75, y-0.75, f'Method 2: Matrix Exponential\n{dag2_formula}\n{dag2_detail}', 
            ha='center', va='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # 箭头
    ax.annotate('', xy=(4.5, y-0.75), xytext=(6, y+0.4), arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.annotate('', xy=(6.5, y-0.75), xytext=(6, y+0.4), arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    y -= 2.5
    
    # ========== Step 5: 归一化 ==========
    norm_box = FancyBboxPatch((1, y-0.3), 10, 1, boxstyle="round,pad=0.1",
                              facecolor=colors['process'], edgecolor='black', linewidth=2)
    ax.add_patch(norm_box)
    norm_formula = r'$G = \text{Sigmoid}(G) = \frac{1}{1 + \exp(-G)}$'
    norm_detail = r'Normalize causal strength to [0, 1]'
    ax.text(6, y+0.2, f'Step 5: Normalization\n{norm_formula}\n{norm_detail}', 
            ha='center', va='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # 箭头
    ax.annotate('', xy=(6, y-0.3), xytext=(6, y-1.5), arrowprops=dict(arrowstyle='->', lw=2))
    y -= 1.5
    
    # ========== 输出 ==========
    output_box = FancyBboxPatch((1, y-0.3), 10, 1, boxstyle="round,pad=0.1",
                                facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    output_formula = r'$G \in \mathbb{R}^{B \times 3 \times 3}, \quad G[i,j,k] = \text{causal strength from } j \text{ to } k$'
    ax.text(6, y+0.2, f'Output: Causal Graph\n{output_formula}', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # 箭头
    ax.annotate('', xy=(6, y-0.3), xytext=(6, y-0.3), arrowprops=dict(arrowstyle='->', lw=2))
    
    plt.tight_layout()
    return fig

def draw_detailed_causal_intervention():
    """创新点3: 因果干预 - 超详细版"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    colors = {
        'input': '#E8F4F8',
        'process': '#B3E5FC',
        'decision': '#FFE082',
        'output': '#4FC3F7',
        'formula': '#FFF9C4'
    }
    
    # 标题
    title_box = FancyBboxPatch((1, 14.5), 10, 1, boxstyle="round,pad=0.2",
                               facecolor='lightblue', edgecolor='black', linewidth=3)
    ax.add_patch(title_box)
    ax.text(6, 15, 'Innovation 3: Causal Intervention (Do-Calculus)\nDetailed Control Flow with Mathematical Formulas', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    y = 13
    
    # ========== 输入 ==========
    input_box = FancyBboxPatch((1, y), 10, 0.8, boxstyle="round,pad=0.1",
                               facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(6, y+0.4, 'Input: Features [z_oct, z_col, z_clin] + Causal Graph G + Variances [σ²_oct, σ²_col, σ²_clin]', 
            ha='center', va='center', fontsize=10, weight='bold')
    y -= 1.2
    
    # ========== Step 1: 不确定性计算 ==========
    unc_box = FancyBboxPatch((1, y-0.3), 10, 1, boxstyle="round,pad=0.1",
                             facecolor=colors['process'], edgecolor='black', linewidth=2)
    ax.add_patch(unc_box)
    unc_formula = r'$U = \frac{1}{3D} \sum_{m=1}^{3} \sum_{d=1}^{D} \sigma_{m,d}^2$'
    unc_detail = r'Average uncertainty across all modalities and dimensions'
    ax.text(6, y+0.2, f'Step 1: Compute Uncertainty\n{unc_formula}\n{unc_detail}', 
            ha='center', va='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # 箭头
    ax.annotate('', xy=(6, y-0.3), xytext=(6, y), arrowprops=dict(arrowstyle='->', lw=2))
    y -= 1.5
    
    # ========== Step 2: 干预强度调度 ==========
    sched_box = FancyBboxPatch((1, y-0.5), 10, 1.5, boxstyle="round,pad=0.1",
                                facecolor=colors['process'], edgecolor='red', linewidth=2)
    ax.add_patch(sched_box)
    sched_formula = r'$\alpha = \text{Scheduler}(U) = \text{Sigmoid}(W \cdot U + b)$'
    sched_detail = r'$\alpha \in [0,1]$: High uncertainty → High intervention strength'
    ax.text(6, y+0.25, f'Step 2: Intervention Strength Scheduling\n{sched_formula}\n{sched_detail}', 
            ha='center', va='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # 箭头
    ax.annotate('', xy=(6, y-0.5), xytext=(6, y-0.3), arrowprops=dict(arrowstyle='->', lw=2))
    y -= 2
    
    # ========== Step 3: Do操作 ==========
    do_box = FancyBboxPatch((1, y-0.5), 10, 1.5, boxstyle="round,pad=0.1",
                            facecolor=colors['process'], edgecolor='red', linewidth=2)
    ax.add_patch(do_box)
    do_formula = r'$P(Y|do(X))$ vs $P(Y|X)$'
    do_detail = r'Cut all incoming edges to X in causal graph G'
    ax.text(6, y+0.25, f'Step 3: Do-Operation\n{do_formula}\n{do_detail}', 
            ha='center', va='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # 箭头
    ax.annotate('', xy=(6, y-0.5), xytext=(6, y-0.5), arrowprops=dict(arrowstyle='->', lw=2))
    y -= 2
    
    # ========== 干预方法选择 ==========
    decision1 = Polygon([(6, y), (7.5, y+0.4), (6, y+0.8), (4.5, y+0.4)], 
                        facecolor=colors['decision'], edgecolor='red', linewidth=3)
    ax.add_patch(decision1)
    ax.text(6, y+0.4, 'Intervention\nMethod?', ha='center', va='center', fontsize=10, weight='bold')
    
    # 方法1: 掩码
    method1_box = FancyBboxPatch((1, y-1.5), 4.5, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#C8E6C9', edgecolor='green', linewidth=2)
    ax.add_patch(method1_box)
    mask_formula = r'$\text{mask} = 1 - G[:, \text{target}]$'
    mask_detail = r'$z_{interv} = z \odot \text{mask}$ (element-wise)'
    ax.text(3.25, y-0.75, f'Method 1: Masking\n{mask_formula}\n{mask_detail}', 
            ha='center', va='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # 方法2: 替换
    method2_box = FancyBboxPatch((6.5, y-1.5), 4.5, 1.5, boxstyle="round,pad=0.1",
                                  facecolor='#FFCCBC', edgecolor='orange', linewidth=2)
    ax.add_patch(method2_box)
    replace_formula = r'$z_{interv} = z + \alpha \cdot (G \cdot z - z)$'
    replace_detail = r'Adjust features by causal graph'
    ax.text(8.75, y-0.75, f'Method 2: Replacement\n{replace_formula}\n{replace_detail}', 
            ha='center', va='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # 箭头
    ax.annotate('', xy=(4.5, y-0.75), xytext=(6, y+0.4), arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.annotate('', xy=(6.5, y-0.75), xytext=(6, y+0.4), arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    y -= 2.5
    
    # ========== Step 4: 因果效应 ==========
    effect_box = FancyBboxPatch((1, y-0.5), 10, 1.5, boxstyle="round,pad=0.1",
                                facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(effect_box)
    effect_formula = r'$\Delta = \mathbb{E}[Y|do(X)] - \mathbb{E}[Y]$'
    effect_detail = r'Causal effect: Quantify contribution of intervention'
    ax.text(6, y+0.25, f'Step 4: Causal Effect Calculation\n{effect_formula}\n{effect_detail}', 
            ha='center', va='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # 箭头
    ax.annotate('', xy=(6, y-0.5), xytext=(3.25, y-1.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(6, y-0.5), xytext=(8.75, y-1.5), arrowprops=dict(arrowstyle='->', lw=2))
    y -= 2
    
    # ========== 输出 ==========
    output_box = FancyBboxPatch((1, y-0.3), 10, 1, boxstyle="round,pad=0.1",
                                facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(6, y+0.2, 'Output: Intervened Features [z_oct_interv, z_col_interv, z_clin_interv] + Causal Effects Δ', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # 箭头
    ax.annotate('', xy=(6, y-0.3), xytext=(6, y-0.5), arrowprops=dict(arrowstyle='->', lw=2))
    
    plt.tight_layout()
    return fig

def draw_detailed_hierarchical_contrastive():
    """创新点4: 分层对比学习 - 超详细版"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    colors = {
        'input': '#E8F4F8',
        'process': '#B3E5FC',
        'decision': '#FFE082',
        'output': '#4FC3F7',
        'formula': '#FFF9C4'
    }
    
    # 标题
    title_box = FancyBboxPatch((1, 14.5), 10, 1, boxstyle="round,pad=0.2",
                               facecolor='lightblue', edgecolor='black', linewidth=3)
    ax.add_patch(title_box)
    ax.text(6, 15, 'Innovation 4: Hierarchical Contrastive Learning (CLIP)\nDetailed Control Flow with Mathematical Formulas', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    y = 13
    
    # ========== 输入 ==========
    input_box = FancyBboxPatch((1, y), 10, 0.8, boxstyle="round,pad=0.1",
                               facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(6, y+0.4, 'Input: Features [z_oct, z_col, z_clin] ∈ ℝ^{B×D} + Labels y ∈ {0,1}^B', 
            ha='center', va='center', fontsize=10, weight='bold')
    y -= 1.2
    
    # ========== Level 1: 特征级 ==========
    level1_box = FancyBboxPatch((1, y-0.8), 10, 2, boxstyle="round,pad=0.1",
                                facecolor='#C8E6C9', edgecolor='green', linewidth=2)
    ax.add_patch(level1_box)
    level1_title = 'Level 1: Feature-Level Contrastive Learning'
    level1_formula1 = r'$L_{feat}^{oct-clin} = \text{InfoNCE}(z_{oct}, z_{clin})$'
    level1_formula2 = r'$L_{feat}^{col-clin} = \text{InfoNCE}(z_{col}, z_{clin})$'
    level1_formula3 = r'$L_{feat}^{oct-col} = \text{InfoNCE}(z_{oct}, z_{col})$'
    level1_total = r'$L_{feat} = \frac{1}{3}(L_{feat}^{oct-clin} + L_{feat}^{col-clin} + L_{feat}^{oct-col})$'
    ax.text(6, y+0.2, f'{level1_title}\n{level1_formula1}\n{level1_formula2}\n{level1_formula3}\n{level1_total}', 
            ha='center', va='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # 箭头
    ax.annotate('', xy=(6, y-0.8), xytext=(6, y), arrowprops=dict(arrowstyle='->', lw=2))
    y -= 2.5
    
    # ========== Level 2: 语义级 ==========
    level2_box = FancyBboxPatch((1, y-0.8), 10, 2, boxstyle="round,pad=0.1",
                                facecolor='#81D4FA', edgecolor='blue', linewidth=2)
    ax.add_patch(level2_box)
    level2_title = 'Level 2: Semantic-Level Contrastive Learning'
    level2_proj = r'$s_m = \text{MLP}_m(z_m) = W_2 \cdot \text{GELU}(\text{LayerNorm}(W_1 \cdot z_m))$'
    level2_formula1 = r'$L_{sem}^{oct-clin} = \text{InfoNCE}(s_{oct}, s_{clin})$'
    level2_formula2 = r'$L_{sem}^{col-clin} = \text{InfoNCE}(s_{col}, s_{clin})$'
    level2_total = r'$L_{sem} = \frac{1}{2}(L_{sem}^{oct-clin} + L_{sem}^{col-clin})$'
    ax.text(6, y+0.2, f'{level2_title}\nProjection: {level2_proj}\n{level2_formula1}\n{level2_formula2}\n{level2_total}', 
            ha='center', va='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # 箭头
    ax.annotate('', xy=(6, y-0.8), xytext=(6, y-0.8), arrowprops=dict(arrowstyle='->', lw=2))
    y -= 2.5
    
    # ========== Level 3: 决策级 ==========
    decision1 = Polygon([(6, y), (7.5, y+0.4), (6, y+0.8), (4.5, y+0.4)], 
                        facecolor=colors['decision'], edgecolor='red', linewidth=3)
    ax.add_patch(decision1)
    ax.text(6, y+0.4, 'Labels\nAvailable?', ha='center', va='center', fontsize=10, weight='bold')
    
    # 有标签分支
    level3_box = FancyBboxPatch((1, y-1.5), 4.5, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#FFCCBC', edgecolor='orange', linewidth=2)
    ax.add_patch(level3_box)
    level3_title = 'Level 3: Decision-Level'
    level3_proj = r'$p_m = \text{Classifier}(z_m)$'
    level3_formula = r'$L_{dec} = \text{InfoNCE}(p_{y=0}, p_{y=1}, \text{negative=True})$'
    level3_detail = r'Push different classes apart'
    ax.text(3.25, y-0.75, f'{level3_title}\n{level3_proj}\n{level3_formula}\n{level3_detail}', 
            ha='center', va='center', fontsize=9, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # 无标签分支
    no_label_box = FancyBboxPatch((6.5, y-1.5), 4.5, 1.5, boxstyle="round,pad=0.1",
                                   facecolor='#E0E0E0', edgecolor='gray', linewidth=2)
    ax.add_patch(no_label_box)
    ax.text(8.75, y-0.75, 'No Labels\n$L_{dec} = 0$', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # 箭头
    ax.annotate('', xy=(4.5, y-0.75), xytext=(6, y+0.4), arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    ax.annotate('', xy=(6.5, y-0.75), xytext=(6, y+0.4), arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    y -= 2.5
    
    # ========== InfoNCE公式详解 ==========
    infonce_box = FancyBboxPatch((1, y-0.8), 10, 2, boxstyle="round,pad=0.1",
                                  facecolor='#FFF9C4', edgecolor='black', linewidth=2)
    ax.add_patch(infonce_box)
    infonce_title = 'InfoNCE Loss Formula (Core of CLIP):'
    infonce_sim = r'$\text{sim}(z_i, z_j) = \frac{z_i^T \cdot z_j}{\|z_i\| \|z_j\|}$ (cosine similarity)'
    infonce_temp = r'$\text{sim}_{temp} = \frac{\text{sim}(z_i, z_j)}{\tau}$ (temperature scaling, $\tau=0.07$)'
    infonce_loss = r'$L_{InfoNCE} = -\log \frac{\exp(\text{sim}_{temp}(z_i, z_j^+))}{\sum_{k=1}^{B} \exp(\text{sim}_{temp}(z_i, z_k))}$'
    infonce_detail = r'Positive: diagonal (matching pairs), Negative: off-diagonal (non-matching pairs)'
    ax.text(6, y+0.2, f'{infonce_title}\n{infonce_sim}\n{infonce_temp}\n{infonce_loss}\n{infonce_detail}', 
            ha='center', va='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    # 箭头
    ax.annotate('', xy=(6, y-0.8), xytext=(3.25, y-1.5), arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    ax.annotate('', xy=(6, y-0.8), xytext=(6, y-0.8), arrowprops=dict(arrowstyle='->', lw=2))
    y -= 2.5
    
    # ========== 加权组合 ==========
    weight_box = FancyBboxPatch((1, y-0.5), 10, 1.5, boxstyle="round,pad=0.1",
                                facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(weight_box)
    weight_formula = r'$w = \text{Softmax}([w_1, w_2, w_3])$ (learned adaptively)'
    weight_total = r'$L_{CLIP} = w_1 \cdot L_{feat} + w_2 \cdot L_{sem} + w_3 \cdot L_{dec}$'
    ax.text(6, y+0.25, f'Weighted Combination:\n{weight_formula}\n{weight_total}', 
            ha='center', va='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['formula'], alpha=0.7))
    
    # 箭头
    ax.annotate('', xy=(6, y-0.5), xytext=(6, y-0.8), arrowprops=dict(arrowstyle='->', lw=2))
    y -= 2
    
    # ========== 输出 ==========
    output_box = FancyBboxPatch((1, y-0.3), 10, 1, boxstyle="round,pad=0.1",
                                facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(6, y+0.2, 'Output: Total Contrastive Loss L_CLIP\nBenefits: Multi-level alignment, Better feature representation', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # 箭头
    ax.annotate('', xy=(6, y-0.3), xytext=(6, y-0.5), arrowprops=dict(arrowstyle='->', lw=2))
    
    plt.tight_layout()
    return fig

def main():
    """主函数：生成所有四个创新点的超详细流程图"""
    print("🎨 正在生成四个创新点的超详细流程图（含公式）...")
    
    # 创新点1
    print("  📊 生成创新点1: 贝叶斯不确定性量化（详细版）...")
    fig1 = draw_detailed_bayesian_uncertainty()
    fig1.savefig('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/innovation1_detailed_bayesian.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    fig1.savefig('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/innovation1_detailed_bayesian.pdf', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    print("    ✅ 已保存")
    
    # 创新点2
    print("  📊 生成创新点2: 自适应因果图学习（详细版）...")
    fig2 = draw_detailed_adaptive_causal_graph()
    fig2.savefig('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/innovation2_detailed_causal_graph.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    fig2.savefig('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/innovation2_detailed_causal_graph.pdf', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print("    ✅ 已保存")
    
    # 创新点3
    print("  📊 生成创新点3: 因果干预（详细版）...")
    fig3 = draw_detailed_causal_intervention()
    fig3.savefig('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/innovation3_detailed_intervention.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    fig3.savefig('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/innovation3_detailed_intervention.pdf', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig3)
    print("    ✅ 已保存")
    
    # 创新点4
    print("  📊 生成创新点4: 分层对比学习（详细版）...")
    fig4 = draw_detailed_hierarchical_contrastive()
    fig4.savefig('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/innovation4_detailed_contrastive.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    fig4.savefig('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/innovation4_detailed_contrastive.pdf', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig4)
    print("    ✅ 已保存")
    
    print("\n🎉 所有创新点超详细流程图（含公式）已生成完成！")
    print("\n生成的文件（详细版）：")
    print("  1. innovation1_detailed_bayesian.png/pdf")
    print("  2. innovation2_detailed_causal_graph.png/pdf")
    print("  3. innovation3_detailed_intervention.png/pdf")
    print("  4. innovation4_detailed_contrastive.png/pdf")

if __name__ == '__main__':
    main()

