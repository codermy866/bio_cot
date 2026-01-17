#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制四个创新点的详细流程图
展示每个模块的内部工作机制
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch, Circle, Rectangle
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def draw_bayesian_uncertainty_flow():
    """创新点1: 贝叶斯不确定性量化流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    colors = {
        'input': '#E8F4F8',
        'mean': '#B3E5FC',
        'var': '#81D4FA',
        'sample': '#4FC3F7',
        'output': '#29B6F6',
        'loss': '#FF6B6B'
    }
    
    # 标题
    ax.text(5, 11.5, 'Innovation 1: Bayesian Uncertainty Quantification', 
            ha='center', va='center', fontsize=16, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    # 输入
    input_box = FancyBboxPatch((1, 9.5), 2, 0.8, boxstyle="round,pad=0.1",
                               facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 9.9, 'Input Feature\nf ∈ R^{B×D}', ha='center', va='center', fontsize=10, weight='bold')
    
    # 均值编码器
    mean_box = FancyBboxPatch((4.5, 9.5), 2, 0.8, boxstyle="round,pad=0.1",
                              facecolor=colors['mean'], edgecolor='black', linewidth=2)
    ax.add_patch(mean_box)
    ax.text(5.5, 9.9, 'Mean Encoder\nμ = MLP(f)', ha='center', va='center', fontsize=9, weight='bold')
    
    # 方差编码器
    var_box = FancyBboxPatch((7.5, 9.5), 2, 0.8, boxstyle="round,pad=0.1",
                             facecolor=colors['var'], edgecolor='black', linewidth=2)
    ax.add_patch(var_box)
    ax.text(8.5, 9.9, 'Variance Encoder\nσ² = Softplus(MLP(f))', ha='center', va='center', fontsize=9, weight='bold')
    
    # 箭头
    ax.annotate('', xy=(4.5, 9.9), xytext=(3, 9.9), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(7.5, 9.9), xytext=(6.5, 9.9), arrowprops=dict(arrowstyle='->', lw=2))
    
    # 采样分支
    y_pos = 7.5
    # 训练时采样
    train_box = FancyBboxPatch((1, y_pos), 3.5, 1.2, boxstyle="round,pad=0.1",
                               facecolor=colors['sample'], edgecolor='red', linewidth=2)
    ax.add_patch(train_box)
    ax.text(2.75, y_pos + 0.6, 'Training: Sampling\nε ~ N(0,1)\nz = μ + ε·√σ²', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # 推理时用均值
    infer_box = FancyBboxPatch((5.5, y_pos), 3.5, 1.2, boxstyle="round,pad=0.1",
                               facecolor=colors['sample'], edgecolor='green', linewidth=2)
    ax.add_patch(infer_box)
    ax.text(7.25, y_pos + 0.6, 'Inference: Mean\nz = μ', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # 从均值/方差到采样
    ax.annotate('', xy=(2.75, y_pos + 1.2), xytext=(5.5, 9.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(7.25, y_pos + 1.2), xytext=(5.5, 9.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(2.75, y_pos + 1.2), xytext=(8.5, 9.5), arrowprops=dict(arrowstyle='->', lw=2))
    
    # 输出
    output_box = FancyBboxPatch((3.5, 5.5), 3, 0.8, boxstyle="round,pad=0.1",
                                facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 5.9, 'Sampled Feature\nz ∈ R^{B×D}', ha='center', va='center', fontsize=10, weight='bold')
    
    # 从采样到输出
    ax.annotate('', xy=(5, 5.5), xytext=(2.75, y_pos), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(5, 5.5), xytext=(7.25, y_pos), arrowprops=dict(arrowstyle='->', lw=2, linestyle='--'))
    
    # KL损失
    kl_box = FancyBboxPatch((1, 3.5), 8, 1, boxstyle="round,pad=0.1",
                            facecolor=colors['loss'], edgecolor='black', linewidth=2)
    ax.add_patch(kl_box)
    ax.text(5, 4, 'KL Divergence Loss\nL_KL = 0.5·Σ(μ² + σ² - log(σ²) - 1)', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # 从均值/方差到KL损失
    ax.annotate('', xy=(5, 3.5), xytext=(5.5, 9.5), arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.annotate('', xy=(5, 3.5), xytext=(8.5, 9.5), arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # 不确定性分解
    unc_box = FancyBboxPatch((1, 1.5), 8, 1, boxstyle="round,pad=0.1",
                             facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(unc_box)
    ax.text(5, 2, 'Uncertainty Decomposition:\nEpistemic (Model) = mean(σ²)  |  Aleatoric (Data) = var(μ)', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # 从方差到不确定性
    ax.annotate('', xy=(5, 1.5), xytext=(8.5, 9.5), arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    
    plt.tight_layout()
    return fig

def draw_adaptive_causal_graph_flow():
    """创新点2: 自适应因果图学习流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    colors = {
        'input': '#E8F4F8',
        'learn': '#B3E5FC',
        'prior': '#81D4FA',
        'dag': '#4FC3F7',
        'output': '#29B6F6'
    }
    
    # 标题
    ax.text(5, 11.5, 'Innovation 2: Adaptive Causal Graph Learning', 
            ha='center', va='center', fontsize=16, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    # 输入特征
    input_box = FancyBboxPatch((1, 9.5), 8, 0.8, boxstyle="round,pad=0.1",
                               facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 9.9, 'Input Features: [z_oct, z_col, z_clin] ∈ R^{B×D}', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Step 1: 拼接
    concat_box = FancyBboxPatch((1, 8), 3, 0.8, boxstyle="round,pad=0.1",
                                facecolor=colors['learn'], edgecolor='black', linewidth=2)
    ax.add_patch(concat_box)
    ax.text(2.5, 8.4, '1. Concatenate\n[B, 3D]', ha='center', va='center', fontsize=9, weight='bold')
    
    # Step 2: 数据驱动学习
    learn_box = FancyBboxPatch((5, 8), 4, 0.8, boxstyle="round,pad=0.1",
                               facecolor=colors['learn'], edgecolor='black', linewidth=2)
    ax.add_patch(learn_box)
    ax.text(7, 8.4, '2. Data-Driven Learning\nG_learned = MLP(concat)', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # 箭头
    ax.annotate('', xy=(5, 8.4), xytext=(4, 8.4), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(1, 8), xytext=(5, 9.5), arrowprops=dict(arrowstyle='->', lw=2))
    
    # Step 3: 先验约束
    prior_box = FancyBboxPatch((1, 6.5), 4, 0.8, boxstyle="round,pad=0.1",
                               facecolor=colors['prior'], edgecolor='red', linewidth=2)
    ax.add_patch(prior_box)
    ax.text(3, 6.9, '3. Prior Constraint\nG = G_learned ⊙ (1-P) + P', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # 先验知识说明
    prior_note = FancyBboxPatch((6, 6.5), 3, 0.8, boxstyle="round,pad=0.1",
                                facecolor='lightyellow', edgecolor='black', linewidth=1)
    ax.add_patch(prior_note)
    ax.text(7.5, 6.9, 'Medical Prior:\nP[i,j]=1: Must exist\nP[i,j]=0: Forbidden', 
            ha='center', va='center', fontsize=8)
    
    # 箭头
    ax.annotate('', xy=(1, 6.9), xytext=(7, 8.4), arrowprops=dict(arrowstyle='->', lw=2))
    
    # Step 4: DAG约束
    dag_box = FancyBboxPatch((1, 5), 4, 0.8, boxstyle="round,pad=0.1",
                             facecolor=colors['dag'], edgecolor='red', linewidth=2)
    ax.add_patch(dag_box)
    ax.text(3, 5.4, '4. DAG Constraint\nEnforce Acyclic', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # DAG说明
    dag_note = FancyBboxPatch((6, 5), 3, 0.8, boxstyle="round,pad=0.1",
                              facecolor='lightyellow', edgecolor='black', linewidth=1)
    ax.add_patch(dag_note)
    ax.text(7.5, 5.4, 'Method:\nUpper Triangular\nor Exp Trace', 
            ha='center', va='center', fontsize=8)
    
    # 箭头
    ax.annotate('', xy=(1, 5.4), xytext=(3, 6.5), arrowprops=dict(arrowstyle='->', lw=2))
    
    # Step 5: 二值化/归一化
    norm_box = FancyBboxPatch((1, 3.5), 4, 0.8, boxstyle="round,pad=0.1",
                              facecolor=colors['dag'], edgecolor='black', linewidth=2)
    ax.add_patch(norm_box)
    ax.text(3, 3.9, '5. Normalization\nG = Sigmoid(G)', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # 箭头
    ax.annotate('', xy=(1, 3.9), xytext=(3, 5), arrowprops=dict(arrowstyle='->', lw=2))
    
    # 输出
    output_box = FancyBboxPatch((1, 2), 8, 0.8, boxstyle="round,pad=0.1",
                                facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 2.4, 'Output: Causal Graph G ∈ R^{B×3×3}\nG[i,j] = causal strength from i to j', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # 箭头
    ax.annotate('', xy=(5, 2), xytext=(3, 3.5), arrowprops=dict(arrowstyle='->', lw=2))
    
    # 个性化因果图（可选）
    pers_box = FancyBboxPatch((6, 3.5), 3, 1.5, boxstyle="round,pad=0.1",
                              facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(pers_box)
    ax.text(7.5, 4.25, 'Optional:\nPersonalized\nCausal Graph\n(Per Patient)', 
            ha='center', va='center', fontsize=8, weight='bold')
    
    plt.tight_layout()
    return fig

def draw_causal_intervention_flow():
    """创新点3: 因果干预流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    colors = {
        'input': '#E8F4F8',
        'graph': '#B3E5FC',
        'uncertainty': '#81D4FA',
        'intervention': '#4FC3F7',
        'output': '#29B6F6'
    }
    
    # 标题
    ax.text(5, 11.5, 'Innovation 3: Causal Intervention (Do-Calculus)', 
            ha='center', va='center', fontsize=16, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    # 输入
    input_box = FancyBboxPatch((1, 9.5), 8, 0.8, boxstyle="round,pad=0.1",
                               facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 9.9, 'Input: Features [z_oct, z_col, z_clin] + Causal Graph G', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Step 1: 不确定性计算
    unc_box = FancyBboxPatch((1, 8), 3.5, 1, boxstyle="round,pad=0.1",
                             facecolor=colors['uncertainty'], edgecolor='black', linewidth=2)
    ax.add_patch(unc_box)
    ax.text(2.75, 8.5, '1. Compute Uncertainty\nU = mean(σ²)\nHigh U → More intervention', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Step 2: 干预强度调度
    sched_box = FancyBboxPatch((5.5, 8), 3.5, 1, boxstyle="round,pad=0.1",
                                facecolor=colors['uncertainty'], edgecolor='black', linewidth=2)
    ax.add_patch(sched_box)
    ax.text(7.25, 8.5, '2. Intervention Strength\nα = Scheduler(U)\nα ∈ [0,1]', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # 箭头
    ax.annotate('', xy=(5.5, 8.5), xytext=(4.5, 8.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(1, 8.5), xytext=(5, 9.5), arrowprops=dict(arrowstyle='->', lw=2))
    
    # Step 3: Do操作（切断边）
    do_box = FancyBboxPatch((1, 6.5), 8, 1, boxstyle="round,pad=0.1",
                            facecolor=colors['intervention'], edgecolor='red', linewidth=2)
    ax.add_patch(do_box)
    ax.text(5, 7, '3. Do-Operation: Do(X = x)\nCut incoming edges to X\nRemove confounding paths', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # 箭头
    ax.annotate('', xy=(5, 6.5), xytext=(2.75, 8), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(5, 6.5), xytext=(7.25, 8), arrowprops=dict(arrowstyle='->', lw=2))
    
    # 干预方法
    method_box = FancyBboxPatch((1, 4.5), 3.5, 1.5, boxstyle="round,pad=0.1",
                                facecolor='lightyellow', edgecolor='black', linewidth=1)
    ax.add_patch(method_box)
    ax.text(2.75, 5.25, 'Method 1: Masking\nz_interv = z ⊙ mask\nmask = 1 - G[:,target]', 
            ha='center', va='center', fontsize=8)
    
    method2_box = FancyBboxPatch((5.5, 4.5), 3.5, 1.5, boxstyle="round,pad=0.1",
                                  facecolor='lightyellow', edgecolor='black', linewidth=1)
    ax.add_patch(method2_box)
    ax.text(7.25, 5.25, 'Method 2: Replacement\nz_interv = z + α·(G@z - z)\nAdjust by causal graph', 
            ha='center', va='center', fontsize=8)
    
    # 箭头
    ax.annotate('', xy=(2.75, 4.5), xytext=(3, 6.5), arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    ax.annotate('', xy=(7.25, 4.5), xytext=(7, 6.5), arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    
    # Step 4: 因果效应计算
    effect_box = FancyBboxPatch((1, 2.5), 8, 1, boxstyle="round,pad=0.1",
                                facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(effect_box)
    ax.text(5, 3, '4. Causal Effect\nΔ = E[Y|do(X)] - E[Y]\nQuantify causal contribution', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # 箭头
    ax.annotate('', xy=(5, 2.5), xytext=(2.75, 4.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(5, 2.5), xytext=(7.25, 4.5), arrowprops=dict(arrowstyle='->', lw=2))
    
    # 输出
    output_box = FancyBboxPatch((1, 0.5), 8, 1, boxstyle="round,pad=0.1",
                                facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 1, 'Output: Intervened Features [z_oct_interv, z_col_interv, z_clin_interv]\n+ Causal Effects Δ', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # 箭头
    ax.annotate('', xy=(5, 0.5), xytext=(5, 2.5), arrowprops=dict(arrowstyle='->', lw=2))
    
    plt.tight_layout()
    return fig

def draw_hierarchical_contrastive_flow():
    """创新点4: 分层对比学习流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    colors = {
        'input': '#E8F4F8',
        'feature': '#B3E5FC',
        'semantic': '#81D4FA',
        'decision': '#4FC3F7',
        'output': '#29B6F6'
    }
    
    # 标题
    ax.text(5, 11.5, 'Innovation 4: Hierarchical Contrastive Learning (CLIP)', 
            ha='center', va='center', fontsize=16, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    # 输入
    input_box = FancyBboxPatch((1, 9.5), 8, 0.8, boxstyle="round,pad=0.1",
                               facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 9.9, 'Input: Features [z_oct, z_col, z_clin] + Labels y', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Level 1: 特征级对比
    feat_box = FancyBboxPatch((1, 7.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                              facecolor=colors['feature'], edgecolor='red', linewidth=2)
    ax.add_patch(feat_box)
    ax.text(2.25, 8.25, 'Level 1:\nFeature Level\nRaw Features', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    feat_detail = FancyBboxPatch((1, 6.5), 2.5, 0.8, boxstyle="round,pad=0.05",
                                 facecolor='white', edgecolor='gray', linewidth=1)
    ax.add_patch(feat_detail)
    ax.text(2.25, 6.9, 'InfoNCE(z_oct, z_clin)\nInfoNCE(z_col, z_clin)\nInfoNCE(z_oct, z_col)', 
            ha='center', va='center', fontsize=7)
    
    # Level 2: 语义级对比
    sem_box = FancyBboxPatch((4, 7.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                             facecolor=colors['semantic'], edgecolor='red', linewidth=2)
    ax.add_patch(sem_box)
    ax.text(5.25, 8.25, 'Level 2:\nSemantic Level\nHigh-level Semantics', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    sem_detail = FancyBboxPatch((4, 6.5), 2.5, 0.8, boxstyle="round,pad=0.05",
                                facecolor='white', edgecolor='gray', linewidth=1)
    ax.add_patch(sem_detail)
    ax.text(5.25, 6.9, 's_oct = MLP(z_oct)\ns_col = MLP(z_col)\ns_clin = MLP(z_clin)\nInfoNCE(s_oct, s_clin)...', 
            ha='center', va='center', fontsize=7)
    
    # Level 3: 决策级对比
    dec_box = FancyBboxPatch((7, 7.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                             facecolor=colors['decision'], edgecolor='red', linewidth=2)
    ax.add_patch(dec_box)
    ax.text(8.25, 8.25, 'Level 3:\nDecision Level\nPredictions', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    dec_detail = FancyBboxPatch((7, 6.5), 2.5, 0.8, boxstyle="round,pad=0.05",
                                facecolor='white', edgecolor='gray', linewidth=1)
    ax.add_patch(dec_detail)
    ax.text(8.25, 6.9, 'p_oct = Classifier(z_oct)\np_col = Classifier(z_col)\nContrast by labels', 
            ha='center', va='center', fontsize=7)
    
    # 箭头
    ax.annotate('', xy=(2.25, 7.5), xytext=(5, 9.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(5.25, 7.5), xytext=(5, 9.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(8.25, 7.5), xytext=(5, 9.5), arrowprops=dict(arrowstyle='->', lw=2))
    
    # InfoNCE公式
    infonce_box = FancyBboxPatch((1, 4.5), 8, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(infonce_box)
    ax.text(5, 5.25, 'InfoNCE Loss Formula:\nL = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))\n' +
            'where sim(z_i, z_j) = z_i^T · z_j / ||z_i|| ||z_j|| (cosine similarity)', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # 箭头
    ax.annotate('', xy=(5, 4.5), xytext=(2.25, 6.5), arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    ax.annotate('', xy=(5, 4.5), xytext=(5.25, 6.5), arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    ax.annotate('', xy=(5, 4.5), xytext=(8.25, 6.5), arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    
    # 加权组合
    weight_box = FancyBboxPatch((1, 2.5), 8, 1, boxstyle="round,pad=0.1",
                                facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(weight_box)
    ax.text(5, 3, 'Weighted Combination:\nL_total = w1·L_feature + w2·L_semantic + w3·L_decision\n' +
            'weights learned adaptively', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # 箭头
    ax.annotate('', xy=(5, 2.5), xytext=(5, 4.5), arrowprops=dict(arrowstyle='->', lw=2))
    
    # 输出
    output_box = FancyBboxPatch((1, 0.5), 8, 1.5, boxstyle="round,pad=0.1",
                                facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 1.25, 'Output: Total Contrastive Loss L_CLIP\n' +
            'Benefits: Multi-level alignment, Better feature representation, Robust to noise', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # 箭头
    ax.annotate('', xy=(5, 0.5), xytext=(5, 2.5), arrowprops=dict(arrowstyle='->', lw=2))
    
    plt.tight_layout()
    return fig

def main():
    """主函数：生成所有四个创新点的详细流程图"""
    print("🎨 正在生成四个创新点的详细流程图...")
    
    # 创新点1: 贝叶斯不确定性量化
    print("  📊 生成创新点1: 贝叶斯不确定性量化...")
    fig1 = draw_bayesian_uncertainty_flow()
    fig1.savefig('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/innovation1_bayesian_uncertainty.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    fig1.savefig('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/innovation1_bayesian_uncertainty.pdf', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    print("    ✅ 已保存")
    
    # 创新点2: 自适应因果图学习
    print("  📊 生成创新点2: 自适应因果图学习...")
    fig2 = draw_adaptive_causal_graph_flow()
    fig2.savefig('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/innovation2_adaptive_causal_graph.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    fig2.savefig('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/innovation2_adaptive_causal_graph.pdf', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print("    ✅ 已保存")
    
    # 创新点3: 因果干预
    print("  📊 生成创新点3: 因果干预...")
    fig3 = draw_causal_intervention_flow()
    fig3.savefig('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/innovation3_causal_intervention.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    fig3.savefig('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/innovation3_causal_intervention.pdf', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig3)
    print("    ✅ 已保存")
    
    # 创新点4: 分层对比学习
    print("  📊 生成创新点4: 分层对比学习...")
    fig4 = draw_hierarchical_contrastive_flow()
    fig4.savefig('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/innovation4_hierarchical_contrastive.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    fig4.savefig('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/innovation4_hierarchical_contrastive.pdf', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig4)
    print("    ✅ 已保存")
    
    print("\n🎉 所有创新点详细流程图已生成完成！")
    print("\n生成的文件：")
    print("  1. innovation1_bayesian_uncertainty.png/pdf")
    print("  2. innovation2_adaptive_causal_graph.png/pdf")
    print("  3. innovation3_causal_intervention.png/pdf")
    print("  4. innovation4_hierarchical_contrastive.png/pdf")

if __name__ == '__main__':
    main()

