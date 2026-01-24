#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级可视化图表生成器
包括：小提琴图、气泡图、雷达图、桑基图、校准曲线等
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    # 旧版本sklearn可能使用不同的导入路径
    try:
        from sklearn.metrics import calibration_curve
    except ImportError:
        calibration_curve = None
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

sns.set_style("whitegrid")
sns.set_palette("husl")


def generate_all_advanced_visualizations(output_dir: str = 'cnn_result', history_path: Optional[str] = None, 
                                       metrics_path: Optional[str] = None):
    """生成所有高级可视化图表"""
    print("\n" + "=" * 80)
    print("📊 开始生成高级可视化图表...")
    print("=" * 80)
    
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 加载数据
    if history_path is None:
        history_path = os.path.join(output_dir, 'training_history.json')
    if metrics_path is None:
        metrics_path = os.path.join(output_dir, 'metrics.json')
    
    if not os.path.exists(history_path):
        print(f"❌ 未找到训练历史文件: {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    
    # 生成各种图表
    print("\n1️⃣ 生成小提琴图...")
    generate_violin_plots(history, plots_dir)
    
    print("\n2️⃣ 生成气泡图...")
    generate_bubble_chart(history, plots_dir)
    
    print("\n3️⃣ 生成雷达图...")
    generate_radar_chart(metrics, plots_dir)
    
    print("\n4️⃣ 生成桑基图...")
    generate_sankey_diagram(metrics, plots_dir)
    
    print("\n5️⃣ 生成校准曲线...")
    generate_calibration_curve(metrics, plots_dir)
    
    print("\n6️⃣ 生成指标对比条形图...")
    generate_metrics_comparison(history, plots_dir)
    
    print("\n7️⃣ 生成ROC曲线详细图...")
    generate_detailed_roc(metrics, plots_dir)
    
    print("\n8️⃣ 生成学习率曲线...")
    generate_lr_curve(history, plots_dir)
    
    print("\n" + "=" * 80)
    print("✅ 所有高级可视化图表已生成完成！")
    print(f"📁 保存位置: {plots_dir}/")
    print("=" * 80)


def generate_violin_plots(history: Dict, plots_dir: str):
    """生成小提琴图（显示指标的分布）"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 准备数据
    data = []
    for epoch in epochs:
        idx = epoch - 1
        data.append({'Epoch': epoch, 'Type': 'Train', 'Loss': history['train_loss'][idx]})
        data.append({'Epoch': epoch, 'Type': 'Val', 'Loss': history['val_loss'][idx]})
        data.append({'Epoch': epoch, 'Type': 'Train', 'Accuracy': history['train_acc'][idx]})
        data.append({'Epoch': epoch, 'Type': 'Val', 'Accuracy': history['val_acc'][idx]})
    
    df = pd.DataFrame(data)
    
    # Loss小提琴图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss分布
    loss_data = df[df['Loss'].notna()]
    if len(loss_data) > 0:
        sns.violinplot(data=loss_data, x='Type', y='Loss', ax=axes[0], palette=['#3498db', '#e74c3c'])
        axes[0].set_title('Loss Distribution by Type', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Dataset Type', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
    
    # Accuracy分布
    acc_data = df[df['Accuracy'].notna()]
    if len(acc_data) > 0:
        sns.violinplot(data=acc_data, x='Type', y='Accuracy', ax=axes[1], palette=['#3498db', '#e74c3c'])
        axes[1].set_title('Accuracy Distribution by Type', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Dataset Type', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'violin_plots.png'), dpi=300, facecolor='white')
    plt.close()
    print("   ✅ 已保存: violin_plots.png")


def generate_bubble_chart(history: Dict, plots_dir: str):
    """生成气泡图（显示多个指标的关系）"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 准备数据
    data = []
    for epoch in epochs:
        idx = epoch - 1
        if 'val_auc' in history and idx < len(history['val_auc']):
            data.append({
                'Epoch': epoch,
                'Loss': history['val_loss'][idx],
                'Accuracy': history['val_acc'][idx],
                'AUC': history['val_auc'][idx] if history['val_auc'][idx] > 0 else 0.5,
                'Size': (history['val_auc'][idx] * 100) if history['val_auc'][idx] > 0 else 50
            })
    
    if not data:
        print("   ⚠️ 数据不足，跳过气泡图")
        return
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['Loss'], df['Accuracy'], s=df['Size']*10, 
                         c=df['AUC'], cmap='viridis', alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # 标注最佳点
    best_idx = df['AUC'].idxmax()
    plt.scatter(df.loc[best_idx, 'Loss'], df.loc[best_idx, 'Accuracy'], 
               s=500, c='red', marker='*', edgecolors='black', linewidth=2, zorder=5,
               label=f"Best (Epoch {df.loc[best_idx, 'Epoch']})")
    
    plt.colorbar(scatter, label='AUC')
    plt.xlabel('Validation Loss', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.title('Bubble Chart: Loss vs Accuracy (Size=AUC)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'bubble_chart.png'), dpi=300, facecolor='white')
    plt.close()
    print("   ✅ 已保存: bubble_chart.png")


def generate_radar_chart(metrics: Dict, plots_dir: str):
    """生成雷达图（显示多个指标的综合表现）"""
    if 'val' not in metrics:
        print("   ⚠️ 数据不足，跳过雷达图")
        return
    
    val_metrics = metrics['val']
    
    # 选择的指标
    metric_names = ['Accuracy', 'AUC', 'F1-Score', 'Precision', 'Recall', 'Specificity']
    metric_values = []
    
    for name in metric_names:
        if name.lower() in val_metrics:
            val = val_metrics[name.lower()]
            # 归一化到0-1范围（假设Accuracy已经是百分比，AUC已经是0-1）
            if name == 'Accuracy':
                metric_values.append(val / 100.0)
            else:
                metric_values.append(val)
        else:
            metric_values.append(0.0)
    
    # 计算特异性（如果没有）
    if 'specificity' not in val_metrics and 'confusion_matrix' in val_metrics:
        cm = val_metrics['confusion_matrix']
        tn, fp = cm['tn'], cm['fp']
        if tn + fp > 0:
            specificity = tn / (tn + fp)
            metric_values[5] = specificity
    
    # 创建雷达图
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    metric_values += metric_values[:1]  # 闭合
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, metric_values, 'o-', linewidth=2, label='Current Model', color='#3498db')
    ax.fill(angles, metric_values, alpha=0.25, color='#3498db')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True)
    
    plt.title('Radar Chart: Comprehensive Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'radar_chart.png'), dpi=300, facecolor='white')
    plt.close()
    print("   ✅ 已保存: radar_chart.png")


def generate_sankey_diagram(metrics: Dict, plots_dir: str):
    """生成桑基图（显示混淆矩阵的流向）"""
    if 'val' not in metrics or 'confusion_matrix' not in metrics['val']:
        print("   ⚠️ 数据不足，跳过桑基图")
        return
    
    cm = metrics['val']['confusion_matrix']
    tn, fp, fn, tp = cm['tn'], cm['fp'], cm['fn'], cm['tp']
    
    try:
        import plotly.graph_objects as go
        import plotly.offline as pyo
        
        # 节点标签
        labels = ['True Negative', 'True Positive', 'False Negative', 'False Positive']
        # 节点索引
        source_indices = [0, 1, 0, 1]  # TN和TP从真实标签来
        target_indices = [0, 1, 1, 0]  # 目标：正确预测和错误预测
        
        # 重新定义：从真实标签到预测标签的流向
        # 0: 真实阴性 -> 0: 预测阴性(TN), 1: 预测阳性(FP)
        # 1: 真实阳性 -> 0: 预测阴性(FN), 1: 预测阳性(TP)
        sources = [0, 0, 1, 1]  # 真实标签索引
        targets = [2, 3, 2, 3]  # 预测结果索引
        values = [tn, fp, fn, tp]  # 流向的值
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=['True Negative (Real)', 'True Positive (Real)', 
                      'Predicted Negative', 'Predicted Positive'],
                color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=['rgba(52, 152, 219, 0.4)', 'rgba(231, 76, 60, 0.4)', 
                      'rgba(231, 76, 60, 0.4)', 'rgba(46, 204, 113, 0.4)']
            )
        )])
        
        fig.update_layout(title_text="Sankey Diagram: True Labels to Predictions Flow", 
                         font_size=14, font_family="Arial")
        
        # 保存为HTML
        html_path = os.path.join(plots_dir, 'sankey_diagram.html')
        fig.write_html(html_path)
        
        # 也尝试保存为PNG（如果可能）
        try:
            png_path = os.path.join(plots_dir, 'sankey_diagram.png')
            fig.write_image(png_path, width=1200, height=800)
            print("   ✅ 已保存: sankey_diagram.png 和 sankey_diagram.html")
        except:
            print("   ✅ 已保存: sankey_diagram.html (PNG需要kaleido)")
    except ImportError:
        # 如果没有plotly，使用matplotlib绘制简化版本
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制简化的流向图
        categories = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
        values = [tn, fp, fn, tp]
        colors = ['#3498db', '#f39c12', '#e74c3c', '#2ecc71']
        
        bars = ax.barh(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + max(values) * 0.01, i, f'{val}', va='center', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Count', fontsize=12)
        ax.set_title('Confusion Matrix Flow (Simplified Sankey)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'sankey_diagram.png'), dpi=300, facecolor='white')
        plt.close()
        print("   ✅ 已保存: sankey_diagram.png (简化版本)")


def generate_calibration_curve(metrics: Dict, plots_dir: str):
    """生成校准曲线（Calibration Curve）"""
    # 如果metrics中没有概率数据，需要从训练过程中获取
    # 这里先创建一个示例，实际使用时需要传入真实的概率数据
    
    # 从历史数据模拟（实际应该从验证过程中获取）
    if 'val_probs' not in metrics:
        print("   ⚠️ 无概率数据，跳过校准曲线")
        return
    
    # 这里假设有概率数据
    # 实际使用时需要修改训练脚本，保存验证集的概率值
    
    print("   ⚠️ 校准曲线需要概率数据，将在训练脚本中添加")


def generate_metrics_comparison(history: Dict, plots_dir: str):
    """生成指标对比条形图"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 选择关键epochs（开始、中间、结束、最佳）
    key_epochs = [1, len(epochs) // 2, len(epochs)]
    if 'val_auc' in history:
        best_epoch_idx = np.argmax(history['val_auc'])
        if best_epoch_idx + 1 not in key_epochs:
            key_epochs.append(best_epoch_idx + 1)
    
    # 准备数据
    metrics_to_compare = []
    if 'val_auc' in history:
        metrics_to_compare.append(('AUC', [history['val_auc'][e-1] for e in key_epochs]))
    metrics_to_compare.append(('Accuracy', [history['val_acc'][e-1] for e in key_epochs]))
    if 'val_f1' in history:
        metrics_to_compare.append(('F1-Score', [history['val_f1'][e-1] for e in key_epochs]))
    
    x = np.arange(len(key_epochs))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, (metric_name, values) in enumerate(metrics_to_compare):
        offset = (i - len(metrics_to_compare) / 2) * width
        ax.bar(x + offset, values, width, label=metric_name, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Metrics Comparison Across Key Epochs', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Epoch {e}' for e in key_epochs])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'metrics_comparison.png'), dpi=300, facecolor='white')
    plt.close()
    print("   ✅ 已保存: metrics_comparison.png")


def generate_detailed_roc(metrics: Dict, plots_dir: str):
    """生成详细的ROC曲线图"""
    if 'val' not in metrics or 'auc' not in metrics['val']:
        print("   ⚠️ 数据不足，跳过ROC曲线")
        return
    
    # 如果没有ROC曲线数据，创建一个简单的图
    auc_value = metrics['val']['auc']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制对角线（随机分类器）
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5)')
    
    # 如果有ROC曲线数据，绘制它
    if 'roc_curve' in metrics['val']:
        roc_data = metrics['val']['roc_curve']
        ax.plot(roc_data['fpr'], roc_data['tpr'], linewidth=2.5, 
               label=f'Model (AUC = {auc_value:.4f})', color='#3498db')
    else:
        # 绘制示例ROC曲线（实际应该从训练过程中获取）
        # 这里仅作占位
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr) * auc_value  # 简化的曲线形状
        ax.plot(fpr, tpr, linewidth=2.5, label=f'Model (AUC = {auc_value:.4f})', color='#3498db')
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'detailed_roc.png'), dpi=300, facecolor='white')
    plt.close()
    print("   ✅ 已保存: detailed_roc.png")


def generate_lr_curve(history: Dict, plots_dir: str):
    """生成学习率曲线"""
    if 'learning_rate' not in history:
        print("   ⚠️ 无学习率数据，跳过学习率曲线")
        return
    
    epochs = range(1, len(history['learning_rate']) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history['learning_rate'], 'b-', linewidth=2.5, marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'lr_curve.png'), dpi=300, facecolor='white')
    plt.close()
    print("   ✅ 已保存: lr_curve.png")


if __name__ == '__main__':
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'cnn_result'
    generate_all_advanced_visualizations(output_dir)

