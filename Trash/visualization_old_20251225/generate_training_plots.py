#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练结果可视化生成器
自动生成训练过程中的各种可视化图表
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from datetime import datetime
try:
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# 设置中文字体和英文字体（确保字体正确显示）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

sns.set_style("whitegrid")
sns.set_palette("husl")


def generate_training_plots(output_dir='cnn_result'):
    """生成训练可视化图表"""
    print("\n" + "=" * 80)
    print("📊 开始生成训练可视化图表...")
    print("=" * 80)
    
    # 加载训练历史
    history_file = os.path.join(output_dir, 'training_history.json')
    if not os.path.exists(history_file):
        print(f"❌ 未找到训练历史文件: {history_file}")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # 处理 null/None 值：将 None 转换为 NaN，然后过滤或使用默认值
    def clean_data(data, default=0.0):
        """清理数据，将 None/null 转换为 NaN 或默认值"""
        cleaned = []
        for val in data:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                cleaned.append(np.nan)
            else:
                cleaned.append(float(val))
        return np.array(cleaned)
    
    # 清理所有数据
    train_loss = clean_data(history.get('train_loss', []))
    val_loss = clean_data(history.get('val_loss', []))
    train_acc = clean_data(history.get('train_acc', []), default=0.0)
    val_acc = clean_data(history.get('val_acc', []), default=0.0)
    val_auc = clean_data(history.get('val_auc', []), default=0.0)
    
    epochs = range(1, len(train_loss) + 1)
    
    # 创建输出目录
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. 训练/验证Loss曲线（使用平滑曲线）
    print("📈 生成 Loss 曲线...")
    plt.figure(figsize=(12, 6))
    
    # 使用移动平均平滑Loss曲线（只处理非NaN值）
    train_loss_clean = train_loss[~np.isnan(train_loss)]
    val_loss_clean = val_loss[~np.isnan(val_loss)]
    
    if len(train_loss_clean) > 1 and HAS_SCIPY:
        window = max(1, len(train_loss_clean) // 10)  # 自适应窗口大小
        try:
            train_loss_smooth = uniform_filter1d(train_loss_clean, size=window, mode='nearest')
            val_loss_smooth = uniform_filter1d(val_loss_clean, size=window, mode='nearest')
            # 重建完整数组（包含NaN）
            train_loss_smooth_full = np.full_like(train_loss, np.nan)
            val_loss_smooth_full = np.full_like(val_loss, np.nan)
            train_loss_smooth_full[~np.isnan(train_loss)] = train_loss_smooth
            val_loss_smooth_full[~np.isnan(val_loss)] = val_loss_smooth
            train_loss_smooth = train_loss_smooth_full
            val_loss_smooth = val_loss_smooth_full
        except:
            train_loss_smooth = train_loss
            val_loss_smooth = val_loss
    else:
        train_loss_smooth = train_loss
        val_loss_smooth = val_loss
    
    plt.plot(epochs, train_loss_smooth, 'b-', label='Train Loss (smoothed)', linewidth=2.5, alpha=0.8)
    plt.plot(epochs, val_loss_smooth, 'r-', label='Val Loss (smoothed)', linewidth=2.5, alpha=0.8)
    # 原始数据点（淡色）
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss (raw)', linewidth=1, alpha=0.3)
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss (raw)', linewidth=1, alpha=0.3)
    
    plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ 已保存: {plots_dir}/loss_curves.png")
    
    # 2. 训练/验证Accuracy曲线
    print("📈 生成 Accuracy 曲线...")
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2.5, alpha=0.8)
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2.5, alpha=0.8)
    plt.title('Training and Validation Accuracy Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'accuracy_curves.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ 已保存: {plots_dir}/accuracy_curves.png")
    
    # 3. AUC曲线
    if 'val_auc' in history and history['val_auc']:
        print("📈 生成 AUC 曲线...")
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, history['val_auc'], 'g-', label='Val AUC', linewidth=2.5, marker='o', markersize=6, alpha=0.8)
        best_auc = max(history['val_auc'])
        best_epoch = history['val_auc'].index(best_auc) + 1
        plt.plot(best_epoch, best_auc, 'ro', markersize=14, label=f'Best AUC: {best_auc:.4f} (Epoch {best_epoch})', zorder=5)
        plt.title('Validation AUC Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('AUC', fontsize=12)
        plt.ylim([0.5, 1.0])
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'auc_curve.png'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   ✅ 已保存: {plots_dir}/auc_curve.png")
    
    # 4. F1-Score, Precision, Recall曲线
    if 'val_f1' in history and history['val_f1']:
        print("📈 生成 F1-Score, Precision, Recall 曲线...")
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, history['val_f1'], 'b-', label='F1-Score', linewidth=2.5, alpha=0.8)
        if 'val_precision' in history:
            plt.plot(epochs, history['val_precision'], 'g-', label='Precision', linewidth=2.5, alpha=0.8)
        if 'val_recall' in history:
            plt.plot(epochs, history['val_recall'], 'r-', label='Recall', linewidth=2.5, alpha=0.8)
        plt.title('Validation F1-Score, Precision, Recall Curves', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'f1_precision_recall_curves.png'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   ✅ 已保存: {plots_dir}/f1_precision_recall_curves.png")
    
    # 5. 综合指标对比图
    print("📈 生成综合指标对比图...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss（使用平滑曲线）
    if len(history['train_loss']) > 1 and HAS_SCIPY:
        window = max(1, len(history['train_loss']) // 10)
        try:
            train_loss_smooth = uniform_filter1d(np.array(history['train_loss']), size=window, mode='nearest')
            val_loss_smooth = uniform_filter1d(np.array(history['val_loss']), size=window, mode='nearest')
        except:
            train_loss_smooth = np.array(history['train_loss'])
            val_loss_smooth = np.array(history['val_loss'])
    else:
        train_loss_smooth = np.array(history['train_loss'])
        val_loss_smooth = np.array(history['val_loss'])
    
    axes[0, 0].plot(epochs, train_loss_smooth, 'b-', label='Train', linewidth=2.5, alpha=0.8)
    axes[0, 0].plot(epochs, val_loss_smooth, 'r-', label='Val', linewidth=2.5, alpha=0.8)
    axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2.5, alpha=0.8)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2.5, alpha=0.8)
    axes[0, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # AUC
    if 'val_auc' in history:
        axes[1, 0].plot(epochs, history['val_auc'], 'g-', linewidth=2.5, marker='o', markersize=5, alpha=0.8)
        best_auc = max(history['val_auc'])
        best_epoch = history['val_auc'].index(best_auc) + 1
        axes[1, 0].plot(best_epoch, best_auc, 'ro', markersize=12, zorder=5)
        axes[1, 0].set_title(f'AUC (Best: {best_auc:.4f})', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('AUC', fontsize=11)
        axes[1, 0].set_ylim([0.5, 1.0])
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # F1-Score
    if 'val_f1' in history:
        axes[1, 1].plot(epochs, history['val_f1'], 'm-', linewidth=2.5, alpha=0.8, label='F1')
        if 'val_precision' in history:
            axes[1, 1].plot(epochs, history['val_precision'], 'g--', linewidth=2, alpha=0.7, label='Precision')
        if 'val_recall' in history:
            axes[1, 1].plot(epochs, history['val_recall'], 'r--', linewidth=2, alpha=0.7, label='Recall')
        axes[1, 1].set_title('F1-Score / Precision / Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch', fontsize=11)
        axes[1, 1].set_ylabel('Score', fontsize=11)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Training Metrics Summary', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_summary.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ 已保存: {plots_dir}/training_summary.png")
    
    # 6. 从metrics.json读取最新的混淆矩阵
    metrics_file = os.path.join(output_dir, 'metrics.json')
    if os.path.exists(metrics_file):
        print("📈 生成混淆矩阵热力图...")
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        if 'val' in metrics and 'confusion_matrix' in metrics['val']:
            cm_data = metrics['val']['confusion_matrix']
            cm = np.array([[cm_data['tn'], cm_data['fp']],
                          [cm_data['fn'], cm_data['tp']]])
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Abnormal'], 
                       yticklabels=['Normal', 'Abnormal'],
                       cbar_kws={'label': 'Count'}, annot_kws={'size': 12, 'weight': 'bold'})
            plt.title('Confusion Matrix Heatmap', fontsize=14, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            
            # 计算并显示指标
            tn, fp, fn, tp = cm_data['tn'], cm_data['fp'], cm_data['fn'], cm_data['tp']
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            info_text = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}'
            plt.text(0.5, -0.15, info_text, transform=plt.gca().transAxes,
                    fontsize=11, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"   ✅ 已保存: {plots_dir}/confusion_matrix.png")
    
    print("\n" + "=" * 80)
    print(f"✅ 所有可视化图表已生成完成！")
    print(f"📁 保存位置: {plots_dir}/")
    print("=" * 80)
    
    # 列出生成的文件
    generated_files = [
        'loss_curves.png',
        'accuracy_curves.png',
        'auc_curve.png',
        'f1_precision_recall_curves.png',
        'training_summary.png',
        'confusion_matrix.png'
    ]
    
    print("\n📊 生成的图表文件:")
    for fname in generated_files:
        filepath = os.path.join(plots_dir, fname)
        if os.path.exists(filepath):
            print(f"   ✅ {fname}")
        else:
            print(f"   ⚠️  {fname} (未生成)")


if __name__ == '__main__':
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'cnn_result'
    generate_training_plots(output_dir)

