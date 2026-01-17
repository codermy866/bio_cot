#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立的可视化脚本 - 用于重新绘制训练结果图表
支持自定义色调、样式等，适合论文发表需求

使用方法:
1. 从训练结果JSON文件加载数据
2. 调用相应的绘图函数
3. 自定义色调和样式（见函数注释）

数据文件位置:
- 训练历史: results_multimodal/results_bio_cot_multimodal_balanced_*.json
- 预测结果: 包含在JSON文件中的 'final_val_metrics' 字段
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

try:
    import seaborn as sns
    sns.set_style("whitegrid")  # 设置seaborn样式
except ImportError:
    sns = None
    print("⚠️ seaborn未安装，将使用matplotlib绘制")

from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy import stats
import json
from typing import Dict, List, Tuple, Optional

# ============================================================================
# 颜色和样式配置（可自定义）
# ============================================================================

# 主色调配置（可根据论文要求修改）
COLOR_PALETTE = {
    'primary': '#2E86AB',      # 主色：蓝色
    'secondary': '#A23B72',    # 次色：紫红色
    'accent': '#F18F01',       # 强调色：橙色
    'success': '#06A77D',      # 成功色：绿色
    'warning': '#F18F01',      # 警告色：橙色
    'error': '#C73E1D',        # 错误色：红色
    'neutral': '#6C757D',      # 中性色：灰色
    'positive': '#28A745',     # 阳性：绿色
    'negative': '#DC3545',     # 阴性：红色
}

# 图表样式配置
PLOT_STYLE = {
    'figsize': (12, 8),           # 默认图表大小
    'dpi': 300,                   # 分辨率
    'fontsize': 12,               # 字体大小
    'title_fontsize': 14,          # 标题字体大小
    'label_fontsize': 11,          # 标签字体大小
    'linewidth': 2.5,              # 线条宽度
    'markersize': 8,               # 标记大小
    'alpha': 0.7,                  # 透明度
    'cmap': 'viridis',             # 默认颜色映射
}

# 热图颜色映射（可选项：'viridis', 'plasma', 'coolwarm', 'RdYlBu', 'Blues'等）
HEATMAP_CMAP = 'viridis'

# ============================================================================
# 工具函数
# ============================================================================

def load_training_results(json_path: str) -> Dict:
    """加载训练结果JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def ensure_numpy_array(data):
    """确保数据是numpy数组"""
    if isinstance(data, list):
        return np.array(data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array([data])

# ============================================================================
# 绘图函数
# ============================================================================

def plot_training_curves(history: Dict, output_dir: Path, timestamp: str, 
                        colors: Optional[Dict] = None, style: Optional[Dict] = None):
    """
    绘制训练曲线（损失和准确率）
    
    自定义参数:
    - colors: 自定义颜色字典，例如 {'train': '#2E86AB', 'val': '#A23B72'}
    - style: 自定义样式字典，例如 {'linewidth': 3, 'alpha': 0.8}
    """
    if colors is None:
        colors = {
            'train': COLOR_PALETTE['primary'],
            'val': COLOR_PALETTE['secondary']
        }
    if style is None:
        style = PLOT_STYLE.copy()
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 训练/验证损失
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', 
                    color=colors['train'], linewidth=style['linewidth'], alpha=style['alpha'])
    axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', 
                    color=colors['val'], linewidth=style['linewidth'], alpha=style['alpha'])
    axes[0, 0].set_xlabel('Epoch', fontsize=style['label_fontsize'])
    axes[0, 0].set_ylabel('Loss', fontsize=style['label_fontsize'])
    axes[0, 0].set_title('Training and Validation Loss', fontsize=style['title_fontsize'], weight='bold')
    axes[0, 0].legend(fontsize=style['label_fontsize'])
    axes[0, 0].grid(True, alpha=0.3)
    
    # 训练/验证准确率
    axes[0, 1].plot(epochs, history['train_acc'], label='Train Acc', 
                    color=colors['train'], linewidth=style['linewidth'], alpha=style['alpha'], marker='o', markersize=style['markersize']//2)
    axes[0, 1].plot(epochs, history['val_acc'], label='Val Acc', 
                    color=colors['val'], linewidth=style['linewidth'], alpha=style['alpha'], marker='s', markersize=style['markersize']//2)
    axes[0, 1].set_xlabel('Epoch', fontsize=style['label_fontsize'])
    axes[0, 1].set_ylabel('Accuracy', fontsize=style['label_fontsize'])
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=style['title_fontsize'], weight='bold')
    axes[0, 1].legend(fontsize=style['label_fontsize'])
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC曲线
    if 'val_auc' in history and len(history['val_auc']) > 0:
        axes[1, 0].plot(epochs, history['val_auc'], label='Val AUC', 
                        color=colors['val'], linewidth=style['linewidth'], alpha=style['alpha'], marker='^', markersize=style['markersize']//2)
        axes[1, 0].set_xlabel('Epoch', fontsize=style['label_fontsize'])
        axes[1, 0].set_ylabel('AUC', fontsize=style['label_fontsize'])
        axes[1, 0].set_title('Validation AUC', fontsize=style['title_fontsize'], weight='bold')
        axes[1, 0].legend(fontsize=style['label_fontsize'])
        axes[1, 0].grid(True, alpha=0.3)
    
    # F1分数
    if 'val_f1' in history and len(history['val_f1']) > 0:
        axes[1, 1].plot(epochs, history['val_f1'], label='Val F1', 
                        color=colors['val'], linewidth=style['linewidth'], alpha=style['alpha'], marker='d', markersize=style['markersize']//2)
        axes[1, 1].set_xlabel('Epoch', fontsize=style['label_fontsize'])
        axes[1, 1].set_ylabel('F1 Score', fontsize=style['label_fontsize'])
        axes[1, 1].set_title('Validation F1 Score', fontsize=style['title_fontsize'], weight='bold')
        axes[1, 1].legend(fontsize=style['label_fontsize'])
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_custom_{timestamp}.png', 
                dpi=style['dpi'], bbox_inches='tight')
    plt.close()
    print(f"✅ 训练曲线已保存: training_curves_custom_{timestamp}.png")


def plot_confusion_matrix_custom(y_true: np.ndarray, y_pred: np.ndarray, 
                                     output_dir: Path, timestamp: str,
                                     cmap: str = 'Blues', 
                                     colors: Optional[Dict] = None):
    """
    绘制混淆矩阵（可自定义颜色映射）
    
    自定义参数:
    - cmap: 颜色映射，例如 'Blues', 'Reds', 'Greens', 'viridis', 'coolwarm'等
    - colors: 自定义颜色字典
    """
    y_true = ensure_numpy_array(y_true)
    y_pred = ensure_numpy_array(y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'],
                    cbar_kws={'label': 'Count'}, ax=ax)
        # 添加百分比标注
        for i in range(2):
            for j in range(2):
                if cm[i, j] > 0:
                    ax.text(j+0.5, i+0.7, f'({cm_percent[i, j]:.1f}%)',
                           ha='center', va='center', fontsize=10, 
                           color='white' if cm[i, j] > cm.max()/2 else 'black', 
                           weight='bold')
    else:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar(im, ax=ax, label='Count')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Negative', 'Positive'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)',
                       ha='center', va='center', 
                       color='white' if cm[i, j] > cm.max()/2 else 'black',
                       fontsize=14, weight='bold')
    
    ax.set_xlabel('Predicted', fontsize=PLOT_STYLE['label_fontsize'])
    ax.set_ylabel('Actual', fontsize=PLOT_STYLE['label_fontsize'])
    ax.set_title('Confusion Matrix (with Percentages)', 
                fontsize=PLOT_STYLE['title_fontsize'], weight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_custom_{timestamp}.png', 
                dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存: confusion_matrix_custom_{timestamp}.png")


def plot_roc_curve_custom(y_true: np.ndarray, y_probs: np.ndarray,
                          output_dir: Path, timestamp: str,
                          color: str = None, style: Optional[Dict] = None):
    """
    绘制ROC曲线（可自定义颜色）
    
    自定义参数:
    - color: 曲线颜色，例如 '#2E86AB', '#A23B72'等
    - style: 自定义样式字典
    """
    y_true = ensure_numpy_array(y_true)
    y_probs = ensure_numpy_array(y_probs)
    
    if color is None:
        color = COLOR_PALETTE['primary']
    if style is None:
        style = PLOT_STYLE.copy()
    
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=style['figsize'])
    plt.plot(fpr, tpr, color=color, linewidth=style['linewidth'],
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', 
             linewidth=style['linewidth']*0.7, alpha=0.5, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=style['label_fontsize'])
    plt.ylabel('True Positive Rate', fontsize=style['label_fontsize'])
    plt.title('ROC Curve', fontsize=style['title_fontsize'], weight='bold')
    plt.legend(loc='lower right', fontsize=style['label_fontsize'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'roc_curve_custom_{timestamp}.png', 
                dpi=style['dpi'], bbox_inches='tight')
    plt.close()
    print(f"✅ ROC曲线已保存: roc_curve_custom_{timestamp}.png")


def plot_prediction_distribution_custom(y_true: np.ndarray, y_probs: np.ndarray,
                                       output_dir: Path, timestamp: str,
                                       colors: Optional[Dict] = None,
                                       style: Optional[Dict] = None):
    """
    绘制预测概率分布（多种图表，可自定义色调）
    
    自定义参数:
    - colors: 自定义颜色字典，例如 {'positive': '#28A745', 'negative': '#DC3545'}
    - style: 自定义样式字典
    """
    y_true = ensure_numpy_array(y_true).flatten()
    y_probs = ensure_numpy_array(y_probs).flatten()
    
    if colors is None:
        colors = {
            'positive': COLOR_PALETTE['positive'],
            'negative': COLOR_PALETTE['negative']
        }
    if style is None:
        style = PLOT_STYLE.copy()
    
    pos_probs = y_probs[y_true == 1]
    neg_probs = y_probs[y_true == 0]
    
    # 创建9个子图
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 直方图
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(neg_probs, bins=30, alpha=0.6, label='Negative', 
            color=colors['negative'], edgecolor='black')
    ax1.hist(pos_probs, bins=30, alpha=0.6, label='Positive', 
            color=colors['positive'], edgecolor='black')
    ax1.set_xlabel('Predicted Probability', fontsize=style['label_fontsize'])
    ax1.set_ylabel('Frequency', fontsize=style['label_fontsize'])
    ax1.set_title('Probability Distribution (Histogram)', fontsize=style['title_fontsize'], weight='bold')
    ax1.legend(fontsize=style['label_fontsize'])
    ax1.grid(True, alpha=0.3)
    
    # 2. 箱线图
    ax2 = plt.subplot(3, 3, 2)
    bp = ax2.boxplot([neg_probs, pos_probs], labels=['Negative', 'Positive'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor(colors['negative'])
    bp['boxes'][1].set_facecolor(colors['positive'])
    ax2.set_ylabel('Predicted Probability', fontsize=style['label_fontsize'])
    ax2.set_title('Probability Distribution (Boxplot)', fontsize=style['title_fontsize'], weight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 小提琴图
    ax3 = plt.subplot(3, 3, 3)
    if sns is not None:
        df = pd.DataFrame({
            'Class': ['Negative'] * len(neg_probs) + ['Positive'] * len(pos_probs),
            'Probability': np.concatenate([neg_probs, pos_probs])
        })
        sns.violinplot(data=df, x='Class', y='Probability', ax=ax3,
                      palette=[colors['negative'], colors['positive']])
    else:
        ax3.violinplot([neg_probs, pos_probs], positions=[0, 1], 
                      showmeans=True, showmedians=True)
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(['Negative', 'Positive'])
    ax3.set_ylabel('Predicted Probability', fontsize=style['label_fontsize'])
    ax3.set_title('Probability Distribution (Violin Plot)', fontsize=style['title_fontsize'], weight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 2D热图（概率分布）
    ax4 = plt.subplot(3, 3, 4)
    H, xedges, yedges = np.histogram2d(y_probs, y_true, bins=20)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax4.imshow(H.T, origin='lower', extent=extent, aspect='auto', cmap=HEATMAP_CMAP)
    ax4.set_xlabel('Predicted Probability', fontsize=style['label_fontsize'])
    ax4.set_ylabel('True Label', fontsize=style['label_fontsize'])
    ax4.set_title('2D Probability Heatmap', fontsize=style['title_fontsize'], weight='bold')
    plt.colorbar(im, ax=ax4, label='Count')
    
    # 5. CDF（累积分布函数）
    ax5 = plt.subplot(3, 3, 5)
    sorted_neg = np.sort(neg_probs)
    sorted_pos = np.sort(pos_probs)
    p_neg = np.arange(1, len(sorted_neg) + 1) / len(sorted_neg)
    p_pos = np.arange(1, len(sorted_pos) + 1) / len(sorted_pos)
    ax5.plot(sorted_neg, p_neg, label='Negative', color=colors['negative'], 
            linewidth=style['linewidth'])
    ax5.plot(sorted_pos, p_pos, label='Positive', color=colors['positive'], 
            linewidth=style['linewidth'])
    ax5.set_xlabel('Predicted Probability', fontsize=style['label_fontsize'])
    ax5.set_ylabel('Cumulative Probability', fontsize=style['label_fontsize'])
    ax5.set_title('Cumulative Distribution Function', fontsize=style['title_fontsize'], weight='bold')
    ax5.legend(fontsize=style['label_fontsize'])
    ax5.grid(True, alpha=0.3)
    
    # 6. Q-Q图
    ax6 = plt.subplot(3, 3, 6)
    stats.probplot(neg_probs, dist="norm", plot=ax6)
    ax6.set_title('Q-Q Plot (Negative)', fontsize=style['title_fontsize'], weight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. 统计摘要
    ax7 = plt.subplot(3, 3, 7)
    ax7.axis('off')
    stats_text = f"""
    Statistics Summary
    
    Negative Class:
      Mean: {np.mean(neg_probs):.4f}
      Std:  {np.std(neg_probs):.4f}
      Min:  {np.min(neg_probs):.4f}
      Max:  {np.max(neg_probs):.4f}
      Median: {np.median(neg_probs):.4f}
    
    Positive Class:
      Mean: {np.mean(pos_probs):.4f}
      Std:  {np.std(pos_probs):.4f}
      Min:  {np.min(pos_probs):.4f}
      Max:  {np.max(pos_probs):.4f}
      Median: {np.median(pos_probs):.4f}
    """
    ax7.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax7.transAxes)
    
    # 8. 密度对比
    ax8 = plt.subplot(3, 3, 8)
    if sns is not None:
        sns.kdeplot(neg_probs, ax=ax8, label='Negative', color=colors['negative'], fill=True)
        sns.kdeplot(pos_probs, ax=ax8, label='Positive', color=colors['positive'], fill=True)
    else:
        ax8.hist(neg_probs, bins=30, alpha=0.5, label='Negative', 
               color=colors['negative'], density=True)
        ax8.hist(pos_probs, bins=30, alpha=0.5, label='Positive', 
               color=colors['positive'], density=True)
    ax8.set_xlabel('Predicted Probability', fontsize=style['label_fontsize'])
    ax8.set_ylabel('Density', fontsize=style['label_fontsize'])
    ax8.set_title('Density Comparison', fontsize=style['title_fontsize'], weight='bold')
    ax8.legend(fontsize=style['label_fontsize'])
    ax8.grid(True, alpha=0.3)
    
    # 9. 直方图+KDE
    ax9 = plt.subplot(3, 3, 9)
    ax9.hist(neg_probs, bins=30, alpha=0.6, label='Negative', 
            color=colors['negative'], density=True, edgecolor='black')
    ax9.hist(pos_probs, bins=30, alpha=0.6, label='Positive', 
            color=colors['positive'], density=True, edgecolor='black')
    if sns is not None:
        sns.kdeplot(neg_probs, ax=ax9, color=colors['negative'], linewidth=style['linewidth'])
        sns.kdeplot(pos_probs, ax=ax9, color=colors['positive'], linewidth=style['linewidth'])
    ax9.set_xlabel('Predicted Probability', fontsize=style['label_fontsize'])
    ax9.set_ylabel('Density', fontsize=style['label_fontsize'])
    ax9.set_title('Histogram with KDE', fontsize=style['title_fontsize'], weight='bold')
    ax9.legend(fontsize=style['label_fontsize'])
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'prediction_distribution_custom_{timestamp}.png', 
                dpi=style['dpi'], bbox_inches='tight')
    plt.close()
    print(f"✅ 预测分布图已保存: prediction_distribution_custom_{timestamp}.png")


def plot_loss_heatmap_custom(history: Dict, output_dir: Path, timestamp: str,
                             cmap: str = 'YlOrRd'):
    """
    绘制损失组件热图（可自定义颜色映射）
    
    自定义参数:
    - cmap: 颜色映射，例如 'YlOrRd', 'viridis', 'plasma', 'coolwarm'等
    """
    epochs = range(1, len(history['train_loss']) + 1)
    loss_data = np.array([
        history['train_cls_loss'],
        history['train_ot_loss'],
        history['train_consist_loss'],
        history['train_adv_loss']
    ])
    
    # 归一化
    loss_data_norm = (loss_data - loss_data.min(axis=1, keepdims=True)) / \
                     (loss_data.max(axis=1, keepdims=True) - loss_data.min(axis=1, keepdims=True) + 1e-10)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 原始损失热图
    im1 = axes[0].imshow(loss_data, cmap=cmap, aspect='auto', interpolation='nearest')
    axes[0].set_yticks(range(4))
    axes[0].set_yticklabels(['CLS Loss', 'OT Loss', 'Consist Loss', 'Adv Loss'])
    axes[0].set_xlabel('Epoch', fontsize=PLOT_STYLE['label_fontsize'])
    axes[0].set_ylabel('Loss Component', fontsize=PLOT_STYLE['label_fontsize'])
    axes[0].set_title('Loss Components Heatmap (Raw)', fontsize=PLOT_STYLE['title_fontsize'], weight='bold')
    plt.colorbar(im1, ax=axes[0], label='Loss Value')
    
    # 归一化损失热图
    im2 = axes[1].imshow(loss_data_norm, cmap=cmap, aspect='auto', interpolation='nearest')
    axes[1].set_yticks(range(4))
    axes[1].set_yticklabels(['CLS Loss', 'OT Loss', 'Consist Loss', 'Adv Loss'])
    axes[1].set_xlabel('Epoch', fontsize=PLOT_STYLE['label_fontsize'])
    axes[1].set_ylabel('Loss Component', fontsize=PLOT_STYLE['label_fontsize'])
    axes[1].set_title('Loss Components Heatmap (Normalized)', fontsize=PLOT_STYLE['title_fontsize'], weight='bold')
    plt.colorbar(im2, ax=axes[1], label='Normalized Loss')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'loss_heatmap_custom_{timestamp}.png', 
                dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    plt.close()
    print(f"✅ 损失热图已保存: loss_heatmap_custom_{timestamp}.png")


# ============================================================================
# 主函数：从JSON文件加载数据并重新绘制
# ============================================================================

def main():
    """主函数：加载数据并重新绘制所有图表"""
    import argparse
    
    parser = argparse.ArgumentParser(description='重新绘制训练结果图表（可自定义色调）')
    parser.add_argument('--json_path', type=str, required=True,
                       help='训练结果JSON文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认：JSON文件所在目录）')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='时间戳（默认：从JSON文件名提取）')
    
    args = parser.parse_args()
    
    # 加载数据
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"❌ 文件不存在: {json_path}")
        return
    
    print(f"📂 加载数据: {json_path}")
    data = load_training_results(str(json_path))
    
    # 确定输出目录和时间戳
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = json_path.parent
    
    if args.timestamp:
        timestamp = args.timestamp
    else:
        # 从文件名提取时间戳
        timestamp = json_path.stem.split('_')[-1] if '_' in json_path.stem else 'custom'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {output_dir}")
    print(f"🏷️  时间戳: {timestamp}")
    
    # 提取数据
    history = data.get('history', {})
    final_val_metrics = data.get('final_val_metrics', {})
    
    if not final_val_metrics:
        # 尝试从其他字段获取
        if 'labels' in data and 'probs' in data:
            final_val_metrics = {
                'labels': data['labels'],
                'probs': data['probs'],
                'preds': data.get('preds', [])
            }
    
    # 绘制所有图表
    print("\n📊 开始绘制图表...")
    
    if history:
        print("1. 绘制训练曲线...")
        plot_training_curves(history, output_dir, timestamp)
        
        print("2. 绘制损失热图...")
        plot_loss_heatmap_custom(history, output_dir, timestamp)
    
    if final_val_metrics and 'labels' in final_val_metrics:
        print("3. 绘制混淆矩阵...")
        y_true = ensure_numpy_array(final_val_metrics['labels'])
        y_pred = ensure_numpy_array(final_val_metrics.get('preds', 
            np.round(final_val_metrics.get('probs', []))).astype(int))
        plot_confusion_matrix_custom(y_true, y_pred, output_dir, timestamp)
        
        print("4. 绘制ROC曲线...")
        y_probs = ensure_numpy_array(final_val_metrics['probs'])
        plot_roc_curve_custom(y_true, y_probs, output_dir, timestamp)
        
        print("5. 绘制预测分布图...")
        plot_prediction_distribution_custom(y_true, y_probs, output_dir, timestamp)
    
    print(f"\n✅ 所有图表已保存到: {output_dir}")
    print("\n💡 提示：")
    print("   - 修改脚本中的 COLOR_PALETTE 和 PLOT_STYLE 可自定义色调和样式")
    print("   - 调用绘图函数时传入 colors 和 style 参数可进一步自定义")


if __name__ == '__main__':
    main()

