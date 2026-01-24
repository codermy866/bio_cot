#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
结果可视化脚本：从训练历史生成详细的可视化图表
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目路径
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from config import BioCOT_v3_Config


def load_latest_history():
    """加载最新的训练历史"""
    config = BioCOT_v3_Config()
    log_dir = Path(config.log_dir)
    
    history_files = sorted(log_dir.glob("training_history_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not history_files:
        print("❌ 未找到训练历史文件！")
        return None, None
    
    latest_file = history_files[0]
    print(f"📥 加载训练历史: {latest_file.name}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    timestamp = latest_file.stem.replace('training_history_', '')
    return history, timestamp


def visualize_comprehensive(history: dict, timestamp: str, output_dir: Path):
    """生成综合可视化图表"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    epochs = np.array(range(1, len(history['train_loss']) + 1))
    
    # 创建大图
    fig = plt.figure(figsize=(24, 16))
    
    # 1. Loss曲线（左上）
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2.5, alpha=0.8)
    ax1.plot(epochs, history['val_loss'], 'r--', label='Val Loss', linewidth=2.5, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Loss Curves', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy曲线（中上）
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2.5, alpha=0.8)
    ax2.plot(epochs, history['val_acc'], 'r--', label='Val Acc', linewidth=2.5, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Accuracy Curves', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. AUC曲线（右上）
    ax3 = plt.subplot(3, 3, 3)
    best_auc = max(history['val_auc'])
    ax3.plot(epochs, history['val_auc'], 'g-', label='Val AUC', linewidth=2.5, marker='o', markersize=5, alpha=0.8)
    ax3.axhline(y=best_auc, color='r', linestyle='--', linewidth=2, label=f'Best: {best_auc:.4f}')
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Validation AUC', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. F1-Score曲线（左中）
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(epochs, history['val_f1'], 'm-', label='Val F1', linewidth=2.5, marker='s', markersize=5, alpha=0.8)
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Validation F1-Score', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # 5. Loss组件（中中）
    ax5 = plt.subplot(3, 3, 5)
    if history.get('cls_loss'):
        ax5.plot(epochs, history['cls_loss'], 'b-', label='Classification', linewidth=2, alpha=0.8)
    if history.get('ot_loss') and any(v > 0 for v in history['ot_loss']):
        ax5.plot(epochs, history['ot_loss'], 'g-', label='OT Loss', linewidth=2, alpha=0.8)
    if history.get('sparse_loss') and any(v > 0 for v in history['sparse_loss']):
        ax5.plot(epochs, history['sparse_loss'], 'orange', label='Sparse Loss', linewidth=2, alpha=0.8)
    if history.get('consist_loss') and any(v > 0 for v in history['consist_loss']):
        ax5.plot(epochs, history['consist_loss'], 'purple', label='Consistency', linewidth=2, alpha=0.8)
    if history.get('adv_loss') and any(v > 0 for v in history['adv_loss']):
        ax5.plot(epochs, history['adv_loss'], 'r-', label='Adversarial', linewidth=2, alpha=0.8)
    ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax5.set_title('(e) Loss Components', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # 6. 综合性能指标（右中）
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(epochs, history['val_auc'], 'g-', label='AUC', linewidth=2.5, marker='o', markersize=5, alpha=0.8)
    ax6.plot(epochs, history['val_acc'], 'b-', label='Accuracy', linewidth=2.5, marker='s', markersize=5, alpha=0.8)
    ax6.plot(epochs, history['val_f1'], 'm-', label='F1-Score', linewidth=2.5, marker='^', markersize=5, alpha=0.8)
    ax6.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax6.set_title('(f) Comprehensive Metrics', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1])
    
    # 7. 训练/验证对比（左下）
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, alpha=0.7)
    ax7.plot(epochs, history['val_loss'], 'r--', label='Val Loss', linewidth=2, alpha=0.7)
    ax7_twin = ax7.twinx()
    ax7_twin.plot(epochs, history['train_acc'], 'b:', label='Train Acc', linewidth=2, alpha=0.7)
    ax7_twin.plot(epochs, history['val_acc'], 'r:', label='Val Acc', linewidth=2, alpha=0.7)
    ax7.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Loss', fontsize=12, fontweight='bold', color='black')
    ax7_twin.set_ylabel('Accuracy', fontsize=12, fontweight='bold', color='gray')
    ax7.set_title('(g) Loss & Accuracy Overlay', fontsize=13, fontweight='bold')
    ax7.legend(loc='upper left', fontsize=10)
    ax7_twin.legend(loc='upper right', fontsize=10)
    ax7.grid(True, alpha=0.3)
    
    # 8. 性能趋势（中下）
    ax8 = plt.subplot(3, 3, 8)
    # 计算移动平均
    window = min(5, len(history['val_auc']) // 3)
    if window > 1:
        auc_smooth = np.convolve(history['val_auc'], np.ones(window)/window, mode='valid')
        epochs_smooth = epochs[window-1:]
        ax8.plot(epochs_smooth, auc_smooth, 'g-', label='AUC (Smoothed)', linewidth=3, alpha=0.8)
    ax8.plot(epochs, history['val_auc'], 'g:', label='AUC (Raw)', linewidth=1.5, alpha=0.5)
    ax8.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax8.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax8.set_title('(h) AUC Trend (Smoothed)', fontsize=13, fontweight='bold')
    ax8.legend(fontsize=11)
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim([0, 1])
    
    # 9. 最终性能总结（右下）
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # 计算统计信息
    best_auc = max(history['val_auc'])
    best_auc_epoch = np.argmax(history['val_auc']) + 1
    final_auc = history['val_auc'][-1]
    final_acc = history['val_acc'][-1]
    final_f1 = history['val_f1'][-1]
    
    summary_text = f"""
    Training Summary
    
    Total Epochs: {len(epochs)}
    
    Best Performance:
    • Best AUC: {best_auc:.4f} (Epoch {best_auc_epoch})
    
    Final Performance:
    • Final AUC: {final_auc:.4f}
    • Final Accuracy: {final_acc:.4f}
    • Final F1-Score: {final_f1:.4f}
    
    Training Progress:
    • Train Loss: {history['train_loss'][0]:.4f} → {history['train_loss'][-1]:.4f}
    • Val Loss: {history['val_loss'][0]:.4f} → {history['val_loss'][-1]:.4f}
    """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 总标题
    fig.suptitle(f'Bio-COT 3.0 Comprehensive Training Analysis (Best AUC: {best_auc:.4f})',
                 fontsize=18, fontweight='bold', y=0.995)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # 保存
    output_path = output_dir / f"comprehensive_analysis_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 综合可视化图表已保存: {output_path}")
    
    return best_auc, final_auc, final_acc, final_f1


def main():
    """主函数"""
    print("=" * 80)
    print("Bio-COT 3.0 结果可视化")
    print("=" * 80)
    
    # 加载历史
    history, timestamp = load_latest_history()
    if history is None:
        return
    
    # 配置
    config = BioCOT_v3_Config()
    output_dir = Path(config.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成可视化
    print("\n📊 正在生成综合可视化图表...")
    best_auc, final_auc, final_acc, final_f1 = visualize_comprehensive(history, timestamp, output_dir)
    
    # 打印总结
    print("\n" + "=" * 80)
    print("📈 训练结果总结")
    print("=" * 80)
    print(f"最佳AUC: {best_auc:.4f}")
    print(f"最终AUC: {final_auc:.4f}")
    print(f"最终准确率: {final_acc:.4f}")
    print(f"最终F1-Score: {final_f1:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()

