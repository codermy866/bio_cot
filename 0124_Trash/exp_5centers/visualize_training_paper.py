#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成适合论文发表的专业训练可视化图表
- 学术风格
- 高分辨率
- 清晰的布局
- 符合期刊要求
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置matplotlib后端（确保可以保存高质量图片）
matplotlib.use('Agg')

# 设置学术论文风格
plt.style.use('seaborn-v0_8-whitegrid')  # 使用seaborn的学术风格
matplotlib.rcParams.update({
    'font.family': 'serif',  # 使用serif字体（Times New Roman风格）
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'SimSun'],
    'font.size': 11,  # 论文常用字体大小
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.0,
    'lines.markersize': 5,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'savefig.dpi': 300,  # 高分辨率
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'pdf.fonttype': 42,  # 确保PDF中的文字可编辑
    'ps.fonttype': 42,
})


def visualize_training_paper(history: dict, log_dir: Path, timestamp: str, best_auc: float):
    """
    生成适合论文发表的专业训练可视化图表
    
    Args:
        history: 训练历史字典
        log_dir: 日志目录
        timestamp: 时间戳
        best_auc: 最佳AUC值
    """
    epochs = np.array(range(1, len(history['train_loss']) + 1))
    
    # 创建两个独立的图表：主图和补充图
    
    # ========== 主图：核心性能指标 ==========
    fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle('Bio-COT 2.0 Training Performance', fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Loss曲线（左上）
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2.5, alpha=0.8)
    ax1.plot(epochs, history['val_loss'], 'r--', label='Validation Loss', linewidth=2.5, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Loss Curves', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([1, len(epochs)])
    
    # 2. Accuracy曲线（右上）
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2.5, alpha=0.8)
    ax2.plot(epochs, history['val_acc'], 'r--', label='Validation Accuracy', linewidth=2.5, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Accuracy Curves', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([1, len(epochs)])
    ax2.set_ylim([0, 1])
    
    # 3. AUC曲线（左下）
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['val_auc'], 'g-', label='Validation AUC', linewidth=2.5, marker='o', 
             markersize=4, markevery=max(1, len(epochs)//20), alpha=0.8)
    ax3.axhline(y=best_auc, color='r', linestyle=':', linewidth=2, 
                label=f'Best AUC: {best_auc:.4f}', alpha=0.7)
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Area Under ROC Curve', fontsize=13, fontweight='bold', pad=10)
    ax3.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim([1, len(epochs)])
    ax3.set_ylim([0.5, 1.0])
    
    # 4. F1-Score曲线（右下）
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['val_f1'], 'm-', label='Validation F1-Score', linewidth=2.5, 
             marker='s', markersize=4, markevery=max(1, len(epochs)//20), alpha=0.8)
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax4.set_title('(d) F1-Score', fontsize=13, fontweight='bold', pad=10)
    ax4.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlim([1, len(epochs)])
    ax4.set_ylim([0, 1])
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # 保存主图
    plot_file_main = log_dir / f"training_curves_paper_main_{timestamp}.png"
    fig1.savefig(plot_file_main, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 主图已保存: {plot_file_main}")
    
    plot_file_main_pdf = log_dir / f"training_curves_paper_main_{timestamp}.pdf"
    fig1.savefig(plot_file_main_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 主图PDF已保存: {plot_file_main_pdf}")
    
    plt.close(fig1)
    
    # ========== 补充图：Loss组件分解 ==========
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Bio-COT 2.0 Loss Components Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Loss组件（线性刻度）
    ax5 = axes[0]
    ax5.plot(epochs, history['cls_loss'], 'b-', label='Classification Loss', linewidth=2.5, alpha=0.8)
    if history.get('ot_loss') and len(history['ot_loss']) > 0:
        ax5.plot(epochs, history['ot_loss'], 'g-', label='Optimal Transport Loss', linewidth=2.5, alpha=0.8)
    if history.get('consist_loss') and len(history['consist_loss']) > 0:
        ax5.plot(epochs, history['consist_loss'], 'orange', label='Consistency Loss', linewidth=2.5, alpha=0.8)
    if history.get('adv_loss') and len(history['adv_loss']) > 0:
        ax5.plot(epochs, history['adv_loss'], 'r-', label='Adversarial Loss', linewidth=2.5, alpha=0.8)
    ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
    ax5.set_title('(a) Loss Components (Linear Scale)', fontsize=13, fontweight='bold', pad=10)
    ax5.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.set_xlim([1, len(epochs)])
    
    # 2. Loss组件（对数刻度）
    ax6 = axes[1]
    ax6.plot(epochs, history['cls_loss'], 'b-', label='Classification Loss', linewidth=2.5, alpha=0.8)
    if history.get('ot_loss') and len(history['ot_loss']) > 0:
        ax6.plot(epochs, history['ot_loss'], 'g-', label='Optimal Transport Loss', linewidth=2.5, alpha=0.8)
    if history.get('consist_loss') and len(history['consist_loss']) > 0:
        ax6.plot(epochs, history['consist_loss'], 'orange', label='Consistency Loss', linewidth=2.5, alpha=0.8)
    if history.get('adv_loss') and len(history['adv_loss']) > 0:
        ax6.plot(epochs, history['adv_loss'], 'r-', label='Adversarial Loss', linewidth=2.5, alpha=0.8)
    ax6.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Loss Value (Log Scale)', fontsize=12, fontweight='bold')
    ax6.set_title('(b) Loss Components (Logarithmic Scale)', fontsize=13, fontweight='bold', pad=10)
    ax6.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.set_xlim([1, len(epochs)])
    ax6.set_yscale('log')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存补充图
    plot_file_supp = log_dir / f"training_curves_paper_supp_{timestamp}.png"
    fig2.savefig(plot_file_supp, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 补充图已保存: {plot_file_supp}")
    
    plot_file_supp_pdf = log_dir / f"training_curves_paper_supp_{timestamp}.pdf"
    fig2.savefig(plot_file_supp_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 补充图PDF已保存: {plot_file_supp_pdf}")
    
    plt.close(fig2)
    
    # ========== 综合性能对比图（可选） ==========
    fig3, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(epochs, history['val_auc'], 'g-', label='AUC', linewidth=2.5, marker='o', 
            markersize=5, markevery=max(1, len(epochs)//20), alpha=0.8)
    ax.plot(epochs, history['val_acc'], 'b-', label='Accuracy', linewidth=2.5, marker='s', 
            markersize=5, markevery=max(1, len(epochs)//20), alpha=0.8)
    ax.plot(epochs, history['val_f1'], 'm-', label='F1-Score', linewidth=2.5, marker='^', 
            markersize=5, markevery=max(1, len(epochs)//20), alpha=0.8)
    
    ax.axhline(y=best_auc, color='r', linestyle=':', linewidth=2, 
               label=f'Best AUC: {best_auc:.4f}', alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Bio-COT 2.0 Comprehensive Performance Metrics', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([1, len(epochs)])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # 保存综合图
    plot_file_comp = log_dir / f"training_curves_paper_comprehensive_{timestamp}.png"
    fig3.savefig(plot_file_comp, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 综合性能图已保存: {plot_file_comp}")
    
    plot_file_comp_pdf = log_dir / f"training_curves_paper_comprehensive_{timestamp}.pdf"
    fig3.savefig(plot_file_comp_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 综合性能图PDF已保存: {plot_file_comp_pdf}")
    
    plt.close(fig3)
    
    print(f"\n✅ 所有论文级可视化图表已生成！")
    print(f"   - 主图: {plot_file_main.name}")
    print(f"   - 补充图: {plot_file_supp.name}")
    print(f"   - 综合图: {plot_file_comp.name}")
    print(f"   - 最佳AUC: {best_auc:.4f}")


def parse_log_file(log_file: Path):
    """从日志文件中解析训练历史"""
    print(f"📖 正在解析日志文件: {log_file}")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_f1': [],
        'cls_loss': [],
        'ot_loss': [],
        'consist_loss': [],
        'adv_loss': []
    }
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        # 解析训练结果
        if 'Train - Loss:' in line and 'Acc:' in line:
            try:
                parts = line.split('Train - Loss:')[1].split(',')
                loss = float(parts[0].strip())
                acc = float(parts[1].split('Acc:')[1].strip())
                history['train_loss'].append(loss)
                history['train_acc'].append(acc)
            except:
                pass
        
        # 解析验证结果
        if 'Val   - Loss:' in line and 'AUC:' in line:
            try:
                parts = line.split('Val   - Loss:')[1]
                loss = float(parts.split(',')[0].strip())
                acc = float(parts.split('Acc:')[1].split(',')[0].strip())
                auc = float(parts.split('AUC:')[1].split(',')[0].strip())
                f1 = float(parts.split('F1:')[1].strip())
                history['val_loss'].append(loss)
                history['val_acc'].append(acc)
                history['val_auc'].append(auc)
                history['val_f1'].append(f1)
            except:
                pass
        
        # 解析Loss组件
        if 'Loss Components -' in line:
            try:
                parts = line.split('Loss Components -')[1].strip()
                if 'CLS:' in parts:
                    cls_loss = float(parts.split('CLS:')[1].strip())
                    history['cls_loss'].append(cls_loss)
                if 'OT:' in parts:
                    ot_loss = float(parts.split('OT:')[1].split(',')[0].strip())
                    history['ot_loss'].append(ot_loss)
                if 'Consist:' in parts:
                    consist_loss = float(parts.split('Consist:')[1].split(',')[0].strip())
                    history['consist_loss'].append(consist_loss)
                if 'Adv:' in parts:
                    adv_loss = float(parts.split('Adv:')[1].split(',')[0].strip())
                    history['adv_loss'].append(adv_loss)
            except:
                pass
    
    # 确保所有列表长度一致
    max_len = max(len(history['train_loss']), len(history['val_loss']))
    for key in history:
        if len(history[key]) < max_len:
            if len(history[key]) > 0:
                history[key].extend([history[key][-1]] * (max_len - len(history[key])))
            else:
                history[key] = [0.0] * max_len
    
    return history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成论文级训练可视化图表')
    parser.add_argument('--json', type=str, help='训练历史JSON文件路径')
    parser.add_argument('--log', type=str, help='训练日志文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    if args.log:
        log_file = Path(args.log)
        if not log_file.exists():
            print(f"❌ 日志文件不存在: {log_file}")
            sys.exit(1)
        history = parse_log_file(log_file)
        best_auc = max(history['val_auc']) if history['val_auc'] else 0.0
        timestamp = log_file.stem.replace('train_bio_cot_v2_', '').replace('_fixed', '')
        output_dir = Path(args.output) if args.output else log_file.parent
        visualize_training_paper(history, output_dir, timestamp, best_auc)
    elif args.json:
        json_file = Path(args.json)
        if not json_file.exists():
            print(f"❌ JSON文件不存在: {json_file}")
            sys.exit(1)
        with open(json_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        best_auc = max(history['val_auc']) if history['val_auc'] else 0.0
        timestamp = json_file.stem.replace('training_history_', '')
        output_dir = Path(args.output) if args.output else json_file.parent
        visualize_training_paper(history, output_dir, timestamp, best_auc)
    else:
        # 自动查找最新的文件
        log_dir = Path(__file__).parent / 'logs'
        log_files = list(log_dir.glob('train_bio_cot_v2*.log'))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            print(f"📁 自动找到最新的日志文件: {latest_log}")
            history = parse_log_file(latest_log)
            best_auc = max(history['val_auc']) if history['val_auc'] else 0.0
            timestamp = latest_log.stem.replace('train_bio_cot_v2_', '').replace('_fixed', '')
            visualize_training_paper(history, log_dir, timestamp, best_auc)
        else:
            print("❌ 未找到训练日志文件！")
            sys.exit(1)



