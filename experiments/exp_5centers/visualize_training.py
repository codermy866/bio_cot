#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从训练历史JSON文件生成可视化图表
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def visualize_from_json(json_file: Path, output_dir: Path = None):
    """
    从JSON文件加载训练历史并生成可视化图表
    
    Args:
        json_file: 训练历史JSON文件路径
        output_dir: 输出目录（如果为None，则使用JSON文件所在目录）
    """
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    # 获取最佳AUC
    best_auc = max(history['val_auc']) if history['val_auc'] else 0.0
    
    # 确定输出目录
    if output_dir is None:
        output_dir = json_file.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取时间戳（从文件名提取）
    timestamp = json_file.stem.replace('training_history_', '')
    
    # 生成可视化
    visualize_training(history, output_dir, timestamp, best_auc)
    
    print(f"✅ 可视化完成！最佳AUC: {best_auc:.4f}")


def visualize_training(history: dict, log_dir: Path, timestamp: str, best_auc: float):
    """
    生成训练过程可视化图表
    
    Args:
        history: 训练历史字典
        log_dir: 日志目录
        timestamp: 时间戳
        best_auc: 最佳AUC值
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 创建图表
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Loss曲线
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy曲线
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. AUC曲线
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs, history['val_auc'], 'g-', label='Val AUC', linewidth=2, marker='o', markersize=4)
    ax3.axhline(y=best_auc, color='r', linestyle='--', linewidth=2, label=f'Best AUC: {best_auc:.4f}')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('AUC', fontsize=12)
    ax3.set_title('Validation AUC', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. F1-Score曲线
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(epochs, history['val_f1'], 'm-', label='Val F1', linewidth=2, marker='s', markersize=4)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('F1-Score', fontsize=12)
    ax4.set_title('Validation F1-Score', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # 5. Loss组件分解
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(epochs, history['cls_loss'], 'b-', label='Classification Loss', linewidth=2)
    if history.get('ot_loss') and len(history['ot_loss']) > 0:
        ax5.plot(epochs, history['ot_loss'], 'g-', label='OT Loss', linewidth=2)
    if history.get('consist_loss') and len(history['consist_loss']) > 0:
        ax5.plot(epochs, history['consist_loss'], 'orange', label='Consistency Loss', linewidth=2)
    if history.get('adv_loss') and len(history['adv_loss']) > 0:
        ax5.plot(epochs, history['adv_loss'], 'r-', label='Adversarial Loss', linewidth=2)
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Loss', fontsize=12)
    ax5.set_title('Loss Components', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')  # 使用对数刻度，因为不同loss的尺度可能差异很大
    
    # 6. 综合性能指标
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(epochs, history['val_auc'], 'g-', label='AUC', linewidth=2, marker='o', markersize=4)
    ax6.plot(epochs, history['val_acc'], 'b-', label='Accuracy', linewidth=2, marker='s', markersize=4)
    ax6.plot(epochs, history['val_f1'], 'm-', label='F1-Score', linewidth=2, marker='^', markersize=4)
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Score', fontsize=12)
    ax6.set_title('Comprehensive Performance Metrics', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1])
    
    # 添加总标题
    fig.suptitle(f'Bio-COT 2.0 Training Progress (Best AUC: {best_auc:.4f})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # 保存图表
    plot_file = log_dir / f"training_curves_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 可视化图表已保存到: {plot_file}")
    
    # 也保存PDF版本（矢量图，更清晰）
    plot_file_pdf = log_dir / f"training_curves_{timestamp}.pdf"
    plt.savefig(plot_file_pdf, bbox_inches='tight')
    print(f"📊 PDF版本已保存到: {plot_file_pdf}")
    
    plt.close()


def parse_log_file(log_file: Path):
    """
    从日志文件中解析训练历史（如果JSON文件不存在）
    
    Args:
        log_file: 日志文件路径
    """
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
    
    for i, line in enumerate(lines):
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
                # 解析各个loss组件
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
            # 用最后一个值填充
            if len(history[key]) > 0:
                history[key].extend([history[key][-1]] * (max_len - len(history[key])))
            else:
                history[key] = [0.0] * max_len
    
    return history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化训练过程')
    parser.add_argument('--json', type=str, help='训练历史JSON文件路径')
    parser.add_argument('--log', type=str, help='训练日志文件路径（如果JSON不存在）')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    if args.json:
        json_file = Path(args.json)
        if not json_file.exists():
            print(f"❌ JSON文件不存在: {json_file}")
            sys.exit(1)
        visualize_from_json(json_file, Path(args.output) if args.output else None)
    elif args.log:
        log_file = Path(args.log)
        if not log_file.exists():
            print(f"❌ 日志文件不存在: {log_file}")
            sys.exit(1)
        history = parse_log_file(log_file)
        best_auc = max(history['val_auc']) if history['val_auc'] else 0.0
        timestamp = log_file.stem.replace('train_bio_cot_v2_', '').replace('_fixed', '')
        output_dir = Path(args.output) if args.output else log_file.parent
        visualize_training(history, output_dir, timestamp, best_auc)
        print(f"✅ 可视化完成！最佳AUC: {best_auc:.4f}")
    else:
        # 自动查找最新的训练历史文件
        log_dir = Path(__file__).parent / 'logs'
        json_files = list(log_dir.glob('training_history_*.json'))
        if json_files:
            latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
            print(f"📁 自动找到最新的训练历史: {latest_json}")
            visualize_from_json(latest_json)
        else:
            log_files = list(log_dir.glob('train_bio_cot_v2*.log'))
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                print(f"📁 自动找到最新的日志文件: {latest_log}")
                history = parse_log_file(latest_log)
                best_auc = max(history['val_auc']) if history['val_auc'] else 0.0
                timestamp = latest_log.stem.replace('train_bio_cot_v2_', '').replace('_fixed', '')
                visualize_training(history, log_dir, timestamp, best_auc)
                print(f"✅ 可视化完成！最佳AUC: {best_auc:.4f}")
            else:
                print("❌ 未找到训练历史文件或日志文件！")
                sys.exit(1)



