#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
比较单模态（OCT）和多模态（OCT+Colposcopy+Clinical）的效果
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(result_dir):
    """加载实验结果"""
    result_files = list(Path(result_dir).glob('results_bio_cot*.json'))
    if not result_files:
        return None
    
    # 按时间排序，取最新的
    result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_file = result_files[0]
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def compare_results():
    """比较单模态和多模态结果"""
    # 单模态结果
    oct_only_dir = Path(__file__).parent / 'results'
    oct_results = load_results(oct_only_dir)
    
    # 多模态结果
    multimodal_dir = Path(__file__).parent / 'results_multimodal'
    multimodal_results = load_results(multimodal_dir)
    
    if oct_results is None:
        print("❌ 未找到单模态实验结果")
        return
    
    if multimodal_results is None:
        print("❌ 未找到多模态实验结果")
        print("   请先运行: python train_bio_cot_multimodal_xiangyang.py")
        return
    
    print("=" * 80)
    print("Bio-COT: 单模态 vs 多模态 效果对比")
    print("=" * 80)
    
    # 创建对比表格
    comparison = {
        '指标': ['验证集准确率', '验证集AUC', '验证集F1', '验证集F1(阳性类)'],
        '单模态 (OCT)': [
            f"{oct_results.get('final_val_acc', 0):.4f}",
            f"{oct_results.get('final_val_auc', 0):.4f}",
            f"{oct_results.get('final_val_f1', 0):.4f}",
            f"{oct_results.get('final_val_f1_pos', 0):.4f}"
        ],
        '多模态 (OCT+Colpo+Clinical)': [
            f"{multimodal_results.get('final_val_acc', 0):.4f}",
            f"{multimodal_results.get('final_val_auc', 0):.4f}",
            f"{multimodal_results.get('final_val_f1', 0):.4f}",
            f"{multimodal_results.get('final_val_f1_pos', 0):.4f}"
        ]
    }
    
    df = pd.DataFrame(comparison)
    print("\n" + df.to_string(index=False))
    
    # 计算改进幅度
    print("\n" + "=" * 80)
    print("改进幅度:")
    print("=" * 80)
    
    metrics = ['final_val_acc', 'final_val_auc', 'final_val_f1', 'final_val_f1_pos']
    metric_names = ['准确率', 'AUC', 'F1', 'F1(阳性类)']
    
    for metric, name in zip(metrics, metric_names):
        oct_val = oct_results.get(metric, 0)
        multi_val = multimodal_results.get(metric, 0)
        if oct_val > 0:
            improvement = ((multi_val - oct_val) / oct_val) * 100
            print(f"{name:15s}: {oct_val:.4f} → {multi_val:.4f} ({improvement:+.2f}%)")
        else:
            print(f"{name:15s}: {oct_val:.4f} → {multi_val:.4f} (N/A)")
    
    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 柱状图对比
    metrics_short = ['Acc', 'AUC', 'F1', 'F1+']
    oct_values = [
        oct_results.get('final_val_acc', 0),
        oct_results.get('final_val_auc', 0),
        oct_results.get('final_val_f1', 0),
        oct_results.get('final_val_f1_pos', 0)
    ]
    multi_values = [
        multimodal_results.get('final_val_acc', 0),
        multimodal_results.get('final_val_auc', 0),
        multimodal_results.get('final_val_f1', 0),
        multimodal_results.get('final_val_f1_pos', 0)
    ]
    
    x = range(len(metrics_short))
    width = 0.35
    
    axes[0].bar([i - width/2 for i in x], oct_values, width, label='单模态 (OCT)', alpha=0.8)
    axes[0].bar([i + width/2 for i in x], multi_values, width, label='多模态 (OCT+Colpo+Clinical)', alpha=0.8)
    axes[0].set_xlabel('指标')
    axes[0].set_ylabel('分数')
    axes[0].set_title('单模态 vs 多模态 性能对比')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_short)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # 改进幅度图
    improvements = [(m - o) / o * 100 if o > 0 else 0 for o, m in zip(oct_values, multi_values)]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    axes[1].bar(metrics_short, improvements, color=colors, alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('指标')
    axes[1].set_ylabel('改进幅度 (%)')
    axes[1].set_title('多模态相对单模态的改进')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / 'results_multimodal' / 'comparison_oct_vs_multimodal.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 对比图已保存: {output_path}")
    
    # 保存对比结果
    comparison_json = {
        'oct_only': oct_results,
        'multimodal': multimodal_results,
        'improvements': {
            metric: {
                'absolute': multi_val - oct_val,
                'relative_percent': ((multi_val - oct_val) / oct_val * 100) if oct_val > 0 else 0
            }
            for metric, oct_val, multi_val in zip(metrics, oct_values, multi_values)
        }
    }
    
    comparison_path = Path(__file__).parent / 'results_multimodal' / 'comparison_results.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison_json, f, indent=2)
    
    print(f"✅ 对比结果已保存: {comparison_path}")

if __name__ == '__main__':
    compare_results()

