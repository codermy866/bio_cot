#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比所有消融实验的结果
生成对比表格和可视化图表
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import re

# 添加父目录到路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 实验列表
EXPERIMENTS = [
    'baseline',
    'w/o_visual_notes',
    'w/o_adaptive_gating',
    'w/o_alignment_loss',
    'w/o_ot_loss',
    'w/o_dual_head',
    'w/o_cross_attn'
]

EXPERIMENT_NAMES = {
    'baseline': 'Baseline',
    'w/o_visual_notes': 'w/o Visual Notes',
    'w/o_adaptive_gating': 'w/o Adaptive Gating',
    'w/o_alignment_loss': 'w/o Alignment Loss',
    'w/o_ot_loss': 'w/o OT Loss',
    'w/o_dual_head': 'w/o Dual Head',
    'w/o_cross_attn': 'w/o Cross-Attention'
}


def parse_log_file(log_path: Path):
    """从日志文件中提取指标"""
    if not log_path.exists():
        return None
    
    metrics = {
        'best_auc': 0.0,
        'best_f1': 0.0,
        'best_recall': 0.0,
        'best_acc': 0.0,
        'final_auc': 0.0,
        'final_f1': 0.0,
        'final_recall': 0.0,
        'final_acc': 0.0,
        'total_epochs': 0
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取验证统计信息
        val_stats_pattern = r'Epoch (\d+) 验证统计:.*?AUC: ([\d.]+).*?对齐召回率.*?([\d.]+).*?F1-Score: ([\d.]+).*?准确率: ([\d.]+)'
        matches = re.findall(val_stats_pattern, content, re.DOTALL)
        
        if matches:
            # 获取最后一个epoch的结果
            last_match = matches[-1]
            metrics['total_epochs'] = int(last_match[0])
            metrics['final_auc'] = float(last_match[1])
            metrics['final_recall'] = float(last_match[2])
            metrics['final_f1'] = float(last_match[3])
            metrics['final_acc'] = float(last_match[4])
            
            # 找到最佳AUC
            best_auc = 0.0
            best_idx = 0
            for i, match in enumerate(matches):
                auc = float(match[1])
                if auc > best_auc:
                    best_auc = auc
                    best_idx = i
            
            if best_idx < len(matches):
                best_match = matches[best_idx]
                metrics['best_auc'] = float(best_match[1])
                metrics['best_recall'] = float(best_match[2])
                metrics['best_f1'] = float(best_match[3])
                metrics['best_acc'] = float(best_match[4])
        
    except Exception as e:
        print(f"⚠️  解析日志文件失败 {log_path}: {e}")
        return None
    
    return metrics


def collect_all_results():
    """收集所有实验的结果"""
    results = []
    
    for exp_id in EXPERIMENTS:
        exp_name = EXPERIMENT_NAMES.get(exp_id, exp_id)
        log_dir = ROOT / 'ablation_studies' / exp_id / 'logs'
        
        if not log_dir.exists():
            print(f"⚠️  实验 '{exp_name}' 的日志目录不存在: {log_dir}")
            continue
        
        # 查找最新的日志文件
        log_files = list(log_dir.glob('train_bio_cot_v3_*.log'))
        if not log_files:
            print(f"⚠️  实验 '{exp_name}' 没有找到日志文件")
            continue
        
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        metrics = parse_log_file(latest_log)
        
        if metrics:
            results.append({
                'Experiment': exp_name,
                'Best AUC': f"{metrics['best_auc']:.4f}",
                'Best F1': f"{metrics['best_f1']:.4f}",
                'Best Recall@1': f"{metrics['best_recall']:.4f}",
                'Best Acc': f"{metrics['best_acc']:.4f}",
                'Final AUC': f"{metrics['final_auc']:.4f}",
                'Final F1': f"{metrics['final_f1']:.4f}",
                'Final Recall@1': f"{metrics['final_recall']:.4f}",
                'Final Acc': f"{metrics['final_acc']:.4f}",
                'Epochs': metrics['total_epochs']
            })
        else:
            print(f"⚠️  实验 '{exp_name}' 无法解析指标")
    
    return results


def print_comparison_table(results):
    """打印对比表格"""
    if not results:
        print("❌ 没有找到任何实验结果")
        return
    
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 100)
    print("📊 消融实验对比表")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    
    # 保存为CSV
    output_path = ROOT / 'ablation_studies' / 'comparison_results.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 结果已保存到: {output_path}")
    
    # 找出最佳结果
    print("\n" + "=" * 100)
    print("🏆 最佳结果")
    print("=" * 100)
    
    # 转换为数值以便排序
    df_numeric = df.copy()
    for col in ['Best AUC', 'Best F1', 'Best Recall@1', 'Best Acc']:
        df_numeric[col] = df_numeric[col].astype(float)
    
    best_auc_idx = df_numeric['Best AUC'].idxmax()
    best_f1_idx = df_numeric['Best F1'].idxmax()
    best_recall_idx = df_numeric['Best Recall@1'].idxmax()
    
    print(f"最佳 AUC:  {df.iloc[best_auc_idx]['Experiment']:30s} - {df.iloc[best_auc_idx]['Best AUC']}")
    print(f"最佳 F1:   {df.iloc[best_f1_idx]['Experiment']:30s} - {df.iloc[best_f1_idx]['Best F1']}")
    print(f"最佳 Recall@1: {df.iloc[best_recall_idx]['Experiment']:30s} - {df.iloc[best_recall_idx]['Best Recall@1']}")
    print("=" * 100)


def main():
    print("=" * 100)
    print("🔍 开始收集消融实验结果...")
    print("=" * 100)
    
    results = collect_all_results()
    
    if results:
        print_comparison_table(results)
    else:
        print("❌ 没有找到任何实验结果")
        print("   请确保已经运行了消融实验")


if __name__ == '__main__':
    main()

