#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速检查实验执行状态和结果
"""

import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

# 添加项目路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_running_processes():
    """检查正在运行的实验进程"""
    import subprocess
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        lines = result.stdout.split('\n')
        running = []
        for line in lines:
            if any(x in line for x in ['train_simple_fusion', 'train_standard_clip', 'train_vit_clinical', 'run_and_visualize']):
                if 'grep' not in line:
                    running.append(line)
        return running
    except:
        return []


def check_results():
    """检查已生成的结果"""
    results_dir = ROOT / 'comparison_experiments' / 'results'
    if not results_dir.exists():
        return {}
    
    experiments = {}
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        results_file = exp_dir / 'results' / 'all_results.json'
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            experiments[exp_dir.name] = {
                'num_runs': len(data),
                'results': data
            }
    
    return experiments


def print_status():
    """打印状态信息"""
    print(f"\n{'='*80}")
    print(f"对比实验执行状态检查")
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # 检查运行中的进程
    print("📊 运行中的进程:")
    running = check_running_processes()
    if running:
        for line in running[:5]:  # 只显示前5个
            parts = line.split()
            if len(parts) > 10:
                print(f"  - PID {parts[1]}: {parts[10]} {parts[11] if len(parts) > 11 else ''}")
    else:
        print("  (无运行中的实验)")
    print()
    
    # 检查结果
    print("📁 已完成的实验:")
    results = check_results()
    if results:
        for exp_name, exp_data in results.items():
            num_runs = exp_data['num_runs']
            print(f"  ✅ {exp_name}: {num_runs} 次运行")
            
            # 计算平均指标
            if exp_data['results']:
                df = pd.DataFrame(exp_data['results'])
                if 'auc' in df.columns:
                    avg_auc = df['auc'].mean()
                    avg_acc = df['accuracy'].mean() if 'accuracy' in df.columns else 0
                    print(f"     平均 AUC: {avg_auc:.4f}, 平均 Acc: {avg_acc:.4f}")
    else:
        print("  (暂无完成的结果)")
    print()
    
    # 检查结果目录结构
    results_dir = ROOT / 'comparison_experiments' / 'results'
    if results_dir.exists():
        print("📂 结果目录结构:")
        for exp_dir in sorted(results_dir.iterdir()):
            if exp_dir.is_dir():
                results_subdir = exp_dir / 'results'
                if results_subdir.exists():
                    files = list(results_subdir.glob('*'))
                    print(f"  - {exp_dir.name}/")
                    for f in files:
                        size = f.stat().st_size / 1024  # KB
                        print(f"      {f.name} ({size:.1f} KB)")
    print()
    
    # 检查可视化
    vis_dir = ROOT / 'comparison_experiments' / 'visualizations'
    if vis_dir.exists():
        vis_files = list(vis_dir.glob('*.png'))
        if vis_files:
            print("🖼️  可视化文件:")
            for f in vis_files:
                size = f.stat().st_size / 1024  # KB
                print(f"  - {f.name} ({size:.1f} KB)")
        else:
            print("🖼️  可视化文件: (尚未生成)")
    else:
        print("🖼️  可视化文件: (目录不存在)")
    print()
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    print_status()

