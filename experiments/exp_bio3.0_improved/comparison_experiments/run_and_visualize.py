#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动执行所有对比实验并生成可视化结果
"""

import sys
from pathlib import Path
import subprocess
import argparse
from datetime import datetime
import time

# 添加项目路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description='自动执行所有对比实验并生成可视化')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='每个实验的运行次数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--data_root', type=str,
                       default='/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal',
                       help='数据根目录')
    parser.add_argument('--skip_existing', action='store_true',
                       help='跳过已完成的实验')
    
    args = parser.parse_args()
    
    # 创建日志目录
    log_dir = ROOT / 'comparison_experiments' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_experiments_{timestamp}.log"
    
    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        with open(log_file, 'a', encoding='utf-8') as f:
            print(*args, **kwargs, file=f)
    
    log_print(f"\n{'='*80}")
    log_print(f"对比实验自动执行系统")
    log_print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"{'='*80}")
    log_print(f"实验配置:")
    log_print(f"  - 每个实验运行次数: {args.num_runs}")
    log_print(f"  - Batch Size: {args.batch_size}")
    log_print(f"  - 训练轮数: {args.num_epochs}")
    log_print(f"  - 数据根目录: {args.data_root}")
    log_print(f"{'='*80}\n")
    
    # 步骤1: 运行所有baseline实验
    log_print("\n" + "="*80)
    log_print("步骤1: 运行所有Baseline实验")
    log_print("="*80 + "\n")
    
    run_script = ROOT / 'comparison_experiments' / 'run_all_baselines.py'
    
    cmd = [
        'python', str(run_script),
        '--num_runs', str(args.num_runs),
        '--batch_size', str(args.batch_size),
        '--num_epochs', str(args.num_epochs),
        '--data_root', args.data_root,
    ]
    
    if args.skip_existing:
        cmd.append('--skip_existing')
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
        )
        
        log_print(result.stdout)
        elapsed_time = time.time() - start_time
        log_print(f"\n✅ 所有Baseline实验完成! 耗时: {elapsed_time/60:.2f} 分钟")
        
    except subprocess.CalledProcessError as e:
        log_print(f"\n❌ Baseline实验执行失败!")
        log_print(f"错误输出: {e.stdout}")
        log_print(f"返回码: {e.returncode}")
        return
    
    # 步骤2: 生成可视化结果
    log_print("\n" + "="*80)
    log_print("步骤2: 生成可视化结果")
    log_print("="*80 + "\n")
    
    visualize_script = ROOT / 'comparison_experiments' / 'visualize_results.py'
    
    cmd = [
        'python', str(visualize_script),
        '--results_dir', 'comparison_experiments/results',
        '--output_dir', 'comparison_experiments/visualizations'
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
        )
        
        log_print(result.stdout)
        elapsed_time = time.time() - start_time
        log_print(f"\n✅ 可视化结果生成完成! 耗时: {elapsed_time/60:.2f} 分钟")
        
    except subprocess.CalledProcessError as e:
        log_print(f"\n❌ 可视化生成失败!")
        log_print(f"错误输出: {e.stdout}")
        log_print(f"返回码: {e.returncode}")
    
    # 总结
    total_elapsed_time = time.time() - start_time
    log_print(f"\n{'='*80}")
    log_print(f"所有任务完成!")
    log_print(f"总耗时: {total_elapsed_time/60:.2f} 分钟")
    log_print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"日志文件: {log_file}")
    log_print(f"{'='*80}\n")
    
    print(f"\n✅ 所有任务完成! 日志已保存: {log_file}")


if __name__ == '__main__':
    main()

