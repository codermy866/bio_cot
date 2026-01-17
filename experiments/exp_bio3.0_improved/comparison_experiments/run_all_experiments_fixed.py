#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复版：自动执行所有对比实验
确保所有实验都能正确运行并保存结果
"""

import sys
from pathlib import Path
import subprocess
import argparse
import json
from datetime import datetime
import time

# 添加项目路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 实验配置
BASELINE_EXPERIMENTS = [
    {
        'name': 'baseline_simple_fusion',
        'script': 'comparison_experiments/baselines/simple_fusion/train_simple_fusion.py',
        'description': 'Simple Fusion Baseline - 特征拼接 + MLP'
    },
    {
        'name': 'baseline_standard_clip',
        'script': 'comparison_experiments/baselines/standard_clip/train_standard_clip.py',
        'description': 'Standard CLIP Baseline - 标准CLIP方法'
    },
    {
        'name': 'baseline_vit_clinical_fusion',
        'script': 'comparison_experiments/baselines/vit_clinical_fusion/train_vit_clinical.py',
        'description': 'ViT + Clinical Fusion - ViT特征 + 临床特征融合'
    }
]


def run_experiment_direct(script_path, experiment_name, num_runs=5, batch_size=48, num_epochs=100, data_root=None):
    """直接运行实验脚本（不使用subprocess，避免输出问题）"""
    print(f"\n{'='*80}")
    print(f"开始运行实验: {experiment_name}")
    print(f"脚本: {script_path}")
    print(f"{'='*80}\n")
    
    # 检查脚本是否存在
    if not script_path.exists():
        print(f"❌ 脚本不存在: {script_path}")
        return False, 0.0
    
    start_time = time.time()
    
    try:
        # 直接导入并运行
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_script", script_path)
        module = importlib.util.module_from_spec(spec)
        
        # 设置命令行参数
        import sys
        original_argv = sys.argv
        sys.argv = [
            str(script_path),
            '--experiment_name', experiment_name,
            '--num_runs', str(num_runs),
            '--batch_size', str(batch_size),
            '--num_epochs', str(num_epochs),
        ]
        if data_root:
            sys.argv.extend(['--data_root', data_root])
        
        spec.loader.exec_module(module)
        
        # 恢复原始argv
        sys.argv = original_argv
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ 实验 {experiment_name} 完成! 耗时: {elapsed_time/60:.2f} 分钟")
        return True, elapsed_time
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ 实验 {experiment_name} 失败! 耗时: {elapsed_time/60:.2f} 分钟")
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed_time


def main():
    parser = argparse.ArgumentParser(description='自动执行所有对比实验（修复版）')
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
    
    # 创建结果目录
    results_dir = ROOT / 'comparison_experiments' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 执行日志
    execution_log = {
        'start_time': datetime.now().isoformat(),
        'config': {
            'num_runs': args.num_runs,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'data_root': args.data_root
        },
        'experiments': []
    }
    
    print(f"\n{'='*80}")
    print(f"对比实验自动执行系统（修复版）")
    print(f"{'='*80}")
    print(f"实验数量: {len(BASELINE_EXPERIMENTS)}")
    print(f"每个实验运行次数: {args.num_runs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"{'='*80}\n")
    
    total_start_time = time.time()
    
    for idx, exp in enumerate(BASELINE_EXPERIMENTS, 1):
        print(f"\n[{idx}/{len(BASELINE_EXPERIMENTS)}] {exp['description']}")
        
        # 检查是否已存在结果
        if args.skip_existing:
            exp_result_dir = results_dir / exp['name'] / 'results'
            if exp_result_dir.exists() and (exp_result_dir / 'all_results.json').exists():
                print(f"⏭️  跳过已存在的实验: {exp['name']}")
                execution_log['experiments'].append({
                    'name': exp['name'],
                    'status': 'skipped',
                    'reason': 'already_exists'
                })
                continue
        
        # 运行实验
        script_path = ROOT / exp['script']
        
        success, elapsed_time = run_experiment_direct(
            script_path,
            exp['name'],
            num_runs=args.num_runs,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            data_root=args.data_root
        )
        
        execution_log['experiments'].append({
            'name': exp['name'],
            'status': 'success' if success else 'failed',
            'elapsed_time': elapsed_time
        })
    
    total_elapsed_time = time.time() - total_start_time
    execution_log['end_time'] = datetime.now().isoformat()
    execution_log['total_elapsed_time'] = total_elapsed_time
    
    # 保存执行日志
    log_file = results_dir / 'execution_log_fixed.json'
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(execution_log, f, indent=2, ensure_ascii=False)
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"所有实验执行完成!")
    print(f"{'='*80}")
    print(f"总耗时: {total_elapsed_time/60:.2f} 分钟 ({total_elapsed_time/3600:.2f} 小时)")
    print(f"\n实验状态:")
    for exp_log in execution_log['experiments']:
        status_icon = '✅' if exp_log['status'] == 'success' else '⏭️' if exp_log['status'] == 'skipped' else '❌'
        print(f"  {status_icon} {exp_log['name']}: {exp_log['status']}")
    print(f"{'='*80}\n")
    
    print(f"执行日志已保存: {log_file}")
    print(f"结果目录: {results_dir}")


if __name__ == '__main__':
    main()

