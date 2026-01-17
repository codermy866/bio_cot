#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动执行所有对比实验（Baselines）
"""

import sys
from pathlib import Path
import subprocess
import argparse
import json
from datetime import datetime
import time

# 添加项目路径
ROOT = Path(__file__).resolve().parents[1]  # exp_bio3.0_improved目录
sys.path.insert(0, str(ROOT))

# 实验配置
# 注意：现在包含SOTA方法和简单baseline的混合方案

# SOTA方法（优先级最高）
SOTA_BASELINE_EXPERIMENTS = [
    {
        'name': 'baseline_medclip',
        'script': 'comparison_experiments/baselines/sota_baselines/medclip/train_medclip.py',
        'description': 'MedCLIP - 医学领域专用CLIP模型',
        'priority': 'high',
        'category': 'sota'
    },
    {
        'name': 'baseline_convirt',
        'script': 'comparison_experiments/baselines/sota_baselines/convirt/train_convirt.py',
        'description': 'ConVIRT - 对比学习的医学Vision-Representation Transformer',
        'priority': 'high',
        'category': 'sota'
    },
    {
        'name': 'baseline_mmformer',
        'script': 'comparison_experiments/baselines/sota_baselines/mmformer/train_mmformer.py',
        'description': 'mmFormer - 多模态医学Transformer',
        'priority': 'high',
        'category': 'sota'
    },
    # 以下方法待实现
    # {
    #     'name': 'baseline_hifuse',
    #     'script': 'comparison_experiments/baselines/sota_baselines/hifuse/train_hifuse.py',
    #     'description': 'HiFuse - 层次多尺度特征融合',
    #     'priority': 'high',
    #     'category': 'sota'
    # },
    # {
    #     'name': 'baseline_m4oe',
    #     'script': 'comparison_experiments/baselines/sota_baselines/m4oe/train_m4oe.py',
    #     'description': 'M4oE - 医学多模态专家混合模型',
    #     'priority': 'high',
    #     'category': 'sota'
    # },
]

# 简单baseline（保留作为参考）
SIMPLE_BASELINE_EXPERIMENTS = [
    {
        'name': 'baseline_simple_fusion',
        'script': 'comparison_experiments/baselines/simple_fusion/train_simple_fusion.py',
        'description': 'Simple Fusion Baseline - 特征拼接 + MLP',
        'priority': 'medium',
        'category': 'simple'
    },
    {
        'name': 'baseline_standard_clip',
        'script': 'comparison_experiments/baselines/standard_clip/train_standard_clip.py',
        'description': 'Standard CLIP Baseline - 标准CLIP方法',
        'priority': 'medium',
        'category': 'simple'
    },
]

# 合并所有实验（SOTA优先）
BASELINE_EXPERIMENTS = SOTA_BASELINE_EXPERIMENTS + SIMPLE_BASELINE_EXPERIMENTS

# 我们的方法（Bio-COT 3.0）
OUR_METHOD = {
    'name': 'bio_cot_v3_improved',
    'script': 'training/train_bio_cot_v3.py',
    'description': 'Bio-COT 3.0 Improved - 我们的完整方法'
}


def run_experiment(script_path, experiment_name, num_runs=5, batch_size=48, num_epochs=100, data_root=None):
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"开始运行实验: {experiment_name}")
    print(f"脚本: {script_path}")
    print(f"{'='*80}\n")
    
    # 确保脚本路径是相对于ROOT的
    if script_path.is_absolute():
        script_rel_path = script_path.relative_to(ROOT)
    else:
        script_rel_path = script_path
    
    cmd = [
        'python', str(script_rel_path),
        '--experiment_name', experiment_name,
        '--num_runs', str(num_runs),
        '--batch_size', str(batch_size),
        '--num_epochs', str(num_epochs),
    ]
    
    if data_root:
        cmd.extend(['--data_root', data_root])
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ 实验 {experiment_name} 完成! 耗时: {elapsed_time/60:.2f} 分钟")
        return True, elapsed_time
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ 实验 {experiment_name} 失败! 耗时: {elapsed_time/60:.2f} 分钟")
        print(f"错误: {e}")
        return False, elapsed_time


def main():
    parser = argparse.ArgumentParser(description='自动执行所有对比实验')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='每个实验的运行次数')
    parser.add_argument('--batch_size', type=int, default=48,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--data_root', type=str,
                       default='/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal',
                       help='数据根目录')
    parser.add_argument('--include_our_method', action='store_true',
                       help='是否包含我们的方法（Bio-COT 3.0）')
    parser.add_argument('--skip_existing', action='store_true',
                       help='跳过已完成的实验')
    
    args = parser.parse_args()
    
    # 创建结果目录
    results_dir = ROOT / 'comparison_experiments' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 实验列表
    experiments = BASELINE_EXPERIMENTS.copy()
    if args.include_our_method:
        experiments.append(OUR_METHOD)
    
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
    print(f"对比实验自动执行系统")
    print(f"{'='*80}")
    print(f"实验数量: {len(experiments)}")
    print(f"每个实验运行次数: {args.num_runs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"{'='*80}\n")
    
    total_start_time = time.time()
    
    for idx, exp in enumerate(experiments, 1):
        print(f"\n[{idx}/{len(experiments)}] {exp['description']}")
        
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
        if not script_path.exists():
            # 尝试绝对路径
            script_path = Path(exp['script'])
            if not script_path.is_absolute():
                script_path = ROOT / exp['script']
            if not script_path.exists():
                print(f"❌ 脚本不存在: {script_path}")
                execution_log['experiments'].append({
                    'name': exp['name'],
                    'status': 'failed',
                    'reason': 'script_not_found'
                })
                continue
        
        success, elapsed_time = run_experiment(
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
    log_file = results_dir / 'execution_log.json'
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

