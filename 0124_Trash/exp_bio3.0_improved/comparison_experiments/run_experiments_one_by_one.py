#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按顺序一个一个执行实验，确保每个实验完成后再执行下一个
"""

import sys
from pathlib import Path
import subprocess
import argparse
import time
from datetime import datetime
import glob
import json

# 添加项目路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 所有需要运行的实验（按顺序）
EXPERIMENTS_TO_RUN = [
    {
        'name': 'baseline_medclip',
        'script': 'comparison_experiments/baselines/sota_baselines/medclip/train_medclip.py',
        'description': 'MedCLIP - 医学领域专用CLIP模型',
        'category': 'sota'
    },
    {
        'name': 'baseline_convirt',
        'script': 'comparison_experiments/baselines/sota_baselines/convirt/train_convirt.py',
        'description': 'ConVIRT - 对比学习的医学Vision-Representation Transformer',
        'category': 'sota'
    },
    {
        'name': 'baseline_mmformer',
        'script': 'comparison_experiments/baselines/sota_baselines/mmformer/train_mmformer.py',
        'description': 'mmFormer - 多模态医学Transformer',
        'category': 'sota'
    },
    {
        'name': 'baseline_cnn',
        'script': 'comparison_experiments/baselines/cnn_baseline/train_cnn.py',
        'description': 'CNN Baseline - CNN编码器 + 多模态融合',
        'category': 'backbone'
    },
    {
        'name': 'baseline_swin_t',
        'script': 'comparison_experiments/baselines/swin_t_baseline/train_swin.py',
        'description': 'Swin-T Baseline - Swin-T编码器 + 多模态融合',
        'category': 'backbone'
    },
    {
        'name': 'baseline_vmamba',
        'script': 'comparison_experiments/baselines/vmamba_baseline/train_vmamba.py',
        'description': 'VMamba Baseline - VMamba编码器 + 多模态融合',
        'category': 'backbone'
    },
]


def check_experiment_completed(exp_name):
    """检查实验是否已完成"""
    possible_paths = [
        ROOT / 'comparison_experiments' / 'results' / exp_name / 'results' / 'all_results.json',
        ROOT / 'comparison_experiments' / 'results' / f"{exp_name}_test" / 'results' / 'all_results.json',
    ]
    
    for pattern in [f'**/{exp_name}/**/all_results.json', f'**/{exp_name}_test/**/all_results.json']:
        for path in glob.glob(str(ROOT / pattern), recursive=True):
            possible_paths.append(Path(path))
    
    for result_file in possible_paths:
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                if len(data) > 0:
                    return True, result_file, len(data)
            except:
                continue
    return False, None, 0


def run_single_experiment(exp_config, num_runs, num_epochs, batch_size, data_root, output_dir):
    """运行单个实验，等待完成"""
    exp_name = exp_config['name']
    script_path = ROOT / exp_config['script']
    
    print(f"\n{'='*80}")
    print(f"实验 {exp_config['description']}")
    print(f"实验名称: {exp_name}")
    print(f"类别: {exp_config['category']}")
    print(f"{'='*80}\n")
    
    # 检查是否已完成
    completed, result_file, runs = check_experiment_completed(exp_name)
    if completed:
        print(f"✅ {exp_name} 已完成 ({runs} 次运行)")
        print(f"   结果文件: {result_file}")
        return True
    
    # 检查脚本是否存在
    if not script_path.exists():
        print(f"❌ 脚本不存在: {script_path}")
        return False
    
    # 构建命令
    cmd = [
        'python', '-u', str(script_path),
        '--experiment_name', exp_name,
        '--num_runs', str(num_runs),
        '--num_epochs', str(num_epochs),
        '--batch_size', str(batch_size),
        '--data_root', data_root,
        '--output_dir', output_dir
    ]
    
    # 创建日志文件
    log_dir = ROOT / 'comparison_experiments' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{exp_name}_one_by_one_{timestamp}.log"
    
    print(f"日志文件: {log_file}")
    print(f"执行命令: {' '.join(cmd)}")
    print(f"\n开始执行...\n")
    sys.stdout.flush()
    
    # 执行命令并等待完成
    start_time = time.time()
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 实时输出日志
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
                sys.stdout.flush()
            
            process.wait()
            return_code = process.returncode
        
        elapsed_time = time.time() - start_time
        
        if return_code == 0:
            # 再次检查结果文件
            completed, result_file, runs = check_experiment_completed(exp_name)
            if completed:
                print(f"\n✅ {exp_name} 执行成功！")
                print(f"   耗时: {elapsed_time/60:.1f} 分钟")
                print(f"   结果: {runs} 次运行")
                print(f"   结果文件: {result_file}")
                return True
            else:
                print(f"\n⚠️  {exp_name} 执行完成但未找到结果文件")
                return False
        else:
            print(f"\n❌ {exp_name} 执行失败，返回码: {return_code}")
            print(f"   请查看日志: {log_file}")
            return False
            
    except Exception as e:
        print(f"\n❌ {exp_name} 执行出错: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='按顺序一个一个执行实验')
    parser.add_argument('--num_runs', type=int, default=3,
                       help='每个实验的运行次数')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--data_root', type=str,
                       default='/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal',
                       help='数据根目录')
    parser.add_argument('--output_dir', type=str,
                       default='comparison_experiments/results',
                       help='输出目录')
    parser.add_argument('--start_from', type=int, default=0,
                       help='从第几个实验开始（0-based）')
    
    args = parser.parse_args()
    
    # 过滤已完成的实验
    experiments_to_run = []
    for exp in EXPERIMENTS_TO_RUN:
        completed, _, runs = check_experiment_completed(exp['name'])
        if completed:
            print(f"⏭️  跳过 {exp['name']}（已完成，{runs} 次运行）")
        else:
            experiments_to_run.append(exp)
    
    if args.start_from > 0:
        experiments_to_run = experiments_to_run[args.start_from:]
        print(f"\n从第 {args.start_from + 1} 个实验开始执行")
    
    if not experiments_to_run:
        print("\n✅ 所有实验都已完成，无需运行！")
        return
    
    print(f"\n{'='*80}")
    print(f"按顺序执行实验（一个一个来）")
    print(f"{'='*80}")
    print(f"需要运行的实验数: {len(experiments_to_run)}")
    for idx, exp in enumerate(experiments_to_run, 1):
        print(f"  {idx}. {exp['name']}: {exp['description']}")
    print(f"每个实验运行次数: {args.num_runs}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"{'='*80}\n")
    
    results = {}
    total_start_time = time.time()
    
    for idx, exp_config in enumerate(experiments_to_run, 1):
        exp_name = exp_config['name']
        
        print(f"\n{'#'*80}")
        print(f"# [{idx}/{len(experiments_to_run)}] {exp_config['description']}")
        print(f"{'#'*80}\n")
        
        # 运行实验（等待完成）
        success = run_single_experiment(
            exp_config,
            args.num_runs,
            args.num_epochs,
            args.batch_size,
            args.data_root,
            args.output_dir
        )
        
        results[exp_name] = 'success' if success else 'failed'
        
        # 等待GPU显存释放
        if idx < len(experiments_to_run):
            print(f"\n等待20秒，确保GPU显存完全释放...")
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(20)
            print("继续下一个实验...\n")
    
    # 总结
    total_elapsed = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print(f"所有实验执行完成！")
    print(f"{'='*80}")
    print(f"总耗时: {total_elapsed/3600:.2f} 小时")
    print(f"\n实验结果:")
    for exp_name, status in results.items():
        status_icon = "✅" if status == 'success' else "❌"
        print(f"  {status_icon} {exp_name}: {status}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

