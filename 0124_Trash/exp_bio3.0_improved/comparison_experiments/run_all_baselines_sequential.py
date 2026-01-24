#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按顺序执行所有对比实验（包括SOTA和简单baseline）
避免显存不足，确保所有实验都能完成
"""

import sys
from pathlib import Path
import subprocess
import argparse
import time
from datetime import datetime

# 添加项目路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 所有对比实验配置（按顺序执行）
ALL_BASELINE_EXPERIMENTS = [
    # SOTA方法（优先级最高）
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
    # 简单baseline（作为参考）
    {
        'name': 'baseline_simple_fusion',
        'script': 'comparison_experiments/baselines/simple_fusion/train_simple_fusion.py',
        'description': 'Simple Fusion - 特征拼接 + MLP',
        'category': 'simple'
    },
    {
        'name': 'baseline_standard_clip',
        'script': 'comparison_experiments/baselines/standard_clip/train_standard_clip.py',
        'description': 'Standard CLIP - 标准CLIP方法',
        'category': 'simple'
    },
    {
        'name': 'baseline_vit_clinical_fusion',
        'script': 'comparison_experiments/baselines/vit_clinical_fusion/train_vit_clinical.py',
        'description': 'ViT + Clinical Fusion - ViT特征 + 临床特征融合',
        'category': 'simple'
    },
    # Backbone对比
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
    }
]


def run_experiment(exp_config, num_runs, num_epochs, batch_size, data_root, output_dir):
    """运行单个实验"""
    exp_name = exp_config['name']
    script_path = ROOT / exp_config['script']
    
    print(f"\n{'='*80}")
    print(f"开始运行: {exp_config['description']}")
    print(f"实验名称: {exp_name}")
    print(f"类别: {exp_config['category']}")
    print(f"脚本路径: {script_path}")
    print(f"{'='*80}\n")
    
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
    log_file = log_dir / f"{exp_name}_sequential_{timestamp}.log"
    
    print(f"日志文件: {log_file}")
    print(f"执行命令: {' '.join(cmd)}")
    print(f"\n开始执行...\n")
    sys.stdout.flush()
    
    # 执行命令
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
            print(f"\n✅ {exp_name} 执行成功！")
            print(f"   耗时: {elapsed_time/60:.1f} 分钟")
            return True
        else:
            print(f"\n❌ {exp_name} 执行失败，返回码: {return_code}")
            return False
            
    except Exception as e:
        print(f"\n❌ {exp_name} 执行出错: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='按顺序执行所有对比实验')
    parser.add_argument('--num_runs', type=int, default=3,
                       help='每个实验的运行次数')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (优化后的值)')
    parser.add_argument('--data_root', type=str,
                       default='/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal',
                       help='数据根目录')
    parser.add_argument('--output_dir', type=str,
                       default='comparison_experiments/results',
                       help='输出目录')
    parser.add_argument('--skip_completed', action='store_true',
                       help='跳过已完成的实验')
    parser.add_argument('--skip_sota', action='store_true',
                       help='跳过SOTA方法（如果已经运行）')
    
    args = parser.parse_args()
    
    # 过滤实验列表
    experiments = ALL_BASELINE_EXPERIMENTS
    if args.skip_sota:
        experiments = [exp for exp in experiments if exp['category'] != 'sota']
    
    print(f"\n{'='*80}")
    print(f"所有对比实验顺序执行")
    print(f"{'='*80}")
    print(f"实验总数: {len(experiments)}")
    print(f"  - SOTA方法: {len([e for e in experiments if e['category'] == 'sota'])}")
    print(f"  - 简单baseline: {len([e for e in experiments if e['category'] == 'simple'])}")
    print(f"  - Backbone对比: {len([e for e in experiments if e['category'] == 'backbone'])}")
    print(f"每个实验运行次数: {args.num_runs}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"{'='*80}\n")
    
    results = {}
    total_start_time = time.time()
    
    for idx, exp_config in enumerate(experiments, 1):
        exp_name = exp_config['name']
        
        print(f"\n{'#'*80}")
        print(f"# 实验 {idx}/{len(experiments)}: {exp_config['description']}")
        print(f"# 类别: {exp_config['category']}")
        print(f"{'#'*80}\n")
        
        # 检查是否已完成
        if args.skip_completed:
            result_dir = Path(args.output_dir) / exp_name / 'results'
            if (result_dir / 'all_results.json').exists():
                print(f"⏭️  跳过 {exp_name}（已完成）")
                results[exp_name] = 'skipped'
                continue
        
        # 运行实验
        success = run_experiment(
            exp_config,
            args.num_runs,
            args.num_epochs,
            args.batch_size,
            args.data_root,
            args.output_dir
        )
        
        results[exp_name] = 'success' if success else 'failed'
        
        # 等待GPU显存释放
        if idx < len(experiments):
            print(f"\n等待15秒，确保GPU显存释放...")
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(15)
    
    # 总结
    total_elapsed = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print(f"所有实验执行完成！")
    print(f"{'='*80}")
    print(f"总耗时: {total_elapsed/3600:.2f} 小时")
    print(f"\n实验结果:")
    for exp_name, status in results.items():
        status_icon = "✅" if status == 'success' else "⏭️" if status == 'skipped' else "❌"
        print(f"  {status_icon} {exp_name}: {status}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

