#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
只运行缺失的实验（SOTA方法和我们的方法）
避免重复执行已有结果的实验
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

# 所有可能的实验（需要检查）
ALL_POSSIBLE_EXPERIMENTS = [
    # SOTA方法
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
    # 简单Baseline
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
    },
]

# 我们的方法（如果需要）
OUR_METHOD = {
    'name': 'bio_cot_v3_improved',
    'script': 'training/train_bio_cot_v3.py',
    'description': 'Bio-COT 3.0 Improved - 我们的完整方法',
    'category': 'our_method'
}


def check_experiment_completed(exp_name):
    """检查实验是否已完成（支持多种路径格式）"""
    # 可能的路径格式
    possible_paths = [
        ROOT / 'comparison_experiments' / 'results' / exp_name / 'results' / 'all_results.json',
        ROOT / 'comparison_experiments' / 'results' / f"{exp_name}_test" / 'results' / 'all_results.json',
        ROOT / 'results' / exp_name / 'results' / 'all_results.json',
        ROOT / 'results' / f"{exp_name}_test" / 'results' / 'all_results.json',
    ]
    
    # 也搜索所有可能的位置
    import glob
    search_patterns = [
        str(ROOT / '**' / exp_name / '**' / 'all_results.json'),
        str(ROOT / '**' / f"{exp_name}_test" / '**' / 'all_results.json'),
    ]
    
    for pattern in search_patterns:
        for path in glob.glob(pattern, recursive=True):
            possible_paths.append(Path(path))
    
    for result_file in possible_paths:
        if result_file.exists():
            try:
                import json
                with open(result_file, 'r') as f:
                    data = json.load(f)
                if len(data) > 0:
                    print(f"  ✅ 找到结果: {result_file} ({len(data)} 次运行)")
                    return True
            except Exception as e:
                continue
    
    return False


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
    log_file = log_dir / f"{exp_name}_missing_{timestamp}.log"
    
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
    parser = argparse.ArgumentParser(description='只运行缺失的实验')
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
    parser.add_argument('--include_our_method', action='store_true',
                       help='包含我们的方法（Bio-COT 3.0 Improved）')
    
    args = parser.parse_args()
    
    # 检查所有实验，过滤已完成的
    print("\n" + "="*80)
    print("检查所有实验的完成状态...")
    print("="*80)
    
    experiments_to_run = []
    completed_experiments = []
    
    for exp in ALL_POSSIBLE_EXPERIMENTS:
        exp_name = exp['name']
        print(f"\n检查: {exp_name} ({exp['description']})")
        if check_experiment_completed(exp_name):
            completed_experiments.append(exp)
            print(f"  ⏭️  跳过（已完成）")
        else:
            # 也检查带_test后缀的版本
            test_name = f"{exp_name}_test"
            if check_experiment_completed(test_name):
                completed_experiments.append(exp)
                print(f"  ⏭️  跳过（已完成，在_test目录）")
            else:
                experiments_to_run.append(exp)
                print(f"  ❌ 需要运行")
    
    print("\n" + "="*80)
    print(f"已完成: {len(completed_experiments)} 个实验")
    print(f"需要运行: {len(experiments_to_run)} 个实验")
    print("="*80)
    
    # 如果需要，添加我们的方法
    if args.include_our_method and not check_experiment_completed(OUR_METHOD['name']):
        experiments_to_run.append(OUR_METHOD)
    
    if not experiments_to_run:
        print("\n✅ 所有实验都已完成，无需运行！")
        return
    
    print(f"\n{'='*80}")
    print(f"缺失实验顺序执行")
    print(f"{'='*80}")
    print(f"需要运行的实验数: {len(experiments_to_run)}")
    for exp in experiments_to_run:
        print(f"  - {exp['name']}: {exp['description']}")
    print(f"每个实验运行次数: {args.num_runs}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"{'='*80}\n")
    
    results = {}
    total_start_time = time.time()
    
    for idx, exp_config in enumerate(experiments_to_run, 1):
        exp_name = exp_config['name']
        
        print(f"\n{'#'*80}")
        print(f"# 实验 {idx}/{len(experiments_to_run)}: {exp_config['description']}")
        print(f"# 类别: {exp_config['category']}")
        print(f"{'#'*80}\n")
        
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
        if idx < len(experiments_to_run):
            print(f"\n等待15秒，确保GPU显存释放...")
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(15)
    
    # 总结
    total_elapsed = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print(f"所有缺失实验执行完成！")
    print(f"{'='*80}")
    print(f"总耗时: {total_elapsed/3600:.2f} 小时")
    print(f"\n实验结果:")
    for exp_name, status in results.items():
        status_icon = "✅" if status == 'success' else "❌"
        print(f"  {status_icon} {exp_name}: {status}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

