#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一运行消融实验的脚本
用法:
    python run_ablation.py --experiment baseline
    python run_ablation.py --experiment all  # 运行所有实验
    python run_ablation.py --list  # 列出所有可用实验
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 添加父目录到路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 实验列表（注意：由于文件夹名包含斜杠，实际路径是 w/o_xxx）
EXPERIMENTS = {
    'baseline': {
        'name': 'Baseline',
        'config': 'baseline/config.py',
        'description': '移除所有高级模块'
    },
    'w/o_visual_notes': {
        'name': 'w/o Visual Notes',
        'config': 'w/o_visual_notes/config.py',  # 实际路径: w/o_visual_notes/config.py
        'description': '移除增强型视觉笔记模块'
    },
    'w/o_adaptive_gating': {
        'name': 'w/o Adaptive Gating',
        'config': 'w/o_adaptive_gating/config.py',
        'description': '移除自适应模态门控'
    },
    'w/o_alignment_loss': {
        'name': 'w/o Alignment Loss',
        'config': 'w/o_alignment_loss/config.py',
        'description': '移除语义-视觉对齐损失'
    },
    'w/o_ot_loss': {
        'name': 'w/o OT Loss',
        'config': 'w/o_ot_loss/config.py',
        'description': '移除 Optimal Transport 损失'
    },
    'w/o_dual_head': {
        'name': 'w/o Dual Head',
        'config': 'w/o_dual_head/config.py',
        'description': '移除双头因果解耦模块'
    },
    'w/o_cross_attn': {
        'name': 'w/o Cross-Attention',
        'config': 'w/o_cross_attn/config.py',
        'description': '移除 Cross-Attention 机制'
    }
}


def list_experiments():
    """列出所有可用实验"""
    print("=" * 80)
    print("可用的消融实验:")
    print("=" * 80)
    for exp_id, exp_info in EXPERIMENTS.items():
        print(f"  {exp_id:25s} - {exp_info['name']:30s} ({exp_info['description']})")
    print("=" * 80)
    print("\n使用方法:")
    print("  python run_ablation.py --experiment <experiment_id>")
    print("  python run_ablation.py --experiment all  # 运行所有实验")
    print()


def run_experiment(exp_id: str, gpu: int = None):
    """运行单个实验"""
    if exp_id not in EXPERIMENTS:
        print(f"❌ 错误: 未知实验 '{exp_id}'")
        print(f"使用 --list 查看所有可用实验")
        return False
    
    exp_info = EXPERIMENTS[exp_id]
    config_path = ROOT / 'ablation_studies' / exp_info['config']
    
    if not config_path.exists():
        print(f"❌ 错误: 配置文件不存在: {config_path}")
        return False
    
    print("=" * 80)
    print(f"🚀 开始运行消融实验: {exp_info['name']}")
    print(f"   描述: {exp_info['description']}")
    print(f"   配置: {config_path}")
    print("=" * 80)
    
    # 切换到实验目录
    exp_dir = ROOT / 'ablation_studies' / Path(exp_info['config']).parent
    os.chdir(str(ROOT))
    
    # 构建训练命令
    train_script = ROOT / 'training' / 'train_bio_cot_v3.py'
    cmd = [
        sys.executable,
        str(train_script),
        '--config', str(config_path)
    ]
    
    if gpu is not None:
        cmd.extend(['--gpu', str(gpu)])
    
    # 运行训练
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ 实验 '{exp_info['name']}' 完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 实验 '{exp_info['name']}' 失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  实验 '{exp_info['name']}' 被用户中断")
        return False


def run_all_experiments(gpu: int = None):
    """运行所有实验"""
    print("=" * 80)
    print("🚀 开始运行所有消融实验")
    print("=" * 80)
    
    results = {}
    for exp_id in EXPERIMENTS.keys():
        print(f"\n{'='*80}")
        print(f"进度: {list(EXPERIMENTS.keys()).index(exp_id) + 1}/{len(EXPERIMENTS)}")
        print(f"{'='*80}\n")
        
        success = run_experiment(exp_id, gpu)
        results[exp_id] = success
        
        if not success:
            print(f"\n⚠️  实验 '{exp_id}' 失败，是否继续? (y/n): ", end='')
            choice = input().strip().lower()
            if choice != 'y':
                print("❌ 用户选择停止")
                break
    
    # 打印总结
    print("\n" + "=" * 80)
    print("📊 实验总结")
    print("=" * 80)
    for exp_id, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {EXPERIMENTS[exp_id]['name']:30s} - {status}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='运行消融实验')
    parser.add_argument('--experiment', '-e', type=str, 
                       help='实验ID (使用 --list 查看所有可用实验)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='列出所有可用实验')
    parser.add_argument('--all', '-a', action='store_true',
                       help='运行所有实验')
    parser.add_argument('--gpu', '-g', type=int, default=None,
                       help='指定GPU ID (默认: 自动选择)')
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    if args.all:
        run_all_experiments(args.gpu)
        return
    
    if args.experiment:
        if args.experiment == 'all':
            run_all_experiments(args.gpu)
        else:
            run_experiment(args.experiment, args.gpu)
    else:
        print("❌ 请指定实验ID或使用 --list 查看所有可用实验")
        print("   用法: python run_ablation.py --experiment <experiment_id>")
        print("   或:   python run_ablation.py --list")


if __name__ == '__main__':
    main()

