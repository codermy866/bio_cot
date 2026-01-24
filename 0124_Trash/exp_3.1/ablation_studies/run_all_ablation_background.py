#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
后台运行所有消融实验的脚本
每个实验在后台运行，可以并行或顺序执行
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

# 添加父目录到路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 实验列表（按优先级排序）
EXPERIMENTS = [
    'baseline',
    'w/o_visual_notes',
    'w/o_alignment_loss',
    'w/o_ot_loss',
    'w/o_dual_head',
    'w/o_adaptive_gating',
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

CONFIG_PATHS = {
    'baseline': 'ablation_studies/baseline/config.py',
    'w/o_visual_notes': 'ablation_studies/w/o_visual_notes/config.py',
    'w/o_adaptive_gating': 'ablation_studies/w/o_adaptive_gating/config.py',
    'w/o_alignment_loss': 'ablation_studies/w/o_alignment_loss/config.py',
    'w/o_ot_loss': 'ablation_studies/w/o_ot_loss/config.py',
    'w/o_dual_head': 'ablation_studies/w/o_dual_head/config.py',
    'w/o_cross_attn': 'ablation_studies/w/o_cross_attn/config.py',
}


def run_experiment_background(exp_id: str, gpu: int = 0):
    """在后台运行单个实验"""
    config_path = ROOT / CONFIG_PATHS[exp_id]
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return None
    
    exp_name = EXPERIMENT_NAMES.get(exp_id, exp_id)
    print(f"\n{'='*80}")
    print(f"🚀 启动实验: {exp_name}")
    print(f"   配置: {config_path}")
    print(f"   GPU: {gpu}")
    print(f"{'='*80}")
    
    # 创建日志目录
    log_dir = ROOT / 'ablation_studies' / exp_id / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ablation_{exp_id}_{timestamp}.log"
    nohup_file = log_dir / f"nohup_{exp_id}_{timestamp}.log"
    
    # 构建命令
    train_script = ROOT / 'training' / 'train_bio_cot_v3.py'
    cmd = [
        sys.executable,
        str(train_script),
        '--config', str(config_path),
        '--gpu', str(gpu)
    ]
    
    # 后台运行
    try:
        with open(nohup_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(ROOT)
            )
        
        # 保存进程ID
        pid_file = log_dir / f"pid_{exp_id}_{timestamp}.txt"
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))
        
        print(f"✅ 实验 '{exp_name}' 已在后台启动")
        print(f"   PID: {process.pid}")
        print(f"   日志: {nohup_file}")
        print(f"   PID文件: {pid_file}")
        
        return process.pid
    except Exception as e:
        print(f"❌ 启动实验失败: {e}")
        return None


def check_experiment_status(exp_id: str):
    """检查实验状态"""
    log_dir = ROOT / 'ablation_studies' / exp_id / 'logs'
    if not log_dir.exists():
        return "未开始"
    
    # 查找最新的日志文件
    log_files = list(log_dir.glob('train_bio_cot_v3_*.log'))
    if not log_files:
        return "运行中（无日志）"
    
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    
    # 检查进程是否还在运行
    pid_files = list(log_dir.glob('pid_*.txt'))
    if pid_files:
        latest_pid_file = max(pid_files, key=lambda p: p.stat().st_mtime)
        try:
            with open(latest_pid_file, 'r') as f:
                pid = int(f.read().strip())
            # 检查进程是否存在
            try:
                os.kill(pid, 0)  # 不发送信号，只检查进程是否存在
                return f"运行中 (PID: {pid})"
            except OSError:
                return "已完成"
        except:
            pass
    
    # 尝试从日志判断
    try:
        with open(latest_log, 'r') as f:
            content = f.read()
            if "训练完成" in content or "✅ 训练完成" in content:
                return "已完成"
            elif "Epoch" in content:
                # 提取最后一个epoch
                import re
                epochs = re.findall(r'Epoch (\d+)/', content)
                if epochs:
                    last_epoch = max([int(e) for e in epochs])
                    return f"运行中 (Epoch {last_epoch})"
    except:
        pass
    
    return "运行中"


def main():
    import argparse
    parser = argparse.ArgumentParser(description='后台运行所有消融实验')
    parser.add_argument('--gpu', type=int, default=0, help='指定GPU ID')
    parser.add_argument('--sequential', action='store_true', 
                       help='顺序运行（等待前一个完成）而不是并行')
    parser.add_argument('--experiments', nargs='+', default=None,
                       help='指定要运行的实验（默认：全部）')
    parser.add_argument('--status', action='store_true',
                       help='只检查实验状态，不运行新实验')
    
    args = parser.parse_args()
    
    # 检查状态模式
    if args.status:
        print("=" * 80)
        print("📊 消融实验状态")
        print("=" * 80)
        experiments_to_check = args.experiments if args.experiments else EXPERIMENTS
        for exp_id in experiments_to_check:
            status = check_experiment_status(exp_id)
            exp_name = EXPERIMENT_NAMES.get(exp_id, exp_id)
            print(f"  {exp_name:30s} - {status}")
        print("=" * 80)
        return
    
    # 确定要运行的实验
    experiments_to_run = args.experiments if args.experiments else EXPERIMENTS
    
    print("=" * 80)
    print("🚀 开始运行消融实验")
    print("=" * 80)
    print(f"实验数量: {len(experiments_to_run)}")
    print(f"运行模式: {'顺序' if args.sequential else '并行'}")
    print(f"GPU: {args.gpu}")
    print("=" * 80)
    
    pids = {}
    
    for i, exp_id in enumerate(experiments_to_run):
        exp_name = EXPERIMENT_NAMES.get(exp_id, exp_id)
        print(f"\n[{i+1}/{len(experiments_to_run)}] 处理实验: {exp_name}")
        
        # 检查是否已经在运行
        status = check_experiment_status(exp_id)
        if "运行中" in status:
            print(f"⚠️  实验 '{exp_name}' 已在运行中，跳过")
            continue
        elif status == "已完成":
            print(f"✅ 实验 '{exp_name}' 已完成，跳过")
            continue
        
        # 运行实验
        pid = run_experiment_background(exp_id, args.gpu)
        if pid:
            pids[exp_id] = pid
        
        # 如果是顺序模式，等待完成
        if args.sequential and pid:
            print(f"\n⏳ 等待实验 '{exp_name}' 完成...")
            while True:
                time.sleep(60)  # 每分钟检查一次
                status = check_experiment_status(exp_id)
                if "已完成" in status or "运行中" not in status:
                    print(f"✅ 实验 '{exp_name}' 已完成")
                    break
                print(f"   状态: {status}")
        
        # 如果是并行模式，给一点时间让进程启动
        elif not args.sequential:
            time.sleep(10)  # 等待10秒再启动下一个
    
    # 打印总结
    print("\n" + "=" * 80)
    print("📊 实验启动总结")
    print("=" * 80)
    for exp_id, pid in pids.items():
        exp_name = EXPERIMENT_NAMES.get(exp_id, exp_id)
        print(f"  {exp_name:30s} - PID: {pid}")
    print("=" * 80)
    print("\n💡 提示:")
    print("   - 使用 'python run_all_ablation_background.py --status' 查看实验状态")
    print("   - 使用 'python compare_results.py' 对比实验结果")
    print("   - 日志文件保存在各自的 logs/ 目录中")


if __name__ == '__main__':
    main()

