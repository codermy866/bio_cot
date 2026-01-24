#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
监控消融实验进度，自动启动下一个实验
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 实验列表（按顺序）
EXPERIMENTS = [
    'baseline',
    'w/o_visual_notes',
    'w/o_alignment_loss',
    'w/o_ot_loss',
    'w/o_dual_head',
    'w/o_adaptive_gating',
    'w/o_cross_attn'
]

CONFIG_PATHS = {
    'baseline': 'ablation_studies/baseline/config.py',
    'w/o_visual_notes': 'ablation_studies/w/o_visual_notes/config.py',
    'w/o_adaptive_gating': 'ablation_studies/w/o_adaptive_gating/config.py',
    'w/o_alignment_loss': 'ablation_studies/w/o_alignment_loss/config.py',
    'w/o_ot_loss': 'ablation_studies/w/o_ot_loss/config.py',
    'w/o_dual_head': 'ablation_studies/w/o_dual_head/config.py',
    'w/o_cross_attn': 'ablation_studies/w/o_cross_attn/config.py',
}


def is_experiment_running(exp_id: str):
    """检查实验是否正在运行"""
    log_dir = ROOT / 'ablation_studies' / exp_id / 'logs'
    if not log_dir.exists():
        return False
    
    # 检查PID文件
    pid_files = list(log_dir.glob('pid_*.txt'))
    for pid_file in pid_files:
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            # 检查进程是否存在
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                pass
        except:
            pass
    
    return False


def is_experiment_completed(exp_id: str):
    """检查实验是否已完成"""
    log_dir = ROOT / 'ablation_studies' / exp_id / 'logs'
    if not log_dir.exists():
        return False
    
    # 查找训练日志
    log_files = list(log_dir.glob('train_bio_cot_v3_*.log'))
    if not log_files:
        return False
    
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            content = f.read()
            # 检查是否有完成标记
            if "训练完成" in content or "✅ 训练完成" in content:
                return True
            # 检查是否达到最大epoch
            epochs = re.findall(r'Epoch (\d+)/100', content)
            if epochs:
                max_epoch = max([int(e) for e in epochs])
                if max_epoch >= 100:
                    return True
    except:
        pass
    
    return False


def start_experiment(exp_id: str, gpu: int = 0):
    """启动实验"""
    config_path = ROOT / CONFIG_PATHS[exp_id]
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return None
    
    log_dir = ROOT / 'ablation_studies' / exp_id / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nohup_file = log_dir / f"nohup_{exp_id}_{timestamp}.log"
    pid_file = log_dir / f"pid_{exp_id}_{timestamp}.txt"
    
    train_script = ROOT / 'training' / 'train_bio_cot_v3.py'
    cmd = [
        sys.executable,
        str(train_script),
        '--config', str(config_path),
        '--gpu', str(gpu)
    ]
    
    try:
        with open(nohup_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(ROOT)
            )
        
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))
        
        print(f"✅ 实验 '{exp_id}' 已启动 (PID: {process.pid})")
        return process.pid
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return None


def get_experiment_status(exp_id: str):
    """获取实验状态"""
    if is_experiment_running(exp_id):
        return "运行中"
    elif is_experiment_completed(exp_id):
        return "已完成"
    else:
        return "未开始"


def main():
    import argparse
    parser = argparse.ArgumentParser(description='监控并自动运行消融实验')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--check-only', action='store_true', help='只检查状态，不启动新实验')
    parser.add_argument('--interval', type=int, default=300, help='检查间隔（秒，默认300）')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔍 消融实验监控系统")
    print("=" * 80)
    
    while True:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 检查实验状态...")
        print("-" * 80)
        
        all_completed = True
        current_running = None
        
        for i, exp_id in enumerate(EXPERIMENTS):
            status = get_experiment_status(exp_id)
            print(f"  [{i+1}/{len(EXPERIMENTS)}] {exp_id:25s} - {status}")
            
            if status == "运行中":
                all_completed = False
                current_running = exp_id
            elif status == "未开始":
                all_completed = False
                # 检查前面的实验是否都完成了
                prev_all_done = all(get_experiment_status(e) == "已完成" 
                                  for e in EXPERIMENTS[:i])
                if prev_all_done and not args.check_only:
                    print(f"\n🚀 前面的实验已完成，启动: {exp_id}")
                    start_experiment(exp_id, args.gpu)
                    break
        
        if all_completed:
            print("\n" + "=" * 80)
            print("✅ 所有消融实验已完成！")
            print("=" * 80)
            print("\n💡 运行以下命令查看结果对比:")
            print("   python ablation_studies/compare_results.py")
            break
        
        if args.check_only:
            break
        
        print(f"\n⏳ 等待 {args.interval} 秒后再次检查...")
        time.sleep(args.interval)


if __name__ == '__main__':
    main()

