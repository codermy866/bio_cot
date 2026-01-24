#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查baseline是否完成，完成后运行下一个消融实验
"""

import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
    'w/o_alignment_loss': 'ablation_studies/w/o_alignment_loss/config.py',
    'w/o_ot_loss': 'ablation_studies/w/o_ot_loss/config.py',
    'w/o_dual_head': 'ablation_studies/w/o_dual_head/config.py',
    'w/o_adaptive_gating': 'ablation_studies/w/o_adaptive_gating/config.py',
    'w/o_cross_attn': 'ablation_studies/w/o_cross_attn/config.py',
}


def is_experiment_completed(exp_id: str):
    """检查实验是否完成"""
    log_dir = ROOT / 'ablation_studies' / exp_id / 'logs'
    if not log_dir.exists():
        return False
    
    log_files = list(log_dir.glob('train_bio_cot_v3_*.log'))
    if not log_files:
        return False
    
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            content = f.read()
            if "训练完成" in content or "✅ 训练完成" in content:
                return True
            # 检查是否达到20个epoch
            import re
            epochs = re.findall(r'Epoch (\d+)/20', content)
            if epochs:
                max_epoch = max([int(e) for e in epochs])
                if max_epoch >= 20:
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
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nohup_file = log_dir / f"nohup_{exp_id}_{timestamp}.log"
    
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
        
        print(f"✅ 实验 '{exp_id}' 已启动 (PID: {process.pid})")
        print(f"   日志: {nohup_file}")
        return process.pid
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description='检查并运行下一个消融实验')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--experiment', type=str, default=None, help='指定要检查的实验')
    
    args = parser.parse_args()
    
    if args.experiment:
        exp_id = args.experiment
        if exp_id not in EXPERIMENTS:
            print(f"❌ 未知实验: {exp_id}")
            return
        
        if is_experiment_completed(exp_id):
            print(f"✅ 实验 '{exp_id}' 已完成")
        else:
            print(f"⏳ 实验 '{exp_id}' 尚未完成")
        return
    
    # 检查baseline是否完成
    print("=" * 80)
    print("🔍 检查Baseline实验状态")
    print("=" * 80)
    
    if is_experiment_completed('baseline'):
        print("✅ Baseline实验已完成！")
        
        # 找出下一个未完成的实验
        next_exp = None
        for exp_id in EXPERIMENTS[1:]:  # 跳过baseline
            if not is_experiment_completed(exp_id):
                next_exp = exp_id
                break
        
        if next_exp:
            print(f"\n🚀 启动下一个实验: {next_exp}")
            start_experiment(next_exp, args.gpu)
        else:
            print("\n✅ 所有实验已完成！")
    else:
        print("⏳ Baseline实验尚未完成，请等待...")
        print("\n💡 提示：运行以下命令手动检查：")
        print("   python ablation_studies/check_and_run_next.py")


if __name__ == '__main__':
    main()

