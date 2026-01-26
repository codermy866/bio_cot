#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
等待训练完成，然后自动生成可视化
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import re
import json
import pandas as pd
from datetime import datetime
import shutil

# 配置路径
ROOT = Path(__file__).parent
VISUALIZATION_DIR = ROOT / 'visualization' / 'code'
OUTPUT_DIR = ROOT / 'newlog_0126'
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)
LOGS_DIR = ROOT / 'logs'

print("=" * 80)
print("⏳ 等待训练完成并生成可视化")
print("=" * 80)

# 读取日志文件路径
log_file_path = OUTPUT_DIR / 'log_file.txt'
if log_file_path.exists():
    with open(log_file_path, 'r') as f:
        log_file = Path(f.read().strip())
else:
    # 查找最新的日志文件
    log_files = sorted(LOGS_DIR.glob('train_bio_cot_v3.2_auto_*.log'), key=os.path.getmtime, reverse=True)
    if not log_files:
        print("❌ 未找到训练日志文件")
        sys.exit(1)
    log_file = log_files[0]

print(f"📝 监控日志: {log_file}")
print()

# 检查训练是否完成
def is_training_complete(log_path):
    """检查训练是否完成"""
    if not log_path.exists():
        return False
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            # 读取最后几行
            lines = f.readlines()
            if len(lines) < 10:
                return False
            
            last_lines = ''.join(lines[-50:])
            
            # 检查完成标志
            if any(keyword in last_lines for keyword in [
                '训练完成', 'Training completed', '✅ 所有epoch训练完成',
                'Best model saved', '训练结束'
            ]):
                return True
            
            # 检查是否到了最后一个epoch
            epoch_matches = re.findall(r'📊 Epoch (\d+)/(\d+)', last_lines)
            if epoch_matches:
                last_epoch, total_epochs = map(int, epoch_matches[-1])
                if last_epoch >= total_epochs - 1:
                    return True
    except:
        pass
    
    return False

# 等待训练完成
print("⏳ 等待训练完成...")
print("   (每60秒检查一次，按Ctrl+C可中断)")
print()

max_wait = 3600 * 8  # 最多8小时
check_interval = 60  # 每60秒检查
elapsed = 0

try:
    while elapsed < max_wait:
        if is_training_complete(log_file):
            print("\n✅ 训练已完成！")
            break
        
        time.sleep(check_interval)
        elapsed += check_interval
        
        if elapsed % 600 == 0:  # 每10分钟显示
            print(f"   已等待 {elapsed // 60} 分钟...")
    else:
        print("\n⚠️  等待超时，继续执行可视化...")
except KeyboardInterrupt:
    print("\n⚠️  等待被中断，继续执行可视化...")

print()

# ========================================
# 提取最佳结果并生成可视化
# ========================================
print("=" * 80)
print("📊 提取结果并生成可视化")
print("=" * 80)

# 运行监控和可视化脚本
monitor_script = ROOT / 'monitor_and_visualize_0126.py'
if monitor_script.exists():
    print(f"🚀 运行可视化脚本: {monitor_script}")
    result = subprocess.run(
        [sys.executable, str(monitor_script)],
        cwd=str(ROOT)
    )
    
    if result.returncode == 0:
        print("\n✅ 所有任务完成！")
        print(f"📁 查看结果: {OUTPUT_DIR}")
    else:
        print(f"\n⚠️  可视化脚本返回错误码: {result.returncode}")
else:
    print(f"❌ 未找到可视化脚本: {monitor_script}")

