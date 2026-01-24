#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""监控CLIP训练进度"""

import time
from pathlib import Path

def monitor_training():
    log_file = Path('causal_bayesian_clip_training.log')
    results_dir = Path('causal_bayesian_clip_results')
    
    print("🔍 监控CLIP训练...")
    print("=" * 60)
    
    if log_file.exists():
        lines = open(log_file).readlines()
        print(f"📄 日志文件: {log_file}")
        print(f"📊 总行数: {len(lines)}")
        print("\n📝 最后10行:")
        print("-" * 60)
        for line in lines[-10:]:
            print(line.rstrip())
    else:
        print("❌ 日志文件尚未创建")
    
    print("\n" + "=" * 60)
    
    if results_dir.exists():
        files = list(results_dir.glob('*'))
        print(f"📁 结果目录: {results_dir}")
        print(f"📊 文件数: {len(files)}")
        for f in files[:5]:
            print(f"  - {f.name}")
    else:
        print("📁 结果目录尚未创建")

if __name__ == '__main__':
    monitor_training()

