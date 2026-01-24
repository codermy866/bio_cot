#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练监控脚本：实时监控Bio-COT 4.0训练进度
"""

import os
import time
import subprocess
from pathlib import Path
from datetime import datetime

def get_latest_log_file(log_dir):
    """获取最新的日志文件"""
    log_path = Path(log_dir)
    if not log_path.exists():
        return None
    
    log_files = sorted(log_path.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
    if log_files:
        return log_files[0]
    return None

def get_gpu_usage():
    """获取GPU使用情况"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return "N/A"

def monitor_training(log_dir="logs", check_interval=60):
    """监控训练进度"""
    print("=" * 80)
    print("🔍 Bio-COT 4.0 训练监控")
    print("=" * 80)
    print(f"日志目录: {log_dir}")
    print(f"检查间隔: {check_interval}秒")
    print("=" * 80)
    print("\n按 Ctrl+C 退出监控\n")
    
    last_epoch = -1
    best_auc = 0.0
    
    try:
        while True:
            log_file = get_latest_log_file(log_dir)
            
            if log_file and log_file.exists():
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    
                # 查找最新的epoch信息
                current_epoch = -1
                current_auc = 0.0
                current_loss = 0.0
                current_acc = 0.0
                
                for line in reversed(lines[-100:]):  # 只看最后100行
                    if "Epoch" in line and "训练统计" in line:
                        try:
                            # 提取epoch号
                            import re
                            match = re.search(r'Epoch (\d+)', line)
                            if match:
                                current_epoch = int(match.group(1))
                        except:
                            pass
                    
                    if "AUC:" in line and "Epoch" in line:
                        try:
                            import re
                            match = re.search(r'AUC: ([\d.]+)', line)
                            if match:
                                current_auc = float(match.group(1))
                                if current_auc > best_auc:
                                    best_auc = current_auc
                        except:
                            pass
                    
                    if "平均损失:" in line:
                        try:
                            import re
                            match = re.search(r'平均损失: ([\d.]+)', line)
                            if match:
                                current_loss = float(match.group(1))
                        except:
                            pass
                    
                    if "准确率:" in line and "Epoch" in line:
                        try:
                            import re
                            match = re.search(r'准确率: ([\d.]+)', line)
                            if match:
                                current_acc = float(match.group(1))
                        except:
                            pass
                
                if current_epoch > last_epoch:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 📊 训练进度更新:")
                    print(f"   当前Epoch: {current_epoch}")
                    if current_loss > 0:
                        print(f"   当前损失: {current_loss:.6f}")
                    if current_acc > 0:
                        print(f"   当前准确率: {current_acc:.4f}")
                    if current_auc > 0:
                        print(f"   当前AUC: {current_auc:.4f}")
                    print(f"   最佳AUC: {best_auc:.4f}")
                    
                    # GPU使用情况
                    gpu_info = get_gpu_usage()
                    if "N/A" not in gpu_info:
                        print(f"   GPU状态:\n{gpu_info}")
                    
                    last_epoch = current_epoch
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ 等待训练日志文件...")
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n✅ 监控已停止")

if __name__ == "__main__":
    import sys
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "logs"
    monitor_training(log_dir)

