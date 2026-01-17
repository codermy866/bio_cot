#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练监控脚本：实时查看训练进度和日志
"""

import sys
from pathlib import Path
import json
import time
from datetime import datetime

# 添加项目路径
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from config import BioCOT_v3_Config


def monitor_training():
    """监控训练进度"""
    config = BioCOT_v3_Config()
    log_dir = Path(config.log_dir)
    
    print("=" * 80)
    print("Bio-COT 3.0 训练监控")
    print("=" * 80)
    print(f"日志目录: {log_dir}")
    print(f"检查点目录: {config.checkpoint_dir}")
    print("=" * 80)
    
    # 查找最新的日志文件
    log_files = sorted(log_dir.glob("train_bio_cot_v3_*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
    history_files = sorted(log_dir.glob("training_history_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    checkpoint_files = sorted(Path(config.checkpoint_dir).glob("best_model_v3_*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if log_files:
        latest_log = log_files[0]
        print(f"\n📄 最新日志文件: {latest_log.name}")
        print(f"   修改时间: {datetime.fromtimestamp(latest_log.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   文件大小: {latest_log.stat().st_size / 1024:.2f} KB")
        
        # 显示最后20行
        print(f"\n📊 最新日志内容（最后20行）:")
        print("-" * 80)
        try:
            with open(latest_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[-20:]:
                    print(line.rstrip())
        except Exception as e:
            print(f"⚠️ 读取日志失败: {e}")
    
    if history_files:
        latest_history = history_files[0]
        print(f"\n📈 最新训练历史: {latest_history.name}")
        
        try:
            with open(latest_history, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            if history.get('val_auc'):
                epochs = len(history['val_auc'])
                best_auc = max(history['val_auc'])
                latest_auc = history['val_auc'][-1] if history['val_auc'] else 0
                latest_acc = history['val_acc'][-1] if history['val_acc'] else 0
                
                print(f"   已完成Epoch: {epochs}/{config.num_epochs}")
                print(f"   最佳AUC: {best_auc:.4f}")
                print(f"   最新AUC: {latest_auc:.4f}")
                print(f"   最新准确率: {latest_acc:.4f}")
        except Exception as e:
            print(f"⚠️ 读取历史失败: {e}")
    
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[0]
        print(f"\n💾 最新检查点: {latest_checkpoint.name}")
        print(f"   修改时间: {datetime.fromtimestamp(latest_checkpoint.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   文件大小: {latest_checkpoint.stat().st_size / 1024 / 1024:.2f} MB")
    
    print("\n" + "=" * 80)
    print("💡 提示: 使用 'tail -f <日志文件>' 实时查看训练日志")
    print("=" * 80)


if __name__ == '__main__':
    monitor_training()

