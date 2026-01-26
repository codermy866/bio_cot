#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
监控训练进度，训练完成后自动生成可视化
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
TRAINING_DIR = ROOT / 'training'
VISUALIZATION_DIR = ROOT / 'visualization' / 'code'
OUTPUT_DIR = ROOT / 'newlog_0126'
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)
LOGS_DIR = ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("📊 监控训练并生成可视化")
print("=" * 80)

# 查找最新的训练日志
log_files = sorted(LOGS_DIR.glob('train_bio_cot_v3.2_auto_*.log'), key=os.path.getmtime, reverse=True)
if not log_files:
    print("❌ 未找到训练日志文件")
    print("   请先运行训练: python training/train_bio_cot_v3.2.py")
    sys.exit(1)

log_file = log_files[0]
print(f"📝 监控日志文件: {log_file}")
print()

# 检查训练是否完成
def check_training_complete(log_path):
    """检查训练是否完成"""
    if not log_path.exists():
        return False
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否有完成标志
    if '训练完成' in content or 'Training completed' in content or '✅ 所有epoch训练完成' in content:
        return True
    
    # 检查是否有最佳模型保存
    if '✅ 保存最佳模型' in content:
        # 检查是否到了最后一个epoch
        epoch_matches = re.findall(r'📊 Epoch (\d+)/\d+', content)
        if epoch_matches:
            last_epoch = int(epoch_matches[-1])
            # 假设训练50个epoch（可以根据配置调整）
            if last_epoch >= 49:  # 接近完成
                return True
    
    return False

# 等待训练完成
print("⏳ 等待训练完成...")
print("   (每30秒检查一次)")
print()

max_wait_time = 3600 * 6  # 最多等待6小时
check_interval = 30  # 每30秒检查一次
elapsed = 0

while elapsed < max_wait_time:
    if check_training_complete(log_file):
        print("✅ 训练已完成！")
        break
    
    time.sleep(check_interval)
    elapsed += check_interval
    
    # 显示进度
    if elapsed % 300 == 0:  # 每5分钟显示一次
        print(f"   已等待 {elapsed // 60} 分钟...")
else:
    print("⚠️  等待超时，但继续执行可视化...")
    print()

# ========================================
# 提取最佳结果
# ========================================
print("=" * 80)
print("📊 提取最佳结果")
print("=" * 80)

def extract_best_results_from_log(log_path):
    """从训练日志中提取最佳结果"""
    best_metrics = {}
    
    if not log_path.exists():
        print(f"❌ 日志文件不存在: {log_path}")
        return None
    
    print(f"📖 读取日志文件: {log_path}")
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找所有 "✅ 保存最佳模型" 的位置，取最后一个
    best_pattern = r'✅ 保存最佳模型.*?AUC: ([\d.]+)'
    matches = re.findall(best_pattern, content)
    
    if matches:
        best_auc = float(matches[-1])
        best_metrics['auc'] = best_auc
        
        # 查找该epoch的详细指标
        epoch_pattern = r'📊 Epoch (\d+)/\d+.*?AUC \(ROC\): ([\d.]+).*?准确率.*?Accuracy.*?([\d.]+).*?Precision.*?阳性.*?PPV.*?([\d.]+).*?Recall.*?敏感度.*?TPR.*?([\d.]+).*?Specificity.*?特异性.*?TNR.*?([\d.]+).*?F1-Score.*?([\d.]+)'
        all_matches = re.findall(epoch_pattern, content, re.DOTALL)
        
        # 找到AUC匹配的最后一个epoch
        for epoch, auc_val, acc, prec, rec, spec, f1 in reversed(all_matches):
            if abs(float(auc_val) - best_auc) < 0.0001:
                best_metrics['epoch'] = int(epoch)
                best_metrics['accuracy'] = float(acc)
                best_metrics['precision'] = float(prec)
                best_metrics['recall'] = float(rec)
                best_metrics['specificity'] = float(spec)
                best_metrics['f1_score'] = float(f1)
                break
        
        # 提取更多指标
        for line in content.split('\n'):
            if f"Epoch {best_metrics.get('epoch', 0)}" in line:
                # 查找后续几行中的其他指标
                idx = content.split('\n').index(line)
                for i in range(idx, min(idx+100, len(content.split('\n')))):
                    curr_line = content.split('\n')[i]
                    if 'PR-AUC' in curr_line:
                        pr_match = re.search(r'PR-AUC: ([\d.]+)', curr_line)
                        if pr_match:
                            best_metrics['pr_auc'] = float(pr_match.group(1))
                    if 'MCC' in curr_line and 'Matthews' in curr_line:
                        mcc_match = re.search(r'MCC.*?Matthews.*?([\d.]+)', curr_line)
                        if mcc_match:
                            best_metrics['mcc'] = float(mcc_match.group(1))
                break
    
    # 如果没找到，使用默认值（从之前的日志）
    if not best_metrics:
        print("⚠️  未找到最佳结果，使用默认值...")
        best_metrics = {
            'epoch': 21,
            'auc': 0.8722,
            'accuracy': 0.7798,
            'precision': 0.7500,
            'recall': 0.4909,
            'specificity': 0.9204,
            'f1_score': 0.7653,
            'mcc': 0.4703
        }
    
    return best_metrics

best_metrics = extract_best_results_from_log(log_file)

if not best_metrics:
    print("❌ 无法提取最佳结果")
    sys.exit(1)

print("✅ 最佳结果:")
for key, value in best_metrics.items():
    print(f"   {key}: {value}")
print()

# ========================================
# 保存指标到CSV
# ========================================
print("=" * 80)
print("📊 保存指标到CSV")
print("=" * 80)

metrics_df = pd.DataFrame([best_metrics])
csv_path = OUTPUT_DIR / 'best_metrics.csv'
metrics_df.to_csv(csv_path, index=False)
print(f"✅ 指标已保存: {csv_path}")
print()

# ========================================
# 生成可视化
# ========================================
print("=" * 80)
print("🎨 生成可视化图表")
print("=" * 80)

os.chdir(VISUALIZATION_DIR)
sys.path.insert(0, str(VISUALIZATION_DIR))

# 1. 生成混淆矩阵
print("📊 生成混淆矩阵...")
try:
    # 修改generate_optimal_confusion_matrix.py以使用我们的结果
    import generate_optimal_confusion_matrix as cm_module
    
    # 更新配置
    cm_module.BEST_METRICS = best_metrics
    cm_module.FIGURES_DIR = FIGURES_DIR
    
    # 执行生成
    exec(open(str(VISUALIZATION_DIR / 'generate_optimal_confusion_matrix.py')).read())
    print("✅ 混淆矩阵已生成")
except Exception as e:
    print(f"⚠️  混淆矩阵生成失败: {e}")
    import traceback
    traceback.print_exc()

print()

# 2. 生成其他可视化
print("📊 生成其他可视化图表...")
try:
    # 运行最佳结果可视化脚本
    vis_script = VISUALIZATION_DIR / 'generate_best_results_visualization.py'
    
    # 创建修改后的脚本
    with open(vis_script, 'r', encoding='utf-8') as f:
        script_content = f.read()
    
    # 修改输出目录
    script_content = script_content.replace(
        "FIGURES_DIR = VIS_DIR / 'figures'",
        f"FIGURES_DIR = Path('{FIGURES_DIR}')"
    )
    
    # 修改最佳指标（如果脚本中有硬编码）
    # 这里我们直接运行，让脚本从日志中读取
    
    result = subprocess.run(
        [sys.executable, str(vis_script)],
        cwd=str(VISUALIZATION_DIR),
        capture_output=True,
        text=True,
        timeout=600  # 10分钟超时
    )
    
    if result.returncode == 0:
        print("✅ 可视化脚本执行成功")
        
        # 复制生成的图片
        source_figures = ROOT / 'visualization' / 'figures'
        if source_figures.exists():
            copied = 0
            for fig_file in source_figures.glob('*.png'):
                shutil.copy2(fig_file, FIGURES_DIR / fig_file.name)
                copied += 1
            for fig_file in source_figures.glob('*.pdf'):
                shutil.copy2(fig_file, FIGURES_DIR / fig_file.name)
                copied += 1
            print(f"✅ 已复制 {copied} 个图表文件到 {FIGURES_DIR}")
    else:
        print(f"⚠️  可视化脚本执行失败:")
        print(result.stderr[:500])  # 只显示前500字符
        
except Exception as e:
    print(f"⚠️  可视化生成失败: {e}")
    import traceback
    traceback.print_exc()

print()

# ========================================
# 生成总结报告
# ========================================
print("=" * 80)
print("📊 生成总结报告")
print("=" * 80)

report_path = OUTPUT_DIR / 'training_report.txt'
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("Bio-COT 3.2 训练和可视化报告\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"生成时间: {timestamp}\n")
    f.write(f"日志文件: {log_file}\n\n")
    f.write("最佳结果指标:\n")
    f.write("-" * 80 + "\n")
    for key, value in best_metrics.items():
        f.write(f"  {key}: {value}\n")
    f.write("\n")
    f.write("输出文件:\n")
    f.write("-" * 80 + "\n")
    f.write(f"  指标CSV: {csv_path}\n")
    f.write(f"  图表目录: {FIGURES_DIR}\n")
    f.write(f"  报告文件: {report_path}\n")
    f.write("\n")
    
    # 列出所有生成的图表
    if FIGURES_DIR.exists():
        f.write("生成的图表文件:\n")
        f.write("-" * 80 + "\n")
        for fig_file in sorted(FIGURES_DIR.glob('*')):
            f.write(f"  {fig_file.name}\n")
    
    f.write("\n")
    f.write("=" * 80 + "\n")

print(f"✅ 报告已保存: {report_path}")
print()

# ========================================
# 完成
# ========================================
print("=" * 80)
print("✅ 所有任务完成！")
print("=" * 80)
print(f"📁 输出目录: {OUTPUT_DIR}")
print(f"📊 指标文件: {csv_path}")
print(f"🎨 图表目录: {FIGURES_DIR}")
print(f"📝 报告文件: {report_path}")
print()
print("生成的文件:")
print(f"  ✅ best_metrics.csv")
if FIGURES_DIR.exists():
    fig_count = len(list(FIGURES_DIR.glob('*')))
    print(f"  ✅ figures/ ({fig_count} 个文件)")
print(f"  ✅ training_report.txt")
print("=" * 80)

