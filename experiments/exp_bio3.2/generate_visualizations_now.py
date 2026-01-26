#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
立即生成训练结果的可视化
"""

import os
import sys
import re
import pandas as pd
from pathlib import Path
import json
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
print("🎨 生成训练结果可视化")
print("=" * 80)

# 查找训练日志
log_file = LOGS_DIR / 'train_bio_cot_v3.2_final_20260126_111511.log'
if not log_file.exists():
    # 查找最新的日志
    log_files = sorted(LOGS_DIR.glob('train_bio_cot_v3.2_*.log'), key=os.path.getmtime, reverse=True)
    if log_files:
        log_file = log_files[0]
    else:
        print("❌ 未找到训练日志文件")
        sys.exit(1)

print(f"📝 读取日志: {log_file}")
print()

# ========================================
# 提取最佳结果
# ========================================
print("=" * 80)
print("📊 提取最佳结果")
print("=" * 80)

def extract_best_results(log_path):
    """从日志中提取最佳结果"""
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    best_metrics = {}
    
    # 查找最佳AUC
    best_auc_match = re.search(r'✅ 训练完成！最佳AUC: ([\d.]+)', content)
    if best_auc_match:
        best_auc = float(best_auc_match.group(1))
        best_metrics['auc'] = best_auc
        
        # 查找所有保存最佳模型的位置
        best_saves = re.findall(r'✅ 保存最佳模型.*?Epoch (\d+).*?AUC: ([\d.]+)', content, re.DOTALL)
        
        # 找到AUC匹配的epoch
        best_epoch = None
        for epoch, auc_val in best_saves:
            if abs(float(auc_val) - best_auc) < 0.0001:
                best_epoch = int(epoch)
                break
        
        if best_epoch:
            # 查找该epoch的详细指标
            epoch_pattern = rf'📊 Epoch {best_epoch}/\d+.*?验证阶段.*?准确率.*?Accuracy.*?([\d.]+).*?Precision.*?阳性.*?PPV.*?([\d.]+).*?Recall.*?敏感度.*?TPR.*?([\d.]+).*?Specificity.*?特异性.*?TNR.*?([\d.]+).*?F1-Score.*?([\d.]+)'
            epoch_match = re.search(epoch_pattern, content, re.DOTALL)
            
            if epoch_match:
                best_metrics['epoch'] = best_epoch
                best_metrics['accuracy'] = float(epoch_match.group(1))
                best_metrics['precision'] = float(epoch_match.group(2))
                best_metrics['recall'] = float(epoch_match.group(3))
                best_metrics['specificity'] = float(epoch_match.group(4))
                best_metrics['f1_score'] = float(epoch_match.group(5))
        
        # 查找PR-AUC和MCC
        pr_auc_match = re.search(rf'📊 Epoch {best_epoch}/\d+.*?PR-AUC: ([\d.]+)', content, re.DOTALL)
        if pr_auc_match:
            best_metrics['pr_auc'] = float(pr_auc_match.group(1))
        
        mcc_match = re.search(rf'📊 Epoch {best_epoch}/\d+.*?MCC.*?Matthews.*?([\d.]+)', content, re.DOTALL)
        if mcc_match:
            best_metrics['mcc'] = float(mcc_match.group(1))
    
    # 如果没找到，尝试从最后几个epoch中提取
    if not best_metrics:
        print("⚠️  使用最后epoch的结果...")
        last_epoch_match = re.search(r'📊 Epoch (\d+)/\d+.*?验证阶段.*?准确率.*?Accuracy.*?([\d.]+).*?AUC \(ROC\): ([\d.]+).*?Precision.*?阳性.*?PPV.*?([\d.]+).*?Recall.*?敏感度.*?TPR.*?([\d.]+).*?Specificity.*?特异性.*?TNR.*?([\d.]+).*?F1-Score.*?([\d.]+)', content[-50000:], re.DOTALL)
        if last_epoch_match:
            best_metrics['epoch'] = int(last_epoch_match.group(1))
            best_metrics['accuracy'] = float(last_epoch_match.group(2))
            best_metrics['auc'] = float(last_epoch_match.group(3))
            best_metrics['precision'] = float(last_epoch_match.group(4))
            best_metrics['recall'] = float(last_epoch_match.group(5))
            best_metrics['specificity'] = float(last_epoch_match.group(6))
            best_metrics['f1_score'] = float(last_epoch_match.group(7))
    
    return best_metrics

best_metrics = extract_best_results(log_file)

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
print("📊 1. 生成混淆矩阵...")
try:
    import generate_optimal_confusion_matrix as cm_module
    
    # 更新配置
    cm_module.BEST_METRICS = best_metrics
    cm_module.FIGURES_DIR = FIGURES_DIR
    
    # 执行生成
    exec(open(str(VISUALIZATION_DIR / 'generate_optimal_confusion_matrix.py')).read())
    
    # 复制生成的文件
    source_cm = ROOT / 'visualization' / 'figures' / 'Confusion_Matrix_Best_Results.png'
    if source_cm.exists():
        shutil.copy2(source_cm, FIGURES_DIR / 'Confusion_Matrix_Best_Results.png')
    source_cm_pdf = ROOT / 'visualization' / 'figures' / 'Confusion_Matrix_Best_Results.pdf'
    if source_cm_pdf.exists():
        shutil.copy2(source_cm_pdf, FIGURES_DIR / 'Confusion_Matrix_Best_Results.pdf')
    
    print("✅ 混淆矩阵已生成")
except Exception as e:
    print(f"⚠️  混淆矩阵生成失败: {e}")
    import traceback
    traceback.print_exc()

print()

# 2. 生成其他可视化（使用generate_best_results_visualization.py）
print("📊 2. 生成其他可视化图表...")
try:
    # 修改脚本以使用我们的日志和输出目录
    vis_script = VISUALIZATION_DIR / 'generate_best_results_visualization.py'
    
    # 读取脚本内容
    with open(vis_script, 'r', encoding='utf-8') as f:
        script_content = f.read()
    
    # 创建临时脚本
    temp_script = VISUALIZATION_DIR / 'generate_best_results_temp.py'
    with open(temp_script, 'w', encoding='utf-8') as f:
        # 替换日志文件路径
        script_content = script_content.replace(
            "LOG_FILE = EXP_DIR / 'logs' / 'train_bio_cot_v3.2_50epochs_20260124_161327.log'",
            f"LOG_FILE = Path('{log_file}')"
        )
        # 替换输出目录
        script_content = script_content.replace(
            "FIGURES_DIR = VIS_DIR / 'figures'",
            f"FIGURES_DIR = Path('{FIGURES_DIR}')"
        )
        f.write(script_content)
    
    # 执行脚本
    import subprocess
    result = subprocess.run(
        [sys.executable, str(temp_script)],
        cwd=str(VISUALIZATION_DIR),
        capture_output=True,
        text=True,
        timeout=600
    )
    
    if result.returncode == 0:
        print("✅ 可视化脚本执行成功")
        print(result.stdout[-500:] if result.stdout else "")
    else:
        print(f"⚠️  可视化脚本执行失败:")
        print(result.stderr[-1000:] if result.stderr else "")
    
    # 清理临时脚本
    if temp_script.exists():
        temp_script.unlink()
    
    # 复制生成的图片
    source_figures = ROOT / 'visualization' / 'figures'
    if source_figures.exists():
        copied = 0
        for fig_file in source_figures.glob('*.png'):
            if 'Best_Results' in fig_file.name or 'High_Contrast' in fig_file.name:
                shutil.copy2(fig_file, FIGURES_DIR / fig_file.name)
                copied += 1
        for fig_file in source_figures.glob('*.pdf'):
            if 'Best_Results' in fig_file.name or 'High_Contrast' in fig_file.name:
                shutil.copy2(fig_file, FIGURES_DIR / fig_file.name)
                copied += 1
        print(f"✅ 已复制 {copied} 个图表文件")
        
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
    fig_files = list(FIGURES_DIR.glob('*'))
    print(f"  ✅ figures/ ({len(fig_files)} 个文件)")
    for fig_file in sorted(fig_files)[:10]:
        print(f"     - {fig_file.name}")
    if len(fig_files) > 10:
        print(f"     ... 还有 {len(fig_files) - 10} 个文件")
print(f"  ✅ training_report.txt")
print("=" * 80)

