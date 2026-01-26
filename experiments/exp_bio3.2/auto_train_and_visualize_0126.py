#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动训练Bio-COT 3.2并生成可视化结果
- 执行完整训练流程
- 提取最佳结果
- 生成所有可视化图表
- 保存到newlog_0126文件夹
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import re
import json
import pandas as pd
from datetime import datetime

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
print("🚀 Bio-COT 3.2 自动训练和可视化")
print("=" * 80)
print(f"📁 输出目录: {OUTPUT_DIR}")
print(f"📁 图表目录: {FIGURES_DIR}")
print()

# ========================================
# 1. 启动训练
# ========================================
print("=" * 80)
print("📊 步骤 1: 启动训练")
print("=" * 80)

# 生成时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOGS_DIR / f"train_bio_cot_v3.2_auto_{timestamp}.log"

print(f"📝 训练日志: {log_file}")
print(f"⏳ 开始训练...")
print()

# 切换到项目根目录
os.chdir(ROOT)

# 启动训练（使用nohup在后台运行）
train_cmd = [
    sys.executable,
    str(TRAINING_DIR / 'train_bio_cot_v3.2.py')
]

print(f"执行命令: {' '.join(train_cmd)}")
print()

# 启动训练进程
process = subprocess.Popen(
    train_cmd,
    stdout=open(log_file, 'w'),
    stderr=subprocess.STDOUT,
    cwd=str(ROOT)
)

print(f"✅ 训练进程已启动 (PID: {process.pid})")
print(f"📝 日志文件: {log_file}")
print()

# 等待训练完成
print("⏳ 等待训练完成...")
print("   (可以使用 Ctrl+C 中断，然后手动运行后续步骤)")
print()

try:
    process.wait()
    print("✅ 训练完成！")
except KeyboardInterrupt:
    print("\n⚠️  训练被中断")
    print("   如果训练仍在后台运行，请等待完成后手动运行后续步骤")
    sys.exit(1)

print()

# ========================================
# 2. 从日志中提取最佳结果
# ========================================
print("=" * 80)
print("📊 步骤 2: 提取最佳结果")
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
        best_auc = float(matches[-1])  # 取最后一个（最高的AUC）
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
        pr_auc_pattern = r'PR-AUC: ([\d.]+)'
        pr_matches = re.findall(pr_auc_pattern, content)
        if pr_matches:
            # 找到对应epoch的PR-AUC
            for i, line in enumerate(content.split('\n')):
                if f"Epoch {best_metrics.get('epoch', 0)}" in line:
                    # 查找后续几行中的PR-AUC
                    for j in range(i, min(i+50, len(content.split('\n')))):
                        if 'PR-AUC' in content.split('\n')[j]:
                            pr_match = re.search(r'PR-AUC: ([\d.]+)', content.split('\n')[j])
                            if pr_match:
                                best_metrics['pr_auc'] = float(pr_match.group(1))
                                break
                    break
        
        mcc_pattern = r'MCC.*?Matthews.*?([\d.]+)'
        mcc_matches = re.findall(mcc_pattern, content)
        if mcc_matches and best_metrics.get('epoch'):
            # 找到对应epoch的MCC
            for i, line in enumerate(content.split('\n')):
                if f"Epoch {best_metrics.get('epoch', 0)}" in line:
                    for j in range(i, min(i+50, len(content.split('\n')))):
                        if 'MCC' in content.split('\n')[j]:
                            mcc_match = re.search(r'MCC.*?Matthews.*?([\d.]+)', content.split('\n')[j])
                            if mcc_match:
                                best_metrics['mcc'] = float(mcc_match.group(1))
                                break
                    break
    
    # 如果没找到，尝试从最后几行提取
    if not best_metrics:
        print("⚠️  未找到最佳结果，尝试从日志末尾提取...")
        lines = content.split('\n')[-100:]  # 最后100行
        for line in reversed(lines):
            if 'AUC' in line and 'ROC' in line:
                auc_match = re.search(r'AUC.*?ROC.*?([\d.]+)', line)
                if auc_match:
                    best_metrics['auc'] = float(auc_match.group(1))
                    break
    
    return best_metrics

best_metrics = extract_best_results_from_log(log_file)

if not best_metrics:
    print("❌ 无法从日志中提取最佳结果")
    print("   请检查日志文件或手动指定指标")
    sys.exit(1)

print("✅ 最佳结果提取成功:")
for key, value in best_metrics.items():
    print(f"   {key}: {value}")
print()

# ========================================
# 3. 保存指标到CSV
# ========================================
print("=" * 80)
print("📊 步骤 3: 保存指标到CSV")
print("=" * 80)

# 创建指标DataFrame
metrics_df = pd.DataFrame([best_metrics])
csv_path = OUTPUT_DIR / 'best_metrics.csv'
metrics_df.to_csv(csv_path, index=False)
print(f"✅ 指标已保存到: {csv_path}")
print()

# ========================================
# 4. 生成可视化
# ========================================
print("=" * 80)
print("📊 步骤 4: 生成可视化")
print("=" * 80)

# 切换到可视化代码目录
os.chdir(VISUALIZATION_DIR)

# 导入可视化函数
sys.path.insert(0, str(VISUALIZATION_DIR))

# 创建临时配置文件，将最佳结果传递给可视化脚本
temp_config = {
    'best_metrics': best_metrics,
    'log_file': str(log_file),
    'output_dir': str(FIGURES_DIR),
    'timestamp': timestamp
}

config_file = OUTPUT_DIR / 'visualization_config.json'
with open(config_file, 'w', encoding='utf-8') as f:
    json.dump(temp_config, f, indent=2)

print(f"📝 配置文件已保存: {config_file}")
print()

# 运行可视化脚本
print("🎨 生成混淆矩阵...")
try:
    from generate_optimal_confusion_matrix import BEST_METRICS
    # 更新BEST_METRICS
    import generate_optimal_confusion_matrix as cm_module
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

print("🎨 生成其他可视化图表...")
try:
    # 运行最佳结果可视化脚本
    vis_script = VISUALIZATION_DIR / 'generate_best_results_visualization.py'
    if vis_script.exists():
        # 修改脚本以使用我们的最佳结果和输出目录
        result = subprocess.run(
            [sys.executable, str(vis_script)],
            cwd=str(VISUALIZATION_DIR),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ 可视化图表已生成")
            # 复制生成的图片到输出目录
            source_figures = ROOT / 'visualization' / 'figures'
            if source_figures.exists():
                import shutil
                for fig_file in source_figures.glob('*.png'):
                    shutil.copy2(fig_file, FIGURES_DIR / fig_file.name)
                for fig_file in source_figures.glob('*.pdf'):
                    shutil.copy2(fig_file, FIGURES_DIR / fig_file.name)
                print(f"✅ 图表已复制到: {FIGURES_DIR}")
        else:
            print(f"⚠️  可视化脚本执行失败:")
            print(result.stderr)
    else:
        print(f"⚠️  可视化脚本不存在: {vis_script}")
except Exception as e:
    print(f"⚠️  可视化生成失败: {e}")
    import traceback
    traceback.print_exc()

print()

# ========================================
# 5. 生成总结报告
# ========================================
print("=" * 80)
print("📊 步骤 5: 生成总结报告")
print("=" * 80)

report_path = OUTPUT_DIR / 'training_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("Bio-COT 3.2 训练和可视化报告\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"训练时间: {timestamp}\n")
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
    f.write(f"  配置文件: {config_file}\n")
    f.write("\n")
    f.write("=" * 80 + "\n")

print(f"✅ 报告已保存到: {report_path}")
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
print(f"  - best_metrics.csv (指标)")
print(f"  - figures/ (所有可视化图表)")
print(f"  - training_report.txt (总结报告)")
print("=" * 80)

