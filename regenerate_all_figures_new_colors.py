#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新生成所有图表 - 使用新的配色方案 (D69584 到 C7CCD6)
"""

import os
import sys
from pathlib import Path
import subprocess

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / 'newlog_0126'
FIGURES_DIR = OUTPUT_DIR / 'figures'

print("=" * 80)
print("🎨 重新生成所有图表 - 新配色方案 (D69584 到 C7CCD6)")
print("=" * 80)

# 需要重新生成的脚本列表
scripts_to_run = [
    'generate_advanced_visualizations.py',
    'generate_regression_visualizations.py',
    'generate_mantel_test_final.py',
    'generate_visualizations_fast.py',
]

# 更新所有脚本的配色
print("\n📝 步骤 1: 更新所有脚本的配色方案...")

# 新的配色方案
NEW_COLORS = {
    'positive': '#D69584',
    'negative': '#C7CCD6',
    'center_0': '#D69584',
    'center_1': '#D2A392',
    'center_2': '#CEB1A0',
    'center_3': '#CABFAE',
    'center_4': '#C7CCD6',
}

# 需要更新的文件
files_to_update = [
    'generate_advanced_visualizations.py',
    'generate_regression_visualizations.py',
    'generate_mantel_test_final.py',
    'generate_mantel_test_standard.py',
    'generate_visualizations_fast.py',
]

for script_file in files_to_update:
    file_path = ROOT / script_file
    if file_path.exists():
        print(f"  📝 更新 {script_file}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换配色
        content = content.replace("'positive': '#DDAB9F'", "'positive': '#D69584'")
        content = content.replace("'center_0': '#DDAB9F'", "'center_0': '#D69584'")
        content = content.replace("'center_1': '#D4B3A8'", "'center_1': '#D2A392'")
        content = content.replace("'center_2': '#CCBBB1'", "'center_2': '#CEB1A0'")
        content = content.replace("'center_3': '#C9C3BA'", "'center_3': '#CABFAE'")
        content = content.replace("'DDAB9F'", "'D69584'")
        content = content.replace("'D4B3A8'", "'D2A392'")
        content = content.replace("'CCBBB1'", "'CEB1A0'")
        content = content.replace("'C9C3BA'", "'CABFAE'")
        content = content.replace('custom_ddab9f_c7ccd6', 'custom_d69584_c7ccd6')
        content = content.replace("colors_list = ['#DDAB9F'", "colors_list = ['#D69584'")
        content = content.replace("colors_list = ['#D4B3A8'", "colors_list = ['#D2A392'")
        content = content.replace("colors_list = ['#CCBBB1'", "colors_list = ['#CEB1A0'")
        content = content.replace("colors_list = ['#C9C3BA'", "colors_list = ['#CABFAE'")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"    ✅ {script_file} 已更新")

print("\n📊 步骤 2: 重新生成所有图表...")

# 运行所有脚本
for script in scripts_to_run:
    script_path = ROOT / script
    if script_path.exists():
        print(f"\n🔄 运行 {script}...")
        try:
            result = subprocess.run(
                ['python', str(script_path)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                print(f"  ✅ {script} 执行成功")
            else:
                print(f"  ⚠️  {script} 执行有警告:")
                print(result.stderr[:500])
        except subprocess.TimeoutExpired:
            print(f"  ⚠️  {script} 执行超时")
        except Exception as e:
            print(f"  ❌ {script} 执行失败: {e}")

print("\n" + "=" * 80)
print("✅ 所有图表重新生成完成！")
print("=" * 80)
print(f"📁 输出目录: {FIGURES_DIR}")
print(f"\n生成的文件:")
for fig_file in sorted(FIGURES_DIR.glob('*.png')):
    size = fig_file.stat().st_size / 1024
    print(f"  ✅ {fig_file.name} ({size:.1f} KB)")

