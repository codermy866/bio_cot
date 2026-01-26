#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
更新所有脚本中的Label显示：将0/1改为Negative/Positive
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).parent

# 需要更新的文件列表
files_to_update = [
    'generate_advanced_visualizations.py',
    'generate_regression_visualizations.py',
    'generate_visualizations_fast.py',
    'visualization/code/generate_all_plots.py',
    'visualization/code/generate_additional_plots.py',
    'visualization/code/generate_best_results_visualization.py',
    'visualization/code/visualize_results_v2.py',
]

# 标签映射函数（添加到脚本中）
label_mapping_code = '''
# 标签映射函数
def get_label_name(label_val):
    """将Label值转换为英文显示名称"""
    if label_val == 0:
        return 'Negative'
    elif label_val == 1:
        return 'Positive'
    else:
        return f'Label {label_val}'
'''

print("=" * 80)
print("🔄 更新Label显示：0/1 → Negative/Positive")
print("=" * 80)

for file_path in files_to_update:
    full_path = ROOT / file_path
    if not full_path.exists():
        print(f"⚠️  文件不存在: {file_path}")
        continue
    
    print(f"\n📝 更新 {file_path}...")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. 替换 f'Label {label_val}' 或 'Label {label_val}'
    content = re.sub(r"f?'Label\s*\{\s*label_val\s*\}'", "f'Label: {get_label_name(label_val)}'", content)
    content = re.sub(r"f?'Label\s*=\s*\{\s*label_val\s*\}'", "f'Label: {get_label_name(label_val)}'", content)
    content = re.sub(r"'Label\s+(\d+)'", r"get_label_name(\1)", content)
    content = re.sub(r'"Label\s+(\d+)"', r"get_label_name(\1)", content)
    
    # 2. 替换 'Label 0' 和 'Label 1'
    content = content.replace("'Label 0'", "'Negative'")
    content = content.replace('"Label 0"', '"Negative"')
    content = content.replace("'Label 1'", "'Positive'")
    content = content.replace('"Label 1"', '"Positive"')
    content = content.replace("f'Label {0}'", "'Negative'")
    content = content.replace("f'Label {1}'", "'Positive'")
    
    # 3. 替换 label_val 在字符串中的使用
    content = re.sub(r"label=f'Fit\s*\(Label=\{\s*label_val\s*\}\)'", 
                    r"label=f'Fit ({get_label_name(label_val)})'", content)
    content = re.sub(r"label=f'Label\s*\{\s*label_val\s*\}'", 
                    r"label=f'{get_label_name(label_val)}'", content)
    
    # 4. 在DataFrame操作中，添加Label名称列
    # 如果看到 feature_df['Label']，添加一个 Label_Name 列
    if "feature_df['Label']" in content and "Label_Name" not in content:
        # 在数据加载后添加Label_Name列
        pattern = r"(feature_df\['Label'\]\s*=\s*labels)"
        replacement = r"\1\n        feature_df['Label_Name'] = feature_df['Label'].map({0: 'Negative', 1: 'Positive'})"
        content = re.sub(pattern, replacement, content)
    
    # 5. 在stripplot等seaborn函数中，使用Label_Name而不是Label
    content = re.sub(r"sns\.(stripplot|boxplot|violinplot|swarmplot)\([^)]*x=['\"]Label['\"]", 
                    r"sns.\1(\g<0>.replace(\"x='Label'\", \"x='Label_Name'\").replace('x=\"Label\"', 'x=\"Label_Name\"')", content)
    
    # 6. 添加标签映射函数（如果还没有）
    if 'def get_label_name' not in content:
        # 在COLORS定义后添加
        if 'COLORS = {' in content:
            pattern = r"(COLORS\s*=\s*\{[^}]+\})"
            replacement = r"\1\n\n" + label_mapping_code.strip()
            content = re.sub(pattern, replacement, content, count=1)
        else:
            # 在文件开头添加
            content = label_mapping_code + "\n\n" + content
    
    if content != original_content:
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ {file_path} 已更新")
    else:
        print(f"  ℹ️  {file_path} 无需更新")

print("\n" + "=" * 80)
print("✅ 所有文件更新完成！")
print("=" * 80)

