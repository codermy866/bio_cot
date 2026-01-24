#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目结构分析脚本
分析核心运行逻辑，识别需要移动到Trash的文件
"""

import os
import ast
import re
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, List

class ProjectAnalyzer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.imports_map = defaultdict(set)  # 文件 -> 导入的模块集合
        self.used_by_map = defaultdict(set)   # 模块 -> 使用它的文件集合
        self.core_files = set()
        self.trash_files = set()
        
    def is_python_file(self, path: Path) -> bool:
        """判断是否是Python文件"""
        return path.suffix == '.py' and path.is_file()
    
    def extract_imports(self, file_path: Path) -> Set[str]:
        """提取Python文件的导入语句"""
        imports = set()
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                tree = ast.parse(content, filename=str(file_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split('.')[0])
        except:
            pass
        return imports
    
    def analyze_core_logic(self):
        """分析核心运行逻辑"""
        print("=" * 80)
        print("分析核心运行逻辑...")
        print("=" * 80)
        
        # 1. 核心训练入口
        train_entries = [
            'experiments/exp1_causal_bayesian_clip/train.py',
            'experiments/exp1_causal_bayesian_clip/train_vlm.py',
            'exp1_Causal_Bayesian_clip/code/vlm_enhanced_causal_clip.py',
            'exp1_Causal_Bayesian_clip/code/enhanced_causal_clip.py',
        ]
        
        # 2. 核心源代码目录
        core_dirs = [
            'src/',
            'models/',
            'utils/',
            'util/',
            'configs/',
        ]
        
        # 3. 数据处理
        data_dirs = [
            'src/data/',
            'data/',
        ]
        
        # 4. 评估和分析
        eval_dirs = [
            'src/evaluation/',
            'src/eval/',
            'analysis/',
        ]
        
        # 标记核心文件
        for entry in train_entries:
            path = self.project_root / entry
            if path.exists():
                self.core_files.add(entry)
                print(f"✓ 核心训练入口: {entry}")
        
        for dir_name in core_dirs + data_dirs + eval_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                for py_file in dir_path.rglob('*.py'):
                    rel_path = py_file.relative_to(self.project_root)
                    self.core_files.add(str(rel_path))
        
        print(f"\n核心文件数量: {len(self.core_files)}")
        
    def identify_trash_files(self):
        """识别需要移动到Trash的文件"""
        print("\n" + "=" * 80)
        print("识别需要移动到Trash的文件...")
        print("=" * 80)
        
        # 1. 开发笔记目录（README说明可以删除）
        trash_patterns = [
            'notes_file/',
            '__pycache__/',
            '*.pyc',
            '*.pyo',
            '*.log',
        ]
        
        # 2. 独立的实验目录（如果不在核心逻辑中）
        independent_experiments = [
            'paper1_hierarchical_multimodal/',
            'lancet_primary_care/',
        ]
        
        # 3. 备份和临时文件
        backup_patterns = [
            '*_backup*',
            '*_old*',
            '*_legacy*',
            '*.bak',
            '*.tmp',
        ]
        
        # 4. 重复的脚本（如果已有更好的版本）
        duplicate_scripts = [
            'scripts/legacy/',
            'scripts/reorganize_project.py',
            'scripts/final_reorganize.py',
            'scripts/final_reorganize_optimized.py',
            'scripts/organize_files.py',
        ]
        
        # 检查独立实验是否被核心代码引用
        for exp_dir in independent_experiments:
            exp_path = self.project_root / exp_dir
            if exp_path.exists():
                # 检查是否有核心文件引用它
                is_referenced = False
                for core_file in self.core_files:
                    core_path = self.project_root / core_file
                    if core_path.exists() and self.is_python_file(core_path):
                        imports = self.extract_imports(core_path)
                        if any(exp_dir.replace('/', '.').replace('_', '') in imp for imp in imports):
                            is_referenced = True
                            break
                
                if not is_referenced:
                    self.trash_files.add(exp_dir)
                    print(f"✗ 独立实验（未引用）: {exp_dir}")
        
        # 添加其他trash文件
        for pattern in trash_patterns + duplicate_scripts:
            if '*' in pattern:
                # 使用glob搜索
                for path in self.project_root.rglob(pattern):
                    rel_path = path.relative_to(self.project_root)
                    self.trash_files.add(str(rel_path))
            else:
                path = self.project_root / pattern
                if path.exists():
                    self.trash_files.add(pattern)
                    print(f"✗ Trash文件: {pattern}")
        
        print(f"\nTrash文件数量: {len(self.trash_files)}")
    
    def generate_report(self):
        """生成分析报告"""
        report = []
        report.append("# 项目运行逻辑分析报告\n")
        report.append(f"**项目根目录**: {self.project_root}\n\n")
        
        report.append("## 核心运行逻辑\n\n")
        report.append("### 1. 数据流\n")
        report.append("```\n")
        report.append("数据源 (data/) \n")
        report.append("  → 数据处理 (src/data/)\n")
        report.append("  → 数据集 (utils/enhanced_multimodal_dataset.py)\n")
        report.append("  → 训练脚本\n")
        report.append("```\n\n")
        
        report.append("### 2. 模型架构\n")
        report.append("```\n")
        report.append("Backbone (models/SwinT/, models/ViT/, models/MedicalViT/)\n")
        report.append("  → 核心模型 (src/models/)\n")
        report.append("  → 因果框架 (src/models/causal/)\n")
        report.append("  → 融合模块 (src/models/fusion/)\n")
        report.append("```\n\n")
        
        report.append("### 3. 训练流程\n")
        report.append("```\n")
        report.append("训练入口 (experiments/exp1_causal_bayesian_clip/train.py)\n")
        report.append("  → 模型加载 (src/models/)\n")
        report.append("  → 数据加载 (utils/)\n")
        report.append("  → 训练循环\n")
        report.append("  → 评估 (src/evaluation/)\n")
        report.append("  → 结果保存\n")
        report.append("```\n\n")
        
        report.append("### 4. 核心文件列表\n\n")
        report.append(f"共 {len(self.core_files)} 个核心文件\n\n")
        
        report.append("### 5. 需要移动到Trash的文件\n\n")
        report.append(f"共 {len(self.trash_files)} 个文件/目录\n\n")
        for trash in sorted(self.trash_files):
            report.append(f"- {trash}\n")
        
        return ''.join(report)
    
    def run(self):
        """运行分析"""
        self.analyze_core_logic()
        self.identify_trash_files()
        report = self.generate_report()
        
        # 保存报告
        report_path = self.project_root / 'PROJECT_LOGIC_ANALYSIS.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n" + "=" * 80)
        print(f"分析完成！报告已保存到: {report_path}")
        print("=" * 80)
        
        return self.trash_files

if __name__ == '__main__':
    analyzer = ProjectAnalyzer('/data2/hmy/VLM_Caus_Rm_Mics')
    trash_files = analyzer.run()

