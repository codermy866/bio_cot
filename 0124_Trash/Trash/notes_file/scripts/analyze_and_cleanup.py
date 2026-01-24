#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目分析和清理脚本
分析根目录和备份目录的区别，并整理项目结构
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import json

class ProjectAnalyzer:
    """项目分析器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self.backup_dir = self.project_root / "backup_before_reorganize"
        
    def analyze_structure(self):
        """分析项目结构"""
        print("=" * 80)
        print("📊 项目结构分析")
        print("=" * 80)
        
        # 分析根目录
        root_dirs = [d for d in self.project_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
        root_files = [f for f in self.project_root.iterdir() if f.is_file() and not f.name.startswith('.')]
        
        # 分析备份目录
        backup_dirs = []
        if self.backup_dir.exists():
            backup_dirs = [d for d in self.backup_dir.iterdir() if d.is_dir()]
        
        print("\n📁 根目录结构:")
        print(f"  目录数: {len(root_dirs)}")
        print(f"  文件数: {len(root_files)}")
        
        print("\n📦 备份目录结构:")
        print(f"  目录数: {len(backup_dirs)}")
        
        # 找出重复的目录
        root_dir_names = {d.name for d in root_dirs}
        backup_dir_names = {d.name for d in backup_dirs}
        
        common_dirs = root_dir_names & backup_dir_names
        root_only = root_dir_names - backup_dir_names
        backup_only = backup_dir_names - root_dir_names
        
        print("\n🔍 目录对比:")
        print(f"  共同目录 ({len(common_dirs)}): {sorted(common_dirs)}")
        print(f"  仅根目录有 ({len(root_only)}): {sorted(root_only)}")
        print(f"  仅备份有 ({len(backup_only)}): {sorted(backup_only)}")
        
        return {
            'root_dirs': [d.name for d in root_dirs],
            'root_files': [f.name for f in root_files],
            'backup_dirs': [d.name for d in backup_dirs],
            'common_dirs': sorted(common_dirs),
            'root_only': sorted(root_only),
            'backup_only': sorted(backup_only)
        }
    
    def categorize_root_files(self):
        """分类根目录文件"""
        print("\n" + "=" * 80)
        print("📄 根目录文件分类")
        print("=" * 80)
        
        root_files = [f for f in self.project_root.iterdir() if f.is_file() and not f.name.startswith('.')]
        
        categories = {
            '文档': [],
            '脚本': [],
            '配置': [],
            '其他': []
        }
        
        for f in root_files:
            name = f.name
            if name.endswith('.md'):
                categories['文档'].append(name)
            elif name.endswith('.py') or name.endswith('.sh'):
                categories['脚本'].append(name)
            elif name.endswith('.txt') or name.endswith('.json') or name.endswith('.yaml') or name.endswith('.yml'):
                categories['配置'].append(name)
            else:
                categories['其他'].append(name)
        
        for cat, files in categories.items():
            if files:
                print(f"\n{cat} ({len(files)}):")
                for f in sorted(files):
                    print(f"  - {f}")
        
        return categories
    
    def find_duplicate_dirs(self):
        """找出需要清理的重复目录"""
        print("\n" + "=" * 80)
        print("🔄 重复目录分析")
        print("=" * 80)
        
        # 这些目录在重组后应该已经移动到新位置，可以清理
        old_dirs_to_check = [
            'training',  # 应该移动到 experiments/
            'util',      # 应该移动到 src/utils/
            'utils',     # 应该移动到 src/utils/
            'visualization',  # 应该移动到 src/utils/ 或 scripts/
            'analysis',  # 应该移动到 src/evaluation/
            'models',    # 应该移动到 src/models/backbones/
        ]
        
        cleanup_candidates = []
        for dir_name in old_dirs_to_check:
            dir_path = self.project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                # 检查是否为空或只包含__pycache__
                has_content = any(
                    f.name != '__pycache__' and not f.name.endswith('.pyc')
                    for f in dir_path.rglob('*')
                    if f.is_file()
                )
                if has_content:
                    cleanup_candidates.append(dir_name)
                    print(f"⚠️  {dir_name}/ - 可能可以清理（已重组到新位置）")
                else:
                    print(f"✅ {dir_name}/ - 空目录，可以删除")
        
        return cleanup_candidates
    
    def generate_cleanup_plan(self, analysis_result, file_categories, duplicate_dirs):
        """生成清理计划"""
        print("\n" + "=" * 80)
        print("📋 清理计划")
        print("=" * 80)
        
        plan = {
            'files_to_move': {},
            'dirs_to_cleanup': [],
            'files_to_keep': []
        }
        
        # 文件移动计划
        file_mappings = {
            # 文档 -> docs/
            'CAUSAL_CONSTRAINED_CLIP_DEEP_ANALYSIS.md': 'docs/paper/MICCAI2025/',
            'DEEP_TECHNICAL_ANALYSIS.md': 'docs/paper/MICCAI2025/',
            'EXPERIMENTAL_METHODS.md': 'docs/paper/MICCAI2025/',
            'REGULARIZATION_OPTIMIZATION.md': 'docs/paper/MICCAI2025/',
            'TECHNICAL_MODULES_ANALYSIS.md': 'docs/paper/MICCAI2025/',
            'TECHNICAL_WORK_SUMMARY.md': 'docs/paper/MICCAI2025/',
            
            # 脚本 -> scripts/
            'final_reorganize_optimized.py': 'scripts/',
            
            # 保留在根目录的文件
            'README.md': 'KEEP',
            'README_GITHUB.md': 'KEEP',
            'LICENSE': 'KEEP',
            'setup.py': 'KEEP',
            'requirements.txt': 'KEEP',
            'requirements_enhanced.txt': 'KEEP',
            '.gitignore': 'KEEP',
        }
        
        for file_name, target in file_mappings.items():
            file_path = self.project_root / file_name
            if file_path.exists():
                if target == 'KEEP':
                    plan['files_to_keep'].append(file_name)
                else:
                    plan['files_to_move'][file_name] = target
        
        # 目录清理计划
        plan['dirs_to_cleanup'] = duplicate_dirs
        
        # 打印计划
        print("\n📝 文件移动计划:")
        for file_name, target in plan['files_to_move'].items():
            print(f"  {file_name} -> {target}")
        
        print("\n🗑️  目录清理计划:")
        for dir_name in plan['dirs_to_cleanup']:
            print(f"  {dir_name}/ - 检查后决定是否删除")
        
        print("\n✅ 保留在根目录的文件:")
        for file_name in plan['files_to_keep']:
            print(f"  {file_name}")
        
        return plan
    
    def execute_cleanup(self, plan, dry_run=True):
        """执行清理"""
        print("\n" + "=" * 80)
        print(f"{'🔍 模拟执行' if dry_run else '🚀 执行清理'}")
        print("=" * 80)
        
        moved_count = 0
        cleaned_count = 0
        
        # 移动文件
        for file_name, target_dir in plan['files_to_move'].items():
            src = self.project_root / file_name
            dst = self.project_root / target_dir / file_name
            
            if not src.exists():
                continue
            
            if dst.exists():
                print(f"⚠️  目标已存在，跳过: {target_dir}{file_name}")
                continue
            
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                print(f"✅ 移动: {file_name} -> {target_dir}")
            else:
                print(f"📋 将移动: {file_name} -> {target_dir}")
            moved_count += 1
        
        # 清理目录（仅标记，不实际删除）
        for dir_name in plan['dirs_to_cleanup']:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                file_count = len(list(dir_path.rglob('*.py')))
                print(f"📋 {dir_name}/ - 包含 {file_count} 个Python文件，需要手动检查")
        
        if dry_run:
            print(f"\n📊 模拟结果: 将移动 {moved_count} 个文件")
            print("💡 提示: 这是模拟运行，实际文件未移动")
        else:
            print(f"\n✅ 清理完成: 移动了 {moved_count} 个文件")
        
        return moved_count


if __name__ == "__main__":
    import sys
    
    project_root = "/data2/hmy/VLM_Caus_Rm"
    execute = False
    
    # 解析参数
    args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    flags = [arg for arg in sys.argv[1:] if arg.startswith('--')]
    
    if len(args) > 0:
        project_root = args[0]
    
    if '--execute' in flags:
        execute = True
    
    analyzer = ProjectAnalyzer(project_root)
    
    # 分析
    analysis_result = analyzer.analyze_structure()
    file_categories = analyzer.categorize_root_files()
    duplicate_dirs = analyzer.find_duplicate_dirs()
    
    # 生成计划
    plan = analyzer.generate_cleanup_plan(analysis_result, file_categories, duplicate_dirs)
    
    # 保存分析报告
    report = {
        'analysis': analysis_result,
        'file_categories': {k: v for k, v in file_categories.items() if v},
        'duplicate_dirs': duplicate_dirs,
        'cleanup_plan': plan
    }
    
    report_path = analyzer.project_root / "PROJECT_ANALYSIS_REPORT.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 分析报告已保存: {report_path}")
    
    # 执行清理
    if execute:
        print("\n执行实际清理...")
        analyzer.execute_cleanup(plan, dry_run=False)
    else:
        print("\n💡 提示: 使用 --execute 参数执行实际清理")
        analyzer.execute_cleanup(plan, dry_run=True)

