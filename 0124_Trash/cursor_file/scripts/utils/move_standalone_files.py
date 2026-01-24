#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件生成信息:
- 生成时间: 2025-01-XX 15:20:00
- 生成需求: 用户要求将根目录下与项目核心运行逻辑无强关联的文件移动到cursor_file文件夹统一管理
- 生成原因: 项目重组需要，清理根目录，将辅助文件（文档、分析报告等）统一放在cursor_file目录
- 相关任务: 项目结构重组和文件整理

文件功能: 移动根目录下孤零零的文件到cursor_file文件夹，这些文件与项目核心运行逻辑无强关联
"""

import shutil
from pathlib import Path

def move_standalone_files(project_root: str):
    """移动根目录下的孤立文件到cursor_file"""
    root = Path(project_root)
    cursor_dir = root / 'cursor_file'
    cursor_dir.mkdir(exist_ok=True)
    
    # 需要移动的文件（与核心运行逻辑无关的文档和分析文件）
    standalone_files = [
        # 文档和分析报告
        'EMPTY_DIRS_CLEANUP.md',
        'FILE_STRUCTURE_ANALYSIS.md',
        'FINAL_CLEANUP_SUMMARY.md',
        'PROJECT_CLEANUP_REPORT.md',
        'PROJECT_LOGIC_ANALYSIS.md',
        'PROJECT_STRUCTURE_REORGANIZED.md',
        'REORGANIZATION_SUMMARY.md',
        'README_GITHUB.md',  # GitHub专用README，不影响运行
        
        # 工具脚本（已完成任务）
        'find_empty_dirs.py',
    ]
    
    # 保留的核心文件（与运行逻辑相关）
    keep_files = [
        'README.md',           # 主README
        'LICENSE',             # 许可证
        'setup.py',            # 安装配置
        'requirements.txt',    # 依赖列表
        'requirements_enhanced.txt',  # 增强依赖
        '.gitignore',          # Git配置
    ]
    
    moved_items = []
    skipped_items = []
    
    for file_name in standalone_files:
        source = root / file_name
        if source.exists():
            dest = cursor_dir / file_name
            try:
                # 如果目标已存在，先删除
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                
                # 移动文件
                shutil.move(str(source), str(dest))
                moved_items.append(file_name)
                print(f"✓ 已移动: {file_name} -> cursor_file/{file_name}")
            except Exception as e:
                skipped_items.append((file_name, str(e)))
                print(f"✗ 移动失败: {file_name} - {e}")
        else:
            skipped_items.append((file_name, "文件不存在"))
            print(f"⚠ 跳过（不存在）: {file_name}")
    
    print("\n" + "=" * 80)
    print(f"移动完成！")
    print(f"成功移动: {len(moved_items)} 个文件")
    print(f"跳过/失败: {len(skipped_items)} 个文件")
    print("=" * 80)
    
    return moved_items, skipped_items

if __name__ == '__main__':
    move_standalone_files('/data2/hmy/VLM_Caus_Rm_Mics')

