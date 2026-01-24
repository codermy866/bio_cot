#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
移动废弃文件到Trash文件夹
"""

import shutil
from pathlib import Path

def move_to_trash(project_root: str):
    """移动废弃文件到Trash"""
    root = Path(project_root)
    trash_dir = root / 'Trash'
    trash_dir.mkdir(exist_ok=True)
    
    # 需要移动的文件和目录
    items_to_move = [
        'notes_file',
        'paper1_hierarchical_multimodal',
        'lancet_primary_care',
        'scripts/legacy',
        'scripts/reorganize_project.py',
        'scripts/final_reorganize.py',
        'scripts/final_reorganize_optimized.py',
        'scripts/organize_files.py',
        'cleanup_project.sh',  # 清理脚本也可以移动
        'analyze_project_structure.py',  # 分析脚本完成后也可以移动
    ]
    
    moved_items = []
    skipped_items = []
    
    for item in items_to_move:
        source = root / item
        if source.exists():
            dest = trash_dir / item
            try:
                # 如果目标已存在，先删除
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                
                # 创建目标目录的父目录
                dest.parent.mkdir(parents=True, exist_ok=True)
                
                # 移动文件或目录
                shutil.move(str(source), str(dest))
                moved_items.append(item)
                print(f"✓ 已移动: {item} -> Trash/{item}")
            except Exception as e:
                skipped_items.append((item, str(e)))
                print(f"✗ 移动失败: {item} - {e}")
        else:
            skipped_items.append((item, "文件不存在"))
            print(f"⚠ 跳过（不存在）: {item}")
    
    print("\n" + "=" * 80)
    print(f"移动完成！")
    print(f"成功移动: {len(moved_items)} 项")
    print(f"跳过/失败: {len(skipped_items)} 项")
    print("=" * 80)
    
    return moved_items, skipped_items

if __name__ == '__main__':
    move_to_trash('/data2/hmy/VLM_Caus_Rm_Mics')

