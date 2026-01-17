#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查找并删除空文件夹
"""

import os
from pathlib import Path
from collections import defaultdict

def is_empty_dir(path: Path, ignore_patterns=None) -> bool:
    """检查目录是否为空（忽略特定文件）"""
    if ignore_patterns is None:
        ignore_patterns = ['.gitkeep', '.gitignore']
    
    if not path.is_dir():
        return False
    
    try:
        items = list(path.iterdir())
        # 过滤掉忽略的文件
        items = [item for item in items if item.name not in ignore_patterns]
        return len(items) == 0
    except PermissionError:
        return False

def find_empty_dirs(root: Path, exclude_dirs=None) -> list:
    """递归查找所有空文件夹"""
    if exclude_dirs is None:
        exclude_dirs = ['Trash', '__pycache__', '.git', 'my_retfound', '.ipynb_checkpoints']
    
    empty_dirs = []
    
    # 排除的目录路径
    exclude_paths = [root / d for d in exclude_dirs]
    
    for dirpath, dirnames, filenames in os.walk(root):
        dir_path = Path(dirpath)
        
        # 跳过排除的目录
        if any(str(dir_path).startswith(str(exclude)) for exclude in exclude_paths):
            continue
        
        # 跳过虚拟环境
        if 'my_retfound' in str(dir_path):
            continue
        
        # 检查是否为空
        if is_empty_dir(dir_path):
            empty_dirs.append(dir_path)
    
    return empty_dirs

def delete_empty_dirs(empty_dirs: list, dry_run=True):
    """删除空文件夹"""
    deleted = []
    failed = []
    
    # 按深度排序，先删除深层目录
    empty_dirs.sort(key=lambda p: len(p.parts), reverse=True)
    
    for dir_path in empty_dirs:
        try:
            if not dry_run:
                dir_path.rmdir()
                deleted.append(dir_path)
                print(f"✓ 已删除: {dir_path.relative_to(dir_path.parts[0])}")
            else:
                print(f"[DRY RUN] 将删除: {dir_path}")
                deleted.append(dir_path)
        except OSError as e:
            failed.append((dir_path, str(e)))
            print(f"✗ 删除失败: {dir_path} - {e}")
    
    return deleted, failed

if __name__ == '__main__':
    import sys
    
    project_root = Path('/data2/hmy/VLM_Caus_Rm_Mics')
    
    # 检查是否有命令行参数
    auto_delete = len(sys.argv) > 1 and sys.argv[1] == '--delete'
    
    print("=" * 80)
    print("查找空文件夹...")
    print("=" * 80)
    
    empty_dirs = find_empty_dirs(project_root)
    
    print(f"\n找到 {len(empty_dirs)} 个空文件夹:\n")
    for dir_path in empty_dirs:
        rel_path = dir_path.relative_to(project_root)
        print(f"  - {rel_path}")
    
    if empty_dirs:
        if auto_delete:
            print("\n" + "=" * 80)
            print("自动删除模式: 开始删除空文件夹...")
            print("=" * 80)
            deleted, failed = delete_empty_dirs(empty_dirs, dry_run=False)
            print("\n" + "=" * 80)
            print(f"删除完成!")
            print(f"成功删除: {len(deleted)} 个文件夹")
            if failed:
                print(f"失败: {len(failed)} 个文件夹")
                for dir_path, error in failed:
                    print(f"  - {dir_path}: {error}")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("提示: 使用 'python find_empty_dirs.py --delete' 来自动删除这些空文件夹")
            print("=" * 80)
    else:
        print("\n✓ 没有找到空文件夹")

