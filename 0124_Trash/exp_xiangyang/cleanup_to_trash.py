#!/usr/bin/env python3
"""
清理脚本：将不需要的文件移动到Trash文件夹而不是直接删除
"""
import os
import shutil
import glob
from pathlib import Path
from datetime import datetime

def move_to_trash(file_path, trash_dir="Trash"):
    """
    将文件移动到Trash文件夹
    
    Args:
        file_path: 要移动的文件路径
        trash_dir: Trash文件夹路径（相对于当前目录）
    """
    if not os.path.exists(file_path):
        return False
    
    # 创建Trash文件夹（如果不存在）
    trash_path = Path(trash_dir)
    trash_path.mkdir(exist_ok=True)
    
    # 获取文件信息
    file_name = os.path.basename(file_path)
    file_dir = os.path.dirname(file_path)
    
    # 在Trash中创建相同的目录结构
    relative_path = os.path.relpath(file_dir, os.getcwd())
    if relative_path == '.':
        trash_file_path = trash_path / file_name
    else:
        trash_file_path = trash_path / relative_path / file_name
        trash_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 如果目标文件已存在，添加时间戳
    if trash_file_path.exists():
        name, ext = os.path.splitext(file_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trash_file_path = trash_path / relative_path / f"{name}_{timestamp}{ext}"
        if relative_path != '.':
            trash_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        shutil.move(file_path, trash_file_path)
        return True
    except Exception as e:
        print(f"⚠️  无法移动 {file_path} 到Trash: {e}")
        return False


def cleanup_results_to_trash(keep_files=None, trash_dir="Trash"):
    """
    清理结果文件到Trash
    
    Args:
        keep_files: 要保留的文件列表（完整路径）
        trash_dir: Trash文件夹路径
    """
    if keep_files is None:
        keep_files = []
    
    keep_files = set(keep_files)
    
    # 查找所有结果文件
    result_patterns = ['results/*', 'results_multimodal/*', 'results_multimodal_balanced/*']
    all_files = []
    for pattern in result_patterns:
        all_files.extend(glob.glob(pattern))
    
    # 要移动到Trash的文件
    files_to_move = [f for f in all_files if f not in keep_files]
    
    print(f"🗑️  将移动 {len(files_to_move)} 个结果文件到Trash")
    
    moved_count = 0
    for f in files_to_move:
        if move_to_trash(f, trash_dir):
            moved_count += 1
            if moved_count <= 10:
                print(f"   ✅ 已移动: {f}")
    
    if moved_count > 10:
        print(f"   ... 还有 {moved_count - 10} 个文件已移动")
    
    print(f"✅ 已移动 {moved_count} 个结果文件到Trash")
    return moved_count


def cleanup_logs_to_trash(keep_logs=None, keep_timestamps=None, trash_dir="Trash"):
    """
    清理日志和CSV文件到Trash
    
    Args:
        keep_logs: 要保留的日志文件列表
        keep_timestamps: 要保留的时间戳列表（用于匹配CSV文件）
        trash_dir: Trash文件夹路径
    """
    if keep_logs is None:
        keep_logs = []
    if keep_timestamps is None:
        keep_timestamps = []
    
    keep_logs = set(keep_logs)
    
    # 查找所有日志和CSV文件
    all_logs = glob.glob('logs/*.log')
    all_csvs = glob.glob('logs/*.csv')
    
    # 要保留的CSV文件（基于时间戳）
    keep_csvs = []
    if keep_timestamps:
        import re
        for csv in all_csvs:
            match = re.search(r'(\d{8}_\d{6})', csv)
            if match and match.group(1) in keep_timestamps:
                keep_csvs.append(csv)
    
    keep_csvs = set(keep_csvs)
    
    # 要移动的文件
    logs_to_move = [f for f in all_logs if f not in keep_logs]
    csvs_to_move = [f for f in all_csvs if f not in keep_csvs]
    
    print(f"🗑️  将移动 {len(logs_to_move)} 个日志文件和 {len(csvs_to_move)} 个CSV文件到Trash")
    
    moved_logs = 0
    for f in logs_to_move:
        if move_to_trash(f, trash_dir):
            moved_logs += 1
            if moved_logs <= 10:
                print(f"   ✅ 已移动日志: {os.path.basename(f)}")
    
    if moved_logs > 10:
        print(f"   ... 还有 {moved_logs - 10} 个日志文件已移动")
    
    moved_csvs = 0
    for f in csvs_to_move:
        if move_to_trash(f, trash_dir):
            moved_csvs += 1
            if moved_csvs <= 10:
                print(f"   ✅ 已移动CSV: {os.path.basename(f)}")
    
    if moved_csvs > 10:
        print(f"   ... 还有 {moved_csvs - 10} 个CSV文件已移动")
    
    print(f"✅ 已移动 {moved_logs} 个日志文件和 {moved_csvs} 个CSV文件到Trash")
    return moved_logs + moved_csvs


if __name__ == "__main__":
    print("=" * 80)
    print("🗑️  清理工具：将文件移动到Trash文件夹")
    print("=" * 80)
    print("\n使用方法：")
    print("  from cleanup_to_trash import cleanup_results_to_trash, cleanup_logs_to_trash")
    print("  cleanup_results_to_trash(keep_files=['path/to/keep.json'])")
    print("  cleanup_logs_to_trash(keep_logs=['logs/keep.log'], keep_timestamps=['20260106_220640'])")
    print("\n或者直接运行此脚本进行清理（需要修改keep_files等参数）")

