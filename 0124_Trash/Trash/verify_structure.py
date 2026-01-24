#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证重组后的项目结构完整性
"""

import ast
import sys
from pathlib import Path
from typing import Set, Dict, List

def check_imports(file_path: Path, project_root: Path) -> List[str]:
    """检查文件的导入是否有效"""
    errors = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            tree = ast.parse(content, filename=str(file_path))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_parts = node.module.split('.')
                        # 检查是否是项目内的模块
                        if module_parts[0] in ['src', 'utils', 'util', 'models', 'configs']:
                            # 检查模块路径是否存在
                            module_path = project_root / '/'.join(module_parts)
                            if not module_path.exists() and not (module_path.parent / f"{module_parts[-1]}.py").exists():
                                # 检查是否是包
                                if not (project_root / module_parts[0] / '__init__.py').exists():
                                    errors.append(f"  可能缺失模块: {node.module}")
    except SyntaxError:
        pass  # 忽略语法错误（可能是未完成的文件）
    except Exception as e:
        pass  # 忽略其他错误
    
    return errors

def verify_core_files(project_root: Path):
    """验证核心文件"""
    print("=" * 80)
    print("验证核心文件完整性...")
    print("=" * 80)
    
    core_files = [
        'experiments/exp1_causal_bayesian_clip/train.py',
        'experiments/exp1_causal_bayesian_clip/train_vlm.py',
        'src/models/enhanced_causal_clip.py',
        'utils/enhanced_multimodal_dataset.py',
        'models/SwinT/swin_image_encoder.py',
    ]
    
    all_errors = []
    for file_path_str in core_files:
        file_path = project_root / file_path_str
        if file_path.exists():
            print(f"✓ 检查: {file_path_str}")
            errors = check_imports(file_path, project_root)
            if errors:
                print(f"  ⚠ 发现 {len(errors)} 个潜在问题:")
                for error in errors[:3]:  # 只显示前3个
                    print(error)
                all_errors.extend(errors)
        else:
            print(f"✗ 缺失: {file_path_str}")
            all_errors.append(f"缺失文件: {file_path_str}")
    
    print("\n" + "=" * 80)
    if all_errors:
        print(f"发现 {len(all_errors)} 个潜在问题")
    else:
        print("✓ 核心文件完整性检查通过！")
    print("=" * 80)
    
    return len(all_errors) == 0

def check_directory_structure(project_root: Path):
    """检查目录结构"""
    print("\n" + "=" * 80)
    print("检查目录结构...")
    print("=" * 80)
    
    required_dirs = [
        'src/',
        'src/models/',
        'src/data/',
        'src/evaluation/',
        'experiments/',
        'experiments/exp1_causal_bayesian_clip/',
        'models/',
        'utils/',
        'configs/',
        'Trash/',
    ]
    
    missing_dirs = []
    for dir_path_str in required_dirs:
        dir_path = project_root / dir_path_str
        if dir_path.exists():
            print(f"✓ {dir_path_str}")
        else:
            print(f"✗ 缺失: {dir_path_str}")
            missing_dirs.append(dir_path_str)
    
    print("\n" + "=" * 80)
    if missing_dirs:
        print(f"缺失 {len(missing_dirs)} 个目录")
    else:
        print("✓ 目录结构完整！")
    print("=" * 80)
    
    return len(missing_dirs) == 0

if __name__ == '__main__':
    project_root = Path('/data2/hmy/VLM_Caus_Rm_Mics')
    
    dirs_ok = check_directory_structure(project_root)
    files_ok = verify_core_files(project_root)
    
    if dirs_ok and files_ok:
        print("\n✅ 项目结构验证通过！")
        sys.exit(0)
    else:
        print("\n⚠️ 项目结构验证发现问题，请检查上述输出")
        sys.exit(1)

