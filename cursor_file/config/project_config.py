#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件生成信息:
- 生成时间: 2025-01-XX 15:30:00
- 生成需求: 用户要求统一管理项目路径、虚拟环境路径等配置信息，方便后续开发使用
- 生成原因: 项目重组后需要统一的配置管理，避免路径硬编码，提高代码可维护性
- 相关任务: 项目结构重组和MICCAI论文准备

文件功能: 项目配置文件，记录项目路径、虚拟环境路径等配置信息，提供统一的配置接口
"""

from pathlib import Path

# ==================== 项目路径配置 ====================
PROJECT_ROOT = Path('/data2/hmy/VLM_Caus_Rm_Mics')

# 虚拟环境路径
VENV_PATH = PROJECT_ROOT / 'my_retfound'
VENV_PYTHON = VENV_PATH / 'bin' / 'python'
VENV_ACTIVATE = VENV_PATH / 'bin' / 'activate'

# 数据路径
DATA_DIR = PROJECT_ROOT / 'data'
DATA_5CENTERS_MULTI = DATA_DIR / '5centers_multi'
DATA_5CENTERS_INTERNAL_EXTERNAL = DATA_DIR / '5centers_multi_internal_external_final'

# 结果路径
RESULTS_DIR = PROJECT_ROOT / 'results'

# 模型路径（已移动到Trash，2025-12-25）
# MODELS_DIR = PROJECT_ROOT / 'models'  # 已移动到Trash/models_old_20251225
# SRC_MODELS_DIR = PROJECT_ROOT / 'src' / 'models'  # 已移动到Trash/src_old_20251225

# 实验路径
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'
EXP1_DIR = EXPERIMENTS_DIR / 'exp1_causal_bayesian_clip'
# EXP1_DEV_DIR已移动到Trash/exp1_Causal_Bayesian_clip_old (2025-12-25)

# 可视化路径（已移动到Trash，2025-12-25）
# VISUALIZATION_DIR = PROJECT_ROOT / 'visualization'  # 已移动到Trash/visualization_old_20251225
# FIGURES_DIR = PROJECT_ROOT / 'figures'  # 已移动到Trash/figures_old_20251225

# Cursor文件目录（用于存放与核心逻辑无关的文件）
CURSOR_FILE_DIR = PROJECT_ROOT / 'cursor_file'

# ==================== 配置验证 ====================
def check_paths():
    """检查关键路径是否存在"""
    paths_to_check = {
        '项目根目录': PROJECT_ROOT,
        '虚拟环境': VENV_PATH,
        '数据目录': DATA_DIR,
        '源代码': PROJECT_ROOT / 'src',
        '实验脚本': EXPERIMENTS_DIR,
    }
    
    print("=" * 80)
    print("检查项目路径...")
    print("=" * 80)
    
    all_ok = True
    for name, path in paths_to_check.items():
        if path.exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} (不存在)")
            all_ok = False
    
    print("=" * 80)
    if all_ok:
        print("✓ 所有路径检查通过")
    else:
        print("⚠ 部分路径不存在，请检查配置")
    print("=" * 80)
    
    return all_ok

# ==================== 获取配置 ====================
def get_config():
    """获取项目配置字典"""
    return {
        'project_root': str(PROJECT_ROOT),
        'venv_path': str(VENV_PATH),
        'venv_python': str(VENV_PYTHON),
        'data_dir': str(DATA_DIR),
        'results_dir': str(RESULTS_DIR),
        # 'models_dir': str(MODELS_DIR),  # 已移动到Trash (2025-12-25)
        'experiments_dir': str(EXPERIMENTS_DIR),
        # 'visualization_dir': str(VISUALIZATION_DIR),  # 已移动到Trash (2025-12-25)
        # 'figures_dir': str(FIGURES_DIR),  # 已移动到Trash (2025-12-25)
        'cursor_file_dir': str(CURSOR_FILE_DIR),
    }

if __name__ == '__main__':
    check_paths()
    print("\n项目配置:")
    config = get_config()
    for key, value in config.items():
        print(f"  {key}: {value}")

