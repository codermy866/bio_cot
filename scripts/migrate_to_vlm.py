#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迁移项目到新的VLM实验目录
将相关文件从旧目录迁移到 /data2/hmy/VLM_Caus_Rm/
"""

import os
import shutil
from pathlib import Path
import json

# 源目录和目标目录
SOURCE_DIR = Path("/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713")
TARGET_DIR = Path("/data2/hmy/VLM_Caus_Rm")

# 需要迁移的目录和文件
MIGRATE_DIRS = [
    "exp1_Causal_Bayesian_clip",  # 核心实验目录
    "training",  # 训练相关
    "models",  # 模型定义
    "utils",  # 工具函数
    "util",  # 工具函数（另一个）
    "src",  # 源代码
    "configs",  # 配置文件
    "scripts",  # 脚本
    "docs",  # 文档
    "visualization",  # 可视化
    "analysis",  # 分析脚本
    "lancet_primary_care",  # Lancet相关
    "paper1_hierarchical_multimodal",  # 论文相关
]

# 需要迁移的单个文件
MIGRATE_FILES = [
    "requirements_enhanced.txt",
    "README.md",
    "EXPERIMENTAL_METHODS.md",
    "TECHNICAL_WORK_SUMMARY.md",
    "TECHNICAL_MODULES_ANALYSIS.md",
    "DEEP_TECHNICAL_ANALYSIS.md",
    "CAUSAL_CONSTRAINED_CLIP_DEEP_ANALYSIS.md",
    "PROJECT_STRUCTURE.md",
    "README_ORGANIZATION.md",
    "REGULARIZATION_OPTIMIZATION.md",
    "MINDMAP_GUIDE.md",
    "LICENSE",
    ".gitignore",
]

# 需要迁移的脚本文件（根目录下的.py和.sh）
MIGRATE_SCRIPTS = [
    "main_causal_finetune.py",
    "models_causal_gnn.py",
    "generate_causal_visualizations.py",
    "generate_detailed_medical_causal_graph.py",
    "generate_training_causal_visualizations.py",
    "check_training_progress.py",
    "compare_methods.py",
    "organize_files.py",
    "draw_academic_roadmap.py",
    "test_training_script.py",
    "auto_start_training.py",
    "run_optimized_training.sh",
    "start_optimized_clip_training.sh",
    "start_stage4_training.sh",
    "download_and_install_arial.sh",
    "install_arial_font.sh",
    "migrate_to_vlm_project.py",  # 迁移脚本本身
]

# 不需要迁移的目录（数据集、结果、备份等）
EXCLUDE_DIRS = [
    "5centers_multi",  # 数据集
    "5centers_multi_internal_external",  # 数据集
    "5centers_multi_internal_external_recommended",  # 数据集
    "my_retfound",  # 虚拟环境
    "__pycache__",  # Python缓存
    "*.pyc",  # Python编译文件
    "backup_*",  # 备份目录
    "causal_bayesian_clip_results",  # 旧结果
    "enhanced_causal_clip_results",  # 旧结果
    "adaptive_causal_intervention_results",  # 旧结果
    "cnn_result*",  # 旧结果
    "vmamba_result*",  # 旧结果
    "swin_large_results",  # 旧结果
    "comparison_results",  # 旧结果
    "results",  # 旧结果
    "reports",  # 旧报告
    "paper_figures",  # 旧图片
    "pth",  # 旧模型
    "pretrained_weights",  # 预训练权重（如果需要可以单独处理）
    "hf_models",  # HuggingFace模型（如果需要可以单独处理）
    "oct_cache_optimized",  # 缓存
    "violin_pic",  # 旧图片
    "causal_analysis_detailed",  # 旧分析结果
]

def should_exclude(path: Path) -> bool:
    """检查路径是否应该被排除"""
    path_str = str(path)
    
    # 检查是否在排除列表中
    for exclude in EXCLUDE_DIRS:
        if exclude in path_str:
            return True
    
    # 排除隐藏文件和缓存
    if path.name.startswith('.') and path.name != '.gitignore':
        return True
    
    if '__pycache__' in path_str:
        return True
    
    if path.suffix == '.pyc':
        return True
    
    return False

def copy_directory(src: Path, dst: Path, exclude_patterns=None):
    """递归复制目录，排除指定模式"""
    if not src.exists():
        print(f"⚠️  源目录不存在: {src}")
        return False
    
    if should_exclude(src):
        print(f"⏭️  跳过排除目录: {src}")
        return False
    
    try:
        # 创建目标目录
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        # 如果是文件，直接复制
        if src.is_file():
            shutil.copy2(src, dst)
            print(f"✅ 复制文件: {src.name}")
            return True
        
        # 如果是目录，递归复制
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            
            for item in src.iterdir():
                if should_exclude(item):
                    continue
                
                dst_item = dst / item.name
                copy_directory(item, dst_item)
            
            return True
        
    except Exception as e:
        print(f"❌ 复制失败 {src} -> {dst}: {e}")
        return False

def update_paths_in_file(file_path: Path, old_path: str, new_path: str):
    """更新文件中的路径引用"""
    if not file_path.exists() or not file_path.is_file():
        return
    
    try:
        # 读取文件
        content = file_path.read_text(encoding='utf-8')
        
        # 替换路径
        if old_path in content:
            new_content = content.replace(old_path, new_path)
            file_path.write_text(new_content, encoding='utf-8')
            print(f"✅ 更新路径: {file_path.name}")
    except Exception as e:
        print(f"⚠️  更新路径失败 {file_path}: {e}")

def main():
    """主迁移函数"""
    print("=" * 80)
    print("🚀 开始迁移项目到新目录")
    print("=" * 80)
    print(f"源目录: {SOURCE_DIR}")
    print(f"目标目录: {TARGET_DIR}")
    print()
    
    # 创建目标目录
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # 统计
    copied_dirs = 0
    copied_files = 0
    skipped = 0
    
    # 1. 迁移目录
    print("📁 迁移目录...")
    for dir_name in MIGRATE_DIRS:
        src_dir = SOURCE_DIR / dir_name
        dst_dir = TARGET_DIR / dir_name
        
        if src_dir.exists():
            print(f"\n迁移目录: {dir_name}")
            if copy_directory(src_dir, dst_dir):
                copied_dirs += 1
            else:
                skipped += 1
        else:
            print(f"⚠️  目录不存在: {dir_name}")
            skipped += 1
    
    # 2. 迁移单个文件
    print("\n📄 迁移文件...")
    for file_name in MIGRATE_FILES:
        src_file = SOURCE_DIR / file_name
        dst_file = TARGET_DIR / file_name
        
        if src_file.exists() and src_file.is_file():
            if copy_directory(src_file, dst_file):
                copied_files += 1
            else:
                skipped += 1
        else:
            print(f"⚠️  文件不存在: {file_name}")
            skipped += 1
    
    # 3. 迁移脚本文件
    print("\n🔧 迁移脚本文件...")
    for script_name in MIGRATE_SCRIPTS:
        src_script = SOURCE_DIR / script_name
        dst_script = TARGET_DIR / script_name
        
        if src_script.exists() and src_script.is_file():
            if copy_directory(src_script, dst_script):
                copied_files += 1
            else:
                skipped += 1
    
    # 4. 创建数据集软链接（可选）
    print("\n🔗 创建数据集软链接...")
    dataset_dirs = [
        "5centers_multi",
        "5centers_multi_internal_external",
    ]
    
    for dataset_dir in dataset_dirs:
        src_dataset = SOURCE_DIR / dataset_dir
        dst_link = TARGET_DIR / "data" / dataset_dir
        
        if src_dataset.exists():
            try:
                dst_link.parent.mkdir(parents=True, exist_ok=True)
                if dst_link.exists():
                    if dst_link.is_symlink():
                        dst_link.unlink()
                    else:
                        print(f"⚠️  {dataset_dir} 已存在，跳过")
                        continue
                
                dst_link.symlink_to(src_dataset)
                print(f"✅ 创建软链接: {dataset_dir} -> {src_dataset}")
            except Exception as e:
                print(f"❌ 创建软链接失败 {dataset_dir}: {e}")
    
    # 5. 更新路径引用
    print("\n🔄 更新路径引用...")
    old_data_path = str(SOURCE_DIR / "5centers_multi")
    new_data_path = str(TARGET_DIR / "data" / "5centers_multi")
    
    # 更新常见文件中的路径
    files_to_update = [
        TARGET_DIR / "exp1_Causal_Bayesian_clip" / "README_exp1.md",
        TARGET_DIR / "exp1_Causal_Bayesian_clip" / "code" / "train_enhanced_causal_clip.py",
        TARGET_DIR / "exp1_Causal_Bayesian_clip" / "code" / "train_vlm_causal_clip.py",
        TARGET_DIR / "exp1_Causal_Bayesian_clip" / "code" / "split_dataset_by_centers.py",
        TARGET_DIR / "exp1_Causal_Bayesian_clip" / "code" / "visualize_dataset_split.py",
    ]
    
    for file_path in files_to_update:
        if file_path.exists():
            update_paths_in_file(file_path, old_data_path, new_data_path)
            # 也更新旧的绝对路径
            update_paths_in_file(file_path, str(SOURCE_DIR), str(TARGET_DIR))
    
    # 6. 创建新的README
    print("\n📝 创建新项目README...")
    readme_content = f"""# VLM-Enhanced Causal Bayesian Framework

## 项目概述

本项目实现了基于Vision-Language Model (VLM)增强的因果贝叶斯CLIP框架，用于医学多模态诊断。

## 目录结构

```
VLM_Caus_Rm/
├── exp1_Causal_Bayesian_clip/     # 核心实验代码和文档
├── code/                           # 通用代码
├── training/                       # 训练脚本
├── models/                         # 模型定义
├── utils/                          # 工具函数
├── configs/                        # 配置文件
├── docs/                           # 文档
├── data/                           # 数据集软链接
│   ├── 5centers_multi -> [原数据集路径]
│   └── 5centers_multi_internal_external -> [原数据集路径]
└── README.md                       # 本文件
```

## 快速开始

### 1. 环境配置

```bash
# 使用原虚拟环境（或创建新的）
source /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/my_retfound/bin/activate

# 或创建新环境
# python -m venv venv
# source venv/bin/activate
# pip install -r exp1_Causal_Bayesian_clip/requirements_enhanced.txt
```

### 2. 数据集

数据集保留在原位置，通过软链接访问：
- `data/5centers_multi` -> 原始数据集
- `data/5centers_multi_internal_external` -> 划分后的数据集

### 3. 运行VLM增强实验

详见: `exp1_Causal_Bayesian_clip/docs/VLM_ENHANCED_EXPERIMENT_PLAN.md`

```bash
cd exp1_Causal_Bayesian_clip/code
python train_vlm_causal_clip.py \\
    --data_path ../../data/5centers_multi \\
    --vlm_model Qwen/Qwen2-VL-2B-Instruct \\
    --batch_size 10 \\
    --use_amp
```

## 相关文档

- **实验方案**: `exp1_Causal_Bayesian_clip/docs/VLM_ENHANCED_EXPERIMENT_PLAN.md`
- **快速开始**: `exp1_Causal_Bayesian_clip/docs/VLM_QUICK_START.md`
- **A6000配置**: `exp1_Causal_Bayesian_clip/docs/VLM_A6000_CONFIGURATION.md`
- **实施路线图**: `exp1_Causal_Bayesian_clip/docs/VLM_IMPLEMENTATION_ROADMAP.md`

## 迁移信息

- **迁移日期**: {Path(__file__).stat().st_mtime}
- **源目录**: {SOURCE_DIR}
- **目标目录**: {TARGET_DIR}

## 注意事项

1. 数据集未迁移，使用软链接访问原数据集
2. 虚拟环境路径可能需要更新
3. 部分结果目录未迁移，需要重新训练生成
4. 请检查并更新所有路径引用

"""
    
    readme_path = TARGET_DIR / "README.md"
    readme_path.write_text(readme_content, encoding='utf-8')
    print(f"✅ 创建README: {readme_path}")
    
    # 7. 创建迁移报告
    print("\n📊 生成迁移报告...")
    report = {
        "source_dir": str(SOURCE_DIR),
        "target_dir": str(TARGET_DIR),
        "copied_dirs": copied_dirs,
        "copied_files": copied_files,
        "skipped": skipped,
        "migrated_dirs": MIGRATE_DIRS,
        "migrated_files": MIGRATE_FILES + MIGRATE_SCRIPTS,
        "excluded_dirs": EXCLUDE_DIRS,
    }
    
    report_path = TARGET_DIR / "MIGRATION_REPORT.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✅ 迁移报告: {report_path}")
    
    # 总结
    print("\n" + "=" * 80)
    print("✅ 迁移完成！")
    print("=" * 80)
    print(f"✅ 已迁移目录: {copied_dirs}")
    print(f"✅ 已迁移文件: {copied_files}")
    print(f"⏭️  跳过项目: {skipped}")
    print(f"\n📁 新项目位置: {TARGET_DIR}")
    print(f"📄 迁移报告: {report_path}")
    print("\n⚠️  请检查并更新以下内容:")
    print("  1. 虚拟环境路径")
    print("  2. 数据集路径引用")
    print("  3. 配置文件中的路径")
    print("  4. 训练脚本中的路径")

if __name__ == "__main__":
    main()

