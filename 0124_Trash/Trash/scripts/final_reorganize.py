#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终重组脚本：备份、整理根目录文件、创建新项目目录
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json

class FinalReorganizer:
    """最终重组器"""
    
    def __init__(self, project_root: str, new_project_name: str = "MICCAI2026_VLM"):
        self.project_root = Path(project_root).resolve()
        self.new_project_root = self.project_root.parent / new_project_name
        self.backup_dir = self.project_root / "backup_before_reorganize"
        
        # 根目录文件分类映射
        self.root_file_mappings = {
            # 训练相关脚本 -> scripts/
            "auto_start_training.py": "scripts/auto_start_training.py",
            "check_training_progress.py": "scripts/check_training_progress.py",
            "test_training_script.py": "scripts/test_training.py",
            "main_causal_finetune.py": "scripts/main_causal_finetune.py",
            
            # 可视化脚本 -> scripts/
            "generate_training_causal_visualizations.py": "scripts/generate_training_visualizations.py",
            
            # 迁移和工具脚本 -> scripts/
            "migrate_to_vlm_project.py": "scripts/migrate_to_vlm.py",
            "organize_files.py": "scripts/organize_files.py",
            "reorganize_project.py": "scripts/reorganize_project.py",
            "compare_methods.py": "scripts/compare_methods.py",
            
            # 模型文件 -> src/models/
            "models_causal_gnn.py": "src/models/causal/causal_gnn.py",
            
            # Shell脚本 -> scripts/
            "check_venv_copy_status.sh": "scripts/check_venv_status.sh",
            "download_and_install_arial.sh": "scripts/install_arial_font.sh",
            "fix_venv_paths.sh": "scripts/fix_venv_paths.sh",
            "install_arial_font.sh": "scripts/install_arial_font_legacy.sh",
            "run_optimized_training.sh": "scripts/run_optimized_training.sh",
            "start_optimized_clip_training.sh": "scripts/start_optimized_clip_training.sh",
            "start_stage4_training.sh": "scripts/start_stage4_training.sh",
            
            # 文档文件 -> docs/
            "CAUSAL_CONSTRAINED_CLIP_DEEP_ANALYSIS.md": "docs/paper/MICCAI2025/CAUSAL_CONSTRAINED_CLIP_DEEP_ANALYSIS.md",
            "DEEP_TECHNICAL_ANALYSIS.md": "docs/paper/MICCAI2025/DEEP_TECHNICAL_ANALYSIS.md",
            "EXPERIMENTAL_METHODS.md": "docs/paper/MICCAI2025/EXPERIMENTAL_METHODS.md",
            "PROJECT_REORGANIZATION_PLAN.md": "docs/PROJECT_REORGANIZATION_PLAN.md",
            "PROJECT_STRUCTURE.md": "docs/PROJECT_STRUCTURE.md",
            "PROJECT_STRUCTURE_VLM.md": "docs/PROJECT_STRUCTURE_VLM.md",
            "README_ORGANIZATION.md": "docs/README_ORGANIZATION.md",
            "REGULARIZATION_OPTIMIZATION.md": "docs/paper/MICCAI2025/REGULARIZATION_OPTIMIZATION.md",
            "REORGANIZATION_GUIDE.md": "docs/REORGANIZATION_GUIDE.md",
            "TECHNICAL_MODULES_ANALYSIS.md": "docs/paper/MICCAI2025/TECHNICAL_MODULES_ANALYSIS.md",
            "TECHNICAL_WORK_SUMMARY.md": "docs/paper/MICCAI2025/TECHNICAL_WORK_SUMMARY.md",
            "MINDMAP_GUIDE.md": "docs/MINDMAP_GUIDE.md",
            "UPDATE_COMPLETE.md": "docs/UPDATE_COMPLETE.md",
            "VENV_MIGRATION_GUIDE.md": "docs/VENV_MIGRATION_GUIDE.md",
            
            # 报告文件 -> docs/
            "MIGRATION_REPORT.json": "docs/MIGRATION_REPORT.json",
            "REORGANIZATION_REPORT.json": "docs/REORGANIZATION_REPORT.json",
            
            # 工具脚本 -> scripts/
            "draw_academic_roadmap.py": "scripts/draw_academic_roadmap.py",
            
            # requirements文件 -> 根目录保留（但备份）
            "requirements_enhanced.txt": "requirements_enhanced.txt",
        }
        
    def create_complete_backup(self):
        """创建完整备份"""
        print("=" * 80)
        print("📦 步骤1: 创建完整备份")
        print("=" * 80)
        
        if self.backup_dir.exists():
            print(f"⚠️  备份目录已存在，删除旧备份...")
            shutil.rmtree(self.backup_dir)
        
        self.backup_dir.mkdir(parents=True)
        
        # 备份所有重要目录和文件
        items_to_backup = [
            "src/",
            "experiments/",
            "models/",
            "training/",
            "analysis/",
            "utils/",
            "util/",
            "scripts/",
            "configs/",
            "data/",
            "docs/",
            "figures/",
            "results/",
            "tests/",
            "visualization/",
            "exp1_Causal_Bayesian_clip/",
            "lancet_primary_care/",
            "paper1_hierarchical_multimodal/",
            "README.md",
            "README_GITHUB.md",
            "LICENSE",
            "setup.py",
            "requirements.txt",
            ".gitignore",
        ]
        
        # 备份根目录下的孤立文件
        root_files = [
            "auto_start_training.py",
            "check_training_progress.py",
            "compare_methods.py",
            "generate_training_causal_visualizations.py",
            "main_causal_finetune.py",
            "migrate_to_vlm_project.py",
            "models_causal_gnn.py",
            "organize_files.py",
            "reorganize_project.py",
            "test_training_script.py",
            "draw_academic_roadmap.py",
            "*.sh",
            "*.md",
            "*.json",
            "requirements_enhanced.txt",
        ]
        
        backed_up = 0
        for item in items_to_backup:
            src = self.project_root / item
            if src.exists():
                dst = self.backup_dir / item
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                backed_up += 1
                print(f"✅ 备份: {item}")
        
        # 备份根目录文件
        for pattern in root_files:
            if '*' in pattern:
                # 处理通配符
                import glob
                for file_path in glob.glob(str(self.project_root / pattern)):
                    file_name = Path(file_path).name
                    dst = self.backup_dir / file_name
                    shutil.copy2(file_path, dst)
                    backed_up += 1
                    print(f"✅ 备份: {file_name}")
            else:
                src = self.project_root / pattern
                if src.exists():
                    dst = self.backup_dir / pattern
                    shutil.copy2(src, dst)
                    backed_up += 1
                    print(f"✅ 备份: {pattern}")
        
        print(f"\n✅ 备份完成: {backed_up} 个项目已备份到 {self.backup_dir}")
        return backed_up
    
    def organize_root_files(self):
        """整理根目录文件"""
        print("\n" + "=" * 80)
        print("📁 步骤2: 整理根目录文件")
        print("=" * 80)
        
        moved_count = 0
        skipped_count = 0
        
        for old_path, new_path in self.root_file_mappings.items():
            src = self.project_root / old_path
            dst = self.project_root / new_path
            
            if not src.exists():
                continue
            
            if dst.exists():
                print(f"⚠️  目标已存在，跳过: {new_path}")
                skipped_count += 1
                continue
            
            # 创建目标目录
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            # 移动文件
            shutil.move(str(src), str(dst))
            print(f"✅ {old_path} -> {new_path}")
            moved_count += 1
        
        print(f"\n📊 整理统计: 移动 {moved_count} 个文件, 跳过 {skipped_count} 个")
        return moved_count
    
    def create_new_project_directory(self, auto_mode=False):
        """创建新项目目录"""
        print("\n" + "=" * 80)
        print("📂 步骤3: 创建新项目目录")
        print("=" * 80)
        
        if self.new_project_root.exists():
            print(f"⚠️  目标目录已存在: {self.new_project_root}")
            if auto_mode:
                print("自动模式：删除并重新创建...")
                shutil.rmtree(self.new_project_root)
            else:
                try:
                    response = input("是否删除并重新创建？(yes/no): ")
                    if response.lower() in ['yes', 'y']:
                        shutil.rmtree(self.new_project_root)
                    else:
                        print("❌ 已取消")
                        return False
                except EOFError:
                    print("非交互式环境：删除并重新创建...")
                    shutil.rmtree(self.new_project_root)
        
        self.new_project_root.mkdir(parents=True)
        print(f"✅ 创建新项目目录: {self.new_project_root}")
        return True
    
    def copy_organized_project(self):
        """复制整理后的项目到新目录"""
        print("\n" + "=" * 80)
        print("📋 步骤4: 复制整理后的项目到新目录")
        print("=" * 80)
        
        # 要复制的目录和文件
        items_to_copy = [
            "src/",
            "experiments/",
            "scripts/",
            "configs/",
            "docs/",
            "figures/",
            "results/",
            "tests/",
            "README.md",
            "README_GITHUB.md",
            "LICENSE",
            "setup.py",
            "requirements.txt",
            "requirements_enhanced.txt",
            ".gitignore",
        ]
        
        # 保留但可能不复制的大目录（可选）
        optional_items = [
            "data/",  # 通常是软链接，保留
            "models/",  # 如果还有残留
            "training/",  # 如果还有残留
            "analysis/",  # 如果还有残留
            "utils/",  # 如果还有残留
            "util/",  # 如果还有残留
            "visualization/",  # 如果还有残留
        ]
        
        copied_count = 0
        
        for item in items_to_copy:
            src = self.project_root / item
            if src.exists():
                dst = self.new_project_root / item
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                copied_count += 1
                print(f"✅ 复制: {item}")
        
        # 复制可选项目（如果存在）
        for item in optional_items:
            src = self.project_root / item
            if src.exists() and src.is_dir():
                # 检查是否为空或只包含__pycache__
                has_content = any(
                    f.name != '__pycache__' and not f.name.endswith('.pyc')
                    for f in src.rglob('*')
                    if f.is_file()
                )
                if has_content:
                    dst = self.new_project_root / item
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    copied_count += 1
                    print(f"✅ 复制: {item}")
        
        print(f"\n✅ 复制完成: {copied_count} 个项目已复制到 {self.new_project_root}")
        return copied_count
    
    def create_project_summary(self):
        """创建项目总结"""
        summary = {
            "reorganization_date": datetime.now().isoformat(),
            "source_directory": str(self.project_root),
            "target_directory": str(self.new_project_root),
            "backup_location": str(self.backup_dir),
            "root_files_organized": len(self.root_file_mappings),
            "steps_completed": [
                "Complete backup created",
                "Root files organized",
                "New project directory created",
                "Organized project copied to new directory"
            ]
        }
        
        summary_path = self.new_project_root / "PROJECT_REORGANIZATION_SUMMARY.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 项目总结已保存: {summary_path}")
    
    def run(self, auto_mode=False):
        """执行完整重组流程"""
        print("=" * 80)
        print("🚀 最终项目重组开始")
        print("=" * 80)
        print(f"源目录: {self.project_root}")
        print(f"目标目录: {self.new_project_root}")
        print(f"备份目录: {self.backup_dir}")
        print()
        
        try:
            # 步骤1: 创建完整备份
            self.create_complete_backup()
            
            # 步骤2: 整理根目录文件
            self.organize_root_files()
            
            # 步骤3: 创建新项目目录
            if not self.create_new_project_directory(auto_mode=auto_mode):
                return
            
            # 步骤4: 复制整理后的项目
            self.copy_organized_project()
            
            # 步骤5: 创建项目总结
            self.create_project_summary()
            
            print("\n" + "=" * 80)
            print("✅ 最终重组完成！")
            print("=" * 80)
            print(f"\n📋 总结:")
            print(f"  - 备份位置: {self.backup_dir}")
            print(f"  - 新项目位置: {self.new_project_root}")
            print(f"\n📝 下一步:")
            print(f"  1. 检查新项目目录: {self.new_project_root}")
            print(f"  2. 测试新项目是否正常工作")
            print(f"  3. 更新文档中的路径引用")
            
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n⚠️  如果出错，可以从备份恢复: {self.backup_dir}")


if __name__ == "__main__":
    import sys
    
    # 解析参数（过滤掉标志参数）
    project_root = "/data2/hmy/VLM_Caus_Rm"
    new_project_name = "MICCAI2026_VLM"
    auto_mode = '--yes' in sys.argv or '--auto' in sys.argv
    
    # 获取位置参数（排除标志参数）
    positional_args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    
    if len(positional_args) > 0:
        project_root = positional_args[0]
    if len(positional_args) > 1:
        new_project_name = positional_args[1]
    
    reorganizer = FinalReorganizer(project_root, new_project_name)
    
    if not auto_mode:
        print(f"项目根目录: {project_root}")
        print(f"新项目名称: {new_project_name}")
        print(f"新项目路径: {reorganizer.new_project_root}")
        try:
            response = input("\n是否开始最终重组？(yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("已取消")
                sys.exit(0)
        except EOFError:
            print("⚠️  非交互式环境，自动执行...")
            auto_mode = True
    
    reorganizer.run(auto_mode=auto_mode)

