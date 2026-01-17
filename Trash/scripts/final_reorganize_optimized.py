#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版最终重组脚本：添加进度显示和错误处理
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json
import sys

class FinalReorganizerOptimized:
    """优化版最终重组器"""
    
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
            "final_reorganize.py": "scripts/final_reorganize.py",
            
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
        }
        
    def safe_copy_tree(self, src, dst, desc=""):
        """安全复制目录，带进度显示"""
        try:
            if desc:
                print(f"  正在复制: {desc}...", end="", flush=True)
            
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            
            if desc:
                print(" ✅")
            return True
        except Exception as e:
            if desc:
                print(f" ❌ 错误: {e}")
            else:
                print(f"❌ 复制失败 {src} -> {dst}: {e}")
            return False
    
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
            try:
                shutil.move(str(src), str(dst))
                print(f"✅ {old_path} -> {new_path}")
                moved_count += 1
            except Exception as e:
                print(f"❌ 移动失败 {old_path}: {e}")
        
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
                try:
                    shutil.rmtree(self.new_project_root)
                    print("✅ 旧目录已删除")
                except Exception as e:
                    print(f"❌ 删除失败: {e}")
                    return False
            else:
                return False
        
        try:
            self.new_project_root.mkdir(parents=True)
            print(f"✅ 创建新项目目录: {self.new_project_root}")
            return True
        except Exception as e:
            print(f"❌ 创建目录失败: {e}")
            return False
    
    def copy_organized_project(self):
        """复制整理后的项目到新目录"""
        print("\n" + "=" * 80)
        print("📋 步骤4: 复制整理后的项目到新目录")
        print("=" * 80)
        
        # 要复制的目录和文件（按优先级）
        items_to_copy = [
            ("src/", "源代码"),
            ("experiments/", "实验代码"),
            ("scripts/", "启动脚本"),
            ("configs/", "配置文件"),
            ("docs/", "文档"),
            ("figures/", "图表"),
            ("results/", "结果"),
            ("tests/", "测试"),
            ("README.md", "README"),
            ("README_GITHUB.md", "GitHub README"),
            ("LICENSE", "许可证"),
            ("setup.py", "安装脚本"),
            ("requirements.txt", "依赖列表"),
            (".gitignore", "Git忽略文件"),
        ]
        
        copied_count = 0
        failed_count = 0
        
        for item, desc in items_to_copy:
            src = self.project_root / item
            if src.exists():
                dst = self.new_project_root / item
                if self.safe_copy_tree(src, dst, desc=desc):
                    copied_count += 1
                else:
                    failed_count += 1
            else:
                print(f"⚠️  源不存在，跳过: {item}")
        
        # 检查是否有 requirements_enhanced.txt
        req_enhanced = self.project_root / "requirements_enhanced.txt"
        if req_enhanced.exists():
            dst = self.new_project_root / "requirements_enhanced.txt"
            if self.safe_copy_tree(req_enhanced, dst, desc="增强依赖列表"):
                copied_count += 1
        
        print(f"\n✅ 复制完成: {copied_count} 个成功, {failed_count} 个失败")
        return copied_count
    
    def run(self, auto_mode=False):
        """执行完整重组流程"""
        print("=" * 80)
        print("🚀 最终项目重组开始（优化版）")
        print("=" * 80)
        print(f"源目录: {self.project_root}")
        print(f"目标目录: {self.new_project_root}")
        print(f"备份目录: {self.backup_dir}")
        print()
        
        try:
            # 步骤1: 检查备份（如果已存在，跳过备份步骤）
            if self.backup_dir.exists():
                print("✅ 备份目录已存在，跳过备份步骤")
            else:
                print("⚠️  备份目录不存在，建议先运行备份")
            
            # 步骤2: 整理根目录文件
            self.organize_root_files()
            
            # 步骤3: 创建新项目目录
            if not self.create_new_project_directory(auto_mode=auto_mode):
                if not auto_mode:
                    print("❌ 目录创建失败或被取消")
                    return
                else:
                    # 自动模式下，如果目录存在，尝试删除
                    print("尝试删除现有目录...")
                    try:
                        shutil.rmtree(self.new_project_root)
                        self.new_project_root.mkdir(parents=True)
                        print("✅ 目录已重新创建")
                    except Exception as e:
                        print(f"❌ 无法创建目录: {e}")
                        return
            
            # 步骤4: 复制整理后的项目
            copied = self.copy_organized_project()
            
            if copied > 0:
                # 步骤5: 创建项目总结
                summary = {
                    "reorganization_date": datetime.now().isoformat(),
                    "source_directory": str(self.project_root),
                    "target_directory": str(self.new_project_root),
                    "backup_location": str(self.backup_dir),
                    "files_copied": copied,
                }
                
                summary_path = self.new_project_root / "PROJECT_REORGANIZATION_SUMMARY.json"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                
                print(f"\n✅ 项目总结已保存: {summary_path}")
            
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
            
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n⚠️  如果出错，可以从备份恢复: {self.backup_dir}")


if __name__ == "__main__":
    project_root = "/data2/hmy/VLM_Caus_Rm"
    new_project_name = "MICCAI2026_VLM"
    auto_mode = '--yes' in sys.argv or '--auto' in sys.argv
    
    positional_args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    
    if len(positional_args) > 0:
        project_root = positional_args[0]
    if len(positional_args) > 1:
        new_project_name = positional_args[1]
    
    reorganizer = FinalReorganizerOptimized(project_root, new_project_name)
    reorganizer.run(auto_mode=True)  # 强制自动模式

