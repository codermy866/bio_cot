#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整理笔记文件脚本
将对话过程中创建的总结性文件整理到 notes_file/ 目录
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

class NotesOrganizer:
    """笔记整理器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self.notes_dir = self.project_root / "notes_file"
        
        # 需要整理的总结性文件（我创建的文件）
        # 包括：.md文档、.py脚本、.json报告、.log日志等
        self.notes_files = {
            # 项目重组相关文档
            "PROJECT_REORGANIZATION_PLAN.md": "项目重组计划",
            "REORGANIZATION_GUIDE.md": "重组使用指南",
            "PROJECT_STRUCTURE_ANALYSIS.md": "项目结构分析报告",
            "FINAL_PROJECT_STATUS.md": "最终项目状态报告",
            "PROJECT_STRUCTURE.md": "项目结构说明",
            "PROJECT_STRUCTURE_VLM.md": "VLM项目结构说明",
            "README_ORGANIZATION.md": "文件组织说明",
            
            # 分析脚本和报告
            "analyze_and_cleanup.py": "项目分析和清理脚本",
            "organize_notes.py": "笔记整理脚本",
            "reorganize_project.py": "项目重组脚本",
            "final_reorganize.py": "最终重组脚本",
            "final_reorganize_optimized.py": "优化重组脚本",
            "PROJECT_ANALYSIS_REPORT.json": "项目分析报告（JSON）",
            "REORGANIZATION_REPORT.json": "重组报告（JSON）",
            "MIGRATION_REPORT.json": "迁移报告（JSON）",
            
            # 日志文件
            "venv_copy.log": "虚拟环境复制日志",
        }
        
    def create_notes_directory(self):
        """创建笔记目录"""
        if not self.notes_dir.exists():
            self.notes_dir.mkdir(parents=True)
            print(f"✅ 创建笔记目录: {self.notes_dir}")
        else:
            print(f"✅ 笔记目录已存在: {self.notes_dir}")
        
        # 创建子目录
        subdirs = {
            "reorganization": "项目重组相关",
            "analysis": "分析报告",
            "scripts": "整理脚本",
            "logs": "日志文件",
        }
        
        for subdir, desc in subdirs.items():
            subdir_path = self.notes_dir / subdir
            if not subdir_path.exists():
                subdir_path.mkdir()
                print(f"✅ 创建子目录: {subdir}/{desc}")
    
    def organize_files(self):
        """整理文件"""
        print("\n" + "=" * 80)
        print("📁 开始整理笔记文件")
        print("=" * 80)
        
        moved_count = 0
        skipped_count = 0
        
        # 文件分类映射
        file_categories = {
            "reorganization": [
                "PROJECT_REORGANIZATION_PLAN.md",
                "REORGANIZATION_GUIDE.md",
                "PROJECT_STRUCTURE_ANALYSIS.md",
                "FINAL_PROJECT_STATUS.md",
                "PROJECT_STRUCTURE.md",
                "PROJECT_STRUCTURE_VLM.md",
                "README_ORGANIZATION.md",
            ],
            "analysis": [
                "PROJECT_ANALYSIS_REPORT.json",
                "REORGANIZATION_REPORT.json",
                "MIGRATION_REPORT.json",
            ],
            "scripts": [
                "analyze_and_cleanup.py",
                "organize_notes.py",
                "reorganize_project.py",
                "final_reorganize.py",
                "final_reorganize_optimized.py",
            ],
            "logs": [
                "venv_copy.log",
            ]
        }
        
        # 整理文件
        for category, files in file_categories.items():
            for file_name in files:
                src = self.project_root / file_name
                if src.exists():
                    dst = self.notes_dir / category / file_name
                    if dst.exists():
                        print(f"⚠️  目标已存在，跳过: {category}/{file_name}")
                        skipped_count += 1
                    else:
                        try:
                            shutil.move(str(src), str(dst))
                            print(f"✅ 移动: {file_name} -> notes_file/{category}/")
                            moved_count += 1
                        except Exception as e:
                            print(f"❌ 移动失败 {file_name}: {e}")
                else:
                    print(f"ℹ️  文件不存在，跳过: {file_name}")
        
        # 检查其他可能的文件（通过文件扩展名自动识别）
        extensions_to_check = {
            ".md": "reorganization",  # 其他md文件
            ".py": "scripts",         # 其他py脚本
            ".json": "analysis",      # 其他json报告
            ".log": "logs",           # 日志文件
        }
        
        # 获取所有根目录文件
        root_files = [f for f in self.project_root.iterdir() 
                     if f.is_file() and not f.name.startswith('.')]
        
        # 已处理的文件集合
        processed_files = set()
        for category, files in file_categories.items():
            processed_files.update(files)
        
        # 处理其他文件
        for file_path in root_files:
            file_name = file_path.name
            if file_name in processed_files:
                continue
            
            # 检查文件扩展名
            for ext, category in extensions_to_check.items():
                if file_name.endswith(ext):
                    # 检查是否是项目核心文件（应该保留在根目录）
                    core_files = [
                        "README.md", "README_GITHUB.md", "LICENSE",
                        "setup.py", "requirements.txt", "requirements_enhanced.txt",
                        ".gitignore"
                    ]
                    if file_name in core_files:
                        continue
                    
                    dst = self.notes_dir / category / file_name
                    if not dst.exists():
                        try:
                            shutil.move(str(file_path), str(dst))
                            print(f"✅ 移动: {file_name} -> notes_file/{category}/")
                            moved_count += 1
                        except Exception as e:
                            print(f"❌ 移动失败 {file_name}: {e}")
                    break
        
        print(f"\n📊 整理统计: 移动 {moved_count} 个文件, 跳过 {skipped_count} 个")
        return moved_count
    
    def create_readme(self):
        """创建笔记目录的README"""
        readme_content = """# 笔记文件目录

本目录包含项目整理过程中生成的总结性文件和文档。

## 📁 目录结构

```
notes_file/
├── reorganization/     # 项目重组相关文档
│   ├── PROJECT_REORGANIZATION_PLAN.md      # 项目重组计划
│   ├── REORGANIZATION_GUIDE.md             # 重组使用指南
│   ├── PROJECT_STRUCTURE_ANALYSIS.md       # 项目结构分析
│   └── FINAL_PROJECT_STATUS.md             # 最终项目状态
│
├── analysis/           # 分析报告
│   ├── PROJECT_ANALYSIS_REPORT.json        # 项目分析报告（JSON格式）
│   └── REORGANIZATION_REPORT.json          # 重组报告
│
├── scripts/            # 整理脚本
│   ├── analyze_and_cleanup.py              # 项目分析和清理脚本
│   └── organize_notes.py                   # 笔记整理脚本
│
└── logs/               # 日志文件
    └── venv_copy.log                       # 虚拟环境复制日志
```

## 📝 文件说明

### 项目重组相关
- **PROJECT_REORGANIZATION_PLAN.md**: 详细的项目重组计划，包括目录结构设计和文件映射
- **REORGANIZATION_GUIDE.md**: 重组脚本的使用指南和注意事项
- **PROJECT_STRUCTURE_ANALYSIS.md**: 项目结构对比分析报告
- **FINAL_PROJECT_STATUS.md**: 项目最终状态总结，包括根目录和备份目录的区别

### 分析报告
- **PROJECT_ANALYSIS_REPORT.json**: 机器可读的项目分析数据

### 整理脚本
- **analyze_and_cleanup.py**: 用于分析和清理项目结构的Python脚本

## 💡 说明

这些文件是在项目整理过程中生成的辅助文档和工具，不是项目的核心代码。
它们主要用于：
- 记录项目重组的过程和计划
- 提供项目结构的分析报告
- 提供整理工具和脚本

如果需要查看项目的主要文档，请参考：
- `README.md` - 项目主README
- `docs/` - 项目文档目录
- `docs/paper/MICCAI2025/` - 论文相关文档

---

**创建时间**: {date}
**说明**: 这些文件可以随时删除，不影响项目核心功能
""".format(date=datetime.now().strftime("%Y-%m-%d"))
        
        readme_path = self.notes_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"\n✅ 创建README: {readme_path}")
    
    def run(self):
        """执行整理"""
        print("=" * 80)
        print("📚 笔记文件整理")
        print("=" * 80)
        print(f"项目根目录: {self.project_root}")
        print(f"笔记目录: {self.notes_dir}")
        print()
        
        try:
            # 创建目录
            self.create_notes_directory()
            
            # 整理文件
            moved = self.organize_files()
            
            # 创建README
            self.create_readme()
            
            print("\n" + "=" * 80)
            print("✅ 笔记文件整理完成！")
            print("=" * 80)
            print(f"\n📋 总结:")
            print(f"  - 移动了 {moved} 个文件到 notes_file/")
            print(f"  - 笔记目录: {self.notes_dir}")
            print(f"\n💡 提示:")
            print(f"  - 这些文件是辅助文档，不影响项目核心功能")
            print(f"  - 可以随时查看或删除")
            
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    project_root = "/data2/hmy/VLM_Caus_Rm"
    
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    
    organizer = NotesOrganizer(project_root)
    organizer.run()

