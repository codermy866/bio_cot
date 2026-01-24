#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目重组脚本 - 将当前项目重组为标准的学术项目结构
用于MICCAI论文发表和GitHub开源准备
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

class ProjectReorganizer:
    """项目重组器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self.backup_dir = self.project_root / "backup_before_reorganize"
        
        # 文件映射表：当前路径 -> 新路径
        self.file_mappings = {
            # 核心模型文件
            "exp1_Causal_Bayesian_clip/code/enhanced_causal_clip.py": 
                "src/models/enhanced_causal_clip.py",
            "src/models/enhanced_causal_clip.py": 
                "src/models/enhanced_causal_clip.py",  # 如果已存在，保留
            "src/models/causal_bayesian_clip_framework.py": 
                "src/models/causal/bayesian_clip_framework.py",
            "src/models/adaptive_causal_graph.py": 
                "src/models/causal/adaptive_causal_graph.py",
            "src/models/causal_intervention.py": 
                "src/models/causal/causal_intervention.py",
            "src/models/enhanced_causal_clip_fixed.py": 
                "src/models/enhanced_causal_clip_fixed.py",  # 备用版本
            
            # 基础编码器
            "models/cnn_multimodal_model.py": 
                "src/models/backbones/cnn_encoder.py",
            "models/vmamba_multimodal_model.py": 
                "src/models/backbones/vmamba_encoder.py",
            "models/SwinT/swin_multimodal_model.py": 
                "src/models/backbones/swin_encoder.py",
            "models/SwinT/swin_image_encoder.py": 
                "src/models/backbones/swin_image_encoder.py",
            "models/ViT/vit_multimodal_model.py": 
                "src/models/backbones/vit_encoder.py",
            "models/ViT/vit_image_encoder.py": 
                "src/models/backbones/vit_image_encoder.py",
            "models/MedicalViT/medical_vit_multimodal_model.py": 
                "src/models/backbones/medical_vit_encoder.py",
            "models/MedicalViT/medical_vit_image_encoder.py": 
                "src/models/backbones/medical_vit_image_encoder.py",
            
            # 融合模块
            "models/hierarchical_multimodal/": 
                "src/models/fusion/hierarchical_fusion.py",
            
            # 不确定性量化
            "src/models/adaptive_causal_intervention_clip.py": 
                "src/models/uncertainty/adaptive_uncertainty.py",
            
            # 数据处理
            "utils/enhanced_multimodal_dataset.py": 
                "src/data/dataset.py",
            "src/data/enhanced_multimodal_dataset.py": 
                "src/data/dataset.py",
            "src/data/enhanced_oct_processing.py": 
                "src/data/preprocessing.py",
            "util/datasets.py": 
                "src/data/dataset_legacy.py",  # 保留旧版本
            "util/datasets_original.py": 
                "src/data/dataset_original.py",
            
            # 训练脚本
            "exp1_Causal_Bayesian_clip/code/train_enhanced_causal_clip.py": 
                "experiments/exp1_causal_bayesian_clip/train.py",
            "training/train_enhanced_causal_clip.py": 
                "experiments/exp1_causal_bayesian_clip/train_legacy.py",
            "exp1_Causal_Bayesian_clip/code/train_vlm_causal_clip.py": 
                "experiments/exp1_causal_bayesian_clip/train_vlm.py",
            "src/train/train_causal_bayesian_clip.py": 
                "experiments/exp1_causal_bayesian_clip/train_causal_bayesian.py",
            "training/train_clip_with_innovations.py": 
                "experiments/exp1_causal_bayesian_clip/train_innovations.py",
            "training/train_adaptive_causal_intervention_clip.py": 
                "experiments/exp1_causal_bayesian_clip/train_adaptive.py",
            "training/train_clip_optimized_swin_style.py": 
                "experiments/exp1_causal_bayesian_clip/train_swin_optimized.py",
            "training/optimized_2class_training.py": 
                "experiments/baseline/train_cnn_baseline.py",
            "training/optimized_vmamba_training.py": 
                "experiments/baseline/train_vmamba_baseline.py",
            "training/train_swin_large.py": 
                "experiments/baseline/train_swin_baseline.py",
            
            # 评估代码
            "analysis/decision_curve_analysis.py": 
                "src/evaluation/dca.py",
            "analysis/external_validation_evaluation.py": 
                "src/evaluation/external_validation.py",
            "analysis/find_optimal_threshold.py": 
                "src/evaluation/threshold_optimization.py",
            "src/eval/decision_curve_analysis.py": 
                "src/evaluation/dca.py",
            "src/eval/uncertainty_analysis.py": 
                "src/evaluation/uncertainty_analysis.py",
            "utils/advanced_clinical_metrics.py": 
                "src/evaluation/metrics.py",
            
            # 可视化
            "exp1_Causal_Bayesian_clip/visualization/visualization/advanced_visualizations.py": 
                "src/utils/visualization.py",
            "visualization/generate_advanced_visualizations.py": 
                "src/utils/visualization_advanced.py",
            "generate_causal_visualizations.py": 
                "src/utils/visualize_causal.py",
            "generate_detailed_medical_causal_graph.py": 
                "src/utils/visualize_causal_graph.py",
            
            # 工具函数
            "util/misc.py": 
                "src/utils/misc.py",
            "util/lr_sched.py": 
                "src/training/optimizers.py",
            "util/lr_decay.py": 
                "src/training/lr_scheduler.py",
            "util/pos_embed.py": 
                "src/utils/pos_embed.py",
            "util/graph.py": 
                "src/utils/graph.py",
        }
        
        # 目录映射
        self.dir_mappings = {
            "exp1_Causal_Bayesian_clip/results/": 
                "results/exp1_causal_bayesian_clip/",
            "exp1_Causal_Bayesian_clip/docs/": 
                "docs/paper/MICCAI2025/",
            "analysis/visualization/": 
                "figures/results/",
            "lancet_primary_care/figures/": 
                "figures/lancet/",
        }
        
    def create_backup(self):
        """创建备份"""
        print("📦 创建备份...")
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        self.backup_dir.mkdir(parents=True)
        
        # 备份关键文件
        key_files = [
            "exp1_Causal_Bayesian_clip/",
            "src/",
            "models/",
            "training/",
            "utils/",
            "analysis/",
        ]
        
        for key_file in key_files:
            src = self.project_root / key_file
            if src.exists():
                dst = self.backup_dir / key_file
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
        
        print(f"✅ 备份完成: {self.backup_dir}")
    
    def create_directory_structure(self):
        """创建新目录结构"""
        print("📁 创建目录结构...")
        
        directories = [
            "src/models/backbones",
            "src/models/fusion",
            "src/models/causal",
            "src/models/uncertainty",
            "src/data",
            "src/training",
            "src/evaluation",
            "src/utils",
            "experiments/exp1_causal_bayesian_clip",
            "experiments/baseline",
            "scripts",
            "configs/model_configs",
            "results/exp1_causal_bayesian_clip/checkpoints",
            "results/exp1_causal_bayesian_clip/logs",
            "results/exp1_causal_bayesian_clip/metrics",
            "results/baseline",
            "figures/architecture",
            "figures/results",
            "figures/causal_graphs",
            "tests",
            "docs/paper/MICCAI2025",
            "docs/api",
            "docs/tutorials",
        ]
        
        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            # 创建__init__.py
            if dir_path.startswith("src/") or dir_path.startswith("experiments/"):
                init_file = full_path / "__init__.py"
                if not init_file.exists():
                    init_file.touch()
        
        print("✅ 目录结构创建完成")
    
    def move_file(self, src_path: Path, dst_path: Path) -> bool:
        """移动文件，如果目标已存在则跳过"""
        if not src_path.exists():
            return False
        
        if dst_path.exists():
            print(f"⚠️  目标已存在，跳过: {dst_path}")
            return False
        
        # 创建目标目录
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 移动文件
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            shutil.rmtree(src_path)
        else:
            shutil.copy2(src_path, dst_path)
            src_path.unlink()
        
        return True
    
    def reorganize_files(self):
        """重组文件"""
        print("🔄 开始重组文件...")
        
        moved_count = 0
        skipped_count = 0
        
        # 移动文件
        for old_path, new_path in self.file_mappings.items():
            src = self.project_root / old_path
            dst = self.project_root / new_path
            
            if self.move_file(src, dst):
                print(f"✅ {old_path} -> {new_path}")
                moved_count += 1
            else:
                skipped_count += 1
        
        # 移动目录
        for old_dir, new_dir in self.dir_mappings.items():
            src = self.project_root / old_dir
            dst = self.project_root / new_dir
            
            if src.exists() and src.is_dir():
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                    print(f"✅ 目录移动: {old_dir} -> {new_dir}")
                    moved_count += 1
                else:
                    # 合并目录内容
                    for item in src.iterdir():
                        item_dst = dst / item.name
                        if not item_dst.exists():
                            shutil.move(str(item), str(item_dst))
                    print(f"✅ 目录合并: {old_dir} -> {new_dir}")
                    skipped_count += 1
        
        print(f"\n📊 重组统计: 移动 {moved_count} 个文件/目录, 跳过 {skipped_count} 个")
    
    def create_init_files(self):
        """创建__init__.py文件"""
        print("📝 创建__init__.py文件...")
        
        init_paths = [
            "src/__init__.py",
            "src/models/__init__.py",
            "src/models/backbones/__init__.py",
            "src/models/fusion/__init__.py",
            "src/models/causal/__init__.py",
            "src/models/uncertainty/__init__.py",
            "src/data/__init__.py",
            "src/training/__init__.py",
            "src/evaluation/__init__.py",
            "src/utils/__init__.py",
            "experiments/__init__.py",
            "experiments/exp1_causal_bayesian_clip/__init__.py",
            "experiments/baseline/__init__.py",
        ]
        
        for init_path in init_paths:
            full_path = self.project_root / init_path
            if not full_path.exists():
                full_path.touch()
                print(f"✅ 创建: {init_path}")
    
    def generate_migration_report(self):
        """生成迁移报告"""
        report = {
            "reorganization_date": datetime.now().isoformat(),
            "file_mappings": self.file_mappings,
            "dir_mappings": self.dir_mappings,
            "backup_location": str(self.backup_dir),
        }
        
        report_path = self.project_root / "REORGANIZATION_REPORT.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 迁移报告已保存: {report_path}")
    
    def run(self):
        """执行重组"""
        print("=" * 80)
        print("🚀 项目重组开始")
        print("=" * 80)
        print()
        
        # 1. 创建备份
        self.create_backup()
        print()
        
        # 2. 创建目录结构
        self.create_directory_structure()
        print()
        
        # 3. 重组文件
        self.reorganize_files()
        print()
        
        # 4. 创建__init__.py
        self.create_init_files()
        print()
        
        # 5. 生成报告
        self.generate_migration_report()
        print()
        
        print("=" * 80)
        print("✅ 项目重组完成！")
        print("=" * 80)
        print("\n📋 下一步:")
        print("1. 检查重组后的文件结构")
        print("2. 更新导入路径（可能需要手动调整）")
        print("3. 测试训练脚本是否正常")
        print("4. 更新README文档")


if __name__ == "__main__":
    import sys
    
    # 解析参数
    project_root = "/data2/hmy/VLM_Caus_Rm"
    auto_mode = False
    
    for arg in sys.argv[1:]:
        if arg in ['--yes', '--auto', '-y']:
            auto_mode = True
        elif not arg.startswith('--'):
            project_root = arg
    
    reorganizer = ProjectReorganizer(project_root)
    
    if not auto_mode:
        # 确认
        print(f"项目根目录: {project_root}")
        try:
            response = input("是否开始重组？(yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("已取消重组")
                sys.exit(0)
        except EOFError:
            # 非交互式环境，自动执行
            print("⚠️  非交互式环境，自动执行重组...")
            auto_mode = True
    
    if auto_mode:
        print(f"项目根目录: {project_root}")
        print("🚀 自动模式：开始重组...")
    
    reorganizer.run()

