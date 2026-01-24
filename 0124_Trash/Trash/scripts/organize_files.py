#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件整理脚本：将项目文件按功能分类到对应文件夹
"""

import os
import shutil
from pathlib import Path

# 项目根目录
BASE_DIR = Path('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713')

# 文件分类规则
FILE_CATEGORIES = {
    'training': [
        'optimized_vmamba_training.py',
        'optimized_2class_training.py',
        'advanced_training_pipeline.py',
    ],
    'analysis': [
        'analyze_training_results.py',
        'diagnose_overfitting.py',
        'training_analysis_report.md',
    ],
    'visualization': [
        'generate_advanced_visualizations.py',
        'generate_training_plots.py',
        'advanced_visualizations.py',
        'advanced_visualizations_from_results.py',
        'paper_figure_generator.py',
    ],
    'models': [
        'vmamba_multimodal_model.py',
        'cnn_multimodal_model.py',
    ],
    'scripts': [
        'run_cnn_enhanced.py',
        'start_cnn_training.sh',
        'optimize_to_90_percent.py',
    ],
    'utils': [
        'fix_csv_encoding.py',
        'temperature_scaling.py',
        'test_cuda1_environment.py',
        'test_fix.py',
        'test_training.py',
        'test_causal_learning.py',
        'analyze_weights.py',
        'comprehensive_diagnostic_report.py',
        'create_merged_classification.py',
        'transfer_2class_to_5class.py',
        'visualize_causal_graph.py',
        'advanced_clinical_metrics.py',
        'enhanced_data_processing.py',
        'enhanced_model_architecture.py',
        'enhanced_multimodal_dataset.py',
        'enhanced_oct_processing.py',
    ],
    'reports': [
        # 报告文件可以单独放在reports文件夹
    ]
}

def create_directories():
    """创建分类文件夹"""
    for category in FILE_CATEGORIES.keys():
        dir_path = BASE_DIR / category
        dir_path.mkdir(exist_ok=True)
        print(f"✅ 创建目录: {category}/")

def move_files():
    """移动文件到对应分类文件夹"""
    moved_count = 0
    skipped_count = 0
    
    for category, files in FILE_CATEGORIES.items():
        target_dir = BASE_DIR / category
        
        for filename in files:
            source_path = BASE_DIR / filename
            target_path = target_dir / filename
            
            if source_path.exists() and source_path.is_file():
                try:
                    # 如果目标文件已存在，先备份
                    if target_path.exists():
                        backup_path = target_path.with_suffix(target_path.suffix + '.bak')
                        if backup_path.exists():
                            backup_path.unlink()
                        shutil.move(str(target_path), str(backup_path))
                        print(f"⚠️  备份已存在的文件: {target_path} -> {backup_path}")
                    
                    shutil.move(str(source_path), str(target_path))
                    print(f"✅ 移动: {filename} -> {category}/")
                    moved_count += 1
                except Exception as e:
                    print(f"❌ 移动失败 {filename}: {e}")
                    skipped_count += 1
            else:
                if not source_path.exists():
                    print(f"⚠️  文件不存在: {filename}")
                skipped_count += 1
    
    print(f"\n📊 统计: 移动 {moved_count} 个文件, 跳过 {skipped_count} 个文件")

def create_readme():
    """创建每个分类文件夹的README说明"""
    readme_content = {
        'training': """# 训练脚本目录

本目录包含所有训练相关的脚本：

- `optimized_vmamba_training.py`: VMamba模型优化训练脚本
- `optimized_2class_training.py`: CNN模型优化训练脚本
- `advanced_training_pipeline.py`: 高级训练流程脚本

## 使用方法

```bash
# VMamba训练
python training/optimized_vmamba_training.py --data_path ../5centers_multi --epochs 20

# CNN训练
python training/optimized_2class_training.py --data_path ../5centers_multi --epochs 20
```
""",
        'analysis': """# 分析脚本目录

本目录包含训练结果分析和诊断工具：

- `analyze_training_results.py`: 训练结果综合分析脚本
- `diagnose_overfitting.py`: 过拟合诊断工具
- `training_analysis_report.md`: 训练分析报告

## 使用方法

```bash
# 分析所有训练结果
python analysis/analyze_training_results.py

# 诊断过拟合
python analysis/diagnose_overfitting.py --model_path path/to/model
```
""",
        'visualization': """# 可视化脚本目录

本目录包含所有可视化相关的脚本：

- `generate_advanced_visualizations.py`: 生成高级可视化图表
- `generate_training_plots.py`: 生成训练过程图表
- `advanced_visualizations.py`: 高级可视化工具
- `advanced_visualizations_from_results.py`: 从结果生成可视化
- `paper_figure_generator.py`: 论文图表生成器

## 使用方法

```bash
# 生成训练可视化
python visualization/generate_training_plots.py --results_dir ../results

# 生成论文图表
python visualization/paper_figure_generator.py
```
""",
        'models': """# 模型定义目录

本目录包含模型架构定义：

- `vmamba_multimodal_model.py`: VMamba多模态模型定义
- `cnn_multimodal_model.py`: CNN多模态模型定义

## 模型说明

这些文件定义了模型的架构，被训练脚本导入使用。
""",
        'scripts': """# 启动脚本目录

本目录包含训练启动脚本和工具脚本：

- `run_cnn_enhanced.py`: CNN增强模型训练启动脚本
- `start_cnn_training.sh`: CNN训练启动Shell脚本
- `optimize_to_90_percent.py`: 优化到90%准确率的脚本

## 使用方法

```bash
# 启动CNN增强训练
python scripts/run_cnn_enhanced.py

# 或使用Shell脚本
bash scripts/start_cnn_training.sh
```
""",
        'utils': """# 工具脚本目录

本目录包含各种工具和辅助脚本：

- `fix_csv_encoding.py`: CSV编码修复工具
- `temperature_scaling.py`: 温度缩放校准工具
- `test_*.py`: 各种测试脚本
- `analyze_weights.py`: 权重分析工具
- `comprehensive_diagnostic_report.py`: 综合诊断报告生成器
- `create_merged_classification.py`: 创建合并分类
- `transfer_2class_to_5class.py`: 二分类转五分类工具
- `visualize_causal_graph.py`: 因果图可视化工具

## 使用方法

根据需要运行对应的工具脚本。
"""
    }
    
    for category, content in readme_content.items():
        readme_path = BASE_DIR / category / 'README.md'
        if not readme_path.exists():
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 创建README: {category}/README.md")

def update_imports():
    """更新训练脚本中的导入路径"""
    training_files = [
        BASE_DIR / 'training' / 'optimized_vmamba_training.py',
        BASE_DIR / 'training' / 'optimized_2class_training.py',
    ]
    
    # 导入路径映射
    import_mappings = {
        'from vmamba_multimodal_model import': 'from models.vmamba_multimodal_model import',
        'from cnn_multimodal_model import': 'from models.cnn_multimodal_model import',
        'from advanced_clinical_metrics import': 'from utils.advanced_clinical_metrics import',
        'from enhanced_multimodal_dataset import': 'from utils.enhanced_multimodal_dataset import',
        'from enhanced_oct_processing import': 'from utils.enhanced_oct_processing import',
        'import vmamba_multimodal_model': 'import models.vmamba_multimodal_model',
        'import cnn_multimodal_model': 'import models.cnn_multimodal_model',
        'import advanced_clinical_metrics': 'import utils.advanced_clinical_metrics',
        'import enhanced_multimodal_dataset': 'import utils.enhanced_multimodal_dataset',
        'import enhanced_oct_processing': 'import utils.enhanced_oct_processing',
    }
    
    for file_path in training_files:
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 更新所有导入路径
                updated = False
                for old_import, new_import in import_mappings.items():
                    if old_import in content:
                        content = content.replace(old_import, new_import)
                        updated = True
                
                # 更新sys.path插入（如果需要）
                if 'sys.path.insert' in content:
                    # 保持相对路径，但添加父目录
                    content = content.replace(
                        'sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))',
                        'sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))'
                    )
                
                if updated:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"✅ 更新导入路径: {file_path.name}")
                else:
                    print(f"ℹ️  无需更新导入路径: {file_path.name}")
            except Exception as e:
                print(f"⚠️  更新导入路径失败 {file_path.name}: {e}")

def main():
    print("=" * 60)
    print("📁 开始整理项目文件...")
    print("=" * 60)
    
    # 创建目录
    create_directories()
    
    # 移动文件
    print("\n📦 移动文件...")
    move_files()
    
    # 创建README
    print("\n📝 创建README文件...")
    create_readme()
    
    # 更新导入路径
    print("\n🔧 更新导入路径...")
    update_imports()
    
    print("\n" + "=" * 60)
    print("✅ 文件整理完成！")
    print("=" * 60)
    print("\n📂 新的目录结构:")
    print("  training/      - 训练脚本")
    print("  analysis/      - 分析脚本和报告")
    print("  visualization/ - 可视化脚本")
    print("  models/        - 模型定义")
    print("  scripts/       - 启动脚本")
    print("  utils/         - 工具脚本")

if __name__ == '__main__':
    main()

