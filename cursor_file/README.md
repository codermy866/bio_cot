# 📁 Cursor File 目录结构说明

本目录包含项目开发过程中的文档、脚本和配置文件，已按类型分类整理。

## 📂 目录结构

```
cursor_file/
├── docs/                    # 文档目录
│   ├── analysis/           # 分析报告
│   ├── training/           # 训练相关文档
│   ├── dataset/            # 数据集相关文档
│   ├── miccai/             # MICCAI相关文档
│   └── project/            # 项目组织文档
├── scripts/                # 脚本目录
│   ├── dataset/            # 数据集处理脚本
│   ├── utils/              # 工具脚本
│   └── visualization/      # 可视化脚本
├── config/                 # 配置文件
├── images/                 # 图片文件
├── debug_scripts/          # 调试脚本
└── README.md               # 本文件
```

## 📋 详细说明

### 📚 docs/ - 文档目录

#### docs/analysis/ - 分析报告
- `CCF_A_TASK_ANALYSIS.md` - CCF-A任务分析
- `CLINICAL_BACKGROUND_ANALYSIS.md` - 临床背景分析
- `ORIGINAL_INNOVATION_ANALYSIS.md` - 原始创新分析
- `INNOVATION_FOCUS_SUMMARY.md` - 创新焦点总结
- `PROJECT_LOGIC_ANALYSIS.md` - 项目逻辑分析

#### docs/training/ - 训练相关文档
- `TRAINING_*.md` - 训练状态和报告
- `BATCH_SIZE_*.md` - 批次大小相关
- `MEMORY_ANALYSIS.md` - 内存分析
- `DATA_LOADING_FIXES.md` - 数据加载修复
- `CLASS_DETECTION_FIX_VERIFIED.md` - 类别检测修复验证

#### docs/dataset/ - 数据集相关文档
- `DATASET_*.md` - 数据集分析和设置
- `DATA_*.md` - 数据相关报告
- `LEAVE_CENTERS_OUT_SUMMARY.md` - Leave-Centers-Out总结
- `SCIENTIFIC_DATASET_SPLIT_SUMMARY.md` - 科学数据集划分总结

#### docs/miccai/ - MICCAI相关文档
- `MICCAI_*.md` - MICCAI论文相关文档
- `NEW_EXPERIMENT_PLAN_EVALUATION.md` - 新实验方案评估
- `DETAILED_EXPERIMENT_PLAN.md` - 详细实验计划

#### docs/project/ - 项目组织文档
- `FILE_*.md` - 文件组织相关
- `PROJECT_*.md` - 项目相关
- `*_CLEANUP_*.md` - 清理相关
- `REORGANIZATION_*.md` - 重组相关
- `DIRECTORY_*.md` - 目录相关

### 🔧 scripts/ - 脚本目录

#### scripts/dataset/ - 数据集处理脚本
- `create_*.py` - 创建数据集相关脚本
- `redesign_*.py` - 重新设计数据集脚本
- `check_dataset_integrity.py` - 检查数据集完整性

#### scripts/utils/ - 工具脚本
- `get_timestamp.py` - 获取时间戳
- `find_empty_dirs.py` - 查找空目录
- `move_*.py` - 文件移动脚本

#### scripts/visualization/ - 可视化脚本
- `visualize_*.py` - 可视化脚本
- `create_links_*.py` - 创建链接和可视化

### ⚙️ config/ - 配置文件
- `project_config.py` - 项目配置
- `visualization_config.py` - 可视化配置

### 🖼️ images/ - 图片文件
- `*.png` - PNG图片
- `*.pdf` - PDF文件

### 🐛 debug_scripts/ - 调试脚本
- `debug_data_loading.py` - 数据加载调试
- `quick_test_data_fix.py` - 快速测试数据修复

## 📝 使用说明

1. **查找文档**: 根据类型在对应的 `docs/` 子目录中查找
2. **运行脚本**: 脚本已按功能分类，在对应的 `scripts/` 子目录中
3. **配置文件**: 所有配置文件统一放在 `config/` 目录
4. **图片资源**: 所有图片文件放在 `images/` 目录

## 🔄 维护规则

- 新增文档请根据类型放入对应的 `docs/` 子目录
- 新增脚本请根据功能放入对应的 `scripts/` 子目录
- 保持目录结构清晰，避免在根目录直接放置文件
