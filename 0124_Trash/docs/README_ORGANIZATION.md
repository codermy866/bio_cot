# 项目文件整理说明

## 📋 概述

本项目已按功能分类整理文件，所有相关文件已归类到对应的文件夹中，方便管理和查找。

## 📁 目录结构

```
HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/
│
├── training/              # 📝 训练脚本目录
│   ├── optimized_vmamba_training.py      # VMamba模型训练脚本
│   ├── optimized_2class_training.py      # CNN模型训练脚本
│   ├── advanced_training_pipeline.py     # 高级训练流程
│   └── README.md                         # 训练脚本说明
│
├── analysis/              # 📊 分析脚本和报告目录
│   ├── analyze_training_results.py       # 训练结果综合分析
│   ├── diagnose_overfitting.py          # 过拟合诊断工具
│   ├── training_analysis_report.md       # 训练分析报告
│   └── README.md                         # 分析工具说明
│
├── visualization/         # 📈 可视化脚本目录
│   ├── generate_advanced_visualizations.py      # 高级可视化生成
│   ├── generate_training_plots.py               # 训练图表生成
│   ├── advanced_visualizations.py               # 高级可视化工具
│   ├── advanced_visualizations_from_results.py  # 从结果生成可视化
│   ├── paper_figure_generator.py               # 论文图表生成器
│   └── README.md                                 # 可视化工具说明
│
├── models/               # 🏗️ 模型定义目录
│   ├── vmamba_multimodal_model.py    # VMamba多模态模型
│   ├── cnn_multimodal_model.py       # CNN多模态模型
│   └── README.md                      # 模型说明
│
├── scripts/              # 🚀 启动脚本目录
│   ├── run_cnn_enhanced.py           # CNN增强训练启动脚本
│   ├── start_cnn_training.sh         # CNN训练启动Shell脚本
│   ├── optimize_to_90_percent.py     # 优化到90%准确率脚本
│   └── README.md                      # 启动脚本说明
│
├── utils/                # 🛠️ 工具脚本目录
│   ├── advanced_clinical_metrics.py  # 临床指标计算
│   ├── enhanced_data_processing.py   # 增强数据处理
│   ├── enhanced_model_architecture.py # 增强模型架构
│   ├── enhanced_multimodal_dataset.py # 增强多模态数据集
│   ├── enhanced_oct_processing.py    # 增强OCT处理
│   ├── fix_csv_encoding.py           # CSV编码修复
│   ├── temperature_scaling.py         # 温度缩放校准
│   ├── analyze_weights.py            # 权重分析
│   ├── comprehensive_diagnostic_report.py  # 综合诊断报告
│   ├── create_merged_classification.py     # 创建合并分类
│   ├── transfer_2class_to_5class.py      # 二分类转五分类
│   ├── visualize_causal_graph.py         # 因果图可视化
│   ├── test_*.py                          # 各种测试脚本
│   └── README.md                           # 工具脚本说明
│
├── reports/              # 📄 报告文件目录（可选）
│
├── vmamba_result/        # VMamba训练结果
├── vmamba_result_120/    # VMamba 120帧训练结果
├── vmamba_result_large/  # VMamba大模型训练结果
├── cnn_result/           # CNN训练结果
├── cnn_result_120/       # CNN 120帧训练结果
├── cnn_result_large/     # CNN大模型训练结果
└── cnn_result_enhanced/  # CNN增强模型训练结果
```

## 🔧 使用方法

### 1. 运行文件整理脚本

首次整理文件时，运行：

```bash
cd /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713
python3 organize_files.py
```

脚本会自动：
- ✅ 创建分类目录（training, analysis, visualization, models, scripts, utils）
- ✅ 移动文件到对应目录
- ✅ 创建每个目录的README说明文件
- ✅ 更新训练脚本中的导入路径

### 2. 训练模型

```bash
# VMamba模型训练
python training/optimized_vmamba_training.py \
    --data_path ../5centers_multi \
    --epochs 20 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --output_dir ../vmamba_result

# CNN模型训练
python training/optimized_2class_training.py \
    --data_path ../5centers_multi \
    --epochs 20 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --output_dir ../cnn_result
```

### 3. 分析训练结果

```bash
# 分析所有训练结果
python analysis/analyze_training_results.py

# 诊断过拟合
python analysis/diagnose_overfitting.py --model_path path/to/model
```

### 4. 生成可视化

```bash
# 生成训练图表
python visualization/generate_training_plots.py --results_dir ../results

# 生成论文图表
python visualization/paper_figure_generator.py
```

## 📝 导入路径说明

由于文件已分类到不同目录，导入路径需要相应更新：

### 训练脚本中的导入

```python
# 导入模型
from models.vmamba_multimodal_model import VMambaMultimodalTransformer
from models.cnn_multimodal_model import CNNMultimodalTransformer

# 导入工具
from utils.advanced_clinical_metrics import calculate_clinical_metrics
from utils.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
from utils.enhanced_oct_processing import EnhancedOCTProcessor
```

### sys.path设置

如果需要在训练脚本中导入，可以添加：

```python
import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 然后可以正常导入
from models.vmamba_multimodal_model import VMambaMultimodalTransformer
from utils.advanced_clinical_metrics import calculate_clinical_metrics
```

## ⚠️ 注意事项

1. **运行位置**：训练脚本需要在项目根目录运行，因为使用了相对路径（如`../5centers_multi`）
2. **导入路径**：整理脚本会自动更新训练脚本中的导入路径
3. **备份文件**：如果目标文件已存在，会先备份（添加`.bak`后缀）
4. **结果目录**：训练结果保存在对应的`*_result/`目录下

## 📚 详细说明

每个分类目录下都有对应的`README.md`文件，包含：
- 目录文件说明
- 使用方法
- 注意事项

请查看各目录下的README文件获取详细信息。

## 🔄 文件整理规则

整理脚本按照以下规则分类文件：

| 分类 | 文件类型 | 说明 |
|------|---------|------|
| training | 训练脚本 | 所有模型训练相关的Python脚本 |
| analysis | 分析脚本 | 训练结果分析和诊断工具 |
| visualization | 可视化脚本 | 图表生成和可视化工具 |
| models | 模型定义 | 模型架构定义文件 |
| scripts | 启动脚本 | 训练启动脚本和工具脚本 |
| utils | 工具脚本 | 数据处理、工具函数等 |

## 📊 文件统计

整理完成后，项目结构更加清晰：
- ✅ 训练脚本：3个文件
- ✅ 分析脚本：3个文件
- ✅ 可视化脚本：5个文件
- ✅ 模型定义：2个文件
- ✅ 启动脚本：3个文件
- ✅ 工具脚本：15+个文件

总计：约30+个文件已分类整理

