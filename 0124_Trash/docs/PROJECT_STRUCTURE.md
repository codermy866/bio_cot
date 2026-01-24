# 项目文件结构说明

## 📁 目录结构

```
HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/
├── training/              # 训练脚本目录
│   ├── optimized_vmamba_training.py      # VMamba模型训练脚本
│   ├── optimized_2class_training.py      # CNN模型训练脚本
│   └── advanced_training_pipeline.py    # 高级训练流程
│
├── analysis/              # 分析脚本和报告目录
│   ├── analyze_training_results.py       # 训练结果分析脚本
│   ├── diagnose_overfitting.py          # 过拟合诊断工具
│   └── training_analysis_report.md       # 训练分析报告
│
├── visualization/         # 可视化脚本目录
│   ├── generate_advanced_visualizations.py      # 高级可视化生成
│   ├── generate_training_plots.py               # 训练图表生成
│   ├── advanced_visualizations.py               # 高级可视化工具
│   ├── advanced_visualizations_from_results.py  # 从结果生成可视化
│   └── paper_figure_generator.py               # 论文图表生成器
│
├── models/               # 模型定义目录
│   ├── vmamba_multimodal_model.py    # VMamba多模态模型
│   └── cnn_multimodal_model.py       # CNN多模态模型
│
├── scripts/              # 启动脚本目录
│   ├── run_cnn_enhanced.py           # CNN增强训练启动脚本
│   ├── start_cnn_training.sh         # CNN训练启动Shell脚本
│   └── optimize_to_90_percent.py     # 优化到90%准确率脚本
│
├── utils/                # 工具脚本目录
│   ├── advanced_clinical_metrics.py  # 临床指标计算
│   ├── fix_csv_encoding.py           # CSV编码修复
│   ├── temperature_scaling.py         # 温度缩放校准
│   ├── analyze_weights.py            # 权重分析
│   ├── comprehensive_diagnostic_report.py  # 综合诊断报告
│   ├── create_merged_classification.py    # 创建合并分类
│   ├── transfer_2class_to_5class.py      # 二分类转五分类
│   ├── visualize_causal_graph.py         # 因果图可视化
│   ├── enhanced_data_processing.py       # 增强数据处理
│   ├── enhanced_model_architecture.py     # 增强模型架构
│   ├── enhanced_multimodal_dataset.py     # 增强多模态数据集
│   ├── enhanced_oct_processing.py        # 增强OCT处理
│   └── test_*.py                          # 各种测试脚本
│
├── reports/              # 报告文件目录（可选）
│
├── vmamba_result/        # VMamba训练结果
├── vmamba_result_120/   # VMamba 120帧训练结果
├── vmamba_result_large/ # VMamba大模型训练结果
├── cnn_result/          # CNN训练结果
├── cnn_result_120/      # CNN 120帧训练结果
├── cnn_result_large/    # CNN大模型训练结果
└── cnn_result_enhanced/ # CNN增强模型训练结果
```

## 📋 使用说明

### 训练脚本

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

### 分析脚本

```bash
# 分析所有训练结果
python analysis/analyze_training_results.py

# 诊断过拟合
python analysis/diagnose_overfitting.py --model_path path/to/model
```

### 可视化脚本

```bash
# 生成训练图表
python visualization/generate_training_plots.py --results_dir ../results

# 生成论文图表
python visualization/paper_figure_generator.py
```

### 启动脚本

```bash
# 启动CNN增强训练
python scripts/run_cnn_enhanced.py

# 或使用Shell脚本
bash scripts/start_cnn_training.sh
```

## 🔧 导入路径说明

由于文件已分类到不同目录，导入路径需要相应更新：

```python
# 导入模型
from models.vmamba_multimodal_model import VMambaMultimodalTransformer
from models.cnn_multimodal_model import CNNMultimodalTransformer

# 导入工具
from utils.advanced_clinical_metrics import calculate_clinical_metrics
from utils.enhanced_multimodal_dataset import EnhancedMultimodalDataset
```

## 📝 注意事项

1. **训练脚本**：需要在项目根目录运行，因为使用了相对路径（如`../5centers_multi`）
2. **导入路径**：已自动更新训练脚本中的导入路径
3. **结果目录**：训练结果保存在对应的`*_result/`目录下
4. **备份文件**：移动文件前会备份已存在的目标文件（添加`.bak`后缀）

## 🔄 文件整理

运行以下命令整理文件：

```bash
python organize_files.py
```

脚本会自动：
1. 创建分类目录
2. 移动文件到对应目录
3. 创建README说明文件
4. 更新导入路径

