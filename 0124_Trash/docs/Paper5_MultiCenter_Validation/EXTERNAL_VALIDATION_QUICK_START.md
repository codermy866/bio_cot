# 外部验证快速开始指南

## 📋 概述

本指南提供外部验证数据收集与评估的快速开始步骤。

## 🚀 快速开始

### 1. 数据收集准备

#### 1.1 检查数据质量
```bash
# 使用虚拟环境
cd /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713
source my_retfound/bin/activate

# 运行数据质量检查
python utils/external_data_quality_check.py \
    --data_root /path/to/external_validation_data \
    --output_file external_validation_quality_report.json
```

#### 1.2 数据组织
确保数据按以下结构组织：
```
external_validation_data/
├── center_1/
│   ├── images/
│   │   ├── oct/
│   │   │   └── patient_001/
│   │   │       ├── point_01/
│   │   │       │   ├── frame_001.jpg
│   │   │       │   └── ...
│   │   │       └── ...
│   │   └── colposcopy/
│   │       └── patient_001/
│   │           ├── image_001.jpg
│   │           └── ...
│   ├── metadata.csv
│   └── labels.csv
├── center_2/
│   └── ...
└── center_3/
    └── ...
```

### 2. 运行外部验证评估

#### 2.1 基本评估
```bash
python analysis/external_validation_evaluation.py \
    --model_path models/SwinT/_results/multimodal/best_model.pth \
    --external_data_path /path/to/external_validation_data \
    --model_type swint \
    --output_dir analysis/external_validation_results \
    --device cuda
```

#### 2.2 与训练集比较
```bash
python analysis/external_validation_evaluation.py \
    --model_path models/SwinT/_results/multimodal/best_model.pth \
    --external_data_path /path/to/external_validation_data \
    --model_type swint \
    --training_results models/SwinT/_results/multimodal/metrics.json \
    --output_dir analysis/external_validation_results \
    --device cuda
```

### 3. 查看结果

评估完成后，结果保存在 `analysis/external_validation_results/` 目录下：

- `external_validation_results.json`: 完整评估结果（JSON格式）
- `external_validation_report.md`: 可读性报告（Markdown格式）
- `subgroup_analysis.csv`: 亚组分析结果
- `decision_curve.png`: 决策曲线图
- `forest_plot_*.png`: 亚组分析森林图

## 📊 结果解读

### 性能指标（95% CI）

主要关注：
- **AUC**: ≥ 0.75（95% CI下限 ≥ 0.70）
- **敏感性**: ≥ 0.70
- **特异性**: ≥ 0.70
- **性能下降**: AUC下降 < 0.05（相比训练集）

### 与训练集比较

- **可接受**: AUC下降 < 0.05
- **警告**: AUC下降 0.05-0.10
- **失败**: AUC下降 > 0.10

### 与临床基线比较

模型应优于或至少不劣于：
- HPV单独
- TCT单独
- HPV+TCT联合

## ⚠️ 常见问题

### 1. 数据加载失败
- 检查数据路径是否正确
- 检查图像格式是否支持（JPG/PNG）
- 检查元数据CSV格式是否正确

### 2. 模型加载失败
- 检查模型路径是否正确
- 检查模型类型是否匹配
- 尝试使用非严格模式加载（已在脚本中实现）

### 3. 性能指标异常
- 检查标签是否正确
- 检查预测概率范围（应在0-1之间）
- 检查样本量是否足够（≥100例）

## 📝 下一步

1. **数据收集**: 按照 `docs/EXTERNAL_VALIDATION_PLAN.md` 中的要求收集数据
2. **质量检查**: 运行数据质量检查脚本
3. **评估**: 运行外部验证评估脚本
4. **报告**: 查看评估报告，准备发表材料

## 🔗 相关文档

- [完整方案文档](EXTERNAL_VALIDATION_PLAN.md)
- [Lancet发表评估](LANCET_PUBLICATION_ASSESSMENT.md)

---

**最后更新**: 2025-01-XX


