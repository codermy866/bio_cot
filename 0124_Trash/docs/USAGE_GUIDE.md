# 🎯 柳叶刀论文分析脚本使用指南

## 🚀 快速开始

### 方法1: 一键运行所有分析（推荐）
```bash
# 激活虚拟环境并运行完整分析
bash run_lancet_analysis.sh
```

### 方法2: 使用批次实验运行器
```bash
# 激活虚拟环境
source my_retfound/bin/activate

# 运行批次实验
python batch_experiment_runner.py --config experiment_config.yaml --parallel
```

### 方法3: 单独运行各分析脚本
```bash
# 激活虚拟环境
source my_retfound/bin/activate

# 分中心评估
python center_evaluation.py --model_path cnn_training_latest/best_model.pth --data_path 5centers_multi --model_type cnn --clinical_dim 8 --output_dir center_evaluation_output

# 决策曲线分析
python decision_curve_analysis.py --model_paths cnn_training_latest/best_model.pth --data_path 5centers_multi --model_types cnn --output_dir dca_output

# 不确定性分析
python uncertainty_analysis.py --model_paths cnn_training_latest/best_model.pth --data_path 5centers_multi --model_type cnn --clinical_dim 8 --output_dir uncertainty_output

# 缺失模态评估
python missing_modality_evaluation.py --model_path cnn_training_latest/best_model.pth --data_path 5centers_multi --model_type cnn --clinical_dim 8 --output_dir missing_modality_output

# 可解释性分析
python interpretability_analysis.py --model_path cnn_training_latest/best_model.pth --data_path 5centers_multi --model_type cnn --clinical_dim 8 --output_dir interpretability_output --n_cases 10
```

## 📊 输出文件说明

### 分中心评估 (`center_evaluation_output/`)
- `forest_plot.png`: 森林图 - 显示各中心AUC及置信区间
- `center_summary.csv`: 分中心汇总表格
- `results.json`: 详细结果数据

### 决策曲线分析 (`dca_output/`)
- `decision_curve.png`: 决策曲线图 - 显示不同阈值下的净获益
- `cost_effectiveness.png`: 成本-效果分析图
- `dca_summary.csv`: DCA汇总表格
- `dca_results.json`: 详细结果数据

### 不确定性分析 (`uncertainty_output/`)
- `uncertainty_analysis.png`: 不确定性分析图
- `reliability_diagram.png`: 可靠性图 - 显示校准性能
- `uncertainty_results.json`: 详细结果数据

### 缺失模态评估 (`missing_modality_output/`)
- `missing_modality_analysis.png`: 缺失模态分析图
- `missing_modality_summary.csv`: 缺失模态汇总表格
- `missing_modality_results.json`: 详细结果数据

### 可解释性分析 (`interpretability_output/`)
- `attention_case_*.png`: 各病例的注意力可视化
- `shap_case_*.png`: 各病例的SHAP分析
- `interpretability_summary.txt`: 可解释性汇总报告
- `interpretability_results.json`: 详细结果数据

### 论文配图 (`paper_figures/`)
- `roc_curve.png`: ROC曲线图
- `precision_recall_curve.png`: 精确率-召回率曲线图
- `confusion_matrix.png`: 混淆矩阵图
- `performance_comparison.png`: 性能对比图
- `learning_curves.png`: 学习曲线图
- `statistical_plots.png`: 统计图表
- 各种表格文件 (CSV和LaTeX格式)

## 🔧 配置说明

### 模型配置
- **模型路径**: `cnn_training_latest/best_model.pth` (已自动配置)
- **模型类型**: `cnn` 或 `vmamba`
- **临床特征维度**: `8`

### 数据配置
- **数据路径**: `5centers_multi`
- **输入图像尺寸**: `224x224`
- **OCT帧数**: `48`
- **OCT缓存目录**: `oct_cache`

### 实验配置
- **可解释性分析病例数**: `10`
- **缺失模态缺失率**: `[0.0, 0.1, 0.2, 0.3, 0.5]`
- **Conformal prediction显著性水平**: `0.1`
- **校准评估分箱数量**: `10`

## 📈 关键指标说明

### 性能指标
- **AUC**: 主要性能指标，>0.8为优秀
- **敏感性/特异性**: 临床重要指标
- **F1-Score**: 精确率和召回率的调和平均
- **Brier Score**: 概率预测的准确性

### 临床指标
- **净获益**: 决策曲线分析的核心指标
- **覆盖率**: Conformal prediction的置信度
- **校准误差**: 模型预测的可靠性
- **不确定性**: 模型预测的置信度

## 🎨 论文撰写建议

### 结果展示顺序
1. **分中心评估结果** - 多中心验证，证明泛化性
2. **决策曲线分析** - 临床决策支持，证明实用性
3. **不确定性分析** - 模型可靠性，证明可信度
4. **缺失模态鲁棒性** - 实际应用场景，证明鲁棒性
5. **可解释性分析** - 模型可解释性，证明透明度

### 图表要求
- **分辨率**: 300 DPI
- **格式**: PNG/PDF
- **颜色**: 柳叶刀期刊标准色彩
- **字体**: Arial或类似无衬线字体
- **图例**: 清晰明确
- **统计标注**: 显著性水平

### 表格要求
- **格式**: CSV和LaTeX
- **精度**: 3位小数
- **置信区间**: 95% CI
- **统计检验**: p值标注

## 🚨 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 解决方案：使用CPU模式或减少batch_size
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **数据加载错误**
   ```bash
   # 检查数据路径和文件权限
   ls -la 5centers_multi/
   ```

3. **模型加载失败**
   ```bash
   # 检查模型文件完整性
   python -c "import torch; print(torch.load('cnn_training_latest/best_model.pth').keys())"
   ```

4. **依赖包缺失**
   ```bash
   # 激活虚拟环境并安装
   source my_retfound/bin/activate
   pip install -r requirements_enhanced.txt
   ```

### 调试模式
```bash
# 启用详细日志
export PYTHONPATH=$PYTHONPATH:.
python -u script_name.py --verbose
```

## 📞 技术支持

### 日志文件
- `batch_experiments.log`: 批次实验日志
- `experiment_summary.txt`: 实验汇总报告
- `test_scripts.log`: 测试脚本日志

### 检查脚本
```bash
# 运行完整测试
python test_scripts.py

# 运行快速测试
python quick_test.py
```

## 🎯 下一步行动

1. **立即开始**: 运行 `bash run_lancet_analysis.sh`
2. **检查结果**: 查看生成的图表和表格
3. **调整参数**: 根据结果优化模型或参数
4. **准备论文**: 使用生成的配图和表格
5. **投稿准备**: 确保符合柳叶刀期刊要求

## 📚 相关资源

- **README.md**: 详细使用说明
- **experiment_config.yaml**: 配置文件模板
- **requirements_enhanced.txt**: 依赖包列表
- **test_scripts.py**: 完整测试脚本
- **quick_test.py**: 快速功能测试

---

**🎉 恭喜！你现在拥有了完整的柳叶刀级别论文分析工具包！**

**💡 提示**: 建议先运行 `bash run_lancet_analysis.sh` 体验完整流程，然后根据需要调整参数或运行单个分析脚本。
