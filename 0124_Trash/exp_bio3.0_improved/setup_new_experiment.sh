#!/bin/bash
# 创建新的实验文件夹并整理最新的log和可视化图片

cd /data2/hmy/VLM_Caus_Rm_Mics/experiments

# 创建新的实验文件夹
NEW_EXP_DIR="exp_bio3.0_improved"
mkdir -p "$NEW_EXP_DIR"/{logs,checkpoints,results,visualizations}

echo "=========================================="
echo "创建新的实验文件夹: $NEW_EXP_DIR"
echo "=========================================="

# 源文件夹
SOURCE_DIR="exp_bio3.0"

# 1. 复制最新的日志文件（20260113的）
echo ""
echo "📋 复制最新的日志文件..."
find "$SOURCE_DIR/logs" -type f -name "*20260113*" -o -name "*20260113*" | while read file; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        cp "$file" "$NEW_EXP_DIR/logs/"
        echo "  ✅ $filename"
    fi
done

# 2. 复制最新的可视化图片（PDF和PNG）
echo ""
echo "📊 复制最新的可视化图片..."

# PDF文件
find "$SOURCE_DIR/logs" -type f -name "*.pdf" | while read file; do
    filename=$(basename "$file")
    cp "$file" "$NEW_EXP_DIR/visualizations/"
    echo "  ✅ $filename (PDF)"
done

# PNG文件（最新的）
find "$SOURCE_DIR/logs" -type f -name "*.png" | while read file; do
    filename=$(basename "$file")
    cp "$file" "$NEW_EXP_DIR/visualizations/"
    echo "  ✅ $filename (PNG)"
done

# 3. 复制数据文件
echo ""
echo "💾 复制数据文件..."
find "$SOURCE_DIR/logs" -type f \( -name "*.csv" -o -name "*.pkl" -o -name "*.json" -o -name "*.txt" \) | while read file; do
    filename=$(basename "$file")
    cp "$file" "$NEW_EXP_DIR/logs/"
    echo "  ✅ $filename"
done

# 4. 复制最新的模型检查点
echo ""
echo "💾 复制最新的模型检查点..."
LATEST_CHECKPOINT=$(find "$SOURCE_DIR/checkpoints" -name "best_model_v3_*.pth" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
if [ -n "$LATEST_CHECKPOINT" ] && [ -f "$LATEST_CHECKPOINT" ]; then
    cp "$LATEST_CHECKPOINT" "$NEW_EXP_DIR/checkpoints/"
    echo "  ✅ $(basename $LATEST_CHECKPOINT)"
else
    echo "  ⚠️ 未找到最新的检查点"
fi

# 5. 复制配置文件
echo ""
echo "⚙️ 复制配置文件..."
cp "$SOURCE_DIR/config.py" "$NEW_EXP_DIR/" 2>/dev/null && echo "  ✅ config.py" || echo "  ⚠️ config.py 不存在"

# 6. 创建README
cat > "$NEW_EXP_DIR/README.md" << 'EOF'
# Bio-COT 3.0 Improved Experiment

## 📁 文件夹结构

```
exp_bio3.0_improved/
├── logs/              # 日志文件、数据文件
├── checkpoints/       # 模型检查点
├── results/           # 实验结果
├── visualizations/    # 可视化图片（PDF和PNG）
└── README.md         # 本文件
```

## 📊 实验信息

### 实验日期
- 创建时间: 2025-01-13
- 数据来源: exp_bio3.0

### 关键发现
1. **分类性能问题**: 初始准确率只有53.57%
2. **最优阈值**: 0.580（而非默认0.5）
3. **性能提升**: 使用最优阈值后，准确率提升到73.81%

### 主要改进
- ✅ 找到最优决策阈值（0.580）
- ✅ 性能提升：准确率 +20.24%，特异性 +41.59%
- ✅ 生成了完整的可视化分析

## 📈 性能指标

### 当前性能（阈值=0.580）
- **准确率 (Accuracy)**: 73.81%
- **精确率 (Precision)**: 58.46%
- **召回率 (Recall)**: 69.09%
- **特异性 (Specificity)**: 76.11%
- **F1分数**: 63.33%
- **ROC AUC**: 79.80%

### 改进前性能（阈值=0.5）
- **准确率**: 53.57%
- **精确率**: 40.80%
- **召回率**: 92.73%
- **特异性**: 34.51%
- **F1分数**: 56.67%

## 📁 文件说明

### 日志文件
- `training_history_*.json`: 训练历史记录
- `train_bio_cot_v3_*.log`: 训练日志
- `optimal_threshold.txt`: 最优阈值

### 数据文件
- `visualization_data_*.pkl`: 完整特征数据
- `visualization_data_*.csv`: CSV格式数据
- `feature_importance_data_*.csv`: 特征重要性数据
- `center_comparison_data_*.csv`: 中心对比数据

### 可视化文件
- `roc_pr_curves_*.pdf`: ROC和PR曲线
- `confusion_matrix_*.pdf`: 混淆矩阵
- `calibration_curve_*.pdf`: 校准曲线
- `feature_importance_*.pdf`: 特征重要性分析
- `center_comparison_*.pdf`: 中心性能对比
- `error_analysis_*.pdf`: 错误分析
- `causal_vs_noise_*.pdf`: 因果vs噪声特征对比
- `training_curves_detailed_*.pdf`: 详细训练曲线
- `attention_evolution_*.pdf`: 注意力演化分析
- `feature_correlation_*.pdf`: 特征相关性热图
- `decision_boundary_*.pdf`: 决策边界可视化
- `tsne_umap_3d_*.pdf`: 3D t-SNE/UMAP可视化
- `distribution_3d_*.pdf`: 3D分布可视化
- `classification_improvement_comparison.pdf`: 性能改进对比图

## 🎯 下一步改进方向

1. **调整损失函数权重**
   - 增加分类损失权重（lambda_cls从1.0增加到2.0）
   - 减少其他损失权重

2. **调整类别权重**
   - 在Focal Loss中调整alpha和gamma参数
   - 处理类别不平衡问题

3. **重新训练模型**
   - 使用调整后的参数重新训练
   - 预期准确率可提升到75-80%

## 📝 相关文档

- `../exp_bio3.0/CLASSIFICATION_PERFORMANCE_ANALYSIS.md`: 详细性能分析
- `../exp_bio3.0/ALL_VISUALIZATIONS_SUMMARY.md`: 所有可视化类型总结

---
**最后更新**: 2025-01-13
EOF

echo ""
echo "✅ README.md 已创建"

# 7. 创建文件清单
echo ""
echo "📝 创建文件清单..."
cat > "$NEW_EXP_DIR/FILE_LIST.md" << 'EOF'
# 文件清单

## 日志文件 (logs/)
EOF

echo "## 日志文件 (logs/)" > "$NEW_EXP_DIR/FILE_LIST.md"
find "$NEW_EXP_DIR/logs" -type f -printf "  - %f\n" | sort >> "$NEW_EXP_DIR/FILE_LIST.md"

echo "" >> "$NEW_EXP_DIR/FILE_LIST.md"
echo "## 可视化文件 (visualizations/)" >> "$NEW_EXP_DIR/FILE_LIST.md"
find "$NEW_EXP_DIR/visualizations" -type f -printf "  - %f\n" | sort >> "$NEW_EXP_DIR/FILE_LIST.md"

echo "" >> "$NEW_EXP_DIR/FILE_LIST.md"
echo "## 检查点文件 (checkpoints/)" >> "$NEW_EXP_DIR/FILE_LIST.md"
find "$NEW_EXP_DIR/checkpoints" -type f -printf "  - %f\n" | sort >> "$NEW_EXP_DIR/FILE_LIST.md"

echo "✅ FILE_LIST.md 已创建"

# 8. 统计信息
echo ""
echo "=========================================="
echo "📊 统计信息"
echo "=========================================="
echo "日志文件数量: $(find "$NEW_EXP_DIR/logs" -type f | wc -l)"
echo "可视化文件数量: $(find "$NEW_EXP_DIR/visualizations" -type f | wc -l)"
echo "检查点文件数量: $(find "$NEW_EXP_DIR/checkpoints" -type f | wc -l)"
echo ""
echo "总文件大小:"
du -sh "$NEW_EXP_DIR" 2>/dev/null || echo "无法计算"

echo ""
echo "=========================================="
echo "✅ 新实验文件夹创建完成！"
echo "=========================================="
echo "位置: $(pwd)/$NEW_EXP_DIR"
echo ""
echo "主要文件:"
echo "  - README.md: 实验说明"
echo "  - FILE_LIST.md: 文件清单"
echo "  - logs/: 日志和数据文件"
echo "  - visualizations/: 可视化图片"
echo "  - checkpoints/: 模型检查点"
echo ""

