#!/bin/bash
# 简单的外部验证评估脚本
# 专为技术小白设计，一键运行，自动完成所有步骤

cd "$(dirname "$0")/.." || exit 1

echo "=" | tr -d '\n' | head -c 80 && echo ""
echo "🚀 开始外部验证评估"
echo "=" | tr -d '\n' | head -c 80 && echo ""
echo ""

# 检查虚拟环境
VENV_PYTHON="./my_retfound/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ 错误: 虚拟环境不存在"
    echo "   请确保虚拟环境在: ./my_retfound/bin/python"
    exit 1
fi

# 设置参数
MODEL_PATH="models/SwinT/_results/multimodal/best_model.pth"
EXTERNAL_DATA_PATH="5centers_multi_internal_external_recommended/external_validation"
OUTPUT_DIR="analysis/external_validation_results"
MODEL_TYPE="swint"
DEVICE="cuda"

echo "📋 评估参数:"
echo "   模型路径: $MODEL_PATH"
echo "   外部验证数据: $EXTERNAL_DATA_PATH"
echo "   输出目录: $OUTPUT_DIR"
echo "   模型类型: $MODEL_TYPE"
echo "   设备: $DEVICE"
echo ""

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型文件不存在: $MODEL_PATH"
    echo "   请先训练模型或检查模型路径"
    exit 1
fi

# 检查外部验证数据
if [ ! -d "$EXTERNAL_DATA_PATH" ]; then
    echo "❌ 错误: 外部验证数据目录不存在: $EXTERNAL_DATA_PATH"
    echo "   请先运行数据分割脚本"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "🔄 开始评估..."
echo ""

# 运行评估脚本
CUDA_VISIBLE_DEVICES=0 "$VENV_PYTHON" analysis/external_validation_evaluation.py \
    --model_path "$MODEL_PATH" \
    --external_data_path "$EXTERNAL_DATA_PATH" \
    --model_type "$MODEL_TYPE" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    2>&1 | tee "$OUTPUT_DIR/evaluation.log"

if [ $? -eq 0 ]; then
    echo ""
    echo "=" | tr -d '\n' | head -c 80 && echo ""
    echo "✅ 外部验证评估完成！"
    echo "=" | tr -d '\n' | head -c 80 && echo ""
    echo ""
    echo "📊 结果文件:"
    echo "   - 评估报告: $OUTPUT_DIR/external_validation_report.md"
    echo "   - 详细结果: $OUTPUT_DIR/external_validation_results.json"
    echo "   - 评估日志: $OUTPUT_DIR/evaluation.log"
    echo ""
    echo "📈 主要指标:"
    if [ -f "$OUTPUT_DIR/external_validation_results.json" ]; then
        "$VENV_PYTHON" -c "
import json
import sys
try:
    with open('$OUTPUT_DIR/external_validation_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    if 'auc' in results:
        print(f\"   AUC: {results['auc'].get('formatted', 'N/A')}\")
    if 'sensitivity' in results:
        print(f\"   敏感性: {results['sensitivity'].get('formatted', 'N/A')}\")
    if 'specificity' in results:
        print(f\"   特异性: {results['specificity'].get('formatted', 'N/A')}\")
except Exception as e:
    print(f\"   ⚠️  无法读取结果文件: {e}\")
"
    fi
    echo ""
    echo "💡 提示: 查看详细结果请打开: $OUTPUT_DIR/external_validation_report.md"
else
    echo ""
    echo "❌ 评估失败，请查看日志: $OUTPUT_DIR/evaluation.log"
    exit 1
fi

