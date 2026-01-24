#!/bin/bash
# 运行所有Lancet所需分析
# 包括：Bootstrap CI、亚组分析、DCA分析

cd "$(dirname "$0")/.." || exit 1

VENV_PYTHON="./my_retfound/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "错误: 虚拟环境 Python 不存在: $VENV_PYTHON"
    exit 1
fi

DATA_PATH="${1:-5centers_multi}"
OUTPUT_BASE="${2:-analysis/lancet_results}"

echo "🚀 开始运行Lancet所需的所有分析"
echo "  数据路径: ${DATA_PATH}"
echo "  输出目录: ${OUTPUT_BASE}"
echo ""

# 定义要分析的模型结果目录
declare -a MODEL_DIRS=(
    "models/SwinT/_results/multimodal:swint"
    "swin_large_results/swin_small_strong_aug:swint"
    "swin_large_results/swin_base_strong_aug:swint"
    "cnn_result_unified:cnn"
    "vmamba_result_unified:vmamba"
)

# 步骤1: 为每个模型生成预测结果
echo "📥 步骤1: 为所有模型生成预测结果..."
echo "=" * 60

for model_info in "${MODEL_DIRS[@]}"; do
    IFS=':' read -r result_dir model_type <<< "$model_info"
    
    if [ ! -d "$result_dir" ]; then
        echo "⚠️  跳过不存在的目录: $result_dir"
        continue
    fi
    
    if [ ! -f "$result_dir/best_model.pth" ]; then
        echo "⚠️  跳过没有模型的目录: $result_dir"
        continue
    fi
    
    echo ""
    echo "处理: $result_dir (模型类型: $model_type)"
    
    # 检查是否已有预测结果
    if [ -f "$result_dir/val_labels.npy" ] && [ -f "$result_dir/val_probs.npy" ]; then
        echo "✅ 预测结果已存在，跳过生成"
    else
        echo "🔄 生成预测结果..."
        CUDA_VISIBLE_DEVICES=0 "$VENV_PYTHON" analysis/generate_predictions.py \
            --result_dir "$result_dir" \
            --data_path "$DATA_PATH" \
            --model_type "$model_type" \
            --device cuda
        
        if [ $? -ne 0 ]; then
            echo "❌ 生成预测结果失败: $result_dir"
            continue
        fi
    fi
done

echo ""
echo "✅ 步骤1完成：所有模型的预测结果已生成"
echo ""

# 步骤2: 运行所有分析
echo "📊 步骤2: 运行所有分析（Bootstrap CI、亚组分析、DCA）..."
echo "=" * 60

for model_info in "${MODEL_DIRS[@]}"; do
    IFS=':' read -r result_dir model_type <<< "$model_info"
    
    if [ ! -d "$result_dir" ]; then
        continue
    fi
    
    if [ ! -f "$result_dir/val_labels.npy" ]; then
        echo "⚠️  跳过没有预测结果的目录: $result_dir"
        continue
    fi
    
    # 提取模型名称
    model_name=$(basename "$result_dir")
    output_dir="${OUTPUT_BASE}/${model_name}"
    
    echo ""
    echo "分析模型: $model_name"
    echo "  结果目录: $result_dir"
    echo "  输出目录: $output_dir"
    
    # 运行综合分析
    "$VENV_PYTHON" analysis/run_lancet_analysis.py \
        --result_dir "$result_dir" \
        --data_path "$DATA_PATH" \
        --output_dir "$output_dir" \
        --model_name "$model_name"
    
    if [ $? -eq 0 ]; then
        echo "✅ 分析完成: $model_name"
    else
        echo "❌ 分析失败: $model_name"
    fi
done

echo ""
echo "=" * 60
echo "✅ 所有分析完成！"
echo "   结果保存在: ${OUTPUT_BASE}/"
echo ""

