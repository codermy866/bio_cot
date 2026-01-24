#!/bin/bash
# 自动化Lancet全面分析脚本
# 等待训练完成后自动执行所有必需的分析

cd "$(dirname "$0")/.." || exit 1

VENV_PYTHON="./my_retfound/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "错误: 虚拟环境 Python 不存在: $VENV_PYTHON"
    exit 1
fi

echo "="*80
echo "🚀 自动化Lancet全面分析"
echo "="*80
echo ""
echo "本脚本将自动执行："
echo "  1. 等待训练完成（检查模型文件）"
echo "  2. 内部测试集性能评估（使用最优阈值）"
echo "  3. 外部验证集性能评估"
echo "  4. 内部/外部数据集对比分析"
echo "  5. Bootstrap置信区间计算"
echo "  6. 亚组分析"
echo "  7. 决策曲线分析（DCA）"
echo "  8. 生成完整报告"
echo ""
echo "="*80
echo ""

# 创建输出目录
mkdir -p analysis/lancet_comprehensive_analysis

# 模型配置
MODELS=(
    "models/SwinT/_results/multimodal:swint:Swin-T"
    "cnn_result_unified:cnn:CNN"
    "vmamba_result_unified:vmamba:VMamba"
)

# 检查模型是否可用
check_model_ready() {
    local model_dir=$1
    local model_file="$model_dir/best_model.pth"
    
    if [ -f "$model_file" ]; then
        return 0
    else
        return 1
    fi
}

# 等待模型训练完成（最多等待24小时）
wait_for_models() {
    echo "⏳ 等待模型训练完成..."
    local max_wait=86400  # 24小时
    local check_interval=300  # 5分钟
    local waited=0
    
    while [ $waited -lt $max_wait ]; do
        local all_ready=true
        
        for model_config in "${MODELS[@]}"; do
            IFS=':' read -r model_dir model_type model_name <<< "$model_config"
            
            if ! check_model_ready "$model_dir"; then
                all_ready=false
                echo "  ⏳ $model_name 还在训练中..."
                break
            fi
        done
        
        if [ "$all_ready" = true ]; then
            echo "✅ 所有模型训练完成！"
            return 0
        fi
        
        sleep $check_interval
        waited=$((waited + check_interval))
        echo "  已等待: $((waited / 60)) 分钟"
    done
    
    echo "⚠️  等待超时，使用已完成的模型进行分析"
    return 1
}

# 运行分析
run_analysis() {
    echo ""
    echo "="*80
    echo "📊 开始全面分析"
    echo "="*80
    echo ""
    
    # 运行全面分析脚本
    "$VENV_PYTHON" analysis/comprehensive_lancet_analysis.py \
        --output_dir analysis/lancet_comprehensive_analysis \
        > analysis/lancet_comprehensive_analysis/analysis.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "="*80
        echo "✅ 全面分析完成！"
        echo "="*80
        echo ""
        echo "📁 结果保存在: analysis/lancet_comprehensive_analysis/"
        echo "📄 查看日志: cat analysis/lancet_comprehensive_analysis/analysis.log"
        echo ""
        echo "📊 主要报告文件:"
        echo "  - Swin-T: analysis/lancet_comprehensive_analysis/Swin-T_lancet_report.md"
        echo "  - CNN: analysis/lancet_comprehensive_analysis/CNN_lancet_report.md"
        echo "  - VMamba: analysis/lancet_comprehensive_analysis/VMamba_lancet_report.md"
        echo ""
        return 0
    else
        echo ""
        echo "❌ 分析过程中出现错误"
        echo "📄 查看日志: cat analysis/lancet_comprehensive_analysis/analysis.log"
        return 1
    fi
}

# 主流程
main() {
    # 选项1: 立即运行（如果有已完成的模型）
    # 选项2: 等待训练完成后再运行
    
    if [ "$1" = "--wait" ]; then
        wait_for_models
    fi
    
    # 运行分析
    run_analysis
}

# 执行
main "$@"

