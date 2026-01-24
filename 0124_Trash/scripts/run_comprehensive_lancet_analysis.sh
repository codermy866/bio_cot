#!/bin/bash
# Lancet期刊全面分析脚本
# 自动执行所有必需的分析

cd "$(dirname "$0")/.." || exit 1

VENV_PYTHON="./my_retfound/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "错误: 虚拟环境 Python 不存在: $VENV_PYTHON"
    exit 1
fi

echo "="*80
echo "🚀 启动Lancet期刊全面分析"
echo "="*80
echo ""
echo "本脚本将自动执行："
echo "  1. 内部测试集性能评估（使用最优阈值）"
echo "  2. 外部验证集性能评估"
echo "  3. 内部/外部数据集对比分析"
echo "  4. Bootstrap置信区间计算"
echo "  5. 亚组分析"
echo "  6. 决策曲线分析（DCA）"
echo "  7. 生成完整报告"
echo ""
echo "="*80
echo ""

# 创建输出目录
mkdir -p analysis/lancet_comprehensive_analysis

# 运行全面分析
echo "📊 开始全面分析..."
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
else
    echo ""
    echo "❌ 分析过程中出现错误"
    echo "📄 查看日志: cat analysis/lancet_comprehensive_analysis/analysis.log"
    exit 1
fi

