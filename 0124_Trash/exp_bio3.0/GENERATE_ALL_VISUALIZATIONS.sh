#!/bin/bash
# 生成所有可视化图表的脚本

cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

echo "=========================================="
echo "Bio-COT 3.0 综合可视化生成"
echo "=========================================="
echo ""

# 1. 生成训练曲线图
echo "📊 1/2 生成训练曲线图..."
python visualize_results.py

echo ""

# 2. 生成综合可视化（t-SNE、箱线图、小提琴图、CAM）
echo "📊 2/2 生成综合可视化（t-SNE、箱线图、小提琴图、CAM）..."
python comprehensive_visualization.py

echo ""
echo "=========================================="
echo "✅ 所有可视化已完成！"
echo "=========================================="
echo ""
echo "输出文件位置:"
echo "  - logs/comprehensive_analysis_*.png (训练曲线)"
echo "  - logs/tsne_visualization_*.png (t-SNE)"
echo "  - logs/boxplots_*.png (箱线图)"
echo "  - logs/violinplots_*.png (小提琴图)"
echo "  - logs/cam_samples_*.png (CAM图)"
echo ""

