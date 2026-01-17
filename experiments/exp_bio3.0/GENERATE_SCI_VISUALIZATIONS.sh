#!/bin/bash
# Bio-COT 3.0 SCI顶刊级可视化生成脚本

cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

echo "=========================================="
echo "Bio-COT 3.0 SCI顶刊级可视化生成"
echo "=========================================="
echo ""

echo "📊 开始生成可视化图表..."
echo "   这可能需要几分钟到十几分钟，请耐心等待..."
echo ""

python generate_sci_visualizations.py

echo ""
echo "=========================================="
echo "✅ 可视化生成完成！"
echo "=========================================="
echo ""
echo "生成的文件位置: logs/"
echo "  - knowledge_notes_distribution_*.png"
echo "  - visual_notes_attention_*.png"
echo "  - volcano_plot_*.png"
echo "  - tsne_umap_*.png"
echo "  - violin_plots_comprehensive_*.png"
echo "  - cam_samples_*.png"
echo ""

