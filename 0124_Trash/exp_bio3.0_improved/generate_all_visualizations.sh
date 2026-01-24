#!/bin/bash
# 训练完成后，生成所有可视化结果

cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0_improved
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

echo "=========================================="
echo "生成所有可视化结果"
echo "=========================================="
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 1. 生成基础可视化（SCI论文级别）
echo "📊 1. 生成基础可视化（SCI论文级别）..."
python ../exp_bio3.0/generate_sci_visualizations.py 2>&1 | tee logs/generate_sci_vis_${TIMESTAMP}.log
echo "✅ 基础可视化完成"
echo ""

# 2. 生成3D可视化
echo "📊 2. 生成3D可视化..."
python ../exp_bio3.0/generate_3d_visualizations.py 2>&1 | tee logs/generate_3d_vis_${TIMESTAMP}.log
echo "✅ 3D可视化完成"
echo ""

# 3. 生成补充可视化
echo "📊 3. 生成补充可视化..."
python ../exp_bio3.0/generate_additional_visualizations.py 2>&1 | tee logs/generate_add_vis_${TIMESTAMP}.log
echo "✅ 补充可视化完成"
echo ""

# 4. 移动可视化文件到visualizations文件夹
echo "📁 整理可视化文件..."
find logs -name "*.pdf" -type f -exec mv {} visualizations/ \; 2>/dev/null
find logs -name "*.png" -type f -exec mv {} visualizations/ \; 2>/dev/null
echo "✅ 文件已整理到 visualizations/ 文件夹"
echo ""

echo "=========================================="
echo "✅ 所有可视化已完成！"
echo "=========================================="
echo ""
echo "📁 结果位置:"
echo "  - 可视化图片: visualizations/"
echo "  - 日志文件: logs/"
echo ""

