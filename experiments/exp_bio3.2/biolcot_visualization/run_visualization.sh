#!/bin/bash
# BioLCoT 可视化脚本
# 自动查找最新的 checkpoint 并生成可视化

set -e  # 遇到错误立即退出

# 激活虚拟环境
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

# 进入脚本目录
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/biolcot_visualization

echo "=========================================="
echo "BioLCoT 可视化生成器"
echo "=========================================="
echo ""

# 检查 checkpoint 目录
CHECKPOINT_DIR="../checkpoints"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Checkpoint 目录不存在: $CHECKPOINT_DIR"
    exit 1
fi

# 查找最新的 checkpoint
LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/best_model_*.pth 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ 未找到 checkpoint 文件！"
    echo "   查找目录: $CHECKPOINT_DIR"
    echo "   请确保该目录下有 best_model_*.pth 文件"
    exit 1
fi

echo "📦 找到最新的 checkpoint:"
echo "   $(basename $LATEST_CHECKPOINT)"
echo ""

# 生成 Grad-CAM 可视化
echo "=========================================="
echo "1. 生成 Grad-CAM 可视化"
echo "=========================================="
python generate_gradcam.py \
    --checkpoint "$LATEST_CHECKPOINT" \
    --num_samples 4 \
    --save_dir gradcam_results

echo ""
echo "✅ Grad-CAM 可视化完成！"
echo ""

# 生成 Attention Map 可视化
echo "=========================================="
echo "2. 生成 Attention Map 可视化"
echo "=========================================="
python generate_attention_map.py \
    --checkpoint "$LATEST_CHECKPOINT" \
    --num_samples 4 \
    --save_dir attention_results

echo ""
echo "✅ Attention Map 可视化完成！"
echo ""

# 生成病灶聚焦 Grad-CAM 可视化
echo "=========================================="
echo "3. 生成病灶聚焦 Grad-CAM 可视化"
echo "=========================================="
python generate_lesion_focused_gradcam.py \
    --checkpoint "$LATEST_CHECKPOINT" \
    --num_samples 4 \
    --threshold 0.6 \
    --smooth_sigma 1.0 \
    --save_dir lesion_focused_results

echo ""
echo "✅ 病灶聚焦 Grad-CAM 可视化完成！"
echo ""

echo "=========================================="
echo "🎉 所有可视化生成完成！"
echo "=========================================="
echo ""
echo "📁 输出目录："
echo "   - Grad-CAM: gradcam_results/"
echo "   - Attention Map: attention_results/"
echo "   - 病灶聚焦 Grad-CAM: lesion_focused_results/"
echo ""

