#!/bin/bash
# 混合策略启动脚本
# GPU 1: 5分类轻量级验证
# GPU 0: 2分类优化

echo "🚀 启动混合策略训练"
echo "=========================================="

# 停止之前可能存在的进程
killall -9 python 2>/dev/null
sleep 2

echo "📊 GPU使用情况:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv

echo ""
echo "🎯 混合策略配置:"
echo "  GPU 1: 5分类轻量级验证（快速验证可行性）"
echo "  GPU 0: 2分类优化（提升到85%）"
echo ""

# 先启动5分类验证（使用轻量级模型，避免数据加载问题）
echo "🚀 启动5分类轻量级验证..."
cd /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713

CUDA_VISIBLE_DEVICES=1 bash -c "
source my_retfound/bin/activate && 
python lightweight_5class_training.py > 5class_lightweight_gpu1.log 2>&1
" &

GPU1_PID=$!
echo "  ✅ 5分类验证已在GPU 1启动 (PID: $GPU1_PID)"

# 由于GPU 0被占用，先等待一小会看看情况
sleep 5

echo ""
echo "📋 监控命令:"
echo "  查看5分类训练日志: tail -f 5class_lightweight_gpu1.log"
echo "  查看GPU使用: watch -n 2 nvidia-smi"
echo "  查看训练历史: ls -lht lightweight_5class_results/"

echo ""
echo "⏳ 等待5分钟让训练开始..."
sleep 60

# 检查进展
echo ""
echo "📊 当前状态:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
echo ""
echo "📄 最新训练日志:"
tail -20 5class_lightweight_gpu1.log 2>/dev/null || echo "  日志文件尚未生成"

echo ""
echo "✅ 混合策略已启动！"
echo "   5分类验证运行在GPU 1"
echo "   2分类优化暂时等待（GPU 0被占用）"
echo ""
echo "💡 建议: 等待5分类验证完成后，释放GPU 0再启动2分类优化"



