#!/bin/bash
# 混合策略并行训练
# GPU 0: 2分类优化
# GPU 1: 3分类训练

set -e

echo "🚀 启动混合策略并行训练"
echo "=========================================="

# 激活环境
source my_retfound/bin/activate

# 检查GPU
echo ""
echo "📊 GPU状态检查:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv

echo ""
echo "🎯 任务分配:"
echo "  GPU 0: 2分类模型优化（目标: 85%）"
echo "  GPU 1: 3分类模型训练（预期: 70-75%）"
echo ""

# 停止之前可能存在的进程
pkill -f "train_with_cnn.py" || true
pkill -f "train_3class_model.py" || true
sleep 2

# 创建输出目录
mkdir -p cnn_training_2class_optimized
mkdir -p cnn_training_3class

# 任务1: 2分类优化（GPU 0）
echo "🚀 启动GPU 0: 2分类优化..."
CUDA_VISIBLE_DEVICES=0 python train_with_cnn.py \
    --data_path 5centers_multi \
    --epochs 20 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --output_dir cnn_training_2class_optimized \
    > cnn_2class_optimize.log 2>&1 &

GPU0_PID=$!
echo "  ✅ GPU 0 任务启动 (PID: $GPU0_PID)"

# 等待5秒让第一个任务启动
sleep 5

# 任务2: 3分类训练（GPU 1）
echo ""
echo "🚀 启动GPU 1: 3分类训练..."
CUDA_VISIBLE_DEVICES=1 python train_3class_model.py \
    --data_path 5centers_multi_3class \
    --epochs 15 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --output_dir cnn_training_3class \
    > cnn_3class_train.log 2>&1 &

GPU1_PID=$!
echo "  ✅ GPU 1 任务启动 (PID: $GPU1_PID)"

echo ""
echo "=========================================="
echo "✅ 混合策略已启动！"
echo ""
echo "📋 监控命令:"
echo "  GPU 0 (2分类优化): tail -f cnn_2class_optimize.log"
echo "  GPU 1 (3分类训练):  tail -f cnn_3class_train.log"
echo "  GPU使用情况: watch -n 2 nvidia-smi"
echo ""
echo "📊 预期的训练进度:"
echo "  - GPU 0: 每epoch约30-40分钟，共20 epochs（预计12-14小时）"
echo "  - GPU 1: 每epoch约30-40分钟，共15 epochs（预计8-10小时）"
echo ""
echo "⚡ 3分类训练将先完成，你可以先查看其效果"
echo ""
echo "🎯 正在后台运行中..."
echo ""



