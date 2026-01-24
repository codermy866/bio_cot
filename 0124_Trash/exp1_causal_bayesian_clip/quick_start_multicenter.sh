#!/bin/bash
# 快速启动脚本：多中心对齐实验
# 生成时间: 2025-12-25 22:18:01

set -e  # 遇到错误立即退出

echo "=========================================="
echo "多中心对齐实验 - 快速启动"
echo "=========================================="

# 1. 激活虚拟环境
echo "📦 激活虚拟环境..."
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

# 2. 检查GPU
echo "🔍 检查GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# 3. 创建实验目录
echo "📁 创建实验目录..."
EXPERIMENT_DIR="./exp_multicenter_alignment"
mkdir -p ${EXPERIMENT_DIR}/{checkpoints,logs,results}
echo "✅ 实验目录: ${EXPERIMENT_DIR}"

# 4. 检查数据集
echo "📊 检查数据集..."
DATA_PATH="/data2/hmy/5Center_datas/5centers_multi_leave_centers_out"
if [ ! -d "$DATA_PATH" ]; then
    echo "❌ 数据集路径不存在: $DATA_PATH"
    exit 1
fi
echo "✅ 数据集路径: $DATA_PATH"

# 5. 检查标签文件
echo "📋 检查标签文件..."
if [ ! -f "$DATA_PATH/internal_train/train/train_labels.csv" ]; then
    echo "❌ 训练标签文件不存在"
    exit 1
fi
if [ ! -f "$DATA_PATH/internal_train/val/val_labels.csv" ]; then
    echo "❌ 验证标签文件不存在"
    exit 1
fi
if [ ! -f "$DATA_PATH/external_validation/external_test_labels.csv" ]; then
    echo "❌ 外部测试标签文件不存在"
    exit 1
fi
echo "✅ 标签文件完整"

# 6. 开始训练
echo "🚀 开始训练..."
echo "=========================================="

python train_vlm.py \
    --data_path ${DATA_PATH} \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 3e-4 \
    --learning_rate_other 3e-4 \
    --weight_decay 1e-5 \
    --use_amp \
    --output_dir ${EXPERIMENT_DIR}/checkpoints \
    --save_freq 10 \
    --eval_freq 5 \
    --device cuda:0 \
    --num_workers 4 \
    --seed 42 \
    2>&1 | tee ${EXPERIMENT_DIR}/logs/train_$(date +%Y%m%d_%H%M%S).log

echo "=========================================="
echo "✅ 训练完成！"
echo "📁 检查点保存在: ${EXPERIMENT_DIR}/checkpoints"
echo "📊 日志保存在: ${EXPERIMENT_DIR}/logs"
echo "=========================================="

