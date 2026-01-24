#!/bin/bash
# 在GPU 1上运行5分类训练

echo "🚀 在GPU 1上启动5分类训练"
echo "========================================"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1

# 激活虚拟环境
source my_retfound/bin/activate

# 设置输出日志
LOG_FILE="5class_training_gpu1.log"

echo "📝 日志将保存到: $LOG_FILE"
echo "开始训练..."

# 后台运行训练
nohup python real_data_5class_training.py > "$LOG_FILE" 2>&1 &

# 获取进程ID
PID=$!
echo "✅ 训练已在后台启动 (PID: $PID)"
echo "📊 可使用以下命令查看进度："
echo "   tail -f $LOG_FILE"
echo "   nvidia-smi"
echo "   python monitor_5class_training.py"



