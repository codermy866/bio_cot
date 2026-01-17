#!/bin/bash

# The Lancet Primary Care - 实验执行脚本
# 使用真实的多模态AI模型运行所有实验

# 配置
DATA_PATH="5centers_multi"
OUTPUT_DIR="lancet_primary_care/results"
LOG_DIR="lancet_primary_care/logs"
CUDA_DEVICE=${1:-"0"}  # 默认使用cuda:0

# 创建目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# 激活虚拟环境
source my_retfound/bin/activate

# 日志文件
LOG_FILE="$LOG_DIR/run_experiments_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "The Lancet Primary Care - 实验执行"
echo "=========================================="
echo "数据路径: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "CUDA设备: $CUDA_DEVICE"
echo "日志文件: $LOG_FILE"
echo "=========================================="

# 运行实验
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE PYTHONUNBUFFERED=1 nohup python -u \
    lancet_primary_care/scripts/run_experiments_with_models.py \
    --data_path "$DATA_PATH" \
    --experiments all \
    --output_dir "$OUTPUT_DIR" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "实验进程已启动 (PID: $PID)"
echo "查看日志: tail -f $LOG_FILE"
echo "停止实验: kill $PID"

