#!/bin/bash
# 启动Bio-COT 3.2训练并自动生成可视化

cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

echo "=========================================="
echo "🚀 Bio-COT 3.2 自动训练和可视化"
echo "=========================================="
echo ""

# 创建输出目录
mkdir -p newlog_0126/figures
mkdir -p logs

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_bio_cot_v3.2_auto_${TIMESTAMP}.log"

echo "📝 训练日志: $LOG_FILE"
echo "📁 输出目录: newlog_0126/"
echo ""

# 启动训练（后台运行，使用虚拟环境的Python）
echo "🚀 启动训练..."
echo "🐍 使用虚拟环境: /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python"
nohup /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python training/train_bio_cot_v3.2.py > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo "✅ 训练进程已启动 (PID: $TRAIN_PID)"
echo "📝 日志文件: $LOG_FILE"
echo ""

# 保存PID和日志文件路径
echo "$TRAIN_PID" > newlog_0126/train_pid.txt
echo "$LOG_FILE" > newlog_0126/log_file.txt

echo "📊 监控训练进度:"
echo "   tail -f $LOG_FILE"
echo ""
echo "⏳ 训练完成后，运行以下命令生成可视化:"
echo "   python monitor_and_visualize_0126.py"
echo ""
echo "或者等待训练完成后自动执行:"
echo "   python wait_and_visualize_0126.py"
echo ""

