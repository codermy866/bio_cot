#!/bin/bash
# 启动Bio-COT 3.2训练并自动生成可视化（使用正确的虚拟环境）

cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

echo "=========================================="
echo "🚀 Bio-COT 3.2 自动训练和可视化"
echo "=========================================="
echo ""

# 检查虚拟环境
if [ ! -f "/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python" ]; then
    echo "❌ 错误: 虚拟环境不存在"
    echo "   路径: /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python"
    exit 1
fi

echo "✅ 虚拟环境: /data2/hmy/VLM_Caus_Rm_Mics/my_retfound"
echo "🐍 Python版本: $(/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python --version)"
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

# 检查CUDA
echo "🔍 检查CUDA环境..."
CUDA_AVAILABLE=$(/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$CUDA_AVAILABLE" = "True" ]; then
    CUDA_COUNT=$(/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    echo "✅ CUDA可用，设备数量: $CUDA_COUNT"
else
    echo "⚠️  CUDA不可用，将使用CPU训练"
fi
echo ""

# 启动训练（后台运行，使用虚拟环境的Python）
echo "🚀 启动训练..."
echo "🐍 使用Python: /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=""  # 让训练脚本自动选择GPU

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
echo "   source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate"
echo "   python monitor_and_visualize_0126.py"
echo ""
echo "或者等待训练完成后自动执行:"
echo "   source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate"
echo "   python wait_and_visualize_0126.py"
echo ""

