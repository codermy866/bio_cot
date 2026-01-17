#!/bin/bash
# 检查VLM特征提取进度

echo "=========================================="
echo "VLM特征提取进度检查"
echo "=========================================="
echo ""

# 检查训练集VLM特征提取
TRAIN_LOG="experiments/exp2_bida/exp_bio_cot/logs/extract_vlm_train.log"
if [ -f "$TRAIN_LOG" ]; then
    echo "📊 训练集VLM特征提取日志:"
    echo "----------------------------------------"
    tail -10 "$TRAIN_LOG" | grep -E "Extracting|样本|完成|Error|Exception" || tail -5 "$TRAIN_LOG"
    echo "----------------------------------------"
    
    # 检查进度
    if grep -q "完成" "$TRAIN_LOG"; then
        echo "✅ 训练集VLM特征提取已完成"
    else
        TRAIN_PID=$(ps aux | grep "extract_vlm_features.*train" | grep -v grep | awk '{print $2}' | head -1)
        if [ -n "$TRAIN_PID" ]; then
            echo "⏳ 训练集VLM特征提取进行中 (PID: $TRAIN_PID)"
        else
            echo "❌ 训练集VLM特征提取进程未运行"
        fi
    fi
else
    echo "⏸️  训练集VLM特征提取尚未开始"
fi

echo ""

# 检查输出文件
TRAIN_OUTPUT="/data2/hmy/5Center_datas/5centers_multi_leave_centers_out/vlm_features_cache/train_vlm_features.npy"
if [ -f "$TRAIN_OUTPUT" ]; then
    SIZE=$(du -h "$TRAIN_OUTPUT" | cut -f1)
    echo "✅ 训练集VLM特征文件存在: $SIZE"
else
    echo "⏳ 训练集VLM特征文件尚未生成"
fi

echo ""

# 检查验证集VLM特征提取
VAL_LOG="experiments/exp2_bida/exp_bio_cot/logs/extract_vlm_val.log"
if [ -f "$VAL_LOG" ]; then
    echo "📊 验证集VLM特征提取日志:"
    echo "----------------------------------------"
    tail -10 "$VAL_LOG" | grep -E "Extracting|样本|完成|Error|Exception" || tail -5 "$VAL_LOG"
    echo "----------------------------------------"
    
    if grep -q "完成" "$VAL_LOG"; then
        echo "✅ 验证集VLM特征提取已完成"
    else
        VAL_PID=$(ps aux | grep "extract_vlm_features.*val" | grep -v grep | awk '{print $2}' | head -1)
        if [ -n "$VAL_PID" ]; then
            echo "⏳ 验证集VLM特征提取进行中 (PID: $VAL_PID)"
        else
            echo "⏸️  验证集VLM特征提取尚未开始"
        fi
    fi
else
    echo "⏸️  验证集VLM特征提取尚未开始"
fi

echo ""

VAL_OUTPUT="/data2/hmy/5Center_datas/5centers_multi_leave_centers_out/vlm_features_cache/val_vlm_features.npy"
if [ -f "$VAL_OUTPUT" ]; then
    SIZE=$(du -h "$VAL_OUTPUT" | cut -f1)
    echo "✅ 验证集VLM特征文件存在: $SIZE"
else
    echo "⏳ 验证集VLM特征文件尚未生成"
fi

echo ""

# GPU使用情况
echo "🖥️  GPU使用情况:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "   GPU %s (%s): %s/%s MB, %s%%\n", $1, $2, $3, $4, $5}'

echo ""
echo "=========================================="


