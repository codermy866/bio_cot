#!/bin/bash
# 检查Bio-COT实验进度

echo "=========================================="
echo "Bio-COT 实验进度检查"
echo "=========================================="
echo ""

# 检查VLM特征提取进度
echo "1. VLM特征提取进度:"
if [ -f "/data2/hmy/5Center_datas/5centers_multi_leave_centers_out/vlm_features_cache/train_vlm_features.npy" ]; then
    echo "   ✅ 训练集VLM特征已提取"
    SIZE=$(du -h /data2/hmy/5Center_datas/5centers_multi_leave_centers_out/vlm_features_cache/train_vlm_features.npy | cut -f1)
    echo "   文件大小: $SIZE"
else
    echo "   ⏳ 训练集VLM特征提取中..."
    if [ -f "experiments/exp2_bida/exp_bio_cot/logs/extract_vlm_train.log" ]; then
        echo "   最新日志:"
        tail -5 experiments/exp2_bida/exp_bio_cot/logs/extract_vlm_train.log
    fi
fi

if [ -f "/data2/hmy/5Center_datas/5centers_multi_leave_centers_out/vlm_features_cache/val_vlm_features.npy" ]; then
    echo "   ✅ 验证集VLM特征已提取"
    SIZE=$(du -h /data2/hmy/5Center_datas/5centers_multi_leave_centers_out/vlm_features_cache/val_vlm_features.npy | cut -f1)
    echo "   文件大小: $SIZE"
else
    echo "   ⏳ 验证集VLM特征未提取"
fi

echo ""

# 检查训练进度
echo "2. 训练进度:"
if [ -f "experiments/exp2_bida/exp_bio_cot/best_model.pth" ]; then
    echo "   ✅ 模型已训练"
    SIZE=$(du -h experiments/exp2_bida/exp_bio_cot/best_model.pth | cut -f1)
    echo "   模型大小: $SIZE"
    if [ -f "experiments/exp2_bida/exp_bio_cot/logs/train.log" ]; then
        echo "   最新训练日志:"
        tail -10 experiments/exp2_bida/exp_bio_cot/logs/train.log | grep -E "Epoch|AUC|Acc" || tail -5 experiments/exp2_bida/exp_bio_cot/logs/train.log
    fi
else
    echo "   ⏳ 模型未训练"
fi

echo ""

# 检查GPU使用情况
echo "3. GPU使用情况:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{printf "   GPU %s (%s): %s/%s MB, %s%%\n", $1, $2, $3, $4, $5}'

echo ""
echo "=========================================="

