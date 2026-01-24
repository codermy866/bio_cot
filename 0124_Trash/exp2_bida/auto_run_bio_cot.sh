#!/bin/bash
# Bio-COT自动执行脚本
# 自动完成：VLM特征提取 -> Student Prior预训练 -> 模型训练

set -e  # 遇到错误立即退出

cd /data2/hmy/VLM_Caus_Rm_Mics
DATA_ROOT="/data2/hmy/5Center_datas/5centers_multi_leave_centers_out"
VLM_CACHE_DIR="$DATA_ROOT/vlm_features_cache"
LOG_DIR="experiments/exp2_bida/exp_bio_cot/logs"
mkdir -p "$LOG_DIR" "$VLM_CACHE_DIR"

echo "=========================================="
echo "Bio-COT 自动执行脚本"
echo "=========================================="
echo ""

# Step 1: 检查并提取VLM特征
echo "Step 1: 检查VLM特征提取..."
if [ ! -f "$VLM_CACHE_DIR/train_vlm_features.npy" ]; then
    echo "   开始提取训练集VLM特征..."
    python experiments/exp2_bida/extract_vlm_features.py \
        --data_root "$DATA_ROOT" \
        --split train \
        --output_file "$VLM_CACHE_DIR/train_vlm_features.npy" \
        --device cuda:1 \
        --batch_size 4 \
        2>&1 | tee "$LOG_DIR/extract_vlm_train.log"
    echo "   ✅ 训练集VLM特征提取完成"
else
    echo "   ✅ 训练集VLM特征已存在，跳过"
fi

if [ ! -f "$VLM_CACHE_DIR/val_vlm_features.npy" ]; then
    echo "   开始提取验证集VLM特征..."
    python experiments/exp2_bida/extract_vlm_features.py \
        --data_root "$DATA_ROOT" \
        --split val \
        --output_file "$VLM_CACHE_DIR/val_vlm_features.npy" \
        --device cuda:1 \
        --batch_size 4 \
        2>&1 | tee "$LOG_DIR/extract_vlm_val.log"
    echo "   ✅ 验证集VLM特征提取完成"
else
    echo "   ✅ 验证集VLM特征已存在，跳过"
fi

echo ""

# Step 2: 开始训练
echo "Step 2: 开始Bio-COT模型训练..."
echo "   训练日志: $LOG_DIR/train.log"
python experiments/exp2_bida/train_bio_cot.py \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""
echo "=========================================="
echo "✅ Bio-COT训练完成！"
echo "=========================================="
echo ""
echo "模型保存位置: experiments/exp2_bida/exp_bio_cot/best_model.pth"
echo "训练日志: $LOG_DIR/train.log"
echo ""
echo "下一步: 运行测试脚本"
echo "  python experiments/exp2_bida/test_adaptation.py --model_path experiments/exp2_bida/exp_bio_cot/best_model.pth --use_ttpa"

