#!/bin/bash
# 监控训练进度脚本

cd /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713

echo "📊 训练进度监控"
echo "============================================================"

# 检查MedicalViT训练
if [ -f "models/MedicalViT/_results/multimodal/train.log" ]; then
    echo ""
    echo "🔄 MedicalViT训练状态："
    if ps aux | grep -q "start_medical_vit_multimodal" | grep -v grep; then
        echo "  ✅ 正在运行"
        tail -5 models/MedicalViT/_results/multimodal/train.log 2>/dev/null | grep -E "(Epoch|Loss|Acc|AUC)" | tail -3
    else
        echo "  ⏸️  已停止"
        if grep -q "训练完成\|Training completed" models/MedicalViT/_results/multimodal/train.log 2>/dev/null; then
            echo "  ✅ 训练已完成"
        fi
    fi
fi

# 检查ViT训练
if [ -f "models/ViT/_results/multimodal/train.log" ]; then
    echo ""
    echo "🔄 ViT训练状态："
    if ps aux | grep -q "start_vit_multimodal" | grep -v grep; then
        echo "  ✅ 正在运行"
        tail -5 models/ViT/_results/multimodal/train.log 2>/dev/null | grep -E "(Epoch|Loss|Acc|AUC)" | tail -3
    else
        echo "  ⏸️  已停止"
        if grep -q "训练完成\|Training completed" models/ViT/_results/multimodal/train.log 2>/dev/null; then
            echo "  ✅ 训练已完成"
        fi
    fi
fi

# GPU使用情况
echo ""
echo "💻 GPU使用情况："
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{printf "  GPU %s: %s/%s MB (%.1f%%), 利用率: %s%%\n", $1, $3, $4, ($3/$4)*100, $5}'

echo ""
echo "============================================================"

