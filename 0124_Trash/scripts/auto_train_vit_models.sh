#!/bin/bash
# 自动训练ViT和MedicalViT模型的脚本
# 按顺序训练：先MedicalViT，完成后自动启动ViT

cd /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713
source my_retfound/bin/activate

echo "🚀 自动训练ViT和MedicalViT模型"
echo "============================================================"
echo "训练顺序："
echo "1. MedicalViT (batch_size=2, GPU 0)"
echo "2. ViT (batch_size=2, GPU 0/1)"
echo "============================================================"
echo ""

# 检查MedicalViT训练是否已完成
check_medical_vit_completed() {
    if [ -f "models/MedicalViT/_results/multimodal/train.log" ]; then
        if grep -q "训练完成\|Training completed\|✅ 训练完成" models/MedicalViT/_results/multimodal/train.log 2>/dev/null; then
            return 0  # 已完成
        fi
    fi
    return 1  # 未完成
}

# 检查MedicalViT训练是否正在运行
check_medical_vit_running() {
    if ps aux | grep -q "start_medical_vit_multimodal" | grep -v grep; then
        return 0  # 正在运行
    fi
    return 1  # 未运行
}

# 检查ViT训练是否已完成
check_vit_completed() {
    if [ -f "models/ViT/_results/multimodal/train.log" ]; then
        if grep -q "训练完成\|Training completed\|✅ 训练完成" models/ViT/_results/multimodal/train.log 2>/dev/null; then
            return 0  # 已完成
        fi
    fi
    return 1  # 未完成
}

# 检查ViT训练是否正在运行
check_vit_running() {
    if ps aux | grep -q "start_vit_multimodal" | grep -v grep; then
        return 0  # 正在运行
    fi
    return 1  # 未运行
}

# 步骤1: 启动MedicalViT训练（如果未完成且未运行）
if ! check_medical_vit_completed && ! check_medical_vit_running; then
    echo "📊 步骤1: 启动MedicalViT训练"
    echo "============================================================"
    bash models/MedicalViT/scripts/run_medical_vit_multimodal.sh --gpu 0 --epochs 30 --batch_size 2 --lr 2e-5
    echo "✅ MedicalViT训练已启动"
    echo "📁 日志文件: models/MedicalViT/_results/multimodal/train.log"
    echo ""
    echo "⏳ 等待MedicalViT训练完成..."
    
    # 监控训练进度
    while true; do
        sleep 300  # 每5分钟检查一次
        
        if check_medical_vit_completed; then
            echo "✅ MedicalViT训练已完成！"
            break
        fi
        
        if ! check_medical_vit_running; then
            echo "⚠️  MedicalViT训练已停止，检查日志..."
            tail -20 models/MedicalViT/_results/multimodal/train.log 2>&1 | tail -10
            break
        fi
        
        # 显示进度
        echo "🔄 MedicalViT训练进行中..."
        tail -3 models/MedicalViT/_results/multimodal/train.log 2>&1 | grep -E "(Epoch|Loss|Acc|AUC)" | tail -1
    done
elif check_medical_vit_running; then
    echo "📊 MedicalViT训练正在运行中..."
    echo "⏳ 等待MedicalViT训练完成..."
    
    # 监控训练进度
    while true; do
        sleep 300  # 每5分钟检查一次
        
        if check_medical_vit_completed; then
            echo "✅ MedicalViT训练已完成！"
            break
        fi
        
        if ! check_medical_vit_running; then
            echo "⚠️  MedicalViT训练已停止，检查日志..."
            tail -20 models/MedicalViT/_results/multimodal/train.log 2>&1 | tail -10
            break
        fi
        
        # 显示进度
        echo "🔄 MedicalViT训练进行中..."
        tail -3 models/MedicalViT/_results/multimodal/train.log 2>&1 | grep -E "(Epoch|Loss|Acc|AUC)" | tail -1
    done
elif check_medical_vit_completed; then
    echo "✅ MedicalViT训练已完成，跳过"
fi

# 步骤2: 启动ViT训练（如果MedicalViT已完成且ViT未完成且未运行）
if check_medical_vit_completed && ! check_vit_completed && ! check_vit_running; then
    echo ""
    echo "📊 步骤2: 启动ViT训练"
    echo "============================================================"
    
    # 检查GPU可用性
    GPU_0_USAGE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=0)
    GPU_1_USAGE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=1)
    
    # 选择显存使用较少的GPU
    if [ "$GPU_0_USAGE" -lt "$GPU_1_USAGE" ]; then
        SELECTED_GPU=0
    else
        SELECTED_GPU=1
    fi
    
    echo "💻 使用GPU: $SELECTED_GPU (显存使用: GPU0=${GPU_0_USAGE}MB, GPU1=${GPU_1_USAGE}MB)"
    
    bash models/ViT/scripts/run_vit_multimodal.sh --gpu $SELECTED_GPU --epochs 30 --batch_size 2 --lr 2e-5
    echo "✅ ViT训练已启动"
    echo "📁 日志文件: models/ViT/_results/multimodal/train.log"
    echo ""
    echo "⏳ 等待ViT训练完成..."
    
    # 监控训练进度
    while true; do
        sleep 300  # 每5分钟检查一次
        
        if check_vit_completed; then
            echo "✅ ViT训练已完成！"
            break
        fi
        
        if ! check_vit_running; then
            echo "⚠️  ViT训练已停止，检查日志..."
            tail -20 models/ViT/_results/multimodal/train.log 2>&1 | tail -10
            break
        fi
        
        # 显示进度
        echo "🔄 ViT训练进行中..."
        tail -3 models/ViT/_results/multimodal/train.log 2>&1 | grep -E "(Epoch|Loss|Acc|AUC)" | tail -1
    done
elif check_vit_running; then
    echo "📊 ViT训练正在运行中..."
elif check_vit_completed; then
    echo "✅ ViT训练已完成，跳过"
fi

echo ""
echo "============================================================"
echo "✅ 所有训练任务完成！"
echo "============================================================"
echo ""
echo "📊 训练结果："
echo "  - MedicalViT: models/MedicalViT/_results/multimodal/"
echo "  - ViT: models/ViT/_results/multimodal/"
echo ""
echo "📁 查看训练日志："
echo "  - MedicalViT: tail -f models/MedicalViT/_results/multimodal/train.log"
echo "  - ViT: tail -f models/ViT/_results/multimodal/train.log"

