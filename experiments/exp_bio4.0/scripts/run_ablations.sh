#!/bin/bash
# Bio-COT 4.0: 消融实验自动化脚本
# 用途：依次运行所有消融实验，自动收集结果

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXP_DIR"

echo "=========================================="
echo "🔬 Bio-COT 4.0 消融实验套件"
echo "=========================================="
echo "实验目录: $EXP_DIR"
echo ""

# 创建结果目录
mkdir -p results/ablation_baseline results/ablation_no_vlm results/ablation_no_causal results/full_model
mkdir -p checkpoints/ablation_baseline checkpoints/ablation_no_vlm checkpoints/ablation_no_causal checkpoints/full_model
mkdir -p logs/ablation_baseline logs/ablation_no_vlm logs/ablation_no_causal logs/full_model

# 定义实验列表
EXPERIMENTS=(
    "baseline:A0 - Baseline (Naïve ViT)"
    "no_vlm:A1 - w/o VLM Guidance"
    "no_causal:A2 - w/o Causal Disentangle"
    "full:A3 - Bio-COT 4.0 (Full)"
)

# 运行实验
for exp_info in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_id exp_name <<< "$exp_info"
    echo ""
    echo "=========================================="
    echo "开始实验: $exp_name"
    echo "=========================================="
    
    # 运行训练脚本
    python training/train_bio_cot_v4.py --ablation "$exp_id" || {
        echo "❌ 实验 $exp_name 失败，跳过..."
        continue
    }
    
    echo "✅ 实验 $exp_name 完成"
    echo ""
done

echo ""
echo "=========================================="
echo "🎉 所有消融实验完成！"
echo "=========================================="
echo "结果已保存到:"
echo "  - results/ablation_*/"
echo "  - logs/ablation_*/"
echo "  - checkpoints/ablation_*/"
echo ""
echo "建议下一步：运行 scripts/analyze_ablations.py 生成对比表格"

