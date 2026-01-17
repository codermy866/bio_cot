#!/bin/bash
# 运行所有Baseline对比实验

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目路径
PROJECT_ROOT="/data2/hmy/VLM_Caus_Rm_Mics"
EXPERIMENT_DIR="$PROJECT_ROOT/experiments/exp_bio3.0_improved"
VENV_PATH="$PROJECT_ROOT/my_retfound"

# 激活虚拟环境
source "$VENV_PATH/bin/activate"

# 进入实验目录
cd "$EXPERIMENT_DIR"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}运行所有Baseline对比实验${NC}"
echo -e "${GREEN}========================================${NC}"

# 创建结果目录
RESULTS_DIR="comparison_experiments/results"
mkdir -p "$RESULTS_DIR"

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="comparison_experiments/logs"
mkdir -p "$LOG_DIR"

# 函数：运行单个实验
run_baseline() {
    local baseline_name=$1
    local script_path=$2
    local num_runs=${3:-5}
    
    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}运行Baseline: $baseline_name${NC}"
    echo -e "${YELLOW}运行次数: $num_runs${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    local log_file="$LOG_DIR/${baseline_name}_${TIMESTAMP}.log"
    
    # 运行实验
    python "$script_path" \
        --experiment_name "baseline_${baseline_name}" \
        --num_runs "$num_runs" \
        --output_dir "$RESULTS_DIR" \
        2>&1 | tee "$log_file"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ Baseline $baseline_name 完成${NC}"
    else
        echo -e "${RED}✗ Baseline $baseline_name 失败${NC}"
        return 1
    fi
}

# ==================== Baseline 1: Simple Fusion ====================
run_baseline "simple_fusion" \
    "comparison_experiments/baselines/simple_fusion/train_simple_fusion.py" \
    5

# ==================== Baseline 2: Standard CLIP ====================
run_baseline "standard_clip" \
    "comparison_experiments/baselines/standard_clip/train_standard_clip.py" \
    5

# ==================== Baseline 3: ViT + Clinical Fusion ====================
run_baseline "vit_clinical_fusion" \
    "comparison_experiments/baselines/vit_clinical_fusion/train_vit_clinical.py" \
    5

# ==================== Baseline 4-6: 已有Baseline（可选） ====================
# 如果需要重新运行，取消注释
# run_baseline "cnn" \
#     "../baseline/train_cnn_baseline.py" \
#     5

# run_baseline "swin_t" \
#     "../baseline/train_swin_baseline.py" \
#     5

# run_baseline "vmamba" \
#     "../baseline/train_vmamba_baseline.py" \
#     5

# ==================== 结果分析 ====================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}分析Baseline对比结果${NC}"
echo -e "${GREEN}========================================${NC}"

python comparison_experiments/scripts/analyze_baseline_results.py \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$RESULTS_DIR/analysis" \
    2>&1 | tee "$LOG_DIR/analysis_${TIMESTAMP}.log"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✓ 结果分析完成${NC}"
else
    echo -e "${RED}✗ 结果分析失败${NC}"
fi

# ==================== 完成 ====================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}所有Baseline对比实验完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "结果目录: $RESULTS_DIR"
echo -e "日志目录: $LOG_DIR"

