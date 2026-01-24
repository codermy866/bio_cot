#!/bin/bash
# 运行所有实验的脚本
# 包括Baseline对比和消融实验

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
echo -e "${GREEN}Bio-COT 3.0 Improved 完整实验执行脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 创建结果目录
RESULTS_DIR="experiments/results"
mkdir -p "$RESULTS_DIR"

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="experiments/logs"
mkdir -p "$LOG_DIR"

# 日志文件
MAIN_LOG="$LOG_DIR/experiments_${TIMESTAMP}.log"

# 函数：运行单个实验
run_experiment() {
    local experiment_name=$1
    local script_path=$2
    local num_runs=${3:-5}  # 默认运行5次
    
    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}运行实验: $experiment_name${NC}"
    echo -e "${YELLOW}运行次数: $num_runs${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    local log_file="$LOG_DIR/${experiment_name}_${TIMESTAMP}.log"
    
    # 运行实验
    python "$script_path" \
        --experiment_name "$experiment_name" \
        --num_runs "$num_runs" \
        --output_dir "$RESULTS_DIR" \
        2>&1 | tee "$log_file"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ 实验 $experiment_name 完成${NC}"
    else
        echo -e "${RED}✗ 实验 $experiment_name 失败${NC}"
        return 1
    fi
}

# ==================== 阶段1: Baseline对比实验 ====================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}阶段1: Baseline对比实验${NC}"
echo -e "${GREEN}========================================${NC}"

# Baseline 1: Simple Fusion
run_experiment "baseline_simple_fusion" \
    "experiments/baseline/train_simple_fusion_baseline.py" \
    5

# Baseline 2: Standard CLIP
run_experiment "baseline_standard_clip" \
    "experiments/baseline/train_standard_clip_baseline.py" \
    5

# Baseline 3: ViT + Clinical Fusion
run_experiment "baseline_vit_clinical_fusion" \
    "experiments/baseline/train_vit_clinical_fusion.py" \
    5

# Baseline 4-6: 已有Baseline（如果需要重新运行）
# run_experiment "baseline_cnn" "experiments/baseline/train_cnn_baseline.py" 5
# run_experiment "baseline_swin_t" "experiments/baseline/train_swin_baseline.py" 5
# run_experiment "baseline_vmamba" "experiments/baseline/train_vmamba_baseline.py" 5

# ==================== 阶段2: 消融实验 ====================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}阶段2: 消融实验${NC}"
echo -e "${GREEN}========================================${NC}"

# Ablation 1: Knowledge Notes消融
run_experiment "ablation_no_knowledge_notes" \
    "experiments/ablation/train_ablation_knowledge_notes.py" \
    5

# Ablation 2: Visual Notes消融
run_experiment "ablation_no_visual_notes" \
    "experiments/ablation/train_ablation_visual_notes.py" \
    5

# Ablation 3: Sinkhorn OT消融
run_experiment "ablation_no_ot" \
    "experiments/ablation/train_ablation_ot.py" \
    5

# Ablation 4: Dual-Head消融
run_experiment "ablation_no_dual_head" \
    "experiments/ablation/train_ablation_dual_head.py" \
    5

# Ablation 5: Knowledge + Visual Notes消融
run_experiment "ablation_no_knowledge_visual" \
    "experiments/ablation/train_ablation_knowledge_visual.py" \
    5

# ==================== 阶段3: 完整方法 ====================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}阶段3: 完整方法 (Full Model)${NC}"
echo -e "${GREEN}========================================${NC}"

# Full Model
run_experiment "full_model" \
    "training/train_bio_cot_v3.py" \
    5

# ==================== 阶段4: 结果分析 ====================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}阶段4: 结果分析和统计检验${NC}"
echo -e "${GREEN}========================================${NC}"

# 运行结果分析脚本
python scripts/analyze_all_results.py \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$RESULTS_DIR/analysis" \
    --reference_method "full_model" \
    2>&1 | tee "$LOG_DIR/analysis_${TIMESTAMP}.log"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✓ 结果分析完成${NC}"
else
    echo -e "${RED}✗ 结果分析失败${NC}"
fi

# ==================== 完成 ====================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}所有实验完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "结果目录: $RESULTS_DIR"
echo -e "日志目录: $LOG_DIR"
echo -e "主日志文件: $MAIN_LOG"

