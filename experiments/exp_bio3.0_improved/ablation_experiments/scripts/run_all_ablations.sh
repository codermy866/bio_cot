#!/bin/bash
# 运行所有消融实验

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
echo -e "${GREEN}运行所有消融实验${NC}"
echo -e "${GREEN}========================================${NC}"

# 创建结果目录
RESULTS_DIR="ablation_experiments/results"
mkdir -p "$RESULTS_DIR"

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="ablation_experiments/logs"
mkdir -p "$LOG_DIR"

# 函数：运行单个消融实验
run_ablation() {
    local ablation_name=$1
    local script_path=$2
    local num_runs=${3:-5}
    
    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}运行消融实验: $ablation_name${NC}"
    echo -e "${YELLOW}运行次数: $num_runs${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    local log_file="$LOG_DIR/${ablation_name}_${TIMESTAMP}.log"
    
    # 运行实验
    python "$script_path" \
        --experiment_name "ablation_${ablation_name}" \
        --num_runs "$num_runs" \
        --output_dir "$RESULTS_DIR" \
        2>&1 | tee "$log_file"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ 消融实验 $ablation_name 完成${NC}"
    else
        echo -e "${RED}✗ 消融实验 $ablation_name 失败${NC}"
        return 1
    fi
}

# ==================== Ablation 1: Knowledge Notes消融 ====================
run_ablation "no_knowledge_notes" \
    "ablation_experiments/ablations/no_knowledge_notes/train_ablation_knowledge_notes.py" \
    5

# ==================== Ablation 2: Visual Notes消融 ====================
run_ablation "no_visual_notes" \
    "ablation_experiments/ablations/no_visual_notes/train_ablation_visual_notes.py" \
    5

# ==================== Ablation 3: Sinkhorn OT消融 ====================
run_ablation "no_ot" \
    "ablation_experiments/ablations/no_ot/train_ablation_ot.py" \
    5

# ==================== Ablation 4: Dual-Head消融 ====================
run_ablation "no_dual_head" \
    "ablation_experiments/ablations/no_dual_head/train_ablation_dual_head.py" \
    5

# ==================== Ablation 5: Knowledge + Visual Notes消融 ====================
run_ablation "no_knowledge_visual" \
    "ablation_experiments/ablations/no_knowledge_visual/train_ablation_knowledge_visual.py" \
    5

# ==================== 结果分析 ====================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}分析消融实验结果${NC}"
echo -e "${GREEN}========================================${NC}"

python ablation_experiments/scripts/analyze_ablation_results.py \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$RESULTS_DIR/analysis" \
    --full_model_path "../comparison_experiments/results/full_model" \
    2>&1 | tee "$LOG_DIR/analysis_${TIMESTAMP}.log"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✓ 结果分析完成${NC}"
else
    echo -e "${RED}✗ 结果分析失败${NC}"
fi

# ==================== 完成 ====================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}所有消融实验完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "结果目录: $RESULTS_DIR"
echo -e "日志目录: $LOG_DIR"

