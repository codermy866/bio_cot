#!/bin/bash
# 持续监控消融实验进度

cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.2/ablation_studies

while true; do
    clear
    echo "=================================================================================="
    echo "📊 消融实验实时监控（整合5.0优势的3.2版本）"
    echo "=================================================================================="
    echo "更新时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    python monitor_experiments.py
    
    echo ""
    echo "按 Ctrl+C 退出监控"
    echo "每60秒自动刷新..."
    sleep 60
done

