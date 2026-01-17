#!/bin/bash
# Bio-COT多模态训练启动脚本
# 使用正确的虚拟环境

# 激活虚拟环境
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

# 切换到项目目录
cd /data2/hmy/VLM_Caus_Rm_Mics

# 运行训练脚本（日志会自动保存到logs文件夹）
python experiments/exp_xiangyang/train_bio_cot_multimodal_xiangyang.py

echo "训练任务已启动"
echo "查看日志: tail -f experiments/exp_xiangyang/logs/train_bio_cot_multimodal_*.log"

