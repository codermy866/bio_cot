#!/bin/bash
# CNN训练启动脚本（隐藏Python路径）
cd /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/cnn_result_enhanced
exec ../my_retfound/bin/python ../optimized_2class_training.py --data_path ../5centers_multi --epochs 20 --batch_size 3 --learning_rate 3e-5 --output_dir . >> train.log 2>&1

