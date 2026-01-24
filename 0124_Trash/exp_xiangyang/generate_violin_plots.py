#!/usr/bin/env python3
"""
从保存的结果文件生成小提琴图
需要先加载模型进行推理获取预测概率
"""
import json
import numpy as np
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from train_bio_cot_multimodal_balanced_xiangyang import (
    plot_advanced_violin_analysis,
    plot_loss_component_analysis
)

def generate_violin_from_results():
    """从结果文件生成小提琴图"""
    timestamp = '20260107_093123'
    results_file = Path(f'results_multimodal_balanced/results_bio_cot_multimodal_balanced_{timestamp}.json')
    output_dir = Path('results_multimodal_balanced')
    
    # 加载结果
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    history = results['history']
    
    print("📊 生成损失组件详细分析（包含小提琴图）...")
    plot_loss_component_analysis(history, output_dir, timestamp)
    
    # 注意：预测概率分布的小提琴图需要实际的预测概率数据
    # 如果JSON中没有保存，需要重新运行模型推理
    print("\n⚠️  预测概率分布的小提琴图需要实际的预测概率数据")
    print("   如果JSON文件中没有保存预测概率，需要重新运行模型推理")
    
    # 检查是否有保存的预测概率
    if 'final_val_probs' in results or 'val_probs' in results:
        y_probs = np.array(results.get('final_val_probs', results.get('val_probs')))
        y_labels = np.array(results.get('final_val_labels', results.get('val_labels')))
        
        print("📊 生成高级小提琴图分析...")
        plot_advanced_violin_analysis(y_labels, y_probs, output_dir, timestamp)
    else:
        print("❌ JSON文件中没有找到预测概率数据")
        print("   需要重新运行模型推理来生成预测概率分布的小提琴图")

if __name__ == '__main__':
    generate_violin_from_results()

